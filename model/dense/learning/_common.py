"""学习引擎共享纯函数层"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from model.model_cyrene import DensePCNet

from model.constants import ELIG_GAMMA, LM_TRUST_REGION


class _MixinBase:
    """mixin 公共基类: 让类型检查器认识组合类成员 (self.net)."""

    net: DensePCNet


def _activity_baseline(
    net: DensePCNet, post: torch.Tensor, ema_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (p2, excess): 窗内活动² 与相对慢基线 EMA (0.99/0.01) 的超额部分.

    首窗延迟初始化 ema ← post² (excess 从零起步), 纯局部.
    """
    p2 = (post * post).mean(dim=(0, 1))  # [out] 窗内活动²
    ema = getattr(net, ema_name)[: p2.shape[0]]  # 修剪后 active 收缩: 只写头部切片
    if ema_name not in net._active_ema_init:
        ema.copy_(p2)
        net._active_ema_init.add(ema_name)
        excess = torch.zeros_like(p2)
    else:
        ema.mul_(0.99).add_(0.01 * p2)
        excess = torch.relu(p2 - ema)
    return p2, excess


def _energy_constraint(
    net: DensePCNet, W: torch.Tensor, dW: torch.Tensor, post: torch.Tensor, ema_name: str
) -> torch.Tensor:
    """内建能量约束: 真 Oja + 活动依赖遗忘, 逐输出单元, 纯局部.

    dW ← dW − (α·post² + β·relu(post²−ema)) ⊙ W;  α=cfg.oja_alpha, β=cfg.oja_elasticity.
    """
    p2, excess = _activity_baseline(net, post, ema_name)
    coef = net.cfg.oja_alpha * p2 + net.cfg.oja_elasticity * excess
    return dW - coef.unsqueeze(1) * W


def _elig_accum(net: DensePCNet, wname: str, dW_raw: torch.Tensor) -> torch.Tensor:
    """资格迹: E ← γ·E + dW_raw, 返回当前迹 (与权重同形, 前缘对齐修剪活性切片).

    R 默认 0 → 迹只记录不注入 (ΔW=η·R·E=0); 生存环境接入时经 _survival_signal 启用.
    """
    E = getattr(net, f"{wname}_elig")
    E = E[: dW_raw.shape[0], : dW_raw.shape[1]]
    E.mul_(ELIG_GAMMA).add_(dW_raw)
    return E


def _spectral_radius_guard(W: torch.Tensor, rescale: bool = True, bound: float = 1.5) -> torch.Tensor:
    """递归矩阵谱半径安全约束: 10 次幂迭代估 ρ, ρ>bound 时 W ← W·(bound/ρ).

    正常动力学由能量耗散自然饱和, 守卫只在发散时兜底. rescale=False 仅观测 ρ.
    """
    a = W.shape[0]
    v = torch.randn(a, 1, device=W.device, dtype=W.dtype)
    v = v / (v.norm() + 1e-4)
    for _ in range(10):
        v = W @ v
        v = v / (v.norm() + 1e-4)
    rho = (W @ v).norm()
    if rescale:
        W.mul_(torch.minimum(bound / rho, torch.ones_like(rho)))
    return rho


def _decorr_W(
    W: torch.Tensor,
    E: torch.Tensor,
    coef: float = 1.0,
    max_delta_ratio: float | None = None,
    learn_boost: float = 1.0,
) -> torch.Tensor:
    """权重去主成分投影: W -= β·(W@v)⊗v, v = W 的 top1 奇异方向 (幂迭代 3 次).

    赫布更新把每行收敛到 ±w 单一方向 (投影秩 1); β 超量抑制必须压倒该正反馈.
    """
    dev = W.device
    dim = W.shape[0]
    Wn = W / (W.norm(dim=1, keepdim=True) + 1e-3)
    dE = (Wn @ Wn.T).abs()  # 绝对相关 (诊断: 行收敛指标)
    eye_mask = 1.0 - torch.eye(dim, device=dev, dtype=W.dtype)
    E.data.mul_(0.97).add_((dE * eye_mask) * (0.05 * learn_boost))
    # 幂迭代前缩放为单位 Frobenius 范数: 方向不变, 防小矩阵在 fp16 乘加中下溢成 NaN
    Wn = W / (W.norm() + 1e-8)
    v = torch.randn(W.shape[1], 1, device=dev, dtype=W.dtype) * 0.01
    for _ in range(3):
        v = Wn.T @ (Wn @ v)
        v = v / (v.norm() + 1e-8)
    c = W @ v  # 每行在 top1 方向上的投影系数 (含 ± 符号)
    dW = c @ v.T
    # 范数信任域: 单步扰动上限 = max_delta_ratio·‖W‖_F, 防下游突变
    if max_delta_ratio is not None:
        dmax = max_delta_ratio * W.norm()
        dn = dW.norm()
        if dn > dmax:
            dW = dW * (dmax / dn)
    dW = dW * coef
    W -= dW
    return dW


def _rho_ctrl(dW: torch.Tensor, W_ref: torch.Tensor, tag: str, net: DensePCNet) -> torch.Tensor:
    """通道级塑性控制: ρ_i=‖ΔW_i‖/‖W_i‖, s_i=clip(0.03/ρ_i, 0.005, 1.0) 缩放时间尺度."""
    if not net.cfg.adaptive_rho:
        return dW
    nW_i = W_ref.norm() + 1e-8
    rho = dW.norm() / nW_i
    s_i = (0.03 / (rho + 1e-8)).clamp(0.005, 1.0)
    dW_s = dW * s_i
    net._rho_map[tag] = (rho, dW_s.norm() / nW_i, s_i)
    return dW_s