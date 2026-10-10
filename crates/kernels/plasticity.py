"""
塑性算子

活动基线 / Oja 能量约束 / 资格迹 / 通道级塑性控制; 状态切片与登记由调用侧适配.
"""

from __future__ import annotations

import torch


def activity_baseline(
    post: torch.Tensor, ema: torch.Tensor, initialized: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """窗内活动² 与相对慢基线 EMA (0.99/0.01) 的超额部分.

    首窗延迟初始化 ema ← post² (excess 从零起步), 纯局部.
    ema 为完整缓冲, 本函数按 post 收缩写头部切片; 冷启动登记由调用侧负责.
    """
    p2 = (post * post).mean(dim=(0, 1))  # [out] 窗内活动²
    ema = ema[: p2.shape[0]]  # 修剪后 active 收缩: 只写头部切片
    if not initialized:
        ema.copy_(p2)
        excess = torch.zeros_like(p2)
    else:
        ema.mul_(0.99).add_(0.01 * p2)
        excess = torch.relu(p2 - ema)
    return p2, excess


def energy_constraint(
    W: torch.Tensor,
    dW: torch.Tensor,
    post: torch.Tensor,
    ema: torch.Tensor,
    initialized: bool,
    oja_alpha: float,
    oja_elasticity: float,
) -> torch.Tensor:
    """内建能量约束: 真 Oja + 活动依赖遗忘, 逐输出单元, 纯局部.

    dW ← dW − (α·post² + β·relu(post²−ema)) ⊙ W;  α=oja_alpha, β=oja_elasticity.
    """
    p2, excess = activity_baseline(post, ema, initialized)
    coef = oja_alpha * p2 + oja_elasticity * excess
    return dW - coef.unsqueeze(1) * W


def elig_accum(E: torch.Tensor, dW_raw: torch.Tensor, gamma: float) -> torch.Tensor:
    """资格迹: E ← γ·E + dW_raw, 返回当前迹 (与权重同形, 前缘对齐修剪活性切片).

    R 默认 0 → 迹只记录不注入 (ΔW=η·R·E=0); 生存环境接入时经 _survival_signal 启用.
    """
    E = E[: dW_raw.shape[0], : dW_raw.shape[1]]
    E.mul_(gamma).add_(dW_raw)
    return E


def rho_control(
    dW: torch.Tensor, W_ref: torch.Tensor, adaptive: bool
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """通道级塑性控制: ρ=‖ΔW‖/‖W‖, s=clip(0.03/ρ, 0.005, 1.0) 缩放时间尺度.

    返回 (缩放后 dW, ρ, s); adaptive=False 时原样返回 dW 与 (None, None).
    """
    if not adaptive:
        return dW, None, None
    nW_i = W_ref.norm() + 1e-8
    rho = dW.norm() / nW_i
    s_i = (0.03 / (rho + 1e-8)).clamp(0.005, 1.0)  # tensor-guard: bound (时间尺度设计界)
    return dW * s_i, rho, s_i
