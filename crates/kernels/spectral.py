"""
谱与去相关算子

去主成分投影 (decorr_W) 与递归矩阵谱半径守卫; 缓存按 (dev,dtype,n) 惰性建一次.
"""

from __future__ import annotations

import torch

_PROBE_CACHE: dict[tuple, torch.Tensor] = {}
_EYE_CACHE: dict[tuple, torch.Tensor] = {}

# 主方向显著性闸: ‖W·v‖ 需超过白噪声最大奇异值上界 (√m+√n)/√(mn)·‖W‖_F 的此倍数
SPECTRAL_GATE = 1.30


def _probe_v(dev, dtype, n):
    """幂迭代探针向量: 按 (dev,dtype,n) 惰性建一次, 之后 normal_ 就地刷新 (零热路径新建).
    构建用 torch.empty (值被 normal_ 覆盖): 零 RNG 消耗 — 捕获干跑的重建与 eager 首用
    不产生随机流错位 (CUDA 图逐位同流前提)."""
    key = (dev, dtype, n)
    v = _PROBE_CACHE.get(key)
    if v is None:
        _PROBE_CACHE[key] = torch.empty(n, 1, device=dev, dtype=dtype)  # tensor-guard: rare (每尺寸一次, 无随机)
        v = _PROBE_CACHE[key]
    return v


def _eye_mask(dev, dtype, n):
    """去相关单位掩码: 惰性建一次, 各尺寸共享 (零热路径新建)."""
    key = (str(dev), dtype, n)
    m = _EYE_CACHE.get(key)
    if m is None:
        _EYE_CACHE[key] = torch.eye(n, device=dev, dtype=dtype)  # tensor-guard: rare (每尺寸一次)
        m = _EYE_CACHE[key]
    return m


def spectral_radius_guard(W: torch.Tensor, rescale: bool = True, bound: float = 1.5) -> torch.Tensor:
    """递归矩阵谱半径安全约束: 10 次幂迭代估 ρ, ρ>bound 时 W ← W·(bound/ρ).

    正常动力学由能量耗散自然饱和, 守卫只在发散时兜底. rescale=False 仅观测 ρ.
    """
    a = W.shape[0]
    v = _probe_v(W.device, W.dtype, a)
    v.normal_()
    v = v / (v.norm() + 1e-4)
    for _ in range(10):
        v = W @ v
        v = v / (v.norm() + 1e-4)
    rho = (W @ v).norm()
    if rescale:
        W.mul_(torch.minimum(bound / rho, torch.ones_like(rho)))  # tensor-guard: bound (谱界缩放, 非梯度)
    return rho


def decorr_W(
    W: torch.Tensor,
    E: torch.Tensor,
    coef: float = 1.0,
    max_delta_ratio: float | None = None,
    learn_boost: float = 1.0,
    gate: float = SPECTRAL_GATE,
) -> torch.Tensor:
    """权重去主成分投影: W -= β·(W@v)⊗v, v = W 的 top1 奇异方向 (幂迭代 3 次).

    纯共现更新把每行收敛到 ±w 单一方向 (投影秩 1); β 超量抑制必须压倒该正反馈.
    谱近平时 (最大奇异值不高于白噪声上界) 3 次幂迭代收敛不到主方向: v 是谱顶部
    簇里的随机方向, 写入量恒为 ‖W‖/√rank —— 此时闸关闭, 只更新诊断 E, 不写 W.
    """
    dev = W.device
    dim, cols = W.shape[0], W.shape[1]
    wfro = W.norm()
    Wn = W / (W.norm(dim=1, keepdim=True) + 1e-3)
    dE = (Wn @ Wn.T).abs()  # 绝对相关 (诊断: 行收敛指标)
    eye_mask = 1.0 - _eye_mask(dev, W.dtype, dim)
    E.data.mul_(0.97).add_((dE * eye_mask) * (0.05 * learn_boost))
    # 幂迭代前缩放为单位 Frobenius 范数: 方向不变, 防小矩阵在 fp16 乘加中下溢成 NaN
    Wn = W / (wfro + 1e-8)
    v = _probe_v(dev, W.dtype, cols)
    v.normal_().mul_(0.01)
    for _ in range(3):
        v = Wn.T @ (Wn @ v)
        v = v / (v.norm() + 1e-8)
    c = W @ v  # 每行在 v 方向上的投影系数 (含 ± 符号)
    dW = c @ v.T
    s1 = dW.norm()
    # 显著性闸 (branchless): 上界之下不存在可提取的主方向, 投影无意义
    dW = dW * (s1 > gate * wfro * (dim ** 0.5 + cols ** 0.5) / (dim * cols) ** 0.5).to(dW.dtype)
    # 范数信任域: 单步扰动上限 = max_delta_ratio·‖W‖_F, 防下游突变 (branchless min, 免同步)
    if max_delta_ratio is not None:
        dW = dW * torch.minimum(max_delta_ratio * wfro / (s1 + 1e-8), torch.ones_like(s1))
    dW = dW * coef
    W -= dW
    return dW
