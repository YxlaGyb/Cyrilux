"""
每步稀疏码.

C 活跃 / P 可塑 / r 发放率: 稀疏只做约束, 不参与执行.
只做 bool 乘法, EMA, 单向门 (relu+倒数), topk: 禁 gather/scatter/index_*/reshape/nonzero/to_sparse.
S 存活门控由 crates.neuron 产, 本模块只读.
"""

from __future__ import annotations

import torch

from crates.kernels import topk_wta


def compete(
    h: torch.Tensor, S: torch.Tensor, k: int,
    M_x: torch.Tensor | None = None, lam: float = 0.0,
    g: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """存活门控竞争: 死神经元先被 S 掩零, 再逐位置取 top-k. 返回 (激活 a, 活跃码 C).

    M_x = None → 与无侧抑制旧路径逐位等价 (不发起 0@h 空算子).
    侧抑制场 h − λ·(M_x@h) 为纯稠密 matmul, B·T 折叠一次, 稀疏只存在于 M_x 的约束语义.
    g = None → 率治理关闭, 逐位等价 (不发起多余乘法).
    """
    if M_x is None and g is None:
        return topk_wta(h * S, k)
    n = h.shape[-1]
    h_flat = h.view(-1, n)
    if M_x is not None:
        h_flat = h_flat - lam * (h_flat @ M_x.T)
    if g is not None:
        h_flat = h_flat * g
    return topk_wta(h_flat.view_as(h) * S, k)


def bind(C: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """可塑码 P = C ∩ S: 限定局部更新只作用于既活跃又存活的神经元."""
    return C & S


def rate_ema(r: torch.Tensor, C: torch.Tensor, decay: float) -> torch.Tensor:
    """发放率 EMA (就地更新): 胜者 mask 逐窗折叠占比, 内在自维护量, 不作外部目标."""
    r.mul_(1.0 - decay).add_(decay * C.mean(dim=(0, 1), dtype=torch.float16))
    return r


def g_rate(r: torch.Tensor, rho: float) -> torch.Tensor:
    """率上限门 (单向): r ≤ ρ → 精确 1.0 (安静单元永不因 g_rate 入 topk), 超标才衰减.

    与 θ 家族 g = 1/(1+θ) 同语义族; 只压制超频, 不抬升, 不注入任何外部任务信号.
    """
    return 1.0 / (1.0 + torch.relu(r - rho))
