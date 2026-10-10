"""
每步稀疏码.

C 活跃 / P 可塑 / r 发放率: 稀疏只做约束, 不参与执行.
只做 bool 乘法, EMA, topk: 禁 gather/scatter/index_*/reshape/nonzero/to_sparse.
S 存活门控由 crates.neuron 产, 本模块只读.
"""

from __future__ import annotations

import torch

from crates.kernels import topk_wta


def compete(h: torch.Tensor, S: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """存活门控竞争: 死神经元先被 S 掩零, 再逐位置取 top-k. 返回 (激活 a, 活跃码 C)."""
    return topk_wta(h * S, k)


def bind(C: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """可塑码 P = C ∩ S: 限定 Hebbian 只更新既活跃又存活的神经元."""
    return C & S


def rate_ema(r: torch.Tensor, h: torch.Tensor, decay: float) -> torch.Tensor:
    """发放率稳态 EMA (就地更新, 仅内在约束, 不作外部目标)."""
    r.mul_(1.0 - decay).add_(decay * h.abs().mean(dim=(0, 1)))
    return r
