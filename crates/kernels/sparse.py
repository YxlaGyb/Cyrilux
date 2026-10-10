"""
稀疏竞争算子
"""

from __future__ import annotations

import torch


def topk_wta(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """硬 top-k: 每行保留数值最大的 k 个槽位, 其余置零. 返回 (激活, 胜出掩码)."""
    if k >= x.shape[-1]:
        return x, torch.ones_like(x) > 0
    thresh = x.topk(k, dim=-1).values[..., -1:]
    mask = x >= thresh
    return x * mask.to(x.dtype), mask
