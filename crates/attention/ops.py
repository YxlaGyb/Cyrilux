from __future__ import annotations

import torch


def topk_wta(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    if k >= x.shape[-1]:
        return x, torch.ones_like(x) > 0
    thresh = x.topk(k, dim=-1).values[..., -1:]
    mask = x >= thresh
    return x * mask.to(x.dtype), mask
