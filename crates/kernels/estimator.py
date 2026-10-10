"""
直通估计 (STE) 算子
"""

from __future__ import annotations

import torch


def ste_delta(view: torch.Tensor, e: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    """直通误差: 误差按恒等穿过 WTA, 只落在胜出槽位对应的投影列上.

    非胜出列恒为零, 这是 STE 的可测契约.
    """
    return (view.transpose(0, 1) @ (e * win.to(torch.float16))) / max(1, view.shape[0])
