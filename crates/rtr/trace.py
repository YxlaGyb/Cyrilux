"""
RTR 资格迹适配

取权重对应迹缓冲并累积瞬时共活动标签; γ = ELIG_GAMMA.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from crates.model.constants import ELIG_GAMMA
from crates.rtr.kernel import elig_accum

if TYPE_CHECKING:
    from crates.model.model_cyrene import HybridPCNet


def trace(net: HybridPCNet, wname: str, dW_raw: torch.Tensor) -> torch.Tensor:
    """适配器: 取 net.{wname}_elig, 调 elig_accum (γ = ELIG_GAMMA)."""
    E = getattr(net, f"{wname}_elig")
    return elig_accum(E, dW_raw, ELIG_GAMMA)
