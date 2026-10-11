"""
LearningEngine
稠密迹调制学习
"""

from __future__ import annotations

from crates.kernels import decorr_W as _decorr_W
from crates.kernels import spectral_radius_guard as _spectral_radius_guard
from crates.model.constants import DIFF_TRUST_REGION, LM_TRUST_REGION

from ._common import (
    _activity_baseline,
    _energy_constraint,
    _rho_ctrl,
)
from .engine import EngineCore

__all__ = [
    "LearningEngine",
    "LM_TRUST_REGION",
    "DIFF_TRUST_REGION",
    "_activity_baseline",
    "_decorr_W",
    "_energy_constraint",
    "_rho_ctrl",
    "_spectral_radius_guard",
]


class LearningEngine(EngineCore):
    """学习引擎: 持 net 引用, 复用 _predict 存的 _z* 状态."""
