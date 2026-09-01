"""
LearningEngine
密集 Hebbian 学习
"""

from __future__ import annotations

from ._common import (
    ELIG_GAMMA,
    LM_TRUST_REGION,
    _activity_baseline,
    _decorr_W,
    _elig_accum,
    _energy_constraint,
    _rho_ctrl,
    _spectral_radius_guard,
)
from .engine import EngineCore
from model.constants import DIFF_TRUST_REGION

__all__ = [
    "LearningEngine",
    "LM_TRUST_REGION",
    "ELIG_GAMMA",
    "DIFF_TRUST_REGION",
    "_activity_baseline",
    "_decorr_W",
    "_elig_accum",
    "_energy_constraint",
    "_rho_ctrl",
    "_spectral_radius_guard",
]


class LearningEngine(EngineCore):
    """学习引擎: 持 net 引用, 复用 _predict 存的 _z* 状态."""
