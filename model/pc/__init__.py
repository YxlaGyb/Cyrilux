# model.pc — Predictive Coding 核心模块 (全张量化)

from .tensor_pool import TensorNeuronPool
from .homeostasis import homeostasis_step
from .neuromodulation import (
    compute_uncertainty,
    compute_dopamine,
    compute_ach,
    combine_modulation,
    compute_precision_scales,
)

__all__ = [
    "TensorNeuronPool",
    "homeostasis_step",
    "compute_uncertainty",
    "compute_dopamine",
    "compute_ach",
    "combine_modulation",
    "compute_precision_scales",
]
