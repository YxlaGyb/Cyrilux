# model.pc — Predictive Coding 核心模块 (全张量化)

from .constants import (
    CONN_FEEDBACK,
    CONN_FEEDFORWARD,
    CONN_LATERAL,
    F_BCM_SLOPE,
    F_BCM_ZERO,
    F_EPS,
    F_FIRING_RATE,
    F_MU,
    F_PI,
    F_THRESHOLD,
    F_Z,
    F_Z_PREV,
    HIDDEN_LAYERS,
    LAYER_CONFIG,
    LAYER_L2,
    LAYER_L3,
    LAYER_L4,
    LAYER_L5,
    LAYER_L6,
    LAYER_SENSORY,
    N_STATE_FIELDS,
    TOP_LAYER,
)
from .tensor_pool import TensorNeuronPool
from .neuromodulation import (
    combine_modulation,
    compute_ach,
    compute_dopamine,
    compute_precision_scales,
    compute_uncertainty,
)

__all__ = [
    "TensorNeuronPool",
    "compute_uncertainty",
    "compute_dopamine",
    "compute_ach",
    "combine_modulation",
    "compute_precision_scales",
]
