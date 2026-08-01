"""model.pc

预测编码双后端门面 

(dense: 全 matmul / sparse: 页式槽位).
"""

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
    K_FAN,
    LAYER_CONFIG,
    LAYER_L2,
    LAYER_L3,
    LAYER_L4,
    LAYER_L5,
    LAYER_L6,
    LAYER_SENSORY,
    N_STATE_FIELDS,
    PAGE_NEURONS,
    PAGE_SYNAPSES,
    PAGE_TD,
    TOP_LAYER,
)
from .dense import DensePCConfig, DensePCNet
from .modulation import (
    combine_modulation,
    compute_ach,
    compute_ach_gain,
    compute_dopamine,
    compute_dopamine_gain,
    compute_free_energy,
    compute_uncertainty,
    soft_norm_preserve,
)
from .sparse import TensorNeuronPool, compute_precision_scales

__all__ = [
    "DensePCNet",
    "DensePCConfig",
    "TensorNeuronPool",
    "compute_uncertainty",
    "compute_dopamine",
    "compute_ach",
    "combine_modulation",
    "compute_precision_scales",
    "compute_dopamine_gain",
    "compute_ach_gain",
    "compute_free_energy",
    "soft_norm_preserve",
]
