"""model.pc.sparse

稀疏/事件驱动后端 (页式槽位 + alive 掩码 + 邻接表).

全张量化神经元池, scatter_add 预测聚合, 零 matmul 依赖.
"""

from ..constants import (
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
from .neuromodulation import compute_precision_scales
from .tensor_pool import TensorNeuronPool

__all__ = [
    "TensorNeuronPool",
    "compute_precision_scales",
]
