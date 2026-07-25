# model.pc — Predictive Coding 核心模块

from .neuron_pool import Neuron, Synapse, NeuronPool, SensoryEvent, NetworkEvent
from .sparse_forward import (
    predict_neuron,
    batch_predict,
    predict_layer,
    update_neuron,
    process_sensory_event,
    hebbian_step,
    batch_hebbian,
    emit_if_active,
    emit_active_neurons,
)
from .homeostasis import (
    adjust_threshold,
    batch_adjust_thresholds,
    should_prune,
    prune_network,
    should_grow,
    grow_network,
    homeostasis_step,
)
from .sparse_projections import (
    SparseTemporalSelf,
    SparseTopdown,
    SparseLMHead,
)
from .neuromodulation import (
    compute_uncertainty,
    compute_dopamine,
    compute_ach,
    combine_modulation,
    compute_precision_scales,
)

__all__ = [
    # neuron_pool
    "Neuron",
    "Synapse",
    "NeuronPool",
    "SensoryEvent",
    "NetworkEvent",
    # sparse_forward
    "predict_neuron",
    "batch_predict",
    "predict_layer",
    "update_neuron",
    "process_sensory_event",
    "hebbian_step",
    "batch_hebbian",
    "emit_if_active",
    "emit_active_neurons",
    # homeostasis
    "adjust_threshold",
    "batch_adjust_thresholds",
    "should_prune",
    "prune_network",
    "should_grow",
    "grow_network",
    "homeostasis_step",
    # sparse_projections
    "SparseTemporalSelf",
    "SparseTopdown",
    "SparseLMHead",
    # neuromodulation
    "compute_uncertainty",
    "compute_dopamine",
    "compute_ach",
    "combine_modulation",
    "compute_precision_scales",
]
