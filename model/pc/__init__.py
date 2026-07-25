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
]
