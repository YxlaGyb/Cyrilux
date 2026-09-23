---
description: "model：密集感知-预测-行动网络的配置与权重声明，加上共享神经调制与数值常量。"
kind: "package-reference"
---

# model

English | [中文](README.zh.md)

## Summary

`crates.model` owns the configuration and weight-declaration surface of the dense perception-prediction-action (PPA) closed loop. It exposes `CyreneModel`, a dataclass that enumerates every learning, pruning, metabolism, and oscillator hyperparameter, and `DensePCNet`, the `nn.Module` that declares the full weight and buffer layout and delegates the forward, learning, and pruning engines. The crate also carries the shared neuro-modulation functions (dopamine, ACh, free energy, norm-preservation) and the numeric constants that define layer roles and trust regions. It holds no training loop; all dynamics live in `crates.dense`.

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Further exploration](#further-exploration)

-----

<a id="use-this-package"></a>
## Use this package

Configure and build a network.

| 导出 | 形状 | 含义 |
|---|---|---|
| `CyreneModel(...)` | Hyperparameter dataclass | Call `dims()` for the layer-size map and `param_count()` for an estimated parameter total (feed-forward plus generation and temporal kernels). |
| `DensePCNet(config=None)` | `nn.Module` | Declares all parameters and buffers, initializes weights, and attaches `forward_engine`, `learning_engine`, and `pruner`. `config` defaults to `CyreneModel()`. |
| `DensePCNet.load(path, config=None)` | Checkpoint loader | Rebuilds a network from a complete checkpoint. Dimension fields always come from the checkpoint; `config` only supplies non-dimension values. Raises on a weights-only snapshot or unknown keys. |

Read and write checkpoints.

| 导出 | 形状 | 含义 |
|---|---|---|
| `net.save(path)` | Writer | Writes parameters, buffers, free runtime tensors, and CPU counters as safetensors with a `state_ver` and a `config` snapshot in metadata. |
| `DensePCNet.config_from_checkpoint(path)` | Reader | Reads the `config` snapshot without loading tensors. Useful for diagnostics such as `list --detail`. |

Run the model. These are thin delegations to the dense engines.

| 导出 | 形状 | 含义 |
|---|---|---|
| `net.forward(byte_ids)` | Feed-forward | Inference feed-forward. Returns the future prediction deviation. Deterministic, ACh off. |
| `net.generate(prompt, n_tokens, temperature, dev)` | Generation | Autoregressive generation from a prompt string. Returns bytes. |
| `net.learn(byte_ids, closed_loop, free_run, force_perception)` | Learning | Hebbian learning step with no backpropagation and no teacher targets. Internal metabolism routes the step to perception or echo. |
| `net.maybe_prune(step)` | Pruning | Metabolism-triggered topology remodelling seam. Polled every 8 steps; returns the trigger cause or None. |
| `net.sync_gate()` | Routing | One routing handoff per step that reads the P3-b behavior gate and mirrors it to a CPU flag. |

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

`CyreneModel` declares the hyperparameter surface. `dims()` and `param_count()` derive the declared architecture directly from the configuration.

`DensePCNet.__init__` divides into three responsibilities. First, weight declaration: feed-forward matrices (`W_04` through `W_56`), the world-model predictors (`W_diff`, `W_state_pred`, `W_pred_54`, `W_pred_43`), the competitive concept-binding layer (`W_bind`, `W_bind_self`), the temporal kernels (`W_t*`), and the action readout (`W_act`) and nonlinear mixing head (`W1`, `W_lm`, `W_lm_2`). Second, buffer declaration: slow variables such as activity EMAs, STP resources, the metabolism ledger, and the death-contract counters are registered as buffers and persist across `save` and `load`. Third, engine delegation: the model constructs `ForwardEngine`, `LearningEngine`, and `PruningEngine` and forwards the public calls to them.

`save` writes a `state_ver` anchor, a `config` snapshot, and the free runtime tensors under an `attr.` namespace plus CPU counters under a `py.` namespace. `load` refuses a snapshot that lacks `state_ver`; dimension fields follow the checkpoint shapes (pruning physically slices matrices); any unknown persistent key raises instead of being silently dropped.

The modulation helpers in `modulation.py` are pure and device-agnostic: `compute_dopamine` is a sigmoid of signed free-energy change, `compute_ach` responds to uncertainty, `soft_norm_preserve` bounds row norms in a structured, non-clamping way, and `rms_norm` and `l2_norm` guard against the zero-vector case with masked denominators so fp16 never divides by zero.

`constants.py` defines the neuron state-column indices, the six-layer cortex roles, the per-layer configuration (neuron count, inertia, k-WTA fraction), connection kinds, and the trust regions and eligibility discount that the learning domain consumes.

-----

<a id="further-exploration"></a>
## Further exploration

- [architecture](../../docs/architecture.md)：本 crate 的设计意图记录在这里。
- [dense](../dense/README.md)：本模型委托的那些引擎。
- [learning](../dense/learning/README.md)：逐域学习子 crate。