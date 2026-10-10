---
description: "learning：密集 PPA 循环的纯希伯学习引擎，组织为编排器加七个逐域 mixin：action、bind、feedforward、metabolism、predict、readout、temporal。"
kind: "package-reference"
---

# learning

English | [中文](README.zh.md)

## Summary

`crates.hybrid.learning` implements Hebbian learning for the dense PPA loop: a feed-forward pass, per-domain errors, then local weight updates with no backpropagation and no teacher targets. `LearningEngine` composes `EngineCore` with seven mixins. `engine.py` holds the `learn()` orchestration and the shared per-step context; `_common.py` holds the shared per-network state adapters, delegating the pure operators to `crates.kernels` (eligibility traces, energy constraints, decorrelation, spectral guard); `action.py`, `bind.py`, `feedforward.py`, `metabolism.py`, `predict.py`, `readout.py`, and `temporal.py` each own one update domain. All state flows through the network the engine holds a reference to.

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Further exploration](#further-exploration)

-----

<a id="use-this-package"></a>
## Use this package

| 导出 | 形状 | 含义 |
|---|---|---|
| `LearningEngine(net)` | Engine | Holds the network and runs one learning step. Constructed automatically as `net.learning_engine`. |
| `net.learn(byte_ids, closed_loop, free_run, force_perception)` | Learning step | A single Hebbian step. Routes internally to perception or echo via the P3-b behavior gate; takes no targets. |
| `LearningEngine.learn(...)` | Learning step | The same call, exposed directly by the engine. |

The `learn()` step accepts the following inputs, selected by flags.

| 导出 | 形状 | 含义 |
|---|---|---|
| `byte_ids` | `[N, S]` long tensor | Target bytes. When None (and not free-run) the step is a self-echo seeded from the previous window. |
| `closed_loop` | Flag | Autoregressive exposure training. The input is the first `k` real bytes anchored plus the model's own self-generated continuation; targets are always the real bytes. |
| `free_run` | Flag | Input is zeroed; only internal-dynamics weights update. The byte-domain weights are frozen. |
| `force_perception` | Flag | A P3-b bypass that forces perception (used on the first step and recovery) even if the behavior gate selects echo. |

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

`engine.py` orchestrates a fixed order of domain calls. It first runs `_predict` to gather the layer states and computes layer-wise prediction errors bottom-up, then routes the step to perception, echo, closed-loop, or free-run through the P3-b behavior gate and the C5 freezing contract. The echo phase freezes the world model. It then builds the diff window and the LM signal, updates metabolism before the action readout, applies the feed-forward and temporal domains, updates the readout head, and finally settles the behavior gate for the next step. It exposes telemetry (`free_energy`, `future_err`, `d_polarity`) and detaches the states it stored.

`_common.py` is the shared state-adapter layer. It reads the network's EMA buffers, cold-start flags and `cfg` coefficients, then calls the pure operators in `crates.kernels`: `elig_accum`, `energy_constraint`, `activity_baseline`, `rho_control`, `decorr_W` and `spectral_radius_guard`. `elig_accum` maintains an eligibility trace per weight and returns it, which is only injected into updates when a survival environment exists. `energy_constraint` applies a real Oja term plus activity-dependent forgetting, `decorr_W` removes the top-1 principal direction with a norm trust region, and `spectral_radius_guard` bounds the recurrence when it diverges. `activity_baseline` tracks window activity against a slow EMA.

The seven domains are: `feedforward` (bias steady states and the `W_04`/`W_42`/`W_56`/`W_23`/`W_state_pred` updates with explain-away weighting and homeostatic gains), `temporal` (multi-scale soft time windows over `W_diff` plus the `W_t*` temporal kernels with a spectral guard), `readout` (the LM head `W_lm`/`W_lm_2`/`W1`/`bias_lm` plus entropy-drive traction and opt-in reconstruction arms), `predict` (the self-organizing prediction engine `W_pred_54`/`W_pred_43` with future-discounted targets and RPE gating), `bind` (the Foldiak `M_l5` decorrelation and the `W_bind`/`W_bind_self` concept slots with the intrinsic oscillator), `action` (`W_act` closed-loop self-consistent generation learning), and `metabolism` (the free-energy ledger, survival signal, novelty ledger, and the P3-b behavior gate).

-----

<a id="further-exploration"></a>
## Further exploration

- [model](../model/README.md)：这些域更新的网络与配置声明。
- [dense](../README.md)：前馈与修剪引擎。