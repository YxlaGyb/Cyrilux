---
description: "corebrain: a parallel coordination layer over the dense PPA, with two views, slot competition, cross-view prediction and a window-level correction."
kind: "package-reference"
---

# corebrain

English | [中文](README.zh.md)

## Summary

`crates.corebrain` holds the Core Brain, a parallel layer beside the dense PPA network. It reads the native `z4` state, pools two views (the recent quarter of the window and the whole window), projects both into one shared slot space, and runs a top-k winner-take-all inside each view. A small predictor maps the short-view activation onto the slow target built from the long view, and that prediction error is the only learning signal the layer receives. The forward output is a window-level correction added to the L4 state before the L2 normalisation. It is a mandatory part of the dense network and is always constructed.

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Further exploration](#further-exploration)

-----

<a id="use-this-package"></a>
## Use this package

Construct the layer through the network so it stays a registered submodule.

| Export | Shape | Meaning |
|---|---|---|
| `CoreBrainConfig` | dataclass | Parameter-level defaults: slot count, top-k, learning rates, EMA rate, trust region, regulariser weights, deviation floor, correction scale, attention EMA rate, attention temperature and attention floor. |
| `CoreBrain(cfg)` | module | The layer itself. Holds the two view projections, the slot predictor, the slow target buffer and the mandatory attention modulator. |
| `DensePCNet._core_brain` | module | Constructed automatically by `DensePCNet` and always present. |

Call the layer.

| Export | Shape | Meaning |
|---|---|---|
| `CoreBrain.forward(z4)` | `[N, d_in]` | Window-level correction. The caller broadcasts it over the sequence positions. |
| `CoreBrain.learn()` | dict | One update of the layer's own weights, then the diagnostics below. |
| `CoreBrain.observe(z4=None)` | dict | With `z4`, runs one forward and one update and returns the diagnostics; without `z4`, returns the latest diagnostics. |

Inputs the contract accepts.

| Input | Shape | Meaning |
|---|---|---|
| `z4` | `[N, S, d_in]`, fp16 | The native L4 state. The layer reads the same tensor the rest of the network uses and never writes it back. |
| `n_slots` | int, default 32 | Size of the layer's own slot space. Independent of the concept-binding slots. |
| `top_k` | int, default 4 | Winners kept inside each view's competition. |

Diagnostics returned by `learn` and `observe`.

| Key | Meaning |
|---|---|
| `xv_energy` | Mean squared cross-view prediction error. |
| `xv_energy_shuffled` | The same error with the short-view activation rolled by one batch position. A large gap against `xv_energy` means the prediction carries sample-specific content. |
| `effective_rank` | Participation ratio of the long-view slot activation. A value near one means the activation collapsed onto a single slot. |
| `slot_overlap` | Fraction of slots won by both views. |
| `delta_cos` | Cosine between the current and previous window-level correction. |
| `delta_absmax` | Largest absolute correction entry, the positive-feedback read-out. |

<a id="understand-the-implementation"></a>
## Understand the implementation

The layer sits beside the forward chain, not inside it. In `ForwardEngine._predict`, after the L4 recurrence produces `z4`, the layer reads that native tensor and returns one correction per batch element, which is added before `l2_norm` feeds L2. Every other consumer of `z4` (the world-model difference, the binding layer, the memory units) still reads the native tensor, so the correction touches exactly one edge.

CoreBrain's cross-view prediction error is passed through the mandatory `crates.attention` modulation layer before it updates `Q`, `P_short` and `P_long`. The attention layer keeps its own slow activation state and does not change CoreBrain's JEPA target.

Two views come from one window. `crates/corebrain/center.py::_views` takes the recent quarter of the sequence positions for the short view and every position for the long view, so the short view carries no earlier history while the long view does. Both pool over positions only; no new parameters and no new data are involved.

Each view has its own projection into the shared slot space. `crates/corebrain/project.py` holds `P_short` and `P_long`, learned by a local delta rule with a trust region and a soft row-norm bound. `P_long_ema` is a slow copy of `P_long` that follows it by exponential moving average and receives no error directly, which keeps the target from moving as fast as the online side.

Competition is hard and stays inside each view. `crates/corebrain/compete.py::topk_wta` keeps the largest `top_k` entries per row and zeroes the rest, returning the winner mask alongside the activation. Both views draw on the same slot indices, so a slot can be won by either view.

The predictor is trained by the error; the projections are trained through the winner mask. `crates/corebrain/predict.py` holds `Q`, a square slot-to-slot map updated by a local delta rule. Because the winner-take-all is not differentiable, the error reaches the projections through a straight-through rule: `crates/corebrain/project.py::learn` multiplies the error by the winner mask, so only the columns that produced a winning slot receive an update.

Collapse is held off without gradients. `crates/corebrain/regularize.py` applies two direct rules to the slot activations, not to the weights: an anti-Hebbian de-correlation that removes the off-diagonal of the slot covariance, and a per-slot gain that rescales any slot whose spread falls below the floor. The gain is bounded to the interval `[1, 2]`, so no clamp is needed.

The correction is the predicted long-view code read back into L4 space. `CoreBrain.forward` maps the short-view activation through `Q`, then through the transpose of `P_long`, and scales the result. The layer never decodes to bytes.

The layer learns only from its own error. `LearningEngine.learn` calls `CoreBrain.learn` after the domain updates, and only in the perception phase; free-run and the frozen echo phase skip it. The byte loss never enters the layer, so the correction is not shaped by the byte stream.

<a id="further-exploration"></a>
## Further exploration

- [dense](../dense/README.md): the forward chain and the point where the correction is added.
- [attention](../attention/README.md): the mandatory error-precision modulator.
- [model](../model/README.md): the network declaration and lifecycle.
