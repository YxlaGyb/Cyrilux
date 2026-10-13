---
description: "a parallel coordination layer over the dense PPA, with two views, slot competition, cross-view prediction and a window-level correction."
kind: "package-reference"
---

# corebrain

English | [中文](README.zh.md)

## Summary

`crates.corebrain` holds the Core Brain, a parallel layer beside the dense PPA network. It reads the native `z4` state, pools two views (the last quarter of the window's content and all of its content), projects both into one shared slot space, and runs a top-k winner-take-all inside each view. A small predictor maps the short-view activation onto the slow target built from the long view, and that prediction error is the only learning signal the layer receives. The forward output is a window-level correction added to the L4 state before the L2 normalisation. It is a mandatory part of the dense network and is always constructed. It is not an association cortex: both of its views are tapped from one source, so its shared slots serve one signal's two apertures.

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
| `CoreBrainConfig` | dataclass | Parameter-level defaults: slot count, top-k, learning rates, EMA rate, trust region, regulariser weights, deviation floor, correction scale, declared window length, attention EMA rate, attention temperature and attention floor. |
| `CoreBrain(cfg)` | module | The layer itself. Holds the two view projections, the slot predictor, the slow target buffer and the mandatory attention modulator. |
| `HybridPCNet._core_brain` | module | Constructed automatically by `HybridPCNet` and always present. |

Call the layer.

| Export | Shape | Meaning |
|---|---|---|
| `CoreBrain.forward(z4, valid=None)` | `[N, d_in]` | Window-level correction. The caller broadcasts it over the sequence positions. |
| `CoreBrain.learn()` | dict | One update of the layer's own weights, then the diagnostics below. |
| `CoreBrain.observe(z4=None, valid=None)` | dict | With `z4`, runs one forward and one update and returns the diagnostics; without `z4`, returns the latest diagnostics. |

Inputs the contract accepts.

| Input | Shape | Meaning |
|---|---|---|
| `z4` | `[N, S, d_in]`, fp16 | The native L4 state. The layer reads the same tensor the rest of the network uses and never writes it back. |
| `valid` | `[N, S]`, bool | Content mask: true where the position holds real content. `ForwardEngine._predict` passes `byte_ids != 0`. Both views pool over content only, so the fill bytes of a document's last window cannot move them. |
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

CoreBrain's cross-view prediction error is passed through the mandatory `crates.attention` modulation layer before it updates `Q`, `P_short` and `P_long`. The attention layer keeps its own slow activation state and does not change CoreBrain's target.

Two views come from one window. `crates/corebrain/center.py::_views` takes the last `q` content positions for the short view and all content positions for the long view, so the short view carries no earlier history while the long view does. Both pool over positions only; no new parameters and no new data are involved. `q` is `window_len * short_frac` of the declared window, anchored at the last content position rather than at the last position: the last window of a document is left-aligned and zero-filled, so an anchor at the window end would pool fill bytes into the short view, and for a window whose content is shorter than `q` it would return an all-zero view. Content shorter than `q` degrades to the whole content, and an empty window returns zeros rather than a division by zero. The aperture is a property of the model's window, not of the length of the tensor a caller happens to pass.

Each view has its own projection into the shared slot space. `crates/corebrain/project.py` holds `P_short` and `P_long`, learned by a local delta rule with a trust region and a soft row-norm bound. `P_long_ema` is a slow copy of `P_long` that follows it by exponential moving average and receives no error directly, which keeps the target from moving as fast as the online side.

Competition is hard and stays inside each view. `crates.kernels.topk_wta` (in `crates/kernels/sparse.py`) keeps the largest `top_k` entries per row and zeroes the rest, returning the winner mask alongside the activation. Both views draw on the same slot indices, so a slot can be won by either view.

The predictor is trained by the error; the projections are trained through the winner mask. `crates/corebrain/predict.py` holds `Q`, a square slot-to-slot map updated by a local delta rule. Because the winner-take-all is not differentiable, the error reaches the projections through a straight-through rule: `crates/corebrain/project.py::learn` multiplies the error by the winner mask, so only the columns that produced a winning slot receive an update.

Collapse is held off without gradients. `crates.kernels` (`foldiak_decorrelate` and `variance_floor`) applies two direct rules to the slot activations, not to the weights: an anti-Hebbian de-correlation that removes the off-diagonal of the slot covariance, and a per-slot gain that rescales any slot whose spread falls below the floor. The gain is bounded to the interval `[1, 2]`, so no clamp is needed.

The correction is the predicted long-view code read back into L4 space. `CoreBrain.forward` maps the short-view activation through `Q`, then through the transpose of `P_long`, and scales the result. The layer never decodes to bytes.

The layer learns only from its own error. `LearningEngine.learn` calls `CoreBrain.learn` after the domain updates, and only in the perception phase; free-run and the frozen echo phase skip it. The byte loss never enters the layer, so the correction is not shaped by the byte stream.

**Shared convergence, not an association cortex.** Two apertures over one signal give a temporal contrast, not a multimodal one. Two views tapped from two different containers would give a multimodal one, and that is what this layer exists for; the shared slot space is the place where modalities are meant to meet. The mechanism below is therefore fixed as a contract, and it is recorded as unfinished work in [docs/pending.md](../../docs/pending.md) because it is not built.

What counts as a second source is narrow: a different signal container, that is a second organ `crates/modal/<signal>` plus a tap that turns it into a convergence vector. Another layer, another aperture or another window length over the same signal does not count.

| Aspect | Rule |
|---|---|
| Hierarchy | Each source competes inside its own subspace first (the unimodal zone); its sparse code then enters the shared layer (the convergence zone). |
| Source normalisation | Pooling over content positions is already the simplest divisive normalisation. A source declares its own, so a second source can declare RMS or total-energy division instead of letting scale monopolise the slots. |
| Shared slots | OR type: a slot may be won by either source, which is what `slot_overlap` already measures. AND type selectivity is not written in as code; it is left to be grown by mutual prediction, since a co-occurrence that cannot be predicted produces error and is learned away. |
| Glue | Cross-source mutual prediction: source `i`'s short-aperture activation predicts source `j`'s slow target through `Q_ij`, the error passes the mandatory attention modulation, and each source updates and injects on its own. |
| Read-outs | The primary source keeps today's key names; with more than one source the keys carry a source suffix. |
| Phase | A gate on routing only, reusing the rhythm counters the repository already keeps. It never enters a learning signal. |
| Parameters and migration | The primary source keeps `P_short`, `P_long`, `P_long_ema`, `Q` and every buffer unchanged, so existing checkpoints stay valid and the C9 fail-fast needs no change. The second source adds new parameter keys, and only that change migrates, through `scripts/migrate_ckpt.py`. |
| Unchanged red lines | A source name or source label never enters a learning signal; the byte loss never enters the layer; it never writes `net._z4`; attention stays mandatory and stays a modulator; fp16 only, no clamp, no `.item()`, no tensor built per step. |

<a id="further-exploration"></a>
## Further exploration

- [hybrid](../hybrid/README.md): the forward chain and the point where the correction is added.
- [attention](../attention/README.md): the mandatory error-precision modulator.
- [model](../model/README.md): the network declaration and lifecycle.
