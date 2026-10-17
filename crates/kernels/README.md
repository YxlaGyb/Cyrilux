---
description: "the leaf operator library shared by every dense, CoreBrain and attention crate."
kind: "package-reference"
---

# kernels

English | [中文](README.zh.md)

## Summary

`crates.kernels` is the leaf operator library. It holds every gradient-free, device-agnostic function the learning and forward paths share: activations, normalizations, sparse competition, neuromodulation, regularization, straight-through estimators, plasticity rules, spectral guards, the step-cascade readers the declared forward graph binds, and the alignment views that bind it. It depends only on `torch` and holds no state and no network reference, so `crates.model`, `crates.hybrid`, `crates.corebrain` and `crates.attention` can all depend on it without creating a cycle.

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Further exploration](#further-exploration)

-----

<a id="use-this-package"></a>
## Use this package

Import the operators from the package root.

```python
from crates.kernels import rms_norm, saturate, topk_wta
```

| Export | Module | Shape | Meaning |
|---|---|---|---|
| `saturate(x)` | `activation` | as input | `x / (1 + abs(x))`, the shared flux-limiter. |
| `cubic(x)` | `activation` | as input | `x * (1 - 0.5 * x**2)`, the zero-backprop polynomial activation. |
| `rms_norm(x)` | `normalize` | as input | RMS normalization with a masked denominator for the zero-vector case. |
| `l2_norm(x)` | `normalize` | as input | L2 normalization with a masked denominator for the zero-vector case. |
| `soft_norm_preserve(W)` | `normalize` | in place | Bounded, non-clamping row-norm preservation. |
| `rel_norm(raw, ema, eps)` | `normalize` | scalar | Self-normalized ratio `raw / (ema + raw + eps)` in `[0, 1)`. |
| `topk_wta(x, k)` | `sparse` | `[N, K]` | Keeps the largest `k` entries per row and returns activation plus winner mask. |
| `ste_delta(view, e, win)` | `estimator` | `[N, K]` | Straight-through error that reaches the projection through the winner mask. |
| `compute_free_energy(...)` | `modulation` | scalar | Absolute error energy used as the survival currency. |
| `compute_dopamine(...)` | `modulation` | scalar | Sigmoid of the signed free-energy change. |
| `compute_ach(...)` | `modulation` | scalar | Uncertainty response. |
| `ach_noise(mu3, nov)` | `modulation` | as input | Element-wise ACh jitter `±0.03·(0.5 + nov)` with a random sign: the graph's one stochastic step. |
| `compute_uncertainty(...)` | `modulation` | scalar | Recent-error uncertainty estimate. |
| `combine_modulation(...)` | `modulation` | scalar | Combines the modulation channels into one gain. |
| `gate_psay(...)` | `modulation` | scalar | P3-b behavior gate: say versus look. |
| `foldiak_decorrelate(...)` | `regularize` | in place | Anti-Hebbian de-correlation of the slot covariance. |
| `variance_floor(...)` | `regularize` | in place | Per-slot gain that rescales spread below a floor, bounded to `[1, 2]`. |
| `participation_ratio(...)` | `regularize` | scalar | Spread diagnostic. |
| `activity_baseline(post, ema, initialized)` | `plasticity` | `[out]` | Window activity squared plus its excess over a slow EMA. |
| `energy_constraint(W, dW, post, ema, initialized, oja_alpha, oja_elasticity)` | `plasticity` | `[out, in]` | Real Oja term plus activity-dependent forgetting. |
| `rho_control(dW, W_ref, adaptive)` | `plasticity` | `[out, in]` | Channel-level plasticity control, returning scaled `dW`, `rho` and `s`. |
| `decorr_W(W, E, coef, max_delta_ratio, learn_boost, gate)` | `spectral` | in place | Removes the top-1 principal direction with a norm trust region, when the largest singular value stands above the white-noise upper bound `(√m+√n)/√(mn)·‖W‖_F` by `gate`. |
| `spectral_radius_guard(W, rescale, bound)` | `spectral` | in place | Bounds the recurrence when its estimated radius diverges. |
| `align_rows(w, n)` | `align` | `[n, ...]` | First `n` rows, the view that binds a weight to a static row count. |
| `align_square(m, n)` | `align` | `[n, n]` | Top-left `[n, n]` block. |
| `align_mem_window(m_seq, n, s, k, a)` | `align` | `[n, s, k, a]` | The memory window the memory ring reads. |
| `pack3(a, b, c)` | `align` | tuple | Bundles three tensors into one node output, the STP resource triple. |

The step operators of the declared forward graph (`cascade`):

| Export | Module | Shape | Meaning |
|---|---|---|---|
| `zero_input(dev, n, s, d)` | `cascade` | `[n, s, d]` | The all-zero external input of a free-run step. |
| `one_hot_input(byte_ids, alphabet)` | `cascade` | `[N, S, A]` | Byte one-hot input channel in fp16. |
| `history_join(padmax, z0, alphabet)` | `cascade` | `[N, S, A]` | Temporal input channel: the previous frame's input, padded on the first frame. |
| `join_last(x, y)` | `cascade` | last dim | Concatenates two channel blocks. |
| `project(x, w, b)` | `cascade` | `[N, S, d]` | Linear projection `x @ w.T + b`. |
| `bias_3d(b)` | `cascade` | `[1, 1, d]` | Lifts a bias for per-frame broadcast. |
| `add_scaled(x, y, k)` | `cascade` | as input | Additive modulation `x + k·y`. |
| `add_unsqueezed(x, y)` | `cascade` | as input | Per-frame broadcast add, the CoreBrain correction. |
| `nonzero_mask(byte_ids)` | `cascade` | `[N, S]` bool | Valid-position mask, padding is 0. |
| `osc_phase(tab, cnt, span, s)` | `cascade` | `[1, S, 1]` | Oscillator phase read from a table, no runtime trigonometry. |
| `osc_advance(cnt, s, span)` | `cascade` | in place | Advances the phase counter. |
| `foldiak_gate(z, m)` | `cascade` | as input | Lateral inhibition `z - 0.2·(M @ zᵀ)ᵀ`. |
| `center_last(x)` | `cascade` | as input | Centres the last dimension. |
| `drop_last(x)` | `cascade` | `[N, S-1, ...]` | Drops the last frame to align with a difference. |
| `seq_diff(x)` | `cascade` | `[N, S-1, ...]` | Frame-to-frame difference. |
| `unit_norm(x)` | `cascade` | as input | Zero-guarded unit normalization. |
| `prepend_pad(x, padmax, d)` | `cascade` | `[N, S+1, ...]` | Prepends the first-frame pad. |
| `pred_err(dz, mu)` | `cascade` | scalar | Difference prediction residual, the free energy. |
| `mem_reset(m_seq, mem_m, n)` | `cascade` | in place | Writes the current memory body into the first frame. |
| `mem_advance(m_seq, z4, t, mem_a)` | `cascade` | in place | Leaky-integrator advance of frame `t`. |
| `mem_read(m_seq, z4, k)` | `cascade` | `[N, S, K·a]` | Content-addressed read: a cosine-similarity softmax gate summing to `K`. |
| `mem_last(m_seq)` | `cascade` | `[K, a]` | Last-frame memory averaged over the batch. |
| `nov_reset(z4)` | `cascade` | `[N, S, ...]` | Slow channel starts at the fast channel's first frame. |
| `nov_advance(zslow, z4, t)` | `cascade` | in place | Slow channel advance, a `0.99` / `0.01` leak. |
| `nov_read(z4, zslow)` | `cascade` | `[N, S]` | Fast-slow divergence as per-frame novelty. |

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

The crate is the single home for operators that were previously duplicated across `crates.model`, `crates.hybrid`, `crates.corebrain` and `crates.attention`. Moving them here breaks the reverse dependency from the learning path back to `crates.model`, since pure math never needs the orchestrator.

Every function is pure: it reads its arguments and returns its result, except the in-place writers (`soft_norm_preserve`, `foldiak_decorrelate`, `variance_floor`, `decorr_W`, `spectral_radius_guard`, `osc_advance`, `mem_reset`, `mem_advance`, `nov_advance`) which mutate the tensor they are handed. No function holds a network reference. When an operator needs network state (the slow EMA buffers, the cold-start flags, the `cfg` coefficients), the caller in `crates.hybrid.learning._common` reads that state and passes it in, so the kernel stays testable in isolation.

Two modules exist for the declared forward graph. `cascade` names the step-level operators of the activation path, so each is one declared node instead of an inline line of the engine; it holds no state and makes no control-flow decision. `align` holds the four pure view and pack operators that bind tensors whose static row or frame count sits above the dynamic one, so a shape is named without clamping or copying. `ach_noise` is the one stochastic operator: it draws the ACh jitter from the generator, which keeps the single random source at a named node.

The sparse and modulation paths keep the fp16 contract. `rms_norm` and `l2_norm` guard the zero-vector case with a masked denominator instead of a clamp, and `saturate` and `cubic` bound their tails structurally, so no operator relies on a gradient-breaking clamp. Two module-level caches inside `spectral` (`_PROBE_CACHE` and `_EYE_CACHE`) reuse a probe vector and an identity mask per `(device, dtype, size)`, which keeps the hot path free of new allocations.

-----

<a id="further-exploration"></a>
## Further exploration

- [model](../model/README.md): configuration and the numeric constants the kernels consume.
- [dense](../../crates/hybrid/README.md): the forward and learning paths that call these operators.
- [learning](../hybrid/learning/README.md): the state adapters that delegate to this crate.
