---
description: "the leaf operator library shared by every dense, CoreBrain and attention crate."
kind: "package-reference"
---

# kernels

English | [中文](README.zh.md)

## Summary

`crates.kernels` is the leaf operator library. It holds every gradient-free, device-agnostic function the learning and forward paths share: activations, normalizations, sparse competition, neuromodulation, regularization, straight-through estimators, plasticity rules and spectral guards. It depends only on `torch` and holds no state and no network reference, so `crates.model`, `crates.hybrid`, `crates.corebrain` and `crates.attention` can all depend on it without creating a cycle.

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

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

The crate is the single home for operators that were previously duplicated across `crates.model`, `crates.hybrid`, `crates.corebrain` and `crates.attention`. Moving them here breaks the reverse dependency from the learning path back to `crates.model`, since pure math never needs the orchestrator.

Every function is pure: it reads its arguments and returns its result, except the in-place regularizers (`soft_norm_preserve`, `foldiak_decorrelate`, `variance_floor`, `decorr_W`, `spectral_radius_guard`) which mutate the tensor they are handed. No function holds a network reference. When an operator needs network state (the slow EMA buffers, the cold-start flags, the `cfg` coefficients), the caller in `crates.hybrid.learning._common` reads that state and passes it in, so the kernel stays testable in isolation.

The sparse and modulation paths keep the fp16 contract. `rms_norm` and `l2_norm` guard the zero-vector case with a masked denominator instead of a clamp, and `saturate` and `cubic` bound their tails structurally, so no operator relies on a gradient-breaking clamp. Two module-level caches inside `spectral` (`_PROBE_CACHE` and `_EYE_CACHE`) reuse a probe vector and an identity mask per `(device, dtype, size)`, which keeps the hot path free of new allocations.

-----

<a id="further-exploration"></a>
## Further exploration

- [model](../model/README.md): configuration and the numeric constants the kernels consume.
- [dense](../../crates/hybrid/README.md): the forward and learning paths that call these operators.
- [learning](../hybrid/learning/README.md): the state adapters that delegate to this crate.
