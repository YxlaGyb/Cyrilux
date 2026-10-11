---
description: "attention: error-precision modulation for CoreBrain's cross-view prediction error."
kind: "package-reference"
---

# attention

English | [中文](README.zh.md)

## Summary

`crates.attention` holds the error-precision attention layer that sits above `crates.corebrain`. It reads the short and long slot activations and the current cross-view prediction error, maintains an independent slow activation state, and returns a precision-modulated error. It is a mandatory part of the dense PPA, not an optional switch.

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Further exploration](#further-exploration)

-----

<a id="use-this-package"></a>
## Use this package

The layer is constructed automatically by `CoreBrain`.

| Export | Shape | Meaning |
|---|---|---|
| `AttentionConfig` | dataclass | `ema_tau`, `temperature`, `floor` and the shared `top_k`. |
| `ErrorAttention(n_slots, cfg)` | module | Owns the independent slow activation state and computes the precision gain. |
| `ErrorAttention.forward(a_short, a_long, delta)` | `[N, K]` | Returns `AttentionOutput` with the effective error and diagnostic tensors. |

`AttentionOutput` fields:

| Field | Meaning |
|---|---|
| `effective_error` | `gain * delta`, the error used by CoreBrain's Q and P updates. |
| `gain` | Per-slot precision distribution with mean 1 and a positive floor. |
| `salience` | `abs(a_short - a_long_ema)`, the bottom-up salience signal. |
| `top_down` | `topk_wta(abs(a_long_ema))`, the sparse slow target. |
| `a_long_ema` | The independent slow activation state. |

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

`a_long_ema` is updated once per perception step from the batch mean of the long-view activation. It is independent of `P_long_ema`: `P_long_ema` is the slow projection target used by CoreBrain, while `a_long_ema` is the slow activation state used only by attention.

The bottom-up signal is `abs(a_short - a_long_ema)`. The top-down signal is the top-k magnitude support of `a_long_ema`. After adding the two signals, a softmax produces a positive per-slot distribution, and the affine mapping

```text
g = (1 - floor) * K * softmax(g_raw / temperature) + floor
```

keeps the mean at 1 and the minimum at `floor` without a clamp.

CoreBrain keeps its existing  target `topk_wta(long @ P_long_ema)`. Attention does not replace that target. It only computes `delta_eff = g * delta` and sends that effective error into both the predictor and projection updates. The projection update maps the effective error through `Q.T` before the existing WTA mask is applied.

The layer is active only during perception learning. Free-run and echo-frozen phases do not update `a_long_ema`.

-----

<a id="further-exploration"></a>
## Further exploration

- [corebrain](../corebrain/README.md): the carrier layer whose cross-view error is modulated.
- [model](../model/README.md): network construction and lifecycle.
