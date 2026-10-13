---
description: "trace-modulated eligibility plasticity, the credit-assigned rule every dense weight update now runs through."
kind: "package-reference"
---

# rtr

English | [中文](README.zh.md)

## Summary

`crates.rtr` owns the trace-modulated eligibility rule that replaces the raw co-occurrence outer product on every dense weight. Each update site accumulates an eligibility trace `E` from the instantaneous co-activity label, then applies it modulated by a residual `m = r - w . z_hat`, where `r` is the in-body free-energy currency and `w` is a value ledger with one column per phase over the normalised pooled state. The package holds the kernel, the trace adapter over the per-weight trace buffers, and the single survival-signal write point. It depends on `torch` and `crates.model` constants only.

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Further exploration](#further-exploration)

-----

<a id="use-this-package"></a>
## Use this package

Import the operators from the package root.

```python
from crates.rtr import rtr_apply, trace, update_survival_signal
```

| Export | Module | Shape | Meaning |
|---|---|---|---|
| `elig_accum(E, dW_raw, gamma)` | `kernel` | `[out, in]` | Eligibility trace `E = gamma * E + dW_raw`, in place on the trace buffer. |
| `rtr_residual(w, z, r, eta_v)` | `kernel` | `[1]` | Residual `m = r - w . z_hat` over the normalised state, tracking `w += eta_v * m * z_hat` in place. |
| `rtr_apply(E, m)` | `kernel` | `[out, in]` | Trace-modulated update `m * E / (norm(E) + 1e-6)`. |
| `trace(net, wname, dW_raw)` | `trace` | `[out, in]` | Adapter: fetches `net.{wname}_elig` and accumulates it with `ELIG_GAMMA`. |
| `update_survival_signal(net, ctx)` | `signals` | `None` | Writes `net._survival_signal` once per step from `net._metab_R` and the current phase column of `net._rtr_w`. |

Every site follows the same three lines: build the raw co-activity label, fold it into the trace, then write the modulated trace into the weight.

```python
dW_raw = (pre.transpose(-2, -1) @ rms_norm(post)).mean(dim=0)
E = trace(net, "W_42", dW_raw)
dW = rtr_apply(E, net._survival_signal)
net.W_42[:dim].data += dW * eta
```

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

`elig_accum` keeps the trace as an exponentially forgotten sum of the instantaneous label `dW_raw`. The label is a bare outer product of pre- and post-synaptic activity: it carries no credit on its own and never reaches the weight directly. The trace is what survives across steps and what the residual later modulates.

`rtr_residual` is the credit assignment. `r` is the survival currency (`net._metab_R`, the tanh-compressed free-energy change), and `w` is the ledger column for the current phase. The state is the mean-pooled L4 activation normalised to a unit vector `z_hat`, and the residual `m = r - w . z_hat` measures whether the current state did better or worse than the running expectation along that direction. The update `w += eta_v * m * z_hat` is a least-mean-squares step with no bootstrap term, so it is unconditionally stable for `eta_v < 2` and every step is bounded by `eta_v * |m|`. The ledger is a buffer registered on the model and follows the same per-row pruning as the rest of the L4 state.

`rtr_apply` normalises the trace per site and scales by the residual. The normalisation lives here rather than at the signal write point, so one scalar residual can drive every site at its own trace scale without a division by zero.

`update_survival_signal` is the only write point for `net._survival_signal`, called unconditionally from `_update_metabolism` after `net._metab_R` is settled. Because it runs in the metabolism term of the step, perception steps produce a signal too, not only closed-loop and echo steps. The two ledger columns keep the perception and echo currencies separate, since their free-energy-change scales differ by roughly an order of magnitude.

All arithmetic stays in fp16. The state is a pooled vector that already lives on the device, and the ledger step is one fused scale-and-add, so no gather, no `index_add_`, and no sync is needed.

-----

<a id="further-exploration"></a>
## Further exploration

- [kernels](../kernels/README.md): the leaf operators, including the homeostasis rules that still run beside the trace.
- [learning](../hybrid/learning/README.md): the per-domain sites that call `trace` and `rtr_apply`.
- [model](../model/README.md): the `_rtr_w` ledger buffer and the numeric constants.
