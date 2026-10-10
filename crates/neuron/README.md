---
description: "neuron: dynamic-neuron mechanisms, main-chain topology pruning and the birth and death of the memory units."
kind: "package-reference"
---

# neuron

English | [中文](README.zh.md)

## Summary

`crates.neuron` holds the dynamic-neuron mechanisms of the dense PPA: topology pruning of the main chain and the birth and death of the competing memory units. Both reshape the live network in place, re-registering the affected tensors so every row-mapped weight stays aligned with its downstream columns. The crate is a mandatory part of the network, not an optional switch.

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Further exploration](#further-exploration)

-----

<a id="use-this-package"></a>
## Use this package

Both engines take a single network and keep a reference to it. They are constructed by `HybridPCNet.__init__`.

| Export | Shape | Meaning |
|---|---|---|
| `PruningEngine(net)` | Engine | Holds the network and drives topology remodelling. Constructed automatically as `net.pruner`. |
| `MemoryPool(net)` | Engine | Holds the network and drives the memory-unit life cycle. Constructed automatically as `net.mem_pool`. |
| `ROW_STATE` | dict | Registry of row-indexed state: each entry is name, layer, form, and handler. |

Remodelling. The verdict side only writes the death contract; the compaction side is the only step that changes shapes.

| Export | Shape | Meaning |
|---|---|---|
| `net.maybe_prune(step)` | Seam | Metabolism-triggered remodelling entry, owned by `crates.model`. Polled every 8 steps; returns the trigger cause or None, and delegates to the two methods below. |
| `pruner.decide()` | Verdict | Writes the death flag and probation ledger only: registers a death row, revives one whose row norm recovered above the threshold, expires the probation, then flags the weakest rows. Shapes and `active_size` are untouched. |
| `pruner.compact()` | Compaction | The only physical step: permutes each layer's row-mapped weights together with their downstream columns, gathers the L4 auxiliaries, shrinks to the surviving width, and reclaims memory. |
| `pruner.has_releasable()` | Guard | True when some layer holds at least 1% of its width in closed slots, the precondition for a compaction round. |

Memory pool.

| Export | Shape | Meaning |
|---|---|---|
| `mem_pool.update(ctx, sh)` | Per step | Fitness, gain, then death and birth of the memory units. Called after the LM head update. |
| `mem_pool.famine_kill()` | Death round | Evicts the lowest-gain unit (index 0 immune, K floor 1). Content and identity are lost permanently. |
| `mem_pool.birth(parent)` | Birth | Splits a child off the parent unit and appends a zero `W1` block. |
| `mem_pool.resize(keep)` | Re-index | Re-registers the unit buffers and the `W1`/`W1_elig` rows for a new K. |

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

`PruningEngine` splits a remodelling round in two. `decide()`, polled on the metabolism round, only writes the death contract: it registers a death row, clears the flag and the probation counter of a row whose norm has recovered above `death_threshold`, increments the probation of the rest, expires the ones that outlived `death_probation`, and flags the weakest candidates by relative rank. `active_size` and every tensor shape are untouched, so a live row never loses its slot to a shape collapse. `compact()` is the only physical step: it permutes each layer's feed-forward weight, temporal kernel, bias, eligibility trace, death ledger, and STP resources together with their downstream columns, gathers the L4 auxiliaries (`W1`, `W_diff`, `W_state_pred`, the memory units) and the CoreBrain projector state to the same permutation, shrinks to the surviving width, and reclaims the freed memory. Eviction is probation first, then relative rank by row norm (L4 additionally scores the W1 contribution), down to the configured floor sizes.

The memory pool keeps the K units in a fixed layout: their state lives in `_mem_m`, `_mem_a`, `_mem_g`, and `_mem_q`, and their decoder rows sit in the `W1` block from `dim_4 + bind_slot_dim` onward. Both death and birth re-register that block, so the invariant `net._lm_in == W1.shape[0]` holds after every event: it is the single merge point that keeps the `zh` columns and the `W1` rows from ever misaligning.

Fitness is the batch-mean cosine between a unit's memory and the error projected back into `zh` space, tracked as the slow `_mem_q` EMA. Gain follows multiplicatively with a bounded decay, so a unit with zero contribution drives `g` toward zero. Death fires when `g` stays below `mem_g_min` for `mem_death_steps`; birth fires when the fast error EMA exceeds the slow EMA times `mem_birth_thresh` and the cooldown has expired. Index 0 is immune to death, and `mem_k_max` caps growth.

The famine kill is the direct loss of a metabolic death round: it evicts the lowest-gain unit at once, and that content and identity are gone permanently. A later birth clones a new unit from a surviving parent, so K can rebound; the irreversibility lives in `active_size`, which only ever shrinks.

-----

<a id="further-exploration"></a>
## Further exploration

- [hybrid](../hybrid/README.md): the engines that call into this crate and the network they drive.
- [model](../model/README.md): network construction, configuration, and lifecycle.
- [corebrain](../corebrain/README.md): the slot layer whose projector state is permuted together with L4.
