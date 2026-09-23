# AGENTS.md

Cyrilux is a project based on the concepts of the primal cause of life and self-organization; please read [@docs/architecture.md](docs/architecture.md) before making any modifications.

## Repository layout

## Commands

## Host sandbox failures

If a required gh, pnpm, build, test, or generator command fails because the sandbox blocks credentials, network, IPC, watching, or nested sandbox-exec, retry unchanged with the narrowest host escalation. Require sandbox evidence; never bypass test failures or the product sandbox.

## Conventions

 - Explicit is better than implicit; layered decoupling; composition over inheritance; no enforcement, but extension points are provided; there is no monolithic Trainer, but extension mechanisms such as hooks, FX, and distributed support are available.
 - No comments in code. Write one only where the logic is genuinely hard to follow, never to restate what the code already says.
 - New architecture = delete the old architecture: when adding a new pattern/architecture, remove the corresponding old code and never keep if-else branches coexisting; always ask before acting: any decision involving architecture replacement must be confirmed with the user first; make no additions and no half-measures: do one thing at a time, do it thoroughly and cleanly, and leave no backdoors.

- **FP16 Ironclad Mandate:** this project forever uses only fp16 or lower precision; model weights, activations, gradients, Hebbian updates, and all tensors must remain fp16; never write `.float()` in any function, never convert to fp32 in the training loop, and never discuss whether to switch to fp32—that option does not exist; core principle: numerical instability is sophistry—biological brains do not have such large values or overflow.
 - **FP16 Numerical Stability Guidelines:** core ban: never use clamp to solve fp16 overflow—clamp is brute force, destroys gradient information, and masks root causes; correct approach: structured methods such as pre-norm (RMSNorm/LayerNorm), weight scaling, and gradient clipping; allowed clamp: `loss.clamp(max=100.0)` only before logging and without affecting gradients; no tensor involved in gradients may be clamped; alternatives: if activations are too large → downstream projection overflow, do not `x.clamp(-N, N)`, instead add RMSNorm before projection; if projection output overflows, multiply the output by scaling factor `1/√H`; if weights grow uncontrollably, inspect the Hebbian update rule/Oja term.
 - Numerical instability is sophistry; biological brains do not have such large values or overflow.

- The transmission form of signals is discrete (sparse, event-driven, bandwidth-saving).
 - The semantic content represented is continuous (graded confidence, prediction error, uncertainty).
 - Local learning uses continuous values (Hebbian updates require graded error signals).
 - Long-term memory/reasoning uses discrete symbols (concept binding, compositional generalization).
 - **Evaluator priors must not enter training signals**: Priors along evaluation/discrimination dimensions (e.g., "common character coverage L") may only serve as a ruler, and are forbidden from flowing into weights through feedback paths (forbidden chain such as `L → q → E → R_world → W_act`); any score fed back into Hebbian/R/W_act is uniformly regarded as an external objective replacing self-maintenance, and constitutes a highest-risk violation.
 - **Corpus frequency ≠ universal language-ness**: `Counter(corpus).most_common(N)` is not "universal language-ness"; it is only "closeness to the distribution of one dataset"; if the corpus is Chinese, then this prior is a Chinese prior; such channels may appear only in reporting terms, and must not participate in any learning signal.
 - **Language-specific patches go only into reports**: Adjusting temperature by language/arm, language-detection gating, changing learning rules for a certain arm's metric—any language-specific change is forever allowed only in reporting/measurement, and forever forbidden from entering the model.
 - **Ladder/benchmark metrics are only fossil records**: n-gram ladder, bpb, top-k hit rate are diagnostic fossils; they must not be treated as optimization targets for changing architecture; once structure is changed "to climb the ladder," the ladder turns from a record into a loss function—that is heteronomy, and it slides toward the Transformer camp.
 - **Universal language-ness criterion**: Universal discrimination should "not know what language it is speaking, and react only to structured content"; any discrimination dimension that requires first naming a language to take effect is external injection.

- **One GPU transfer > N `.item()` synchronizations**
 - Ban `.item()` reading GPU tensors — use `.shape[0]` instead of `.sum().item()`
 - Ban creating new GPU tensors in hot paths — preallocate persistently, use `.zero_()` inside loops
 - Ban `clone() + copy_()` for writing slices — always use in-place `+=` or `scatter_add_`
 - Ban for-loop writing GPU tensor scalars — use `torch.rand` / batch indexing instead
 - Ban `.cpu().clone()` — `.cpu()` already returns a new tensor
 - Ban using PyTorch to compute scalars — use `math.tanh`/`math.exp` instead of `torch.tensor(x).item()`

## Documentation

Read [@docs/CONVENTIONS.md](docs/CONVENTIONS.md) before writing or editing any document, README, or language pair; it owns the rules for pairs, README shape, and prose.

## Defensive patterns

Read [@docs/defensive-patterns.md](docs/defensive-patterns.md) before lifecycle, concurrency, subprocess, or teardown work.

## Editing these instructions

`CLAUDE.md` only includes this file; edit `AGENTS.md`. Keep each rule self-contained while linking high-level docs, and condense when clarity survives.