---
name: cyrene-eval
description: |
  Evaluate Cyrene predictive coding models — test language ability, memory, representation quality,
  and compare multiple checkpoints. Use whenever the user asks to test a model, evaluate a checkpoint,
  check model performance, run model comparison, measure PPL, or "see how well this model works."
  Also use when the user asks about model health, wants to know if a model has collapsed,
  or says things like "test model X", "evaluate model Y", "run model Z", "看看模型", "测试语言能力".
---

# Cyrene Model Evaluation

Load a Cyrene checkpoint and run a standard diagnostic battery. The goal is to
understand what the model can and cannot do, not to tune hyperparameters.

## Prerequisites

- `model.model_cyrene.CyreneModel` for loading checkpoints
- `model.training.dataset.DualChannelDataset` for PPL testing
- `torch.utils.data.DataLoader` with batch_size=1

## Step 1: Load and report metadata

```python
m = CyreneModel.load(path)
```

Report: step count, total neurons, neurons per layer (0, 10, 11, 12, 13, 14),
total synapses, top_layer (should be 13 = L5 for new architecture, 10 for migrated).

## Step 2: Input sensitivity

Feed 3-4 different text types (English prose, Python code, math, Chinese UTF-8) and
record for each: entropy of softmax(logits), top-3 predicted bytes. Use `compute_lm_logits`.

Healthy: entropy between 3.0 and 5.5. Entropy near 0 = mode collapse.
Entropy near 5.55 = uniform (model hasn't learned anything yet).
Top-3 predictions should DIFFER between different input types — if they're identical,
the hidden layer has no input selectivity.

## Step 3: Perplexity

Run on `dataset/sft_t2t.jsonl` (or user-specified dataset). Test at least 300 tokens.
Use the standard per-position prediction loop:

- For each position p in sequence, feed `byte_seq[:,:,:p]` as context
- Get logits via `lm_head.predict_logits(pool, top_layer)`
- Compute CE loss against `labels[0, p-1]`
- Track top-1 accuracy

Report: PPL, CE loss, top-1 accuracy, tokens/sec.
Healthy PPL: < 300 for a trained model. PPL near 256 = random baseline.
PPL > 1000 = model is confidently wrong (overfit on byte frequencies without context).

## Step 4: Memory

Feed the SAME text repeatedly (3-5 times) and check if next-byte prediction improves.
Use a short text (~30 bytes). Measure per-round accuracy.
A model that learns from repetition should show accuracy increasing round-to-round.

## Step 5: Generation

For 2-3 prompts, generate ~30-40 tokens with top-k sampling (k=15, temperature=0.7).
Decode as UTF-8, display ASCII-safe output.
Look for: preserved prompt prefix, word-like letter sequences separated by spaces.
Pure garbage bytes = model hasn't learned letter distribution yet.

## Step 6: Hidden layer selectivity

Feed two different text types, extract Z from the top layer (L5).
Compute cosine similarity between the Z vectors.
cos < 0.95 = some selectivity emerging. cos > 0.99 = all inputs produce the same
representation (hidden layer dead or weights uniform).

Also compute MU (F_MU) vectors and compare: MU often has more selectivity than Z,
because Z is smoothed by inertia. The gap between cos(MU) and cos(Z) tells you
how much selectivity the inertia is killing.

## Step 7: Summary table

When testing multiple models, produce a comparison table:

| Model | Steps | PPL | CE | Top-1 | Entropy | cos(Z) | Generation |
|-------|-------|-----|-----|-------|---------|--------|------------|

Also compare against previous baselines if known (e.g., "model5 was 256 PPL before").

## Interpretation guide

| Symptom | Diagnosis |
|---------|-----------|
| Entropy = 0 | LM Head mode collapse |
| Entropy = 5.55, PPL = 256 | Model is random, hasn't learned anything yet |
| PPL > 1000, entropy 2-4 | Confidently wrong — LM Head overfit without selectivity |
| cos(Z) > 0.99 for all inputs | Hidden layer dead or uniform weights |
| cos(MU) < 0.90 but cos(Z) > 0.99 | Inertia killing selectivity that MU has |
| Generation has word-like structure | Model learned letter distribution, next step is spelling |
