"""
model48: load model47, switch to English-only data, 200 step buffer flush, clean argmax gen test.
"""

import sys, math, time

sys.path.insert(0, ".")
import torch

torch.set_grad_enabled(False)
from model.model_cyrene import CyreneModel
from model.core.train import TrainingConfig, TrainingLoop
from model.core.dataset import DualChannelDataset
from model.pc.constants import F_Z, F_MU, TOP_LAYER

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={dev}")

m = CyreneModel.load("out/model47/last.pt")
m._top_layer = 10
print(f"loaded: step={m._step}, N={m.pool.N}")

loop = TrainingLoop(
    TrainingConfig(
        hidden_size=64,
        max_seq_len=256,
        batch_size=1,
        hebbian_base_eta=3e-4,
        out_dir="out/model48",
        save_interval=100000,
    )
)
loop.runner = m

# English-only dataset
ds = DualChannelDataset("dataset/en_mini.jsonl", max_length=256, max_samples=2000)
loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
t0 = time.time()

for bi, (b, l) in enumerate(loader):
    if bi >= 200:
        break
    stats = loop.train_step(b.to(dev), l.to(dev))
    if bi % 50 == 0:
        w = m.pool.lm_weight.float()
        cn = w.norm(dim=0)
        l4 = (m.pool.layer == TOP_LAYER) & m.pool.alive
        bias = m.pool.lm_bias.float()
        print(f"+{bi}: lm_norm={cn.mean():.4f} max={cn.max():.4f} bias_nz={(bias.abs() > 0).sum().item()}/256")

m.save("out/model48/flush.pt")
dt = time.time() - t0
print(f"\n{bi + 1} seqs in {dt:.0f}s = {dt / (bi + 1) * 1000:.0f}ms/seq")

# ── Clean argmax gen ──
print("\n=== Generation ===")
m.reset_hidden_state()
prompt_bs = b"machine learn"
n_prompt = len(prompt_bs)
prompt = torch.zeros(1, 64, dtype=torch.long, device=dev)
for i, b in enumerate(prompt_bs):
    prompt[0, i] = b
generated = bytearray()
for _ in range(64):
    m.step(prompt)
    logits = m.pool.forward.compute_lm_logits(TOP_LAYER)
    pred = int(logits.argmax().item())
    generated.append(pred)
    prompt = prompt.roll(-1, dims=-1)
    prompt[0, -1] = pred
print(f"  gen raw: {generated}")
try:
    print(f"  gen txt: {bytes(generated).decode('utf-8', errors='replace')}")
except:
    print(f"  gen txt: decode error")

# Also print per-position top-1 prediction quality
print("\n=== Per-position ===")
m.reset_hidden_state()
for ti in range(min(20, len(prompt_bs))):
    m.step(prompt[:, ti : ti + 1])
    logits = m.pool.forward.compute_lm_logits(TOP_LAYER)
    pred = int(logits.argmax().item())
    target = prompt_bs[ti] if ti < len(prompt_bs) else -1
    print(
        f"  pos {ti}: pred={pred} ({chr(pred) if 32 <= pred < 127 else '?'}) target={target} ({chr(target) if 32 <= target < 127 and target >= 0 else '?'}) ok={'YES' if pred == target else ''}"
    )
