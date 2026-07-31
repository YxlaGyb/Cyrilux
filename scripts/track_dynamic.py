"""Track dynamic neuron growth on real data — cos(Z5) oscillation."""
import torch
from model.pc.dense.core import DensePCNet, DensePCConfig
from model.core.dataset import DualChannelDataset
from torch.utils.data import DataLoader

torch.set_grad_enabled(False)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={dev}")

cfg = DensePCConfig(
    d_l4=512, d_l2=256, d_l3=256, d_l5=256, d_l6=128,
    prune_interval=500,
    lr_hebbian=3e-4,
    death_threshold=5e-5,
    active_size_lower_bound=32,
    l5_kwta_ratio=0.10,
    l5_rank_gain=True,
    l5_oja_norm_high=1.3,
    l5_oja_norm_low=0.6,
)
net = DensePCNet(cfg).to(dev)
print(f"params={cfg.param_count():,}")

ds = DualChannelDataset("dataset/agent_rl_math.jsonl", max_length=128, max_samples=200)
loader = DataLoader(ds, batch_size=16, shuffle=True)
it = iter(loader)

cos_history = []
active_history = {k: [] for k in ("l4", "l2", "l3", "l5", "l6")}
step = 0

# Measure cos(Z5) between two different contexts from the same batch
while step < 2000:
    try:
        b, l = next(it)
    except StopIteration:
        it = iter(loader)
        b, l = next(it)
    b, l = b.to(dev), l.to(dev)

    net.learn(b, l[:, 1:].long())
    step += 1

    if step % 10 == 0:
        _ = net(b[:1, :64])
        z5_a = net._z5[0, -1].float().clone()
        _ = net(b[1:2, :64])
        z5_b = net._z5[0, -1].float().clone()
        cos_z = torch.nn.functional.cosine_similarity(z5_a.unsqueeze(0), z5_b.unsqueeze(0)).item()
        cos_history.append((step, cos_z))

    if step % 200 == 0:
        sz = f"l4={net.active_size['l4']} l2={net.active_size['l2']} l3={net.active_size['l3']} l5={net.active_size['l5']} l6={net.active_size['l6']}"
        print(f"  step {step:4d}  cos(Z5)={cos_history[-1][1]:.4f}  active=[{sz}]")

print(f"\ncos(Z5) over time:")
for step, c in cos_history[::10]:
    print(f"  {step:4d}: {c:.4f}")

deltas = [abs(cos_history[i+1][1] - cos_history[i][1]) for i in range(len(cos_history)-1)]
mean_delta = sum(deltas) / len(deltas)
max_delta = max(deltas)
print(f"\n  mean abs delta(cos)={mean_delta:.4f}  max delta={max_delta:.4f}")
if mean_delta > 0.001:
    print("  Oscillation detected! Dynamic growth mechanism active.")
else:
    print("  Minimal oscillation.")
