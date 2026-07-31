"""PPA 架构 2000 步验证 — 汇报: 自由能是否下降, cos(z5) 是否分化, 生成连贯性.

红线 (用户批准): 只跑 2000 步. 不看 PPL/Top-1, 看世界模型内在自洽:
- free_energy 稳步下降 → 大脑在构建内在世界模型
- cos(z5) 分化 → 时序核撕开方向坍缩
- 生成文本的抽象程度 → W_future 是否促使序列在时空上连贯 (无突兀乱码断层)
"""
import torch
import torch.nn.functional as F
from model.pc.dense.core import DensePCNet, DensePCConfig
from model.core.dataset import DualChannelDataset
from torch.utils.data import DataLoader

torch.set_grad_enabled(False)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={dev}")

cfg = DensePCConfig(
    d_l4=256, d_l2=96, d_l3=96, d_l5=64, d_l6=64,
    lr_hebbian=3e-4,
    max_seq_len=256,
)
net = DensePCNet(cfg).to(dev)
print(f"params={cfg.param_count():,}")

ds = DualChannelDataset("dataset/agent_rl_math.jsonl", max_length=128, max_samples=200)
loader = DataLoader(ds, batch_size=16, shuffle=True)
it = iter(loader)

TEXTS = [
    "import torch def foo return x + 1 class Model",
    "they often speak very clearly and write simple english prose",
]

def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def measure():
    z5_all = {}
    for i, t in enumerate(TEXTS):
        bv = torch.tensor([list(t.encode("utf-8"))], dtype=torch.long, device=dev)
        _ = net(bv)
        z5_all[i] = net._z5[0, -1].float().clone()
    return cos(z5_all[0], z5_all[1])

STEP_CAP = 2000
fe_hist = []
step = 0

while step < STEP_CAP:
    try:
        b, _ = next(it)
    except StopIteration:
        it = iter(loader)
        b, _ = next(it)
    b = b.to(dev)
    stats = net.learn(b)
    step += 1
    fe_hist.append(float(stats["free_energy"]))
    if step % 200 == 0 or step == STEP_CAP:
        cz = measure()
        print(f"  step {step:4d}  free_energy={fe_hist[-1]:.4f}  "
              f"avg200={sum(fe_hist[-200:])/min(200,len(fe_hist)):.4f}  cos(z5)={cz:.4f}  "
              f"dop={float(stats['dop_gain']):.2f}  ach={float(stats['ach_gain']):.2f}")

cz_f = measure()
fe_first, fe_last = fe_hist[0], fe_hist[-1]
print(f"\nFINAL  cos(z5)={cz_f:.4f}  free_energy: {fe_first:.4f} -> {fe_last:.4f} "
      f"({100*(fe_last-fe_first)/fe_first:+.1f}%)")
print(f"  free_energy 下降={'YES' if fe_last < fe_first * 0.9 else 'NO (未达标)'}")

# ── 生成连贯性 (世界模型解码, 非 LM 分类) ──
print("\n  [Generation]")
for prompt in ["Hello", "import ", "The quick"]:
    out = net.generate(prompt, n_tokens=40, dev=dev)
    safe = out.decode("utf-8", errors="replace")
    ascii_safe = safe[:60].encode("ascii", errors="replace").decode("ascii")
    # 抽象程度: 可打印 ASCII 占比
    printable = sum(32 <= c < 127 for c in out)
    ratio = printable / max(len(out), 1)
    print(f'    "{prompt}" -> "{ascii_safe}"   ascii_ratio={ratio:.2f}')
