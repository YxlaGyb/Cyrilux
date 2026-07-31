"""方案 C 零成本诊断: 对比 cos(mu5) vs cos(z5) 定位方向锁在哪层."""
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
    max_seq_len=256,
)
net = DensePCNet(cfg).to(dev)

ds = DualChannelDataset("dataset/agent_rl_math.jsonl", max_length=128, max_samples=200)
loader = DataLoader(ds, batch_size=16, shuffle=True)
it = iter(loader)

# 一对不同的输入文本 (避免依赖随机 batch)
texts = [
    "import torch def foo return x + 1 class Model",
    "人工智能的未来在于深度学习与神经网络",
]
def _byte(text):
    return torch.tensor([list(text.encode("utf-8"))], dtype=torch.long, device=dev)

results = {}
for label, text in [("A", texts[0]), ("B", texts[1])]:
    bv = _byte(text)
    _ = net(bv)
    # mu5 = z3 @ W_35.T + bias_l5 (预测, 未归一化)
    mu5 = net._z3 @ net.W_35[:net.active_size["l5"]].T + net.bias_l5[:net.active_size["l5"]]
    results[label] = {
        "mu5": mu5[0, -1].float().clone(),
        "z5": net._z5[0, -1].float().clone(),
        "z3": net._z3[0, -1].float().clone(),
    }

def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

print(f"\ncos(mu5) A vs B (不同输入): {cos(results['A']['mu5'], results['B']['mu5']):.4f}")
print(f"cos(z5)  A vs B (不同输入): {cos(results['A']['z5'], results['B']['z5']):.4f}")
print(f"cos(z3)  A vs B (不同输入): {cos(results['A']['z3'], results['B']['z3']):.4f}")

m = cos(results['A']['mu5'], results['B']['mu5'])
print("\n诊断判定:")
if m > 0.95:
    print(f"  cos(mu5)={m:.4f} > 0.95 → 死结在 L5 之前 (L3/L4 被 pre-norm 锁死)，应在下层归一化动刀")
elif m < 0.8:
    print(f"  cos(mu5)={m:.4f} < 0.8 → L3 输入有区分度, 方向锁确凿在 L5 内部权重分配")
else:
    print(f"  cos(mu5)={m:.4f} 中间地带")
