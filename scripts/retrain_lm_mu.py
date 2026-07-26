"""加载 model5, 清零 LM head, 用 MU 快速重训 (via step())."""
import torch, math, time
from model.model_cyrene import CyreneModel
from model.core.dataset import DualChannelDataset
from torch.utils.data import DataLoader

m = CyreneModel.load('out/model5/final.pt')
m.bridge.set_warmup(0)
m.pool.lm_weight.zero_()
m.config.use_mu_lm = True  # LM head 训练和推理都用 MU

# 冻结隐藏层 Hebbian (只训 LM head)
_orig = m.pool.hebbian_pass, m.pool.hebbian_temporal, m.pool.hebbian_topdown
m.pool.hebbian_pass = lambda *a, **kw: 0.0
m.pool.hebbian_temporal = lambda *a, **kw: 0.0
m.pool.hebbian_topdown = lambda *a, **kw: 0.0

print(f'model5: step={m._step} neurons={m.pool.alive.sum().item()}')
print(f'LM head training with MU (use_mu_lm=True)')

# 训练
ds = DualChannelDataset('dataset/sft_t2t.jsonl', max_length=128, max_samples=60)
loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

t0 = time.perf_counter()
total = 0
for byte_seq, labels in loader:
    byte_seq = byte_seq.to(m.device)
    S = byte_seq.shape[-1]
    for pos in range(1, S):
        target = labels[0, pos - 1].item()
        if target == -100:
            continue
        ctx = byte_seq[:, :, :pos].contiguous()
        if ctx.shape[-1] < 13:
            continue
        m.step(ctx, target_byte=target)  # hebbian_lm_head 自动用 MU
        total += 1
        if total >= 2000:
            break
    if total >= 2000:
        break

elapsed = time.perf_counter() - t0
print(f'Trained: {total} steps in {elapsed:.1f}s ({total/elapsed:.0f} s/s)')

# 恢复
m.pool.hebbian_pass, m.pool.hebbian_temporal, m.pool.hebbian_topdown = _orig

# === 测试 ===
ds2 = DualChannelDataset('dataset/sft_t2t.jsonl', max_length=64, max_samples=20)
loader2 = DataLoader(ds2, batch_size=1, shuffle=False, num_workers=0)

def test(use_mu):
    loss_sum = 0; n = 0; correct = 0
    for byte_seq, labels in loader2:
        byte_seq = byte_seq.to(m.device)
        for pos in range(1, byte_seq.shape[-1]):
            target = labels[0, pos - 1].item()
            if target == -100:
                continue
            ctx = byte_seq[:, :, :pos].contiguous()
            if ctx.shape[-1] < 13:
                continue
            m.step(ctx)
            logits = m.lm_head.predict_logits(m.pool, m._top_layer, use_mu=use_mu)
            loss_sum += m.lm_head.cross_entropy_loss(logits, target)
            n += 1
            if max(range(256), key=lambda i: logits[i]) == target:
                correct += 1
            if n >= 500:
                break
        if n >= 500:
            break
    ppl = math.exp(loss_sum / n) if loss_sum / n < 50 else float('inf')
    return ppl, loss_sum / n, 100 * correct / n

print()
ppl_mu, ce_mu, acc_mu = test(use_mu=True)
print(f'PPL(MU→MU) = {ppl_mu:.1f}  CE={ce_mu:.2f}  top1={acc_mu:.1f}%')

ppl_z, ce_z, acc_z = test(use_mu=False)
print(f'PPL(MU→Z)  = {ppl_z:.1f}  CE={ce_z:.2f}  top1={acc_z:.1f}%')

print()
print('=== 对比 ===')
print(f'原始 model5: PPL=256  CE=5.54  (Z→Z, 17K步)')
print(f'新 MU 方案:  PPL={ppl_mu:.0f}  CE={ce_mu:.2f}  (MU→MU, 2K步)')
