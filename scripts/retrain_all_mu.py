"""对 model4/5/6/7 全部做 MU LM head 重训, 然后横评."""
import torch, math, time
from model.model_cyrene import CyreneModel
from model.core.dataset import DualChannelDataset
from torch.utils.data import DataLoader

results = {}

for name in ['model4', 'model5', 'model6', 'model7']:
    print(f'\n{'='*60}')
    print(f'{name}')
    print(f'{"="*60}')
    t_load = time.perf_counter()

    m = CyreneModel.load(f'out/{name}/final.pt')
    m.bridge.set_warmup(0)
    orig_step = m._step
    orig_neurons = m.pool.alive.sum().item()
    orig_syn = m.pool.syn_alive.sum().item()

    # 冻结隐藏层, 清零 LM head, 启用 MU
    _orig = m.pool.learning.hebbian_pass, m.pool.hebbian_temporal, m.pool.hebbian_topdown
    m.pool.learning.hebbian_pass = lambda *a, **kw: 0.0
    m.pool.hebbian_temporal = lambda *a, **kw: 0.0
    m.pool.hebbian_topdown = lambda *a, **kw: 0.0
    m.pool.lm_weight.zero_()
    m.config.use_mu_lm = True

    # 训练
    ds = DualChannelDataset('dataset/sft_t2t.jsonl', max_length=128, max_samples=60)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    total = 0
    t_train = time.perf_counter()
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
            m.step(ctx, target_byte=target)
            total += 1
            if total >= 2000:
                break
        if total >= 2000:
            break
    train_time = time.perf_counter() - t_train

    # 恢复
    m.pool.learning.hebbian_pass, m.pool.hebbian_temporal, m.pool.hebbian_topdown = _orig

    # 测试 PPL (MU→MU)
    ds2 = DualChannelDataset('dataset/sft_t2t.jsonl', max_length=64, max_samples=20)
    loader2 = DataLoader(ds2, batch_size=1, shuffle=False, num_workers=0)
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
            logits = m.lm_head.predict_logits(m.pool, m._top_layer, use_mu=True)
            loss_sum += m.lm_head.cross_entropy_loss(logits, target)
            n += 1
            if max(range(256), key=lambda i: logits[i]) == target:
                correct += 1
            if n >= 500:
                break
        if n >= 500:
            break
    ppl = math.exp(loss_sum / n) if loss_sum / n < 50 else float('inf')

    # 生成
    gen_samples = {}
    for prompt in ['Hello', 'import ']:
        gen = list(prompt.encode())
        for _ in range(30):
            bv = torch.tensor([[b/128.0-1.0 for b in gen[-64:]]], dtype=torch.half, device=m.device)
            m.step(torch.stack([bv, torch.ones_like(bv)], dim=1))
            logits_t = m.pool.forward.compute_lm_logits(m._top_layer, use_mu=True).float() / 0.7
            topv, _ = torch.topk(logits_t, min(15, 256))
            logits_t[logits_t < topv[-1]] = -float('inf')
            probs = torch.softmax(logits_t, dim=-1)
            gen.append(int(torch.multinomial(probs, 1).item()))
        text = bytes(gen).decode('utf-8', errors='replace')
        gen_samples[prompt] = text[:40].encode('ascii', errors='replace').decode('ascii')

    results[name] = {
        'step': orig_step, 'neurons': orig_neurons, 'syn': orig_syn,
        'ppl': ppl, 'ce': loss_sum / n, 'top1': 100 * correct / n,
        'train_s': train_time, 'gen': gen_samples,
    }
    print(f'  orig_step={orig_step} PPL={ppl:.0f} CE={loss_sum/n:.2f} top1={100*correct/n:.1f}% time={train_time:.0f}s')

# === 汇总对比表 ===
print(f'\n\n{"="*80}')
print('四模型横评 (MU LM head, 2000步重训)')
print(f'{"="*80}')
print(f'{"Model":<8} {"原始步数":<10} {"神经元":<8} {"PPL":<10} {"CE":<8} {"Top-1":<8} {"生成(Hello)":<30}')
print(f'{"-"*80}')
for name in ['model4', 'model5', 'model6', 'model7']:
    r = results[name]
    gen = r['gen'].get('Hello', '')
    print(f'{name:<8} {r["step"]:<10} {r["neurons"]:<8} {r["ppl"]:<10.0f} {r["ce"]:<8.2f} {r["top1"]:<8.1f}% {gen:<30}')

print()
print('vs 原始:')
print(f'{"Model":<8} {"原始PPL":<12} {"MU PPL":<12} {"改善":<10}')
for name, orig_ppl in [('model4', 306), ('model5', 256), ('model6', 315), ('model7', 17409)]:
    new_ppl = results[name]['ppl']
    delta = (orig_ppl - new_ppl) / orig_ppl * 100
    print(f'{name:<8} {orig_ppl:<12} {new_ppl:<12.0f} {delta:+.1f}%')
