"""
灾难性遗忘验证实验: 极端分布任务 (中文 → 英文噪声)

Phase 1: 无回放基线 (纯 CE, 无 MemoryBank, 无 Sniffer)
  - 训练 task_zh 到收敛 → 记录 CE
  - 继续训练 task_en (相同模型, 无记忆回放)
  - 回测 task_zh CE → 预期剧烈反弹 (灾难性遗忘)

Phase 2: 有回放对照 (PC + 多巴胺 + MemoryBank + Sniffer)
  - 训练 task_zh → 存入 exemplars
  - 训练 task_en (带回放 + 遗忘嗅探)
  - 回测 task_zh CE → 预期保持稳定

用法: python run_forgetting_test.py
"""
import os, sys, json, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model.pc_layers import PCLocalDynamicMiniMind
from model.model_minimind import MiniMindConfig
from model.pc_core import DopamineSignal
from continual.memory_bank import MemoryBank
from continual.forgetting_sniffer import ForgettingSniffer
from continual.offline_replay import OfflineReplayer
from trainer_utils import get_lr, setup_seed

# ═══════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════
ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT, 'dataset')
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 24
MAX_SEQ_LEN = 256
LR = 3e-4
EPOCHS = 8  # 每任务训练轮数
EVAL_SAMPLES = 500
REPLAY_RATIO = 5
BANK_SIZE = 2000

TASK_ZH = os.path.join(DATASET_DIR, 'task_zh.jsonl')
TASK_EN = os.path.join(DATASET_DIR, 'task_en.jsonl')

print(f'Device: {DEVICE}')
print(f'Batch: {BATCH_SIZE}, SeqLen: {MAX_SEQ_LEN}, LR: {LR}')
print(f'Task zh: {TASK_ZH}')
print(f'Task en: {TASK_EN}')


# ═══════════════════════════════════════════════════════════════════
# 数据集
# ═══════════════════════════════════════════════════════════════════

class ByteDataset(Dataset):
    """与 train_pc_unified.py 完全一致的 _LocalDataset。"""
    def __init__(self, data_path, max_length=128, max_samples=None):
        super().__init__()
        self.byte_tensors = []
        self.label_tensors = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                sample = json.loads(line)
                byte_seq = str(sample['text']).encode('utf-8')[:max_length]
                padded = byte_seq.ljust(max_length, b'\x00')
                t = torch.frombuffer(bytearray(padded), dtype=torch.uint8).clone()
                self.byte_tensors.append(t)
                lbl = t.clone()
                lbl[t == 0x00] = -100
                self.label_tensors.append(lbl.to(torch.long))

    def __len__(self):
        return len(self.byte_tensors)

    def __getitem__(self, index):
        return self.byte_tensors[index].clone(), self.label_tensors[index].clone()


# ═══════════════════════════════════════════════════════════════════
# 评估
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, data_path, max_samples=EVAL_SAMPLES, desc=''):
    ds = ByteDataset(data_path, max_length=MAX_SEQ_LEN, max_samples=max_samples)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    total_ce = 0.0
    total_tokens = 0
    model.eval()
    for bt, lt in loader:
        bt, lt = bt.to(DEVICE), lt.to(DEVICE)
        pos = model.get_position_embeddings(bt.size(1), DEVICE)
        _, ce = model.forward_with_ce(bt, lt, pos)
        n_valid = (lt != -100).sum().item()
        total_ce += ce.item() * n_valid
        total_tokens += n_valid
    avg_ce = total_ce / max(total_tokens, 1)
    ppl = math.exp(min(avg_ce, 20))
    model.train()
    print(f'  [Eval{desc}] CE={avg_ce:.4f}, PPL={ppl:.2f} (n={total_tokens} tokens)')
    return avg_ce, ppl


# ═══════════════════════════════════════════════════════════════════
# 单任务训练 (无回放)
# ═══════════════════════════════════════════════════════════════════

def train_task(model, optimizer, data_path, epochs=EPOCHS, desc=''):
    ds = ByteDataset(data_path, max_length=MAX_SEQ_LEN, max_samples=None)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    total_steps = len(loader) * epochs
    global_step = 0
    best_ce = float('inf')

    for epoch in range(epochs):
        for bt, lt in loader:
            bt, lt = bt.to(DEVICE), lt.to(DEVICE)
            bsz, seq_len = bt.shape
            global_step += 1

            pos = model.get_position_embeddings(seq_len, DEVICE)
            _, ce_loss = model.forward_with_ce(bt, lt, pos)
            total_loss = ce_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            if trainable:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)

            current_lr = get_lr(global_step, total_steps, LR)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr
            optimizer.step()

            ce_val = ce_loss.item()
            if ce_val < best_ce:
                best_ce = ce_val

            if global_step % 200 == 0:
                print(f'  [{desc}] Step {global_step}/{total_steps} CE={ce_val:.4f} lr={current_lr:.2e}')

    print(f'  [{desc}] Done. Best CE={best_ce:.4f}')
    return best_ce


# ═══════════════════════════════════════════════════════════════════
# 持续学习训练 (有回放)
# ═══════════════════════════════════════════════════════════════════

def train_continual(model, optimizer, task_order, epochs=EPOCHS):
    memory_bank = MemoryBank(max_per_task=BANK_SIZE)
    sniffer = ForgettingSniffer(
        memory_bank=memory_bank, model=model,
        check_interval=200, threshold=1.2, repair_steps=10,
    )
    offline_replayer = OfflineReplayer(memory_bank, model)

    for task_id in task_order:
        data_path = os.path.join(DATASET_DIR, f'task_{task_id}.jsonl')
        ds = ByteDataset(data_path, max_length=MAX_SEQ_LEN, max_samples=None)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        total_steps = len(loader) * epochs
        global_step = 0

        print(f'\n--- Continual Task {task_id} ({data_path}) ---')

        for epoch in range(epochs):
            for bt, lt in loader:
                bt, lt = bt.to(DEVICE), lt.to(DEVICE)
                bsz, seq_len = bt.shape
                global_step += 1

                # 主训练步 (纯 CE)
                pos = model.get_position_embeddings(seq_len, DEVICE)
                _, ce_loss = model.forward_with_ce(bt, lt, pos)
                total_loss = ce_loss

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                if trainable:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                current_lr = get_lr(global_step, total_steps, LR)
                for pg in optimizer.param_groups:
                    pg['lr'] = current_lr
                optimizer.step()

                # 记忆回放 (每 REPLAY_RATIO 步)
                if memory_bank.total > 0 and global_step % REPLAY_RATIO == 0 and not sniffer.is_repairing:
                    replay_ex = memory_bank.sample(BATCH_SIZE, strategy='dopamine')
                    if replay_ex:
                        rb = torch.stack([ex.byte_tensor for ex in replay_ex], dim=0).to(DEVICE)
                        rl = torch.stack([ex.label_tensor for ex in replay_ex], dim=0).to(DEVICE)
                        rp = model.get_position_embeddings(rb.size(1), DEVICE)
                        _, rloss = model.forward_with_ce(rb, rl, rp)
                        optimizer.zero_grad(set_to_none=True)
                        rloss.backward()
                        tr = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                        if tr:
                            torch.nn.utils.clip_grad_norm_(tr, 1.0)
                        optimizer.step()

                # 遗忘嗅探
                forgotten = sniffer.check(global_step, DEVICE)
                if forgotten:
                    repair_lr = sniffer.repair_begin(optimizer, current_lr, DEVICE)
                    print(f'    [Sniffer] FORGOTTEN: {forgotten} — repairing (LR={repair_lr:.2e})')
                    for _ in range(10):
                        replay_data = sniffer.get_replay_batch(BATCH_SIZE, DEVICE)
                        if replay_data is None:
                            break
                        rb, rl = replay_data
                        rp = model.get_position_embeddings(rb.size(1), DEVICE)
                        _, rloss = model.forward_with_ce(rb, rl, rp)
                        optimizer.zero_grad(set_to_none=True)
                        rloss.backward()
                        tr = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                        if tr:
                            torch.nn.utils.clip_grad_norm_(tr, 1.0)
                        optimizer.step()
                    sniffer.repair_end(optimizer, current_lr)
                    print(f'    [Sniffer] Repair complete')

                if global_step % 200 == 0:
                    print(f'  Task {task_id} Step {global_step}/{total_steps} CE={ce_loss.item():.4f}')

        # 任务结束: 采样 exemplars 存入 bank
        n_samples = min(200, len(ds))
        idx = torch.randperm(len(ds))[:n_samples].tolist()
        samples = []
        total_bl = 0.0
        with torch.no_grad():
            for i in idx:
                bt, lt = ds[i]
                samples.append((bt, lt))
                x, y = bt.unsqueeze(0).to(DEVICE), lt.unsqueeze(0).to(DEVICE)
                p = model.get_position_embeddings(x.size(1), DEVICE)
                _, bl = model.forward_with_ce(x, y, p)
                total_bl += bl.item()
        avg_bl = total_bl / max(len(idx), 1)
        memory_bank.add_samples(task_id, samples, dopamine_score=0.5, baseline_loss=avg_bl)
        print(f'  Task {task_id}: {n_samples} exemplars → bank (total={memory_bank.total})')

    return memory_bank


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    setup_seed(42)

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: 无回放基线 (致命测试)
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('PHASE 1: 无回放基线 — 训练 zh → 训练 en → 回测 zh')
    print('='*70)

    lm_config = MiniMindConfig(hidden_size=256, num_hidden_layers=4, use_moe=False)
    model1 = PCLocalDynamicMiniMind(lm_config).to(DEVICE)
    params = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    print(f'Model params: {params/1e6:.2f}M')

    optim1 = torch.optim.AdamW(
        list(model1.temporal_proj.parameters()) +
        list(model1.topdown_proj.parameters()) +
        [p for n, p in model1.model.named_parameters() if p.requires_grad],
        lr=LR, betas=(0.9, 0.95), fused=True,
    )

    # 预热 (吸收 cudnn benchmark 首步延迟)
    with torch.no_grad():
        dummy = torch.randint(0, 256, (BATCH_SIZE, MAX_SEQ_LEN), device=DEVICE).long()
        dummy_pos = model1.get_position_embeddings(MAX_SEQ_LEN, DEVICE)
        model1.forward_with_ce(dummy, dummy, dummy_pos)
    print('Warmup done')

    T0 = time.time()

    # 1a: 训练 task_zh
    print('\n--- [Phase 1a] 训练 task_zh ---')
    train_task(model1, optim1, TASK_ZH, epochs=EPOCHS, desc='P1-zh')

    # 1b: 评估 task_zh (训练后)
    ce_zh_p1_before, _ = evaluate(model1, TASK_ZH, desc='P1-zh-after-train')

    # 1c: 训练 task_en (同一模型, 无回放!)
    print('\n--- [Phase 1c] 训练 task_en (无回放, 灾难性遗忘即将发生...) ---')
    train_task(model1, optim1, TASK_EN, epochs=EPOCHS, desc='P1-en')

    # 1d: 评估 task_zh (训练 en 后)
    ce_zh_p1_after, _ = evaluate(model1, TASK_ZH, desc='P1-zh-after-en')
    # 1e: 评估 task_en (训练后 — 学得怎么样?)
    ce_en_p1, _ = evaluate(model1, TASK_EN, desc='P1-en-after-train')

    delta_p1 = ce_zh_p1_after - ce_zh_p1_before
    print(f'\n>>> Phase 1 结果:')
    print(f'    task_zh CE: {ce_zh_p1_before:.4f} → {ce_zh_p1_after:.4f} (Δ={delta_p1:+.4f})')
    print(f'    task_en CE: {ce_en_p1:.4f}')
    verdict_p1 = '✅ 灾难性遗忘确认!' if delta_p1 > 0.5 else '⚠️ 遗忘不明显'
    print(f'    {verdict_p1}')
    time_p1 = time.time() - T0

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: 有回放对照 (PC + MemoryBank + Sniffer)
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('PHASE 2: 有回放对照 — MemoryBank + Sniffer 保护')
    print('='*70)

    setup_seed(42)  # 相同初始化
    model2 = PCLocalDynamicMiniMind(lm_config).to(DEVICE)
    optim2 = torch.optim.AdamW(
        list(model2.temporal_proj.parameters()) +
        list(model2.topdown_proj.parameters()) +
        [p for n, p in model2.model.named_parameters() if p.requires_grad],
        lr=LR, betas=(0.9, 0.95), fused=True,
    )

    # 预热
    with torch.no_grad():
        model2.forward_with_ce(dummy, dummy, dummy_pos)
    print('Warmup done')

    T1 = time.time()

    # 2a: 训练 task_zh
    print('\n--- [Phase 2a] 训练 task_zh ---')
    train_task(model2, optim2, TASK_ZH, epochs=EPOCHS, desc='P2-zh')

    # 2b: 评估 task_zh (训练后)
    ce_zh_p2_before, _ = evaluate(model2, TASK_ZH, desc='P2-zh-after-train')

    # 2c: 收集 exemplars → 存入 bank
    print('\n--- [Phase 2c] 收集 zh exemplars → MemoryBank ---')
    ds_zh = ByteDataset(TASK_ZH, max_length=MAX_SEQ_LEN, max_samples=None)
    n_ex = min(500, len(ds_zh))
    idx = torch.randperm(len(ds_zh))[:n_ex].tolist()
    samples = []
    total_bl = 0.0
    memory_bank = MemoryBank(max_per_task=BANK_SIZE)
    with torch.no_grad():
        for i in idx:
            bt, lt = ds_zh[i]
            samples.append((bt, lt))
            x, y = bt.unsqueeze(0).to(DEVICE), lt.unsqueeze(0).to(DEVICE)
            p = model2.get_position_embeddings(x.size(1), DEVICE)
            _, bl = model2.forward_with_ce(x, y, p)
            total_bl += bl.item()
    avg_bl = total_bl / max(len(idx), 1)
    memory_bank.add_samples('zh', samples, dopamine_score=0.5, baseline_loss=avg_bl)
    print(f'  {n_ex} zh exemplars → bank')

    # 2d: 训练 task_en (带 MemoryBank 回放)
    print('\n--- [Phase 2d] 训练 task_en (带回放! MemoryBank 保护中...) ---')
    sniffer = ForgettingSniffer(
        memory_bank=memory_bank, model=model2,
        check_interval=200, threshold=1.2, repair_steps=10,
    )
    offline_replayer = OfflineReplayer(memory_bank, model2)

    ds_en = ByteDataset(TASK_EN, max_length=MAX_SEQ_LEN, max_samples=None)
    loader_en = DataLoader(ds_en, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    total_steps = len(loader_en) * EPOCHS
    global_step = 0

    for epoch in range(EPOCHS):
        for bt, lt in loader_en:
            bt, lt = bt.to(DEVICE), lt.to(DEVICE)
            global_step += 1

            # 主训练
            pos = model2.get_position_embeddings(bt.size(1), DEVICE)
            _, ce_loss = model2.forward_with_ce(bt, lt, pos)
            optim2.zero_grad(set_to_none=True)
            ce_loss.backward()
            trainable = [p for p in model2.parameters() if p.requires_grad and p.grad is not None]
            if trainable:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            current_lr = get_lr(global_step, total_steps, LR)
            for pg in optim2.param_groups:
                pg['lr'] = current_lr
            optim2.step()

            # 记忆回放
            if memory_bank.total > 0 and global_step % REPLAY_RATIO == 0 and not sniffer.is_repairing:
                replay_ex = memory_bank.sample(BATCH_SIZE, strategy='dopamine')
                if replay_ex:
                    rb = torch.stack([ex.byte_tensor for ex in replay_ex], dim=0).to(DEVICE)
                    rl = torch.stack([ex.label_tensor for ex in replay_ex], dim=0).to(DEVICE)
                    rp = model2.get_position_embeddings(rb.size(1), DEVICE)
                    _, rloss = model2.forward_with_ce(rb, rl, rp)
                    optim2.zero_grad(set_to_none=True)
                    rloss.backward()
                    tr = [p for p in model2.parameters() if p.requires_grad and p.grad is not None]
                    if tr:
                        torch.nn.utils.clip_grad_norm_(tr, 1.0)
                    optim2.step()

            # 遗忘嗅探
            forgotten = sniffer.check(global_step, DEVICE)
            if forgotten:
                repair_lr = sniffer.repair_begin(optim2, current_lr, DEVICE)
                print(f'    [Sniffer] FORGOTTEN: {forgotten} — repairing')
                for _ in range(10):
                    replay_data = sniffer.get_replay_batch(BATCH_SIZE, DEVICE)
                    if replay_data is None:
                        break
                    rb, rl = replay_data
                    rp = model2.get_position_embeddings(rb.size(1), DEVICE)
                    _, rloss = model2.forward_with_ce(rb, rl, rp)
                    optim2.zero_grad(set_to_none=True)
                    rloss.backward()
                    tr = [p for p in model2.parameters() if p.requires_grad and p.grad is not None]
                    if tr:
                        torch.nn.utils.clip_grad_norm_(tr, 1.0)
                    optim2.step()
                sniffer.repair_end(optim2, current_lr)
                print(f'    [Sniffer] Repair complete')

            if global_step % 200 == 0:
                print(f'  Task en Step {global_step}/{total_steps} CE={ce_loss.item():.4f} '
                      f'bank_size={memory_bank.total}')

    # 2e: 评估 task_zh (训练 en 后)
    ce_zh_p2_after, _ = evaluate(model2, TASK_ZH, desc='P2-zh-after-en')
    # 2f: 评估 task_en (训练后 — 被 Sniffer 过度保护了吗?)
    ce_en_p2, _ = evaluate(model2, TASK_EN, desc='P2-en-after-train')

    delta_p2 = ce_zh_p2_after - ce_zh_p2_before
    print(f'\n>>> Phase 2 结果:')
    print(f'    task_zh CE: {ce_zh_p2_before:.4f} → {ce_zh_p2_after:.4f} (Δ={delta_p2:+.4f})')
    print(f'    task_en CE: {ce_en_p2:.4f}')
    verdict_p2 = '✅ 无遗忘! 持续学习有效!' if abs(delta_p2) < 0.5 else '⚠️ 仍有遗忘'
    print(f'    {verdict_p2}')
    time_p2 = time.time() - T1

    # ═══════════════════════════════════════════════════════════════
    # 最终对比
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('最终对比: 灾难性遗忘验证')
    print('='*70)
    print(f'')
    print(f'  {"条件":<25} {"task_zh CE(ex en后)":<19} {"task_en CE":<12} {"Δ_zh":<10} {"耗时":<10}')
    print(f'  {"-"*25} {"-"*19} {"-"*12} {"-"*10} {"-"*10}')
    print(f'  {"无回放 (Phase 1)":<25} {ce_zh_p1_after:<19.4f} {ce_en_p1:<12.4f} {delta_p1:<+10.4f} {time_p1:<10.1f}s')
    print(f'  {"有回放 (Phase 2)":<25} {ce_zh_p2_after:<19.4f} {ce_en_p2:<12.4f} {delta_p2:<+10.4f} {time_p2:<10.1f}s')
    print(f'')

    if delta_p1 > 0.5 and abs(delta_p2) < 0.5:
        print('🎉 结论: 持续学习系统有效防止了灾难性遗忘!')
        print(f'   无回放: CE_zh 上升 {delta_p1:.2f} (灾难性遗忘)')
        print(f'   有回放: CE_zh 仅变化 {delta_p2:.2f} (几乎不变)')
    elif delta_p1 > 0.5 and delta_p2 > 0.5:
        print('⚠️ 两种条件下都有明显遗忘, 但需比较幅度差异。')
        print(f'   无回放 Δ={delta_p1:.4f} vs 有回放 Δ={delta_p2:.4f}')
        if delta_p2 < delta_p1:
            print(f'   有回放遗忘幅度降低 {((delta_p1-delta_p2)/delta_p1*100):.0f}%')
    else:
        print('⚠️ 遗忘不显著, 可能任务不够极端或训练量不足。')
        print(f'   无回放 Δ={delta_p1:.4f}, 有回放 Δ={delta_p2:.4f}')

    print(f'\n实验完成。总耗时: {time_p1+time_p2:.1f}s')


if __name__ == '__main__':
    main()
