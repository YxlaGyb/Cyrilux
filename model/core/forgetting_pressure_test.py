"""多任务灾难性遗忘压力测试 — 无回放 vs MemoryBank+Sniffer 4×4 CE 矩阵.

用法:
  virtuoso test forgetting ...

Phase 1: 无回放 (A→B→C→D 灾难性遗忘基线)
Phase 2: MemoryBank + Sniffer 保护 (A→B→C→D 持续学习)
输出: 4×4 CE 矩阵 (行=训练完第 i 个任务, 列=在第 j 个任务上的 CE)
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

from model.continual.forgetting_sniffer import ForgettingSniffer
from model.continual.memory_bank import MemoryBank
from model.core.dataset import DualChannelDataset
from model.core.globals import DEVICE_STR
from pkg.utils.trainer_utils import setup_seed
from model.model_cyrene import CyreneConfig
from model.pc.local_updates import (
    apply_hebbian_updates,
    compute_hebbian_conv,
    compute_hebbian_swiglu,
    compute_hebbian_temporal,
    compute_hebbian_topdown,
    compute_modulators,
)
from model.pc.pc_layers import CyrenePC

# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def make_model(device):
    cfg = CyreneConfig(hidden_size=256, num_hidden_layers=4)
    model = CyrenePC(cfg).to(device)
    return model

def make_optimizer(model, lr):
    return None

def warmup(model, batch_size, seq_len, device):
    with torch.no_grad():
        dummy_byte = torch.full((batch_size, seq_len), 128.0, device=device)
        dummy_role = torch.full((batch_size, seq_len), 2.0, device=device)
        dummy = torch.stack([dummy_byte, dummy_role], dim=1)  # [bsz, 2, seq]
        dummy_lbl = torch.full((batch_size, seq_len), -100, device=device).long()
        dummy_pos = model.get_position_embeddings(seq_len, device)
        model.forward_with_ce(dummy, dummy_lbl, dummy_pos)

@torch.no_grad()
def evaluate(model, data_path, max_seq_len, batch_size, max_samples, device):
    ds = DualChannelDataset(data_path, max_length=max_seq_len, max_samples=max_samples)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    total_ce, total_tokens = 0.0, 0
    model.eval()
    for bt, lt in loader:
        bt, lt = bt.to(device), lt.to(device)
        seq_len = bt.size(2)  # [bsz, 2, seq]
        pos = model.get_position_embeddings(seq_len, device)
        _, ce = model.forward_with_ce(bt, lt, pos)
        n_valid = (lt != -100).sum().item()
        total_ce += ce.item() * n_valid
        total_tokens += n_valid
    avg_ce = total_ce / max(total_tokens, 1)
    model.train()
    return avg_ce


@torch.no_grad()
def hebbian_step(model, bt, lt, device, base_lr, gamma_rpe, total_steps, cfg):
    """单步 Hebbian 训练, 返回 CE + modulation."""
    bt, lt = bt.to(device), lt.to(device)
    pos = model.get_position_embeddings(bt.size(2), device)  # [bsz, 2, seq]
    z_init, ce = model.forward_with_ce(bt, lt, pos)

    z_detached = [z.detach() for z in z_init]
    _, _, _, F_pred, ε_list = model.spatiotemporal_infer(
        z_detached, pos,
        gamma=cfg.get('gamma', 0.05),
        T=cfg.get('T_infer', 1),
        return_errors=False,
        return_pred_loss=True,
        return_ε=True,
    )

    F_curr = F_pred.item()
    F_prev = getattr(model, '_last_F', F_curr)
    model._last_F = F_curr
    uncertainty = 1.0
    _, _, modulation = compute_modulators(F_curr, F_prev, uncertainty, cfg)

    updates = {}
    updates.update(compute_hebbian_temporal(ε_list, z_detached, modulation, base_lr, gamma_rpe))
    updates.update(compute_hebbian_topdown(ε_list, z_detached, modulation, base_lr, gamma_rpe))

    L = len(ε_list)
    for ℓ in range(L):
        block_idx = ℓ // 2
        is_conv = (ℓ % 2 == 0)
        block = model.model.layers[block_idx]
        if is_conv:
            dW = compute_hebbian_conv(ε_list[ℓ], z_detached[ℓ],
                                       block.local_conv.weight, block.dilation,
                                       modulation, base_lr, gamma_rpe)
            updates[f'model.layers.{block_idx}.local_conv.weight'] = dW
        else:
            mlp_up = compute_hebbian_swiglu(ε_list[ℓ], z_detached[ℓ], block.mlp,
                                             modulation, base_lr, gamma_rpe)
            for k, v in mlp_up.items():
                updates[f'model.layers.{block_idx}.mlp.{k}'] = v

    apply_hebbian_updates(updates, model)
    return ce.item(), modulation


# ═══════════════════════════════════════════════════════════════════
# Phase 1: 无回放基线 — A→B→C→D 顺序学习, 每步评估所有已知任务
# ═══════════════════════════════════════════════════════════════════

def run_phase1(model, task_paths, epochs, max_seq_len, batch_size,
               max_samples, device, base_lr, cfg):
    """训练 A→B→...→N, 纯 Hebbian.
    返回: matrix[n_tasks][n_tasks], matrix[i][j] = 训练完 i 后在任务 j 上的 CE.
    """
    n = len(task_paths)
    matrix = [[0.0] * n for _ in range(n)]
    gamma_rpe = cfg.get('dopamine_gamma', 0.3)

    for i, path in enumerate(task_paths):
        ds = DualChannelDataset(path, max_length=max_seq_len, max_samples=None)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        total_steps = len(loader) * epochs
        gs = 0
        best = float('inf')
        name = chr(65 + i)  # A, B, C, D...

        print(f'\n--- [P1-{name}] {os.path.basename(path)} ---')
        for _ in range(epochs):
            for bt, lt in loader:
                gs += 1
                loss, mod = hebbian_step(model, bt, lt, device, base_lr, gamma_rpe, total_steps, cfg)
                if loss < best:
                    best = loss
                if gs % 200 == 0:
                    print(f'  [{name}] Step {gs}/{total_steps} CE={loss:.4f} mod={mod:.3f}')
        print(f'  [{name}] Done. Best CE={best:.4f}')

        # 评估所有已知任务
        for j in range(i + 1):
            ce = evaluate(model, task_paths[j], max_seq_len, batch_size, max_samples, device)
            matrix[i][j] = ce
            j_name = chr(65 + j)
            print(f'  [P1-{name}] CE on {j_name}={ce:.4f}')

    return matrix


# ═══════════════════════════════════════════════════════════════════
# Phase 2: MemoryBank + Sniffer — A→B→C→D 带保护
# ═══════════════════════════════════════════════════════════════════

def run_phase2(model, task_paths, epochs, max_seq_len, batch_size, max_samples, device,
               threshold, repair_steps, check_interval, bank_size, n_exemplars, replay_ratio, base_lr, cfg):
    """训练 A→B→...→N 带 MemoryBank+Sniffer 保护, 纯 Hebbian.
    每学完一个任务, 收集 exemplars → bank.
    返回: matrix[n_tasks][n_tasks] (同上).
    """
    n = len(task_paths)
    matrix = [[0.0] * n for _ in range(n)]
    memory_bank = MemoryBank(max_per_task=bank_size)
    sniffer = ForgettingSniffer(
        memory_bank=memory_bank, model=model,
        check_interval=check_interval, threshold=threshold, repair_steps=repair_steps,
    )
    gamma_rpe = cfg.get('dopamine_gamma', 0.3)

    # ── 辅助: Hebbian 回放步 ──
    def _hebbian_replay(rb, rl):
        seq_len = rb.size(2)
        rp = model.get_position_embeddings(seq_len, device)
        z_r, _ = model.forward_with_ce(rb, rl, rp)
        z_det = [z.detach() for z in z_r]
        _, _, _, F_pred, ε_list = model.spatiotemporal_infer(
            z_det, rp,
            gamma=cfg.get('gamma', 0.05),
            T=cfg.get('T_infer', 1),
            return_errors=False,
            return_pred_loss=True,
            return_ε=True,
        )
        F_curr = F_pred.item()
        F_prev = getattr(model, '_last_F', F_curr)
        model._last_F = F_curr
        _, _, modulation = compute_modulators(F_curr, F_prev, 1.0, cfg)
        updates = {}
        updates.update(compute_hebbian_temporal(ε_list, z_det, modulation, base_lr, gamma_rpe))
        updates.update(compute_hebbian_topdown(ε_list, z_det, modulation, base_lr, gamma_rpe))
        L = len(ε_list)
        for ℓ in range(L):
            bidx = ℓ // 2
            is_conv = (ℓ % 2 == 0)
            blk = model.model.layers[bidx]
            if is_conv:
                dW = compute_hebbian_conv(ε_list[ℓ], z_det[ℓ], blk.local_conv.weight,
                                           blk.dilation, modulation, base_lr, gamma_rpe)
                updates[f'model.layers.{bidx}.local_conv.weight'] = dW
            else:
                mu = compute_hebbian_swiglu(ε_list[ℓ], z_det[ℓ], blk.mlp,
                                             modulation, base_lr, gamma_rpe)
                for k, v in mu.items():
                    updates[f'model.layers.{bidx}.mlp.{k}'] = v
        apply_hebbian_updates(updates, model)

    # ── 主循环 ──
    for i, path in enumerate(task_paths):
        name = chr(65 + i)

        if i == 0:
            # 第一个任务: 纯 Hebbian 训练 (无保护)
            print(f'\n--- [P2-{name}] {os.path.basename(path)} (首次, 无保护) ---')
            ds = DualChannelDataset(path, max_length=max_seq_len, max_samples=None)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
            total_steps = len(loader) * epochs
            gs = 0
            best = float('inf')
            for _ in range(epochs):
                for bt, lt in loader:
                    gs += 1
                    loss, mod = hebbian_step(model, bt, lt, device, base_lr, gamma_rpe, total_steps, cfg)
                    if loss < best:
                        best = loss
                    if gs % 200 == 0:
                        print(f'  [{name}] Step {gs}/{total_steps} CE={loss:.4f} mod={mod:.3f}')
            print(f'  [{name}] Done. Best CE={best:.4f}')
        else:
            # 后续任务: 带 MemoryBank 回放 + Sniffer 保护
            print(f'\n--- [P2-{name}] {os.path.basename(path)} [回放保护中] ---')
            ds = DualChannelDataset(path, max_length=max_seq_len, max_samples=None)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
            total_steps = len(loader) * epochs
            gs = 0
            best = float('inf')

            for _ in range(epochs):
                for bt, lt in loader:
                    gs += 1

                    # 主训练步
                    loss, mod = hebbian_step(model, bt, lt, device, base_lr, gamma_rpe, total_steps, cfg)
                    if loss < best:
                        best = loss

                    # 记忆回放 (非修复期间)
                    if memory_bank.total > 0 and gs % replay_ratio == 0 and not sniffer.is_repairing:
                        replay_ex = memory_bank.sample(batch_size, strategy='dopamine')
                        if replay_ex:
                            rb = torch.stack([ex.byte_tensor for ex in replay_ex], dim=0).to(device)
                            rl = torch.stack([ex.label_tensor for ex in replay_ex], dim=0).to(device)
                            _hebbian_replay(rb, rl)

                    # 遗忘嗅探
                    if sniffer.is_repairing or (gs % check_interval == 0 and gs > 0):
                        forgotten = sniffer.check(gs, device)
                        if forgotten:
                            print(f'    [Sniffer] FORGOTTEN: {forgotten} — repairing')
                            for _ in range(repair_steps):
                                replay_data = sniffer.get_replay_batch(batch_size, device)
                                if replay_data is None:
                                    break
                                rb, rl = replay_data
                                _hebbian_replay(rb, rl)
                            print('    [Sniffer] Repair complete')

                    if gs % 200 == 0:
                        print(f'  [{name}] Step {gs}/{total_steps} CE={loss:.4f} bank={memory_bank.total}')

            print(f'  [{name}] Done. Best CE={best:.4f}')

        # 收集 exemplars → MemoryBank (每任务都做, 即使第一个任务)
        print(f'  [P2-{name}] 收集 exemplars → MemoryBank...')
        ds_eval = DualChannelDataset(path, max_length=max_seq_len, max_samples=None)
        n_ex = min(n_exemplars, len(ds_eval))
        idx = torch.randperm(len(ds_eval))[:n_ex].tolist()
        samples_list, total_bl = [], 0.0
        with torch.no_grad():
            for idx_i in idx:
                bt, lt = ds_eval[idx_i]
                samples_list.append((bt, lt))
                x, y = bt.unsqueeze(0).to(device), lt.unsqueeze(0).to(device)
                p = model.get_position_embeddings(x.size(2), device)  # [1, 2, seq]
                _, bl = model.forward_with_ce(x, y, p)
                total_bl += bl.item()
        memory_bank.add_samples(name, samples_list, dopamine_score=0.5,
                                baseline_loss=total_bl / max(len(idx), 1))
        print(f'  {n_ex} exemplars → bank (total={memory_bank.total})')

        # 评估所有已知任务
        for j in range(i + 1):
            ce = evaluate(model, task_paths[j], max_seq_len, batch_size, max_samples, device)
            matrix[i][j] = ce
            j_name = chr(65 + j)
            print(f'  [P2-{name}] CE on {j_name}={ce:.4f}')

    return matrix


# ═══════════════════════════════════════════════════════════════════
# 输出 4×4 CE 矩阵 + 结论
# ═══════════════════════════════════════════════════════════════════

def print_matrix(title, matrix, task_names, phase_label):
    """打印 CE 矩阵: 行=训练完第 i 个任务, 列=在第 j 个任务上的 CE."""
    n = len(task_names)
    print(f'\n  {title}')
    print(f'  {"":<12}', end='')
    for name in task_names:
        print(f'{name:>12}', end='')
    print()
    print(f'  {"─"*12}', end='')
    for _ in task_names:
        print(f'{"─"*12}', end='')
    print()

    for i in range(n):
        label = f'{phase_label}{chr(65+i)}之后'
        print(f'  {label:<12}', end='')
        for j in range(n):
            if j <= i:
                print(f'{matrix[i][j]:>12.4f}', end='')
            else:
                print(f'{"":>12}', end='')
        print()
    print()


def print_conclusion(p1_matrix, p2_matrix, task_names):
    """从矩阵中提取遗忘量, 输出结论."""
    n = len(task_names)
    print('=' * 72)
    print('  结论: 灾难性遗忘分析')
    print('=' * 72)

    # 对每个任务: 看它被后续任务覆盖后的 CE 变化
    all_clear = True
    for j in range(n - 1):  # 最后一个任务没有后继
        ce_after_train = p1_matrix[j][j]  # 刚训练完时的 CE
        ce_after_all = p1_matrix[n - 1][j]  # 所有任务训练完后的 CE
        delta_p1 = ce_after_all - ce_after_train

        ce_after_train_p2 = p2_matrix[j][j]
        ce_after_all_p2 = p2_matrix[n - 1][j]
        delta_p2 = ce_after_all_p2 - ce_after_train_p2

        verdict_p1 = '💀 灾难性遗忘!' if delta_p1 > 0.5 else ('⚠️ 部分遗忘' if delta_p1 > 0.2 else '✅ 稳定')
        verdict_p2 = '🛡️ 完全保护' if abs(delta_p2) < 0.3 else ('⚠️ 部分遗忘' if delta_p2 > 0.2 else '✅ 稳定')

        print(f'\n  任务 {task_names[j]}:')
        print(f'    无回放: CE {ce_after_train:.4f} → {ce_after_all:.4f} (Δ={delta_p1:+.4f}) {verdict_p1}')
        print(f'    有回放: CE {ce_after_train_p2:.4f} → {ce_after_all_p2:.4f} (Δ={delta_p2:+.4f}) {verdict_p2}')

        if delta_p2 > 0.3:
            all_clear = False

    # 新任务学习能力对比 (最后一个任务)
    last = n - 1
    ce_b_p1 = p1_matrix[last][last]
    ce_b_p2 = p2_matrix[last][last]
    print(f'\n  新任务 ({task_names[last]}) 学习能力:')
    print(f'    无回放 CE={ce_b_p1:.4f} vs 有回放 CE={ce_b_p2:.4f}')
    if abs(ce_b_p1 - ce_b_p2) < 0.5:
        print('    ✅ 保护机制未影响新任务学习能力')
    else:
        print('    ⚠️ 保护机制对新任务学习有轻微影响')

    if all_clear:
        print('\n  🎉 总体结论: MemoryBank+Sniffer 成功防止了多任务顺序学习中的灾难性遗忘!')
    else:
        print('\n  ⚠️ 总体结论: 保护部分有效, 但部分任务仍有遗忘, 建议微调 --threshold/--repair-steps')


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='多任务灾难性遗忘压力测试 — 无回放 vs MemoryBank+Sniffer 4×4 CE 矩阵',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 数据 — 支持 N 个任务
    parser.add_argument('--tasks', type=str, nargs='+', required=True,
                        help='任务数据路径 (按学习顺序, 至少 2 个)')
    parser.add_argument('--task-names', type=str, nargs='+', default=None,
                        help='任务名称 (对应 --tasks, 默认: A B C D...')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='每任务最大样本数 (默认: 全量)')

    # 训练
    parser.add_argument('--epochs', type=int, default=1,
                        help='每任务训练轮数 (默认: 1)')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max-seq-len', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)

    # 持续学习
    parser.add_argument('--bank-size', type=int, default=2000,
                        help='MemoryBank 每任务最大容量')
    parser.add_argument('--exemplars', type=int, default=500,
                        help='每任务存入 bank 的 exemplars 数量')
    parser.add_argument('--replay-ratio', type=int, default=5,
                        help='每 N 步回放一次')

    # Sniffer 调优
    parser.add_argument('--threshold', type=float, default=1.5,
                        help='Sniffer 触发阈值 (默认: 1.5)')
    parser.add_argument('--repair-steps', type=int, default=5,
                        help='Sniffer 每轮修复步数 (默认: 5)')
    parser.add_argument('--check-interval', type=int, default=500,
                        help='Sniffer 检测间隔 (默认: 500)')

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = DEVICE_STR

    n_tasks = len(args.tasks)
    if n_tasks < 2:
        print('错误: 至少需要 2 个任务')
        sys.exit(1)

    task_names = args.task_names if args.task_names else [chr(65 + i) for i in range(n_tasks)]
    if len(task_names) != n_tasks:
        print(f'错误: --task-names 数量 ({len(task_names)}) 与 --tasks ({n_tasks}) 不匹配')
        sys.exit(1)

    # 解析路径
    root = os.path.dirname(os.path.abspath(__file__))
    task_paths = [p if os.path.isabs(p) else os.path.join(root, p) for p in args.tasks]

    print('=' * 72)
    print(f'  多任务灾难性遗忘压力测试 ({n_tasks} 任务)')
    print('=' * 72)
    for i, (name, path) in enumerate(zip(task_names, task_paths)):
        print(f'  任务 {chr(65+i)} ({name}): {os.path.basename(path)}')
    print(f'  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}, SeqLen: {args.max_seq_len}')
    print(f'  Sniffer: threshold={args.threshold}, repair_steps={args.repair_steps}, check_interval={args.check_interval}')
    print(f'  Device: {device}')
    print()

    cfg = {'dopamine_gamma': 0.3, 'gamma': 0.05, 'T_infer': 1}

    # ═══════════════════════════════════════════════════════════
    # Phase 1 — 无回放
    # ═══════════════════════════════════════════════════════════
    print('=' * 72)
    print('  PHASE 1: 无回放基线 (灾难性遗忘)')
    print('=' * 72)
    setup_seed(args.seed)
    m1 = make_model(device)
    n_params = sum(p.numel() for p in m1.parameters() if p.requires_grad)
    print(f'  Model: {n_params/1e6:.2f}M parameters')
    warmup(m1, args.batch_size, args.max_seq_len, device)

    t0 = time.time()
    p1_matrix = run_phase1(m1, task_paths, args.epochs, args.max_seq_len,
                           args.batch_size, args.max_samples, device, args.lr, cfg)
    p1_time = time.time() - t0
    print_matrix('PHASE 1 CE 矩阵 (无回放)', p1_matrix, task_names, 'P1-')
    print(f'  Phase 1 耗时: {p1_time:.1f}s ({p1_time/60:.1f}min)')

    # ═══════════════════════════════════════════════════════════
    # Phase 2 — 带保护
    # ═══════════════════════════════════════════════════════════
    print('\n' + '=' * 72)
    print('  PHASE 2: MemoryBank + Sniffer 保护')
    print('=' * 72)
    setup_seed(args.seed)
    m2 = make_model(device)
    warmup(m2, args.batch_size, args.max_seq_len, device)

    t0 = time.time()
    p2_matrix = run_phase2(m2, task_paths, args.epochs, args.max_seq_len,
                           args.batch_size, args.max_samples, device,
                           args.threshold, args.repair_steps, args.check_interval,
                           args.bank_size, args.exemplars, args.replay_ratio, args.lr, cfg)
    p2_time = time.time() - t0
    print_matrix('PHASE 2 CE 矩阵 (有回放)', p2_matrix, task_names, 'P2-')
    print(f'  Phase 2 耗时: {p2_time:.1f}s ({p2_time/60:.1f}min)')

    # ═══════════════════════════════════════════════════════════
    # 结论
    # ═══════════════════════════════════════════════════════════
    print_conclusion(p1_matrix, p2_matrix, task_names)
    print()


if __name__ == '__main__':
    main()
