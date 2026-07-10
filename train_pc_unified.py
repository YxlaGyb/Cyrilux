"""
统一训练: 预测编码局部时空 + 全局多巴胺 + 4bit QAT，三合一。

架构:
  Phase 1: forward_with_ce()       ← 共享前向 (有梯度)
  Phase 2: spatiotemporal_infer()  ← T 步推理 (π 可调)
  Phase 3: compute_*               ← F_pred (π 加权) + CE_conv
  Phase 4: Dopamine.update(F) → D  ← 3 级调制 (precision / beta / lr)
  Phase 5: backward + step         ← lr 调制

Ponytail: 从 train_pc_local_hybrid.py 派生，正交叠加多巴胺 + QAT。
"""
import os, sys, json, warnings, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model.pc_layers import PCLocalDynamicMiniMind
from model.model_minimind import MiniMindConfig
from model.pc_core import DopamineSignal
from trainer_utils import get_lr, Logger, setup_seed
from tqdm import tqdm

# 持续学习模块
from continual.memory_bank import MemoryBank
from continual.forgetting_sniffer import ForgettingSniffer
from continual.offline_replay import OfflineReplayer

# ═══════════════════════════════════════════════════════════════════
# 数据集
# ═══════════════════════════════════════════════════════════════════

class _LocalDataset(Dataset):
    """原始 UTF-8 字节数据集 — 预编码到内存, 消除 per-step CPU 编码开销."""
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


warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════
# 跨任务遗忘评估
# ═══════════════════════════════════════════════════════════════════

def evaluate_cross_tasks(model, task_ids, root, device, max_seq_len=128, n_samples=100):
    """评估模型在所有给定任务上的 CE / PPL。返回 {task_id: {ce, ppl}}."""
    results = {}
    model.eval()
    with torch.no_grad():
        for tid in task_ids:
            task_path = os.path.join(root, 'dataset', f'task_{tid}.jsonl')
            ds = _LocalDataset(task_path, max_length=max_seq_len, max_samples=n_samples)
            total_ce = 0.0
            for i in range(len(ds)):
                bt, lt = ds[i]
                x = bt.unsqueeze(0).to(device)
                y = lt.unsqueeze(0).to(device)
                p = model.get_position_embeddings(x.size(1), device)
                _, ce = model.forward_with_ce(x, y, p)
                total_ce += ce.item()
            avg_ce = total_ce / max(len(ds), 1)
            results[tid] = {'ce': avg_ce, 'ppl': math.exp(min(avg_ce, 20))}
    model.train()
    return results


# ═══════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='PC 统一训练: 时空 + 多巴胺 + QAT')

    # 训练基础参数
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--subset', type=int, default=10000,
                        help='训练子集大小')
    parser.add_argument('--seed', type=int, default=42)

    # PC 参数
    parser.add_argument('--T_infer', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--max_beta', type=float, default=2.0,
                        help='CE_local 权重上限')
    parser.add_argument('--max_beta_conv', type=float, default=1.0,
                        help='CE_converged 权重上限')
    parser.add_argument('--ema_lambda', type=float, default=0.001,
                        help='EMA 正则强度 (防坍塌)')
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # 多巴胺
    parser.add_argument('--dopamine', action='store_true',
                        help='启用全局多巴胺 3 级调制')
    parser.add_argument('--dopamine_eta', type=float, default=1.0,
                        help='精度调制强度 η')
    parser.add_argument('--dopamine_beta', type=float, default=0.5,
                        help='学习率调制强度 β')
    parser.add_argument('--dopamine_gamma', type=float, default=0.3,
                        help='loss 平衡调制强度 γ')

    # QAT
    parser.add_argument('--quantize', action='store_true',
                        help='启用 4bit weight-only QAT')
    parser.add_argument('--qat_groupsize', type=int, default=64)
    parser.add_argument('--no_quantize_embed', action='store_true',
                        help='不量化 embed/lm_head (实验性)')
    parser.add_argument('--compile', action='store_true',
                        help='使用 torch.compile 编译前向 (需 PyTorch ≥2.0, ~1.5-2×)')
    parser.add_argument('--fast', action='store_true',
                        help='纯 CE 模式 (跳过 PC, ~3× 速度)')

    # I/O
    parser.add_argument('--out_dir', type=str, default='out_pc_unified',
                        help='输出目录')
    parser.add_argument('--data_path', type=str, default=None,
                        help='数据集路径 (默认 pretrain_t2t_mini.jsonl)')
    parser.add_argument('--save_interval', type=int, default=500)

    # 持续学习 (多巴胺门控)
    parser.add_argument('--continual', action='store_true',
                        help='启用持续学习: 多任务序列训练 + 记忆回放 + 遗忘嗅探')
    parser.add_argument('--task_order', type=str, default=None,
                        help='任务顺序, 逗号分隔 (默认 "a,b,c,d,e")')
    parser.add_argument('--replay_ratio', type=int, default=5,
                        help='每 N 步插入 1 步回放 (默认 5)')
    parser.add_argument('--bank_size', type=int, default=2000,
                        help='每任务最大 exemplar 数 (默认 2000)')
    parser.add_argument('--sniff_interval', type=int, default=200,
                        help='遗忘嗅探检查间隔 (默认 200 步)')
    parser.add_argument('--repair_threshold', type=float, default=1.2,
                        help='遗忘触发阈值: loss_ratio > threshold (默认 1.2)')
    parser.add_argument('--repair_steps', type=int, default=10,
                        help='修复步数 (默认 10)')
    parser.add_argument('--eval_samples', type=int, default=100,
                        help='跨任务遗忘评估每任务样本数 (默认 100)')

    return parser.parse_args()


def train():
    args = parse_args()

    # ── 模型配置 ──
    lm_config = MiniMindConfig(hidden_size=256, num_hidden_layers=4, use_moe=False)

    ROOT = os.path.dirname(os.path.abspath(__file__))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data_path = args.data_path or os.path.join(ROOT, 'dataset', 'pretrain_t2t_mini.jsonl')
    out_dir = os.path.join(ROOT, args.out_dir)
    # ── 构建描述 ──
    desc_parts = ['Hybrid']
    if args.dopamine:
        desc_parts.append(f'DA(η={args.dopamine_eta},β={args.dopamine_beta},γ={args.dopamine_gamma})')
    if args.quantize:
        desc_parts.append(f'QAT(g={args.qat_groupsize})')
    desc = '+'.join(desc_parts)

    Logger(f'=== {desc} Training ===')
    Logger(f'T={args.T_infer}, γ={args.gamma}, lr={args.lr}, batch={args.batch_size}')
    Logger(f'Device: {device}')

    # ── 模型创建 (CPU) — QAT prepare 必须在 CPU ──
    pc_model = PCLocalDynamicMiniMind(lm_config)
    base_params = sum(p.numel() for p in pc_model.parameters() if p.requires_grad)
    Logger(f'Base params: {base_params / 1e6:.2f}M')

    # ── QAT 准备 (CPU 上执行) ──
    quantizer = None
    if args.quantize:
        from torchao.quantization.qat import Int4WeightOnlyQATQuantizer
        quantizer = Int4WeightOnlyQATQuantizer(
            groupsize=args.qat_groupsize,
            inner_k_tiles=4,
            precision=torch.float16,
            scales_precision=torch.bfloat16,
        )
        pc_model = quantizer.prepare(pc_model)
        Logger(f'Int4WeightOnly QAT prepared (groupsize={args.qat_groupsize})')

    # ── 移到设备 ──
    pc_model = pc_model.to(device)
    if device == 'cuda:0':
        torch.set_float32_matmul_precision('medium')  # TF32 matmul (~2× vs fp32)
        torch.backends.cudnn.benchmark = True       # Conv1D 自动选最优算法
        torch.backends.cudnn.allow_tf32 = True      # tf32 matmul
    if args.compile and hasattr(torch, 'compile'):
        try:
            pc_model.forward_with_ce = torch.compile(pc_model.forward_with_ce,
                                                      mode='reduce-overhead')
            Logger('torch.compile 启用 (mode=reduce-overhead)')
        except Exception as e:
            Logger(f'torch.compile 失败 (已忽略): {e}')
    qat_params = sum(p.numel() for p in pc_model.parameters() if p.requires_grad)
    Logger(f'QAT params: {qat_params / 1e6:.2f}M (trainable)')

    # ── 数据 ──
    ds = _LocalDataset(data_path,
                       max_length=args.max_seq_len, max_samples=args.subset)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)
    iters = len(loader)
    Logger(f'Data: {args.subset} samples, {iters} steps/epoch')

    # ── 预热: 吸收 cudnn benchmark 首步算法搜索延迟 ──
    with torch.no_grad():
        dummy = torch.randint(0, 256, (args.batch_size, args.max_seq_len), device=device).long()
        dummy_pos = pc_model.get_position_embeddings(args.max_seq_len, device)
        _, _ = pc_model.forward_with_ce(dummy, dummy, dummy_pos)
    Logger('Warmup done (cudnn benchmark ready)')

    # ── 优化器 ──
    optimizer = torch.optim.AdamW(
        list(pc_model.temporal_proj.parameters()) +
        list(pc_model.topdown_proj.parameters()) +
        [p for n, p in pc_model.model.named_parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), fused=True,
    )

    # ── no AMP (模型 fp32 已在 /255 归一化后稳定) ──
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    os.makedirs(out_dir, exist_ok=True)
    forgetting_log = []  # 遗忘评估日志

    # ── 多巴胺初始化 ──
    dopamine = None
    if args.dopamine:
        dopamine = DopamineSignal(η=args.dopamine_eta, threshold=0.0)
        # 冷启动: 前几步 D=0.5, 避免 F_prev=inf → D≈0
        dopamine.F_prev = None
        _dopamine_warmup_steps = 10

    # ── 精度权重缓存 (per-layer, 上一步的值) ──
    prev_precision_scales = None

    # ── 持续学习初始化 ──
    memory_bank = None
    sniffer = None
    offline_replayer = None
    task_order = None
    is_replay_step = False
    if args.continual:
        memory_bank = MemoryBank(max_per_task=args.bank_size)
        sniffer = ForgettingSniffer(
            memory_bank=memory_bank, model=pc_model,
            check_interval=args.sniff_interval,
            threshold=args.repair_threshold,
            repair_steps=args.repair_steps,
        )
        offline_replayer = OfflineReplayer(memory_bank, pc_model)
        task_list = args.task_order or 'a,b,c,d,e'
        task_order = [t.strip() for t in task_list.split(',')]
        Logger(f'Continual learning: {len(task_order)} tasks → {", ".join(task_order)}')
        Logger(f'  replay_ratio={args.replay_ratio}, bank_size={args.bank_size}')
        Logger(f'  sniff_interval={args.sniff_interval}, repair_threshold={args.repair_threshold}')

    # ── 训练循环 ──
    pc_model.train()
    global_step = 0
    total_steps = iters * args.epochs
    # 非 continual 时 task_order 保持 None

    # EMA 参考 (防坍塌)
    ema_z = None

    # ── 任务序列 (持续学习) ──
    current_task_id = None
    _task_loader = loader  # fallback: 非 continual 用原始 loader

    # 外层: 任务循环 (持续学习) / 单任务 (原始)
    _task_iter = task_order if args.continual else [None]

    for _task_id in _task_iter:
        current_task_id = _task_id
        if args.continual and _task_id is not None:
            task_path = os.path.join(ROOT, 'dataset', f'task_{_task_id}.jsonl')
            Logger(f'\n{"="*60}\nStarting Task {_task_id}: {task_path}\n{"="*60}')
            task_ds = _LocalDataset(task_path,
                                    max_length=args.max_seq_len, max_samples=args.subset)
            _task_loader = DataLoader(task_ds, batch_size=args.batch_size, shuffle=True,
                                      num_workers=0, pin_memory=True)
            # 重启 total_steps (per-task)
            total_steps = len(_task_loader) * args.epochs
            global_step = 0

        for epoch in range(args.epochs):
            pbar = tqdm(_task_loader,
                        desc=f'Task {_task_id or "single"} Epoch {epoch + 1}/{args.epochs} [{desc}]',
                        unit='step', dynamic_ncols=True, ascii=True)

            for step, (byte_seq, labels) in enumerate(pbar):
                byte_seq = byte_seq.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                bsz, seq_len = byte_seq.shape
                global_step += 1

                # ═══════════════════════════════════════════════════════
                # Phase 1: 共享前向 (有梯度)
                # ═══════════════════════════════════════════════════════
                pos_emb = pc_model.get_position_embeddings(seq_len, device)
                z_init, ce_loss = pc_model.forward_with_ce(byte_seq, labels, pos_emb)

            if args.fast:
                # ═════════════════════════════════════════════════
                # Fast mode: 纯 CE, 跳过全部 PC
                # ═════════════════════════════════════════════════
                total_loss = ce_loss
                F_pred = ce_loss.detach()
                F_final = 0.0
                errors_hist = []
                F_hist = [0.0]
                ce_converged = ce_loss
                β_local = 0.0
                β_conv = 0.0
                D = 0.5
                scale_local = 0.0
                scale_conv = 0.0
            else:
                # ═════════════════════════════════════════════════
                # Phase 2+: PC 推理 + 多路损失
                # ═════════════════════════════════════════════════
                z_detached = [z.detach() for z in z_init]
                # ponytail: return_errors 只需用于多巴胺精度调制
                z_converged, errors_hist, F_hist, F_pred = pc_model.spatiotemporal_infer(
                        z_detached, pos_emb, gamma=args.gamma, T=args.T_infer,
                        return_errors=args.dopamine,
                        return_pred_loss=True,
                        precision_scales=prev_precision_scales,
                    )


                # Phase 3: CE_conv — 用 z_init 的 CE 代替, 跳过二次前向
                ce_converged = ce_loss

                # Phase 4: 多巴胺调制
                D = 0.5
                β_local = min(args.max_beta,
                              0.1 + global_step / total_steps * (args.max_beta - 0.1))
                β_conv = min(args.max_beta_conv,
                             0.0 + global_step / total_steps * args.max_beta_conv)

                if args.dopamine:
                    if dopamine.F_prev is None:
                        dopamine.F_prev = float('inf')
                        D = 0.5 if global_step <= _dopamine_warmup_steps else 0.0
                    else:
                        D = dopamine.update(F_pred.item())
                β_local = β_local * (1.0 + args.dopamine_gamma * D)
                β_conv = β_conv * (1.0 + args.dopamine_gamma * D)
                last_errors = errors_hist[-1] if errors_hist else []
                if last_errors:
                    err_norms = torch.tensor([e[1] for e in last_errors], device=device)
                    max_err = err_norms.max() + 1e-8
                    π_list = 1.0 + args.dopamine_eta * D * (err_norms / max_err)
                    prev_precision_scales = π_list.detach().cpu().tolist()
                else:
                    prev_precision_scales = None

                # Phase 4.5: 三路合并 (PC only)
                ce_local_sum = ce_loss * (bsz * seq_len)
                ce_conv_sum = ce_converged * (bsz * seq_len)
                scale_local = (F_pred.detach() / (ce_local_sum.detach() + 1e-8)).clamp(0.1, 10.0)
                scale_conv = (F_pred.detach() / (ce_conv_sum.detach() + 1e-8)).clamp(0.1, 10.0)
                total_loss = F_pred + β_local * scale_local * ce_local_sum \
                                      + β_conv * scale_conv * ce_conv_sum


            # ═══════════════════════════════════════════════════════
            # Phase 5: backward
            # ═══════════════════════════════════════════════════════
            optimizer.zero_grad(set_to_none=True)

            total_loss.backward()

            # 梯度裁剪
            trainable_params = [p for p in pc_model.parameters()
                                if p.requires_grad and p.grad is not None]
            if trainable_params:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)

            # 学习率 (余弦 + 多巴胺调制)
            current_lr = get_lr(global_step, total_steps, args.lr)
            if args.dopamine:
                current_lr = current_lr * (1.0 + args.dopamine_beta * D)

            for pg in optimizer.param_groups:
                pg['lr'] = current_lr

            optimizer.step()

            # ponytail: EMA 跳过 (不影响收敛, 省 per-step clone/add)

            # ═══════════════════════════════════════════════════════
            # 持续学习: 记忆回放 (每 replay_ratio 步插入 1 步纯 CE 回放)
            # ═══════════════════════════════════════════════════════
            if args.continual and memory_bank is not None and memory_bank.total > 0:
                if global_step % args.replay_ratio == 0 and not sniffer.is_repairing:
                    replay_ex = memory_bank.sample(args.batch_size, strategy='dopamine')
                    if replay_ex:
                        replay_byte = torch.stack([ex.byte_tensor for ex in replay_ex], dim=0).to(device)
                        replay_label = torch.stack([ex.label_tensor for ex in replay_ex], dim=0).to(device)
                        replay_pos = pc_model.get_position_embeddings(replay_byte.size(1), device)
                        _, replay_loss = pc_model.forward_with_ce(replay_byte, replay_label, replay_pos)
                        optimizer.zero_grad(set_to_none=True)
                        replay_loss.backward()
                        trainable_rp = [p for p in pc_model.parameters()
                                        if p.requires_grad and p.grad is not None]
                        if trainable_rp:
                            torch.nn.utils.clip_grad_norm_(trainable_rp, args.grad_clip)
                        optimizer.step()

            # ═══════════════════════════════════════════════════════
            # 持续学习: 遗忘嗅探 + 自触发修复
            # ═══════════════════════════════════════════════════════
            if args.continual and sniffer is not None:
                forgotten = sniffer.check(global_step, device)
                if forgotten:
                    current_lr_sniff = current_lr  # 当前已调制的 lr
                    repair_lr = sniffer.repair_begin(optimizer, current_lr_sniff, device)
                    Logger(f'[Sniffer] FORGOTTEN: {forgotten} — repair LR={repair_lr:.2e}')
                    # 修复回放
                    for _ in range(args.repair_steps):
                        replay_data = sniffer.get_replay_batch(args.batch_size, device)
                        if replay_data is None:
                            break
                        rp_byte, rp_label = replay_data
                        rp_pos = pc_model.get_position_embeddings(rp_byte.size(1), device)
                        _, rp_loss = pc_model.forward_with_ce(rp_byte, rp_label, rp_pos)
                        optimizer.zero_grad(set_to_none=True)
                        rp_loss.backward()
                        trainable_rp = [p for p in pc_model.parameters()
                                        if p.requires_grad and p.grad is not None]
                        if trainable_rp:
                            torch.nn.utils.clip_grad_norm_(trainable_rp, args.grad_clip)
                        optimizer.step()
                    sniffer.repair_end(optimizer, current_lr_sniff)
                    Logger(f'[Sniffer] Repair complete — LR restored to {current_lr_sniff:.2e}')

            # ═══════════════════════════════════════════════════════
            # 日志 (ponytail: 跳过 compute_representation_metrics, 太快时不值得)
            # ═══════════════════════════════════════════════════════
            ce_val = ce_loss.item()
            ce_conv_val = ce_converged.item() if hasattr(ce_converged, 'item') else ce_val
            F_val = F_pred.item()
            F_final = F_hist[-1] if F_hist else 0.0

            # ── 进度条日志 ──
            postfix = {
                'CE': f'{ce_val:.4f}',
                'F': f'{F_final:.1f}',
                'CEc': f'{ce_conv_val:.4f}',
            }
            if args.dopamine:
                postfix['D'] = f'{D:.3f}'
            pbar.set_postfix(**postfix)

            # ── 详细日志 (每 100 步) ──
            if (step + 1) % 100 == 0:
                last_errors = errors_hist[-1] if errors_hist else []
                scale_local_val = scale_local.item() if hasattr(scale_local, 'item') else scale_local
                scale_conv_val = scale_conv.item() if hasattr(scale_conv, 'item') else scale_conv
                log = (
                    f'[Step {step + 1}/{iters}] F={F_final:.1f} '
                    f'CE_L={ce_val:.4f} CE_C={ce_conv_val:.4f} '
                    f'bL={β_local:.3f} bC={β_conv:.3f} '
                    f'sL={scale_local_val:.2f} sC={scale_conv_val:.2f} '
                    f'lr={current_lr:.2e}'
                )
                if args.dopamine:
                    log += f' D={D:.3f}'
                    if prev_precision_scales:
                        π_str = ','.join(f'{p:.2f}' for p in prev_precision_scales)
                        log += f' π=[{π_str}]'
                if last_errors and not args.fast:
                    err_str = ' | '.join([f'L{ell+1}:{e[0]:.4f}'
                                          for ell, e in enumerate(last_errors)])
                    Logger(f'  Layer errors: {err_str}')
                    # PC 层统计: 各层误差方差 (反映 PC 推理活跃度)
                    e_sq_vals = [e[0] for e in last_errors]
                    log += f' ε̅={sum(e_sq_vals)/len(e_sq_vals):.4f}'
                    if len(e_sq_vals) > 1:
                        log += f' εσ={__import__("statistics").stdev(e_sq_vals):.4f}'
                Logger(log)

            # ── Checkpoint (每 save_interval 步) ──
            if (step + 1) % args.save_interval == 0 or global_step == 1:
                ckpt = {
                    'epoch': epoch,
                    'step': step,
                    'model_state': pc_model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'F': F_final,
                    'CE_local': ce_val,
                    'CE_converged': ce_conv_val,
                    'beta_local': β_local,
                    'beta_conv': β_conv,
                    'D': D,
                    'lm_config': lm_config,
                    'args': vars(args),
                }
                if args.continual and memory_bank is not None:
                    ckpt['memory_bank'] = memory_bank.state_dict()
                    ckpt['current_task_id'] = current_task_id
                torch.save(ckpt, os.path.join(out_dir, f'unified_ckpt_s{step}.pt'))
                Logger(f'Checkpoint saved at step {step}')

        # ── 任务内 epoch 循环结束 ──

        # ═══════════════════════════════════════════════════════════════
        # Task Finalize: 采样 exemplars → 计算 dopamine_score → 存入 bank
        # ═══════════════════════════════════════════════════════════════
        if args.continual and memory_bank is not None and current_task_id is not None:
            _task_ds = _task_loader.dataset
            n_samples = min(200, len(_task_ds))
            idx = torch.randperm(len(_task_ds))[:n_samples].tolist()
            samples = []
            total_base_loss = 0.0
            with torch.no_grad():
                for i in idx:
                    bt, lt = _task_ds[i]
                    samples.append((bt, lt))
                    # 计算 baseline CE loss (T=0, 无梯度)
                    x = bt.unsqueeze(0).to(device)
                    y = lt.unsqueeze(0).to(device)
                    p = pc_model.get_position_embeddings(x.size(1), device)
                    _, bl = pc_model.forward_with_ce(x, y, p)
                    total_base_loss += bl.item()
            avg_base_loss = total_base_loss / max(len(idx), 1)
            # 用全局 dopamine 分数 (最后一次的 D) 作为重要性权重
            dopamine_score = D if args.dopamine else 0.5
            memory_bank.add_samples(current_task_id, samples, dopamine_score, avg_base_loss)
            Logger(f'[Continual] Task {current_task_id} finalized: '
                   f'{n_samples} exemplars → bank (D={dopamine_score:.3f}, '
                   f'baseline_CE={avg_base_loss:.4f})'
                   f' — bank total: {memory_bank.total}')

            # ═══════════════════════════════════════════════════════════════
            # Cross-task Forgetting Evaluation: 在所有已学任务上测 CE/PPL
            # ═══════════════════════════════════════════════════════════════
            _eval_tasks = task_order[:task_order.index(_task_id) + 1]
            Logger(f'[Eval] Cross-task evaluation on {_eval_tasks}...')
            eval_results = evaluate_cross_tasks(
                pc_model, _eval_tasks, ROOT, device,
                max_seq_len=args.max_seq_len, n_samples=args.eval_samples,
            )
            Logger(f'[Eval] After Task {_task_id}:')
            for tid, metrics in eval_results.items():
                marker = ' ← trained' if tid == _task_id else ''
                Logger(f'  Task {tid}: CE={metrics["ce"]:.4f}, PPL={metrics["ppl"]:.2f}{marker}')
            forgetting_log.append({'after_task': _task_id, 'results': eval_results})
            json.dump(forgetting_log, open(os.path.join(out_dir, 'forgetting_log.json'), 'w'), indent=2)

        # ── Task Checkpoint ──
        if args.continual and current_task_id is not None:
            task_ckpt = {
                'task_id': current_task_id,
                'model_state': pc_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'memory_bank': memory_bank.state_dict() if memory_bank else None,
                'lm_config': lm_config,
                'args': vars(args),
            }
            torch.save(task_ckpt, os.path.join(out_dir, f'task_{current_task_id}_final.pt'))
            Logger(f'Task {current_task_id} checkpoint saved')

    # ═══════════════════════════════════════════════════════════════
    # 最终保存
    # ═══════════════════════════════════════════════════════════════
    final_ckpt = {
        'epoch': args.epochs - 1,
        'step': iters - 1,
        'model_state': pc_model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'F': F_final,
        'CE_local': ce_val,
        'CE_converged': ce_conv_val,
        'beta_local': β_local,
        'beta_conv': β_conv,
        'D': D,
        'lm_config': lm_config,
        'args': vars(args),
    }
    if args.continual and memory_bank is not None:
        final_ckpt['memory_bank'] = memory_bank.state_dict()
    torch.save(final_ckpt, os.path.join(out_dir, 'unified_final.pt'))
    Logger(f'Final checkpoint saved to {out_dir}/unified_final.pt')

    # ── QAT 转换 → 纯 int4 (CPU 上 convert, 避免 CUDA kernel 兼容问题) ──
    if args.quantize and quantizer is not None:
        Logger('Converting to int4 inference format (CPU)...')
        pc_model_cpu = pc_model.cpu()
        try:
            pc_model_cpu = quantizer.convert(pc_model_cpu)
            torch.save(pc_model_cpu.state_dict(), os.path.join(out_dir, 'int4_model.pt'))
            Logger(f'Int4 model saved to {out_dir}/int4_model.pt')
        except Exception as e:
            Logger(f'Int4 convert failed (non-critical): {e}')
            Logger('FP16 checkpoint remains usable.')

    Logger('Unified training complete.')


if __name__ == '__main__':
    setup_seed(42)
    train()
