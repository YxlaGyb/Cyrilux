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
from continual.abstraction_bank import (
    AbstractionBank,
    AbstractionSniffer,
    VariationalReplayer,
    compute_layer_importance,
)

# ═══════════════════════════════════════════════════════════════════
# 数据集
# ═══════════════════════════════════════════════════════════════════

class _LocalDataset(Dataset):
    """双通道 UTF-8 字节数据集 — ch0=原始字节值, ch1=角色编码.

    输入格式:
      - {"conversations": [{"role":..., "content":...}, ...]}: 精确解析 role 分配 ch1
      - {"text": "..."}: ch1 全 2.0 (assistant)
    输出: [2, max_length] float32, [max_length] int64 (labels, pad=-100)
    """
    def __init__(self, data_path, max_length=128, max_samples=None, data_lines=None):
        super().__init__()
        self.dual_tensors = []
        self.label_tensors = []
        role_map = {"user": 1.0, "assistant": 2.0, "system": 3.0, "tool": 4.0}

        if data_lines is not None:
            source = data_lines
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                source = list(f)
        for i, line in enumerate(source):
            if max_samples and i >= max_samples:
                break
            sample = json.loads(line)

            # 从样本提取原始文本 + 角色区间
            if 'conversations' in sample:
                text, roles = self._conversations_to_roles(sample['conversations'])
            else:
                text = str(sample.get('text', ''))
                roles = [(0, len(text), 'assistant')]

            # ch0: 原始 UTF-8 字节值
            byte_seq = text.encode('utf-8')[:max_length]
            padded = byte_seq.ljust(max_length, b'\x00')
            byte_t = torch.frombuffer(bytearray(padded), dtype=torch.uint8).clone().float()

            # ch1: 角色编码 (UTF-8 字节对齐)
            role_t = torch.zeros(max_length, dtype=torch.float)
            for ch_start, ch_end, role_name in roles:
                role_val = role_map.get(role_name, 2.0)
                b_start = len(text[:ch_start].encode('utf-8'))
                b_end = min(len(text[:ch_end].encode('utf-8')), max_length)
                if b_start < max_length:
                    role_t[b_start:b_end] = role_val

            dual_t = torch.stack([byte_t, role_t], dim=0)  # [2, max_length]
            self.dual_tensors.append(dual_t)

            # labels: 仅字节通道, pad 位置 -100
            lbl = byte_t.clone().long()
            lbl[byte_t == 0x00] = -100
            self.label_tensors.append(lbl)

    @staticmethod
    def _conversations_to_roles(conversations):
        """将 conversations 拆解为 (拼接文本, [(字符起始, 字符结束, role名), ...])."""
        raw = ''
        roles = []
        for msg in conversations:
            role = msg.get('role', 'assistant')
            content = msg.get('content', '')
            start = len(raw)
            raw += content
            roles.append((start, len(raw), role))
        return raw, roles

    def __len__(self):
        return len(self.dual_tensors)

    def __getitem__(self, index):
        return self.dual_tensors[index].clone(), self.label_tensors[index].clone()


def _build_task_list(task_files_str, split_size, data_dir, max_length, max_samples):
    """根据 --task_files / --split_size / dataset/ 构建任务列表.

    返回: [(task_id, _LocalDataset), ...]
    """
    tasks = []
    if task_files_str:
        paths = [p.strip() for p in task_files_str.split(',') if p.strip()]
    else:
        if os.path.isdir(data_dir):
            paths = sorted([
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith('.jsonl')
            ])
        else:
            paths = []

    for fp in paths:
        if not os.path.isfile(fp):
            Logger(f'[Data] \u8df3\u8fc7: {fp} (\u4e0d\u5b58\u5728)')
            continue
        base = os.path.splitext(os.path.basename(fp))[0]
        if split_size > 0:
            with open(fp, 'r', encoding='utf-8') as f:
                all_lines = [l for l in f][:max_samples] if max_samples else list(f)
            n_chunks = (len(all_lines) + split_size - 1) // split_size
            for ci in range(n_chunks):
                chunk_lines = all_lines[ci * split_size:(ci + 1) * split_size]
                tid = f'{base}_part{ci + 1}'
                ds = _LocalDataset(fp, max_length=max_length, max_samples=None, data_lines=chunk_lines)
                tasks.append((tid, ds))
            Logger(f'[Data] {base}: {len(all_lines)} \u6761 \u2192 {n_chunks} \u5757')
        else:
            ds = _LocalDataset(fp, max_length=max_length, max_samples=max_samples)
            tasks.append((base, ds))
            Logger(f'[Data] {base}: {len(ds)} \u6761 (\u4e0d\u5206\u5757)')
    return tasks


warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════
# 跨任务遗忘评估
# ═══════════════════════════════════════════════════════════════════

def evaluate_cross_tasks(model, task_ds_list, device, max_seq_len=128, n_samples=100):
    """评估模型在给定任务上的 CE / PPL。

    Args:
        task_ds_list: [(task_id, Dataset), ...]
    Returns: {task_id: {ce, ppl}}
    """
    results = {}
    model.eval()
    with torch.no_grad():
        for tid, ds in task_ds_list:
            n_eval = min(n_samples, len(ds))
            total_ce = 0.0
            for i in range(n_eval):
                bt, lt = ds[i]
                x = bt.unsqueeze(0).to(device)
                y = lt.unsqueeze(0).to(device)
                p = model.get_position_embeddings(x.size(-1), device)
                _, ce = model.forward_with_ce(x, y, p)
                total_ce += ce.item()
            avg_ce = total_ce / max(n_eval, 1)
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
    parser.add_argument('--T_infer', type=int, default=2, help='PC 时空推理步数')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--max_beta', type=float, default=2.0,
                        help='CE_local 权重上限')
    parser.add_argument('--max_beta_conv', type=float, default=1.0,
                        help='CE_converged 权重上限')
    parser.add_argument('--ema_lambda', type=float, default=0.001,
                        help='EMA 正则强度 (防坍塌)')
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # 多巴胺
    parser.add_argument('--dopamine_eta', type=float, default=1.0,
                        help='精度调制强度 η')
    parser.add_argument('--dopamine_beta', type=float, default=0.5,
                        help='学习率调制强度 β')
    parser.add_argument('--dopamine_gamma', type=float, default=0.3,
                        help='loss 平衡调制强度 γ')

    # QAT / 编译
    parser.add_argument('--qat_groupsize', type=int, default=64)
    parser.add_argument('--no_quantize_embed', action='store_true',
                        help='不量化 embed/lm_head (实验性)')

    # 抽象记忆银行参数
    parser.add_argument('--n_prototypes', type=int, default=8,
                        help='每任务原型数 (默认 8)')
    parser.add_argument('--abstraction_replay_interval', type=int, default=200,
                        help='抽象级回放间隔 (默认 200 步)')
    parser.add_argument('--abstraction_sniff_interval', type=int, default=300,
                        help='抽象漂移检测间隔 (默认 300 步)')
    parser.add_argument('--abstraction_drift_threshold', type=float, default=0.7,
                        help='抽象漂移阈值: cosine < threshold 触发修复 (默认 0.7)')

    # I/O
    parser.add_argument('--out_dir', type=str, default='out_pc_unified',
                        help='输出目录')
    parser.add_argument('--data_path', type=str, default=None,
                        help='数据集路径 (默认 datasets 下的 4 任务文件)')
    parser.add_argument('--save_interval', type=int, default=500)

    # 持续学习
    parser.add_argument('--task_order', type=str, default='a,b,c,d',
                        help='任务顺序, 逗号分隔')
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

    # 自定义数据
    parser.add_argument('--task_files', type=str, default=None,
                        help='逗号分隔的 JSONL 路径, 例如 E:/a.jsonl,E:/b.jsonl')
    parser.add_argument('--split_size', type=int, default=0,
                        help='大文件分块: >0 则每块 N 条样本 (0=不分块)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 恢复训练 (.pt 路径)')

    # GUI 集成
    parser.add_argument('--gui', action='store_true',
                        help='启动桌面 GUI 窗口 (Tkinter)')
    parser.add_argument('--gui_mode', action='store_true',
                        help='GUI 模式: 从 --config_file 加载参数')
    parser.add_argument('--config_file', type=str, default=None,
                        help='GUI 模式下的 JSON 配置文件路径')

    return parser.parse_args()


def train():
    args = parse_args()

    # ── GUI 模式: 从 JSON 配置文件覆写参数 ──
    if args.gui_mode and args.config_file:
        with open(args.config_file, 'r', encoding='utf-8') as f:
            gui_cfg = json.load(f)
        for key, val in gui_cfg.items():
            if hasattr(args, key):
                setattr(args, key, val)
        Logger(f'GUI 模式: 从 {args.config_file} 加载配置 ({len(gui_cfg)} 项)')

    # ── 模型配置 ──
    lm_config = MiniMindConfig(hidden_size=256, num_hidden_layers=4, use_moe=False)

    ROOT = os.path.dirname(os.path.abspath(__file__))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data_path = args.data_path or os.path.join(ROOT, 'dataset', '')
    out_dir = os.path.join(ROOT, args.out_dir)
    # ── 构建描述 ──
    desc_parts = ['Hybrid', f'DA(η={args.dopamine_eta},β={args.dopamine_beta},γ={args.dopamine_gamma})', f'QAT(g={args.qat_groupsize})']
    desc = '+'.join(desc_parts)

    Logger(f'=== {desc} Training ===')
    Logger(f'T={args.T_infer}, γ={args.gamma}, lr={args.lr}, batch={args.batch_size}')
    Logger(f'Device: {device}')

    # ── 模型创建 (CPU) — QAT prepare 必须在 CPU ──
    pc_model = PCLocalDynamicMiniMind(lm_config)
    base_params = sum(p.numel() for p in pc_model.parameters() if p.requires_grad)
    Logger(f'Base params: {base_params / 1e6:.2f}M')

    # ── QAT 准备 (CPU 上执行) ──
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
    if hasattr(torch, 'compile'):
        _orig_forward = pc_model.forward_with_ce
        try:
            pc_model.forward_with_ce = torch.compile(_orig_forward,
                                                      mode='reduce-overhead')
            Logger('torch.compile 启用 (mode=reduce-overhead)')
        except Exception as e:
            pc_model.forward_with_ce = _orig_forward
            Logger(f'torch.compile 失败 (已忽略): {e}')
    else:
        _orig_forward = None
    qat_params = sum(p.numel() for p in pc_model.parameters() if p.requires_grad)
    Logger(f'QAT params: {qat_params / 1e6:.2f}M (trainable)')

    # ── 数据 (task_files / dataset/ 自动发现) ──
    all_tasks = _build_task_list(
        args.task_files, args.split_size, data_path,
        args.max_seq_len, args.subset,
    )
    if not all_tasks:
        Logger('[Data] ⚠ 未找到任何数据文件, 退出')
        return
    Logger(f'[Data] {len(all_tasks)} task(s) loaded')

    # 兼容: 非 task_order 模式 → 使用第一任务的 loader
    first_loader = DataLoader(all_tasks[0][1], batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    iters = len(first_loader)

    # ── 预热: 吸收 cudnn benchmark 首步算法搜索延迟 ──
    with torch.no_grad():
        dummy_byte = torch.randint(0, 256, (args.batch_size, args.max_seq_len), device=device)
        dummy = torch.stack([
            dummy_byte.float(),
            torch.full_like(dummy_byte, 2.0, dtype=torch.float, device=device),
        ], dim=1)  # [batch, 2, seq]
        dummy_pos = pc_model.get_position_embeddings(args.max_seq_len, device)
        try:
            _, _ = pc_model.forward_with_ce(dummy, dummy_byte, dummy_pos)
        except Exception as e:
            # torch.compile 惰性编译失败 (Windows 无 Triton) → 回退原始 forward
            if _orig_forward is not None:
                pc_model.forward_with_ce = _orig_forward
                Logger(f'torch.compile 已回退 (首次调用失败: {e})')
                _, _ = pc_model.forward_with_ce(dummy, dummy_byte, dummy_pos)
            else:
                raise
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
    dopamine = DopamineSignal(η=args.dopamine_eta, threshold=0.0)
    # 冷启动: 前几步 D=0.5, 避免 F_prev=inf → D≈0
    dopamine.F_prev = None
    _dopamine_warmup_steps = 10

    # ── 精度权重缓存 (per-layer, 上一步的值) ──
    prev_precision_scales = None

    # ── 持续学习初始化 (始终启用) ──
    memory_bank = MemoryBank(max_per_task=args.bank_size)
    sniffer = ForgettingSniffer(
        memory_bank=memory_bank, model=pc_model,
        check_interval=args.sniff_interval,
        threshold=args.repair_threshold,
        repair_steps=args.repair_steps,
    )
    offline_replayer = OfflineReplayer(memory_bank, pc_model)
    abstraction_bank = AbstractionBank(
        max_entries_per_task=args.bank_size,
        n_prototypes=args.n_prototypes,
        consolidation_frequency=1,
    )
    abstraction_sniffer = AbstractionSniffer(
        bank=abstraction_bank, model=pc_model,
        check_interval=args.abstraction_sniff_interval,
        drift_threshold=args.abstraction_drift_threshold,
    )
    variational_replayer = VariationalReplayer(pc_model, abstraction_bank)
    Logger(f'  AbstractionBank: {args.n_prototypes} proto/task')

    # 从 _build_task_list 提取任务 ID 列表
    task_order = [tid for tid, _ in all_tasks]
    trained_tasks = []  # 记录已学任务 (用于 cross-task eval)
    Logger(f'Continual learning: {len(task_order)} tasks → {", ".join(task_order)}')
    Logger(f'  replay_ratio={args.replay_ratio}, bank_size={args.bank_size}')
    Logger(f'  sniff_interval={args.sniff_interval}, repair_threshold={args.repair_threshold}')

    # ── 训练循环 ──
    pc_model.train()
    global_step = 0
    total_steps = iters * args.epochs

    # EMA 参考 (防坍塌)
    ema_z = None

    # ── 任务序列 (持续学习) ──
    current_task_id = None
    _task_loader = first_loader  # fallback

    # 外层: 任务循环 (遍历 _build_task_list 构建的 all_tasks)
    for _task_id, _task_ds in all_tasks:
        current_task_id = _task_id
        _task_loader = DataLoader(_task_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)
        Logger(f'\n{"="*60}\nStarting Task {_task_id}: {len(_task_ds)} samples\n{"="*60}')
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
                bsz, _, seq_len = byte_seq.shape
                global_step += 1

                # ═══════════════════════════════════════════════════════
                # Phase 1: 共享前向 (有梯度)
                # ═══════════════════════════════════════════════════════
                pos_emb = pc_model.get_position_embeddings(seq_len, device)
                z_init, ce_loss = pc_model.forward_with_ce(byte_seq, labels, pos_emb)

            # ═════════════════════════════════════════════════
            # Phase 2+: PC 推理 + 多路损失
            # ═════════════════════════════════════════════════
            z_detached = [z.detach() for z in z_init]
            # ponytail: return_errors 只需用于多巴胺精度调制
            z_converged, errors_hist, F_hist, F_pred = pc_model.spatiotemporal_infer(
                    z_detached, pos_emb, gamma=args.gamma, T=args.T_infer,
                    return_errors=True,
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
            current_lr = current_lr * (1.0 + args.dopamine_beta * D)

            for pg in optimizer.param_groups:
                pg['lr'] = current_lr

            optimizer.step()

            # ponytail: EMA 跳过 (不影响收敛, 省 per-step clone/add)

            # ═══════════════════════════════════════════════════════
            # 持续学习: 记忆回放 (每 replay_ratio 步插入 1 步纯 CE 回放)
            # ═══════════════════════════════════════════════════════
            if memory_bank.total > 0:
                if global_step % args.replay_ratio == 0 and not sniffer.is_repairing:
                    replay_ex = memory_bank.sample(args.batch_size, strategy='dopamine')
                    if replay_ex:
                        replay_byte = torch.stack([ex.byte_tensor for ex in replay_ex], dim=0).to(device)
                        replay_label = torch.stack([ex.label_tensor for ex in replay_ex], dim=0).to(device)
                        replay_pos = pc_model.get_position_embeddings(replay_byte.size(-1), device)
                        _, replay_loss = pc_model.forward_with_ce(replay_byte, replay_label, replay_pos)
                        optimizer.zero_grad(set_to_none=True)
                        replay_loss.backward()
                        trainable_rp = [p for p in pc_model.parameters()
                                        if p.requires_grad and p.grad is not None]
                        if trainable_rp:
                            torch.nn.utils.clip_grad_norm_(trainable_rp, args.grad_clip)
                        optimizer.step()

            # ═══════════════════════════════════════════════════════
            # 持续学习: AbstractionBank 表示级回放 (每 abstraction_replay_interval 步)
            # ═══════════════════════════════════════════════════════
            if global_step % args.abstraction_replay_interval == 0:
                    r_loss = abstraction_bank.replay_loss(
                        pc_model, batch_size=16, device=device,
                        pos_emb=(None, None),
                    )
                    if r_loss is not None:
                        optimizer.zero_grad(set_to_none=True)
                        r_loss.backward()
                        trainable_rp = [p for p in pc_model.parameters()
                                        if p.requires_grad and p.grad is not None]
                        if trainable_rp:
                            torch.nn.utils.clip_grad_norm_(trainable_rp, args.grad_clip)
                        optimizer.step()
                        Logger(f'[AbstractionBank] Replay step {global_step}: loss={r_loss.item():.4f}')

                    # 定时 consolidate (压缩原型)
                    if global_step % (args.abstraction_replay_interval * 5) == 0:
                        for tid in abstraction_bank._store:
                            abstraction_bank.consolidate(tid)
                            n_p = abstraction_bank.get_num_prototypes(tid)
                            Logger(f'[AbstractionBank] Consolidated {tid}: {n_p} prototypes')

            # ═══════════════════════════════════════════════════════
            # 持续学习: AbstractionSniffer 抽象漂移检测
            # ═══════════════════════════════════════════════════════
            drifted = abstraction_sniffer.check(global_step, device, pos_emb=(None, None))
            if drifted:
                current_lr_sniff = current_lr
                repair_lr = abstraction_sniffer.repair_begin(optimizer, current_lr_sniff)
                Logger(f'[AbstractionSniffer] DRIFT detected: {drifted} — repair LR={repair_lr:.2e}')
                for _ in range(abstraction_sniffer.repair_steps):
                    r_loss = abstraction_bank.replay_loss(
                        pc_model, batch_size=16, device=device,
                        pos_emb=(None, None),
                    )
                    if r_loss is None:
                        break
                    optimizer.zero_grad(set_to_none=True)
                    r_loss.backward()
                    trainable_rp = [p for p in pc_model.parameters()
                                    if p.requires_grad and p.grad is not None]
                    if trainable_rp:
                        torch.nn.utils.clip_grad_norm_(trainable_rp, args.grad_clip)
                    optimizer.step()
                abstraction_sniffer.repair_end(optimizer, current_lr_sniff)
                Logger(f'[AbstractionSniffer] Repair complete — LR restored')

            # ═══════════════════════════════════════════════════════
            # 持续学习: 遗忘嗅探 + 自触发修复
            # ═══════════════════════════════════════════════════════
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
                    rp_pos = pc_model.get_position_embeddings(rp_byte.size(-1), device)
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
            postfix['D'] = f'{D:.3f}'
            pbar.set_postfix(**postfix)

            # ── 详细日志 (每 100 步) ──
            if (step + 1) % 100 == 0:
                last_errors = errors_hist[-1] if errors_hist else []
                scale_local_val = scale_local.item() if hasattr(scale_local, 'item') else scale_local
                scale_conv_val = scale_conv.item() if hasattr(scale_conv, 'item') else scale_conv
                log = (
                    f'[Step {step + 1}/{total_steps}] F={F_final:.1f} '
                    f'CE_L={ce_val:.4f} CE_C={ce_conv_val:.4f} '
                    f'bL={β_local:.3f} bC={β_conv:.3f} '
                    f'sL={scale_local_val:.2f} sC={scale_conv_val:.2f} '
                    f'lr={current_lr:.2e}'
                )
                log += f' D={D:.3f}'
                if prev_precision_scales:
                        π_str = ','.join(f'{p:.2f}' for p in prev_precision_scales)
                        log += f' π=[{π_str}]'
                if last_errors:
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
                ckpt['memory_bank'] = memory_bank.state_dict()
                ckpt['current_task_id'] = current_task_id
                ckpt['abstraction_bank'] = abstraction_bank.state_dict()
                torch.save(ckpt, os.path.join(out_dir, f'unified_ckpt_s{step}.pt'))
                Logger(f'Checkpoint saved at step {step}')

        # ── 任务内 epoch 循环结束 ──

        # ═══════════════════════════════════════════════════════════════
        # Task Finalize: 采样 exemplars → 计算 dopamine_score → 存入 bank
        # ═══════════════════════════════════════════════════════════════
        if current_task_id is not None:
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
                    p = pc_model.get_position_embeddings(x.size(-1), device)
                    _, bl = pc_model.forward_with_ce(x, y, p)
                    total_base_loss += bl.item()
            avg_base_loss = total_base_loss / max(len(idx), 1)
            # 用全局 dopamine 分数 (最后一次的 D) 作为重要性权重
            dopamine_score = D
            memory_bank.add_samples(current_task_id, samples, dopamine_score, avg_base_loss)
            Logger(f'[Continual] Task {current_task_id} finalized: '
                   f'{n_samples} exemplars → bank (D={dopamine_score:.3f}, '
                   f'baseline_CE={avg_base_loss:.4f})'
                   f' — bank total: {memory_bank.total}')

            # ── AbstractionBank: 从 exemplars 收集 PC 收敛后的 z 表示 ──
            z_collected = []
            for bt, lt in samples:
                    x = bt.unsqueeze(0).to(device)       # [1, seq]
                    y = lt.unsqueeze(0).to(device)
                    # 先 forward 的 z = init_z 结果 (梯度关)
                    with torch.no_grad():
                        z_init = pc_model.init_z(x)
                    # PC 推理收敛
                    z_conv, errors_hist, F_hist, _ = pc_model.spatiotemporal_infer(
                        z_init, pos_emb=(None, None),
                        gamma=0.1, T=4,
                        return_errors=False, return_pred_loss=False,
                    )
                    # 用 dopamine 调制
                    D_sample = D
                    z_collected.append(z_conv)

                    if len(z_collected) >= min(n_samples, 100):
                        break

            if z_collected:
                # 计算层重要性 (取第一个样本作为代表)
                layer_imp = compute_layer_importance(
                    z_collected[0], pc_model, (None, None),
                    dopamine_D=D_sample, eta=1.0,
                )
                abstraction_bank.add_z_samples(
                    current_task_id, z_collected,
                    layer_importance=layer_imp,
                    dopamine_score=D_sample,
                )
                abstraction_bank.consolidate(current_task_id)
                n_protos = abstraction_bank.get_num_prototypes(current_task_id)
                Logger(f'[AbstractionBank] Task {current_task_id}: '
                       f'{len(z_collected)} z_states → {n_protos} prototypes'
                       f' (η={abstraction_bank.total_prototypes} total)')

            # ═══════════════════════════════════════════════════════════════
            # Cross-task Forgetting Evaluation: 在所有已学任务上测 CE/PPL
            # ═══════════════════════════════════════════════════════════════
            trained_tasks.append(_task_id)
            _eval_ds_list = [(tid, ds) for tid, ds in all_tasks if tid in trained_tasks]
            Logger(f'[Eval] Cross-task evaluation on {trained_tasks}...')
            eval_results = evaluate_cross_tasks(
                pc_model, _eval_ds_list, device,
                max_seq_len=args.max_seq_len, n_samples=args.eval_samples,
            )
            Logger(f'[Eval] After Task {_task_id}:')
            for tid, metrics in eval_results.items():
                marker = ' ← trained' if tid == _task_id else ''
                Logger(f'  Task {tid}: CE={metrics["ce"]:.4f}, PPL={metrics["ppl"]:.2f}{marker}')
            forgetting_log.append({'after_task': _task_id, 'results': eval_results})
            json.dump(forgetting_log, open(os.path.join(out_dir, 'forgetting_log.json'), 'w'), indent=2)

        # ── Task Checkpoint ──
        if current_task_id is not None:
            task_ckpt = {
                'task_id': current_task_id,
                'model_state': pc_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'memory_bank': memory_bank.state_dict() if memory_bank else None,
                'abstraction_bank': abstraction_bank.state_dict() if abstraction_bank else None,
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
        'step': global_step,
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
    final_ckpt['memory_bank'] = memory_bank.state_dict()
    final_ckpt['abstraction_bank'] = abstraction_bank.state_dict()
    torch.save(final_ckpt, os.path.join(out_dir, 'unified_final.pt'))
    Logger(f'Final checkpoint saved to {out_dir}/unified_final.pt')

    # ── QAT 转换 → 纯 int4 (CPU 上 convert, 避免 CUDA kernel 兼容问题) ──
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


def train_from_config(config_dict: dict, config_path: str = None):
    """
    GUI 调用入口: 通过字典直接启动训练, 绕过 argparse.

    用法:
        cfg = {'batch_size': 48, 'lr': 3e-4, 'data_path': '...', ...}
        train_from_config(cfg)

    如果提供 config_path, 会写入 JSON 并带上 --gui_mode 重新执行.
    否则直接 hack argparse 的 Namespace 后调用 train().
    """
    # 方案 A: 写入临时配置文件 → 子进程启动
    if config_path:
        os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        # 覆写 sys.argv
        sys.argv = [sys.argv[0], '--gui_mode', '--config_file', config_path]
        # 重新运行 train()
        train()
        return

    # 方案 B: 直接 h进入 train() 但 hack argparse
    import argparse
    # 用默认值创建 namespace
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--subset', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--T_infer', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--max_beta', type=float, default=2.0)
    parser.add_argument('--max_beta_conv', type=float, default=1.0)
    parser.add_argument('--ema_lambda', type=float, default=0.001)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--dopamine_eta', type=float, default=1.0)
    parser.add_argument('--dopamine_beta', type=float, default=0.5)
    parser.add_argument('--dopamine_gamma', type=float, default=0.3)
    parser.add_argument('--qat_groupsize', type=int, default=64)
    parser.add_argument('--no_quantize_embed', action='store_true', default=False)
    parser.add_argument('--out_dir', type=str, default='out_pc_unified')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--task_order', type=str, default='a,b,c,d')
    parser.add_argument('--replay_ratio', type=int, default=5)
    parser.add_argument('--bank_size', type=int, default=2000)
    parser.add_argument('--sniff_interval', type=int, default=200)
    parser.add_argument('--repair_threshold', type=float, default=1.2)
    parser.add_argument('--repair_steps', type=int, default=10)
    parser.add_argument('--eval_samples', type=int, default=100)
    parser.add_argument('--task_files', type=str, default=None)
    parser.add_argument('--split_size', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--gui', action='store_true', default=False)
    parser.add_argument('--gui_mode', action='store_true', default=False)
    parser.add_argument('--config_file', type=str, default=None)
    args = parser.parse_args([])
    # 用 config_dict 覆写
    for key, val in config_dict.items():
        if hasattr(args, key):
            setattr(args, key, val)
    # 手动调用训练 (不通过 train() 避免二次 parse)
    _run_train_from_args(args)


def _run_train_from_args(args):
    """被 train_from_config() 调用: 覆写 sys.argv 后委托 train()."""
    import sys as _sys
    # 将 args.Namespace 转成 sys.argv 格式
    argv = [_sys.argv[0]]
    for key, val in vars(args).items():
        key_str = f'--{key.replace("_", "-")}'
        if isinstance(val, bool):
            if val:
                argv.append(key_str)
        elif val is not None:
            argv.extend([key_str, str(val)])
    _sys.argv = argv
    train()


# ═══════════════════════════════════════════════════════════════════
# 桌面 GUI (Tkinter)
# ═══════════════════════════════════════════════════════════════════

def launch_gui():
    """启动 Tkinter 桌面训练 GUI。零外部依赖 (stdlib only)。"""
    # ── Windows DPI 感知 (125% / 150% 缩放) ──
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # per-monitor DPI
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    import tkinter as tk
    from tkinter import ttk, filedialog, scrolledtext
    import threading
    import queue

    # ── 线程安全日志队列: 训练线程入列, 主线程轮询 ──
    _log_queue = queue.Queue()

    class GuiLogger:
        """只往队列写, 不碰 tkinter (线程安全)."""
        def write(self, text):
            if text.strip():
                _log_queue.put(text)
        def flush(self):
            pass

    def _poll_log():
        """主线程每 80ms 轮询日志队列并刷新 GUI."""
        try:
            while True:
                text = _log_queue.get_nowait()
                log_area.insert(tk.END, text)
                log_area.see(tk.END)
        except queue.Empty:
            pass
        except Exception as exc:
            # 日志轮询自身不崩溃
            print(f'[_poll_log] {exc}', file=sys.stderr)
        root.after(80, _poll_log)

    # ── 主窗口 ──
    root = tk.Tk()
    # DPI 缩放适配 (125%=1.25, 150%=1.5 …)
    try:
        root.tk.call('tk', 'scaling', root.tk.call('tk', 'scaling') * 1.25)
    except Exception:
        pass
    root.title('MiniMind PC Unified Trainer')
    root.geometry('900x720')
    root.minsize(800, 640)

    style = ttk.Style()
    style.theme_use('vista' if 'vista' in style.theme_names() else 'clam')

    main_frame = ttk.Frame(root, padding=12)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # ── 数据文件区 ──
    data_lf = ttk.LabelFrame(main_frame, text='数据文件', padding=8)
    data_lf.pack(fill=tk.X, pady=(0, 8))

    file_frame = ttk.Frame(data_lf)
    file_frame.pack(fill=tk.X)

    file_listbox = tk.Listbox(file_frame, height=5, selectmode=tk.EXTENDED)
    file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    file_scroll = ttk.Scrollbar(file_frame, orient=tk.VERTICAL, command=file_listbox.yview)
    file_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    file_listbox.config(yscrollcommand=file_scroll.set)

    btn_frame = ttk.Frame(data_lf)
    btn_frame.pack(fill=tk.X, pady=(4, 0))

    _selected_files = []

    def add_files():
        paths = filedialog.askopenfilenames(
            title='选择 JSONL 文件',
            filetypes=[('JSONL', '*.jsonl'), ('All', '*.*')],
        )
        for p in paths:
            if p not in _selected_files:
                _selected_files.append(p)
                file_listbox.insert(tk.END, p)

    def remove_selected():
        for sel in reversed(file_listbox.curselection()):
            file_listbox.delete(sel)
            del _selected_files[sel]

    ttk.Button(btn_frame, text='添加文件...', command=add_files).pack(side=tk.LEFT, padx=(0, 4))
    ttk.Button(btn_frame, text='移除选中', command=remove_selected).pack(side=tk.LEFT)

    # ── 参数区 (两列) ──
    param_nb = ttk.Notebook(main_frame)
    param_nb.pack(fill=tk.X, pady=(0, 8))

    # Tab 1: 训练参数
    train_tab = ttk.Frame(param_nb, padding=8)
    param_nb.add(train_tab, text='训练参数')

    params = {}
    entries = [
        ('batch_size', '48'), ('max_seq_len', '128'), ('lr', '3e-4'),
        ('epochs', '1'), ('subset', '10000'), ('seed', '42'),
        ('split_size', '0'),
    ]
    for i, (key, default) in enumerate(entries):
        lbl = ttk.Label(train_tab, text=f'{key}:')
        lbl.grid(row=i // 2, column=(i % 2) * 2, sticky=tk.W, padx=(0, 4), pady=2)
        var = tk.StringVar(value=default)
        ent = ttk.Entry(train_tab, textvariable=var, width=18)
        ent.grid(row=i // 2, column=(i % 2) * 2 + 1, sticky=tk.W, padx=(0, 16), pady=2)
        params[key] = var

    # Tab 2: PC / 多巴胺
    pc_tab = ttk.Frame(param_nb, padding=8)
    param_nb.add(pc_tab, text='PC / 多巴胺')

    pc_entries = [
        ('T_infer', '2'), ('gamma', '0.1'), ('max_beta', '2.0'),
        ('max_beta_conv', '1.0'), ('grad_clip', '1.0'),
        ('dopamine_eta', '1.0'), ('dopamine_beta', '0.5'), ('dopamine_gamma', '0.3'),
    ]
    for i, (key, default) in enumerate(pc_entries):
        lbl = ttk.Label(pc_tab, text=f'{key}:')
        lbl.grid(row=i // 2, column=(i % 2) * 3, sticky=tk.W, padx=(0, 4), pady=2)
        var = tk.StringVar(value=default)
        ent = ttk.Entry(pc_tab, textvariable=var, width=14)
        ent.grid(row=i // 2, column=(i % 2) * 3 + 1, sticky=tk.W, padx=(0, 8), pady=2)
        params[key] = var

    # Tab 3: 持续学习
    cl_tab = ttk.Frame(param_nb, padding=8)
    param_nb.add(cl_tab, text='持续学习')

    cl_entries = [
        ('replay_ratio', '5'), ('bank_size', '2000'), ('sniff_interval', '200'),
        ('repair_threshold', '1.2'), ('repair_steps', '10'), ('eval_samples', '100'),
        ('n_prototypes', '8'), ('abstraction_replay_interval', '200'),
    ]
    for i, (key, default) in enumerate(cl_entries):
        lbl = ttk.Label(cl_tab, text=f'{key}:')
        lbl.grid(row=i // 2, column=(i % 2) * 3, sticky=tk.W, padx=(0, 4), pady=2)
        var = tk.StringVar(value=default)
        ent = ttk.Entry(cl_tab, textvariable=var, width=14)
        ent.grid(row=i // 2, column=(i % 2) * 3 + 1, sticky=tk.W, padx=(0, 8), pady=2)
        params[key] = var

    # Tab 4: QAT
    qat_tab = ttk.Frame(param_nb, padding=8)
    param_nb.add(qat_tab, text='QAT')

    qat_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(qat_tab, text='no_quantize_embed', variable=qat_var).grid(row=0, column=0, sticky=tk.W)
    params['no_quantize_embed'] = qat_var

    # ── 控制区 ──
    ctrl_frame = ttk.Frame(main_frame)
    ctrl_frame.pack(fill=tk.X, pady=(0, 8))

    _train_thread = None
    _stop_event = threading.Event()

    def start_training():
        nonlocal _train_thread
        # 立即显示反馈: 写一条初始消息到日志队列 (主线程, 直写 log_area 确保立刻可见)
        log_area.delete(1.0, tk.END)
        log_area.insert(tk.END, '正在初始化训练参数...\n')
        log_area.see(tk.END)
        progress_bar['value'] = 0
        start_btn.config(state=tk.DISABLED)
        stop_btn.config(state=tk.NORMAL)

        # 收集参数
        try:
            cfg = {}
            for key, var in params.items():
                val = var.get()
                if key in ('batch_size', 'max_seq_len', 'epochs', 'subset', 'seed',
                           'T_infer', 'replay_ratio', 'bank_size', 'sniff_interval',
                           'repair_steps', 'eval_samples', 'n_prototypes',
                           'abstraction_replay_interval', 'split_size'):
                    cfg[key] = int(val)
                elif key in ('lr', 'gamma', 'max_beta', 'max_beta_conv', 'grad_clip',
                             'dopamine_eta', 'dopamine_beta', 'dopamine_gamma',
                             'repair_threshold'):
                    cfg[key] = float(val)
                elif key == 'no_quantize_embed':
                    cfg[key] = bool(val)
                else:
                    cfg[key] = val
            cfg['task_files'] = ','.join(_selected_files) if _selected_files else None

            if not _selected_files:
                log_area.insert(tk.END, '⚠ 未添加数据文件, 将使用 dataset/ 自动发现\n')
                log_area.see(tk.END)
        except Exception as e:
            log_area.insert(tk.END, f'❌ 参数解析失败: {e}\n')
            log_area.see(tk.END)
            start_btn.config(state=tk.NORMAL)
            stop_btn.config(state=tk.DISABLED)
            return

        # 重定向 Logger 到 GUI (线程安全: GuiLogger 只写队列)
        import builtins
        logger = GuiLogger()
        _orig_print_builtin = builtins.print

        def _gui_print(*args, **kwargs):
            text = ' '.join(str(a) for a in args)
            logger.write(text + '\n')

        # 在日志中提示训练启动
        _log_queue.put('训练线程已启动, 请等待模型初始化...\n')

        def train_task():
            try:
                builtins.print = _gui_print

                train_from_config(cfg)

                _log_queue.put('\n✅ 训练完成!\n')
                root.after(0, lambda: progress_bar.config(value=100))
            except Exception as e:
                _log_queue.put(f'\n❌ 训练失败: {e}\n')
                import traceback
                _log_queue.put(traceback.format_exc() + '\n')
            finally:
                root.after(0, lambda: start_btn.config(state=tk.NORMAL))
                root.after(0, lambda: stop_btn.config(state=tk.DISABLED))
                builtins.print = _orig_print_builtin

        _stop_event.clear()
        _train_thread = threading.Thread(target=train_task, daemon=True)
        _train_thread.start()

    def stop_training():
        _stop_event.set()
        GuiLogger().write('⏹ 停止信号已发送 (等待当前 step 完成后退出)\n')

    start_btn = ttk.Button(ctrl_frame, text='▶ 开始训练', command=start_training)
    start_btn.pack(side=tk.LEFT, padx=(0, 4))
    stop_btn = ttk.Button(ctrl_frame, text='⏹ 停止', command=stop_training, state=tk.DISABLED)
    stop_btn.pack(side=tk.LEFT)

    # ── 进度条 ──
    progress_bar = ttk.Progressbar(main_frame, mode='determinate')
    progress_bar.pack(fill=tk.X, pady=(0, 8))

    # ── 日志区 ──
    log_lf = ttk.LabelFrame(main_frame, text='训练日志', padding=4)
    log_lf.pack(fill=tk.BOTH, expand=True)

    log_area = scrolledtext.ScrolledText(log_lf, height=14, font=('Consolas', 9), wrap=tk.WORD)
    log_area.pack(fill=tk.BOTH, expand=True)

    # ── 启动日志轮询 (主线程) ──
    root.after(80, _poll_log)

    root.mainloop()


if __name__ == '__main__':
    # ── GUI 模式 → 启动桌面窗口 ──
    if '--gui' in sys.argv or '-g' in sys.argv:
        launch_gui()
    else:
        setup_seed(42)
        train()
