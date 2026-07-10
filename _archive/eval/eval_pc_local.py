"""
时空预测编码评估 — 5 维自监督指标。

  1. F_total      — 总预测误差 (越低越好)
  2. sparsity     — ℓ1/ℓ2 每层稀疏度 (越高越稀疏)
  3. smoothness   — ‖z(t+1)-z(t)‖ 时序平滑度 (越低越平滑)
  4. variance     — 每层表示方差 (抗坍塌, 越高越好)
  5. recon_acc    — LM head 解码 z_L → token 重建准确率

Ponytail: 无 label 依赖, 全自监督。
"""
import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from model.pc_layers import PCDynamicMiniMind
from model.model_minimind import MiniMindConfig
from trainer_utils import Logger
from train_pc_local import _LocalDataset
from transformers import AutoTokenizer


@torch.no_grad()
def evaluate(pc_model, loader, pos_emb, gamma=0.1, T=2, max_batches=20):
    """5 维自监督评估。"""
    pc_model.eval()

    metrics = {
        'F': [],
        'sparsity': {f'L{i}': [] for i in range(1, pc_model.num_sub_layers + 1)},
        'smoothness': {f'L{i}': [] for i in range(1, pc_model.num_sub_layers + 1)},
        'variance': {f'L{i}': [] for i in range(1, pc_model.num_sub_layers + 1)},
        'recon_acc': [],
    }

    for batch_idx, (input_ids, labels) in enumerate(loader):
        if batch_idx >= max_batches:
            break

        input_ids = input_ids.to(next(pc_model.parameters()).device)
        bsz, seq_len = input_ids.shape
        pos = (pos_emb[0][:seq_len].to(input_ids.device),
               pos_emb[1][:seq_len].to(input_ids.device))

        # ── 时空推理 ──
        z_by_layer = pc_model.init_z(input_ids)
        z_by_layer, errors_hist, F_hist = pc_model.spatiotemporal_infer(
            z_by_layer, pos, gamma=gamma, T=T,
        )

        # 1. F_total
        F_val = F_hist[-1] / (bsz * seq_len)
        metrics['F'].append(F_val)

        # 2-4. 表示质量
        rep_metrics = pc_model.compute_representation_metrics(z_by_layer)
        for ℓ in range(1, pc_model.num_sub_layers + 1):
            metrics['sparsity'][f'L{ℓ}'].append(rep_metrics['sparsity'][ℓ - 1])
            metrics['smoothness'][f'L{ℓ}'].append(rep_metrics['temporal_smoothness'][ℓ - 1])
            metrics['variance'][f'L{ℓ}'].append(rep_metrics['variance'][ℓ - 1])

        # 5. 重建准确率 (LM head 解码 z_L → token 预测)
        h_norm = pc_model.model.norm(z_by_layer[pc_model.num_sub_layers])
        logits = pc_model.model.lm_head(h_norm)
        pred_tokens = logits.argmax(dim=-1)

        # 只算有效 token (非 pad)
        valid = labels != -100
        if valid.any():
            acc = (pred_tokens[valid] == input_ids[valid]).float().mean().item()
        else:
            acc = 0.0
        metrics['recon_acc'].append(acc)

        if (batch_idx + 1) % 5 == 0:
            Logger(f'  Evaluated {batch_idx + 1}/{min(max_batches, len(loader))} batches')

    # ── 汇总 ──
    summary = {
        'F_mean': sum(metrics['F']) / len(metrics['F']),
        'F_std': torch.tensor(metrics['F']).std().item(),
        'recon_acc_mean': sum(metrics['recon_acc']) / len(metrics['recon_acc']),
        'recon_acc_std': torch.tensor(metrics['recon_acc']).std().item(),
        'per_layer': {},
    }
    for ℓ in range(1, pc_model.num_sub_layers + 1):
        label = f'L{ℓ}'
        sp = metrics['sparsity'][label]
        sm = metrics['smoothness'][label]
        vr = metrics['variance'][label]
        summary['per_layer'][label] = {
            'sparsity': sum(sp) / len(sp),
            'smoothness': sum(sm) / len(sm),
            'variance': sum(vr) / len(vr),
        }

    return summary


def main():
    lm_config = MiniMindConfig(hidden_size=256, num_hidden_layers=4, use_moe=False)
    ROOT = os.path.dirname(os.path.abspath(__file__))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT, 'model'))

    # ── 加载模型 ──
    pc_model = PCDynamicMiniMind(lm_config).to(device)
    ckpt_path = os.path.join(ROOT, 'out_pc_local', 'pcl_final.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        pc_model.load_state_dict(ckpt['model_state'])
        Logger(f'Loaded checkpoint: {ckpt_path}')
    else:
        Logger(f'No checkpoint found at {ckpt_path}, using untrained model')

    # ── 数据 ──
    data_path = os.path.join(ROOT, 'dataset', 'pretrain_t2t_mini.jsonl')
    ds = _LocalDataset(data_path, tokenizer, max_length=128, max_samples=1000)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

    # ── RoPE (max 512) ──
    freqs_cos, freqs_sin = pc_model.freqs_cos, pc_model.freqs_sin
    pos_emb = (freqs_cos.to(device), freqs_sin.to(device))

    # ── 评估 ──
    Logger('Running 5-dim self-supervised evaluation...')
    summary = evaluate(pc_model, loader, pos_emb, gamma=0.1, T=2, max_batches=20)

    Logger('─' * 50)
    Logger('Spatiotemporal PC Evaluation Results')
    Logger('─' * 50)
    Logger(f'F_total:        {summary["F_mean"]:.6f} ± {summary["F_std"]:.6f}')
    Logger(f'Recon Accuracy: {summary["recon_acc_mean"]:.4f} ± {summary["recon_acc_std"]:.4f}')
    Logger('')
    Logger('Per-Layer Metrics:')
    for label, m in summary['per_layer'].items():
        Logger(f'  {label}: sparsity={m["sparsity"]:.4f}, '
               f'smooth={m["smoothness"]:.4f}, var={m["variance"]:.4f}')
    Logger('─' * 50)

    return summary


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
