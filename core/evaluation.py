"""
统一评估模块 — Perplexity + 自监督指标 + 文本生成。

从 eval_pc_language.py 和 eval_all.py 合并。

用法:
    from core.evaluation import (
        compute_perplexity, generate_text,
        eval_self_supervised, eval_ppl,
        load_with_remap,
    )
"""
import os, sys, math, json, warnings
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from model.pc_layers import PCDynamicMiniMind, PCLocalDynamicMiniMind, load_pc_checkpoint
from model.pc_backbone_local import PCLocalBackbone
from model.model_minimind import MiniMindConfig
from core.dataset import DualChannelDataset
from core.trainer_utils import Logger


def _resolve_pos(pos_emb, seq_len, device):
    """处理 None pos_emb (Conv backbone 无位置编码)。"""
    if pos_emb is None or pos_emb[0] is None:
        return (None, None)
    return (pos_emb[0][:seq_len].to(device), pos_emb[1][:seq_len].to(device))


def load_with_remap(model, ckpt_path, device='cpu'):
    """
    加载检查点并处理 key 重映射。
    旧版 checkpoint 使用 model.model.xxx 键名, 当前代码为 model.xxx。
    自动处理 byte_proj Conv1d 输入通道数变化 (1→2)。
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt['model_state']
    remapped = {}
    for k, v in sd.items():
        if k.startswith('model.model.'):
            remapped['model.' + k[len('model.model.'):]] = v
        else:
            remapped[k] = v

    # 处理 byte_proj 输入通道数变化: 旧=1, 新=2
    model_sd = model.state_dict()
    for key in list(remapped.keys()):
        if key.endswith('byte_proj.weight') and key in model_sd:
            expected = model_sd[key].shape
            actual = remapped[key].shape
            if actual != expected and actual[1] == 1 and expected[1] == 2:
                # 将单通道权重复制到双通道并减半振幅
                remapped[key] = remapped[key].expand(-1, 2, -1) / 2

    model.load_state_dict(remapped, strict=False)
    return model


# ═══════════════════════════════════════════════════════════════
# 1. Perplexity
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_perplexity(pc_model, loader, pos_emb, gamma=0.1, T=2, max_batches=20):
    """
    计算 Perplexity = exp(mean CE loss)。

    Args:
        pc_model: PC 模型
        loader: DataLoader (返回 byte_tensor, labels)
        pos_emb: 位置编码 (cos, sin) 或 (None, None)
        gamma: PC 推理 gamma
        T: PC 推理步数 (0 = 无 PC 推理)
        max_batches: 最大 batch 数

    Returns:
        (ppl, avg_loss)
    """
    pc_model.eval()
    total_loss, total_tokens = 0.0, 0

    for batch_idx, (input_ids, labels) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        input_ids = input_ids.to(next(pc_model.parameters()).device)
        labels = labels.to(input_ids.device)
        bsz, seq_len = input_ids.shape
        pos = _resolve_pos(pos_emb, seq_len, input_ids.device)

        # PC 推理
        z_by_layer = pc_model.init_z(input_ids)
        if T > 0:
            z_by_layer, _, _, _ = pc_model.spatiotemporal_infer(z_by_layer, pos, gamma=gamma, T=T)

        # CE loss
        h_norm = pc_model.model.norm(z_by_layer[pc_model.num_sub_layers])
        logits = pc_model.model.lm_head(h_norm.to(dtype=pc_model.model.lm_head.weight.dtype))
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='sum',
        )
        n_tokens = (shift_labels != -100).sum().item()
        total_loss += loss.item()
        total_tokens += n_tokens

        if (batch_idx + 1) % 5 == 0:
            Logger(f'  Batch {batch_idx + 1}/{min(max_batches, len(loader))}')

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    return ppl, avg_loss


# ═══════════════════════════════════════════════════════════════
# 2. 文本生成
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_text(pc_model, prompt, max_new_tokens=50,
                  gamma=0.1, temperature=0.8, top_k=20):
    """
    字节级自回归文本生成。

    使用模型内置的 generate_with_pc 方法 (纯前向, 无 PC 推理)。

    Args:
        pc_model: PC 模型 (需支持 generate_with_pc)
        prompt: 字符串 prompt
        max_new_tokens: 最大生成 token 数
        gamma: PC 推理 gamma
        temperature: 采样温度
        top_k: Top-K 采样

    Returns:
        生成文本字符串
    """
    pc_model.eval()
    device = next(pc_model.parameters()).device
    byte_seq = list(prompt.encode('utf-8'))
    byte_tensor = torch.tensor([byte_seq], device=device, dtype=torch.long)
    out = pc_model.generate_with_pc(
        byte_tensor, max_new_tokens=max_new_tokens,
        T_infer=0, gamma=gamma, temperature=temperature, top_k=top_k,
        eos_token_id=0x02,
    )
    return bytes(out[0].tolist()).decode('utf-8', errors='replace')


@torch.no_grad()
def generate_text_manual(pc_model, prompt, pos_emb, max_new=50,
                         gamma=0.1, T=2, temp=0.8, top_k=20):
    """
    手动字节级自回归生成 (支持 PC 推理)。

    与 generate_text 不同, 此函数在每步生成时执行 PC 推理。

    Args:
        pc_model: PC 模型
        prompt: 字符串 prompt
        pos_emb: 位置编码
        max_new: 最大新 token 数
        gamma: PC 推理 gamma
        T: PC 推理步数
        temp: 采样温度
        top_k: Top-K 采样

    Returns:
        生成文本字符串
    """
    pc_model.eval()
    device = next(pc_model.parameters()).device
    byte_seq = list(prompt.encode('utf-8'))

    for _ in range(max_new):
        ids = torch.tensor([byte_seq], device=device, dtype=torch.uint8)
        pos = _resolve_pos(pos_emb, min(len(byte_seq), 128), device)

        z = pc_model.init_z(ids)
        if T > 0:
            z, _, _ = pc_model.spatiotemporal_infer(z, pos, gamma=gamma, T=T)

        h_norm = pc_model.model.norm(z[pc_model.num_sub_layers])
        logits = pc_model.model.lm_head(h_norm.to(dtype=pc_model.model.lm_head.weight.dtype))
        next_logits = logits[0, -1, :] / temp

        if top_k > 0:
            tv, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < tv[-1]] = float('-inf')

        nid = torch.multinomial(torch.softmax(next_logits, -1), 1).item()
        byte_seq.append(nid)
        if nid == 0x02:  # EOS
            break

    return bytes(byte_seq).decode('utf-8', errors='replace')


# ═══════════════════════════════════════════════════════════════
# 3. 自监督评估 (5维)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_self_supervised(pc_model, loader, pos_emb, gamma=0.1, T=2, max_batches=20):
    """
    自监督评估: 5 维度指标。

    返回:
        dict {
            'F_mean': float,           # 平均自由能
            'recon_acc_mean': float,   # 平均重建准确率
            'per_layer': {
                'L1': { 'sparsity', 'smoothness', 'variance' },
                'L2': { ... },
                ...
            }
        }
    """
    pc_model.eval()
    metrics = {
        'F': [],
        'recon_acc': [],
        'sparsity': {f'L{i}': [] for i in range(1, pc_model.num_sub_layers + 1)},
        'smoothness': {f'L{i}': [] for i in range(1, pc_model.num_sub_layers + 1)},
        'variance': {f'L{i}': [] for i in range(1, pc_model.num_sub_layers + 1)},
    }

    for batch_idx, (input_ids, labels) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        input_ids = input_ids.to(next(pc_model.parameters()).device)
        bsz, seq_len = input_ids.shape

        pos = _resolve_pos(pos_emb, seq_len, input_ids.device)

        z = pc_model.init_z(input_ids)
        z, _, F_hist = pc_model.spatiotemporal_infer(z, pos, gamma=gamma, T=T)

        metrics['F'].append(F_hist[-1] / (bsz * seq_len))

        rep = pc_model.compute_representation_metrics(z)
        for ℓ in range(1, pc_model.num_sub_layers + 1):
            metrics['sparsity'][f'L{ℓ}'].append(rep['sparsity'][ℓ - 1])
            metrics['smoothness'][f'L{ℓ}'].append(rep['temporal_smoothness'][ℓ - 1])
            metrics['variance'][f'L{ℓ}'].append(rep['variance'][ℓ - 1])

        h_norm = pc_model.model.norm(z[pc_model.num_sub_layers])
        logits = pc_model.model.lm_head(h_norm.to(dtype=pc_model.model.lm_head.weight.dtype))
        valid = labels != -100
        acc = (logits.argmax(-1)[valid] == input_ids[valid]).float().mean().item() if valid.any() else 0.0
        metrics['recon_acc'].append(acc)

        if (batch_idx + 1) % 5 == 0:
            Logger(f'  自监督 batch {batch_idx + 1}/{min(max_batches, len(loader))}')

    summary = {
        'F_mean': sum(metrics['F']) / len(metrics['F']),
        'recon_acc_mean': sum(metrics['recon_acc']) / len(metrics['recon_acc']),
        'per_layer': {},
    }
    for ℓ in range(1, pc_model.num_sub_layers + 1):
        lk = f'L{ℓ}'
        summary['per_layer'][lk] = {
            'sparsity': sum(metrics['sparsity'][lk]) / len(metrics['sparsity'][lk]),
            'smoothness': sum(metrics['smoothness'][lk]) / len(metrics['smoothness'][lk]),
            'variance': sum(metrics['variance'][lk]) / len(metrics['variance'][lk]),
        }
    return summary


# ═══════════════════════════════════════════════════════════════
# 4. PPL 评估 (完整版, 兼容 eval_all)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_ppl(pc_model, loader, pos_emb, gamma=0.1, T=2, max_batches=20):
    """
    PPL 评估 (与 eval_all 兼容)。

    Returns:
        (ppl, avg_loss)
    """
    pc_model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch_idx, (input_ids, labels) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        input_ids = input_ids.to(next(pc_model.parameters()).device)
        labels = labels.to(input_ids.device)
        bsz, seq_len = input_ids.shape
        pos = _resolve_pos(pos_emb, seq_len, input_ids.device)
        z = pc_model.init_z(input_ids)
        if T > 0:
            z, _, _ = pc_model.spatiotemporal_infer(z, pos, gamma=gamma, T=T)
        h_norm = pc_model.model.norm(z[pc_model.num_sub_layers])
        logits = pc_model.model.lm_head(h_norm.to(dtype=pc_model.model.lm_head.weight.dtype))
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), ignore_index=-100, reduction='sum',
        )
        total_loss += loss.item()
        total_tokens += (shift_labels != -100).sum().item()
        if (batch_idx + 1) % 5 == 0:
            Logger(f'  PPL batch {batch_idx + 1}')
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss), avg_loss


# ═══════════════════════════════════════════════════════════════
# 便利函数: 创建 DataLoader
# ═══════════════════════════════════════════════════════════════

def create_eval_loader(data_path: str, max_length=128, max_samples=500,
                       batch_size=8, shuffle=False):
    """创建评估用的 DataLoader。"""
    ds = DualChannelDataset(
        data_path,
        max_length=max_length,
        max_samples=max_samples,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ═══════════════════════════════════════════════════════════════
# 综合评估入口
# ═══════════════════════════════════════════════════════════════

def run_full_evaluation(
    models: dict,
    loader_ppl,
    loader_ss=None,
    gamma=0.1,
    T=2,
    max_batches=20,
    prompts=None,
):
    """
    运行完整评估 (自监督 + PPL + 生成)。

    Args:
        models: dict {name: (model, pos_emb)}
        loader_ppl: PPL 评估用 DataLoader (no shuffle)
        loader_ss: 自监督评估用 DataLoader (shuffle), 默认复用 loader_ppl
        gamma: PC 推理 gamma
        T: PC 推理步数
        max_batches: 最大 batch 数
        prompts: 生成测试用的 prompt 列表

    Returns:
        (results_ss, results_ppl, results_gen)
    """
    if loader_ss is None:
        loader_ss = loader_ppl
    if prompts is None:
        prompts = ['人工智能的未来在于', '小明今天去了公园，他看到', '深度学习是一种']

    results_ss = {}
    results_ppl = {}
    results_gen = {}

    for name, (model, pos) in models.items():
        Logger(f'\n=== {name} ===')

        # 自监督
        try:
            ss = eval_self_supervised(model, loader_ss, pos, gamma=gamma, T=T, max_batches=max_batches)
            results_ss[name] = ss
            Logger(f'  F_total: {ss["F_mean"]:.4f}  Recon Acc: {ss["recon_acc_mean"]:.4f}')
            for l, m in ss['per_layer'].items():
                Logger(f'    {l}: sparsity={m["sparsity"]:.4f} smooth={m["smoothness"]:.4f} var={m["variance"]:.4f}')
        except Exception as e:
            Logger(f'  [SS ERROR] {e}')

        # PPL
        try:
            ppl0, loss0 = eval_ppl(model, loader_ppl, pos, T=0, max_batches=max_batches)
            ppl2, loss2 = eval_ppl(model, loader_ppl, pos, T=T, max_batches=max_batches)
            results_ppl[name] = {'T0_PPL': ppl0, 'T0_CE': loss0, 'T2_PPL': ppl2, 'T2_CE': loss2}
            delta = (ppl2 - ppl0) / ppl0 * 100
            Logger(f'  T=0: PPL={ppl0:.2f}  T={T}: PPL={ppl2:.2f} ({"改善" if ppl2<ppl0 else "劣化"} {abs(delta):.1f}%)')
        except Exception as e:
            Logger(f'  [PPL ERROR] {e}')

        # 生成
        try:
            gen_samples = []
            for prompt in prompts:
                text = generate_text(model, prompt, max_new_tokens=40,
                                     temperature=0.8, top_k=20)
                gen_samples.append((prompt, text))
                Logger(f'  Prompt: {prompt}')
                Logger(f'    → {text}')
            results_gen[name] = gen_samples
        except Exception as e:
            Logger(f'  [GEN ERROR] {e}')

    # 汇总表
    Logger('\n' + '═' * 60)
    Logger('汇总对比')
    Logger('═' * 60)
    header = f'{"Model":25s} {"F↓":>10s} {"Recon↑":>8s} {"PPL(T=0)↓":>12s} {"PPL(T=2)↓":>12s}'
    Logger(header)
    Logger('─' * 70)
    for name in models:
        ss = results_ss.get(name, {})
        pp = results_ppl.get(name, {})
        F = f'{ss.get("F_mean", 0):.2f}' if ss else 'N/A'
        recon = f'{ss.get("recon_acc_mean", 0):.4f}' if ss else 'N/A'
        ppl0 = f'{pp.get("T0_PPL", 0):.2f}' if pp else 'N/A'
        ppl2 = f'{pp.get("T2_PPL", 0):.2f}' if pp else 'N/A'
        Logger(f'{name:25s} {F:>10s} {recon:>8s} {ppl0:>12s} {ppl2:>12s}')

    return results_ss, results_ppl, results_gen
