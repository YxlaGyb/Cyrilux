"""
语言能力对比: 预训练 MiniMind → PC 推理效果

三步:
  1. 加载 pretrain_mvp.pth → PCDynamicMiniMind (骨架 + LM head 预训练)
  2. T=0 (纯前向) vs T=2 (PC 推理) 对比 PPL
  3. 文本生成对比
"""
import os, sys, warnings, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from model.pc_layers import PCDynamicMiniMind
from model.model_minimind import MiniMindConfig
from train_pc_local import _LocalDataset
from transformers import AutoTokenizer

LOG = lambda msg: print(msg)

@torch.no_grad()
def compute_perplexity(pc_model, loader, pos_emb, gamma=0.1, T=2, max_batches=20):
    pc_model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch_idx, (input_ids, labels) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        input_ids = input_ids.to(next(pc_model.parameters()).device)
        labels = labels.to(input_ids.device)
        bsz, seq_len = input_ids.shape
        pos = (pos_emb[0][:seq_len].to(input_ids.device),
               pos_emb[1][:seq_len].to(input_ids.device))
        z_by_layer = pc_model.init_z(input_ids)
        if T > 0:
            z_by_layer, _, _ = pc_model.spatiotemporal_infer(z_by_layer, pos, gamma=gamma, T=T)
        h_norm = pc_model.model.norm(z_by_layer[pc_model.num_sub_layers])
        logits = pc_model.model.lm_head(h_norm)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), ignore_index=-100, reduction='sum',
        )
        total_loss += loss.item()
        total_tokens += (shift_labels != -100).sum().item()
        if (batch_idx + 1) % 5 == 0:
            LOG(f'  Batch {batch_idx + 1}/{min(max_batches, len(loader))}')
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss), avg_loss

@torch.no_grad()
def generate_text(pc_model, tokenizer, prompt, pos_emb,
                  max_new_tokens=50, gamma=0.1, T=2, temperature=0.8, top_k=20):
    """PC 引导的自回归生成。ponytail: 委托给 generate_with_pc。"""
    pc_model.eval()
    device = next(pc_model.parameters()).device
    tokens = [tokenizer.bos_token_id] + tokenizer(prompt, add_special_tokens=False).input_ids
    input_ids = torch.tensor([tokens], device=device)
    out = pc_model.generate_with_pc(
        input_ids, max_new_tokens=max_new_tokens,
        T_infer=T, gamma=gamma, temperature=temperature, top_k=top_k,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0].tolist(), skip_special_tokens=True)


def load_pretrained_backbone(pc_model, pretrain_path, device):
    """将 pretrain_mvp.pth 加载到 PCDynamicMiniMind 的骨架中。"""
    ckpt = torch.load(pretrain_path, map_location='cpu', weights_only=False)
    # PCBackbone key = model.xxx, pretrain_mvp key = model.xxx → 直接匹配
    missing, unexpected = [], []
    for pc_key, pc_param in pc_model.state_dict().items():
        # 跳过 temporal_proj / topdown_proj (预训练中没有)
        if pc_key.startswith('temporal_proj') or pc_key.startswith('topdown_proj'):
            continue
        # 跳过 freqs (register_buffer, 预训练中没有)
        if pc_key in ('freqs_cos', 'freqs_sin'):
            continue
        pt_key = pc_key  # key 格式已对齐, 无需 remap
        if pt_key in ckpt:
            pt_tensor = ckpt[pt_key].to(device)
            if pt_tensor.shape == pc_param.shape:
                pc_param.data.copy_(pt_tensor)
            else:
                missing.append(f'shape mismatch: {pc_key} {pc_param.shape} vs {pt_tensor.shape}')
        else:
            missing.append(f'{pc_key} -> {pt_key} not found')
    if missing:
        LOG(f'  Missing keys (expected for PC layers): {len(missing)}')
    LOG(f'  Loaded pretrain_mvp.pth → PCDynamicMiniMind backbone')
    return pc_model


def main():
    lm_config = MiniMindConfig(hidden_size=256, num_hidden_layers=4, use_moe=False)
    ROOT = os.path.dirname(os.path.abspath(__file__))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT, 'model'))

    # ── 加载预训练骨架 ──
    LOG('─' * 50)
    LOG('加载 pretrain_mvp.pth (预训练 MiniMind)')
    LOG('─' * 50)
    pc_model_pt = PCDynamicMiniMind(lm_config).to(device)
    pretrain_path = os.path.join(ROOT, 'out', 'pretrain_mvp.pth')
    load_pretrained_backbone(pc_model_pt, pretrain_path, device)
    pc_model_pt.eval()

    # ── 同样加载 PC 训练的模型 ──
    LOG('')
    LOG('─' * 50)
    LOG('加载 pcl_final.pt (PC 自组织训练)')
    LOG('─' * 50)
    pc_model_pc = PCDynamicMiniMind(lm_config).to(device)
    ckpt_path = os.path.join(ROOT, 'out_pc_local', 'pcl_final.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        pc_model_pc.load_state_dict(ckpt['model_state'])
        LOG(f'  Loaded: {ckpt_path}')
    pc_model_pc.eval()

    # ── 数据 ──
    data_path = os.path.join(ROOT, 'dataset', 'pretrain_t2t_mini.jsonl')
    ds = _LocalDataset(data_path, tokenizer, max_length=128, max_samples=500)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    # ── RoPE ──
    freqs_cos, freqs_sin = pc_model_pt.freqs_cos, pc_model_pt.freqs_sin
    pos_emb = (freqs_cos.to(device), freqs_sin.to(device))

    # ════════════════════════════════════════════════════════
    # Perplexity: 预训练骨架
    # ════════════════════════════════════════════════════════
    LOG('')
    LOG('─' * 50)
    LOG('Perplexity — 预训练骨架 + PC 推理效果')
    LOG('─' * 50)

    def run_ppl(label, model):
        LOG(f'\n{label}:')
        for t_val in (0, 2):
            ppl, loss = compute_perplexity(model, loader, pos_emb,
                                           gamma=0.1, T=t_val, max_batches=20)
            LOG(f'  T={t_val}  →  CE Loss: {loss:.4f},  PPL: {ppl:.2f}')
            if t_val == 2:
                base_ppl = ppl_0
            else:
                ppl_0 = ppl
                continue
            delta = (ppl - base_ppl) / base_ppl * 100
            LOG(f'           (PC 推理 {"改善" if ppl < base_ppl else "劣化"} {abs(delta):.1f}%)')

    run_ppl('🟢 预训练 MiniMind', pc_model_pt)
    run_ppl('🔵 PC 训练 (自组织)', pc_model_pc)

    # ════════════════════════════════════════════════════════
    # Text Generation (只用预训练模型)
    # ════════════════════════════════════════════════════════
    LOG('')
    LOG('─' * 50)
    LOG('文本生成 — 预训练 MiniMind (T=0 vs T=2)')
    LOG('─' * 50)

    prompts = [
        '人工智能的未来在于',
        '小明今天去了公园，他看到',
        '深度学习是一种',
    ]

    for prompt in prompts:
        LOG(f'\n  Prompt: {prompt}')
        for tag, model in [('T=0 (纯前向)', pc_model_pt),
                            ('T=2 (PC 推理)', pc_model_pt)]:
            t = 0 if 'T=0' in tag else 2
            text = generate_text(pc_model_pt, tokenizer, prompt, pos_emb,
                                 max_new_tokens=40, gamma=0.1, T=t,
                                 temperature=0.8, top_k=20)
            LOG(f'  [{tag}] {text}')

    LOG('')
    LOG('─' * 50)
    LOG('Done.')
    LOG('─' * 50)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
