"""
语言能力验证 — Perplexity + 文本生成

衡量 PC 自组织训练对语言能力的影响:
  1. Perplexity (越低越好): 对 held-out 数据的 next-token 预测能力
  2. 文本生成: 给定 prompt, 逐 token 采样, 看语义连贯性

Ponytail: 只做最小验证, 20 batch PPL  + 3 条生成样本即可。
"""
import os, sys, json, warnings, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import Dataset, DataLoader
from model.pc_layers import PCDynamicMiniMind, PCLocalDynamicMiniMind
from model.model_minimind import MiniMindConfig
from trainer_utils import Logger


class _LocalDataset(Dataset):
    """原始 UTF-8 字节数据集 — 无 tokenizer, 无离散边界."""
    def __init__(self, data_path, max_length=128, max_samples=None):
        super().__init__()
        self.max_length = max_length
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        byte_seq = str(sample['text']).encode('utf-8')[:self.max_length]
        padded = byte_seq.ljust(self.max_length, b'\x00')
        byte_tensor = torch.frombuffer(bytearray(padded), dtype=torch.uint8).clone()
        labels = byte_tensor.clone()
        labels[byte_tensor == 0x00] = -100
        return byte_tensor, labels


@torch.no_grad()
def compute_perplexity(pc_model, loader, pos_emb, gamma=0.1, T=2, max_batches=20):
    """计算 Perplexity = exp(mean CE loss)。"""
    pc_model.eval()
    total_loss, total_tokens = 0.0, 0

    for batch_idx, (input_ids, labels) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        input_ids = input_ids.to(next(pc_model.parameters()).device)
        labels = labels.to(input_ids.device)
        bsz, seq_len = input_ids.shape
        # ponytail: pos_emb may be (None, None) for Conv backbone
        if pos_emb[0] is not None:
            pos = (pos_emb[0][:seq_len].to(input_ids.device),
                   pos_emb[1][:seq_len].to(input_ids.device))
        else:
            pos = (None, None)

        # PC 推理: 精炼表示 (T=0 时直接返回 init_z 结果)
        z_by_layer = pc_model.init_z(input_ids)
        if T > 0:
            z_by_layer, _, _, _ = pc_model.spatiotemporal_infer(z_by_layer, pos, gamma=gamma, T=T)

        # CE loss (原始实现, 保持 reduction='sum' 以精确计算 PPL)
        h_norm = pc_model.model.norm(z_by_layer[pc_model.num_sub_layers])
        logits = pc_model.model.lm_head(h_norm)
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


@torch.no_grad()
def generate_text(pc_model, prompt, max_new_tokens=50,
                  gamma=0.1, temperature=0.8, top_k=20):
    """字节级自回归生成 (纯前向, 无 PC 推理)."""
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint path (default: out_pc_local/pcl_final.pt)')
    parser.add_argument('--local', action='store_true',
                        help='Use PCLocalDynamicMiniMind (Conv backbone)')
    args = parser.parse_args()

    lm_config = MiniMindConfig(hidden_size=256, num_hidden_layers=4, use_moe=False)
    ROOT = os.path.dirname(os.path.abspath(__file__))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # ── 加载模型 ──
    ModelClass = PCLocalDynamicMiniMind if args.local else PCDynamicMiniMind
    pc_model = ModelClass(lm_config).to(device)
    ckpt_path = args.ckpt or os.path.join(ROOT, 'out_pc_local', 'pcl_final.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        pc_model.load_state_dict(ckpt['model_state'], strict=False)
        Logger(f'Loaded trained checkpoint: {ckpt_path}')
    else:
        Logger(f'No checkpoint found, using UNTRAINED model (baseline)')

    pc_model.eval()

    # ── 数据 (用于 PPL) ──
    data_path = os.path.join(ROOT, 'dataset', 'pretrain_t2t_mini.jsonl')
    ds = _LocalDataset(data_path, max_length=128, max_samples=500)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    # ── 位置编码 ──
    if args.local:
        pos_emb = pc_model.get_position_embeddings(128, device)  # (None, None)
    else:
        freqs_cos, freqs_sin = pc_model.freqs_cos, pc_model.freqs_sin
        pos_emb = (freqs_cos.to(device), freqs_sin.to(device))

    # ════════════════════════════════════════════════════════
    # 1. Perplexity
    # ════════════════════════════════════════════════════════
    Logger('─' * 50)
    Logger('1. Perplexity Evaluation')
    Logger('─' * 50)
    # ── T=2 (PC 推理后) ──
    Logger('T=2 (with PC spatiotemporal inference):')
    ppl, avg_loss = compute_perplexity(pc_model, loader, pos_emb, gamma=0.1, T=2, max_batches=20)
    Logger(f'  CE Loss: {avg_loss:.4f}, PPL: {ppl:.4f}')

    # ── T=0 (纯前向, 无 PC 推理) ──
    Logger('T=0 (forward pass only, no PC inference):')
    ppl0, avg_loss0 = compute_perplexity(pc_model, loader, pos_emb, gamma=0.1, T=0, max_batches=20)
    Logger(f'  CE Loss: {avg_loss0:.4f}, PPL: {ppl0:.4f}')
    Logger('')

    # ════════════════════════════════════════════════════════
    # 2. Text Generation
    # ════════════════════════════════════════════════════════
    Logger('─' * 50)
    Logger('2. Text Generation Samples')
    Logger('─' * 50)

    prompts = [
        '人工智能的未来在于',
        '小明今天去了公园，他看到',
        '深度学习是一种',
    ]

    for prompt in prompts:
        text = generate_text(pc_model, prompt,
                             max_new_tokens=50, temperature=0.8, top_k=20)
        Logger(f'\n  Prompt: {prompt}')
        Logger(f'  Output: {text}')
        Logger('')

    Logger('─' * 50)
    Logger('Language evaluation complete.')
    Logger('─' * 50)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
