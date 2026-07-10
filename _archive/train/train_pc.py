"""
PC 预训练 — 预测编码 + 多巴胺替代反向传播。
Ponytail: 最小验证，T=1 走通, T=4 出效果。
"""
# -*- coding: utf-8 -*-
import os, sys, json, warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from model.pc_layers import PCMiniMind
from model.model_minimind import MiniMindConfig
from trainer.pc_infer import pc_infer_with_tracking
from trainer.updaters import PCUpdater
from trainer_utils import get_lr, Logger, setup_seed
from tqdm import tqdm
from transformers import AutoTokenizer


# ── 纯 json 数据集 (零 datasets 依赖) ───────────────────────
class _PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, max_samples=None):
        super().__init__()
        self.tokenizer = tokenizer
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
        tokens = self.tokenizer(
            str(sample['text']),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


warnings.filterwarnings('ignore')


def train():
    # ── 配置 ──
    lm_config = MiniMindConfig(hidden_size=256, num_hidden_layers=4, use_moe=False)

    ROOT = os.path.dirname(os.path.abspath(__file__))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16
    batch_size = 8
    accum_steps = 4
    max_seq_len = 128
    epochs = 1
    lr = 5e-4
    # PC 参数
    T_infer = 2       # ponytail: T=2 速度/效果平衡, T=4 更精细但 2x 慢
    gamma = 0.1       # 推理步长
    η_dopamine = 1.0  # 多巴胺强度
    β_dopamine = 0.5  # 多巴胺学习率调制

    data_path = os.path.join(ROOT, 'dataset', 'pretrain_t2t_mini.jsonl')
    out_dir = os.path.join(ROOT, 'out_pc')
    tokenizer_path = os.path.join(ROOT, 'model')

    # ── 模型 & tokenizer ──
    Logger(f'PC Training | T={T_infer}, γ={gamma}, η={η_dopamine}, β={β_dopamine}')
    Logger(f'Device: {device} | Loading tokenizer from {tokenizer_path}')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    pc_model = PCMiniMind(lm_config).to(device)
    Logger(f'PC model params: {sum(p.numel() for p in pc_model.parameters() if p.requires_grad) / 1e6:.2f}M')

    # ── 数据 ──
    Logger(f'Loading data from {data_path}')
    subset_size = 50000
    ds = _PretrainDataset(data_path, tokenizer, max_length=max_seq_len, max_samples=subset_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    iters = len(loader)
    Logger(f'Subset: {subset_size} samples, {iters} steps/epoch')

    # ── 优化器 & 更新器 ──
    optimizer = optim.AdamW(pc_model.parameters(), lr=lr, betas=(0.9, 0.95))
    updater = PCUpdater(pc_model, optimizer, base_lr=lr, η=η_dopamine, β=β_dopamine)

    os.makedirs(out_dir, exist_ok=True)

    # ── 训练循环 ──
    pc_model.train()
    global_step = 0

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f'Epoch {epoch + 1}/{epochs} [PC]', unit='step', dynamic_ncols=True)

        for step, (input_ids, labels) in enumerate(pbar):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bsz, seq_len = input_ids.shape
            global_step += 1

            # ── Phase 1: PC 推理 (T 步自由能最小化) ──
            pos_emb = pc_model.get_position_embeddings(seq_len, device)
            z, layer_errors, track = pc_infer_with_tracking(
                pc_model, input_ids, labels, pos_emb,
                gamma=gamma, T=T_infer,
            )

            # ── Phase 2: PC 权重更新 (梯度累积) ──
            F_this = updater.backward(z, pos_emb, labels, input_ids=input_ids, div_factor=accum_steps)

            info = {}
            if (step + 1) % accum_steps == 0 or (step + 1) == iters:
                info = updater.optimizer_step()

            # ── 日志 ──
            ce_final = track['ce'][-1]
            F_final = track['F'][-1]
            last_errors = layer_errors[-1]
            avg_error = sum(e[0] for e in last_errors) / len(last_errors) if last_errors else 0
            D_display = info.get('dopamine', 0.0)

            pbar.set_postfix(
                ce=f'{ce_final:.4f}',
                F=f'{F_final:.4f}',
                D=f'{D_display:.3f}',
                err=f'{avg_error:.4f}',
            )

            # ── 详细日志 (每 100 步) ──
            if (step + 1) % 100 == 0:
                Logger(
                    f'[Step {step + 1}/{iters}] CE={ce_final:.4f} | '
                    f'F={F_final:.4f} | D={D_display:.3f} | '
                    f'lr={info.get("lr", 0):.2e}'
                )
                err_str = ' | '.join([f'L{ℓ+1}:{e[0]:.4f}' for ℓ, e in enumerate(last_errors)])
                Logger(f'  Layer errors: {err_str}')

            # ── Checkpoint ──
            if (step + 1) % 500 == 0:
                ckpt = {
                    'epoch': epoch,
                    'step': step,
                    'model_state': pc_model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'loss': ce_final,
                    'F': F_final,
                    'lm_config': lm_config,
                }
                torch.save(ckpt, os.path.join(out_dir, f'pc_ckpt_s{step}.pt'))
                Logger(f'Checkpoint saved at step {step}')

    # ── 最终保存 ──
    ckpt_path = os.path.join(out_dir, 'pc_final.pt')
    torch.save({
        'epoch': epochs - 1,
        'step': iters - 1,
        'model_state': pc_model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': ce_final,
        'F': F_final,
        'lm_config': lm_config,
    }, ckpt_path)
    Logger(f'Final checkpoint saved to {ckpt_path}')
    Logger('PC training complete.')


if __name__ == '__main__':
    setup_seed(42)
    train()
