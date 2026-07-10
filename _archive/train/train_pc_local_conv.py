"""
CE-only 训练 — 局部 Conv 语言模型 (无 PC 推理)。

梯度路径: Conv1D → MLP → backbone → lm_head → CE

Ponytail: 标准 LM 训练, PCLocalBackbone 直接 forward, 无 PC 组件.
"""
import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model.pc_backbone_local import PCLocalBackbone
from model.model_minimind import MiniMindConfig
from trainer_utils import get_lr, Logger, setup_seed
from tqdm import tqdm
from transformers import AutoTokenizer


class _LocalDataset(Dataset):
    """标准语言建模数据集."""
    def __init__(self, data_path, tokenizer, max_length=128, max_samples=None):
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
    batch_size = 8
    max_seq_len = 128
    epochs = 1
    lr = 3e-4
    grad_clip = 1.0

    data_path = os.path.join(ROOT, 'dataset', 'pretrain_t2t_mini.jsonl')
    out_dir = os.path.join(ROOT, 'out_pc_local_conv')
    tokenizer_path = os.path.join(ROOT, 'model')

    Logger('CE-only Conv LM Training')
    Logger(f'Device: {device}')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = PCLocalBackbone(lm_config).to(device)
    Logger(f'Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M')

    # ── 数据 ──
    subset_size = 10000
    ds = _LocalDataset(data_path, tokenizer, max_length=max_seq_len, max_samples=subset_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    iters = len(loader)
    Logger(f'Data: {subset_size} samples, {iters} steps/epoch')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    os.makedirs(out_dir, exist_ok=True)

    # ── 训练循环 ──
    model.train()
    global_step = 0
    total_steps = iters * epochs

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f'Epoch {epoch + 1}/{epochs} [CE-only Conv]',
                    unit='step', dynamic_ncols=True)

        for step, (input_ids, labels) in enumerate(pbar):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            global_step += 1

            _, loss = model(input_ids, labels=labels)

            optimizer.zero_grad()
            loss.backward()

            trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            if trainable:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)

            current_lr = get_lr(global_step, total_steps, lr)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr

            optimizer.step()

            pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{current_lr:.2e}')

            if (step + 1) % 200 == 0:
                Logger(f'[Step {step + 1}/{iters}] loss={loss.item():.4f} lr={current_lr:.2e}')

            if (step + 1) % 500 == 0 or global_step == 1:
                ckpt = {
                    'epoch': epoch, 'step': step,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'loss': loss.item(),
                    'lm_config': lm_config,
                }
                torch.save(ckpt, os.path.join(out_dir, f'conv_ckpt_s{step}.pt'))
                Logger(f'Checkpoint saved at step {step}')

    ckpt_path = os.path.join(out_dir, 'conv_final.pt')
    torch.save({
        'epoch': epochs - 1, 'step': iters - 1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': loss.item(),
        'lm_config': lm_config,
    }, ckpt_path)
    Logger(f'Final checkpoint saved to {ckpt_path}')
    Logger('CE-only Conv training complete.')


if __name__ == '__main__':
    setup_seed(42)
    train()
