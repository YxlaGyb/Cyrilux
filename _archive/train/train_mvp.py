"""
MVP pretrain — 极小模型 + 数据子集 + 进度条。
Ponytail: 够用就行，跑通看效果。零依赖 minimind 项目。
"""
import os, sys, warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer_utils import get_lr, Logger, setup_seed
from tqdm import tqdm
from transformers import AutoTokenizer

# torchao Int4 Weight-Only QAT — weight-only 无 activation 模拟 4bit 训练
from torchao.quantization.qat import Int4WeightOnlyQATQuantizer


# 纯 json 加载 PretrainDataset，零 datasets 依赖
class _PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path, 'r', encoding='utf-8') as f:
            self.samples = [json.loads(line) for line in f]

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
    # ── 极小模型配置 ──
    lm_config = MiniMindConfig(hidden_size=256, num_hidden_layers=4, use_moe=False)

    ROOT = os.path.dirname(os.path.abspath(__file__))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16
    batch_size = 8
    accum_steps = 4  # effective batch = 32
    max_seq_len = 128
    epochs = 1
    lr = 5e-4
    save_interval = 500
    data_path = os.path.join(ROOT, 'dataset', 'pretrain_t2t_mini.jsonl')
    out_dir = os.path.join(ROOT, 'out')
    tokenizer_path = os.path.join(ROOT, 'model')

    # ── 模型 & tokenizer ──
    Logger(f'Loading tokenizer from {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 先创建在 CPU → QAT prepare → 再 .to(device)
    model = MiniMindForCausalLM(lm_config)
    Logger(f'Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M')

    quantizer = Int4WeightOnlyQATQuantizer(
        groupsize=64,
        inner_k_tiles=4,
        precision=torch.float16,
        scales_precision=torch.bfloat16,
    )
    model = quantizer.prepare(model)
    Logger('Prepared Int4WeightOnly QAT (groupsize=64, inner_k_tiles=4)')
    model = model.to(device)

    # ── 数据子集 (前50K) ──
    Logger(f'Loading data from {data_path}')
    full_ds = _PretrainDataset(data_path, tokenizer, max_length=max_seq_len)
    subset_size = min(50000, len(full_ds))
    ds = Subset(full_ds, range(subset_size))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    iters = len(loader)
    Logger(f'Subset: {subset_size} samples, {iters} steps/epoch')

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='step', dynamic_ncols=True)
        for step, (input_ids, labels) in enumerate(pbar):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            lr_val = get_lr(epoch * iters + step, epochs * iters, lr)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_val

            with autocast_ctx:
                res = model(input_ids, labels=labels)
                loss = (res.loss + res.aux_loss) / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            loss_item = loss.item() * accum_steps
            pbar.set_postfix(loss=f'{loss_item:.4f}', lr=f'{lr_val:.2e}')

            if (step + 1) % save_interval == 0:
                ckpt = {
                    'epoch': epoch,
                    'step': step,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'loss': loss_item,
                    'lm_config': lm_config,
                }
                torch.save(ckpt, os.path.join(out_dir, f'qat_ckpt_s{step}.pt'))
                Logger(f'Saved checkpoint at step {step}')

    # 最终 checkpoint
    ckpt_path = os.path.join(out_dir, 'qat_final.pt')
    torch.save({
        'epoch': epochs - 1,
        'step': iters - 1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': loss_item,
        'lm_config': lm_config,
    }, ckpt_path)
    Logger(f'Final checkpoint saved to {ckpt_path}')

    # 转换为纯 int4 推理格式
    Logger('Converting to int4 inference format...')
    model = quantizer.convert(model)
    torch.save(model.state_dict(), os.path.join(out_dir, 'int4_model.pt'))
    Logger('Int4 model saved to out/int4_model.pt')


if __name__ == '__main__':
    setup_seed(42)
    train()
