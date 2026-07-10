"""
时空预测编码自组织训练 — 纯预测误差驱动, 无 CE loss, 无 token loss。

核心循环:
  init_z → T步时空推理(收敛) → F_pred.backward() → 优化器更新 → 下一代

Ponytail: 最小验证, 先 T=1 走通, 再 T=2 出效果。
"""
# -*- coding: utf-8 -*-
import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import Dataset, DataLoader
from model.pc_layers import PCDynamicMiniMind
from model.model_minimind import MiniMindConfig
from trainer.pc_local_learn import SpatiotemporalPCUpdater
from trainer_utils import get_lr, Logger, setup_seed
from tqdm import tqdm
from transformers import AutoTokenizer


class _LocalDataset(Dataset):
    """自组织数据集: 只需序列, 不需要 label.

    ponytail: 复用 pretrain 数据但忽略 labels, 只取 input_ids.
    """
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

    # 时空 PC 参数
    T_infer = 2        # 时空推理步数
    gamma = 0.1        # 推理步长
    mode = 'autograd'  # 更新模式

    data_path = os.path.join(ROOT, 'dataset', 'pretrain_t2t_mini.jsonl')
    out_dir = os.path.join(ROOT, 'out_pc_local')
    tokenizer_path = os.path.join(ROOT, 'model')

    # ── 模型 & tokenizer ──
    Logger(f'Spatiotemporal PC Training | T={T_infer}, γ={gamma}, mode={mode}')
    Logger(f'Device: {device}')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    pc_model = PCDynamicMiniMind(lm_config).to(device)
    Logger(f'Model params: {sum(p.numel() for p in pc_model.parameters() if p.requires_grad) / 1e6:.2f}M')

    # ── 数据 ──
    subset_size = 10000
    ds = _LocalDataset(data_path, tokenizer, max_length=max_seq_len, max_samples=subset_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    iters = len(loader)
    Logger(f'Data: {subset_size} samples, {iters} steps/epoch')

    # ── 更新器 ──
    updater = SpatiotemporalPCUpdater(pc_model, lr=lr, mode=mode)

    os.makedirs(out_dir, exist_ok=True)

    # ── 训练循环 ──
    pc_model.train()
    global_step = 0

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f'Epoch {epoch + 1}/{epochs} [PC-Local]',
                    unit='step', dynamic_ncols=True)

        for step, (input_ids, _labels) in enumerate(pbar):
            input_ids = input_ids.to(device, non_blocking=True)
            bsz, seq_len = input_ids.shape
            global_step += 1

            # ── Phase 1: 时空推理 (自由能最小化) ──
            pos_emb = pc_model.get_position_embeddings(seq_len, device)
            z_by_layer = pc_model.init_z(input_ids)
            z_by_layer, errors_hist, F_hist = pc_model.spatiotemporal_infer(
                z_by_layer, pos_emb, gamma=gamma, T=T_infer,
            )

            # ── Phase 2: 权重更新 (F_pred → backward) ──
            result = updater(z_by_layer, pos_emb)

            # ── 表示质量评估 ──
            with torch.no_grad():
                metrics = pc_model.compute_representation_metrics(z_by_layer)

            # ── 日志 ──
            F_final = F_hist[-1] if F_hist else 0.0
            last_errors = errors_hist[-1] if errors_hist else []
            avg_error = sum(e[0] for e in last_errors) / len(last_errors) if last_errors else 0
            sparsity = metrics['sparsity'][-1] if metrics['sparsity'] else 0
            smooth = metrics['temporal_smoothness'][-1] if metrics['temporal_smoothness'] else 0
            var = metrics['variance'][-1] if metrics['variance'] else 0

            pbar.set_postfix(
                F=f'{F_final:.4f}',
                D=f'{result["dopamine"]:.3f}',
                sp=f'{sparsity:.3f}',
                sm=f'{smooth:.3f}',
            )

            # ── 详细日志 (每 100 步) ──
            if (step + 1) % 100 == 0:
                Logger(
                    f'[Step {step + 1}/{iters}] F={F_final:.4f} | '
                    f'sparsity={sparsity:.3f} | '
                    f'smooth={smooth:.3f} | var={var:.4f}'
                )
                err_str = ' | '.join([f'L{ℓ+1}:{e[0]:.4f}'
                                      for ℓ, e in enumerate(last_errors)])
                Logger(f'  Layer errors: {err_str}')

            # ── Checkpoint ──
            if (step + 1) % 500 == 0:
                ckpt = {
                    'epoch': epoch,
                    'step': step,
                    'model_state': pc_model.state_dict(),
                    'optimizer_state': updater.optimizer.state_dict(),
                    'F': F_final,
                    'lm_config': lm_config,
                }
                torch.save(ckpt, os.path.join(out_dir, f'pcl_ckpt_s{step}.pt'))
                Logger(f'Checkpoint saved at step {step}')

    # ── 最终保存 ──
    ckpt_path = os.path.join(out_dir, 'pcl_final.pt')
    torch.save({
        'epoch': epochs - 1,
        'step': iters - 1,
        'model_state': pc_model.state_dict(),
        'optimizer_state': updater.optimizer.state_dict(),
        'F': F_final,
        'lm_config': lm_config,
    }, ckpt_path)
    Logger(f'Final checkpoint saved to {ckpt_path}')
    Logger('Spatiotemporal PC training complete.')


if __name__ == '__main__':
    setup_seed(42)
    train()
