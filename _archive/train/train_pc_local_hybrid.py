"""
F_pred + CE 混合训练 — 时空预测编码自组织 + 语言能力联合优化。

核心: 一个训练步同时优化两个互补目标:
  F_pred = Σ½·‖z_ℓ - μ_total‖²  预测误差 (自组织 + 群体编码)
  CE     = -Σ log p(token)        交叉熵 (语言能力)

梯度路径:
  CE → backbone 所有层 + LM head  (语言能力)
  F_pred → backbone + temporal_proj + topdown_proj  (自组织)

Ponytail: 从 train_pc_local.py 派生, beta warmup + 尺度对齐。
"""
import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model.pc_layers import PCLocalDynamicMiniMind
from model.model_minimind import MiniMindConfig
from trainer_utils import get_lr, Logger, setup_seed
from tqdm import tqdm
from transformers import AutoTokenizer


class _LocalDataset(Dataset):
    """混合训练数据集: 需要 label (CE 训练用)."""
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
    T_infer = 2
    gamma = 0.1

    # 混合参数
    max_beta = 2.0       # CE_local 权重上限 (warmup: 0.1 → max_beta)
    max_beta_conv = 1.0  # CE_converged 权重上限 (warmup: 0.0 → max_beta_conv)
    grad_clip = 1.0       # 梯度裁剪

    data_path = os.path.join(ROOT, 'dataset', 'pretrain_t2t_mini.jsonl')
    out_dir = os.path.join(ROOT, 'out_pc_local_hybrid')
    tokenizer_path = os.path.join(ROOT, 'model')

    # ── 模型 & tokenizer ──
    Logger(f'F_pred + CE Hybrid Training | T={T_infer}, gamma={gamma}, max_beta={max_beta}')
    Logger(f'Device: {device}')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    pc_model = PCLocalDynamicMiniMind(lm_config).to(device)
    Logger(f'Model params: {sum(p.numel() for p in pc_model.parameters() if p.requires_grad) / 1e6:.2f}M')

    # ── 数据 ──
    subset_size = 10000
    ds = _LocalDataset(data_path, tokenizer, max_length=max_seq_len, max_samples=subset_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    iters = len(loader)
    Logger(f'Data: {subset_size} samples, {iters} steps/epoch')

    # ── 优化器 (与 SpatiotemporalPCUpdater 相同的参数分组) ──
    optimizer = torch.optim.AdamW(
        list(pc_model.temporal_proj.parameters()) +
        list(pc_model.topdown_proj.parameters()) +
        [p for n, p in pc_model.model.named_parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.95),
    )

    os.makedirs(out_dir, exist_ok=True)

    # ── 训练循环 ──
    pc_model.train()
    global_step = 0
    total_steps = iters * epochs

    # EMA 参考 (防坍塌, 只作用于 PC 表示)
    ema_z = None
    ema_lambda = 0.001

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f'Epoch {epoch + 1}/{epochs} [Hybrid]',
                    unit='step', dynamic_ncols=True, ascii=True)

        for step, (input_ids, labels) in enumerate(pbar):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bsz, seq_len = input_ids.shape
            global_step += 1

            # ── Phase 1: 共享前向 (有梯度) ──
            pos_emb = pc_model.get_position_embeddings(seq_len, device)
            z_init, ce_loss = pc_model.forward_with_ce(input_ids, labels, pos_emb)

            # ── Phase 2: PC 推理 (z detach, 无梯度) ──
            z_detached = [z.detach() for z in z_init]
            z_converged, errors_hist, F_hist = pc_model.spatiotemporal_infer(
                z_detached, pos_emb, gamma=gamma, T=T_infer,
            )

            # ── Phase 3: F_pred ──
            F_pred = pc_model.compute_spatiotemporal_loss(z_converged, pos_emb)

            # ── Phase 3.5: CE 从 PC 收敛后的 z 计算 (新增!) ──
            ce_converged = pc_model.compute_ce_loss(z_converged, labels)

            # EMA 正则 (防表示塌塌)
            if ema_z is not None and ema_lambda > 0:
                reg = 0.0
                for ell in range(1, pc_model.num_sub_layers + 1):
                    reg = reg + ((z_converged[ell] - ema_z[ell]) ** 2).sum()
                F_pred = F_pred + 0.5 * ema_lambda * reg

            # ── Phase 4: 尺度对齐 + 合并 (三路) ──
            # CE_local: 从 z_init 计算, 快速 warmup — 学习局部 n-gram 映射
            beta_local = min(max_beta, 0.1 + global_step / total_steps * (max_beta - 0.1))
            # CE_converged: 从 PC 收敛后的 z 计算, 慢热 — 教 lm_head 读 PC 精炼表示
            beta_conv = min(max_beta_conv, 0.0 + global_step / total_steps * max_beta_conv)

            # ponytail: 尺度分别对齐, 防止任一目标主导
            ce_local_sum = ce_loss * (bsz * seq_len)
            ce_conv_sum = ce_converged * (bsz * seq_len)
            scale_local = (F_pred.detach() / (ce_local_sum.detach() + 1e-8)).clamp(0.1, 10.0)
            scale_conv = (F_pred.detach() / (ce_conv_sum.detach() + 1e-8)).clamp(0.1, 10.0)
            total_loss = F_pred + beta_local * scale_local * ce_local_sum \
                                  + beta_conv * scale_conv * ce_conv_sum

            # ── Phase 5: backward ──
            optimizer.zero_grad()
            total_loss.backward()

            # 梯度裁剪 (所有可训练参数)
            trainable_params = [p for p in pc_model.parameters() if p.requires_grad and p.grad is not None]
            if trainable_params:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

            # 余弦学习率
            current_lr = get_lr(global_step, total_steps, lr)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr

            optimizer.step()

            # ── 更新 EMA ──
            with torch.no_grad():
                if ema_z is None:
                    ema_z = [z.detach().clone() for z in z_converged]
                else:
                    alpha = 0.99
                    for ell in range(len(z_converged)):
                        ema_z[ell] = (alpha * ema_z[ell] +
                                       (1 - alpha) * z_converged[ell].detach())

            # ── 表示质量评估 ──
            with torch.no_grad():
                metrics = pc_model.compute_representation_metrics(z_converged)

            # ── 日志 ──
            ce_val = ce_loss.item()
            ce_conv_val = ce_converged.item()
            F_val = F_pred.item()
            F_final = F_hist[-1] if F_hist else 0.0
            last_errors = errors_hist[-1] if errors_hist else []
            avg_error = sum(e[0] for e in last_errors) / len(last_errors) if last_errors else 0
            sparsity = metrics['sparsity'][-1] if metrics['sparsity'] else 0
            smooth = metrics['temporal_smoothness'][-1] if metrics['temporal_smoothness'] else 0
            var = metrics['variance'][-1] if metrics['variance'] else 0

            scale_local_val = scale_local.item() if hasattr(scale_local, 'item') else scale_local
            scale_conv_val = scale_conv.item() if hasattr(scale_conv, 'item') else scale_conv

            pbar.set_postfix(
                F=f'{F_final:.2f}',
                CE=f'{ce_val:.4f}',
                CEv=f'{ce_conv_val:.4f}',
                bL=f'{beta_local:.3f}',
                bC=f'{beta_conv:.3f}',
                sp=f'{sparsity:.3f}',
                sm=f'{smooth:.3f}',
            )

            # ── 详细日志 (每 100 步) ──
            if (step + 1) % 100 == 0:
                Logger(
                    f'[Step {step + 1}/{iters}] F={F_final:.2f} '
                    f'CE_L={ce_val:.4f} CE_C={ce_conv_val:.4f} '
                    f'bL={beta_local:.3f} bC={beta_conv:.3f} '
                    f'sL={scale_local_val:.2f} sC={scale_conv_val:.2f} '
                    f'lr={current_lr:.2e} | '
                    f'sparsity={sparsity:.3f} smooth={smooth:.3f} var={var:.4f}'
                )
                err_str = ' | '.join([f'L{ell+1}:{e[0]:.4f}'
                                      for ell, e in enumerate(last_errors)])
                Logger(f'  Layer errors: {err_str}')

            # ── Checkpoint (每 500 步) ──
            if (step + 1) % 500 == 0 or global_step == 1:
                ckpt = {
                    'epoch': epoch,
                    'step': step,
                    'model_state': pc_model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'F': F_final,
                    'CE_local': ce_val,
                    'CE_converged': ce_conv_val,
                    'beta_local': beta_local,
                    'beta_conv': beta_conv,
                    'lm_config': lm_config,
                }
                torch.save(ckpt, os.path.join(out_dir, f'hybrid_ckpt_s{step}.pt'))
                Logger(f'Checkpoint saved at step {step}')

    # ── 最终保存 ──
    ckpt_path = os.path.join(out_dir, 'hybrid_final.pt')
    torch.save({
        'epoch': epochs - 1,
        'step': iters - 1,
        'model_state': pc_model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'F': F_final,
        'CE_local': ce_val,
        'CE_converged': ce_conv_val,
        'beta_local': beta_local,
        'beta_conv': beta_conv,
        'lm_config': lm_config,
    }, ckpt_path)
    Logger(f'Final checkpoint saved to {ckpt_path}')
    Logger('Hybrid training complete.')


if __name__ == '__main__':
    setup_seed(42)
    train()
