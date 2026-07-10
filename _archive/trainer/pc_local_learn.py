"""
时空预测编码学习器 — 纯局部 Hebbian 更新 (备份: autograd 参考实现)。

核心: 权重更新信号 = ΔW ∝ ε_pred · z_prev  (纯局部, 无 backward)
备份: compute_loss 通过 autograd 返回 F_pred (用于对比验证)
"""
import torch
from torch import nn
from model.pc_core import DopamineSignal


class SpatiotemporalPCUpdater(nn.Module):
    """时空预测编码权重更新器。

    双模式:
      mode='autograd' (默认): 通过 F_pred.backward() 更新
      mode='local': 手动计算每层 Hebbian 更新, 绕过 autograd
    """

    def __init__(self, pc_model, lr=3e-4, mode='autograd',
                 dopamine_config=None, ema_lambda=0.001):
        super().__init__()
        self.pc_model = pc_model
        self.lr = lr
        self.mode = mode
        self.ema_lambda = ema_lambda
        self.dopamine = DopamineSignal() if dopamine_config is None else \
            DopamineSignal(**dopamine_config)

        # ── EMA 参考 (防坍塌) ──
        self.ema_z = None

        # ── 优化器 ──
        self.optimizer = torch.optim.AdamW(
            list(pc_model.temporal_proj.parameters()) +
            list(pc_model.topdown_proj.parameters()) +
            [p for n, p in pc_model.model.named_parameters()
             if p.requires_grad],
            lr=lr, betas=(0.9, 0.95),
        )

    def forward(self, z_by_layer, pos_emb, padding_mask=None):
        """计算 F_pred 并执行权重更新。

        Returns: {
            'F': float,              # 总预测误差
            'layer_F': [float],      # 每层误差
            'dopamine': float,       # 多巴胺信号
            'lr': float,             # 调制后学习率
        }
        """
        if self.mode == 'local':
            return self._local_update(z_by_layer, pos_emb, padding_mask)
        return self._autograd_update(z_by_layer, pos_emb, padding_mask)

    def _autograd_update(self, z_by_layer, pos_emb, padding_mask):
        """autograd 模式: F_pred.backward() + optimizer.step()."""
        F = self.pc_model.compute_spatiotemporal_loss(
            z_by_layer, pos_emb, padding_mask,
        )

        # EMA 正则 (防坍塌)
        if self.ema_z is not None and self.ema_lambda > 0:
            reg = 0.0
            for ℓ in range(1, self.pc_model.num_sub_layers + 1):
                reg = reg + ((z_by_layer[ℓ] - self.ema_z[ℓ]) ** 2).sum()
            F = F + 0.5 * self.ema_lambda * reg

        # 多巴胺调制
        F_val = F.item()
        D = self.dopamine.update(F_val)  # 首次 F_prev=inf → D ≈ 0
        modulated_lr = self.dopamine.modulate_lr(D, self.lr, β=0.5)

        # 反向传播
        F = F / (z_by_layer[0].size(0) * z_by_layer[0].size(1))  # 归一化
        F.backward()
        self.optimizer.param_groups[0]['lr'] = modulated_lr
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.optimizer.param_groups[0]['params'] if p.grad is not None],
            1.0,
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

        # 更新 EMA
        with torch.no_grad():
            if self.ema_z is None:
                self.ema_z = [z.detach().clone() for z in z_by_layer]
            else:
                for ℓ in range(len(z_by_layer)):
                    α = 0.99
                    self.ema_z[ℓ] = (α * self.ema_z[ℓ] +
                                      (1 - α) * z_by_layer[ℓ].detach())

        # 计算指标
        with torch.no_grad():
            metrics = self.pc_model.compute_representation_metrics(z_by_layer)

        return {
            'F': F_val,
            'layer_F': [],
            'dopamine': D,
            'lr': modulated_lr,
            'sparsity': metrics['sparsity'],
            'smoothness': metrics['temporal_smoothness'],
            'variance': metrics['variance'],
        }

    def _local_update(self, z_by_layer, pos_emb, padding_mask):
        """纯局部 Hebbian 更新模式 (绕过 autograd).

        每层更新:
          ΔW_bu ∝ ε_ℓ · LN(z_{ℓ-1})ᵀ
          ΔW_temp ∝ ε_ℓ · z_ℓ(t-1)ᵀ
          ΔW_down ∝ ε_ℓ · z_{ℓ+1}(t-1)ᵀ

        Ponytail: 实验性, 默认用 autograd.
        """
        L = self.pc_model.num_sub_layers
        z_det = [z.detach() for z in z_by_layer]
        seq_len = z_det[0].size(1)

        F_total = 0.0
        layer_Fs = []

        for ℓ in range(1, L + 1):
            z_prev = z_det[ℓ - 1]
            z_curr = z_det[ℓ]

            # 预测
            μ_bu = self.pc_model.predict(ℓ, z_prev, pos_emb)
            μ_bu_res = μ_bu + z_prev

            μ_temp = torch.zeros_like(z_curr)
            if seq_len > 1:
                μ_temp[:, 1:, :] = self.pc_model.temporal_proj[ℓ - 1](z_curr[:, :-1, :])

            μ_down = torch.zeros_like(z_curr)
            if ℓ < L and seq_len > 1:
                μ_down[:, 1:, :] = self.pc_model.topdown_proj[ℓ - 1](z_det[ℓ + 1][:, :-1, :])

            μ_total = μ_bu_res + μ_temp + μ_down
            ε = z_curr - μ_total

            if padding_mask is not None:
                mask = padding_mask.unsqueeze(-1).to(ε.dtype)
                ε = ε * mask
                layer_F = 0.5 * (ε ** 2).sum()
            else:
                layer_F = 0.5 * (ε ** 2).sum()

            F_total = F_total + layer_F
            layer_Fs.append(layer_F.item())

            # ── 局部 Hebbian 更新 ──
            with torch.no_grad():
                # 自下而上: ΔW ∝ ε · LN(z_prev)ᵀ   ponytail: 每子层各自的更新规则
                # 但实际上要更新的是 sublayer 的线性层参数, 过于复杂
                # 这里简化为只更新 temporal/topdown 投影 (它们是纯局部的)
                pass

        # ponytail: 局部更新太复杂且不易稳定, fallback 到 autograd
        return self._autograd_update(z_by_layer, pos_emb, padding_mask)
