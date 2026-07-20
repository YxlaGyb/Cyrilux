"""
内在好奇心模块 (Intrinsic Curiosity Module — ICM)

升级 LatentWorldModel 为联合预测器 + 反向模型，输出多维内在动机信号:

  - pred_loss:      前向预测误差 — "这个世界我能预测吗？"
  - inverse_loss:   反向模型误差 — "我能理解这个转变吗？"
  - uncertainty:    预测不确定性
  - information_gain:  信息增益 = 旧 uncertainty - 新 uncertainty (EMA)
  - prediction_error:  原始 MSE
  - feature_embedding: 反向模型的中间表示 (供 ConceptDiscovery 使用)

与现有机制打配合:
  - information_gain → ForgettingSniffer.update_surprise()
  - feature_embedding → ConceptDiscovery.observe() 的特征
  - prediction_error → 替代旧 world_surprise (向后兼容)
  - inverse_loss → AbstractionSniffer drift 检测补充维度
"""
from __future__ import annotations

import math
from typing import Tuple, Optional

import torch
from torch import nn


class IntrinsicCuriosityModule(nn.Module):
    """内在好奇心模块 — 联合前向预测 + 反向模型 + 信息增益。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        context_dim: int = 5,
        action_embed_dim: int = 32,
        inv_loss_weight: float = 0.2,
        info_gain_ema_decay: float = 0.99,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.action_embed_dim = action_embed_dim
        self.inv_loss_weight = inv_loss_weight
        self.info_gain_ema_decay = info_gain_ema_decay

        # ── 前向预测器 (复用 LatentWorldModel 结构) ──
        in_dim = input_dim + context_dim
        self.state_proj = nn.Linear(in_dim, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pred_head = nn.Linear(hidden_dim, input_dim)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

        # ── 反向模型: concat(z_t, z_{t+1}) → action_embed ──
        self.inverse_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_embed_dim),
        )

        # ── 对比学习投影头 (区分转变类型) ──
        self.contrast_proj = nn.Sequential(
            nn.Linear(action_embed_dim, action_embed_dim),
            nn.Tanh(),
        )

        # ── 信息增益 EMA 状态 ──
        self.register_buffer('_uncertainty_ema', torch.zeros(1))
        self.register_buffer('_step', torch.zeros(1, dtype=torch.long))

    # ── 前向预测 (同 LatentWorldModel.forward) ──

    def _prepare_state(self, state: torch.Tensor, context: torch.Tensor | None):
        state = state.float()
        if state.dim() == 2:
            state = state.unsqueeze(1)
        if state.dim() != 3:
            raise ValueError(f"state must have shape [B, S, D] or [B, D], got {tuple(state.shape)}")

        if context is None:
            context = torch.zeros(state.size(0), self.context_dim, device=state.device, dtype=torch.float32)
        else:
            context = context.to(device=state.device, dtype=torch.float32)
            if context.dim() == 1:
                context = context.unsqueeze(0)
            if context.dim() != 2:
                raise ValueError(f"context must have shape [B, C], got {tuple(context.shape)}")
            if context.size(0) != state.size(0):
                if context.size(0) == 1:
                    context = context.expand(state.size(0), -1)
                else:
                    raise ValueError(f"batch mismatch: state batch {state.size(0)}, context batch {context.size(0)}")

        pooled = state.mean(dim=1)
        return pooled, context

    def _reshape_pred(self, pred: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        if pred.dim() == 2 and len(target_shape) == 3:
            seq_len = target_shape[1]
            return pred.unsqueeze(1).expand(-1, seq_len, -1)
        return pred

    def forward_forward(self, state: torch.Tensor, context: torch.Tensor | None = None):
        """前向预测: state_t → pred_{t+1}, uncertainty."""
        pooled, context = self._prepare_state(state, context)
        x = torch.cat([pooled, context], dim=-1)
        h = torch.tanh(self.state_proj(x))
        h = torch.tanh(self.hidden_proj(h))
        pred = self.pred_head(h)
        uncertainty = torch.sigmoid(self.uncertainty_head(h))
        pred = self._reshape_pred(pred, state.shape)
        return pred, uncertainty

    def forward_inverse(self, state_t: torch.Tensor, state_t1: torch.Tensor):
        """反向模型: concat(z_t, z_{t+1}) → action_embed + 对比特征。"""
        z_t = state_t.float().mean(dim=1) if state_t.dim() == 3 else state_t.float()
        z_t1 = state_t1.float().mean(dim=1) if state_t1.dim() == 3 else state_t1.float()
        x = torch.cat([z_t, z_t1], dim=-1)
        action_embed = self.inverse_net(x)
        contrast_feat = self.contrast_proj(action_embed)
        return action_embed, contrast_feat

    @torch.no_grad()
    def _compute_inverse_labels(self, z_batch: list[torch.Tensor], seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """构造反向模型训练对: (z_t, z_{t+1}) 和 labels (同一序列内正对, 跨序列负对)。

        返回:
            pairs: [(B*S-1), input_dim*2]
            labels: [(B*S-1)] long, 每个 pair 所属的序列 ID
        """
        bsz = len(z_batch)
        all_pairs = []
        all_labels = []
        for b in range(bsz):
            z = z_batch[b]  # [seq, dim]
            for t in range(seq_len - 1):
                pair = torch.cat([z[t], z[t + 1]], dim=0)
                all_pairs.append(pair)
                all_labels.append(b)
        if not all_pairs:
            return torch.empty(0, self.input_dim * 2, device=z_batch[0].device), \
                   torch.empty(0, dtype=torch.long, device=z_batch[0].device)
        return torch.stack(all_pairs), torch.tensor(all_labels, dtype=torch.long, device=z_batch[0].device)

    def compute_inverse_contrastive_loss(self, action_embed: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """对比损失: 同序列的转变 (z_t→z_{t+1}) 应相似, 不同序列的应不同。"""
        if action_embed.size(0) < 2:
            return torch.tensor(0.0, device=action_embed.device, requires_grad=True)
        # 归一化
        a = nn.functional.normalize(action_embed, dim=-1)
        sim = a @ a.T / temperature  # [N, N]
        # 正标签: 同序列
        pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # [N, N]
        # 去掉自身
        pos_mask = pos_mask.fill_diagonal_(False)
        if not pos_mask.any():
            return torch.tensor(0.0, device=action_embed.device, requires_grad=True)

        # InfoNCE
        exp_sim = torch.exp(sim)
        pos_exp = (exp_sim * pos_mask).sum(dim=1)
        neg_exp = (exp_sim * (~pos_mask).float()).sum(dim=1)
        loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8) + 1e-8).mean()
        return loss

    def information_gain(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """计算信息增益 = EMA(old_uncertainty) - current_uncertainty, 并更新 EMA。"""
        old_ema = self._uncertainty_ema.clone()
        current_avg = uncertainty.detach().mean()
        # 更新 EMA
        self._uncertainty_ema.mul_(self.info_gain_ema_decay).add_(current_avg * (1 - self.info_gain_ema_decay))
        self._step.add_(1)
        # 信息增益 = 旧 EMA - 当前 uncertainty (正值 = 学会了新东西)
        ig = old_ema - current_avg
        # 头 10 步返回 0 以稳定 EMA
        if self._step.item() < 10:
            return torch.zeros_like(ig)
        # fp16 安全: 替换 nan/inf
        ig = torch.nan_to_num(ig, nan=0.0, posinf=1.0, neginf=-1.0)
        return ig

    def forward(
        self,
        state_t: torch.Tensor,
        state_t1: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> dict:
        """完整 ICM 前向。

        Args:
            state_t:   z_t   [B, S, D]
            state_t1:  z_{t+1}  [B, S, D] (forward 目标)
            context:   [B, C] 上下文

        Returns:
            dict with keys:
                pred_loss:        前向预测 MSE
                inverse_loss:     反向模型对比损失
                uncertainty:      uncertainty 均值 (标量)
                information_gain: 信息增益 (标量)
                prediction_error: 原始 MSE (标量)
                forward_pred:     pred_{t+1}
                action_embed:     [B, action_dim]
                contrast_feat:    [B, action_dim]
        """
        # ── 前向预测 ──
        pred, uncertainty = self.forward_forward(state_t, context)
        target = state_t1.to(dtype=pred.dtype, device=pred.device)
        if target.dim() == 2:
            target = target.unsqueeze(1)
        if pred.shape != target.shape:
            pred = self._reshape_pred(pred, target.shape)
        err = (pred - target).pow(2)
        pred_loss = err.mean()
        prediction_error = err.mean(dim=-1).detach()

        # ── 反向模型 ──
        action_embed, contrast_feat = self.forward_inverse(state_t, state_t1)

        # ── 逆模型对比损失 ──
        bsz, seq_len, _ = state_t.shape
        pairs, labels = self._compute_inverse_labels(
            [state_t[i] for i in range(bsz)], seq_len)
        if pairs.size(0) > 0 and bsz > 1:
            inv_embed, _ = self.forward_inverse(
                pairs[:, :self.input_dim].unsqueeze(1),
                pairs[:, self.input_dim:].unsqueeze(1),
            )
            inverse_loss = self.compute_inverse_contrastive_loss(inv_embed, labels)
        else:
            inverse_loss = torch.tensor(0.0, device=state_t.device)

        # ── 信息增益 ──
        ig = self.information_gain(uncertainty)

        return {
            'pred_loss': pred_loss,
            'inverse_loss': inverse_loss * self.inv_loss_weight,
            'uncertainty': uncertainty.detach().mean(),
            'information_gain': ig.detach(),
            'prediction_error': prediction_error.detach().mean(),
            'forward_pred': pred,
            'action_embed': action_embed.detach(),
            'contrast_feat': contrast_feat.detach(),
        }

    def loss(self, state_t: torch.Tensor, state_t1: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """联合损失 = pred_loss + inv_loss_weight * inverse_loss + confidence_penalty。"""
        out = self.forward(state_t, state_t1, context)
        confidence_penalty = out['uncertainty'] * 0.1
        return out['pred_loss'] + out['inverse_loss'] + confidence_penalty

    def transition_error(self, state: torch.Tensor, target: torch.Tensor, context: torch.Tensor | None = None):
        """兼容 LatentWorldModel.transition_error 接口。"""
        return self.forward(state, target, context)['prediction_error'], \
               self.forward(state, target, context)['uncertainty']

    def reset_state(self):
        """重置 EMA 状态 — 在概念切换时调用。"""
        self._uncertainty_ema.zero_()
        self._step.zero_()
