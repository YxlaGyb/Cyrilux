"""Latent-space world model for the PC continual-learning loop.

This module is intentionally lightweight and practical: it learns a compact
latent transition model over the PC hidden states and produces an uncertainty
signal that can be used to decide whether a training step should trigger a full
update, a light update, or a memory consolidation step.
"""
from __future__ import annotations

import torch
from torch import nn


class LatentWorldModel(nn.Module):
    """Small latent dynamics model over PC hidden states.

    The model takes a latent state tensor of shape [B, S, D] and an optional
    context vector [B, C], and predicts a next latent state of the same shape.
    It also outputs a scalar uncertainty estimate for each batch element.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, context_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        in_dim = input_dim + context_dim
        self.state_proj = nn.Linear(in_dim, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pred_head = nn.Linear(hidden_dim, input_dim)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

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

    def forward(self, state: torch.Tensor, context: torch.Tensor | None = None):
        """Predict the next latent state and a per-batch uncertainty score."""
        pooled, context = self._prepare_state(state, context)
        x = torch.cat([pooled, context], dim=-1)
        h = torch.tanh(self.state_proj(x))
        h = torch.tanh(self.hidden_proj(h))
        pred = self.pred_head(h)
        uncertainty = torch.sigmoid(self.uncertainty_head(h))
        pred = self._reshape_pred(pred, state.shape)
        return pred, uncertainty

    def loss(self, state: torch.Tensor, target: torch.Tensor, context: torch.Tensor | None = None):
        """Compute a practical latent transition loss.

        The loss combines latent transition error with a small regularizer that
        discourages the model from becoming overly confident on all states.
        """
        pred, uncertainty = self(state, context)
        target = target.to(dtype=pred.dtype, device=pred.device)
        if target.dim() == 2:
            target = target.unsqueeze(1)
        if pred.shape != target.shape:
            pred = self._reshape_pred(pred, target.shape)

        err = (pred - target).pow(2).mean(dim=-1, keepdim=True)
        mse = err.mean()
        confidence_penalty = uncertainty.mean() * 0.1
        return mse + confidence_penalty

    def transition_error(self, state: torch.Tensor, target: torch.Tensor, context: torch.Tensor | None = None):
        """Convenience helper returning the raw transition error and uncertainty."""
        pred, uncertainty = self(state, context)
        target = target.to(dtype=pred.dtype, device=pred.device)
        if target.dim() == 2:
            target = target.unsqueeze(1)
        if pred.shape != target.shape:
            pred = self._reshape_pred(pred, target.shape)
        err = (pred - target).pow(2).mean(dim=-1, keepdim=True)
        return err.mean(), uncertainty.mean()

    def transition_error_with_features(self, state: torch.Tensor, target: torch.Tensor, context: torch.Tensor | None = None):
        """Extended version returning raw error + uncertainty + placeholder features.

        Provides compatibility bridge between LatentWorldModel and IntrinsicCuriosityModule.
        """
        pred, uncertainty = self(state, context)
        target = target.to(dtype=pred.dtype, device=pred.device)
        if target.dim() == 2:
            target = target.unsqueeze(1)
        if pred.shape != target.shape:
            pred = self._reshape_pred(pred, target.shape)
        err = (pred - target).pow(2).mean(dim=-1, keepdim=True)
        dummy_feat = torch.zeros(state.size(0), 32, device=state.device, dtype=state.dtype)
        return err.mean(), uncertainty.mean(), dummy_feat, 0.0

    def reset_state(self):
        """Reset any internal state (MLP model is stateless — placeholder for future use)."""
        pass
