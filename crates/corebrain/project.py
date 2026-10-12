"""
视图投影

P_short / P_long 两组独立投影, 局部 delta 规则学.
P_long_ema 是慢速目标编码器, 只由 EMA 跟随 P_long, 不接收误差 (stop-grad).
"""

from __future__ import annotations

import torch
from torch import nn

from crates.kernels import soft_norm_preserve, ste_delta


class ViewProjector(nn.Module):
    """短/长两视图 → 共享槽位空间."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        scale = 1.0 / float(cfg.d_in) ** 0.5
        self.P_short = nn.Parameter(torch.randn(cfg.d_in, cfg.n_slots, dtype=torch.float16) * scale)
        self.P_long = nn.Parameter(torch.randn(cfg.d_in, cfg.n_slots, dtype=torch.float16) * scale)
        soft_norm_preserve(self.P_short)
        soft_norm_preserve(self.P_long)
        self.register_buffer("P_long_ema", self.P_long.detach().clone())

    def forward(self, short: torch.Tensor, long: torch.Tensor):
        return short @ self.P_short, long @ self.P_long, long @ self.P_long_ema

    def _trust(self, grad: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """信任域: ‖ΔW‖ ≤ tr·‖W‖ (谱界缩放, 非 clamp)."""
        nw = weight.norm() + 1e-8
        ng = grad.norm() + 1e-8
        return grad * torch.minimum(self.cfg.trust_region * nw / ng, torch.ones_like(ng))

    @torch.no_grad()
    def learn(self, short, long, e, win_short, win_long) -> None:
        self.P_short.sub_(self.cfg.eta_proj * self._trust(ste_delta(short, e, win_short), self.P_short))
        self.P_long.sub_(self.cfg.eta_proj * self._trust(ste_delta(long, e, win_long), self.P_long))
        soft_norm_preserve(self.P_short)
        soft_norm_preserve(self.P_long)

    @torch.no_grad()
    def ema(self) -> None:
        self.P_long_ema.mul_(self.cfg.ema_tau).add_(self.P_long, alpha=1.0 - self.cfg.ema_tau)
