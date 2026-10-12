"""
槽位预测器

Q 学 短视图激活 → EMA 目标 的映射, 局部 delta 规则
"""

from __future__ import annotations

import torch
from torch import nn

from crates.kernels import soft_norm_preserve


class SlotPredictor(nn.Module):
    """
    跨视图预测器: A_short → ŝ(A_long 的 EMA 目标).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.Q = nn.Parameter(torch.eye(cfg.n_slots, dtype=torch.float16) * 0.5)

    def forward(self, a_short: torch.Tensor) -> torch.Tensor:
        return a_short @ self.Q

    @torch.no_grad()
    def learn(self, a_short: torch.Tensor, e: torch.Tensor) -> None:
        n = max(1, a_short.shape[0])
        g = (a_short.transpose(0, 1) @ e) / n
        nw = self.Q.norm() + 1e-8
        ng = g.norm() + 1e-8
        g = g * torch.minimum(self.cfg.trust_region * nw / ng, torch.ones_like(ng))
        self.Q.sub_(self.cfg.eta_q * g)
        soft_norm_preserve(self.Q)
