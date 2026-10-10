from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from crates.attention.config import AttentionConfig
from crates.kernels import topk_wta


@dataclass
class AttentionOutput:
    effective_error: torch.Tensor
    gain: torch.Tensor
    salience: torch.Tensor
    top_down: torch.Tensor
    a_long_ema: torch.Tensor


class ErrorAttention(nn.Module):
    def __init__(self, n_slots: int, cfg: AttentionConfig | None = None):
        super().__init__()
        self.cfg = cfg or AttentionConfig()
        self.register_buffer("_a_long_ema", torch.zeros(n_slots, dtype=torch.float16))

    @torch.no_grad()
    def forward(
        self,
        a_short: torch.Tensor,
        a_long: torch.Tensor,
        delta: torch.Tensor,
    ) -> AttentionOutput:
        mean_long = a_long.mean(dim=0)
        self._a_long_ema.mul_(self.cfg.ema_tau).add_(mean_long, alpha=1.0 - self.cfg.ema_tau)
        salience = (a_short - self._a_long_ema).abs()
        top_down, _ = topk_wta(self._a_long_ema.abs(), self.cfg.top_k)
        g_raw = salience + top_down
        p = torch.softmax(g_raw / self.cfg.temperature, dim=-1)
        gain = (1.0 - self.cfg.floor) * float(p.shape[-1]) * p + self.cfg.floor
        return AttentionOutput(
            effective_error=gain * delta,
            gain=gain,
            salience=salience,
            top_down=top_down,
            a_long_ema=self._a_long_ema,
        )
