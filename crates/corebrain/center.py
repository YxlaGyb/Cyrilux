"""
中心脑编排

读原生 z4, 在自己的一组槽位上做多视图竞争与跨视图预测, 输出窗级 Δ 修正量
只由跨视图预测误差更新, 不接收字节误差, 不写回 net._z4
"""

from __future__ import annotations

import torch
from torch import nn

from crates.attention import AttentionConfig, ErrorAttention
from crates.corebrain.config import CoreBrainConfig
from crates.corebrain.predict import SlotPredictor
from crates.corebrain.project import ViewProjector
from crates.kernels import (
    foldiak_decorrelate,
    participation_ratio,
    topk_wta,
    variance_floor,
)


class CoreBrain(nn.Module):
    """平行新层, z4 → 两视图 → 各自 WTA → 跨视图预测 → 窗级 Δ."""

    def __init__(self, cfg: CoreBrainConfig | None = None):
        super().__init__()
        self.cfg = cfg or CoreBrainConfig()
        self.projector = ViewProjector(self.cfg)
        self.predictor = SlotPredictor(self.cfg)
        self.attention = ErrorAttention(
            self.cfg.n_slots,
            AttentionConfig(
                ema_tau=self.cfg.attention_ema_tau,
                temperature=self.cfg.attention_temperature,
                floor=self.cfg.attention_floor,
                top_k=self.cfg.top_k,
            ),
        )
        k = self.cfg.n_slots
        self.register_buffer("_cov_s", torch.zeros(k, k, dtype=torch.float16))
        self.register_buffer("_cov_l", torch.zeros(k, k, dtype=torch.float16))
        self.register_buffer("_win_ema", torch.zeros(2, k, dtype=torch.float16))
        self.register_buffer("_delta_prev", torch.zeros(self.cfg.d_in, dtype=torch.float16))
        self._state: dict[str, torch.Tensor] | None = None
        self._stats: dict[str, torch.Tensor] = {}

    def _views(
        self, z4: torch.Tensor, valid: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        短视图 = 最后 q 个内容位池化,
        长视图 = 全内容池化, 范围按内容长度, 与填充位置无关.

        valid 是填充边界掩码, 非填充位 True.
        q 是声明窗口的 short_frac 份, 不随传入张量长度变.
        尾窗左对齐, 短窗锚在内容末尾 max(0, n - q) 而非位置末尾 S - q:
            按位置切会把填充当内容, 内容不足 q 时短窗自然退化为整段内容.
        """
        q = max(1, int(self.cfg.window_len * self.cfg.short_frac))
        if valid is None:
            valid = torch.ones(z4.shape[:2], dtype=torch.bool, device=z4.device)  # tensor-guard: rare (诊断未传掩码)
        v = valid.unsqueeze(-1).to(torch.float16)
        w_short = (valid & (valid.cumsum(dim=1) > (valid.sum(dim=1, keepdim=True) - q)))
        w_short = w_short.unsqueeze(-1).to(torch.float16)
        short = (z4 * w_short).sum(dim=1) / w_short.sum(dim=1).clamp(min=1.0)
        long = (z4 * v).sum(dim=1) / v.sum(dim=1).clamp(min=1.0)
        return short, long

    @torch.no_grad()
    def forward(self, z4: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        """窗级 Δ, [N, d_in]. 平铺即得逐位置修正量."""
        short, long = self._views(z4, valid)
        p_short, p_long, p_target = self.projector(short, long)
        a_short, win_short = topk_wta(p_short, self.cfg.top_k)
        a_long, win_long = topk_wta(p_long, self.cfg.top_k)
        a_target, _ = topk_wta(p_target, self.cfg.top_k)
        a_short = variance_floor(
            foldiak_decorrelate(a_short, self._cov_s, self.cfg.vicreg_cov), self.cfg.std_floor
        )
        a_long = variance_floor(
            foldiak_decorrelate(a_long, self._cov_l, self.cfg.vicreg_cov), self.cfg.std_floor
        )
        pred = self.predictor(a_short)
        delta = (pred @ self.projector.P_long.T) * self.cfg.delta_scale
        self._state = {
            "short": short,
            "long": long,
            "a_short": a_short,
            "a_long": a_long,
            "pred": pred,
            "target": a_target,
            "win_short": win_short,
            "win_long": win_long,
            "delta": delta,
        }
        return delta

    @torch.no_grad()
    def learn(self) -> dict[str, torch.Tensor]:
        """
        一步跨视图预测更新, 只动中心脑自己的权重, 字节损失不参与.

        `_state` 不在此清除: 静态图回放期 Python 不执行, 前向与学习分居两张图,
        状态必须跨图存活; 下一次 forward 原位覆盖.
        """
        st = self._state
        if st is None:
            return dict(self._stats)
        e = st["pred"] - st["target"]
        attn = self.attention(st["a_short"], st["a_long"], e)
        delta_eff = attn.effective_error
        e_p = delta_eff @ self.predictor.Q.T
        self.predictor.learn(st["a_short"], delta_eff)
        self.projector.learn(st["short"], st["long"], e_p, st["win_short"], st["win_long"])
        self.projector.ema()
        shuffled = self.predictor(st["a_short"].roll(1, dims=0))
        self._win_ema[0].mul_(0.99).add_(0.01 * st["win_short"].to(torch.float16).mean(dim=0))
        self._win_ema[1].mul_(0.99).add_(0.01 * st["win_long"].to(torch.float16).mean(dim=0))
        delta_mean = st["delta"].mean(dim=0)
        delta_cos = (delta_mean * self._delta_prev).sum() / (delta_mean.norm() * self._delta_prev.norm() + 1e-6)
        self._delta_prev.copy_(delta_mean)
        self._stats = {
            "xv_energy": e.square().mean(),
            "xv_energy_effective": delta_eff.square().mean(),
            "xv_energy_shuffled": (shuffled - st["target"]).square().mean(),
            "effective_rank": participation_ratio(st["a_long"]),
            "slot_overlap": (st["win_short"] & st["win_long"]).to(torch.float16).mean(),
            "delta_cos": delta_cos,
            "delta_absmax": st["delta"].abs().max(),
            "attention_gain_mean": attn.gain.mean(),
            "attention_gain_min": attn.gain.min(),
            "attention_gain_var": attn.gain.var(unbiased=False),
            "attention_salience_norm": attn.salience.norm(dim=-1).mean(),
            "attention_top_down_norm": attn.top_down.norm(),
        }
        return dict(self._stats)

    @torch.no_grad()
    def observe(
        self, z4: torch.Tensor | None = None, valid: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        诊断读数, 传 z4 则跑一步 forward+learn, 否则返回最近一步读数.
        """
        if z4 is not None:
            self.forward(z4, valid)
            return self.learn()
        return dict(self._stats)
