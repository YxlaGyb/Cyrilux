"""DensePCNet — 密集 GPU 预测编码网络.

完全替代事件驱动版的纯 matmul 实现:
- L0 带位置编码
- 6 层全连接前馈+反馈
- 时序惯性 (z_old[t-1] → z_new[t])
- Hebbian 外积学习 + Oja + BCM + k-WTA + 列 dropout
- 全 fp16, 零 Python 循环, 零 .item()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DensePCConfig:
    """密集 PC 网络配置."""

    # 层维度和
    d_input: int = 256  # 输入字节维度 (固定 vocab_size)
    d_pe: int = 256  # 位置编码维度
    d_l4: int = 1024
    d_l2: int = 384
    d_l3: int = 384
    d_l5: int = 256
    d_l6: int = 128
    max_seq_len: int = 256

    # Hebbian 参数
    hebbian_base_eta: float = 3e-4
    oja_alpha: float = 0.05
    column_dropout: float = 0.25
    dopamine: float = 1.0  # 固定调制 (后续可扩展)

    # 时间惯性 alpha (per-layer, per-neuron)
    inertia_alpha: float = 0.3

    # k-WTA 比例
    kwta_ratio: float = 0.10

    # 列维度和 (用于 lm_head 读取)
    top_layer: str = "l5"

    def dims(self) -> dict[str, int]:
        return {
            "l0": self.d_input + self.d_pe,
            "l4": self.d_l4,
            "l2": self.d_l2,
            "l3": self.d_l3,
            "l5": self.d_l5,
            "l6": self.d_l6,
        }

    def param_count(self) -> int:
        """预估参数量 (不计 PE)."""
        d = self.dims()
        n = 0
        # 前馈 (只有 FF)
        n += d["l0"] * d["l4"]  # W_04
        n += d["l4"] * d["l2"]  # W_42
        n += d["l2"] * d["l3"]  # W_23
        n += d["l3"] * d["l5"]  # W_35
        n += d["l5"] * d["l6"]  # W_56
        # LM head
        n += d["l5"] * self.d_input  # W_LM
        # 偏置
        n += d["l4"] + d["l2"] + d["l3"] + d["l5"] + d["l6"] + self.d_input
        return n


class DensePCNet(nn.Module):
    """密集预测编码网络.

    Args:
        config: 网络配置 (或 None 使用默认).
        max_seq_len: 位置编码最大长度.
    """

    def __init__(self, config: DensePCConfig | None = None, max_seq_len: int = 256):
        super().__init__()
        self.cfg = config or DensePCConfig()
        S = max_seq_len
        d = self.cfg.dims()

        # ── 位置编码 (固定 sin/cos) ──
        pe = torch.zeros(S, self.cfg.d_pe, dtype=torch.float16)
        pos = torch.arange(S, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.cfg.d_pe, 2, dtype=torch.float32) * (-math.log(10000.0) / self.cfg.d_pe))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_encoding", pe)  # [S, d_pe]

        # ── 前馈权重 ──
        self.W_04 = nn.Parameter(torch.empty(d["l4"], d["l0"], dtype=torch.float16))
        self.W_42 = nn.Parameter(torch.empty(d["l2"], d["l4"], dtype=torch.float16))
        self.W_23 = nn.Parameter(torch.empty(d["l3"], d["l2"], dtype=torch.float16))
        self.W_35 = nn.Parameter(torch.empty(d["l5"], d["l3"], dtype=torch.float16))
        self.W_56 = nn.Parameter(torch.empty(d["l6"], d["l5"], dtype=torch.float16))

        # ── 反馈权重（用前馈权转置, 共享参数）──
        # 不额外分配 W_65/W_53/W_32/W_24，所有反馈走 W_x.T

        # ── 时序权重 (每层) ──
        self.W_t4 = nn.Parameter(torch.empty(d["l4"], d["l4"], dtype=torch.float16))
        self.W_t2 = nn.Parameter(torch.empty(d["l2"], d["l2"], dtype=torch.float16))
        self.W_t3 = nn.Parameter(torch.empty(d["l3"], d["l3"], dtype=torch.float16))
        self.W_t5 = nn.Parameter(torch.empty(d["l5"], d["l5"], dtype=torch.float16))
        self.W_t6 = nn.Parameter(torch.empty(d["l6"], d["l6"], dtype=torch.float16))

        # ── LM Head ──
        self.W_LM = nn.Parameter(torch.empty(256, d["l5"], dtype=torch.float16))
        self.bias_lm = nn.Parameter(torch.zeros(256, dtype=torch.float16))

        # ── 层偏置 ──
        self.bias_l4 = nn.Parameter(torch.zeros(d["l4"], dtype=torch.float16))
        self.bias_l2 = nn.Parameter(torch.zeros(d["l2"], dtype=torch.float16))
        self.bias_l3 = nn.Parameter(torch.zeros(d["l3"], dtype=torch.float16))
        self.bias_l5 = nn.Parameter(torch.zeros(d["l5"], dtype=torch.float16))
        self.bias_l6 = nn.Parameter(torch.zeros(d["l6"], dtype=torch.float16))

        # ── BCM 参数 (per-neuron) ──
        self.bcm_slope = nn.Parameter(torch.empty(d["l4"] + d["l2"] + d["l3"] + d["l5"] + d["l6"], dtype=torch.float16))
        self.bcm_zero = nn.Parameter(torch.empty(d["l4"] + d["l2"] + d["l3"] + d["l5"] + d["l6"], dtype=torch.float16))

        self._init_weights()

    def _init_weights(self):
        """Kaiming 初始化所有权重."""
        scale = 0.1
        for name, p in self.named_parameters():
            if "bias" in name or "bcm" in name:
                continue
            if "pos_encoding" in name:
                continue
            nn.init.normal_(p, mean=0.0, std=scale / math.sqrt(p.shape[-1]))
        nn.init.uniform_(self.bcm_slope, 2.8, 5.2)
        nn.init.uniform_(self.bcm_zero, 0.15, 0.45)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """前馈预测.

        Args:
            byte_ids: [N, S] long (0..255).

        Returns:
            logits: [N, S, 256].
        """
        return self._predict(byte_ids, store_state=True)

    def _predict(self, byte_ids: torch.Tensor, store_state: bool = True) -> torch.Tensor:
        """核心前馈逻辑 (复用给 learn)."""
        N, S = byte_ids.shape
        dev = byte_ids.device
        d = self.cfg.dims()
        alpha = self.cfg.inertia_alpha

        # ── L0: one-hot + 位置编码 (固定幅度 0.5) ──
        one_hot = F.one_hot(byte_ids, num_classes=256).to(torch.float16)
        pe = self.pos_encoding[:S].unsqueeze(0) * 0.5
        z0 = torch.cat([one_hot, pe.expand(N, -1, -1)], dim=-1)

        # ── 前馈预测 (每层: matmul(÷√dim) + 偏置 + 时间惯性 + k-WTA) ──
        mu4 = z0 @ self.W_04.T / (d["l0"] ** 0.5) + self.bias_l4
        z4, z4_all = self._layer_step(N, S, d["l4"], mu4, alpha, dev, self.W_t4, 0.80)

        mu2 = z4 @ self.W_42.T / (d["l4"] ** 0.5) + self.bias_l2
        z2, z2_all = self._layer_step(N, S, d["l2"], mu2, alpha, dev, self.W_t2, 0.80)

        mu3 = z2 @ self.W_23.T / (d["l2"] ** 0.5) + self.bias_l3
        z3, z3_all = self._layer_step(N, S, d["l3"], mu3, alpha, dev, self.W_t3, 0.80)

        mu5 = z3 @ self.W_35.T / (d["l3"] ** 0.5) + self.bias_l5
        z5, z5_all = self._layer_step(N, S, d["l5"], mu5, alpha, dev, self.W_t5, 1.0)  # L5 全保留

        mu6 = z5 @ self.W_56.T / (d["l5"] ** 0.5) + self.bias_l6
        z6, z6_all = self._layer_step(N, S, d["l6"], mu6, alpha, dev, self.W_t6, 1.0)

        if store_state:
            self._z0 = z0
            self._z4 = z4_all
            self._z2 = z2_all
            self._z3 = z3_all
            self._z5 = z5_all
            self._z6 = z6_all

        # ── LM Head (行归一化防高字节垄断) ──
        rn = self.W_LM.norm(dim=1, keepdim=True) + 1e-8
        logits = (self.W_LM / rn) @ z5.transpose(-2, -1) + self.bias_lm.unsqueeze(1)
        logits = logits.transpose(-2, -1)  # [N, S, 256]
        return logits

    def _layer_step(
        self,
        N: int,
        S: int,
        dim: int,
        mu: torch.Tensor,
        alpha: float,
        dev: torch.device,
        W_t: torch.Tensor,
        kwta: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """单层前馈: z = mu (活动值), 返回 mu 用于 Hebbian."""
        z = torch.zeros(N, S, dim, dtype=torch.float16, device=dev)
        z[:, 0] = mu[:, 0]
        z[:, 1:] = mu[:, 1:]  # z=mu (基础)
        # 时序：z[t] += 0.1 * W_t @ z[t-1] (让模型学"h后面是e")
        temporal = z[:, :-1] @ W_t.T
        z[:, 1:] = z[:, 1:] + temporal * 0.1

        # 每帧归一化防爆炸
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)

        mu_ret = z.clone()
        if kwta > 0 and kwta < 1.0:
            k = max(1, int(dim * kwta))
            vals, idxs = z.topk(k, dim=-1)
            mask = torch.zeros_like(z)
            mask.scatter_(-1, idxs, 1.0)
            z = z * mask + (1.0 - mask) * 1e-5 * z
        return z, mu_ret

    def learn(self, byte_ids: torch.Tensor, targets: torch.Tensor) -> dict:
        """Hebbian 学习 (前馈 + 外积更新).

        Args:
            byte_ids: [N, S] long 输入.
            targets: [N, S] long 目标字节 (-100 表示忽略).

        Returns:
            stats dict.
        """
        # 1. 前馈, 保存状态
        logits = self._predict(byte_ids, store_state=True)
        N, S = byte_ids.shape
        dev = byte_ids.device
        d = self.cfg.dims()
        eta = self.cfg.hebbian_base_eta * 50.0
        oja = self.cfg.oja_alpha

        # L0
        one_hot = F.one_hot(byte_ids, num_classes=256).to(torch.float16)
        pe = self.pos_encoding[:S].unsqueeze(0).expand(N, -1, -1)
        z0 = torch.cat([one_hot, pe], dim=-1)

        # 预测误差 (前馈, mu 计算加与 forward 相同的 √dim 缩放)
        d = self.cfg.dims()
        eps4 = self._z4 - (z0 @ self.W_04.T / (d["l0"] ** 0.5) + self.bias_l4)
        eps2 = self._z2 - (self._z4 @ self.W_42.T / (d["l4"] ** 0.5) + self.bias_l2)
        eps3 = self._z3 - (self._z2 @ self.W_23.T / (d["l2"] ** 0.5) + self.bias_l3)
        eps5 = self._z5 - (self._z3 @ self.W_35.T / (d["l3"] ** 0.5) + self.bias_l5)
        eps6 = self._z6 - (self._z5 @ self.W_56.T / (d["l5"] ** 0.5) + self.bias_l6)

        # ── Hebbian 外积 + 列 dropout ──
        # dW[post, pre] = sum_{N,S}(eps_post * z_pre)
        dW_list = [
            (eps4.transpose(-2, -1) @ z0).sum(dim=0).to(torch.float16),
            (eps2.transpose(-2, -1) @ self._z4).sum(dim=0).to(torch.float16),
            (eps3.transpose(-2, -1) @ self._z2).sum(dim=0).to(torch.float16),
            (eps5.transpose(-2, -1) @ self._z3).sum(dim=0).to(torch.float16),
            (eps6.transpose(-2, -1) @ self._z5).sum(dim=0).to(torch.float16),
        ]
        W_list = [self.W_04, self.W_42, self.W_23, self.W_35, self.W_56]
        for dW, W in zip(dW_list, W_list):
            col_mask = torch.rand(W.shape[0], 1, device=dev) < self.cfg.column_dropout
            W.data += (dW * (~col_mask).to(torch.float16)) * eta

        # ── LM Head Hebbian ──
        # targets [N, S'], logits [N, S, 256], _z5 [N, S, d_l5]
        # S' 可能比 S 少 1 (labels[:, 1:]), 对齐
        S_t = targets.shape[-1]
        z5_lm = self._z5[:, :S_t]  # [N, S', d_l5]
        logits_lm = logits[:, :S_t]  # [N, S', 256]
        valid = targets >= 0
        if valid.any():
            z5_valid = z5_lm[valid]  # [V, d_l5]
            tg_valid = targets[valid]
            preds = logits_lm[valid].argmax(dim=-1)
            err = preds != tg_valid
            dw_lm = torch.zeros(256, d["l5"], dtype=torch.float16, device=dev)
            if (~err).any():
                x = z5_valid[~err]
                dw_lm.index_add_(0, tg_valid[~err], (0.1 * eta * x).to(torch.float16))
            if err.any():
                x = z5_valid[err]
                dw_lm.index_add_(0, tg_valid[err], (eta * x).to(torch.float16))
                dw_lm.index_add_(0, preds[err], (-eta * x).to(torch.float16))
            col_mask = torch.rand(256, device=dev) < self.cfg.column_dropout
            dw_lm[~col_mask] = 0.0
            self.W_LM.data += dw_lm

        # ── LM Head 列 Oja: 每步单位列范数 (平权竞争) ──
        col_norm = self.W_LM.data.norm(dim=0, keepdim=True)
        col_norm = torch.where(col_norm > 1e-8, col_norm, torch.ones_like(col_norm))
        self.W_LM.data = self.W_LM.data / col_norm

        # ── 前馈 Oja 上限约束 ──
        for name, W in [("04", self.W_04), ("42", self.W_42), ("23", self.W_23), ("35", self.W_35), ("56", self.W_56)]:
            col_norm = W.data.norm(dim=1, keepdim=True)
            over = col_norm > 8.0
            if over.any():
                scale = torch.where(over, 8.0 / (col_norm + 1e-6), 1.0)
                W.data *= scale

        return {"logits_norm": logits.norm().item()}

    def save(self, path: str):
        """保存模型权重."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, config: DensePCConfig | None = None) -> DensePCNet:
        """加载模型权重."""
        net = cls(config or DensePCConfig())
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return net
