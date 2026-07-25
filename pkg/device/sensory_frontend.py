"""SensoryFrontend — GPU 感官前端 (纯 Conv1D, 无 MLP)。

独立于 CyreneBackbone, 仅保留局部卷积部分:
  byte_proj → 6 × SensoryConvBlock → 7 个 h_conv 张量

H_front 是唯一固定维度 (默认 64, 可配)。
没有 SwiGLU MLP, 没有 post_attention_layernorm。
没有 train/eval: 永恒 no_grad 运行, 纯特征提取。
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class DopamineGateRMSNorm(nn.Module):
    """多巴胺门控 RMSNorm — 用调制信号动态缩放增益.

    移植自旧世界 local_blocks.DopamineGateRMSNorm.
    D 为组合调制信号 (0.5·D + 0.5·ACh), ∈ (0,1).
    高 D → gain 增大 (增强该层响应), 低 D → gain 减小 (抑制).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5, dopamine_eta: float = 0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.dopamine_eta = dopamine_eta

    def forward(self, x: torch.Tensor, dopamine_D: float | None = None) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        if dopamine_D is not None:
            gain = self.weight * (1 + self.dopamine_eta * (2 * dopamine_D - 1))
        else:
            gain = self.weight
        return gain * x_norm


class SensoryConvBlock(nn.Module):
    """单层感官卷积 block — 因果 dilated conv + salience gate + 残差.

    MLP 和 post_attention_layernorm 已被移除 ——
    SwiGLU MLP 的密集计算由 CPU NeuronPool 的稀疏连接取代.
    使用 DopamineGateRMSNorm 替代 LayerNorm (fp16 安全).

    Args:
        channels: 特征通道数 (H_front, 默认 64)
        dilation: 扩张率 (1, 2, 4, 8, 16, 32)
        block_id: 层索引 (0..5), 用于权重加载
    """

    def __init__(self, channels: int, dilation: int, block_id: int):
        super().__init__()
        self.channels = channels
        self.dilation = dilation
        self.block_id = block_id

        # RMSNorm (fp16 安全) → causal dilated conv1d(k=3) → salience gate → residual
        self.norm = DopamineGateRMSNorm(channels, eps=1e-5)
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=3, bias=False, dilation=dilation, padding=0
        )
        self.gate = nn.Parameter(torch.full([channels], 5.0))  # σ(5)≈0.993 (全开)

    def forward(self, h: torch.Tensor, dopamine_D: float | None = None) -> torch.Tensor:
        """前向: [1, C, S] → [1, C, S] (因果, 不改变序列长度).

        Args:
            h: [1, channels, S] 输入特征, C=H_front, S=序列位置数
            dopamine_D: 可选的多巴胺调制值, 传入 norm 动态缩放 gain

        Returns:
            h_out: [1, channels, S]
        """
        # Norm (DopamineGateRMSNorm, fp16 安全)
        h_t = h.transpose(1, 2)  # [1, S, C]
        h_norm = self.norm(h_t, dopamine_D)  # RMSNorm on last dim
        h_norm = h_norm.transpose(1, 2)  # [1, C, S]

        # Causal dilated conv1d (k=3, d=dilation)
        pad = 2 * self.dilation
        h_pad = F.pad(h_norm, (pad, 0))  # pad left only (causal)

        h_conv = self.conv(h_pad)

        # Salience gate: h * sigmoid(gate)
        gate = torch.sigmoid(self.gate)  # [C]
        h_gated = h_conv * gate.view(1, -1, 1)

        # Residual
        return h + h_gated


class SensoryFrontend(nn.Module):
    """感官前端 — 字节序列 → 多尺度特征张量列表。

    与旧 CyreneBackbone 的兼容性:
      - byte_proj 权重形状: [H_front, 2, 13]
      - 6 conv blocks 权重形状: [H_front, H_front, 3] (dilation 感知)
      - 可从现有 checkpoint 加载

    Args:
        h_front: 特征通道数, 默认 64 (唯一固定维度)
        dilations: 扩张率序列, 默认 (1, 2, 4, 8, 16, 32)
    """

    def __init__(self, h_front: int = 64, dilations: Sequence[int] = (1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.h_front = h_front

        # 字节投影: Conv1d(2 → H_front, k=13, causal, no bias)
        self.byte_proj = nn.Conv1d(2, h_front, kernel_size=13, padding=0, bias=False)

        # 6 层感官卷积 (无 MLP)
        self.blocks = nn.ModuleList(
            [SensoryConvBlock(h_front, d, bid) for bid, d in enumerate(dilations)]
        )

        # 本项目只使用 fp16 — .half() 原地转换参数
        self.byte_proj.half()
        for block in self.blocks:
            block.half()

    def forward(self, byte_seq: torch.Tensor) -> list[torch.Tensor]:
        """前向: 字节序列 → 7 个 h_conv 张量。

        Args:
            byte_seq: [1, 2, S] fp16 — 双通道字节编码 (ch0=字节值, ch1=角色掩码)

        Returns:
            h_list: list[torch.Tensor; 7] 每张量形状 [1, H_front, S]
                     h_list[0] = byte_proj 输出
                     h_list[1..6] = conv block 1..6 输出
        """
        # causal pad (12,0) for k=13
        x = F.pad(byte_seq, (12, 0))
        h = self.byte_proj(x)  # [1, H_front, S]

        h_list = [h]
        for block in self.blocks:
            h = block(h, dopamine_D=None)  # 前端推理阶段不使用多巴胺调制
            h_list.append(h)

        return h_list

    @torch.inference_mode()
    def forward_streaming(self, byte_seq: torch.Tensor) -> list[torch.Tensor]:
        """推断模式前向 (等效于 forward, 仅语义标记为推理)。

        Args:
            byte_seq: [1, 2, S]

        Returns:
            h_list: list[Tensor; 7] 每张量 [1, H_front, S]
        """
        return self.forward(byte_seq)

    def load_pretrained_conv_weights(self, state_dict: dict, h_front: int, old_h: int = 256):
        """从旧 CyreneBackbone checkpoint 加载 conv 权重。

        Args:
            state_dict: 旧模型的 state_dict
            h_front: 新 H_front 维度
            old_h: 旧 hidden_size (默认 256)
        """
        # byte_proj
        key = "byte_proj.weight"
        if key in state_dict:
            w = state_dict[key]  # [old_h, 2, 13]
            if h_front <= old_h:
                self.byte_proj.weight.data.copy_(w[:h_front])
            else:
                self.byte_proj.weight.data[:old_h] = w
                # 新通道用小随机初始化
                with torch.no_grad():
                    self.byte_proj.weight.data[old_h:] *= 0.01

        # conv blocks
        for i, block in enumerate(self.blocks):
            # 旧 backbone 中 conv 权重在 blocks[i].local_conv.weight
            conv_key = f"layers.{i}.local_conv.weight"
            if conv_key in state_dict:
                w = state_dict[conv_key]  # [old_h, old_h, 3, dilation?]
                # Conv1d weight: [C_out, C_in/groups, k]
                # dilated conv: same shape, dilation is separate param
                if w.dim() == 4:
                    w = w.squeeze(-2)  # [old_h, old_h, 3]
                c_out = min(h_front, old_h)
                c_in = min(h_front, old_h)
                block.conv.weight.data[:c_out, :c_in] = w[:c_out, :c_in]

    def get_output_channels(self) -> int:
        """返回特征通道数 H_front。"""
        return self.h_front

    def extra_repr(self):
        return f"h_front={self.h_front}, blocks={len(self.blocks)}"
