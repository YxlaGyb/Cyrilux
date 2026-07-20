"""Local Conv blocks — 纯局部 Conv1D 操作。
Conv1D(k=3, causal) + SwiGLU MLP, 接口兼容 CyreneBackbone。
"""
import torch
import torch.nn.functional as F
from torch import nn

from model.model_cyrene import _ACT2FN, CyreneConfig, FeedForward, RMSNorm


class LateralInhibition(nn.Module):
    """侧向抑制模块 — 生物启发的跨层竞争机制。

    高误差层抑制相邻层的误差信号，模拟生物神经网络中
    活跃神经元抑制邻居的侧向抑制现象。效果:
      - 锐化误差对比度 (高ε更强, 低ε更弱)
      - 促进层间专业化 (避免所有层学习相似特征)
      - 加速收敛 (减少冗余表示)

    Args:
        num_layers: PC 子层数 (默认 12)
        k: 每个抑制层影响的邻居半径 (默认 3)
        inhibition_strength: 抑制强度 (0=无抑制, 1=完全抑制)
    """

    def __init__(self, num_layers: int = 12, k: int = 3,
                 inhibition_strength: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.k = min(k, num_layers // 2)  # 对称半径
        self.inhibition_strength = inhibition_strength

        # 抑制权重核: Gaussian 形状, 中心为零, 邻域为正
        kernel = torch.zeros(2 * self.k + 1)
        for i in range(2 * self.k + 1):
            dist = abs(i - self.k)
            if dist == 0:
                kernel[i] = 0.0  # 不自抑制
            else:
                kernel[i] = torch.exp(torch.tensor(-dist ** 2 / (2 * (self.k / 2) ** 2)))
        # 归一化使抑制总和 = 1
        kernel = kernel / (kernel.sum() + 1e-8)
        self.register_buffer('_inhibition_kernel', kernel)

    def forward(self, ε_list):
        """应用侧向抑制到误差列表。

        Args:
            ε_list: list[Tensor], 每层误差 [B, S, H] 或 [B, S, H, F]

        Returns:
            ε_inhibited: list[Tensor], 抑制后的误差
        """
        if self.inhibition_strength <= 0 or self.num_layers <= 2:
            return ε_list

        L = len(ε_list)
        device = ε_list[0].device

        # 1) 计算每层误差范数 → [L]
        norms = torch.tensor([ε.float().norm().item() for ε in ε_list], device=device)

        # 2) 对范数进行 softmax 归一化 → 竞争权重
        w = F.softmax(norms / (norms.mean() + 1e-8), dim=0)

        # 3) 用事先计算的核做 1D 卷积 → 邻域抑制信号
        w_pad = F.pad(w.unsqueeze(0).unsqueeze(0),  # [1, 1, L]
                      pad=(self.k, self.k), mode='replicate')
        # 显式 float() 确保与 w_pad 类型一致 (模型 .half() 后 buffer 为 fp16)
        kernel = self._inhibition_kernel.float().view(1, 1, -1).to(device)
        inhibition = F.conv1d(w_pad, kernel)[0, 0]  # [L]

        # 4) 从原始 ε 中减去抑制信号
        ε_inhibited = []
        for ℓ, ε in enumerate(ε_list):
            inh_factor = self.inhibition_strength * inhibition[ℓ]
            ε_mod = ε * (1.0 - inh_factor)
            ε_inhibited.append(ε_mod)

        return ε_inhibited

    def extra_repr(self):
        return (f'L={self.num_layers}, k={self.k}, '
                f'inhibition={self.inhibition_strength:.2f}')


class SalienceGate(nn.Module):
    """每通道可学习显著性门控。

    每个隐藏维度维护一个可学习的 salience logit,
    z_out = z * σ(logit / τ), 其中 σ 是 sigmoid。
    温度 τ 控制门的硬度: τ → 0 时趋近离散 0/1。

    EMA 追踪激活水平, 供神经发生控制器 (Phase C) 使用。

    Config:
        hidden_size: 隐藏维度
        temperature: sigmoid 温度 (默认 0.1, 越低越硬)
        init_open: 初始化为全开 (True) 或全关 (False)
    """

    def __init__(self, hidden_size: int, temperature: float = 0.1, init_open: bool = True):
        super().__init__()
        init_val = 5.0 if init_open else -5.0  # σ(5)≈0.993, σ(-5)≈0.007
        self.logits = nn.Parameter(torch.full([hidden_size], init_val))
        self.temperature = temperature

        # EMA 追踪 (供 Phase C 神经发生控制器使用)
        self.register_buffer('activation_ema', torch.zeros(hidden_size))
        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """应用门控。训练时软门控 (可微), 推理时硬门控 (阈值 0.5)。

        Args:
            z: [*, hidden_size] 输入

        Returns:
            z_gated: [*, hidden_size] 门控输出
        """
        if self.training:
            # 软门控: sigmoid, 可微
            gate = torch.sigmoid(self.logits / self.temperature)
        else:
            # 硬门控: 阈值 0.5
            gate = (torch.sigmoid(self.logits) > 0.5).to(z.dtype)

        # 更新激活 EMA
        if self.training:
            with torch.no_grad():
                # 对除最后一维外的所有维度求平均 → 每通道均值
                act = z.abs().mean(dim=tuple(range(z.ndim - 1)))  # [hidden_size]
                decay = 0.99
                self.activation_ema.mul_(decay).add_(act * (1 - decay))
                self._step += 1

        return z * gate.view(1, 1, -1)

    @torch.no_grad()
    def get_gate_values(self) -> torch.Tensor:
        """返回当前门控值 [hidden_size], 范围 (0,1)。"""
        return torch.sigmoid(self.logits / self.temperature)

    @torch.no_grad()
    def get_active_ratio(self, threshold: float = 0.01) -> float:
        """活性通道比例 (gate > threshold)。"""
        return (self.get_gate_values() > threshold).float().mean().item()

    @torch.no_grad()
    def get_sparsity_loss(self, β: float = 0.01) -> torch.Tensor:
        """门控稀疏正则损失: β · Σ(1 - σ(logits))²

        鼓励 logits 要么很大 (全开) 要么很小 (全关), 避免中间态。
        """
        gate = torch.sigmoid(self.logits)
        return β * ((1.0 - gate) ** 2).sum()

    def extra_repr(self):
        active = self.get_active_ratio()
        return f'H={self.logits.size(0)}, τ={self.temperature}, active={active:.2%}'


class LocalConvBlock(nn.Module):
    """纯局部 Conv 块: Conv1D → residual → SwiGLU MLP → residual

    接口与 CyreneBackbone 的 LocalConvBlock 兼容:
      .input_layernorm, .local_conv (替代 .self_attn),
      .post_attention_layernorm, .mlp

    ponytail: 因果 Conv1D(k=3) 替代全局 self-attention。
    每位置只看过去 2 个 token, 无跨位置全局混合。
    位置信息由 conv kernel 的相对位置天然编码, 无需 RoPE。
    """

    def __init__(self, layer_id, config, dilation=1):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        self.dilation = dilation

        # ── Conv 子层 (替代 Attention) ──
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # kernel_size=3, bias=False, padding=0 → 手动 causal pad
        self.local_conv = nn.Conv1d(
            config.hidden_size, config.hidden_size,
            kernel_size=3, padding=0, dilation=dilation, bias=False,
        )

        # ── MLP 子层 (SwiGLU) ──
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        use_fused = getattr(config, 'use_fused_mlp', True)
        self.mlp = FusedFeedForward(config) if use_fused else FeedForward(config)

    def forward(self, hidden_states, position_embeddings=None, **kwargs):
        """前向: Conv → residual → MLP → residual

        Args:
            hidden_states: [bsz, seq_len, hidden_size]
            position_embeddings: 忽略 (接口兼容)

        Returns:
            hidden_states: [bsz, seq_len, hidden_size]
            None: 接口兼容 CyreneBlock (past_kv placeholder)
        """
        # ── Conv 子层 ──
        residual = hidden_states
        # 因果填充: 左侧 pad 2*dilation 个零, 右侧不 pad
        h = F.pad(hidden_states, (0, 0, 2 * self.dilation, 0))  # [bsz, seq_len+2*dilation, hidden]
        h = self.input_layernorm(h)
        h = h.transpose(1, 2)  # [bsz, hidden, seq_len+2]
        h = self.local_conv(h)  # [bsz, hidden, seq_len]
        h = h.transpose(1, 2)  # [bsz, seq_len, hidden]
        hidden_states = residual + h

        # ── MLP 子层 ──
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = residual + hidden_states

        return hidden_states, None

    def extra_repr(self):
        return f'layer_id={self.layer_id}, conv={self.local_conv}'

class FusedFeedForward(nn.Module):
    """Fused SwiGLU MLP — 将 gate_proj + up_proj 合并为单次 matmul.

    标准 SwiGLU:
      gate = gate_proj(x)   # [B, S, H] → [B, S, inter]
      up   = up_proj(x)     # [B, S, H] → [B, S, inter]
      out  = silu(gate) * up
      y    = down_proj(out) # [B, S, inter] → [B, S, H]

    融合后:
      fused = gate_up_proj(x)  # [B, S, H] → [B, S, 2*inter]
      gate, up = fused.split(inter, dim=-1)
      out = silu(gate) * up
      y   = down_proj(out)

    减少 1 次 matmul / 前向, 33% MLP 计算量减少。
    """

    def __init__(self, config: CyreneConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        # 融合 gate + up 为一个更大的权重矩阵
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = _ACT2FN[config.hidden_act]

    def forward(self, x):
        """fp32 前向 — 显式权重提升防 silu 溢出."""
        w_gu = self.gate_up_proj.weight.float()
        w_d = self.down_proj.weight.float()
        x32 = x.float()
        fused = F.linear(x32, w_gu)  # [B, S, 2*inter]
        gate, up = fused.chunk(2, dim=-1)
        return F.linear(self.act_fn(gate) * up, w_d).to(x.dtype)