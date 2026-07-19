"""
Local Conv blocks — 替代 Transformer Attention 的纯局部操作。
Ponytail: Conv1D(k=3, causal) + SwiGLU MLP, 接口兼容 MiniMindBlock。
"""
import torch
import torch.nn.functional as F
from torch import nn
from model.model_minimind import RMSNorm, FeedForward


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

    接口与 MiniMindBlock 兼容:
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

        # ── MLP 子层 (SwiGLU, 与 MiniMindBlock 相同) ──
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, position_embeddings=None, **kwargs):
        """前向: Conv → residual → MLP → residual

        Args:
            hidden_states: [bsz, seq_len, hidden_size]
            position_embeddings: 忽略 (接口兼容)

        Returns:
            hidden_states: [bsz, seq_len, hidden_size]
            None: 接口兼容 MiniMindBlock (past_kv placeholder)
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
