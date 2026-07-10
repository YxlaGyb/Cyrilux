"""
Local Conv blocks — 替代 Transformer Attention 的纯局部操作。
Ponytail: Conv1D(k=3, causal) + SwiGLU MLP, 接口兼容 MiniMindBlock。
"""
import torch.nn.functional as F
from torch import nn
from model.model_minimind import RMSNorm, FeedForward


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
