import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

# Local activation function map — replaces transformers.activations.ACT2FN
_ACT2FN = {"silu": F.silu, "gelu": F.gelu, "relu": F.relu}


class CyreneConfig:
    def __init__(self, hidden_size=768, num_hidden_layers=8, **kwargs):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = kwargs.get("dropout", 0.0)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.intermediate_size = kwargs.get(
            "intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64
        )
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.use_fused_mlp = kwargs.get("use_fused_mlp", True)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self.norm(x)


class FeedForward(nn.Module):
    def __init__(self, config: CyreneConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = _ACT2FN[config.hidden_act]

    def forward(self, x):
        """fp16 前向 (直接使用 fp16 权重)."""
        g = self.act_fn(F.linear(x, self.gate_proj.weight))
        u = F.linear(x, self.up_proj.weight)
        return F.linear(g * u, self.down_proj.weight)
