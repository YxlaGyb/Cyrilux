"""
归一化与范数保持算子.
"""

from __future__ import annotations

import torch


def soft_norm_preserve(W: torch.Tensor) -> None:
    """行范数保持 0.8-1.2, 结构化: 幅度差异保留, 权重有界防 fp16 溢出.

    范数归一化收缩 + 软缩放: 行范数归一到 1 后乘 soft 缩放因子,
    超范数行强收缩 (≥1.2 收缩), 弱行微放大 (≤0.8 放大), 中段近恒等.
    原实现 rn/(rn+1e-4) 在范数 2 时饱和到 1, 收缩近零.
    """
    rn = W.norm(dim=1, keepdim=True)
    soft = 0.8 + 0.4 * (rn / (rn + 1.0))  # [0.8, 1.2), 范数大时趋向 1.2 但乘上归一化强收缩
    W.mul_(soft / (rn + 1e-4))


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """零向量保护 RMS 归一化 (fp16 下 1e-8 舍为 0, 需掩码保护分母)."""
    rms = x.square().mean(dim=-1, keepdim=True)
    alive = (rms > 1e-8).to(x.dtype)
    denom = torch.where(alive > 0, (rms * 1.01).sqrt(), torch.ones_like(rms))
    return x * alive / denom


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    """零向量保护 L2 归一化 (与 rms_norm 量纲不同: 分母为范数非均方根)."""
    nrm = x.norm(dim=-1, keepdim=True)
    alive = (nrm > 1e-8).to(x.dtype)
    denom = torch.where(alive > 0, nrm * 1.01, torch.ones_like(nrm))
    return x * alive / denom


def rel_norm(raw: float, ema: float, eps: float = 1e-6) -> float:
    """自归一化相对量 (rel_n 家族, bind.py 同款): raw/(ema+raw+eps) ∈ [0,1)."""
    return raw / (ema + raw + eps)
