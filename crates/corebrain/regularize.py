"""
梯度无关 VICReg

协方差项用 Foldiak 反希伯去相关; 方差项用逐槽稳态增益保底.
两项作用在槽位激活上, 不作用在权重上.
"""

from __future__ import annotations

import torch


def foldiak_decorrelate(a: torch.Tensor, cov: torch.Tensor, rate: float) -> torch.Tensor:
    """反希伯去相关: a ← a − rate·(a·off-diag(aᵀa/N)). 单样本批次跳过 (协方差退化)."""
    n = a.shape[0]
    if n < 2:
        return a
    cov.zero_()
    cov.addmm_(a.transpose(0, 1), a, alpha=1.0 / n)
    cov.diagonal().zero_()
    return a - rate * (a @ cov)


def variance_floor(a: torch.Tensor, std_floor: float) -> torch.Tensor:
    """逐槽稳态增益: 该槽在 batch 上的 std 低于下限时整列重标. 增益落在 [1, 2]."""
    std = a.square().mean(dim=0).sqrt()
    lo = torch.full_like(std, std_floor * 0.5)
    gain = std_floor / (torch.maximum(std, lo) + 1e-6)
    gain = torch.maximum(gain, torch.ones_like(gain))
    return a * gain


def participation_ratio(a: torch.Tensor) -> torch.Tensor:
    """逐槽能量的参与比: 有效维数读数 (单值集中 → 1, 均摊 → n)."""
    v = a.square().mean(dim=0)
    return v.sum().square() / (v.square().sum() + 1e-8)
