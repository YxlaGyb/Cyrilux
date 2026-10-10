"""
逐元素非线性激活算子

分流域与前向热路径共用; 抽取自内联表达式以统一口径.
"""

from __future__ import annotations

import torch


def saturate(x: torch.Tensor) -> torch.Tensor:
    """分流抑制: x/(1+|x|), 有界单调饱和, 掐断下游尖峰正反馈."""
    return x / (1.0 + x.abs())


def cubic(x: torch.Tensor) -> torch.Tensor:
    """多项式激活: x·(1 − 0.5x²), 零 BP 的奇次软饱和校正."""
    return x * (1.0 - 0.5 * x.pow(2))
