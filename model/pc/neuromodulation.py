"""神经调制 — 张量化多巴胺/乙酰胆碱/精度调度.

基于 torch tensor 操作, 替代旧的 Python math 标量循环.
"""

from __future__ import annotations

import math

import torch

from .tensor_pool import TensorNeuronPool


def compute_uncertainty(F_hist: list[float], window: int = 10) -> float:
    """从自由能历史计算不确定度 (标量, 滑动窗口太小不值得 GPU)."""
    if len(F_hist) < 3:
        return 0.5
    window = min(window, len(F_hist))
    recent = F_hist[-window:]
    mean_F = sum(recent) / window
    if mean_F < 1e-8:
        return 0.0
    var_F = sum((f - mean_F) ** 2 for f in recent) / window
    cv = (var_F ** 0.5) / mean_F
    return float(math.tanh(cv * 3.0))


def compute_dopamine(F_curr: float, F_prev: float, beta: float = 0.5) -> float:
    """多巴胺 RPE: D = sigmoid(beta * (F_prev - F_curr))."""
    if not (math.isfinite(F_curr) and math.isfinite(F_prev)):
        return 0.5
    x = -beta * (F_prev - F_curr)
    if x > 50:
        return 1.0
    if x < -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def compute_ach(uncertainty: float, beta_0: float = 0.0) -> float:
    """乙酰胆碱: ACh = sigmoid(-uncertainty + beta_0)."""
    x = uncertainty - beta_0
    if abs(x) > 50:
        return 1.0 if x < 0 else 0.0
    return 1.0 / (1.0 + math.exp(x))


def combine_modulation(D: float, ACh: float) -> float:
    """组合调制信号: 0.5*D + 0.5*ACh."""
    return 0.5 * D + 0.5 * ACh


def compute_precision_scales(
    pool: TensorNeuronPool,
    D: float,
    ACh: float,
    eta: float = 1.0,
) -> None:
    """逐神经元精度权重: pi = 1 + eta*D*|eps| + eta*ACh*|eps| (批量 tensor)."""
    pool.learning.compute_precision_scales(D, ACh, eta)
