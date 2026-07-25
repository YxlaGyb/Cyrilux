"""神经调制 — 多巴胺 RPE、乙酰胆碱、不确定度、精度调度。

从旧世界移植的核心思想:
  - compute_uncertainty:  自由能历史变异系数 → 不确定度 ∈ [0,1)
  - compute_dopamine:     RPE 多巴胺 D = σ(β·ΔF)
  - compute_ach:          乙酰胆碱 ACh = σ(-uncertainty + β₀)
  - combine_modulation:   0.5·D + 0.5·ACh → 学习速率调制
  - compute_precision_scales: 逐神经元 π = 1 + η·D·|ε| + η·ACh·|ε|

所有函数纯标量运算, 无 torch 依赖 (除 sigmoid/tanh 初始化)。
"""

from __future__ import annotations

import math

from .neuron_pool import NeuronPool


# ── 不确定度 ──────────────────────────────────────────────────────


def compute_uncertainty(F_hist: list[float], window: int = 10) -> float:
    """从自由能历史计算不确定度.

    uncertainty = tanh(var(F_window) / (mean(F_window) + 1e-8))
    近期 F 波动越大 → uncertainty 越高.

    Args:
        F_hist: list[float] — 自由能历史
        window: int — 滑动窗口大小 (默认 10)

    Returns:
        uncertainty: float ∈ [0, 1)
    """
    if len(F_hist) < 3:
        return 0.5  # 冷启动: 中等不确定度
    window = min(window, len(F_hist))
    recent = F_hist[-window:]
    mean_F = sum(recent) / window
    if mean_F < 1e-8:
        return 0.0
    var_F = sum((f - mean_F) ** 2 for f in recent) / window
    cv = (var_F ** 0.5) / mean_F  # 变异系数
    # tanh 映射到 [0, 1)
    return float(math.tanh(cv * 3.0))


# ── 多巴胺 RPE ────────────────────────────────────────────────────


def compute_dopamine(F_curr: float, F_prev: float, beta: float = 0.5) -> float:
    """计算多巴胺 RPE 信号.

    D = sigmoid(β · (F_prev - F_curr))
      F_prev > F_curr (自由能下降) → D → 1 (正奖赏)
      F_prev < F_curr (自由能上升) → D → 0 (负奖赏)

    NaN/Inf 防护: 异常输入返回中性值 D=0.5.

    Args:
        F_curr: 当前步自由能
        F_prev: 上一步自由能
        beta:   RPE 敏感度 (默认 0.5)

    Returns:
        D: float ∈ (0, 1)
    """
    if not math.isfinite(F_curr):
        return 0.5
    if not math.isfinite(F_prev):
        return 0.5
    δ = F_prev - F_curr
    # sigmoid 手动实现 (避免 torch 依赖)
    try:
        x = beta * δ
        if x > 20.0:
            return 1.0
        if x < -20.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))
    except (OverflowError, ValueError):
        return 0.5


# ── 乙酰胆碱 ──────────────────────────────────────────────────────


def compute_ach(uncertainty: float, beta_0: float = 0.0) -> float:
    """计算乙酰胆碱信号.

    ACh = sigmoid(-uncertainty + β₀)
      高不确定度 → ACh 低 (不确信时降低学习率)
      低不确定度 → ACh 高 (确信时加速学习)

    Args:
        uncertainty: 不确定度 ∈ [0, 1)
        beta_0:      ACh 基线偏移 (默认 0.0)

    Returns:
        ACh: float ∈ (0, 1)
    """
    x = -uncertainty + beta_0
    if x > 20.0:
        return 1.0
    if x < -20.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def combine_modulation(D: float, ACh: float) -> float:
    """组合多巴胺与乙酰胆碱为统一学习调制信号.

    modulation = 0.5·D + 0.5·ACh

    Args:
        D:   多巴胺值 ∈ (0, 1)
        ACh: 乙酰胆碱值 ∈ (0, 1)

    Returns:
        modulation: float ∈ (0, 1)
    """
    return 0.5 * D + 0.5 * ACh


# ── 精度调度 ──────────────────────────────────────────────────────


def compute_precision_scales(
    pool: NeuronPool,
    D: float,
    ACh: float,
    eta: float = 1.0,
) -> None:
    """为所有神经元更新精度权重 π.

    π_n = 1 + η·D·|ε_n| + η·ACh·|ε_n|

    在 free_energy 计算中: F += 0.5 * π_n * ε_n^2
    高误差 + 高调制 → 更高精度 → 该神经元对自由能贡献更大.

    Args:
        pool: NeuronPool 实例 (每个神经元的 π 字段会被就地更新)
        D:    多巴胺值 ∈ (0, 1)
        ACh:  乙酰胆碱值 ∈ (0, 1)
        eta:  精度敏感度 (默认 1.0)
    """
    for neuron in pool.neurons.values():
        e_abs = abs(neuron.ε)
        neuron.π = 1.0 + eta * D * e_abs + eta * ACh * e_abs
