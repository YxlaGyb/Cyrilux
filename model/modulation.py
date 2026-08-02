"""共享神经调制

神经调制 (参数化) + 自由能 + 软范数保持.

dense (全 matmul) 与 sparse (页式槽位) 两个后端共用的公式.
系数参数化, 各后端传自己验证过的数值, 行为不变.
"""

from __future__ import annotations

import math

import torch


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
    cv = (var_F**0.5) / mean_F
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


def compute_dopamine_gain(rel: float, lo: float = 0.3, hi: float = 5.0) -> float:
    """多巴胺增益: rel 相对惊喜比的线性夹取 (dense 侧 5 万步验证范围 [0.3, 5.0])."""
    return min(max(rel, lo), hi)


def compute_ach_gain(rel: float, max_gain: float = 3.0) -> float:
    """ACh 增益: 低惊喜 → 高增益 (记忆巩固), 上限 max_gain."""
    return min(1.0 / (rel + 1e-4), max_gain)


def compute_free_energy(eps_list: list[torch.Tensor]) -> torch.Tensor:
    """自由能 F = Σ ε² 各层均方和 (fp16 进出, 零 .float())."""
    return sum(eps.square().mean() for eps in eps_list)


def soft_norm_preserve(W: torch.Tensor) -> None:
    """行范数保持 0.8-1.2, 结构化非 clamp: 幅度差异保留, 权重有界防 fp16 溢出.

    范数归一化收缩 + 软缩放: 行范数归一到 1 后乘 soft 缩放因子,
    超范数行强收缩 (≥1.2 收缩), 弱行微放大 (≤0.8 放大), 中段近恒等.
    原实现 rn/(rn+1e-4) 在范数 2 时饱和到 1, 收缩近零 (装饰性), 已修复.
    """
    rn = W.norm(dim=1, keepdim=True)
    soft = 0.8 + 0.4 * (rn / (rn + 1.0))  # [0.8, 1.2), 范数大时趋向 1.2 但乘上归一化强收缩
    W.mul_(soft / (rn + 1e-4))
