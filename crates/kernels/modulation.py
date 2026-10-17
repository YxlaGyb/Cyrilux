"""
神经调制纯算子

标量调制 (多巴胺/ACh/不确定度) + 自由能 + 行为门
"""

from __future__ import annotations

import math

import torch


def compute_uncertainty(F_hist: list[float], window: int = 10) -> float:
    """
    从自由能历史计算不确定度，标量，滑动窗口太小不值得 GPU
    """
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
    """多巴胺 RPE: D = sigmoid(beta * (F_curr - F_prev))."""
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


def ach_noise(mu3: torch.Tensor, nov: torch.Tensor) -> torch.Tensor:
    """ACh 抖动，逐元素 ±(0.03·(0.5+nov)) 随机符号叠加到激活上"""
    return mu3 + torch.sign(2.0 * (torch.rand_like(mu3) - 0.5)) * (0.03 * (0.5 + nov))


def combine_modulation(D: float, ACh: float) -> float:
    """组合调制信号: 0.5*D + 0.5*ACh."""
    return 0.5 * D + 0.5 * ACh


def compute_free_energy(eps_list: list[torch.Tensor]) -> torch.Tensor:
    """自由能 F = Σ ε² 各层均方和."""
    return sum(eps.square().mean() for eps in eps_list)


def gate_psay(g_eco: float, g_perc: float, stress: float, A: float, nov: float,
              k_g: float, k_s: float, k_a: float, k_n: float) -> float:
    """行为门纯函数 (测试锚; GPU 版 _update_gate 与之公式一一对应).

    p_say = σ(κ_g·(g_eco−g_perc) − κ_s·stress + κ_A·(A−0.5) − κ_n·(nov−0.5)).
    符号: 账本差 >0 (说比看近期更赚) → 想说; 应激高 (能量紧) → 省着少说;
    振荡器 A 高相位 → 说 (节律基因); 新奇高 → 多看少说. σ 软饱和 = 结构性有界.
    """
    logit = (k_g * (g_eco - g_perc) - k_s * stress
             + k_a * (A - 0.5) - k_n * (nov - 0.5))
    return 1.0 / (1.0 + math.exp(-logit))
