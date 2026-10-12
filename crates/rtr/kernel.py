"""
RTR 核心算子

资格迹累加 / 逐状态价值残差 / 迹调制更新; 状态切片与登记由调用侧适配.
"""

from __future__ import annotations

import torch


def elig_accum(E: torch.Tensor, dW_raw: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    资格迹: E ← γ·E + dW_raw, 返回当前迹, 与权重同形, 前缘对齐修剪活性切片

    dW_raw 为瞬时突触共活动标签; 迹是它的指数遗忘累积, 承载信用分配所需的时间信用.
    """
    E = E[: dW_raw.shape[0], : dW_raw.shape[1]]
    E.mul_(gamma).add_(dW_raw)
    return E


def rtr_residual(w: torch.Tensor, z: torch.Tensor, r: torch.Tensor, eta_v: float) -> torch.Tensor:
    """
    逐状态残差 m = r − w·ẑ, ẑ = z/(‖z‖+1e-6), 并就地追踪 w ← w + η_v·m·ẑ.

    w 是账本权向量: 地址长在权重里, 状态相近则更新落在重叠的同一批分量上, 无需划分状态.
    状态自归一使 ‖Δw‖ = η_v·|m| 有界; 追踪无自举项, 无条件稳定.
    """
    z_hat = z / (z.norm() + 1e-6)
    m = r - torch.dot(w, z_hat)
    w.add_(eta_v * m * z_hat)
    return m.reshape(1)


def rtr_apply(E: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """
    迹调制更新 ΔW = m·Ê, Ê = E/(‖E‖+1e-6) (逐域自归一, 免 /0 爆炸).
    """
    return E * (m / (E.norm() + 1e-6))
