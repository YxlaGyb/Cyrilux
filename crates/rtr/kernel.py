"""
RTR 核心算子

资格迹累加 / 逐状态价值残差 / 迹调制更新; 状态切片与登记由调用侧适配.
"""

from __future__ import annotations

import torch


def elig_accum(E: torch.Tensor, dW_raw: torch.Tensor, gamma: float) -> torch.Tensor:
    """资格迹: E ← γ·E + dW_raw, 返回当前迹, 与权重同形, 前缘对齐修剪活性切片

    dW_raw 为瞬时突触共活动标签; 迹是它的指数遗忘累积, 承载信用分配所需的时间信用.
    """
    E = E[: dW_raw.shape[0], : dW_raw.shape[1]]
    E.mul_(gamma).add_(dW_raw)
    return E


def rtr_residual(
    v_row: torch.Tensor, index: torch.Tensor, r: torch.Tensor, eta_v: float
) -> torch.Tensor:
    """三阶残差 m = r − V[idx], 并就地追踪 V[idx] ← V[idx] + η_v·m.

    r 为体内货币 (生存信号原语), V 为逐状态价值表. index 为 GPU 常驻整数下标张量,
    全程 gather / index_add_, 零 host 同步.
    """
    v = v_row.index_select(0, index)
    m = r - v
    v_row.index_add_(0, index, eta_v * m)
    return m


def rtr_apply(E: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """迹调制更新 ΔW = m·Ê, Ê = E/(‖E‖+1e-6) (逐域自归一, 免 /0 爆炸)."""
    return E * (m / (E.norm() + 1e-6))
