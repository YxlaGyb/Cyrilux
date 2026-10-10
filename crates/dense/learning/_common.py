"""
学习引擎共享状态适配层
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from crates.model.model_cyrene import DensePCNet

from crates.kernels import (
    activity_baseline,
    elig_accum,
    energy_constraint,
    rho_control,
)
from crates.model.constants import ELIG_GAMMA


class _MixinBase:
    """mixin 公共基类: 让类型检查器认识组合类成员 (self.net)."""

    net: DensePCNet


def _mark_ema_init(net: DensePCNet, ema_name: str) -> None:
    """冷启动登记: CPU 侧集合 + 持久化 mask 位 (每个名字至多置一次, 零同步).

    集合本身不持久化 → 续跑后 17 个活动 EMA 会全部重新快照而非继续, 故 mask 记录谁曾初始化.
    """
    net._active_ema_init.add(ema_name)
    net._active_ema_init_mask.add_(1 << net._active_ema_names.index(ema_name))


def restore_active_ema_init(net: DensePCNet) -> set[str]:
    """load 边界: 由 mask 重建 CPU 侧冷启动集合 (一次性同步)."""
    mask = int(net._active_ema_init_mask.item())
    return {n for i, n in enumerate(net._active_ema_names) if (mask >> i) & 1}


def _activity_baseline(
    net: DensePCNet, post: torch.Tensor, ema_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """适配器: 取 EMA 缓冲与冷启动标志, 调 kernels.activity_baseline, 首窗登记."""
    initialized = ema_name in net._active_ema_init
    ema = getattr(net, ema_name)
    p2, excess = activity_baseline(post, ema, initialized)
    if not initialized:
        _mark_ema_init(net, ema_name)
    return p2, excess


def _energy_constraint(
    net: DensePCNet, W: torch.Tensor, dW: torch.Tensor, post: torch.Tensor, ema_name: str
) -> torch.Tensor:
    """适配器: 取 EMA/冷启动标志与 cfg 系数, 调 kernels.energy_constraint, 首窗登记."""
    initialized = ema_name in net._active_ema_init
    ema = getattr(net, ema_name)
    out = energy_constraint(
        W, dW, post, ema, initialized, net.cfg.oja_alpha, net.cfg.oja_elasticity
    )
    if not initialized:
        _mark_ema_init(net, ema_name)
    return out


def _elig_accum(net: DensePCNet, wname: str, dW_raw: torch.Tensor) -> torch.Tensor:
    """适配器: 取权重对应资格迹缓冲, 调 kernels.elig_accum (γ=ELIG_GAMMA).

    R 默认 0 → 迹只记录不注入 (ΔW=η·R·E=0); 生存环境接入时经 _survival_signal 启用.
    """
    E = getattr(net, f"{wname}_elig")
    return elig_accum(E, dW_raw, ELIG_GAMMA)


def _rho_ctrl(dW: torch.Tensor, W_ref: torch.Tensor, tag: str, net: DensePCNet) -> torch.Tensor:
    """适配器: 调 kernels.rho_control, 把通道读数写入 net._rho_map (诊断)."""
    dW_s, rho, s_i = rho_control(dW, W_ref, net.cfg.adaptive_rho)
    if rho is None:
        return dW_s
    nW_i = W_ref.norm() + 1e-8
    net._rho_map[tag] = (rho, dW_s.norm() / nW_i, s_i)
    return dW_s
