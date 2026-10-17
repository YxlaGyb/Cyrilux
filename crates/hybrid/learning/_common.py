"""
学习引擎共享状态适配层
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from crates.model.model_cyrene import HybridPCNet

    from .engine import LearnCtx

from crates.kernels import (
    activity_baseline,
    energy_constraint,
    rho_control,
)


@dataclass
class LayerErrs:
    """逐层预测误差，精度加权版"""

    eps4: torch.Tensor
    eps2: torch.Tensor
    eps3: torch.Tensor
    eps5: torch.Tensor
    eps6: torch.Tensor
    eps2_precise: torch.Tensor
    eps6_precise: torch.Tensor


class _MixinBase:
    """mixin 公共基类: 让类型检查器认识组合类成员 (self.net)."""

    net: HybridPCNet


def _mark_ema_init(net: HybridPCNet, ema_name: str) -> None:
    """冷启动登记: CPU 侧集合 + 持久化 mask 位 (每个名字至多置一次, 零同步).

    集合本身不持久化 → 续跑后 17 个活动 EMA 会全部重新快照而非继续, 故 mask 记录谁曾初始化.
    """
    net._active_ema_init.add(ema_name)
    net._active_ema_init_mask.add_(1 << net._active_ema_names.index(ema_name))


def restore_active_ema_init(net: HybridPCNet) -> set[str]:
    """load 边界: 由 mask 重建 CPU 侧冷启动集合 (一次性同步)."""
    mask = int(net._active_ema_init_mask.item())
    return {n for i, n in enumerate(net._active_ema_names) if (mask >> i) & 1}


def _activity_baseline(
    net: HybridPCNet, post: torch.Tensor, ema_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """适配器: 取 EMA 缓冲与冷启动标志, 调 kernels.activity_baseline, 首窗登记."""
    initialized = ema_name in net._active_ema_init
    ema = getattr(net, ema_name)
    p2, excess = activity_baseline(post, ema, initialized)
    if not initialized:
        _mark_ema_init(net, ema_name)
    return p2, excess


def _energy_constraint(
    net: HybridPCNet, W: torch.Tensor, dW: torch.Tensor, post: torch.Tensor, ema_name: str
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


def _plastic(post: torch.Tensor, net: HybridPCNet, layer: str, pos=None) -> torch.Tensor:
    """可塑码门控 P_l = C_l ∩ S_l: 死/不活跃神经元不进权重更新.

    post 为逐位置突触后信号; pos 对齐 P_l 与 post 的位置轴 (切片或布尔掩码).
    """
    P = getattr(net, f"_P_{layer}")
    if pos is not None:
        P = P[:, pos]
    return post * P.to(post.dtype)


def _rho_ctrl(dW: torch.Tensor, W_ref: torch.Tensor, tag: str, net: HybridPCNet) -> torch.Tensor:
    """适配器: 调 kernels.rho_control, 把通道读数写入 net._rho_map (诊断)."""
    dW_s, rho, s_i = rho_control(dW, W_ref, net.cfg.adaptive_rho)
    if rho is None:
        return dW_s
    nW_i = W_ref.norm() + 1e-8
    net._rho_map[tag] = (rho, dW_s.norm() / nW_i, s_i)
    return dW_s


def layer_errs(ctx: LearnCtx) -> LayerErrs:
    """
    逐层预测误差块 (自下而上 PC): L5 用时序差分误差, L6 用时间自预测.

    逐位置误差先过可塑码门控 P_l (死/不活跃神经元不进权重更新).
    """
    net = ctx.net
    dim_4, dim_2, dim_3, dim_6 = ctx.dim_4, ctx.dim_2, ctx.dim_3, ctx.dim_6
    z0, z4, z2, z3, z5, z6 = net._z0, net._z4, net._z2, net._z3, net._z5, net._z6
    eps4 = _plastic(z4 - (z0 @ net.W_04[:dim_4].T + net.bias_l4[:dim_4]), net, "l4")
    eps2 = _plastic(z2 - (z4 @ net.W_42[:dim_2].T + net.bias_l2[:dim_2]), net, "l2")
    eps3 = _plastic(z3 - (z2 @ net.W_23[:dim_3].T + net.bias_l3[:dim_3]), net, "l3")
    z6_pre = torch.cat([net._padmax[:, :, :dim_6], z6[:, :-1]], dim=1)
    eps6 = _plastic(z6 - z6_pre, net, "l6")
    eps5 = _plastic(z5[:, 1:] - z5[:, :-1], net, "l5", pos=slice(1, None))
    fe = net.forward_engine
    return LayerErrs(eps4, eps2, eps3, eps5, eps6, fe._precise(eps2), fe._precise(eps6))
