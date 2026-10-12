"""
RTR 生存信号

体内 ΔF 货币 → 逐状态价值残差, 每步无条件写点, 与行为门无关
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from crates.model.constants import RTR_ETA_V
from crates.rtr.kernel import rtr_residual

if TYPE_CHECKING:
    from crates.model.model_cyrene import HybridPCNet


def update_survival_signal(net: HybridPCNet, ctx) -> None:
    """
    生存信号写点
    状态 = 池化 z4, 账本 = 权向量 _rtr_w[:, 列], 列 = 感知 / 回声 双轨.

    残差 m = _metab_R − w·ẑ 由 rtr_residual 就地追踪后写入 net._survival_signal.
    _metab_R 由 _update_metabolism 在本调用前结算 (行为门之外, 感知相亦有值).
    """
    pooled = net._z4.mean(dim=(0, 1))
    col = 1 if ctx.echo_loop else 0
    net._survival_signal = rtr_residual(net._rtr_w[:, col], pooled, net._metab_R, RTR_ETA_V)
