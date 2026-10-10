"""
存活掩码 S_l.

生死机制的对外读数: 每个物理工位是否存活 (1/0), 结构与学习数学无关.
hybrid/mask.py 只读它做门控; 压实点由 topology 重写.

物理宽度 = 活体数起步 (S 全一); 死亡关位后 S 置零, 压实后 S 收缩回全一.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from crates.model.model_cyrene import HybridPCNet

LAYERS = ("l4", "l2", "l3", "l5", "l6")


def init_survival(net: HybridPCNet) -> None:
    """登记每层存活掩码 S_l (物理宽度起步全一). row 索引登记见 registry.ROW_STATE."""
    for layer in LAYERS:
        net.register_buffer(
            f"_S_{layer}", torch.ones(net.active_size[layer], dtype=torch.bool)
        )


def survival_mask(net: HybridPCNet, layer: str) -> torch.Tensor:
    """读某层存活掩码 [N_l] bool."""
    return getattr(net, f"_S_{layer}")
