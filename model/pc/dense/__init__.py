"""密集 GPU PC 网络 — 全 matmul, 零事件驱动, 零 Python 循环.

与现有稀疏/事件驱动版本 `model/pc/` 完全独立。保留相同的 6 层 PC 算法，
但全部用密集 matmul 表达，GPU 利用率目标 >90%。

Usage:
    from model.pc.dense import DensePCNet
    net = DensePCNet().cuda()
    logits = net(byte_ids)           # [N, S, 256]
    stats = net.learn(byte_ids, targets)
"""

from .core import DensePCNet, DensePCConfig

__all__ = ["DensePCNet", "DensePCConfig"]
