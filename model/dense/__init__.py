"""密集 GPU PPA 闭环网络

全 matmul, 零事件驱动, 零 Python 循环.

PPA: 感知-预测-行动闭环, 全 fp16, 零反向传播, 自由能驱动。

Usage:
    from model.dense import DensePCNet
    net = DensePCNet().cuda()
    out = net(byte_ids)                # dict: mu_diff / diff_err / free_energy
    stats = net.learn(byte_ids)        # Hebbian 更新, 返回 free_energy 等
"""

from .network import DensePCConfig, DensePCNet

__all__ = ["DensePCNet", "DensePCConfig"]
