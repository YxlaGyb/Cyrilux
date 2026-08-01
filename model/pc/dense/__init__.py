"""密集 GPU PPA 闭环网络

全 matmul, 零事件驱动, 零 Python 循环.

与稀疏/事件驱动版本 `model/pc/sparse/`

完全独立。PPA: 感知-预测-行动闭环, 全 fp16, 零反向传播, 自由能驱动。

Usage:
    from model.pc.dense import DensePCNet
    net = DensePCNet().cuda()
    out = net(byte_ids)                # dict: mu4_top / eps4 / rpe / free_energy
    stats = net.learn(byte_ids)        # Hebbian 更新, 返回 free_energy 等
"""

from .core import DensePCConfig, DensePCNet

__all__ = ["DensePCNet", "DensePCConfig"]
