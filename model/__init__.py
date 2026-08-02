"""Cyrilux 模型核心

双后端预测编码:
  dense  — 全 matmul PPA 闭环 (DensePCNet)
  sparse — 页式事件驱动 (CyreneModel 驱动 TensorNeuronPool)

模型定义主文件: model.model_cyrene (CyreneModel/CyreneConfig)
"""

from model.dense import DensePCConfig, DensePCNet
from model.model_cyrene import CyreneConfig, CyreneModel
from model.sparse import TensorNeuronPool

__all__ = [
    "DensePCNet",
    "DensePCConfig",
    "CyreneModel",
    "CyreneConfig",
    "TensorNeuronPool",
]
