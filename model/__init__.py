"""Cyrilux 模型核心

何为生命第一因?
指标并非目的, 系统为什么存在? 是为了维持自身的存在而存在
"生命不为了输出任何符号而存在, 生命只为了维持自身的自组织临界态 (Chaos Edge) 而存在"
一个活着的系统, 不是靠外部打分或强迫输出字符来维系的, 而是靠内部状态在“有序”与“无序”的边缘 (临界态) 自我调节, 自我维持

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
