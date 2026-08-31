"""model._archived_sparse — DEPRECATED · 稀疏存档命名空间

117 轮 Round 1: 稀疏 CyreneModel 从活动机体移除, 归档于此作参考。
不参与 model/__init__ 的活动导出链。如需运行旧 sparse 管线, 从这里独立导入。

活跃主线: model.dense.DensePCNet (见 model/model_cyrene.py dense 门面)。
"""

from .cyrene import CyreneConfig, CyreneModel, LMHead, create_cyrene

__all__ = ["CyreneModel", "CyreneConfig", "LMHead", "create_cyrene"]