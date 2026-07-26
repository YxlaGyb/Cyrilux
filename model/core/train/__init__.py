"""训练子包.

Public API:
    TrainingLoop    训练循环入口
    TrainingConfig  统一训练配置
    ProgressCallback 训练进度回调类型
"""

from .config import ProgressCallback, TrainingConfig
from .loop import TrainingLoop

__all__ = [
    "TrainingLoop",
    "TrainingConfig",
    "ProgressCallback",
]
