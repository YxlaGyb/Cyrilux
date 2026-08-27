"""LearningEngine

密集 PPA Hebbian 学习 (零反传, 零误差回路).

机制:
- 前馈权重: 逐层预测误差驱动 (标准 PC 自下而上), L3 加随机增益+门控种子
- 微柱 W_35: 块内 BCM 滑阈 + 样本显著性加权 + 增益/剪切/门控掩码
- W_diff: 增量预测 (dz5 = z5[t]-z5[t-1]), 多尺度软窗 + 4 步时间窗 + 独立 BCM
- 时序 W_t: 共现 Hebbian, 静止帧掩码 + homeostatic 列增益
- 内建能量约束 (第 78 轮): 全部可塑性权重施加真 Oja + 活动依赖遗忘项
  (_energy_constraint, 逐输出单元, 纯局部, 零全局统计)
- 学习率: 恒基准值 (lr_hebbian), 无全局调制 (第 78 轮裁决: 无上帝之手)

全 fp16, 零 .float(), 零 autograd.

解耦 (2026-08-27): 原单文件 1475 行按学习域拆为子包 —
engine.py (learn 编排) / _common.py (纯函数) / feedforward.py / temporal.py /
readout.py / predict.py / bind.py / action.py. 行为与原 learning.py 逐位等价.

兼容导出: LearningEngine + 模块级函数 (_activity_baseline/_energy_constraint/
_elig_accum/_spectral_radius_guard) + LM_TRUST_REGION (外部 tests/scripts 引用).
"""

from __future__ import annotations

from ._common import (
    ELIG_GAMMA,
    LM_TRUST_REGION,
    _activity_baseline,
    _decorr_W,
    _elig_accum,
    _energy_constraint,
    _rho_ctrl,
    _spectral_radius_guard,
)
from .engine import EngineCore

__all__ = [
    "LearningEngine",
    "LM_TRUST_REGION",
    "ELIG_GAMMA",
    "_activity_baseline",
    "_decorr_W",
    "_elig_accum",
    "_energy_constraint",
    "_rho_ctrl",
    "_spectral_radius_guard",
]


class LearningEngine(EngineCore):
    """学习引擎: 持 net 引用, 复用 _predict 存的 _z* 状态."""
