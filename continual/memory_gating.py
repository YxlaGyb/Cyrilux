"""
自主记忆门控 (Autonomous Memory Gating)

三级经验价值评估，让系统自主决定存/删/巩固哪些经验:

  Level 1 — 即时门控:  这条经验值得存吗？
  Level 2 — 保留门控:  这条经验该保留多久？
  Level 3 — 巩固门控:  这条经验值得做 abstraction 吗？

与现有机制打配合:
  - Level 1 → MemoryBank.add_samples() 前置过滤
  - Level 2 → MemoryBank FIFO 淘汰改为 priority × decay 排序
  - Level 3 → AbstractionBank.add_z_samples() 的 min_required_value
  - 自适应阈值: 每小时 (1000 步) 自动校准
"""
from __future__ import annotations

import math
from typing import Optional


class MemoryGate:
    """三级记忆门控。"""

    def __init__(
        self,
        threshold_low: float = 0.05,
        threshold_high: float = 0.5,
        target_storage_ratio: float = 0.30,
        target_high_value_ratio: float = 0.10,
        decay_base: float = 0.05,
        adaptation_window: int = 1000,
    ):
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.target_storage_ratio = target_storage_ratio
        self.target_high_value_ratio = target_high_value_ratio
        self.decay_base = decay_base
        self.adaptation_window = adaptation_window

        # 滚动统计 (用于自适应)
        self._recent_intrinsic_values: list[float] = []
        self._n_stored: int = 0
        self._n_total: int = 0
        self._n_high_value: int = 0
        self._steps_since_adapt: int = 0

    # ── Level 1: 即时门控 ──

    def should_store(self, intrinsic_value: float) -> bool:
        """判断经验是否值得存储。"""
        self._n_total += 1
        self._recent_intrinsic_values.append(intrinsic_value)
        if len(self._recent_intrinsic_values) > self.adaptation_window:
            self._recent_intrinsic_values.pop(0)

        if intrinsic_value >= self.threshold_high:
            self._n_stored += 1
            self._n_high_value += 1
            return True
        elif intrinsic_value >= self.threshold_low:
            self._n_stored += 1
            return True
        else:
            return False

    # ── Level 2: 保留门控 ──

    def compute_retention_decay(self, intrinsic_value: float, time_since_stored: int) -> float:
        """保留衰减因子: 低价值 + 久远 = 更快遗忘。

        Returns:
            [0, 1] 衰减系数, 1.0 = 完全保留, 0.0 = 立即淘汰
        """
        return math.exp(-self.decay_base * (1.0 / max(intrinsic_value, 0.01)) * time_since_stored)

    def compute_replay_priority(self, intrinsic_value: float, dopamine_score: float, age: int) -> float:
        """综合回放优先级 = 内在价值 × 多巴胺 × 衰减。"""
        decay = self.compute_retention_decay(intrinsic_value, age)
        return decay * (intrinsic_value + 0.1) * (dopamine_score + 0.1)

    # ── Level 3: 巩固门控 ──

    def should_consolidate(self, intrinsic_value: float) -> bool:
        """判断经验是否值得进入 AbstractionBank。"""
        return intrinsic_value >= self.threshold_high

    # ── 自适应阈值 ──

    def adapt_thresholds(self, force: bool = False) -> bool:
        """自适应调整阈值使存储率接近 target_storage_ratio。

        Returns:
            True 如果阈值被调整了
        """
        self._steps_since_adapt += 1
        if not force and self._steps_since_adapt < self.adaptation_window:
            return False
        self._steps_since_adapt = 0

        if self._n_total < 10:
            return False

        storage_ratio = self._n_stored / max(self._n_total, 1)
        high_value_ratio = self._n_high_value / max(self._n_total, 1)

        old_low = self.threshold_low
        old_high = self.threshold_high

        # 调整 threshold_low 使存储率接近 target
        if storage_ratio > self.target_storage_ratio * 1.5:
            self.threshold_low *= 1.1  # 提高阈值 → 存更少
        elif storage_ratio < self.target_storage_ratio * 0.5:
            self.threshold_low *= 0.9  # 降低阈值 → 存更多

        # 调整 threshold_high 使高价值率接近 target
        if high_value_ratio > self.target_high_value_ratio * 1.5:
            self.threshold_high *= 1.1
        elif high_value_ratio < self.target_high_value_ratio * 0.5 and self._n_high_value > 0:
            self.threshold_high *= 0.9

        # 重置统计
        self._n_stored = 0
        self._n_total = 0
        self._n_high_value = 0

        return old_low != self.threshold_low or old_high != self.threshold_high

    # ── 统计 ──

    def get_stats(self) -> dict:
        return {
            'threshold_low': self.threshold_low,
            'threshold_high': self.threshold_high,
            'storage_ratio': self._n_stored / max(self._n_total, 1) if self._n_total > 0 else 0.0,
            'n_stored': self._n_stored,
            'n_total': self._n_total,
            'n_high_value': self._n_high_value,
        }

    def state_dict(self) -> dict:
        return {
            'threshold_low': self.threshold_low,
            'threshold_high': self.threshold_high,
        }

    def load_state_dict(self, state: dict):
        self.threshold_low = state.get('threshold_low', self.threshold_low)
        self.threshold_high = state.get('threshold_high', self.threshold_high)
