"""
遗忘嗅探器 — 检测旧任务 CE loss 超标 → 自触发修复.

原理:
  每隔 check_interval 步, 从 MemoryBank 采样 N 条 exemplars 跑纯前向 CE loss.
  如果某任务的当前 loss 超过 baseline_loss × threshold, 进入修复模式:
    - LR 降至 repair_lr_factor × current_lr
    - 强制回放 repair_steps 步旧数据
    - 直到所有任务的 loss_ratio < threshold

Ponytail: 嗅探器只做 T=0 纯前向 — 无 PC 推理, 开销约 = 1 步训练.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn

from model.continual.memory_bank import MemoryBank


class ForgettingSniffer:
    """遗忘嗅探 + 自触发修复.

    支持两级遗忘检测:
      - Task 级: 原逻辑, 按 task_id 分组
      - Concept 级: 按 concept_id 分组, 由 ICM composite signal 增强
    """

    def __init__(
        self,
        memory_bank: MemoryBank,
        model: nn.Module,
        check_interval: int = 200,
        threshold: float = 1.2,
        repair_steps: int = 10,
        repair_lr_factor: float = 0.3,
        eval_n: int = 32,
        # 动态间隔参数
        enable_dynamic_interval: bool = True,
        min_interval: int = 50,
        max_interval: int = 500,
        surprise_threshold: float = 0.25,
        surprise_window: int = 20,
        # 概念级遗忘参数
        enable_concept_detection: bool = True,
        concept_threshold: float = 1.5,
    ):
        self.memory_bank = memory_bank
        self.model = model
        self.check_interval = check_interval
        self.threshold = threshold
        self.repair_steps = repair_steps
        self.repair_lr_factor = repair_lr_factor
        self.eval_n = eval_n
        self._repairing = False
        self._repair_counter = 0
        self._last_ratios: dict[str, float] = {}
        self._last_concept_ratios: dict[str, float] = {}
        # 动态间隔
        self.enable_dynamic_interval = enable_dynamic_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.surprise_threshold = surprise_threshold
        self._surprise_buffer: list[float] = []
        self._surprise_window = surprise_window
        self._effective_interval = check_interval
        # 概念级遗忘
        self.enable_concept_detection = enable_concept_detection
        self.concept_threshold = concept_threshold

    # ── 属性 ──────────────────────────────────────────────────────────

    @property
    def is_repairing(self) -> bool:
        return self._repairing

    @property
    def last_ratios(self) -> dict:
        return dict(self._last_ratios)

    @property
    def last_concept_ratios(self) -> dict:
        return dict(self._last_concept_ratios)

    @property
    def effective_interval(self) -> int:
        return self._effective_interval

    # ── 世界模型 surprise / ICM 复合信号 ──────────────────────────

    def update_surprise(self, surprise: float, icm_signal: dict | None = None):
        """每步接收 surprise 以动态调整嗅探频率.

        Args:
            surprise: 预测误差 (world model or ICM forward_pred_loss)
            icm_signal: ICM 复合信号 dict (来自 IntrinsicCuriosityModule.forward())
                        — 支持 information_gain, uncertainty, prediction_error
        """
        # 使用 ICM information_gain 增强 surprise 信号
        if icm_signal is not None:
            info_gain = icm_signal.get('information_gain', 0.0)
            uncertainty = icm_signal.get('uncertainty', 0.0)
            surprise = surprise + info_gain * 0.5 + uncertainty * 0.3

        self._surprise_buffer.append(surprise)
        if len(self._surprise_buffer) > self._surprise_window:
            self._surprise_buffer.pop(0)
        if not self.enable_dynamic_interval or len(self._surprise_buffer) < 5:
            return
        recent_high = sum(1 for s in self._surprise_buffer if s > self.surprise_threshold)
        ratio = recent_high / len(self._surprise_buffer)
        if ratio > 0.6:
            self._effective_interval = max(self.min_interval, self.check_interval // 2)
        elif ratio < 0.2:
            self._effective_interval = min(self.max_interval, int(self.check_interval * 1.5))
        else:
            self._effective_interval = self.check_interval

    # ── 核心 ──────────────────────────────────────────────────────────

    def check(self, global_step: int, device: str) -> Optional[List[str]]:
        """嗅探: 检测是否有任务被遗忘.

        每 check_interval 步触发一次.
        如果正在 repair 中, 每次都会检测是否恢复.

        Returns:
            如果触发遗忘, 返回遗忘任务 ID 列表; 否则 None.
        """
        if self.memory_bank.total == 0:
            return None

        # 修复模式下, 每步都检测
        if not self._repairing and global_step % self._effective_interval != 0:
            return None

        results = self.memory_bank.evaluate(self.model, device, N=self.eval_n)
        self._last_ratios = {tid: r['ratio'] for tid, r in results.items()}

        forgotten = [tid for tid, r in results.items() if r['ratio'] > self.threshold]
        return forgotten if forgotten else None

    def check_concept(self, global_step: int, device: str, concept_ids: list[str]) -> list[str] | None:
        """概念级遗忘检测 — 按概念检测 CE loss 变化.

        Returns:
            遗忘的概念 ID 列表 (ratio > concept_threshold)
        """
        if not self.enable_concept_detection or not concept_ids:
            return None
        results = self.memory_bank.evaluate(self.model, device, N=self.eval_n)
        # 按 concept_id 聚合
        concept_results: dict[str, list[float]] = {}
        for tid, r in results.items():
            cid = tid  # 简化: task_id 即 concept_id (实际需 memory_bank 按 concept 分组)
            concept_results.setdefault(cid, []).append(r['ratio'])
        ratios = {}
        for cid, ratios_list in concept_results.items():
            ratios[cid] = sum(ratios_list) / max(len(ratios_list), 1)
        self._last_concept_ratios = ratios
        forgotten_c = [cid for cid, r in ratios.items() if r > self.concept_threshold]
        return forgotten_c if forgotten_c else None

    def repair_begin(self, optimizer, current_lr: float, device: str) -> float:
        """进入修复模式: 降低 LR, 准备强制回放.

        Returns: 修复用的 LR
        """
        self._repairing = True
        self._repair_counter = 0
        repair_lr = current_lr * self.repair_lr_factor
        for pg in optimizer.param_groups:
            pg['lr'] = repair_lr
        return repair_lr

    def repair_end(self, optimizer, restore_lr: float):
        """退出修复模式: 恢复 LR."""
        self._repairing = False
        self._repair_counter = 0
        for pg in optimizer.param_groups:
            pg['lr'] = restore_lr

    def repair_step(self) -> bool:
        """执行一步修复计数器. 返回 True 表示还需继续修复."""
        if not self._repairing:
            return False
        self._repair_counter += 1
        if self._repair_counter >= self.repair_steps:
            self._repairing = False
            return False
        return True

    def get_replay_batch(self, batch_size: int, device: str, strategy: str = 'dopamine'):
        """获取用于修复回放的 batch.

        Args:
            strategy: MemoryBank.sample() 策略 ('dopamine', 'world_model' 等)

        Returns: (byte_seq, labels) 或 None (bank 为空时)
        """
        if self.memory_bank.total == 0:
            return None
        exemplars = self.memory_bank.sample(batch_size, strategy=strategy)
        if not exemplars:
            return None
        byte_seq = torch.stack([ex.byte_tensor for ex in exemplars], dim=0).to(device)
        labels = torch.stack([ex.label_tensor for ex in exemplars], dim=0).to(device)
        return byte_seq, labels

    # ── 序列化 ────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            '_repairing': self._repairing,
            '_repair_counter': self._repair_counter,
            '_last_ratios': dict(self._last_ratios),
            '_last_concept_ratios': dict(self._last_concept_ratios),
        }

    def load_state_dict(self, state: dict):
        self._repairing = state.get('_repairing', False)
        self._repair_counter = state.get('_repair_counter', 0)
        self._last_ratios = state.get('_last_ratios', {})
        self._last_concept_ratios = state.get('_last_concept_ratios', {})
