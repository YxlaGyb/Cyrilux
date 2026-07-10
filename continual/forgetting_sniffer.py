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

from continual.memory_bank import MemoryBank


class ForgettingSniffer:
    """遗忘嗅探 + 自触发修复."""

    def __init__(
        self,
        memory_bank: MemoryBank,
        model: nn.Module,
        check_interval: int = 200,
        threshold: float = 1.2,
        repair_steps: int = 10,
        repair_lr_factor: float = 0.3,
        eval_n: int = 32,
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

    # ── 属性 ──────────────────────────────────────────────────────────

    @property
    def is_repairing(self) -> bool:
        return self._repairing

    @property
    def last_ratios(self) -> dict:
        return dict(self._last_ratios)

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
        if not self._repairing and global_step % self.check_interval != 0:
            return None

        results = self.memory_bank.evaluate(self.model, device, N=self.eval_n)
        self._last_ratios = {tid: r['ratio'] for tid, r in results.items()}

        forgotten = [tid for tid, r in results.items() if r['ratio'] > self.threshold]
        return forgotten if forgotten else None

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

    def get_replay_batch(self, batch_size: int, device: str):
        """获取用于修复回放的 batch.

        Returns: (byte_seq, labels) 或 None (bank 为空时)
        """
        if self.memory_bank.total == 0:
            return None
        exemplars = self.memory_bank.sample(batch_size, strategy='dopamine')
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
        }

    def load_state_dict(self, state: dict):
        self._repairing = state.get('_repairing', False)
        self._repair_counter = state.get('_repair_counter', 0)
        self._last_ratios = state.get('_last_ratios', {})
