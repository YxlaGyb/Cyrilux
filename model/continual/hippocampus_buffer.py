"""
海马体缓冲 (Hippocampus Buffer) — 快速循环缓冲 + 信息增益优先保留。

生物动机:
  海马体临时快速编码新经验, 随后逐步整合到新皮层。
  此处实现一个轻量级"快速写入 + 基于信息增益的优先保留"缓冲,
  独立于 ConsolidationPipeline (慢速长期存储)。

核心机制:
  - FIFO 环形缓冲, 容量较小 (默认 200)
  - 新样本按信息增益阈值过滤后才加入
  - 缓冲满时自动驱逐最低信息增益的样本
  - 支持按优先级采样用于快速回放

用法 (在 training.py 的 train_step 中):
  # 每步添加高信息增益样本
  if info_gain > min_threshold:
      hippocampus.add(z_states, byte_seq, labels, info_gain)

  # 定期回放
  if step % replay_interval == 0:
      batch = hippocampus.sample_for_replay(batch_size)
      # → 对 batch 执行一次 Hebbian update
"""

from __future__ import annotations

import torch


class HippocampusEntry:
    """单条海马体条目。"""

    __slots__ = ("z_states", "byte_tensor", "label_tensor", "info_gain", "step")

    def __init__(
        self,
        z_states: list[torch.Tensor],
        byte_tensor: torch.Tensor,
        label_tensor: torch.Tensor,
        info_gain: float,
        step: int,
    ):
        self.z_states = z_states
        self.byte_tensor = byte_tensor
        self.label_tensor = label_tensor
        self.info_gain = info_gain
        self.step = step


class HippocampusBuffer:
    """海马体缓冲 — 快速循环缓冲 + 信息增益优先保留。

    Args:
        capacity: 最大条目数 (默认 200)
        min_info_gain: 最小信息增益阈值 (默认 0.03), 低于此值丢弃
    """

    def __init__(self, capacity: int = 200, min_info_gain: float = 0.03):
        self.capacity = capacity
        self.min_info_gain = min_info_gain
        self._buffer: list[HippocampusEntry] = []
        self._step_counter: int = 0

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) >= self.capacity

    def add(
        self,
        z_states: list[torch.Tensor],
        byte_tensor: torch.Tensor,
        label_tensor: torch.Tensor,
        info_gain: float,
        step: int | None = None,
    ):
        """添加新条目。低信息增益直接丢弃; 满时驱逐最低增益条目。"""
        # 低增益过滤
        if info_gain < self.min_info_gain:
            return

        if step is None:
            step = self._step_counter
        self._step_counter += 1

        entry = HippocampusEntry(
            z_states=[z.detach().cpu() for z in z_states],
            byte_tensor=byte_tensor.cpu(),
            label_tensor=label_tensor.cpu(),
            info_gain=info_gain,
            step=step,
        )

        if not self.is_full:
            self._buffer.append(entry)
        else:
            # 驱逐最低信息增益的条目
            min_idx = 0
            min_ig = self._buffer[0].info_gain
            for i, e in enumerate(self._buffer):
                if e.info_gain < min_ig:
                    min_ig = e.info_gain
                    min_idx = i

            # 仅当新条目增益 > 最低增益时才替换
            if info_gain > min_ig:
                self._buffer[min_idx] = entry

    def sample_for_replay(
        self, n: int, device: str = "cuda:0"
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """按信息增益加权采样 N 条条目, 返回 (byte_tensor, label_tensor) batch。

        Returns:
            (byte_batch, label_batch) 或 None (缓冲为空)
        """
        if not self._buffer:
            return None

        n = min(n, len(self._buffer))

        # 信息增益加权采样
        gains = torch.tensor(
            [max(e.info_gain, self.min_info_gain) for e in self._buffer],
            dtype=torch.float16,
        )
        weights = gains / (gains.sum() + 1e-8)

        if len(self._buffer) <= n:
            indices = list(range(len(self._buffer)))
        else:
            indices = torch.multinomial(weights, n, replacement=False).tolist()

        batch_bytes = []
        batch_labels = []
        for idx in indices:
            e = self._buffer[idx]
            batch_bytes.append(e.byte_tensor.unsqueeze(0).to(device))
            batch_labels.append(e.label_tensor.unsqueeze(0).to(device))

        return (torch.cat(batch_bytes, dim=0), torch.cat(batch_labels, dim=0))

    def clear(self):
        """清空缓冲。"""
        self._buffer.clear()
        self._step_counter = 0

    def state_dict(self) -> dict:
        """序列化 (不保存 z_states, 仅保存统计)。"""
        return {
            "capacity": self.capacity,
            "min_info_gain": self.min_info_gain,
            "n_entries": len(self._buffer),
            "step_counter": self._step_counter,
        }

    def load_state_dict(self, state: dict):
        """恢复统计状态 (缓冲内容重置)。"""
        self.capacity = state.get("capacity", self.capacity)
        self.min_info_gain = state.get("min_info_gain", self.min_info_gain)
        self._step_counter = state.get("step_counter", 0)
        # 不恢复具体条目 (冷启动)
