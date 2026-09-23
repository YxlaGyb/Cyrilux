"""TensorRingBuffer — GPU 侧固定大小环形缓冲.

替代旧 EventBridge 中的 deque[SensoryEvent] Python 对象队列.
单生产者单消费者, 无锁设计.

ponytail: 当前 EventBridge 直接用 tensor slice 管理,
此模块预留用于未来异步多流场景.
"""

from __future__ import annotations

import torch


class TensorRingBuffer:
    """固定大小 GPU tensor 环形缓冲.

    Args:
        capacity: 最大条目数
        shape: 每条目的形状 (不含批次维)
        dtype: 数据类型
        device: 存储设备
    """

    def __init__(
        self, capacity: int, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ):
        self.capacity = capacity
        self.buffer = torch.zeros(capacity, *shape, dtype=dtype, device=device)
        self._write: int = 0
        self._count: int = 0
        self.device = device

    def push(self, data: torch.Tensor) -> int:
        """推入数据, 返回实际写入数."""
        n = data.shape[0]
        if n == 0:
            return 0

        # 截断
        available = self.capacity - self._count
        n = min(n, available)
        if n <= 0:
            return 0

        end = min(self._write + n, self.capacity)
        space = end - self._write
        self.buffer[self._write : end] = data[:space]
        if n > space:
            self.buffer[: n - space] = data[space:n]

        self._write = (self._write + n) % self.capacity
        self._count += n
        return n

    def pop(self, n: int) -> torch.Tensor:
        """消费 n 条数据."""
        n = min(n, self._count)
        if n == 0:
            return torch.zeros(
                0, *self.buffer.shape[1:], dtype=self.buffer.dtype, device=self.device
            )

        read_start = (self._write - self._count) % self.capacity
        read_end = read_start + n

        if read_end <= self.capacity:
            result = self.buffer[read_start:read_end].clone()
        else:
            first = self.capacity - read_start
            result = torch.cat([self.buffer[read_start:], self.buffer[: n - first]])

        self._count -= n
        return result

    def __len__(self) -> int:
        return self._count
