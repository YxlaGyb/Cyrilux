"""CUDA Stream 管理 — 多流并行支持.

ponytail: 先留接口不实现。encode/predict/hebbian 三阶段用同一默认流。
等 profile 确认瓶颈在流等待后再开启 stream pool.
"""

from __future__ import annotations

import torch


class StreamPool:
    """CUDA stream 池 (stub — 当前使用默认流).

    Args:
        n_streams: stream 数量
    """

    def __init__(self, n_streams: int = 3):
        self.n_streams = n_streams
        self._streams: list[torch.cuda.Stream] | None = None

    def _ensure(self):
        if self._streams is None and torch.cuda.is_available():
            self._streams = [torch.cuda.Stream() for _ in range(self.n_streams)]

    @property
    def encode(self) -> torch.cuda.Stream | None:
        self._ensure()
        return self._streams[0] if self._streams else None

    @property
    def predict(self) -> torch.cuda.Stream | None:
        self._ensure()
        return self._streams[1] if self._streams and len(self._streams) > 1 else None

    @property
    def hebbian(self) -> torch.cuda.Stream | None:
        self._ensure()
        return self._streams[2] if self._streams and len(self._streams) > 2 else None

    def synchronize(self):
        """等待所有 stream 完成."""
        if self._streams:
            for s in self._streams:
                s.synchronize()
