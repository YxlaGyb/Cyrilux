"""
有效宽度对齐纯算子

按 active 宽度取前缀视图:
    宿主只传递全量权重与宽度标量,
    对齐本身是图内算子, 逐步路径的张量工作全部有图内来源
"""

from __future__ import annotations

import torch


def align_rows(w: torch.Tensor, n: int) -> torch.Tensor:
    """首维取前缀视图 [n, ...]."""
    return w[:n]


def align_square(m: torch.Tensor, n: int) -> torch.Tensor:
    """首两维取前缀视图 [n, n]."""
    return m[:n, :n]


def align_mem_window(m_seq: torch.Tensor, n: int, s: int, k: int, a: int) -> torch.Tensor:
    """记忆序列窗口视图 [n, s, k, a]."""
    return m_seq[:n, :s, :k, :a]


def pack3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """短时可塑性三通道打包."""
    return (a, b, c)
