"""稳态可塑性 — 张量化阈值调节、修剪与生长决策.

基于 TensorNeuronPool 的批量 tensor 操作, 替代旧的逐神经元 Python 循环.
"""

from __future__ import annotations

import torch

from .tensor_pool import TensorNeuronPool


def homeostasis_step(
    pool: TensorNeuronPool,
    current_step: int,
    target_rate: float = 0.01,
    prune_interval: int = 100,
    grow_interval: int = 200,
) -> dict:
    """执行完整稳态维护步 (委托到 TensorNeuronPool)."""
    return pool.homeostasis_step(
        current_step=current_step,
        target_rate=target_rate,
        prune_interval=prune_interval,
        grow_interval=grow_interval,
    )


# compute_prune_mask 已删除 — 死代码，从未被调用。
# 剪枝逻辑在 TensorNeuronPool.homeostasis_step() 内部。
