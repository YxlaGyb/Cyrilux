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


def compute_prune_mask(
    pool: TensorNeuronPool,
    current_step: int,
    max_inactive: int = 1000,
    min_age: int = 100,
) -> torch.Tensor:
    """计算应修剪的神经元掩码 (惰性删除决策).

    Returns:
        [N] bool mask.
    """
    alive = pool.alive
    age = current_step - pool.created_at
    inactive_for = current_step - pool.last_active

    orphan = torch.zeros(pool.N, dtype=torch.bool, device=pool.device)
    negligible = (pool.state[:, 2].abs() < 1e-6) & (age > 100) & alive
    inactive = (inactive_for > max_inactive) & (pool.state[:, 4] < 0.001) & alive
    too_young = age < min_age

    return (orphan | negligible | inactive) & ~too_young & alive
