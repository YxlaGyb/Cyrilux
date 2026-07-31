"""稀疏侧神经调制

依赖 pool 的精度调度.
"""

from __future__ import annotations

from .tensor_pool import TensorNeuronPool


def compute_precision_scales(
    pool: TensorNeuronPool,
    D: float,
    ACh: float,
    eta: float = 1.0,
) -> None:
    """逐神经元精度权重: pi = 1 + eta*D*|eps| + eta*ACh*|eps| (批量 tensor)."""
    pool.learning.compute_precision_scales(D, ACh, eta)
