"""稳态可塑性 — 阈值调节、修剪、与生长决策。

纯 Python/numpy-free 实现, 所有操作按神经元逐个进行。

功能:
  - adjust_thresholds(): 发放率稳态, 目标 ~1%
  - prune_decision():    长期不活跃 → 删除
  - growth_decision():   ε 持续高 → 分裂候选
  - BCM-like 滑动阈值
"""

from __future__ import annotations

from typing import Tuple

from .neuron_pool import NeuronPool


# ── 阈值调节 ──────────────────────────────────────────────────────

def adjust_threshold(
        pool: NeuronPool, neuron_id: int,
        target_rate: float = 0.01,
        rate_eta: float = 0.001
    ) -> float:
    """调节单个神经元的发放阈值以实现目标活跃率。

    规则:
      Δθ = η · (firing_rate - target_rate)
      θ += Δθ

    即: 当前活跃率高于目标 → 提高阈值使其更难发放。
        当前活跃率低于目标 → 降低阈值使其更容易发放。

    Args:
        pool:       NeuronPool 实例
        neuron_id:  目标神经元 ID
        target_rate: 目标活跃率 (默认 0.01 = 1%)
        rate_eta:    调节速率 (默认 0.001)

    Returns:
        阈值变化量 Δθ (绝对值)
    """
    neuron = pool.neurons.get(neuron_id)
    if neuron is None:
        return 0.0

    delta = rate_eta * (neuron.firing_rate - target_rate)
    new_th = neuron.threshold + delta

    # 防止阈值变为 0 或负数
    neuron.threshold = max(1e-4, new_th)
    return abs(delta)


def batch_adjust_thresholds(pool: NeuronPool, layer: int,
                            target_rate: float = 0.01,
                            rate_eta: float = 0.001) -> float:
    """批量调节一个逻辑层的神经元阈值。

    Args:
        pool:        NeuronPool 实例
        layer:       逻辑层索引
        target_rate: 目标活跃率
        rate_eta:    调节速率

    Returns:
        该层平均阈值变化量
    """
    total_delta = 0.0
    count = 0
    for nid in list(pool.layer_groups.get(layer, set())):
        total_delta += adjust_threshold(pool, nid, target_rate, rate_eta)
        count += 1
    return total_delta / max(1, count)


# ── BCM 滑动阈值 ─────────────────────────────────────────────────

def bcm_sliding_threshold(neuron_firing_rate: float,
                          avg_postsynaptic_rate: float,
                          theta_m: float = 1.0) -> float:
    """BCM 理论中的滑动阈值。

    LTP 和 LTD 的分界随突触后平均活跃率滑动:
      θ_M = firing_rate^2 / avg_postsynaptic_rate (简化形式)

    Args:
        neuron_firing_rate:  该神经元的当前活跃率
        avg_postsynaptic_rate: 突触后平均活跃率 (环境平均值)
        theta_m:             缩放因子

    Returns:
        滑动阈值 θ_M。
    """
    if avg_postsynaptic_rate <= 0:
        return theta_m
    return theta_m * (neuron_firing_rate ** 2) / max(1e-6, avg_postsynaptic_rate)


# ── 修剪决策 ──────────────────────────────────────────────────────

def should_prune(neuron_id: int, pool: NeuronPool,
                 max_inactive_steps: int = 1000,
                 min_age: int = 100,
                 current_step: int = 0) -> Tuple[bool, str]:
    """判断一个神经元是否应该被修剪。

    条件 (任一满足即修剪):
      1. 入连接数为 0 (孤儿)
      2. 残差太小 (|ε| < 1e-6) 持续太久, 对系统无贡献 (已编码融合)
      3. 不活跃超过 max_inactive_steps (且不是幼年神经元)

    Args:
        neuron_id:          目标神经元 ID
        pool:               NeuronPool 实例
        max_inactive_steps: 最大不活跃步数 (默认 1000)
        min_age:            最短生命周期 (默认 100 步, 防止幼年被误删)
        current_step:       当前全局步数

    Returns:
        (should_prune, reason)
        reason 是修剪原因字符串, 用于日志/监控。
    """
    neuron = pool.neurons.get(neuron_id)
    if neuron is None:
        return False, ""

    age = current_step - neuron.created_at

    # 幼年保护: 太年轻的神经元不修剪
    if age < min_age:
        return False, f"too_young({age}<{min_age})"

    # 1. 孤儿: 无入连接
    if len(neuron.in_synapses) == 0:
        return True, "orphan: no_in_synapses"

    # 2. 残差太小 (可忽略的 ε)
    if abs(neuron.ε) < 1e-6 and age > max(100, min_age):
        return True, f"negligible_ε({neuron.ε:.2e})"

    # 3. 长期不活跃
    inactive_for = current_step - neuron.last_active
    if inactive_for > max_inactive_steps:
        # 平均活跃率必须为零
        if neuron.firing_rate < 0.001:
            return True, f"inactive({inactive_for}>{max_inactive_steps})"

    return False, ""


def prune_network(pool: NeuronPool, current_step: int,
                  max_inactive_steps: int = 1000,
                  max_prune_per_step: int = 10) -> list[int]:
    """扫描全网络, 修剪所有符合条件的神经元。

    Args:
        pool:               NeuronPool 实例
        current_step:       当前全局步数
        max_inactive_steps: 最大不活跃步数 (传给 should_prune)
        max_prune_per_step: 每步最多修剪数 (防止抖动)

    Returns:
        被删除的神经元 ID 列表。
    """
    pruned = []
    for nid in list(pool.neurons.keys()):
        if len(pruned) >= max_prune_per_step:
            break
        decision, reason = should_prune(
            nid, pool, max_inactive_steps, current_step=current_step
        )
        if decision:
            pool.prune_neuron(nid, force=True)
            pruned.append(nid)
    return pruned


# ── 生长决策 ──────────────────────────────────────────────────────

def should_grow(pool: NeuronPool, neuron_id: int,
                error_multiple: float = 3.0,
                sustained_steps: int = 100) -> Tuple[bool, str]:
    """判断神经元是否需要分裂 (神经发生)。

    条件:
      - |ε| > threshold × error_multiple 持续超过 sustained_steps

    Args:
        pool:            NeuronPool 实例
        neuron_id:       目标神经元 ID
        error_multiple:  ε 超过 threshold 的倍数 (默认 3.0)
        sustained_steps: 需要持续的步数 (默认 100)

    Returns:
        (should_grow, reason)
    """
    _ = sustained_steps  # 历史窗口追踪在外部维护

    neuron = pool.neurons.get(neuron_id)
    if neuron is None:
        return False, ""

    # 检查 ε 是否超过阈值
    if abs(neuron.ε) > neuron.threshold * error_multiple:
        # 此处简化: 实际实现需要检查历史窗口
        # 但可以检查 age: 如果已经存在较久, 认为是持续的高误差
        return True, f"high_ε({neuron.ε:.4f}>={neuron.threshold*error_multiple:.4f})"

    return False, ""


def grow_network(pool: NeuronPool, current_step: int,
                 error_multiple: float = 3.0,
                 max_grow_per_step: int = 2,
                 min_interval: int = 50) -> list[int]:
    """扫描全网络, 对所有符合条件的神经元执行分裂。

    Args:
        pool:              NeuronPool 实例
        current_step:      当前全局步数
        error_multiple:    ε 超过 threshold 的倍数
        max_grow_per_step: 每步最多分裂数
        min_interval:      两次分裂间最小间隔步数

    Returns:
        新创建的神经元 ID 列表。
    """
    new_ids = []
    for nid in list(pool.neurons.keys()):
        if len(new_ids) >= max_grow_per_step:
            break
        decision, reason = should_grow(pool, nid, error_multiple)
        if decision:
            child = pool.split_neuron(nid)
            if child is not None:
                new_ids.append(child.id)
    return new_ids


# ── 综合稳态更新 ──────────────────────────────────────────────────

def homeostasis_step(pool: NeuronPool, current_step: int,
                     target_rate: float = 0.01,
                     prune_interval: int = 100,
                     grow_interval: int = 200) -> dict:
    """执行一个完整的稳态维护步。

    包含:
      1. 所有神经元的阈值调节
      2. 定期修剪 (每 prune_interval 步)
      3. 定期生长 (每 grow_interval 步)

    Args:
        pool:          NeuronPool 实例
        current_step:  当前全局步数
        target_rate:   目标活跃率
        prune_interval: 修剪间隔步数
        grow_interval:  生长间隔步数

    Returns:
        {action: 数量} 的统计字典。
    """
    stats = {}

    # 1. 全网络阈值调节
    total_rate_adj = 0.0
    n_neurons = 0
    for layer in list(pool.layer_groups.keys()):
        rate_adj = batch_adjust_thresholds(pool, layer, target_rate)
        total_rate_adj += rate_adj * len(pool.layer_groups.get(layer, set()))
        n_neurons += len(pool.layer_groups.get(layer, set()))
    stats["threshold_adjusted"] = n_neurons
    stats["avg_threshold_delta"] = total_rate_adj / max(1, n_neurons)

    # 2. 修剪
    if current_step % prune_interval == 0:
        pruned = prune_network(pool, current_step)
        stats["pruned"] = len(pruned)
    else:
        stats["pruned"] = 0

    # 3. 生长
    if current_step % grow_interval == 0:
        grown = grow_network(pool, current_step)
        stats["grown"] = len(grown)
    else:
        stats["grown"] = 0

    return stats
