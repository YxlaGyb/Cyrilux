"""稀疏前向计算 — 事件驱动的神经元级推理与学习。

所有操作按神经元逐个进行, 不涉及密集矩阵乘法。
每个神经元仅遍历其 ~32 个入连接, 复杂度 O(活跃神经元 × 连接度)。

核心函数:
  - predict_neuron():  从入连接权重 × 前活动值 计算 μ
  - update_neuron():   新的外部输入 z, 重新计算 ε
  - hebbian_step():    单步 Hebbian + Oja 规则更新
  - emit_events():     生成 NetworkEvent 供下游消费

FP16 约定:
  - 所有权重 (Synapse.weight) 以 fp16 标量存储
  - 计算时使用 Python float, 写回时截断为 fp16 范围
"""

from __future__ import annotations

import math
from typing import Optional

from .neuron_pool import NeuronPool, SensoryEvent, NetworkEvent


# ── 预测 ──────────────────────────────────────────────────────────

def predict_neuron(pool: NeuronPool, neuron_id: int) -> float:
    """计算单个神经元的预测值 `μ = Σ w_ij · z_j`。

    只遍历活跃的入连接 (is_active=True), 忽略已修剪的连接。

    Args:
        pool:      NeuronPool 实例
        neuron_id: 目标神经元 ID

    Returns:
        预测值 μ (标量 fp16)。

    Raises:
        KeyError: neuron_id 不存在
    """
    neuron = pool.neurons.get(neuron_id)
    if neuron is None:
        raise KeyError(f"neuron_id {neuron_id} 不存在")

    μ = 0.0
    n_active = 0
    for syn_id in neuron.in_synapses:
        syn = pool.synapses.get(syn_id)
        if syn is None or not syn.is_active:
            continue
        pre = pool.neurons.get(syn.pre_id)
        if pre is None:
            continue
        μ += syn.weight * pre.z
        n_active += 1

    # μ 更新后存入神经元
    neuron.μ = _to_fp16(μ)
    return neuron.μ


def batch_predict(pool: NeuronPool, neuron_ids: list[int]) -> dict[int, float]:
    """批量预测多个神经元的 μ。

    比逐次调用 predict_neuron() 更高效 (减少 dict lookup 次数)。

    Args:
        pool:        NeuronPool 实例
        neuron_ids:  目标神经元 ID 列表

    Returns:
        {neuron_id: μ}
    """
    results = {}
    for nid in neuron_ids:
        results[nid] = predict_neuron(pool, nid)
    return results


def predict_layer(pool: NeuronPool, layer: int) -> dict[int, float]:
    """计算指定逻辑层全部神经元的 μ。

    Args:
        pool:  NeuronPool 实例
        layer: 逻辑层索引

    Returns:
        {neuron_id: μ}
    """
    ids = list(pool.layer_groups.get(layer, set()))
    return batch_predict(pool, ids)


# ── 更新 ──────────────────────────────────────────────────────────

def update_neuron(pool: NeuronPool, neuron_id: int,
                  z_new: Optional[float] = None) -> float:
    """更新神经元的 z 并重新计算 ε = z - μ。

    这是 PC 推理的核心步骤: 新感知进入 → z 更新 → ε 变化 → 向上/下传播。

    Args:
        pool:      NeuronPool 实例
        neuron_id: 目标神经元 ID
        z_new:     新的 z 值。None = 使用当前 z (仅重算 ε)。

    Returns:
        新的 ε (标量 fp16)。
    """
    neuron = pool.neurons.get(neuron_id)
    if neuron is None:
        raise KeyError(f"neuron_id {neuron_id} 不存在")

    if z_new is not None:
        neuron.z = _to_fp16(z_new)

    # 确保 μ 最新
    predict_neuron(pool, neuron_id)

    neuron.ε = _to_fp16(neuron.z - neuron.μ)
    return neuron.ε


def process_sensory_event(pool: NeuronPool, event: SensoryEvent) -> dict[int, float]:
    """处理一个 SensoryEvent: 找到对应感觉神经元并更新其 z。

    如果神经元不存在则自动创建。

    Args:
        pool:  NeuronPool 实例
        event: 来自 GPU conv 的 SensoryEvent

    Returns:
        {neuron_id: ε} 所有受影响的神经元的 ε
    """
    # 找匹配的感觉神经元
    target_id = None
    for n in pool.get_neurons_by_layer(event.layer):
        if n.position == event.pos and n.channel == event.channel:
            target_id = n.id
            break

    if target_id is None:
        # 自动创建新的感觉神经元
        new_n = pool.create_neuron(
            layer=event.layer,
            position=event.pos,
            channel=event.channel,
            threshold=0.05,
        )
        target_id = new_n.id

    ε = update_neuron(pool, target_id, z_new=event.value)
    return {target_id: ε}


# ── Hebbian 更新 ─────────────────────────────────────────────────

def hebbian_step(pool: NeuronPool, neuron_id: int,
                 eta: float = 3e-4,
                 oja_alpha: float = 0.05,
                 dopamine: float = 1.0) -> float:
    """对目标神经元的入连接执行 Hebbian 更新。

    规则:
      Δw = η · D · ε · z_pre - α · D · ε² · w  (Oja 约束)

    Args:
        pool:      NeuronPool 实例
        neuron_id: 目标神经元 ID
        eta:       Hebbian 学习率 (基本率)
        oja_alpha: Oja 归一化强度
        dopamine:  多巴胺调制系数 D (default 1.0)

    Returns:
        总权重变化量 Σ|Δw|, 用于监控。
    """
    neuron = pool.neurons.get(neuron_id)
    if neuron is None:
        raise KeyError(f"neuron_id {neuron_id} 不存在")

    weight_delta = 0.0
    π = getattr(neuron, 'π', 1.0)
    η_eff = eta * dopamine * π

    for syn_id in neuron.in_synapses:
        syn = pool.synapses.get(syn_id)
        if syn is None or not syn.is_active:
            continue
        pre = pool.neurons.get(syn.pre_id)
        if pre is None:
            continue

        # Hebbian: Δw = η · D · ε · z_pre
        hebb = η_eff * neuron.ε * pre.z

        # Oja: -α · D · ε² · w
        oja = -oja_alpha * dopamine * (neuron.ε ** 2) * syn.weight

        dw = hebb + oja
        syn.weight = _to_fp16(syn.weight + dw)
        syn.age += 1

        # 更新 STDP trace
        syn.trace = _to_fp16(syn.trace * 0.9 + abs(pre.z) * 0.1)

        weight_delta += abs(dw)

    return weight_delta


def batch_hebbian(pool: NeuronPool, neuron_ids: list[int],
                  eta: float = 3e-4,
                  oja_alpha: float = 0.05,
                  dopamine: float = 1.0) -> float:
    """批量 Hebbian 更新。

    Args:
        pool:       NeuronPool 实例
        neuron_ids: 目标神经元 ID 列表
        eta:        Hebbian 学习率
        oja_alpha:  Oja 归一化强度
        dopamine:   多巴胺调制系数

    Returns:
        总绝对权重变化量 (监控用)
    """
    total_delta = 0.0
    for nid in neuron_ids:
        total_delta += hebbian_step(pool, nid, eta, oja_alpha, dopamine)
    return total_delta


# ── 发放与事件传播 ────────────────────────────────────────────────

def emit_if_active(pool: NeuronPool, neuron_id: int,
                   current_time: int,
                   activation_multiple: float = 1.0) -> Optional[NetworkEvent]:
    """如果 |ε| > threshold, 发出 NetworkEvent。

    Args:
        pool:                NeuronPool 实例
        neuron_id:           目标神经元 ID
        current_time:        当前时间步
        activation_multiple: 活跃阈值的乘数因子 (默认 1.0)

    Returns:
        如果活跃则返回 NetworkEvent, 否则 None。
    """
    neuron = pool.neurons.get(neuron_id)
    if neuron is None:
        return None

    firing_threshold = neuron.threshold * activation_multiple
    if abs(neuron.ε) <= firing_threshold:
        # 更新 firing rate (指数衰减, 无事件时不更新也可)
        return None

    # 标记活跃
    neuron.last_active = current_time
    # EMA firing rate
    neuron.firing_rate = _to_fp16(neuron.firing_rate * 0.95 + 0.05)

    # 创建 NetworkEvent
    event = NetworkEvent(
        time=current_time,
        neuron_id=neuron_id,
        ε=neuron.ε,
        value=neuron.z,
        target_layers=[neuron.layer + 1],   # 默认向上传播
    )
    return event


def emit_active_neurons(pool: NeuronPool, layer: int,
                        current_time: int) -> list[NetworkEvent]:
    """扫描指定层, 收集所有活跃神经元的事件。

    Args:
        pool:         NeuronPool 实例
        layer:        逻辑层索引
        current_time: 当前时间步

    Returns:
        NetworkEvent 列表 (仅含活跃神经元)。
    """
    events = []
    for nid in list(pool.layer_groups.get(layer, set())):
        event = emit_if_active(pool, nid, current_time)
        if event is not None:
            events.append(event)
    return events


# ── 工具 ──────────────────────────────────────────────────────────

def _to_fp16(v: float) -> float:
    """模拟 fp16 精度截断。

    实际运行中不需要此函数，权重本身就在 fp16 中。
    此处为纯 Python 实现提供精度边界。

    使用 round() 模拟 fp16 的 ~3.3 位有效数字精度。
    """
    # f16 有 10 位尾数 → ~log10(2^10) ≈ 3.01 位有效数字
    if v == 0.0 or not math.isfinite(v):
        return 0.0
    # 钳位到 fp16 范围 [-65504, 65504]
    v = max(-65504.0, min(65504.0, v))
    # 舍入模拟精度损失
    digits = -int(math.floor(math.log10(abs(v)))) + 3
    return round(v, max(0, digits))
