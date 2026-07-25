"""NeuronPool — 动态神经元对象池。

数据结构层: Neuron, Synapse, NeuronPool, SensoryEvent, NetworkEvent。
纯 Python + dataclass, 不依赖 torch。

核心设计:
  - 没有 hidden_size 超参数。神经元数量由活动数据动态决定。
  - 每个神经元有 ~H/r 个入连接（稀疏）, 不是全连接。
  - 层概念保留为逻辑分组 (layer_groups), 但层宽不固定。
  - 所有操作 O(1) 或 O(活跃神经元), 不遍历全部。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ── 突触连接 ──────────────────────────────────────────────────────

@dataclass
class Synapse:
    """单个突触连接。

    每个突触连接一个前神经元和一个后神经元, 用标量权重存储连接强度。
    age 用于修剪决策, trace 用于 STDP 追踪, is_active 用于惰性删除。
    """
    id: int
    pre_id: int
    post_id: int
    weight: float       # fp16-compatible 标量
    age: int = 0
    trace: float = 0.0  # 脉冲追踪 (STDP eligibility trace)
    is_active: bool = True


# ── 神经元 ────────────────────────────────────────────────────────

@dataclass
class Neuron:
    """单个神经元。

    不是张量的一行 —— 每个神经元有独立的标量状态和稀疏连接。
    layer 决定其在 PC 层级中的逻辑位置, 层宽由该层神经元数量决定。

    z:          膜电位 (活动值)
    μ:          当前入连接对 z 的预测
    ε:          预测误差 z - μ
    threshold:  自适应发放阈值 (|ε| > threshold → 发放)
    firing_rate: EMA 活跃率, 用于稳态可塑性
    """
    id: int
    layer: int                    # 逻辑层索引 (0=sensory, 1..=hidden)
    z: float = 0.0                # 膜电位 (标量)
    μ: float = 0.0                # 预测值
    ε: float = 0.0                # 预测误差
    threshold: float = 0.1        # 自适应发放阈值
    firing_rate: float = 0.0      # EMA 活跃率
    out_synapses: list[int] = field(default_factory=list)   # 发出连接 ID 列表
    in_synapses: list[int] = field(default_factory=list)    # 接收连接 ID 列表
    created_at: int = 0           # 创建时的全局步数
    last_active: int = 0          # 最后一次发放的步数
    position: int = -1            # 对于 sensory neurons: 序列位置
    channel: int = -1             # 对于 sensory neurons: 特征通道
    π: float = 1.0                # 精度权重, 由多巴胺/乙酰胆碱调制
    _z_prev: float = 0.0          # 上一步的 z (用于 temporal 预测)


# ── 事件 ──────────────────────────────────────────────────────────

@dataclass(order=True)
class SensoryEvent:
    """来自 GPU 感官前端的事件，按时间排序。"""
    time: int
    block_id: int = 0       # 0..5 (6 个 conv block)
    pos: int = 0            # 序列位置
    channel: int = 0        # 特征通道 (0..H_front-1)
    value: float = 0.0      # h_conv 的值
    layer: int = 0          # 目标 PC 层 (默认 0=sensory)


@dataclass(order=True)
class NetworkEvent:
    """网络内部神经元间事件, 按时间排序。"""
    time: int
    neuron_id: int = 0
    ε: float = 0.0
    source: str = ""        # 'bottom_up' | 'temporal' | 'topdown'
    value: float = 0.0      # 传播的激活值 (z 或 ε 的某种函数)
    target_layers: list[int] = field(default_factory=list)  # 传往哪些层


# ── 神经元池 ──────────────────────────────────────────────────────

class NeuronPool:
    """所有神经元的容器, 支持动态增删。

    神经元数量不固定, 由活动数据自动决定。
    不是 nn.ModuleList —— 因为神经元可以增减。

    Attributes:
        neurons:      id → Neuron
        synapses:     id → Synapse
        layer_groups: layer → set[id]  (逻辑层分组)
        next_id:     下一个可用 ID (自增)
        max_neurons: 软上限, 防止无限增长 (默认 1024)
    """

    def __init__(self, max_neurons: int = 8192):
        self.neurons: dict[int, Neuron] = {}
        self.synapses: dict[int, Synapse] = {}
        self.layer_groups: dict[int, set[int]] = {}
        self._next_id: int = 0
        self._next_syn_id: int = 0
        self.max_neurons = max_neurons

        # 统计
        self._total_created: int = 0
        self._total_pruned: int = 0
        self._total_synapses_created: int = 0
        self._total_synapses_pruned: int = 0

    # ── 神经元创建 ──

    def create_neuron(self, layer: int, **kwargs) -> Neuron:
        """创建新神经元并分配 ID。

        Args:
            layer: 逻辑层索引。0=感觉输入层, 1..=隐藏层。
            **kwargs: 传给 Neuron 构造函数的额外参数 (z, threshold 等)。

        Returns:
            创建的 Neuron 实例。

        Raises:
            RuntimeError: 超过 max_neurons 上限。
        """
        if len(self.neurons) >= self.max_neurons:
            # 尝试回收最久未活跃的神经元
            freed = self._force_recycle()
            if not freed:
                raise RuntimeError(
                    f"NeuronPool 已达上限 {self.max_neurons}, "
                    f"且无神经元可回收"
                )

        neuron_id = self._next_id
        self._next_id += 1
        neuron = Neuron(id=neuron_id, layer=layer, **kwargs)
        neuron.created_at = self._total_created

        self.neurons[neuron_id] = neuron
        if layer not in self.layer_groups:
            self.layer_groups[layer] = set()
        self.layer_groups[layer].add(neuron_id)
        self._total_created += 1
        return neuron

    def create_sensory_neurons(self, n_channels: int, n_positions: int,
                               layer: int = 0) -> list[Neuron]:
        """批量创建感觉神经元 (来自 GPU conv 输出)。

        每个 (channel, position) 组合对应一个感觉神经元。
        初始 threshold 设为 0.05 (低阈值, warmup 后自适应升高)。

        Args:
            n_channels: 特征通道数 (H_front)
            n_positions: 序列位置数
            layer: 感觉所在逻辑层 (默认 0)

        Returns:
            创建的 Neuron 列表。
        """
        created = []
        for pos in range(n_positions):
            for ch in range(n_channels):
                n = self.create_neuron(
                    layer=layer,
                    threshold=0.05,
                    position=pos,
                    channel=ch,
                )
                created.append(n)
        return created

    # ── 突触创建 ──

    def create_synapse(self, pre_id: int, post_id: int,
                       weight: Optional[float] = None) -> Synapse:
        """在两个已存在的神经元之间创建突触连接。

        Args:
            pre_id:  前神经元 ID
            post_id: 后神经元 ID
            weight:  初始权重。None=随机初始化 ~N(0, 1/sqrt(fan_in))。

        Returns:
            创建的 Synapse 实例。
        """
        assert pre_id in self.neurons, f"pre_id {pre_id} 不存在"
        assert post_id in self.neurons, f"post_id {post_id} 不存在"

        if weight is None:
            # Xavier-style 初始化
            pre = self.neurons[pre_id]
            fan_in = max(1, len(pre.in_synapses) + 1)
            weight = random.gauss(0, 1.0 / math.sqrt(fan_in))
            weight = max(-1.0, min(1.0, weight))  # 钳位防止极端值

        syn_id = self._next_syn_id
        self._next_syn_id += 1
        syn = Synapse(id=syn_id, pre_id=pre_id, post_id=post_id, weight=float(weight))

        self.synapses[syn_id] = syn
        self.neurons[pre_id].out_synapses.append(syn_id)
        self.neurons[post_id].in_synapses.append(syn_id)
        self._total_synapses_created += 1
        return syn

    def connect_layer(self, from_layer: int, to_layer: int,
                      connection_density: float = 0.5) -> int:
        """在两个逻辑层之间建立随机稀疏连接。

        Args:
            from_layer: 源层
            to_layer:   目标层
            connection_density: 连接密度 [0,1], 默认 0.5 (每个后神经元连 ~50% 源)

        Returns:
            创建的连接数。
        """
        from_ids = list(self.layer_groups.get(from_layer, set()))
        to_ids = list(self.layer_groups.get(to_layer, set()))
        if not from_ids or not to_ids:
            return 0

        count = 0
        for post_id in to_ids:
            # 随机选 ~density × |from| 个前神经元
            k = max(1, int(len(from_ids) * connection_density))
            sampled = random.sample(from_ids, min(k, len(from_ids)))
            for pre_id in sampled:
                self.create_synapse(pre_id, post_id)
                count += 1
        return count

    # ── 神经元分裂 (Neurogenesis) ──

    def split_neuron(self, neuron_id: int,
                     noise_scale: float = 0.05) -> Optional[Neuron]:
        """分裂一个高活跃神经元为两个。

        模拟生物神经发生: 当一个神经元长期高 ε, 分裂为两个
        子神经元, 继承父神经元的输入连接(各半)和 1/3 的输出连接。

        Args:
            neuron_id: 要分裂的神经元 ID
            noise_scale: 分裂后权重噪声比例

        Returns:
            新创建的 Neuron, 或 None (分裂失败)。
        """
        parent = self.neurons.get(neuron_id)
        if parent is None:
            return None

        # 创建子神经元, 与父神经元同层
        child = self.create_neuron(
            layer=parent.layer,
            threshold=parent.threshold,
            position=parent.position,
            channel=parent.channel,
            z=parent.z * 0.5,  # 膜电位减半
        )
        if child is None:
            return None

        # 平分输入连接
        in_ids = list(parent.in_synapses)
        random.shuffle(in_ids)
        mid = len(in_ids) // 2
        parent.in_synapses = in_ids[:mid] + parent.in_synapses[mid:]  # 保持完整性
        for syn_id in in_ids[mid:]:
            syn = self.synapses.get(syn_id)
            if syn is not None:
                # 将连接重定向到子神经元
                syn.post_id = child.id
                # 加噪声
                syn.weight *= (1.0 + random.gauss(0, noise_scale))
                syn.weight = max(-1.0, min(1.0, syn.weight))
                child.in_synapses.append(syn_id)

        # 继承 1/3 的输出连接 (加噪声)
        out_ids = list(parent.out_synapses)
        random.shuffle(out_ids)
        inherit_count = max(1, len(out_ids) // 3)
        for syn_id in out_ids[:inherit_count]:
            syn = self.synapses.get(syn_id)
            if syn is not None:
                # 克隆输出连接给子神经元
                child_out = self.create_synapse(
                    pre_id=child.id,
                    post_id=syn.post_id,
                    weight=syn.weight * (1.0 + random.gauss(0, noise_scale)),
                )
                if child_out.weight is not None:
                    child_out.weight = max(-1.0, min(1.0, child_out.weight))

        # 父神经元的 threshold 升高 (降低其活跃度)
        parent.threshold *= 1.1
        return child

    # ── 修剪 ──

    def prune_neuron(self, neuron_id: int, force: bool = False) -> bool:
        """标记神经元为死亡并清理其所有连接。

        Args:
            neuron_id: 要删除的神经元 ID
            force: 即使神经元有活跃连接也强制删除

        Returns:
            True=成功删除, False=神经元不存在。
        """
        neuron = self.neurons.pop(neuron_id, None)
        if neuron is None:
            return False

        # 从层组中移除
        layer_set = self.layer_groups.get(neuron.layer)
        if layer_set is not None:
            layer_set.discard(neuron_id)

        # 清理所有入连接
        for syn_id in list(neuron.in_synapses):
            self._remove_synapse(syn_id, from_post=True)

        # 清理所有出连接
        for syn_id in list(neuron.out_synapses):
            self._remove_synapse(syn_id, from_pre=True)

        self._total_pruned += 1
        return True

    def _remove_synapse(self, syn_id: int,
                        from_pre: bool = False,
                        from_post: bool = False) -> None:
        """从 pre/post 的列表中移除突触引用, 然后删除。

        生物类比: 突触修剪 —— 标记为不活跃而非立即物理删除,
        但为节省内存此处完全删除。
        """
        syn = self.synapses.pop(syn_id, None)
        if syn is None:
            return
        # 从 pre 的 out_synapses 中移除
        if from_pre:
            pre_neuron = self.neurons.get(syn.pre_id)
            if pre_neuron is not None and syn_id in pre_neuron.out_synapses:
                pre_neuron.out_synapses.remove(syn_id)
        # 从 post 的 in_synapses 中移除
        if from_post:
            post_neuron = self.neurons.get(syn.post_id)
            if post_neuron is not None and syn_id in post_neuron.in_synapses:
                post_neuron.in_synapses.remove(syn_id)
        self._total_synapses_pruned += 1

    def _force_recycle(self) -> bool:
        """当池满时, 回收长期未活跃的神经元。

        找 last_active 最小的神经元 (最久未发放), 删除它。

        Returns:
            True=成功回收至少一个, False=无神经元可回收。
        """
        if not self.neurons:
            return False
        # 找最久未活跃的
        oldest_id = min(self.neurons, key=lambda nid: self.neurons[nid].last_active)
        return self.prune_neuron(oldest_id, force=True)

    # ── 查询 ──

    def get_neurons_by_layer(self, layer: int) -> list[Neuron]:
        """返回指定逻辑层的所有神经元列表。"""
        ids = self.layer_groups.get(layer, set())
        return [self.neurons[nid] for nid in ids if nid in self.neurons]

    def get_layer_width(self, layer: int) -> int:
        """返回指定逻辑层的当前宽度 (神经元数量)。"""
        return len(self.layer_groups.get(layer, set()))

    def get_active_neurons(self, layer: int,
                           threshold: Optional[float] = None) -> list[Neuron]:
        """返回指定层中 |ε| > threshold 的活跃神经元。

        Args:
            layer: 逻辑层索引
            threshold: 活跃阈值。None = 使用各神经元的自适应阈值。

        Returns:
            活跃的 Neuron 列表。
        """
        active = []
        for nid in self.layer_groups.get(layer, set()):
            n = self.neurons.get(nid)
            if n is None:
                continue
            th = threshold if threshold is not None else n.threshold
            if abs(n.ε) > th:
                active.append(n)
        return active

    def get_total_neurons(self) -> int:
        """当前存活的神经元总数。"""
        return len(self.neurons)

    def get_total_synapses(self) -> int:
        """当前存活的突触总数。"""
        return len(self.synapses)

    def get_activity_stats(self) -> dict:
        """返回全网络的活跃统计。

        Returns:
            dict with keys:
              - total_neurons, total_synapses
              - layer_widths: {layer: count}
              - avg_firing_rate, avg_threshold
              - n_pruned, n_created
        """
        rates = [n.firing_rate for n in self.neurons.values()]
        thresholds = [n.threshold for n in self.neurons.values()]
        return {
            "total_neurons": self.get_total_neurons(),
            "total_synapses": self.get_total_synapses(),
            "layer_widths": {
                layer: len(ids)
                for layer, ids in self.layer_groups.items()
            },
            "avg_firing_rate": sum(rates) / len(rates) if rates else 0.0,
            "avg_threshold": sum(thresholds) / len(thresholds) if thresholds else 0.1,
            "n_created": self._total_created,
            "n_pruned": self._total_pruned,
            "n_syn_created": self._total_synapses_created,
            "n_syn_pruned": self._total_synapses_pruned,
        }

    def find_high_error_neurons(self, layer: int,
                                 error_multiple: float = 3.0) -> list[Neuron]:
        """找到指定层中 ε 超过 threshold × error_multiple 的神经元。

        这些神经元往往是分裂候选 (Phase E)。
        """
        candidates = []
        for n in self.get_neurons_by_layer(layer):
            if abs(n.ε) > n.threshold * error_multiple:
                candidates.append(n)
        return candidates
