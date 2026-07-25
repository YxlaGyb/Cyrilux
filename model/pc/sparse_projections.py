"""稀疏投影模块 — temporal self-connection, topdown, lm_head。

所有投影以神经元对象为单位, 不使用密集矩阵乘法。
每个投影类型维护一个轻量权重字典。

设计:
  - SparseTemporalSelf: 每个神经元有可选的"自连接"权重, 用于时序预测
  - SparseTopdown: 高→低层的自上而下预测, 仅在活跃神经元之间建立
  - SparseLMHead: 256 个固定输出神经元, 从顶层接收稀疏连接
"""

from __future__ import annotations

import math
import random
from typing import Optional

from .neuron_pool import NeuronPool


# ── 时序自连接 ───────────────────────────────────────────────────


class SparseTemporalSelf:
    """时序自连接 — 每个神经元对其自身前一步活动值的线性投影。

    替代旧的密集 `Linear(H, H)` temporal_proj。
    权重存储为 `dict[neuron_id, float]`, 仅对已分配的神经元存在。

    Biology: 神经元的自连接或循环连接, 维持时间上下文。

    Args:
        init_scale: 初始权重缩放 (默认 0.1, 弱时序依赖起步)
    """

    def __init__(self, init_scale: float = 0.1):
        self._weights: dict[int, float] = {}
        self.init_scale = init_scale

    def connect(self, neuron_id: int) -> bool:
        """为一个神经元分配初始 temporal 权重。

        Args:
            neuron_id: 神经元 ID

        Returns:
            True=成功分配, False=已存在。
        """
        if neuron_id in self._weights:
            return False
        self._weights[neuron_id] = random.gauss(0, self.init_scale)
        return True

    def connect_batch(self, neuron_ids: list[int]) -> int:
        """批量为神经元分配 temporal 权重。"""
        count = 0
        for nid in neuron_ids:
            if self.connect(nid):
                count += 1
        return count

    def predict(self, neuron_id: int, z_prev: float) -> Optional[float]:
        """时序预测: z_temp = W_temp · z(t-1)。

        Args:
            neuron_id: 目标神经元 ID
            z_prev: 该神经元上一步的活动值 (z(t-1))

        Returns:
            时序预测值, 或 None (如果该神经元未连接)。
        """
        w = self._weights.get(neuron_id)
        if w is None:
            return None
        return _to_fp16(w * z_prev)

    def hebbian_step(
        self,
        neuron_id: int,
        z_prev: float,
        epsilon: float,
        eta: float = 1e-4,
        dopamine: float = 1.0,
    ) -> float:
        """时序权重的 Hebbian 更新: Δw = η · D · ε · z(t-1) - α · w。

        Args:
            neuron_id: 目标神经元 ID
            z_prev: z(t-1)
            epsilon: 当前 ε (z - μ)
            eta: 学习率 (默认 1e-4, 慢于感官 Hebbian)
            dopamine: 多巴胺调制

        Returns:
            |Δw| (监控用)
        """
        w = self._weights.get(neuron_id)
        if w is None:
            return 0.0
        dw = eta * dopamine * epsilon * z_prev - 0.01 * w  # 带 weight decay
        self._weights[neuron_id] = _to_fp16(w + dw)
        return abs(dw)

    def disconnect(self, neuron_id: int):
        """移除一个神经元的 temporal 权重 (修剪时调用)。"""
        self._weights.pop(neuron_id, None)

    def __len__(self) -> int:
        return len(self._weights)

    def __contains__(self, neuron_id: int) -> bool:
        return neuron_id in self._weights

    def get_stats(self) -> dict:
        ws = list(self._weights.values())
        return {
            "n_connected": len(ws),
            "mean_w": sum(ws) / len(ws) if ws else 0.0,
            "max_w": max(ws) if ws else 0.0,
        }


# ── 自上而下投影 ─────────────────────────────────────────────────


class SparseTopdown:
    """自上而下投影 — 层 ℓ+1 → 层 ℓ 的稀疏预测连接。

    不同于密集 topdown_proj(H, H), 这里只在活跃神经元之间建立连接。
    连接密度动态: 有多少活跃的上层神经元 × 每个连多少个下层神经元。

    Args:
        connection_density: 每个上层活跃神经元连接的下层比例 (默认 0.3)
        init_scale: 初始权重缩放 (默认 0.05, 弱预测起步)
    """

    def __init__(self, connection_density: float = 0.3, init_scale: float = 0.05):
        self.connection_density = connection_density
        self.init_scale = init_scale
        # (pre_id, post_id) → weight
        self._weights: dict[tuple[int, int], float] = {}

    def ensure_connection(self, pre_id: int, post_id: int) -> float:
        """确保两个神经元间存在 topdown 连接, 没有则创建。

        Args:
            pre_id:  上层神经元 ID (预测者)
            post_id: 下层神经元 ID (被预测者)

        Returns:
            当前权重值。
        """
        key = (pre_id, post_id)
        if key not in self._weights:
            self._weights[key] = random.gauss(0, self.init_scale)
        return self._weights[key]

    def connect_active(
        self,
        pool: NeuronPool,
        upper_layer: int,
        lower_layer: int,
        threshold_multiple: float = 1.0,
        max_per_upper: int = 8,
    ) -> int:
        """在上层活跃神经元和下层活跃神经元之间建立 topdown 连接。

        Args:
            pool: NeuronPool 实例
            upper_layer: 上层索引
            lower_layer: 下层索引
            threshold_multiple: 活跃判定乘数
            max_per_upper: 每个上层神经元最多连几个下层

        Returns:
            新建连接数。
        """
        upper_neurons = pool.get_neurons_by_layer(upper_layer)
        lower_neurons = pool.get_neurons_by_layer(lower_layer)

        if not upper_neurons or not lower_neurons:
            return 0

        count = 0
        for upper in upper_neurons:
            if abs(upper.ε) <= upper.threshold * threshold_multiple:
                continue
            # 随机采样 max_per_upper 个下层神经元
            candidates = random.sample(
                lower_neurons,
                min(max_per_upper, len(lower_neurons)),
            )
            for lower in candidates:
                self.ensure_connection(upper.id, lower.id)
                count += 1

        return count

    def predict(self, pool: NeuronPool, lower_neuron_id: int) -> float:
        """从所有连接到该下层神经元的上层节点计算 topdown 预测。

        Args:
            pool: NeuronPool
            lower_neuron_id: 下层神经元 ID

        Returns:
            μ_topdown (标量)。
        """
        μ_td = 0.0
        for (pre_id, post_id), w in self._weights.items():
            if post_id != lower_neuron_id:
                continue
            pre = pool.neurons.get(pre_id)
            if pre is not None:
                μ_td += w * pre.z
        return _to_fp16(μ_td)

    def hebbian_step(
        self,
        pool: NeuronPool,
        lower_neuron_id: int,
        epsilon: float,
        eta: float = 1e-4,
        dopamine: float = 1.0,
    ) -> float:
        """topdown 权重的 Hebbian 更新。

        Args:
            pool: NeuronPool
            lower_neuron_id: 下层神经元 ID
            epsilon: 下层神经元的 ε
            eta: 学习率
            dopamine: 多巴胺调制

        Returns:
            Σ|Δw|
        """
        total_delta = 0.0
        for (pre_id, post_id), w in list(self._weights.items()):
            if post_id != lower_neuron_id:
                continue
            pre = pool.neurons.get(pre_id)
            if pre is None:
                continue
            dw = eta * dopamine * epsilon * pre.z
            self._weights[(pre_id, post_id)] = _to_fp16(w + dw)
            total_delta += abs(dw)
        return total_delta

    def disconnect_neuron(self, neuron_id: int):
        """移除与该神经元相关的所有 topdown 连接 (修剪时调用)。"""
        keys = [k for k in self._weights if k[0] == neuron_id or k[1] == neuron_id]
        for k in keys:
            del self._weights[k]

    def __len__(self) -> int:
        return len(self._weights)

    def get_stats(self) -> dict:
        ws = list(self._weights.values())
        return {
            "n_connections": len(ws),
            "mean_w": sum(ws) / len(ws) if ws else 0.0,
        }


# ── LM Head ──────────────────────────────────────────────────────


class SparseLMHead:
    """稀疏输出投影 — 256 个字节 logits 从顶层稀疏读出。

    替代旧的密集 `Linear(H, 256)` lm_head。
    每个输出字节对应一个 logit neuron, 从顶层活跃神经元接收稀疏连接。

    当顶层神经元数量变化时, lm_head 的输入维度自动适配。
    这就是"自动容量" —— 不需要重新配置模型。

    Args:
        connections_per_logit: 每个 logit 连接的顶层神经元数 (默认 16)
        init_scale: 初始权重缩放 (默认 0.02)
    """

    def __init__(self, connections_per_logit: int = 16, init_scale: float = 0.02):
        self.connections_per_logit = connections_per_logit
        self.init_scale = init_scale
        # (logit_idx, neuron_id) → weight
        self._weights: dict[tuple[int, int], float] = {}
        self._initialized_logits: set[int] = set()

    def ensure_top_layer_connected(self, pool: NeuronPool, top_layer: int) -> int:
        """确保顶层所有神经元已连接到 lm_head 的某些 logit。

        每个顶层神经元随机连接到 connections_per_logit 个 logits。
        每个 logit 有 ~min(connections_per_logit, n_top) 个输入。

        Args:
            pool: NeuronPool
            top_layer: 顶层索引 (最高隐藏层)

        Returns:
            新建连接数。
        """
        top_ids = list(pool.layer_groups.get(top_layer, set()))
        if not top_ids:
            return 0

        count = 0
        for nid in top_ids:
            # 检查该神经元是否已有连接到 lm_head
            connected_logits = {logit for (logit, neuron) in self._weights if neuron == nid}
            if connected_logits:
                continue  # 已有连接

            # 随机连接到 connections_per_logit 个 logits
            for logit_idx in random.sample(
                range(256),
                min(self.connections_per_logit, 256),
            ):
                key = (logit_idx, nid)
                if key not in self._weights:
                    self._weights[key] = random.gauss(0, self.init_scale)
                    count += 1
        return count

    def predict_logits(self, pool: NeuronPool, top_layer: int) -> list[float]:
        """计算 256 个字节的 logits。

        每个 logit = Σ W[logit, neuron] · z_neuron。
        只遍历有连接的神经元, 不遍历全量。

        Args:
            pool: NeuronPool
            top_layer: 顶层索引

        Returns:
            [256] logits 列表 (标量 fp16)。
        """
        logits = [0.0] * 256

        # 遍历所有连接
        for (logit_idx, nid), w in self._weights.items():
            n = pool.neurons.get(nid)
            if n is not None:
                logits[logit_idx] += w * n.z

        return [_to_fp16(v) for v in logits]

    def cross_entropy_loss(self, logits: list[float], target_byte: int) -> float:
        """计算交叉熵损失 (给定目标字节)。

        Args:
            logits: [256] float
            target_byte: 0..255

        Returns:
            CE loss (标量)。
        """
        # softmax 手工计算 (数值稳定版)
        max_logit = max(logits)
        shifted = [v - max_logit for v in logits]
        exp_sum = sum(math.exp(v) for v in shifted)
        log_probs = [v - math.log(exp_sum) for v in shifted]
        return -log_probs[target_byte]

    def hebbian_step(
        self,
        pool: NeuronPool,
        top_layer: int,
        logits: list[float],
        target_byte: int,
        eta: float = 1e-4,
        dopamine: float = 1.0,
    ) -> float:
        """lm_head 权重的 Hebbian 更新。

        使用 logit-gap 信号: 对 target_byte 的连接做 Hebbian 增强,
        对其他字节的连接做轻微衰减 (Oja-like 约束)。

        Args:
            pool: NeuronPool
            top_layer: 顶层索引
            logits: [256] logit 值
            target_byte: 目标字节
            eta: 学习率
            dopamine: 多巴胺调制

        Returns:
            Σ|Δw|
        """
        total_delta = 0.0
        η_eff = eta * dopamine

        for (logit_idx, nid), w in list(self._weights.items()):
            n = pool.neurons.get(nid)
            if n is None:
                continue

            # 目标字节: 增强连接; 非目标: 抑制
            if logit_idx == target_byte:
                dw = η_eff * n.z  # Hebbian 增长
            else:
                dw = -η_eff * 0.01 * w  # 轻微衰减

            self._weights[(logit_idx, nid)] = _to_fp16(w + dw)
            total_delta += abs(dw)

        return total_delta

    def logit_gap_loss(self, logits: list[float], target_byte: int, margin: float = 2.0) -> float:
        """Logit-gap loss: max(0, margin - (logit_target - max_other))。

        鼓励目标字节的 logit 至少比其他高 margin。
        """
        target = logits[target_byte]
        other_max = max(logits[i] for i in range(256) if i != target_byte)
        gap = target - other_max
        return max(0.0, margin - gap)

    def disconnect_neuron(self, neuron_id: int):
        """移除与某神经元相关的 lm_head 连接。"""
        keys = [k for k in self._weights if k[1] == neuron_id]
        for k in keys:
            del self._weights[k]

    def __len__(self) -> int:
        return len(self._weights)

    def get_stats(self) -> dict:
        ws = list(self._weights.values())
        # 每个 logit 的输入数
        import collections

        logit_counts = collections.Counter(k[0] for k in self._weights)
        return {
            "n_connections": len(ws),
            "mean_w": sum(ws) / len(ws) if ws else 0.0,
            "avg_connections_per_logit": (sum(logit_counts.values()) / max(1, len(logit_counts))),
        }


# ── 工具 ──────────────────────────────────────────────────────────


def _to_fp16(v: float) -> float:
    """模拟 fp16 精度截断。"""
    if v == 0.0 or not math.isfinite(v):
        return 0.0
    v = max(-65504.0, min(65504.0, v))
    digits = -int(math.floor(math.log10(abs(v)))) + 3
    return round(v, max(0, digits))
