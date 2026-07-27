"""PoolQuery — 只读查询与统计 (委托到 TensorNeuronPool 的张量)."""

from __future__ import annotations

import torch


class PoolQuery:
    """只读查询: 神经元/突触/层信息、活跃检测、感官事件匹配.

    通过 self.pool 引用访问共享张量, 扩容后自动指向新张量.
    """

    def __init__(self, pool):
        self.pool = pool

    def get_neurons_by_layer(self, layer: int) -> torch.Tensor:
        """返回指定层的神经元索引."""
        mask = (self.pool.layer == layer) & self.pool.alive
        return torch.where(mask)[0]

    def get_layer_width(self, layer: int) -> int:
        """返回指定层宽度."""
        return int(((self.pool.layer == layer) & self.pool.alive).sum().item())

    def get_active_neurons(self, layer: int | None = None) -> torch.Tensor:
        """返回 |epsilon| > threshold 的活跃神经元索引.

        Args:
            layer: 限定层 (None = 全网络)
        """
        active_mask = (
            self.pool.state[:, 2].abs() > self.pool.state[:, 3]
        ) & self.pool.alive  # F_EPS=2, F_THRESHOLD=3
        if layer is not None:
            active_mask &= self.pool.layer == layer
        return torch.where(active_mask)[0]

    def get_total_neurons(self) -> int:
        """存活神经元总数."""
        return int(self.pool.alive.sum().item())

    def get_total_synapses(self) -> int:
        """存活突触总数."""
        return int(self.pool.syn_alive.sum().item())

    def get_activity_stats(self) -> dict:
        """返回全网络活跃统计."""
        alive_mask = self.pool.alive
        n_alive = int(alive_mask.sum().item())
        if n_alive == 0:
            return {
                "total_neurons": 0,
                "total_synapses": 0,
                "layer_widths": {},
                "avg_firing_rate": 0.0,
                "avg_threshold": 0.1,
                "n_created": self.pool._total_created,
                "n_pruned": self.pool._total_pruned,
                "n_syn_created": self.pool._total_syn_created,
                "n_syn_pruned": self.pool._total_syn_pruned,
            }

        rates = self.pool.state[alive_mask, 4]  # F_FIRING_RATE=4
        thresholds = self.pool.state[alive_mask, 3]  # F_THRESHOLD=3
        layers = self.pool.layer[alive_mask]

        layer_widths = {}
        for L in layers.unique().tolist():
            layer_widths[int(L)] = int((layers == L).sum().item())

        return {
            "total_neurons": n_alive,
            "total_synapses": self.get_total_synapses(),
            "layer_widths": layer_widths,
            "avg_firing_rate": float(rates.mean().item()) if len(rates) > 0 else 0.0,
            "avg_threshold": float(thresholds.mean().item()) if len(thresholds) > 0 else 0.1,
            "n_created": self.pool._total_created,
            "n_pruned": self.pool._total_pruned,
            "n_syn_created": self.pool._total_syn_created,
            "n_syn_pruned": self.pool._total_syn_pruned,
        }

    def match_sensory_events(
        self,
        layers: list[int],
        positions: list[int],
        channels: list[int],
    ) -> tuple[list[int], list[int], list[int]]:
        """批量匹配感官事件到现有神经元 (O(1) hash 索引, 纯 Python).

        Args:
            layers: [E] 逻辑层
            positions: [E] 序列位置
            channels: [E] 特征通道

        Returns:
            (matched_nids, matched_event_indices, unmatched_event_indices)
            所有返回值均为 Python list.
        """
        E = len(layers)
        idx = self.pool._sensory_index

        matched_nids: list[int] = []
        matched_ev_idx: list[int] = []
        unmatched_ev_idx: list[int] = []

        for e in range(E):
            key = (layers[e], positions[e], channels[e])
            nid = idx.get(key)
            if nid is not None:
                matched_nids.append(nid)
                matched_ev_idx.append(e)
            else:
                unmatched_ev_idx.append(e)

        return matched_nids, matched_ev_idx, unmatched_ev_idx
