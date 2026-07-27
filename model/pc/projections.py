"""ProjectionManager — 时序/topdown/LM head 结构连接."""

from __future__ import annotations

import random

import torch


class ProjectionManager:
    """投影连接管理: 时序自连接、topdown 连接、LM head 连接.

    通过 self.pool 引用访问共享张量, 扩容后自动指向新张量.
    """

    def __init__(self, pool):
        self.pool = pool

    def temporal_connect(self, nids: list[int], init_scale: float = 0.1):
        """为神经元建立时序自连接."""
        for nid in nids:
            self.pool.t_connected[nid] = True
            self.pool.t_weight[nid] = random.gauss(0, init_scale)

    def temporal_disconnect(self, nid: int):
        """移除时序连接."""
        self.pool.t_connected[nid] = False
        self.pool.t_weight[nid] = 0.0

    def topdown_ensure(self, pre: int, post: int, init_scale: float = 0.05) -> int:
        """确保两神经元间有 topdown 连接, 返回连接索引."""
        for i in range(self.pool._td_count):
            if (
                self.pool.td_alive[i]
                and int(self.pool.td_pre[i].item()) == pre
                and int(self.pool.td_post[i].item()) == post
            ):
                return i
        idx = self.pool._td_count
        if idx >= self.pool.S_td:
            dead = torch.where(~self.pool.td_alive)[0]
            if len(dead) > 0:
                idx = int(dead[0].item())
            else:
                return -1
        self.pool.td_pre[idx] = pre
        self.pool.td_post[idx] = post
        self.pool.td_weight[idx] = random.gauss(0, init_scale)
        self.pool.td_alive[idx] = True
        if idx == self.pool._td_count:
            self.pool._td_count += 1
        return idx

    def topdown_connect_layer(
        self, upper_layer: int, lower_layer: int, density: float = 0.2
    ) -> int:
        """上层所有神经元 → 下层所有神经元的 topdown 连接 (批量, 免扫描)."""
        upper = self.pool.query.get_neurons_by_layer(upper_layer).tolist()
        lower = self.pool.query.get_neurons_by_layer(lower_layer).tolist()
        if not upper or not lower:
            return 0

        count = 0
        k = max(1, int(len(lower) * density))
        for pre in upper:
            candidates = random.sample(lower, min(k, len(lower)))
            for post in candidates:
                idx = self.pool._td_count
                if idx >= self.pool.S_td:
                    dead = torch.where(~self.pool.td_alive)[0]
                    if len(dead) > 0:
                        idx = int(dead[0].item())
                    else:
                        continue
                self.pool.td_pre[idx] = pre
                self.pool.td_post[idx] = post
                self.pool.td_weight[idx] = random.gauss(0, 0.05)
                self.pool.td_alive[idx] = True
                if idx == self.pool._td_count:
                    self.pool._td_count += 1
                count += 1
        return count

    def topdown_connect_active(
        self,
        upper_layer: int,
        lower_layer: int,
        max_per_upper: int = 8,
    ) -> int:
        """上层活跃神经元 → 下层神经元的 topdown 连接."""
        upper = self.pool.query.get_active_neurons(upper_layer).tolist()
        lower = self.pool.query.get_neurons_by_layer(lower_layer).tolist()
        if not upper or not lower:
            return 0

        count = 0
        for pre in upper:
            candidates = random.sample(lower, min(max_per_upper, len(lower)))
            for post in candidates:
                self.topdown_ensure(pre, post)
                count += 1
        return count

    def lm_ensure_top_connected(self, top_layer: int, connections_per_logit: int = 16):
        """确保顶层神经元已连接到 LM head."""
        top_mask = (self.pool.layer == top_layer) & self.pool.alive
        top_ids = torch.where(top_mask)[0].tolist()
        if not top_ids:
            return

        for nid in top_ids:
            if self.pool.lm_weight[:, nid].abs().sum() > 0.001:
                continue
            # 全连接: 每个 L5 神经元连到所有 256 个 logit (零初始化, Hebbian 从零起步)
            self.pool.lm_weight[:, nid] = torch.zeros(
                256, dtype=torch.float16, device=self.pool.device
            )
