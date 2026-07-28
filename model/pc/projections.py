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
        td_slice = slice(0, self.pool._td_count)
        if self.pool._td_count > 0:
            mask = (
                (self.pool.td_pre[td_slice] == pre)
                & (self.pool.td_post[td_slice] == post)
                & self.pool.td_alive[td_slice]
            )
            found = torch.where(mask)[0]
            if len(found) > 0:
                return int(found[0].item())
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
        """上层所有神经元 → 下层所有神经元的批量 topdown 连接."""
        upper = self.pool.query.get_neurons_by_layer(upper_layer)
        lower = self.pool.query.get_neurons_by_layer(lower_layer)
        if len(upper) == 0 or len(lower) == 0:
            return 0

        n_upper = len(upper)
        n_lower = len(lower)
        k = max(1, int(n_lower * density))
        n_conn = n_upper * k

        # 预分配连接批量
        pre_list = upper.repeat_interleave(k).tolist()
        post_list = lower[torch.randint(0, n_lower, (n_conn,))].tolist()
        w_list = [random.gauss(0, 0.05) for _ in range(n_conn)]

        # 找到空闲的 td 槽位
        start = self.pool._td_count
        need = n_conn
        if start + need <= self.pool.S_td:
            idx_t = torch.arange(start, start + need, device=self.pool.device)
            self.pool._td_count = start + need
        else:
            dead = torch.where(~self.pool.td_alive)[0]
            n_dead = len(dead)
            if n_dead < need:
                need = n_dead
            idx_t = dead[:need]
            # 优先用连续的死槽位
            if need != n_conn:
                pre_list = pre_list[:need]
                post_list = post_list[:need]
                w_list = w_list[:need]

        self.pool.td_pre[idx_t] = torch.tensor(pre_list, dtype=torch.int32, device=self.pool.device)
        self.pool.td_post[idx_t] = torch.tensor(post_list, dtype=torch.int32, device=self.pool.device)
        self.pool.td_weight[idx_t] = torch.tensor(w_list, dtype=torch.float16, device=self.pool.device)
        self.pool.td_alive[idx_t] = True
        return need

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
