"""
SynapseManager

突触生命周期管理 (创建/批量创建/层连接/周转).
"""

from __future__ import annotations

import math
import random

import torch

from ..constants import CONN_FEEDBACK, CONN_FEEDFORWARD, LAYER_L4, LAYER_SENSORY
from .page_storage import MemoryBudgetError


class SynapseManager:
    """
    突触 CRUD: 创建、批量创建、层连接、突触周转.

    通过 self.pool 引用访问共享张量, 扩容后自动指向新张量.
    """

    def __init__(self, pool):
        self.pool = pool

    def create_synapse(self, pre: int, post: int, weight: float | None = None) -> int:
        """在两个神经元间创建突触."""
        try:
            sid = self.pool._storage.alloc_synapse_slot()
        except MemoryBudgetError:
            dead = torch.where(~self.pool.syn_alive)[0]
            if len(dead) == 0:
                raise RuntimeError("TensorNeuronPool 突触内存预算耗尽, 无死槽位可回收")
            sid = int(dead[0].item())
        self.pool._sync_storage_refs()

        if weight is None:
            fan_in = max(1, int(self.pool._in_counts[pre].item()) + 1)
            w = random.gauss(0, 1.0 / math.sqrt(fan_in))
            w = max(-1.0, min(1.0, w))
        else:
            w = float(weight)

        self.pool.syn_alive[sid] = True
        self.pool.pre_id[sid] = pre
        self.pool.post_id[sid] = post
        self.pool.weight[sid] = w
        self.pool.trace[sid] = 0.0
        self.pool.syn_age[sid] = 0
        self.pool._occupied_synapses += 1
        self.pool._fan_in_dirty = True

        in_c = int(self.pool._in_counts[post].item())
        out_c = int(self.pool._out_counts[pre].item())
        if in_c < self.pool.K:
            self.pool.in_ptrs[post, in_c] = sid
            self.pool._in_counts[post] = in_c + 1
        if out_c < self.pool.K:
            self.pool.out_ptrs[pre, out_c] = sid
            self.pool._out_counts[pre] = out_c + 1

        self.pool._total_syn_created += 1
        return sid

    def create_synapses_batch(
        self,
        pre_ids: list[int],
        post_ids: list[int],
        weights: list[float] | None = None,
        conn_type: int = 0,
        init_scale: float = 1.0,
    ) -> int:
        """批量创建突触, 一次 GPU 写入, 替代逐元素 create_synapse."""
        n = len(pre_ids)
        if n == 0:
            return 0

        sids = []
        while len(sids) < n:
            try:
                sids.append(self.pool._storage.alloc_synapse_slot())
            except MemoryBudgetError:
                dead = torch.where(~self.pool.syn_alive)[0]
                if len(dead) == 0:
                    raise RuntimeError(f"突触内存预算耗尽: 需要 {n}")
                for d in dead[: n - len(sids)].tolist():
                    sids.append(int(d))
        self.pool._sync_storage_refs()
        sids_t = torch.tensor(sids, dtype=torch.int32, device=self.pool.device)

        pre_t = torch.tensor(pre_ids, dtype=torch.int32, device=self.pool.device)
        post_t = torch.tensor(post_ids, dtype=torch.int32, device=self.pool.device)

        if weights is None:
            fan_in_cache: dict[int, int] = {}
            w_list = []
            for post_i in post_ids:
                if post_i not in fan_in_cache:
                    fan_in_cache[post_i] = max(1, int(self.pool._in_counts[post_i].item()) + 1)
            for pre_i, post_i in zip(pre_ids, post_ids):
                fan_in = fan_in_cache[post_i]
                w = random.gauss(0, init_scale / math.sqrt(fan_in))
                w = max(-1.0, min(1.0, w))
                w_list.append(w)
            w_t = torch.tensor(w_list, dtype=torch.float16, device=self.pool.device)
        else:
            w_t = torch.tensor(weights, dtype=torch.float16, device=self.pool.device)

        self.pool.syn_alive[sids_t] = True
        self.pool.pre_id[sids_t] = pre_t
        self.pool.post_id[sids_t] = post_t
        self.pool.weight[sids_t] = w_t
        self.pool.trace[sids_t] = 0.0
        self.pool.syn_age[sids_t] = 0
        self.pool.conn_type[sids_t] = conn_type
        self.pool._occupied_synapses += n
        self.pool._fan_in_dirty = True

        # in_ptrs: 分组后逐后神经元写入 (满时向量化置换)
        _post_to_sids: dict[int, list[int]] = {}
        for sid, post_i in zip(sids, post_ids):
            _post_to_sids.setdefault(post_i, []).append(sid)
        for post_i, syn_list in _post_to_sids.items():
            in_c = int(self.pool._in_counts[post_i].item())
            avail = min(len(syn_list), self.pool.K - in_c)
            if avail > 0:
                self.pool.in_ptrs[post_i, in_c : in_c + avail] = torch.tensor(
                    syn_list[:avail], dtype=torch.int32, device=self.pool.device
                )
                self.pool._in_counts[post_i] = in_c + avail
            remain = syn_list[avail:]
            if not remain:
                continue
            # K 满了: 向量化置换 — randint 选 slot, 批量杀旧写新
            n_r = len(remain)
            dev = self.pool.device
            evict = torch.randint(0, self.pool.K, (n_r,), device=dev)
            old_sids = self.pool.in_ptrs[post_i, evict]  # [n_r]
            self.pool.syn_alive[old_sids.long()] = False
            self.pool._occupied_synapses -= n_r
            self.pool.in_ptrs[post_i, evict] = torch.tensor(
                remain, dtype=torch.int32, device=dev
            )
            # _in_counts[post_i] 保持 K (已满)

        # out_ptrs: 分组后逐前神经元写入
        _pre_to_sids: dict[int, list[int]] = {}
        for sid, pre_i in zip(sids, pre_ids):
            _pre_to_sids.setdefault(pre_i, []).append(sid)
        for pre_i, syn_list in _pre_to_sids.items():
            out_c = int(self.pool._out_counts[pre_i].item())
            avail = min(len(syn_list), self.pool.K - out_c)
            if avail > 0:
                self.pool.out_ptrs[pre_i, out_c : out_c + avail] = torch.tensor(
                    syn_list[:avail], dtype=torch.int32, device=self.pool.device
                )
                self.pool._out_counts[pre_i] = out_c + avail

        self.pool._total_syn_created += n
        return n

    def connect_layer(
        self,
        from_layer: int,
        to_layer: int,
        density: float = 0.5,
        bias_strength: float = 0.7,
        conn_type: int = 0,
        init_scale: float = 1.0,
    ) -> int:
        """两层间有偏置的稀疏连接, 纯 GPU 向量化."""
        from_mask = (self.pool.layer == from_layer) & self.pool.alive
        to_mask = (self.pool.layer == to_layer) & self.pool.alive
        from_ids = torch.where(from_mask)[0]
        to_ids = torch.where(to_mask)[0]
        n_from = len(from_ids)
        n_to = len(to_ids)
        if n_from == 0 or n_to == 0:
            return 0

        k = min(self.pool.K, max(1, int(n_from * density)))
        n_local = int(k * bias_strength)
        n_global = k - n_local

        # 每个后神经元: 取 n_local 个偏好源 + n_global 个随机源
        # 偏好 = 在所有源中随机取 10%
        pref_size = max(1, int(n_from * 0.1))

        # 纯 torch 采样替代 random.sample Python 循环
        device = self.pool.device
        to_ids_t = to_ids.to(device)

        pre_list = []
        post_list = []

        perm = torch.randperm(n_from, device=device)
        pref_mask = perm[:pref_size]  # [pref_size]
        rest_mask = perm[pref_size:]  # [n_from - pref_size]
        pref_ids = from_ids[pref_mask]
        rest_ids = from_ids[rest_mask]

        n_pref = min(n_local, pref_size)
        n_rest = min(n_global, len(rest_ids))

        # 全向量化: 用广播生成所有连接
        # 每个后神经元独立随机取 n_pref 个偏好源 + n_rest 个随机源
        all_pre = torch.cat([
            pref_ids[torch.randint(0, len(pref_ids), (n_to, n_pref), device=device)],
            rest_ids[torch.randint(0, len(rest_ids), (n_to, n_rest), device=device)],
        ], dim=1)  # [n_to, n_pref + n_rest]

        all_post = to_ids_t.unsqueeze(1).expand(-1, n_pref + n_rest)  # [n_to, k]

        pre_list = all_pre.reshape(-1).tolist()
        post_list = all_post.reshape(-1).tolist()

        return self.create_synapses_batch(
            pre_list, post_list, conn_type=conn_type, init_scale=init_scale
        )

    def synapse_turnover(self, step: int, rate: float = 0.02) -> int:
        """剪掉最弱的 rate% 突触, 创建等量新试探突触."""
        alive = self.pool.syn_alive
        if not alive.any():
            return 0

        w_abs = self.pool.weight[alive].abs()
        trace_val = self.pool.trace[alive]
        score = 0.5 * w_abs + 0.5 * trace_val

        n_total = alive.sum().item()
        n_turnover = max(1, int(n_total * rate))

        alive_indices = torch.where(alive)[0]
        _, weak_idx = torch.topk(score, n_turnover, largest=False)
        dead_sids = alive_indices[weak_idx]

        types_present = self.pool.conn_type[dead_sids].unique()

        total_rebuilt = 0
        if types_present.numel() == 0:
            return 0

        # 按 (from_layer, ct) 合并后批量重建
        rebuild_map: dict[int, tuple[int, list[int], list[int]]] = {}
        for ct in types_present.tolist():
            type_mask = self.pool.conn_type[dead_sids] == ct
            type_dead = dead_sids[type_mask]
            type_posts = self.pool.post_id[type_dead]

            self.pool.syn_alive[type_dead] = False
            self.pool._occupied_synapses -= len(type_dead)
            for sid in type_dead.tolist():
                self.pool._free_synapses.append(sid)

            # 获取 from_layer
            sample_post = int(type_posts[0].item())
            post_layer = int(self.pool.layer[sample_post].item())
            if ct == CONN_FEEDFORWARD:
                from_layer = LAYER_SENSORY if post_layer == LAYER_L4 else post_layer - 1
            elif ct == CONN_FEEDBACK:
                from_layer = post_layer + 1
            else:
                from_layer = post_layer

            # 统计每个 post 被删的突触数
            uniq, counts = type_posts.unique(return_counts=True)
            rebuild_map[ct] = (from_layer, uniq.tolist(), counts.tolist())

        for ct, (from_layer, posts, counts) in rebuild_map.items():
            from_mask = (self.pool.layer == from_layer) & self.pool.alive
            from_ids = torch.where(from_mask)[0]
            if len(from_ids) == 0:
                continue

            n_pairs = sum(counts)
            # 向量化一次构造所有 pre-post 对
            posts_expanded = torch.tensor(posts, device=self.pool.device).repeat_interleave(
                torch.tensor(counts, device=self.pool.device))
            pre_indices = torch.randint(0, len(from_ids), (n_pairs,), device=self.pool.device)
            pre_alist = from_ids[pre_indices].tolist()

            if pre_alist:
                self.create_synapses_batch(
                    pre_alist, posts_expanded.tolist(), conn_type=ct)
                total_rebuilt += len(pre_alist)

        return total_rebuilt
