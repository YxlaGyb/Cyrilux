"""
NeuronManager
神经元生命周期管理 (创建/删除/分裂).
"""

from __future__ import annotations

import random

import torch

from ..constants import (
    F_BCM_SLOPE,
    F_BCM_ZERO,
    F_EPS,
    F_FIRING_RATE,
    F_MU,
    F_PI,
    F_THRESHOLD,
    F_Z,
    F_Z_PREV,
)
from .page_storage import MemoryBudgetError


class NeuronManager:
    """神经元 CRUD: 创建、批量创建、修剪、分裂.

    通过 self.pool 引用访问共享张量, 扩容后自动指向新张量.
    """

    def __init__(self, pool):
        self.pool = pool

    def create_neuron(self, layer: int, **kwargs) -> int:
        """创建新神经元, 返回槽位索引."""
        nid = self.pool._storage.alloc_neuron_slot()
        self.pool._sync_storage_refs()
        self.pool.alive[nid] = True
        self.pool.layer[nid] = layer
        self.pool._occupied_neurons += 1

        thr = kwargs.get("threshold", 0.1)
        pos = kwargs.get("position", -1)
        ch = kwargs.get("channel", -1)
        z_val = kwargs.get("z", 0.0)

        self.pool.state[nid, F_THRESHOLD] = thr
        self.pool.state[nid, F_Z] = z_val if z_val != 0.0 or layer == 0 else 0.1
        self.pool.state[nid, F_MU] = 0.0
        self.pool.state[nid, F_EPS] = 0.0
        self.pool.state[nid, F_FIRING_RATE] = 0.0
        self.pool.state[nid, F_PI] = 1.0
        self.pool.state[nid, F_Z_PREV] = 0.0
        if layer > 0:
            self.pool.state[nid, F_BCM_SLOPE] = 2.8 + 2.4 * random.random()
            self.pool.state[nid, F_BCM_ZERO] = 0.15 + 0.30 * random.random()
        else:
            self.pool.state[nid, F_BCM_SLOPE] = 4.0
            self.pool.state[nid, F_BCM_ZERO] = 0.25
        self.pool.position[nid] = pos
        self.pool.channel[nid] = ch
        self.pool.created_at[nid] = self.pool._total_created

        if pos >= 0:
            self.pool._sensory_index[(layer, int(pos), int(ch))] = nid

        self.pool._total_created += 1
        return nid

    def create_neurons_batch(
        self,
        layers: list[int],
        positions: list[int] | None = None,
        channels: list[int] | None = None,
        thresholds: list[float] | None = None,
        z_vals: list[float] | None = None,
    ) -> list[int]:
        """批量创建神经元, 一次 GPU 写入, 替代逐元素 create_neuron."""
        n_create = len(layers)
        if n_create == 0:
            return []

        nids = []
        while len(nids) < n_create:
            try:
                nids.append(self.pool._storage.alloc_neuron_slot())
            except MemoryBudgetError:
                alive_ids = torch.where(self.pool.alive)[0]
                if len(alive_ids) == 0:
                    raise RuntimeError("TensorNeuronPool 内存预算耗尽, 且无存活神经元可回收")
                need = n_create - len(nids)
                k = min(need, len(alive_ids))
                _, oldest_rel = torch.topk(self.pool.last_active[alive_ids], k, largest=False)
                for nid_val in alive_ids[oldest_rel].tolist():
                    self.prune_neuron(int(nid_val))
                    try:
                        nids.append(self.pool._storage.alloc_neuron_slot())
                    except MemoryBudgetError:
                        continue
                    if len(nids) >= n_create:
                        break
        self.pool._sync_storage_refs()
        nids_t = torch.tensor(nids, dtype=torch.int32, device=self.pool.device)

        self.pool.alive[nids_t] = True
        self.pool._occupied_neurons += n_create

        pos = positions or [-1] * n_create
        ch = channels or [-1] * n_create
        thr = thresholds or [0.1] * n_create
        zv = z_vals or [0.0] * n_create

        pos_t = torch.tensor(pos, dtype=torch.int16, device=self.pool.device)
        ch_t = torch.tensor(ch, dtype=torch.int16, device=self.pool.device)
        layer_t = torch.tensor(layers, dtype=torch.int16, device=self.pool.device)

        self.pool.layer[nids_t] = layer_t
        self.pool.position[nids_t] = pos_t
        self.pool.channel[nids_t] = ch_t
        self.pool.created_at[nids_t] = self.pool._total_created

        self.pool.state[nids_t, F_THRESHOLD] = torch.tensor(
            thr, dtype=torch.float16, device=self.pool.device
        )
        self.pool.state[nids_t, F_Z] = torch.tensor(
            zv, dtype=torch.float16, device=self.pool.device
        )
        self.pool.state[nids_t, F_MU] = 0.0
        self.pool.state[nids_t, F_EPS] = 0.0
        self.pool.state[nids_t, F_FIRING_RATE] = 0.0
        self.pool.state[nids_t, F_PI] = 1.0
        self.pool.state[nids_t, F_Z_PREV] = 0.0
        # BCM 初始化 (向量化, 免 Python loop)
        layer_t_gpu = layer_t.to(device=self.pool.device)
        hidden_mask = layer_t_gpu > 0
        n_hidden = int(hidden_mask.sum().item())
        slopes = torch.full((n_create,), 4.0, dtype=torch.float16, device=self.pool.device)
        zeros = torch.full((n_create,), 0.25, dtype=torch.float16, device=self.pool.device)
        if n_hidden > 0:
            slopes = slopes.index_put(
                (torch.arange(n_create, device=self.pool.device)[hidden_mask],),
                2.8 + 2.4 * torch.rand(n_hidden, dtype=torch.float16, device=self.pool.device)
            )
            zeros = zeros.index_put(
                (torch.arange(n_create, device=self.pool.device)[hidden_mask],),
                0.15 + 0.30 * torch.rand(n_hidden, dtype=torch.float16, device=self.pool.device)
            )
        self.pool.state[nids_t, F_BCM_SLOPE] = slopes
        self.pool.state[nids_t, F_BCM_ZERO] = zeros

        for i, nid in enumerate(nids):
            if pos[i] >= 0:
                self.pool._sensory_index[(layers[i], pos[i], ch[i])] = nid

        self.pool._total_created += n_create
        return nids

    def prune_neuron(self, nid: int, force: bool = False) -> bool:
        """
        惰性删除神经元:

        标记 alive=False, 关联突触标记 syn_alive=False.
        """
        if not self.pool.alive[nid]:
            return False

        pos = int(self.pool.position[nid].item())
        if pos >= 0:
            key = (int(self.pool.layer[nid].item()), pos, int(self.pool.channel[nid].item()))
            self.pool._sensory_index.pop(key, None)

        self.pool.alive[nid] = False
        self.pool.layer[nid] = -1
        self.pool._occupied_neurons -= 1
        self.pool._storage.free_neuron_slot(nid)

        in_ptr_slice = self.pool.in_ptrs[nid].cpu()
        for i in range(self.pool.K):
            sid = int(in_ptr_slice[i].item())
            if sid >= 0:
                self.pool.syn_alive[sid] = False
                self.pool._storage.free_synapse_slot(sid)
                self.pool._occupied_synapses -= 1
        self.pool._fan_in_dirty = True
        out_ptr_slice = self.pool.out_ptrs[nid].cpu()
        for i in range(self.pool.K):
            sid = int(out_ptr_slice[i].item())
            if sid >= 0:
                self.pool.syn_alive[sid] = False
                self.pool._storage.free_synapse_slot(sid)
                self.pool._occupied_synapses -= 1

        self.pool._in_counts[nid] = 0
        self.pool._out_counts[nid] = 0

        self.pool.t_connected[nid] = False
        self.pool.t_weight[nid] = 0.0
        self._remove_td_for_neuron(nid)

        self.pool._total_pruned += 1
        return True

    def split_neuron(self, nid: int, noise_scale: float = 0.05) -> int | None:
        """
        分裂神经元:

        创建子神经元, 平分入突触.
        """
        if not self.pool.alive[nid]:
            return None

        parent_layer = int(self.pool.layer[nid].item())
        parent_thr = float(self.pool.state[nid, F_THRESHOLD].item())
        parent_z = float(self.pool.state[nid, F_Z].item())
        parent_pos = int(self.pool.position[nid].item())
        parent_ch = int(self.pool.channel[nid].item())

        child = self.create_neuron(
            layer=parent_layer,
            threshold=parent_thr,
            position=parent_pos,
            channel=parent_ch,
            z=parent_z * 0.5,
        )

        in_sids = []
        in_c = int(self.pool._in_counts[nid].item())
        for i in range(min(in_c, self.pool.K)):
            sid = int(self.pool.in_ptrs[nid, i].item())
            if sid >= 0 and self.pool.syn_alive[sid]:
                in_sids.append(sid)

        random.shuffle(in_sids)
        mid = len(in_sids) // 2

        for sid in in_sids[mid:]:
            self.pool.post_id[sid] = child
            new_w = float(self.pool.weight[sid].item()) * (1.0 + random.gauss(0, noise_scale))
            self.pool.weight[sid] = max(-1.0, min(1.0, new_w))
            child_in = int(self.pool._in_counts[child].item())
            if child_in < self.pool.K:
                self._remove_from_in_ptrs(nid, sid)
                self.pool.in_ptrs[child, child_in] = sid
                self.pool._in_counts[child] = child_in + 1

        out_sids = []
        out_c = int(self.pool._out_counts[nid].item())
        for i in range(min(out_c, self.pool.K)):
            sid = int(self.pool.out_ptrs[nid, i].item())
            if sid >= 0 and self.pool.syn_alive[sid]:
                out_sids.append(sid)

        random.shuffle(out_sids)
        inherit_n = max(1, len(out_sids) // 3)
        for sid in out_sids[:inherit_n]:
            post = int(self.pool.post_id[sid].item())
            src_w = float(self.pool.weight[sid].item())
            new_w = src_w * (1.0 + random.gauss(0, noise_scale))
            self.pool.synapse.create_synapse(child, post, weight=new_w)

        self.pool.state[nid, F_THRESHOLD] = min(1.0, parent_thr * 1.1)
        return child

    def _remove_from_in_ptrs(self, nid: int, sid: int):
        """从神经元入邻接表中移除指定突触."""
        in_c = int(self.pool._in_counts[nid].item())
        for i in range(min(in_c, self.pool.K)):
            if int(self.pool.in_ptrs[nid, i].item()) == sid:
                last_valid = in_c - 1
                if i < last_valid:
                    self.pool.in_ptrs[nid, i] = self.pool.in_ptrs[nid, last_valid]
                self.pool.in_ptrs[nid, last_valid] = -1
                self.pool._in_counts[nid] = in_c - 1
                break

    def _remove_td_for_neuron(self, nid: int):
        """移除神经元的所有 topdown 连接 (向量化布尔索引)."""
        td_pre_slice = self.pool.td_pre[: self.pool._td_count]
        td_post_slice = self.pool.td_post[: self.pool._td_count]
        mask = (td_pre_slice == nid) | (td_post_slice == nid)
        self.pool.td_alive[: self.pool._td_count][mask] = False

    def grow_hidden_neurons(self, layer: int, count: int) -> list[int]:
        """在隐藏层创建 count 个新神经元 (含 BCM 初始化 + 时序自连接 + 信道标签)."""
        nids = self.create_neurons_batch(
            layers=[layer] * count,
            thresholds=[0.1] * count,
        )
        # 信道标签: 优先填补该层非满的通道
        alive_nids = torch.where((self.pool.layer == layer) & self.pool.alive)[0]
        ch_counts = torch.zeros(8, dtype=torch.int32)
        for nid in alive_nids.tolist():
            ch = int(self.pool.channel[nid].item())
            if ch >= 0:
                ch_counts[ch] += 1
        target_ch = int(ch_counts.argmin().item())
        for i, nid in enumerate(nids):
            self.pool.channel[nid] = (target_ch + i) % 8
        # 时序自连接
        self.pool.projections.temporal_connect(nids)
        return nids

    def _force_recycle(self) -> bool:
        """池满时回收最久未活跃的神经元."""
        alive_ids = torch.where(self.pool.alive)[0]
        if len(alive_ids) == 0:
            return False
        oldest_idx = int(self.pool.last_active[alive_ids].argmin().item())
        return self.prune_neuron(int(alive_ids[oldest_idx].item()))
