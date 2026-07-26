"""TensorNeuronPool — 全张量化神经元池.

替代旧的 NeuronPool (dict/dataclass), 所有状态存储为 device-agnostic tensor.
可在 CPU 或 CUDA 上运行, 计算操作批量执行.

Tensor 布局:
  N = max_neurons (默认 65536)
  S = max_synapses (默认 8M, 约 128 连接/神经元)
  K = max_fan_in/out (默认 64, 缺填 -1)

  神经元状态:    state[N, 7] fp16  (z, mu, epsilon, threshold, firing_rate, pi, z_prev)
  神经元元数据:   layer[N] int16, position[N] int16, channel[N] int16
                 created_at[N] int32, last_active[N] int32
  存活:          alive[N] bool

  突触:         pre_id[S] int32, post_id[S] int32
                weight[S] fp16, trace[S] fp16, age[S] int32
  存活:          syn_alive[S] bool

  邻接:         in_ptrs[N, K] int32, out_ptrs[N, K] int32  (缺填 -1)

  投影:
    temporal:   t_weight[N] fp16, t_connected[N] bool
    topdown:    td_pre[S_td] int32, td_post[S_td] int32, td_weight[S_td] fp16
    lm_head:    lm_weight[256, N] fp16  (32MB 稠密, 可接受)
"""

from __future__ import annotations

import math
import random

import torch


# ── 神经元状态列索引 ─────────────────────────────────────────────────
F_Z = 0
F_MU = 1
F_EPS = 2
F_THRESHOLD = 3
F_FIRING_RATE = 4
F_PI = 5
F_Z_PREV = 6
N_STATE_FIELDS = 7


class TensorNeuronPool:
    """全张量化神经元池.

    Args:
        max_neurons: 神经元槽位数 (默认 65536)
        max_synapses: 突触槽位数 (默认 8M)
        K: 每神经元最大入/出连接数 (默认 128)
        S_td: topdown 连接槽位数 (默认 5M)
        device: 张量存储设备
    """

    def __init__(
        self,
        max_neurons: int = 65536,
        max_synapses: int = 8_000_000,
        K: int = 128,
        S_td: int = 5_000_000,
        device: torch.device | str = "cpu",
    ):
        self.N = max_neurons
        self.S = max_synapses
        self.K = K
        self.S_td = S_td
        self.device = torch.device(device) if isinstance(device, str) else device

        # ── 神经元张量 ──
        self.state = torch.zeros(
            max_neurons, N_STATE_FIELDS, dtype=torch.float16, device=self.device
        )
        self.layer = torch.full((max_neurons,), -1, dtype=torch.int16, device=self.device)
        self.position = torch.full((max_neurons,), -1, dtype=torch.int16, device=self.device)
        self.channel = torch.full((max_neurons,), -1, dtype=torch.int16, device=self.device)
        self.created_at = torch.zeros(max_neurons, dtype=torch.int32, device=self.device)
        self.last_active = torch.zeros(max_neurons, dtype=torch.int32, device=self.device)
        self.alive = torch.zeros(max_neurons, dtype=torch.bool, device=self.device)

        # ── 突触张量 ──
        self.pre_id = torch.zeros(max_synapses, dtype=torch.int32, device=self.device)
        self.post_id = torch.zeros(max_synapses, dtype=torch.int32, device=self.device)
        self.weight = torch.zeros(max_synapses, dtype=torch.float16, device=self.device)
        self.trace = torch.zeros(max_synapses, dtype=torch.float16, device=self.device)
        self.syn_age = torch.zeros(max_synapses, dtype=torch.int32, device=self.device)
        self.syn_alive = torch.zeros(max_synapses, dtype=torch.bool, device=self.device)

        # ── 邻接表 (pad -1 表示空槽) ──
        self.in_ptrs = torch.full((max_neurons, K), -1, dtype=torch.int32, device=self.device)
        self.out_ptrs = torch.full((max_neurons, K), -1, dtype=torch.int32, device=self.device)
        # 每神经元当前连接数 (CPU 侧辅助, 用于快速找到空槽)
        self._in_counts = torch.zeros(max_neurons, dtype=torch.int32)
        self._out_counts = torch.zeros(max_neurons, dtype=torch.int32)

        # ── 时序投影 ──
        self.t_weight = torch.zeros(max_neurons, dtype=torch.float16, device=self.device)
        self.t_connected = torch.zeros(max_neurons, dtype=torch.bool, device=self.device)

        # ── 自上而下投影 (COO) ──
        self.td_pre = torch.zeros(S_td, dtype=torch.int32, device=self.device)
        self.td_post = torch.zeros(S_td, dtype=torch.int32, device=self.device)
        self.td_weight = torch.zeros(S_td, dtype=torch.float16, device=self.device)
        self.td_alive = torch.zeros(S_td, dtype=torch.bool, device=self.device)
        self._td_count: int = 0

        # ── LM Head (稠密 [256, N], 32MB) ──
        self.lm_weight = torch.zeros(256, max_neurons, dtype=torch.float16, device=self.device)

        # ── CPU 侧辅助 ──
        self._free_neurons: list[int] = list(range(max_neurons))
        self._free_synapses: list[int] = list(range(max_synapses))
        self._occupied_neurons: int = 0
        self._occupied_synapses: int = 0
        # 感官神经元 hash 索引: (layer, position, channel) → nid
        self._sensory_index: dict[tuple[int, int, int], int] = {}

        # 统计
        self._total_created: int = 0
        self._total_pruned: int = 0
        self._total_syn_created: int = 0
        self._total_syn_pruned: int = 0

    # ═══════════════════════════════════════════════════════════════
    # 结构操作 (CPU 侧)
    # ═══════════════════════════════════════════════════════════════

    def create_neuron(self, layer: int, **kwargs) -> int:
        """创建新神经元, 返回槽位索引.

        Args:
            layer: 逻辑层索引
            **kwargs: position, channel, threshold, z 等初始值

        Returns:
            神经元索引 (0..N-1).
        """
        if not self._free_neurons:
            if not self._force_recycle():
                raise RuntimeError(f"TensorNeuronPool 已达上限 {self.N}, 无法回收")

        nid = self._free_neurons.pop()
        self.alive[nid] = True
        self.layer[nid] = layer
        self._occupied_neurons += 1

        thr = kwargs.get("threshold", 0.1)
        pos = kwargs.get("position", -1)
        ch = kwargs.get("channel", -1)
        z_val = kwargs.get("z", 0.0)

        self.state[nid, F_THRESHOLD] = thr
        self.state[nid, F_Z] = z_val if z_val != 0.0 or layer == 0 else 0.1
        self.state[nid, F_MU] = 0.0
        self.state[nid, F_EPS] = 0.0
        self.state[nid, F_FIRING_RATE] = 0.0
        self.state[nid, F_PI] = 1.0
        self.state[nid, F_Z_PREV] = 0.0
        self.position[nid] = pos
        self.channel[nid] = ch
        self.created_at[nid] = self._total_created

        if pos >= 0:
            self._sensory_index[(layer, int(pos), int(ch))] = nid

        self._total_created += 1
        return nid

    def create_neurons_batch(
        self,
        layers: list[int],
        positions: list[int] | None = None,
        channels: list[int] | None = None,
        thresholds: list[float] | None = None,
        z_vals: list[float] | None = None,
    ) -> list[int]:
        """批量创建神经元 — 一次 GPU 写入, 替代逐元素 create_neuron.

        Args:
            layers: 每神经元的逻辑层
            positions: 每神经元序列位置 (-1 = 非感官)
            channels: 每神经元特征通道
            thresholds: 初始阈值
            z_vals: 初始 z 值

        Returns:
            新神经元的 nid 列表.
        """
        n_create = len(layers)
        if n_create == 0:
            return []

        # 确保有空槽 (批量回收最久未活跃神经元, 避免逐元素 GPU sync)
        while len(self._free_neurons) < n_create:
            alive_ids = torch.where(self.alive)[0]
            if len(alive_ids) == 0:
                raise RuntimeError(
                    f"TensorNeuronPool 空间不足: 需要 {n_create}, 无存活神经元可回收"
                )
            need = n_create - len(self._free_neurons)
            """
            一次 topk 找到最旧的 N 个神经元
            逐个 prune (prune 本身有 .item() 但远少于 2000 次全扫描)
            """
            k = min(need, len(alive_ids))
            _, oldest_rel = torch.topk(self.last_active[alive_ids], k, largest=False)
            oldest_nids = alive_ids[oldest_rel]
            for nid_val in oldest_nids.tolist():
                self.prune_neuron(int(nid_val))
                if len(self._free_neurons) >= n_create:
                    break

        nids = [self._free_neurons.pop() for _ in range(n_create)]
        nids_t = torch.tensor(nids, dtype=torch.int32, device=self.device)

        # 批量写入 alive 和 layer
        self.alive[nids_t] = True
        self._occupied_neurons += n_create

        pos = positions or [-1] * n_create
        ch = channels or [-1] * n_create
        thr = thresholds or [0.1] * n_create
        zv = z_vals or [0.0] * n_create

        pos_t = torch.tensor(pos, dtype=torch.int16, device=self.device)
        ch_t = torch.tensor(ch, dtype=torch.int16, device=self.device)
        layer_t = torch.tensor(layers, dtype=torch.int16, device=self.device)

        self.layer[nids_t] = layer_t
        self.position[nids_t] = pos_t
        self.channel[nids_t] = ch_t
        self.created_at[nids_t] = self._total_created

        # 批量写入神经元状态
        self.state[nids_t, F_THRESHOLD] = torch.tensor(thr, dtype=torch.float16, device=self.device)
        self.state[nids_t, F_Z] = torch.tensor(zv, dtype=torch.float16, device=self.device)
        self.state[nids_t, F_MU] = 0.0
        self.state[nids_t, F_EPS] = 0.0
        self.state[nids_t, F_FIRING_RATE] = 0.0
        self.state[nids_t, F_PI] = 1.0
        self.state[nids_t, F_Z_PREV] = 0.0

        # 维护感官索引
        for i, nid in enumerate(nids):
            if pos[i] >= 0:
                self._sensory_index[(layers[i], pos[i], ch[i])] = nid

        self._total_created += n_create
        return nids

    def create_synapse(self, pre: int, post: int, weight: float | None = None) -> int:
        """在两个神经元间创建突触.

        Args:
            pre: 前神经元索引
            post: 后神经元索引
            weight: 初始权重 (None = Xavier 随机)

        Returns:
            突触索引 (0..S-1).
        """
        if not self._free_synapses:
            # 扩展: 在可用槽位中找未使用的位置
            dead = torch.where(~self.syn_alive)[0]
            if len(dead) == 0:
                raise RuntimeError(f"TensorNeuronPool 突触已达上限 {self.S}")
            sid = int(dead[0].item())
        else:
            sid = self._free_synapses.pop()

        if weight is None:
            fan_in = max(1, int(self._in_counts[pre].item()) + 1)
            w = random.gauss(0, 1.0 / math.sqrt(fan_in))
            w = max(-1.0, min(1.0, w))
        else:
            w = float(weight)

        self.syn_alive[sid] = True
        self.pre_id[sid] = pre
        self.post_id[sid] = post
        self.weight[sid] = w
        self.trace[sid] = 0.0
        self.syn_age[sid] = 0
        self._occupied_synapses += 1

        # 写入邻接表
        in_c = int(self._in_counts[post].item())
        out_c = int(self._out_counts[pre].item())
        if in_c < self.K:
            self.in_ptrs[post, in_c] = sid
            self._in_counts[post] = in_c + 1
        if out_c < self.K:
            self.out_ptrs[pre, out_c] = sid
            self._out_counts[pre] = out_c + 1

        self._total_syn_created += 1
        return sid

    def create_synapses_batch(
        self,
        pre_ids: list[int],
        post_ids: list[int],
        weights: list[float] | None = None,
    ) -> int:
        """批量创建突触 — 一次 GPU 写入, 替代逐元素 create_synapse.

        Args:
            pre_ids: 前神经元索引列表
            post_ids: 后神经元索引列表
            weights: 初始权重 (None = Xavier 随机)

        Returns:
            创建数.
        """
        n = len(pre_ids)
        if n == 0:
            return 0

        # 分配槽位
        while len(self._free_synapses) < n:
            dead = torch.where(~self.syn_alive)[0]
            if len(dead) == 0:
                raise RuntimeError(f"突触槽位不足: 需要 {n}")
            self._free_synapses.extend(dead[: n - len(self._free_synapses)].tolist())

        sids = [self._free_synapses.pop() for _ in range(n)]
        sids_t = torch.tensor(sids, dtype=torch.int32, device=self.device)

        # 批量写入突触数据
        pre_t = torch.tensor(pre_ids, dtype=torch.int32, device=self.device)
        post_t = torch.tensor(post_ids, dtype=torch.int32, device=self.device)

        if weights is None:
            # 按 post 分组预读 fan_in, 减少 .item() 同步
            fan_in_cache: dict[int, int] = {}
            w_list = []
            for post_i in post_ids:
                if post_i not in fan_in_cache:
                    fan_in_cache[post_i] = max(1, int(self._in_counts[post_i].item()) + 1)
            for pre_i, post_i in zip(pre_ids, post_ids):
                fan_in = fan_in_cache[post_i]
                w = random.gauss(0, 1.0 / math.sqrt(fan_in))
                w = max(-1.0, min(1.0, w))
                w_list.append(w)
            w_t = torch.tensor(w_list, dtype=torch.float16, device=self.device)
        else:
            w_t = torch.tensor(weights, dtype=torch.float16, device=self.device)

        self.syn_alive[sids_t] = True
        self.pre_id[sids_t] = pre_t
        self.post_id[sids_t] = post_t
        self.weight[sids_t] = w_t
        self.trace[sids_t] = 0.0
        self.syn_age[sids_t] = 0
        self._occupied_synapses += n

        # 批量更新邻接表 — 按神经元分组, 每神经元仅一次 .item()
        _post_to_sids: dict[int, list[int]] = {}
        for sid, post_i in zip(sids, post_ids):
            _post_to_sids.setdefault(post_i, []).append(sid)
        for post_i, syn_list in _post_to_sids.items():
            in_c = int(self._in_counts[post_i].item())
            avail = min(len(syn_list), self.K - in_c)
            if avail > 0:
                self.in_ptrs[post_i, in_c : in_c + avail] = torch.tensor(
                    syn_list[:avail], dtype=torch.int32, device=self.device
                )
                self._in_counts[post_i] = in_c + avail

        _pre_to_sids: dict[int, list[int]] = {}
        for sid, pre_i in zip(sids, pre_ids):
            _pre_to_sids.setdefault(pre_i, []).append(sid)
        for pre_i, syn_list in _pre_to_sids.items():
            out_c = int(self._out_counts[pre_i].item())
            avail = min(len(syn_list), self.K - out_c)
            if avail > 0:
                self.out_ptrs[pre_i, out_c : out_c + avail] = torch.tensor(
                    syn_list[:avail], dtype=torch.int32, device=self.device
                )
                self._out_counts[pre_i] = out_c + avail

        self._total_syn_created += n
        return n

    def prune_neuron(self, nid: int, force: bool = False) -> bool:
        """惰性删除神经元: 标记 alive=False, 关联突触标记 syn_alive=False.

        Args:
            nid: 神经元索引
            force: 未使用 (保持 API 兼容)

        Returns:
            True=成功.
        """
        if not self.alive[nid]:
            return False

        # 清理感官索引
        pos = int(self.position[nid].item())
        if pos >= 0:
            key = (int(self.layer[nid].item()), pos, int(self.channel[nid].item()))
            self._sensory_index.pop(key, None)

        self.alive[nid] = False
        self.layer[nid] = -1
        self._occupied_neurons -= 1
        self._free_neurons.append(nid)

        # 惰性删除关联突触 (回收槽位到 _free_synapses)
        in_c = int(self._in_counts[nid].item())
        for i in range(min(in_c, self.K)):
            sid = int(self.in_ptrs[nid, i].item())
            if sid >= 0:
                self.syn_alive[sid] = False
                self._free_synapses.append(sid)
                self._occupied_synapses -= 1
        out_c = int(self._out_counts[nid].item())
        for i in range(min(out_c, self.K)):
            sid = int(self.out_ptrs[nid, i].item())
            if sid >= 0:
                self.syn_alive[sid] = False
                self._free_synapses.append(sid)
                self._occupied_synapses -= 1

        # 重置邻接计数
        self._in_counts[nid] = 0
        self._out_counts[nid] = 0

        # 清理时序和 topdown
        self.t_connected[nid] = False
        self.t_weight[nid] = 0.0
        self._remove_td_for_neuron(nid)

        self._total_pruned += 1
        return True

    def split_neuron(self, nid: int, noise_scale: float = 0.05) -> int | None:
        """分裂神经元: 创建子神经元, 平分入突触.

        Args:
            nid: 父神经元索引
            noise_scale: 分裂权重噪声

        Returns:
            子神经元索引, 或 None.
        """
        if not self.alive[nid]:
            return None

        parent_layer = int(self.layer[nid].item())
        parent_thr = float(self.state[nid, F_THRESHOLD].item())
        parent_z = float(self.state[nid, F_Z].item())
        parent_pos = int(self.position[nid].item())
        parent_ch = int(self.channel[nid].item())

        child = self.create_neuron(
            layer=parent_layer,
            threshold=parent_thr,
            position=parent_pos,
            channel=parent_ch,
            z=parent_z * 0.5,
        )

        # 收集父神经元的入突触, 平分给子神经元
        in_sids = []
        in_c = int(self._in_counts[nid].item())
        for i in range(min(in_c, self.K)):
            sid = int(self.in_ptrs[nid, i].item())
            if sid >= 0 and self.syn_alive[sid]:
                in_sids.append(sid)

        random.shuffle(in_sids)
        mid = len(in_sids) // 2

        for sid in in_sids[mid:]:
            self.post_id[sid] = child
            new_w = float(self.weight[sid].item()) * (1.0 + random.gauss(0, noise_scale))
            self.weight[sid] = max(-1.0, min(1.0, new_w))
            # 加入子神经元的入邻接
            child_in = int(self._in_counts[child].item())
            if child_in < self.K:
                # 从父神经元移除
                self._remove_from_in_ptrs(nid, sid)
                self.in_ptrs[child, child_in] = sid
                self._in_counts[child] = child_in + 1

        # 复制部分出突触
        out_sids = []
        out_c = int(self._out_counts[nid].item())
        for i in range(min(out_c, self.K)):
            sid = int(self.out_ptrs[nid, i].item())
            if sid >= 0 and self.syn_alive[sid]:
                out_sids.append(sid)

        random.shuffle(out_sids)
        inherit_n = max(1, len(out_sids) // 3)
        for sid in out_sids[:inherit_n]:
            post = int(self.post_id[sid].item())
            src_w = float(self.weight[sid].item())
            new_w = src_w * (1.0 + random.gauss(0, noise_scale))
            self.create_synapse(child, post, weight=new_w)

        # 父神经元阈值升高
        self.state[nid, F_THRESHOLD] = min(1.0, parent_thr * 1.1)
        return child

    def connect_layer(self, from_layer: int, to_layer: int, density: float = 0.5) -> int:
        """两层间随机稀疏连接 (批量创建, 一次 GPU 写入).

        Args:
            from_layer: 源层
            to_layer: 目标层
            density: 连接密度

        Returns:
            创建的连接数.
        """
        from_mask = (self.layer == from_layer) & self.alive
        to_mask = (self.layer == to_layer) & self.alive
        from_ids = torch.where(from_mask)[0].tolist()
        to_ids = torch.where(to_mask)[0].tolist()

        if not from_ids or not to_ids:
            return 0

        # 预计算所有 (pre, post) 对
        pre_list: list[int] = []
        post_list: list[int] = []
        for post in to_ids:
            k = max(1, int(len(from_ids) * density))
            sampled = random.sample(from_ids, min(k, len(from_ids)))
            for pre in sampled:
                pre_list.append(pre)
                post_list.append(post)

        return self.create_synapses_batch(pre_list, post_list)

    def _force_recycle(self) -> bool:
        """池满时回收最久未活跃的神经元."""
        alive_ids = torch.where(self.alive)[0]
        if len(alive_ids) == 0:
            return False
        oldest_idx = int(self.last_active[alive_ids].argmin().item())
        return self.prune_neuron(int(alive_ids[oldest_idx].item()))

    def _remove_from_in_ptrs(self, nid: int, sid: int):
        """从神经元入邻接表中移除指定突触."""
        in_c = int(self._in_counts[nid].item())
        for i in range(min(in_c, self.K)):
            if int(self.in_ptrs[nid, i].item()) == sid:
                # 将最后一个有效条目移到此位置
                last_valid = in_c - 1
                if i < last_valid:
                    self.in_ptrs[nid, i] = self.in_ptrs[nid, last_valid]
                self.in_ptrs[nid, last_valid] = -1
                self._in_counts[nid] = in_c - 1
                break

    def _remove_td_for_neuron(self, nid: int):
        """移除神经元的所有 topdown 连接 (向量化布尔索引)."""
        td_pre_slice = self.td_pre[: self._td_count]
        td_post_slice = self.td_post[: self._td_count]
        mask = (td_pre_slice == nid) | (td_post_slice == nid)
        self.td_alive[: self._td_count][mask] = False

    # ═══════════════════════════════════════════════════════════════
    # 查询 (tensor 原生)
    # ═══════════════════════════════════════════════════════════════

    def get_neurons_by_layer(self, layer: int) -> torch.Tensor:
        """返回指定层的神经元索引."""
        mask = (self.layer == layer) & self.alive
        return torch.where(mask)[0]

    def get_layer_width(self, layer: int) -> int:
        """返回指定层宽度."""
        return int(((self.layer == layer) & self.alive).sum().item())

    def get_active_neurons(self, layer: int | None = None) -> torch.Tensor:
        """返回 |epsilon| > threshold 的活跃神经元索引.

        Args:
            layer: 限定层 (None = 全网络)
        """
        active_mask = (self.state[:, F_EPS].abs() > self.state[:, F_THRESHOLD]) & self.alive
        if layer is not None:
            active_mask &= self.layer == layer
        return torch.where(active_mask)[0]

    def get_total_neurons(self) -> int:
        """存活神经元总数."""
        return int(self.alive.sum().item())

    def get_total_synapses(self) -> int:
        """存活突触总数."""
        return int(self.syn_alive.sum().item())

    def get_activity_stats(self) -> dict:
        """返回全网络活跃统计."""
        alive_mask = self.alive
        n_alive = int(alive_mask.sum().item())
        if n_alive == 0:
            return {
                "total_neurons": 0,
                "total_synapses": 0,
                "layer_widths": {},
                "avg_firing_rate": 0.0,
                "avg_threshold": 0.1,
                "n_created": self._total_created,
                "n_pruned": self._total_pruned,
                "n_syn_created": self._total_syn_created,
                "n_syn_pruned": self._total_syn_pruned,
            }

        rates = self.state[alive_mask, F_FIRING_RATE]
        thresholds = self.state[alive_mask, F_THRESHOLD]
        layers = self.layer[alive_mask]

        layer_widths = {}
        for L in layers.unique().tolist():
            layer_widths[int(L)] = int((layers == L).sum().item())

        return {
            "total_neurons": n_alive,
            "total_synapses": self.get_total_synapses(),
            "layer_widths": layer_widths,
            "avg_firing_rate": float(rates.mean().item()) if len(rates) > 0 else 0.0,
            "avg_threshold": float(thresholds.mean().item()) if len(thresholds) > 0 else 0.1,
            "n_created": self._total_created,
            "n_pruned": self._total_pruned,
            "n_syn_created": self._total_syn_created,
            "n_syn_pruned": self._total_syn_pruned,
        }

    # ═══════════════════════════════════════════════════════════════
    # 批量前向操作
    # ═══════════════════════════════════════════════════════════════

    def predict_all(self) -> None:
        """全网 scatter-add 预测: mu = 1/sqrt(K) * scatter_add(weight * z_pre, post).

        1/sqrt(K) 缩放防止 fp16 溢出 (K = 入连接数).
        写入 state[:, F_MU] 和 state[:, F_EPS].
        """
        alive_syn = self.syn_alive
        if not alive_syn.any():
            return

        syn_mask = alive_syn
        pre = self.pre_id[syn_mask]
        post = self.post_id[syn_mask]
        w = self.weight[syn_mask]
        z_pre = self.state[pre.long(), F_Z]

        # z_pre RMSNorm (投影前归一化, 防止上游激活值爆炸)
        rms = (z_pre * z_pre).mean().sqrt()
        rms = rms + 1e-6
        z_pre = z_pre / rms

        contrib = w * z_pre
        mu = torch.zeros(self.N, dtype=torch.float16, device=self.device)
        mu.scatter_add_(0, post.long(), contrib)

        # 1/sqrt(K) 缩放 (K = 每神经元入连接数)
        alive = self.alive
        fan_in = (self.in_ptrs[alive] >= 0).sum(dim=-1, keepdim=True).to(torch.float16)
        scale = torch.rsqrt(fan_in + 1e-6)
        self.state[alive, F_MU] = mu[alive] * scale.squeeze(-1)
        self.state[alive, F_EPS] = self.state[alive, F_Z] - self.state[alive, F_MU]

        # 非感官层神经元: z 向 mu 靠拢但不瞬间跟随 (膜电位时间常数)
        # z_new = 0.9*z_old + 0.1*mu — 神经元有自己的惯性, 不是预测的镜子
        non_sensory = alive & (self.layer > 0)
        if non_sensory.any():
            z_old = self.state[non_sensory, F_Z]
            mu_ns = self.state[non_sensory, F_MU]
            z_new = 0.9 * z_old + 0.1 * mu_ns

            # 逐层 k-WTA 侧抑制: 每层独立竞争 (生物: 侧抑制在同一皮层内运作)
            # 效果: 每层保留 top-10% 最强神经元 → 稀疏、可分的表示
            for L in self.layer[non_sensory].unique().tolist():
                if L <= 0:
                    continue
                in_layer = self.layer[non_sensory] == L  # 在 z_new 中的位置
                n_L = in_layer.sum().item()
                if n_L <= 4:
                    continue
                z_L = z_new[in_layer]
                k = max(4, n_L // 10)
                _, top_idx = torch.topk(z_L.abs(), k)
                winners = torch.zeros(n_L, dtype=torch.bool, device=self.device)
                winners[top_idx] = True
                z_new[in_layer] = torch.where(winners, z_L, z_L * 0.1)

            self.state[non_sensory, F_Z] = z_new
            self.state[non_sensory, F_EPS] = z_new - mu_ns

    def predict_neurons(self, nids: torch.Tensor) -> torch.Tensor:
        """批量预测指定神经元的 mu.

        Args:
            nids: [E] 神经元索引

        Returns:
            mu: [E] 预测值
        """
        E = nids.shape[0]
        if E == 0:
            return torch.zeros(0, dtype=torch.float16, device=self.device)

        # 收集这些神经元的入突触
        in_s = self.in_ptrs[nids.long()]  # [E, K]
        valid = (in_s >= 0) & self.syn_alive[in_s.long()]
        in_s = in_s.long()
        pre_ids = self.pre_id[in_s]  # [E, K]
        valid &= pre_ids >= 0
        w = self.weight[in_s]  # [E, K]
        z_pre = torch.where(valid, self.state[pre_ids.long(), F_Z], torch.zeros_like(w))
        w_masked = torch.where(valid, w, torch.zeros_like(w))

        mu = (w_masked * z_pre).sum(dim=-1)  # [E]
        return mu

    def update_batch(self, nids: torch.Tensor, z_new: torch.Tensor) -> None:
        """批量更新 z, 重算 mu 和 eps.

        Args:
            nids: [E] 神经元索引
            z_new: [E] 新 z 值
        """
        if nids.shape[0] == 0:
            return
        self.state[nids.long(), F_Z] = z_new
        mu = self.predict_neurons(nids)
        self.state[nids.long(), F_MU] = mu
        self.state[nids.long(), F_EPS] = z_new - mu

    def temporal_topdown_pass(self, top_layer: int):
        """时序预测 + 自上而下预测, 更新 mu/epsilon.

        Args:
            top_layer: 最高隐藏层索引 (>0 才做 topdown)
        """
        alive = self.alive

        # 时序预测: mu += t_weight * z_prev
        connected = self.t_connected & alive
        if connected.any():
            mu_temp = self.t_weight[connected] * self.state[connected, F_Z_PREV]
            self.state[connected, F_MU] += mu_temp
            self.state[connected, F_EPS] = self.state[connected, F_Z] - self.state[connected, F_MU]

        # 自上而下预测: 对感觉层 (layer==0) 做 scatter_add
        if top_layer <= 0:
            return

        td_alive = self.td_alive
        if not td_alive.any():
            return

        td_mask = td_alive
        td_pre = self.td_pre[td_mask].long()
        td_post = self.td_post[td_mask].long()
        td_w = self.td_weight[td_mask]
        z_upper = self.state[td_pre, F_Z]

        contrib = td_w * z_upper
        mu_td = torch.zeros(self.N, dtype=torch.float16, device=self.device)
        mu_td.scatter_add_(0, td_post, contrib)

        sensory = (self.layer == 0) & alive
        if sensory.any():
            self.state[sensory, F_MU] += mu_td[sensory]
            self.state[sensory, F_EPS] = self.state[sensory, F_Z] - self.state[sensory, F_MU]

    def compute_free_energy(self) -> torch.Tensor:
        """F = sum(epsilon^2) over alive neurons."""
        return (self.state[self.alive, F_EPS] ** 2).sum()

    def compute_lm_logits(self, top_layer: int) -> torch.Tensor:
        """计算 256 个字节 logits.

        logits = lm_weight @ z[top_layer_mask]  (选择顶层活跃神经元)

        Args:
            top_layer: 顶层索引

        Returns:
            logits: [256] fp16
        """
        top_mask = (self.layer == top_layer) & self.alive
        n_top = int(top_mask.sum().item())
        if n_top == 0:
            return torch.zeros(256, dtype=torch.float16, device=self.device)

        z_top = self.state[top_mask, F_Z]  # [n_top]
        w_sub = self.lm_weight[:, top_mask]  # [256, n_top]
        logits = w_sub @ z_top  # [256]
        return logits

    def compute_cross_entropy(self, logits: torch.Tensor, target_byte: int) -> torch.Tensor:
        """计算交叉熵损失."""
        return torch.nn.functional.cross_entropy(
            logits.unsqueeze(0).float(),
            torch.tensor([target_byte], device=self.device),
        )

    # ═══════════════════════════════════════════════════════════════
    # 批量学习
    # ═══════════════════════════════════════════════════════════════

    def hebbian_pass(
        self,
        active_nids: torch.Tensor,
        eta: float,
        oja_alpha: float,
        dopamine: float,
    ) -> float:
        """批量 Hebbian + Oja 更新活跃神经元的入突触.

        dw = eta * D * pi * eps * z_pre - oja_alpha * D * eps^2 * w

        Args:
            active_nids: [A] 活跃神经元索引
            eta: 基础学习率
            oja_alpha: Oja 约束强度
            dopamine: 多巴胺调制

        Returns:
            总绝对权重变化量.
        """
        A = active_nids.shape[0]
        if A == 0:
            return 0.0

        # 收集活跃神经元的入突触
        in_s = self.in_ptrs[active_nids.long()]  # [A, K]
        valid = (in_s >= 0) & self.syn_alive[in_s.long()]
        in_s = in_s.long()
        pre_ids = self.pre_id[in_s]
        valid &= pre_ids >= 0

        w = self.weight[in_s]  # [A, K]
        eps_post = self.state[active_nids.long(), F_EPS].unsqueeze(-1)  # [A, 1]
        pi_post = self.state[active_nids.long(), F_PI].unsqueeze(-1)  # [A, 1]
        z_pre = torch.where(valid, self.state[pre_ids.long(), F_Z], torch.zeros_like(w))

        eta_eff = eta * dopamine * pi_post

        # Hebb: eta * eps * z_pre
        hebb = eta_eff * eps_post * z_pre
        # Oja: -alpha * D * eps^2 * w  — 仅在 |w| > 0.8 时介入 (突触资源物理上限)
        oja_trigger = (w.abs() > 0.8).to(torch.float16)
        oja = -oja_alpha * dopamine * (eps_post**2) * w * oja_trigger

        dw = hebb + oja
        dw = torch.where(valid, dw, torch.zeros_like(dw))

        # 更新权重
        self.weight.scatter_add_(
            0,
            in_s[valid].long(),
            dw[valid],
        )
        # 更新 trace 和 age
        self.trace[in_s[valid].long()] = (
            self.trace[in_s[valid].long()] * 0.9 + z_pre[valid].abs() * 0.1
        )
        self.syn_age[in_s[valid].long()] += 1

        return float(dw.abs().sum().item())

    def hebbian_temporal(
        self,
        active_nids: torch.Tensor,
        eta: float,
        dopamine: float,
    ) -> float:
        """时序权重的 Hebbian 更新: dw = eta * D * eps * z_prev - 0.01 * w."""
        A = active_nids.shape[0]
        if A == 0:
            return 0.0

        connected = self.t_connected[active_nids.long()]
        if not connected.any():
            return 0.0

        nids_c = active_nids[connected]
        w = self.t_weight[nids_c.long()]
        eps = self.state[nids_c.long(), F_EPS]
        z_prev = self.state[nids_c.long(), F_Z_PREV]

        dw = eta * dopamine * eps * z_prev - 0.01 * w
        self.t_weight[nids_c.long()] += dw
        return float(dw.abs().sum().item())

    def hebbian_topdown(
        self,
        active_nids: torch.Tensor,
        eta: float,
        dopamine: float,
    ) -> float:
        """Topdown 权重的 Hebbian 更新 (向量化, 零逐元素 .item()).

        仅更新连接到活跃神经元的 topdown 权重: dw = eta * dopamine * eps * z_pre.
        """
        if active_nids.shape[0] == 0:
            return 0.0

        td_alive = self.td_alive
        if not td_alive.any():
            return 0.0

        td_indices = torch.where(td_alive)[0]
        post = self.td_post[td_indices].long()
        pre = self.td_pre[td_indices].long()

        # GPU 侧布尔掩码: post 在 active_nids 中 且 pre 存活
        active_mask = torch.isin(post, active_nids) & self.alive[pre]
        if not active_mask.any():
            return 0.0

        idx = td_indices[active_mask]
        eps = self.state[post[active_mask], F_EPS]
        z_pre_val = self.state[pre[active_mask], F_Z]
        w = self.td_weight[idx]
        dw = eta * dopamine * eps * z_pre_val
        self.td_weight[idx] = w + dw
        return float(dw.abs().sum().item())

    def hebbian_lm_head(
        self,
        top_layer: int,
        target_byte: int,
        eta: float,
        dopamine: float,
        pred_byte: int = -1,
    ) -> float:
        """LM head 权重的生物启发更新 — 误差门控 + 异突触竞争.

        机制:
        1. 误差门控: 猜对时不强化 (防止高频字节权重发散)
        2. 异突触可塑性: 目标字节 LTP 的同时, 错误预测字节承担 LTD
           (同一树突上的资源守恒 — 一个增强, 另一个减弱)

        Args:
            top_layer: 顶层索引
            target_byte: 目标字节 (0..255), -1 = 无监督
            eta: 学习率
            dopamine: 多巴胺调制
            pred_byte: 当前模型预测 (argmax logits), -1 = 未计算

        Returns:
            总绝对权重变化量.
        """
        top_mask = (self.layer == top_layer) & self.alive
        n_top = int(top_mask.sum().item())
        if n_top == 0:
            return 0.0

        z_top = self.state[top_mask, F_Z]  # [n_top]
        eta_eff = eta * dopamine

        # 计算预测 (若调用方未传入)
        if pred_byte < 0:
            logits = self.lm_weight[:, top_mask] @ z_top  # [256]
            pred_byte = int(logits.argmax().item())

        error = pred_byte != target_byte

        # 误差门控: 猜对时学习率降低到 10% (避免无限制强化)
        gate = 1.0 if error else 0.1

        # 异突触可塑性: dw 初始化为零, 仅更新参与错误的字节
        dw = torch.zeros(256, n_top, dtype=torch.float16, device=self.device)

        # LTP: 目标字节增强 (不管猜对猜错都轻微调整, 门控控制强度)
        dw[target_byte] = eta_eff * gate * z_top

        if error:
            # LTD: 错误预测字节削弱 (它"用错了资源", 需要被抑制)
            dw[pred_byte] = -eta_eff * gate * z_top * 0.5

        self.lm_weight[:, top_mask] += dw
        return float(dw.abs().sum().item())

    def adjust_thresholds(self, target_rate: float = 0.01, rate_eta: float = 0.01):
        """每步阈值调节 (稳态 homeostatic plasticity)."""
        alive = self.alive
        if alive.any():
            delta = rate_eta * (self.state[alive, F_FIRING_RATE] - target_rate)
            self.state[alive, F_THRESHOLD] = torch.clamp(
                self.state[alive, F_THRESHOLD] + delta, min=1e-4, max=0.5
            )

    def homeostasis_step(
        self,
        current_step: int,
        target_rate: float = 0.01,
        rate_eta: float = 0.001,
        prune_interval: int = 100,
        grow_interval: int = 200,
        max_prune: int = 10,
        max_grow: int = 2,
        max_inactive: int = 1000,
    ) -> dict:
        """周期性维护: 修剪 + 生长 (阈值调节已移至每步 adjust_thresholds).

        Returns:
            {action: count} 统计字典.
        """
        stats: dict = {"threshold_adjusted": 0}

        alive = self.alive

        # 1. 修剪 (包含孤儿检查 + 不活跃检查)
        if current_step % prune_interval == 0:
            alive_ids = torch.where(alive)[0]
            age = current_step - self.created_at[alive_ids]
            inactive_for = current_step - self.last_active[alive_ids]
            in_counts = self._in_counts[alive_ids.cpu()].to(self.device)

            # 隐藏层神经元需要更长 min_age
            layer_of_alive = self.layer[alive_ids]
            min_age = torch.where(layer_of_alive > 0, 500, 100)

            orphan_mask = (in_counts == 0) & (age > 500)  # 孤儿: 无入连接且活了很久
            inactive_mask = (
                (inactive_for > max_inactive)
                & (self.state[alive_ids, F_FIRING_RATE] < 0.001)
                & (age > min_age)
            )
            prune_mask = orphan_mask | inactive_mask
            candidates = alive_ids[prune_mask][:max_prune]
            for nid in candidates.tolist():
                self.prune_neuron(int(nid))
            stats["pruned"] = len(candidates)
        else:
            stats["pruned"] = 0

        # 2. 生长 (分裂高误差神经元)
        if current_step % grow_interval == 0:
            alive_ids = torch.where(alive)[0]
            high_err = self.state[alive_ids, F_EPS].abs() > self.state[alive_ids, F_THRESHOLD] * 3.0
            candidates = alive_ids[high_err][:max_grow]
            grown = 0
            for nid in candidates.tolist():
                if self.split_neuron(int(nid)) is not None:
                    grown += 1
            stats["grown"] = grown
        else:
            stats["grown"] = 0

        return stats

    def homeostasis_lm_head(self, top_layer: int):
        """LM head 稳态缩放: 每列减去列均值 (模拟突触资源守恒).

        防止高频字节权重无限增长 → 低频字节被永久锁死.
        生物对应: 异突触长时程抑制 + 稳态缩放.

        列均值归零等价于: sum_i(lm_weight[i, j]) = 0 对所有隐藏神经元 j 成立.
        """
        top_mask = (self.layer == top_layer) & self.alive
        if not top_mask.any():
            return
        w_sub = self.lm_weight[:, top_mask]  # [256, n_top]
        col_mean = w_sub.mean(dim=0, keepdim=True)  # [1, n_top]
        self.lm_weight[:, top_mask] -= col_mean

    def compute_precision_scales(self, D: float, ACh: float, eta: float = 1.0):
        """更新精度权重 pi = 1 + eta*D*|eps| + eta*ACh*|eps|."""
        alive = self.alive
        if alive.any():
            e_abs = self.state[alive, F_EPS].abs()
            self.state[alive, F_PI] = 1.0 + eta * D * e_abs + eta * ACh * e_abs

    def finalize_step(self):
        """保存 z 到 z_prev (供下一步 temporal 预测)."""
        self.state[:, F_Z_PREV] = self.state[:, F_Z]

    def emit_active(self, current_time: int) -> torch.Tensor:
        """扫描全网络, 更新 firing_rate, 返回活跃神经元索引.

        firing: |eps| > threshold.
        """
        firing = (self.state[:, F_EPS].abs() > self.state[:, F_THRESHOLD]) & self.alive
        if firing.any():
            self.state[firing, F_FIRING_RATE] = self.state[firing, F_FIRING_RATE] * 0.95 + 0.05
            self.last_active[firing] = current_time
        return torch.where(firing)[0]

    # ═══════════════════════════════════════════════════════════════
    # 事件处理
    # ═══════════════════════════════════════════════════════════════

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
        idx = self._sensory_index

        matched_nids: list[int] = []
        matched_ev_idx: list[int] = []
        unmatched_ev_idx: list[int] = []

        for e in range(E):
            key = (layers[e], positions[e], channels[e])
            nid = idx.get(key)
            if nid is not None:
                # 索引中的神经元一定存活 (prune 时已从索引移除)
                matched_nids.append(nid)
                matched_ev_idx.append(e)
            else:
                unmatched_ev_idx.append(e)

        return matched_nids, matched_ev_idx, unmatched_ev_idx

    # ═══════════════════════════════════════════════════════════════
    # 时序投影操作
    # ═══════════════════════════════════════════════════════════════

    def temporal_connect(self, nids: list[int], init_scale: float = 0.1):
        """为神经元建立时序自连接."""
        for nid in nids:
            self.t_connected[nid] = True
            self.t_weight[nid] = random.gauss(0, init_scale)

    def temporal_disconnect(self, nid: int):
        """移除时序连接."""
        self.t_connected[nid] = False
        self.t_weight[nid] = 0.0

    # ═══════════════════════════════════════════════════════════════
    # Topdown 投影操作
    # ═══════════════════════════════════════════════════════════════

    def topdown_ensure(self, pre: int, post: int, init_scale: float = 0.05) -> int:
        """确保两神经元间有 topdown 连接, 返回连接索引."""
        # 检查是否已有
        for i in range(self._td_count):
            if (
                self.td_alive[i]
                and int(self.td_pre[i].item()) == pre
                and int(self.td_post[i].item()) == post
            ):
                return i
        # 新建
        idx = self._td_count
        if idx >= self.S_td:
            # 找已删除的槽位
            dead = torch.where(~self.td_alive)[0]
            if len(dead) > 0:
                idx = int(dead[0].item())
            else:
                return -1
        self.td_pre[idx] = pre
        self.td_post[idx] = post
        self.td_weight[idx] = random.gauss(0, init_scale)
        self.td_alive[idx] = True
        if idx == self._td_count:
            self._td_count += 1
        return idx

    def topdown_connect_active(
        self,
        upper_layer: int,
        lower_layer: int,
        max_per_upper: int = 8,
    ) -> int:
        """上层活跃神经元 → 下层神经元的 topdown 连接."""
        upper = self.get_active_neurons(upper_layer).tolist()
        lower = self.get_neurons_by_layer(lower_layer).tolist()
        if not upper or not lower:
            return 0

        count = 0
        for pre in upper:
            candidates = random.sample(lower, min(max_per_upper, len(lower)))
            for post in candidates:
                self.topdown_ensure(pre, post)
                count += 1
        return count

    # ═══════════════════════════════════════════════════════════════
    # LM Head 操作
    # ═══════════════════════════════════════════════════════════════

    def lm_ensure_top_connected(self, top_layer: int, connections_per_logit: int = 16):
        """确保顶层神经元已连接到 LM head."""
        top_mask = (self.layer == top_layer) & self.alive
        top_ids = torch.where(top_mask)[0].tolist()
        if not top_ids:
            return

        for nid in top_ids:
            # 检查是否已有连接
            if self.lm_weight[:, nid].abs().sum() > 0:
                continue
            # 随机选 connections_per_logit 个 logit
            chosen = random.sample(range(256), min(connections_per_logit, 256))
            for logit_idx in chosen:
                self.lm_weight[logit_idx, nid] = random.gauss(0, 0.02)

    # ═══════════════════════════════════════════════════════════════
    # 序列化
    # ═══════════════════════════════════════════════════════════════

    def state_dict(self) -> dict:
        """返回可序列化的状态字典 (CPU tensors)."""
        return {
            "state": self.state.cpu().clone(),
            "layer": self.layer.cpu().clone(),
            "position": self.position.cpu().clone(),
            "channel": self.channel.cpu().clone(),
            "created_at": self.created_at.cpu().clone(),
            "last_active": self.last_active.cpu().clone(),
            "alive": self.alive.cpu().clone(),
            "pre_id": self.pre_id.cpu().clone(),
            "post_id": self.post_id.cpu().clone(),
            "weight": self.weight.cpu().clone(),
            "trace": self.trace.cpu().clone(),
            "syn_age": self.syn_age.cpu().clone(),
            "syn_alive": self.syn_alive.cpu().clone(),
            "in_ptrs": self.in_ptrs.cpu().clone(),
            "out_ptrs": self.out_ptrs.cpu().clone(),
            "t_weight": self.t_weight.cpu().clone(),
            "t_connected": self.t_connected.cpu().clone(),
            "td_pre": self.td_pre.cpu().clone(),
            "td_post": self.td_post.cpu().clone(),
            "td_weight": self.td_weight.cpu().clone(),
            "td_alive": self.td_alive.cpu().clone(),
            "td_count": self._td_count,
            "lm_weight": self.lm_weight.cpu().clone(),
            "occupied_neurons": self._occupied_neurons,
            "occupied_synapses": self._occupied_synapses,
            "total_created": self._total_created,
            "total_pruned": self._total_pruned,
            "total_syn_created": self._total_syn_created,
            "total_syn_pruned": self._total_syn_pruned,
        }

    def load_state_dict(self, sd: dict):
        """从状态字典恢复."""
        for key in [
            "state",
            "layer",
            "position",
            "channel",
            "created_at",
            "last_active",
            "alive",
            "pre_id",
            "post_id",
            "weight",
            "trace",
            "syn_age",
            "syn_alive",
            "in_ptrs",
            "out_ptrs",
            "t_weight",
            "t_connected",
            "td_pre",
            "td_post",
            "td_weight",
            "td_alive",
            "lm_weight",
        ]:
            if key in sd:
                t = sd[key].to(self.device)
                setattr(self, key, t)

        self._td_count = sd.get("td_count", 0)
        self._occupied_neurons = sd.get("occupied_neurons", int(self.alive.sum().item()))
        self._occupied_synapses = sd.get("occupied_synapses", int(self.syn_alive.sum().item()))
        self._total_created = sd.get("total_created", self._occupied_neurons)
        self._total_pruned = sd.get("total_pruned", 0)
        self._total_syn_created = sd.get("total_syn_created", self._occupied_synapses)
        self._total_syn_pruned = sd.get("total_syn_pruned", 0)

        # 重建 CPU 辅助 (用 tensor 操作, 避免逐元素 .item())
        dead = torch.where(~self.alive)[0].tolist()
        self._free_neurons = dead if dead else list(range(self.N))
        dead_syn = torch.where(~self.syn_alive)[0].tolist()
        self._free_synapses = dead_syn if dead_syn else list(range(self.S))
        # 批量计算邻接计数 (CPU 侧)
        valid_in = self.in_ptrs >= 0
        valid_out = self.out_ptrs >= 0
        self._in_counts = valid_in.sum(dim=-1).cpu().to(torch.int32)
        self._out_counts = valid_out.sum(dim=-1).cpu().to(torch.int32)
        # 维护感官索引
        self._sensory_index.clear()
        for nid in torch.where(self.alive & (self.position >= 0))[0].tolist():
            self._sensory_index[
                (
                    int(self.layer[nid].item()),
                    int(self.position[nid].item()),
                    int(self.channel[nid].item()),
                )
            ] = nid
