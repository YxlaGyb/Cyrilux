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


import torch


from .neuron import NeuronManager
from .page_storage import PAGE_NEURONS, PAGE_SYNAPSES, PageStorage
from .query import PoolQuery
from .forward import ForwardEngine
from .learning import LearningEngine
from .projections import ProjectionManager
from .synapse import SynapseManager


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
        K: int = 128,
        device: torch.device | str = "cpu",
        max_memory_bytes: int = 2_147_483_648,
        initial_neurons: int = PAGE_NEURONS,
        initial_synapses: int = PAGE_SYNAPSES,
    ):
        self.K = K
        self.device = torch.device(device) if isinstance(device, str) else device
        self._max_memory_bytes = max_memory_bytes

        # ── 页式存储 (对外暴露所有张量属性) ──
        self._storage = PageStorage(
            device=self.device,
            max_memory_bytes=max_memory_bytes,
            initial_neurons=initial_neurons,
            initial_synapses=initial_synapses,
        )

        # ── 张量属性 (从 _storage 直接引用, 扩容时 _storage 会更新引用) ──
        self.state = self._storage.state
        self.layer = self._storage.layer
        self.position = self._storage.position
        self.channel = self._storage.channel
        self.created_at = self._storage.created_at
        self.last_active = self._storage.last_active
        self.alive = self._storage.alive
        self.pre_id = self._storage.pre_id
        self.post_id = self._storage.post_id
        self.weight = self._storage.weight
        self.trace = self._storage.trace
        self.syn_age = self._storage.syn_age
        self.syn_alive = self._storage.syn_alive
        self.conn_type = self._storage.conn_type
        self.in_ptrs = self._storage.in_ptrs
        self.out_ptrs = self._storage.out_ptrs
        self._in_counts = self._storage._in_counts
        self._out_counts = self._storage._out_counts
        self.t_weight = self._storage.t_weight
        self.t_connected = self._storage.t_connected
        self.td_pre = self._storage.td_pre
        self.td_post = self._storage.td_post
        self.td_weight = self._storage.td_weight
        self.td_alive = self._storage.td_alive
        self.lm_weight = self._storage.lm_weight
        self.lm_bias = self._storage.lm_bias

        # ── 容量属性 (代理到 _storage) ──
        self._free_neurons: list[int] = self._storage._free_neurons
        self._free_synapses: list[int] = self._storage._free_synapses

        # ── 状态 ──
        self._td_count: int = 0
        self._occupied_neurons: int = 0
        self._occupied_synapses: int = 0
        self._sensory_index: dict[tuple[int, int, int], int] = {}

        # 统计
        self._total_created: int = 0
        self._total_pruned: int = 0
        self._total_syn_created: int = 0
        self._total_syn_pruned: int = 0

        # ── 组合子模块 ──
        self.query = PoolQuery(self)
        self.neuron = NeuronManager(self)
        self.synapse = SynapseManager(self)
        self.forward = ForwardEngine(self)
        self.learning = LearningEngine(self)
        self.projections = ProjectionManager(self)

    # ═══════════════════════════════════════════════════════════════
    # 容量属性 (代理到 PageStorage)
    # ═══════════════════════════════════════════════════════════════

    @property
    def N(self) -> int:
        return self._storage.N

    @property
    def S(self) -> int:
        return self._storage.S

    @property
    def S_td(self) -> int:
        return self._storage.S_td

    def _sync_storage_refs(self):
        """扩容后刷新张量引用 (_storage 可能已分配新 tensor)."""
        self.state = self._storage.state
        self.layer = self._storage.layer
        self.position = self._storage.position
        self.channel = self._storage.channel
        self.created_at = self._storage.created_at
        self.last_active = self._storage.last_active
        self.alive = self._storage.alive
        self.pre_id = self._storage.pre_id
        self.post_id = self._storage.post_id
        self.weight = self._storage.weight
        self.trace = self._storage.trace
        self.syn_age = self._storage.syn_age
        self.syn_alive = self._storage.syn_alive
        self.conn_type = self._storage.conn_type
        self.in_ptrs = self._storage.in_ptrs
        self.out_ptrs = self._storage.out_ptrs
        self._in_counts = self._storage._in_counts
        self._out_counts = self._storage._out_counts
        self.t_weight = self._storage.t_weight
        self.t_connected = self._storage.t_connected
        self.td_pre = self._storage.td_pre
        self.td_post = self._storage.td_post
        self.td_weight = self._storage.td_weight
        self.td_alive = self._storage.td_alive
        self.lm_weight = self._storage.lm_weight
        self.lm_bias = self._storage.lm_bias
        self._free_neurons = self._storage._free_neurons
        self._free_synapses = self._storage._free_synapses

    # ═══════════════════════════════════════════════════════════════
    # 结构操作 (委托到 NeuronManager / SynapseManager)
    # ═══════════════════════════════════════════════════════════════

    def create_neuron(self, layer: int, **kwargs) -> int:
        return self.neuron.create_neuron(layer, **kwargs)

    def create_neurons_batch(
        self, layers, positions=None, channels=None, thresholds=None, z_vals=None
    ) -> list[int]:
        return self.neuron.create_neurons_batch(layers, positions, channels, thresholds, z_vals)

    def create_synapse(self, pre: int, post: int, weight: float | None = None) -> int:
        return self.synapse.create_synapse(pre, post, weight)

    def create_synapses_batch(self, pre_ids, post_ids, weights=None, conn_type=0) -> int:
        return self.synapse.create_synapses_batch(pre_ids, post_ids, weights, conn_type)

    def prune_neuron(self, nid: int, force: bool = False) -> bool:
        return self.neuron.prune_neuron(nid, force)

    def split_neuron(self, nid: int, noise_scale: float = 0.05) -> int | None:
        return self.neuron.split_neuron(nid, noise_scale)

    def connect_layer(
        self, from_layer, to_layer, density=0.5, bias_strength=0.7, conn_type=0
    ) -> int:
        return self.synapse.connect_layer(from_layer, to_layer, density, bias_strength, conn_type)

    def _force_recycle(self) -> bool:
        return self.neuron._force_recycle()

    def _remove_from_in_ptrs(self, nid: int, sid: int):
        self.neuron._remove_from_in_ptrs(nid, sid)

    def _remove_td_for_neuron(self, nid: int):
        self.neuron._remove_td_for_neuron(nid)

    # ═══════════════════════════════════════════════════════════════
    # 查询 (委托到 PoolQuery)
    # ═══════════════════════════════════════════════════════════════

    def get_neurons_by_layer(self, layer: int) -> torch.Tensor:
        return self.query.get_neurons_by_layer(layer)

    def get_layer_width(self, layer: int) -> int:
        return self.query.get_layer_width(layer)

    def get_active_neurons(self, layer: int | None = None) -> torch.Tensor:
        return self.query.get_active_neurons(layer)

    def get_total_neurons(self) -> int:
        return self.query.get_total_neurons()

    def get_total_synapses(self) -> int:
        return self.query.get_total_synapses()

    def get_activity_stats(self) -> dict:
        return self.query.get_activity_stats()

    # ═══════════════════════════════════════════════════════════════
    # 批量前向操作 (委托到 ForwardEngine)
    # ═══════════════════════════════════════════════════════════════

    def predict_all(self) -> None:
        self.forward.predict_all()

    def predict_neurons(self, nids: torch.Tensor) -> torch.Tensor:
        return self.forward.predict_neurons(nids)

    def update_batch(self, nids: torch.Tensor, z_new: torch.Tensor) -> None:
        self.forward.update_batch(nids, z_new)

    def temporal_topdown_pass(self, top_layer: int):
        self.forward.temporal_topdown_pass(top_layer)

    def compute_free_energy(self) -> torch.Tensor:
        return self.forward.compute_free_energy()

    def compute_lm_logits(self, top_layer: int, use_mu: bool = False) -> torch.Tensor:
        return self.forward.compute_lm_logits(top_layer, use_mu)

    def compute_cross_entropy(self, logits: torch.Tensor, target_byte: int) -> torch.Tensor:
        return self.forward.compute_cross_entropy(logits, target_byte)

    # ═══════════════════════════════════════════════════════════════
    # 批量学习 (委托到 LearningEngine)
    # ═══════════════════════════════════════════════════════════════

    def hebbian_pass(self, active_nids, eta, oja_alpha, dopamine) -> float:
        return self.learning.hebbian_pass(active_nids, eta, oja_alpha, dopamine)

    def hebbian_temporal(self, active_nids, eta, dopamine) -> float:
        return self.learning.hebbian_temporal(active_nids, eta, dopamine)

    def hebbian_topdown(self, active_nids, eta, dopamine) -> float:
        return self.learning.hebbian_topdown(active_nids, eta, dopamine)

    def hebbian_lm_head(
        self, top_layer, target_byte, eta, dopamine, pred_byte=-1, use_mu=False
    ) -> float:
        return self.learning.hebbian_lm_head(
            top_layer, target_byte, eta, dopamine, pred_byte, use_mu
        )

    def adjust_thresholds(self, target_rate=0.01, rate_eta=0.01):
        self.learning.adjust_thresholds(target_rate, rate_eta)

    def synapse_turnover(self, step: int, rate: float = 0.02) -> int:
        return self.synapse.synapse_turnover(step, rate)

    def homeostasis_step(
        self,
        current_step,
        target_rate=0.01,
        rate_eta=0.001,
        prune_interval=100,
        grow_interval=200,
        max_prune=10,
        max_grow=2,
        max_inactive=1000,
    ) -> dict:
        return self.learning.homeostasis_step(
            current_step,
            target_rate,
            rate_eta,
            prune_interval,
            grow_interval,
            max_prune,
            max_grow,
            max_inactive,
        )

    def homeostasis_lm_head(self, top_layer: int):
        self.learning.homeostasis_lm_head(top_layer)

    def compute_precision_scales(self, D: float, ACh: float, eta: float = 1.0):
        self.learning.compute_precision_scales(D, ACh, eta)

    def finalize_step(self):
        self.learning.finalize_step()

    def emit_active(self, current_time: int) -> torch.Tensor:
        return self.forward.emit_active(current_time)

    # ═══════════════════════════════════════════════════════════════
    # 事件处理
    # ═══════════════════════════════════════════════════════════════

    def match_sensory_events(
        self,
        layers: list[int],
        positions: list[int],
        channels: list[int],
    ) -> tuple[list[int], list[int], list[int]]:
        return self.query.match_sensory_events(layers, positions, channels)

    # ═══════════════════════════════════════════════════════════════
    # 时序投影操作
    # ═══════════════════════════════════════════════════════════════
    # 投影操作 (委托到 ProjectionManager)
    # ═══════════════════════════════════════════════════════════════

    def temporal_connect(self, nids: list[int], init_scale: float = 0.1):
        self.projections.temporal_connect(nids, init_scale)

    def temporal_disconnect(self, nid: int):
        self.projections.temporal_disconnect(nid)

    def topdown_ensure(self, pre: int, post: int, init_scale: float = 0.05) -> int:
        return self.projections.topdown_ensure(pre, post, init_scale)

    def topdown_connect_layer(self, upper_layer, lower_layer, density=0.2) -> int:
        return self.projections.topdown_connect_layer(upper_layer, lower_layer, density)

    def topdown_connect_active(self, upper_layer, lower_layer, max_per_upper=8) -> int:
        return self.projections.topdown_connect_active(upper_layer, lower_layer, max_per_upper)

    def lm_ensure_top_connected(self, top_layer: int, connections_per_logit: int = 16):
        self.projections.lm_ensure_top_connected(top_layer, connections_per_logit)

    # ═══════════════════════════════════════════════════════════════
    # 序列化
    # ═══════════════════════════════════════════════════════════════

    def state_dict(self) -> dict:
        """稀疏序列化: 仅保存 alive 神经元/突触/topdown, 含原始槽位索引.

        _format=2: 稀疏格式, 兼容旧 _format=1 (或无 _format) 的全量格式.
        """
        alive_idx = torch.where(self.alive)[0]
        syn_idx = torch.where(self.syn_alive)[0]
        td_idx = torch.where(self.td_alive)[0]

        return {
            "_format": 2,
            "N": self.N,
            "S": self.S,
            "S_td": self.S_td,
            "K": self.K,
            # ── 神经元 (只存 alive 行 + 原始索引) ──
            "state": self.state[alive_idx].cpu().clone(),
            "alive_idx": alive_idx.cpu().clone(),
            "layer": self.layer[alive_idx].cpu().clone(),
            "position": self.position[alive_idx].cpu().clone(),
            "channel": self.channel[alive_idx].cpu().clone(),
            "created_at": self.created_at[alive_idx].cpu().clone(),
            "last_active": self.last_active[alive_idx].cpu().clone(),
            "t_weight": self.t_weight[alive_idx].cpu().clone(),
            "t_connected": self.t_connected[alive_idx].cpu().clone(),
            "in_ptrs": self.in_ptrs[alive_idx].cpu().clone(),
            "out_ptrs": self.out_ptrs[alive_idx].cpu().clone(),
            # ── 突触 (只存 alive 行 + 原始索引) ──
            "pre_id": self.pre_id[syn_idx].cpu().clone(),
            "post_id": self.post_id[syn_idx].cpu().clone(),
            "weight": self.weight[syn_idx].cpu().clone(),
            "trace": self.trace[syn_idx].cpu().clone(),
            "syn_age": self.syn_age[syn_idx].cpu().clone(),
            "syn_idx": syn_idx.cpu().clone(),
            # ── Topdown (只存 alive 行 + 原始索引) ──
            "td_pre": self.td_pre[td_idx].cpu().clone(),
            "td_post": self.td_post[td_idx].cpu().clone(),
            "td_weight": self.td_weight[td_idx].cpu().clone(),
            "td_idx": td_idx.cpu().clone(),
            # ── LM Head (只存 alive 列 + bias) ──
            "lm_weight": self.lm_weight[:, alive_idx].cpu().clone(),
            "lm_bias": self.lm_bias.cpu().clone(),
            # ── 统计 ──
            "occupied_neurons": self._occupied_neurons,
            "occupied_synapses": self._occupied_synapses,
            "total_created": self._total_created,
            "total_pruned": self._total_pruned,
            "total_syn_created": self._total_syn_created,
            "total_syn_pruned": self._total_syn_pruned,
        }

    def load_state_dict(self, sd: dict):
        """从稀疏格式恢复 (_format=2).  旧格式由迁移脚本处理."""
        alive_idx = sd["alive_idx"].to(self.device)
        syn_idx = sd["syn_idx"].to(self.device)
        td_idx = sd.get("td_idx", torch.zeros(0, dtype=torch.int32)).to(self.device)

        # 确保容量足够
        if len(alive_idx) > 0:
            self._storage.ensure_neuron_capacity(int(alive_idx.max().item()) + 1)
            self._sync_storage_refs()
        if len(syn_idx) > 0:
            self._storage.ensure_synapse_capacity(int(syn_idx.max().item()) + 1)
            self._sync_storage_refs()

        # Scatter 神经元数据
        self.state[alive_idx] = sd["state"].to(self.device)
        self.layer[alive_idx] = sd["layer"].to(self.device)
        self.position[alive_idx] = sd["position"].to(self.device)
        self.channel[alive_idx] = sd["channel"].to(self.device)
        self.created_at[alive_idx] = sd["created_at"].to(self.device)
        self.last_active[alive_idx] = sd["last_active"].to(self.device)
        self.alive[alive_idx] = True
        self.t_weight[alive_idx] = sd["t_weight"].to(self.device)
        self.t_connected[alive_idx] = sd["t_connected"].to(self.device)
        self.in_ptrs[alive_idx] = sd["in_ptrs"].to(self.device)
        self.out_ptrs[alive_idx] = sd["out_ptrs"].to(self.device)

        # Scatter 突触数据
        self.pre_id[syn_idx] = sd["pre_id"].to(self.device)
        self.post_id[syn_idx] = sd["post_id"].to(self.device)
        self.weight[syn_idx] = sd["weight"].to(self.device)
        self.trace[syn_idx] = sd["trace"].to(self.device)
        self.syn_age[syn_idx] = sd["syn_age"].to(self.device)
        self.syn_alive[syn_idx] = True

        # Topdown
        if len(td_idx) > 0:
            self.td_pre[td_idx] = sd["td_pre"].to(self.device)
            self.td_post[td_idx] = sd["td_post"].to(self.device)
            self.td_weight[td_idx] = sd["td_weight"].to(self.device)
            self.td_alive[td_idx] = True
        self._td_count = len(td_idx)

        # LM Head
        self.lm_weight[:, alive_idx] = sd["lm_weight"].to(self.device)
        if "lm_bias" in sd:
            self.lm_bias[:] = sd["lm_bias"].to(self.device)
        # 旧 checkpoint 无 lm_bias → 保持零初始化 (自动学到频次)

        # 统计
        self._occupied_neurons = sd.get("occupied_neurons", int(self.alive.sum().item()))
        self._occupied_synapses = sd.get("occupied_synapses", int(self.syn_alive.sum().item()))
        self._total_created = sd.get("total_created", self._occupied_neurons)
        self._total_pruned = sd.get("total_pruned", 0)
        self._total_syn_created = sd.get("total_syn_created", self._occupied_synapses)
        self._total_syn_pruned = sd.get("total_syn_pruned", 0)

        # 重建 freelists: 取 alive_mask 反向填充
        dead = torch.where(~self.alive)[0].tolist()
        self._storage._free_neurons.clear()
        self._storage._free_neurons.extend(dead if dead else list(range(self.N)))
        self._free_neurons = self._storage._free_neurons

        dead_syn = torch.where(~self.syn_alive)[0].tolist()
        self._storage._free_synapses.clear()
        self._storage._free_synapses.extend(dead_syn if dead_syn else list(range(self.S)))
        self._free_synapses = self._storage._free_synapses

        # 批量计算邻接计数
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
