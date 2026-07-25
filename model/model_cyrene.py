"""Cyrene 模型
字节级 PC 持续学习模型定义。

这是整个项目中唯一的模型入口模块。
CyreneModel 是项目唯一的模型类，包含完整的架构定义和运行逻辑。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

import torch

from model.pc.homeostasis import homeostasis_step
from model.pc.neuron_pool import NeuronPool
from model.pc.neuromodulation import (
    combine_modulation,
    compute_ach,
    compute_dopamine,
    compute_precision_scales,
    compute_uncertainty,
)
from model.pc.sparse_forward import _to_fp16, batch_hebbian
from model.pc.sparse_projections import (
    SparseLMHead,
    SparseTemporalSelf,
    SparseTopdown,
)
from pkg.device.event_bridge import EventBridge
from pkg.device.sensory_frontend import SensoryFrontend


@dataclass
class CyreneConfig:
    """Cyrene 模型配置."""

    hidden_size: int = 64
    num_hidden_layers: int = 1
    max_neurons: int = 65536
    warmup_steps: int = 50
    hebbian_base_eta: float = 3e-4
    oja_alpha: float = 0.05
    ach_beta_0: float = 0.0
    hidden_neurons: int = 256
    connection_density: float = 0.2
    prune_interval: int = 100
    grow_interval: int = 200
    homeostasis_interval: int = 50


class CyreneModel:
    """Cyrene 模型 — 字节级 PC 持续学习模型。

    架构:
        SensoryFrontend (GPU) → EventBridge → NeuronPool (CPU)
        ├─ temporal self-prediction (SparseTemporalSelf)
        ├─ topdown prediction (SparseTopdown)
        └─ lm_head 字节预测 (SparseLMHead)
    可塑性: Hebbian + Oja, 多巴胺/乙酰胆碱调制
    无 train/eval 模式: 启动即持续运行.

    Usage:
        config = CyreneConfig(hidden_size=64)
        model = CyreneModel(config)
        model.add_hidden_layer(256)
        stats = model.step(byte_seq)
    """

    def __init__(self, config: CyreneConfig):
        self.config = config

        # GPU 感官前端 — 将原始字节编码为特征
        self.frontend = SensoryFrontend(h_front=config.hidden_size)
        self.frontend.eval()

        # CPU 动态神经元池
        self.pool = NeuronPool(max_neurons=config.max_neurons)
        self.bridge = EventBridge(
            self.pool,
            h_front=config.hidden_size,
            sensory_threshold=0.05,
        )

        # 稀疏投影模块
        self.temporal = SparseTemporalSelf(init_scale=0.1)
        self.topdown = SparseTopdown(connection_density=0.3, init_scale=0.05)
        self.lm_head = SparseLMHead(connections_per_logit=16, init_scale=0.02)

        # 运行状态
        self._step: int = 0
        self._top_layer: int = 0
        self._hidden_layer_created: bool = False
        self._free_energy_history: list[float] = []
        self._n_sensory_events: int = 0
        self._n_network_events: int = 0

        # 神经调制状态
        self._last_D: float = 0.5
        self._last_ACh: float = 0.5
        self._last_modulation: float = 0.5

        # lm_head 状态
        self._last_logits: list[float] | None = None
        self._last_lm_loss: float = 0.0
        self._last_pred_byte: int = -1

        # warmup
        self.bridge.set_warmup(config.warmup_steps)

    # ═══════════════════════════════════════════════════════════════
    # 架构构建
    # ═══════════════════════════════════════════════════════════════

    def add_hidden_layer(
        self,
        n_neurons: int,
        from_layer: int = 0,
        to_layer: int = 7,
        connection_density: float = 0.2,
    ):
        """添加隐藏层.

        创建神经元, 建立感官→隐藏前馈连接, 分配 temporal 自连接.
        """
        nids = [self.pool.create_neuron(layer=to_layer).id for _ in range(n_neurons)]
        self.pool.connect_layer(from_layer, to_layer, connection_density)
        self.temporal.connect_batch(nids)
        self._top_layer = max(self._top_layer, to_layer)
        self._hidden_layer_created = True

    # ═══════════════════════════════════════════════════════════════
    # 核心运行逻辑
    # ═══════════════════════════════════════════════════════════════

    @torch.inference_mode()
    def step(self, byte_seq: torch.Tensor) -> dict:
        """执行一个完整步的事件驱动处理.

        Args:
            byte_seq: [1, 2, S] fp16 双通道字节编码

        Returns:
            统计字典: free_energy, lm_loss, n_neurons, D, ACh, ...
        """
        self._step += 1
        is_warmup = self._step <= self.config.warmup_steps

        # 1. GPU: 感官编码
        h_list = self.frontend(byte_seq)

        # 2. Bridge: 特征 → 事件队列
        self.bridge.ingest_hlist(h_list, top_k=(0 if is_warmup else 4))

        # 3. CPU: 消费感官事件
        n_sensory = self.bridge.process_sensory_events(max_events=500)
        self._n_sensory_events += n_sensory

        # 4. CPU: 稀疏传播
        n_network = self.bridge.process_network_events(max_events=10)
        self._n_network_events += n_network

        # 5. 时序 + 自上而下预测
        self._temporal_topdown_pass()

        # 6. 自由能 + 字节预测
        free_energy = self._compute_free_energy()
        lm_loss, pred_byte = self._compute_lm()
        self._free_energy_history.append(free_energy)

        # 7. 稳态可塑性
        hs_stats = {}
        if self._step % self.config.homeostasis_interval == 0:
            hs_stats = homeostasis_step(
                self.pool,
                self._step,
                target_rate=0.01,
                prune_interval=self.config.prune_interval,
                grow_interval=self.config.grow_interval,
            )

        # 8. 神经调制 + Hebbian 更新
        D = self._last_D if hasattr(self, "_last_D") else 0.5
        ACh = self._last_ACh if hasattr(self, "_last_ACh") else 0.5
        modulation = self._last_modulation if hasattr(self, "_last_modulation") else 1.0
        uncertainty = 0.5

        if not is_warmup and free_energy > 1e-8:
            uncertainty = compute_uncertainty(self._free_energy_history)
            F_prev = (
                self._free_energy_history[-2]
                if len(self._free_energy_history) >= 2
                else free_energy
            )
            D = compute_dopamine(free_energy, F_prev)
            ACh = compute_ach(uncertainty, self.config.ach_beta_0)
            modulation = combine_modulation(D, ACh)
            self._last_D = D
            self._last_ACh = ACh
            self._last_modulation = modulation
            compute_precision_scales(self.pool, D, ACh)

            active = [n.id for n in self.pool.neurons.values() if abs(n.ε) > n.threshold]
            if active:
                batch_hebbian(
                    self.pool,
                    active,
                    eta=self.config.hebbian_base_eta,
                    oja_alpha=self.config.oja_alpha,
                    dopamine=modulation,
                )
                for nid in active:
                    n = self.pool.neurons[nid]
                    eta_ = self.config.hebbian_base_eta * 0.3
                    self.temporal.hebbian_step(nid, n._z_prev, n.ε, eta=eta_, dopamine=modulation)
                    self.topdown.hebbian_step(self.pool, nid, n.ε, eta=eta_, dopamine=modulation)

        # 9. 保存 z(t) 供下一轮使用
        self._store_prev_z()

        return {
            "step": self._step,
            "n_sensory_events": n_sensory,
            "n_network_events": n_network,
            "free_energy": free_energy,
            "lm_loss": lm_loss,
            "pred_byte": pred_byte,
            "n_neurons": self.pool.get_total_neurons(),
            "n_synapses": self.pool.get_total_synapses(),
            "firing_rate": self.pool.get_activity_stats()["avg_firing_rate"],
            "threshold": self.pool.get_activity_stats()["avg_threshold"],
            "warmup": is_warmup,
            "D": D,
            "ACh": ACh,
            "modulation": modulation,
            "uncertainty": uncertainty,
            **hs_stats,
        }

    # ── 内部辅助 ──────────────────────────────────────────────────

    def _temporal_topdown_pass(self):
        """时序预测 + 自上而下预测，更新神经元 μ / ε."""
        for neuron in self.pool.neurons.values():
            if neuron.id not in self.temporal:
                continue
            mu_temp = self.temporal.predict(neuron.id, neuron._z_prev)
            if mu_temp is not None:
                neuron.μ = _to_fp16(neuron.μ + mu_temp)
                neuron.ε = _to_fp16(neuron.z - neuron.μ)

        if self._top_layer > 0:
            sensory_ids = set(self.pool.layer_groups.get(0, set()))
            for neuron in self.pool.neurons.values():
                if neuron.id not in sensory_ids:
                    continue
                mu_td = self.topdown.predict(self.pool, neuron.id)
                if mu_td != 0.0:
                    neuron.μ = _to_fp16(neuron.μ + mu_td)
                    neuron.ε = _to_fp16(neuron.z - neuron.μ)

    def _compute_free_energy(self) -> float:
        return sum(n.ε**2 for n in self.pool.neurons.values())

    def _compute_lm(self) -> tuple[float, int]:
        if self._top_layer > 0 and self.pool.layer_groups.get(self._top_layer):
            logits = self.lm_head.predict_logits(self.pool, self._top_layer)
            self._last_logits = logits
            pred = max(range(256), key=lambda i: logits[i])
            self._last_pred_byte = pred
            self._last_lm_loss = 0.0
        return self._last_lm_loss, self._last_pred_byte

    def _store_prev_z(self):
        for neuron in self.pool.neurons.values():
            neuron._z_prev = neuron.z

    # ═══════════════════════════════════════════════════════════════
    # 高级 API
    # ═══════════════════════════════════════════════════════════════

    def run(self, byte_seq: torch.Tensor, n_steps: int = 1) -> list[dict]:
        stats_list = []
        for t in range(n_steps):
            start = min(t, max(0, byte_seq.shape[-1] - 128))
            window = byte_seq[..., start : start + 128]
            if window.shape[-1] < 13:
                window = torch.nn.functional.pad(window, (0, 13 - window.shape[-1]))
            stats_list.append(self.step(window))
        return stats_list

    def ingest_stream(
        self, byte_stream: bytes, positions_per_step: int = 128, batch_size: int = 1
    ) -> int:
        processed = 0
        for t in range(0, max(1, len(byte_stream) - 13), positions_per_step):
            chunk = byte_stream[t : t + positions_per_step + 12]
            if len(chunk) < 13:
                break
            byte_vals = (
                torch.tensor([b / 128.0 - 1.0 for b in chunk], dtype=torch.half)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            mask = torch.ones_like(byte_vals)
            seq = torch.cat([byte_vals, mask], dim=1)
            self.step(seq)
            processed += 1
        return processed

    def connect_topdown(self, upper_layer: int, lower_layer: int, max_per_upper: int = 8):
        return self.topdown.connect_active(self.pool, upper_layer, lower_layer, max_per_upper)

    def get_state(self) -> dict:
        activity = self.pool.get_activity_stats()
        return {
            "step": self._step,
            "pool_stats": activity,
            "bridge_stats": self.bridge.get_stats(),
            "free_energy": (self._free_energy_history[-1] if self._free_energy_history else 0.0),
            "total_sensory_events": self._n_sensory_events,
            "total_network_events": self._n_network_events,
            "warmup_remaining": max(0, self.config.warmup_steps - self._step),
            "D": self._last_D,
            "ACh": self._last_ACh,
            "modulation": self._last_modulation,
            "lm_head_stats": self.lm_head.get_stats() if self._top_layer > 0 else {},
            "last_pred_byte": self._last_pred_byte,
            "last_lm_loss": self._last_lm_loss,
            "temporal_stats": self.temporal.get_stats() if self._top_layer > 0 else {},
            "topdown_connections": len(self.topdown),
        }

    def warmup(self, n_steps: int = 20):
        dummy = torch.zeros(1, 2, self.config.hidden_size, dtype=torch.half)
        for _ in range(n_steps):
            self.step(dummy)

    def save(self, path: str):
        state = {
            "config": {f.name: getattr(self.config, f.name) for f in fields(CyreneConfig)},
            "stats": self.get_state(),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(state, path)

    @classmethod
    def load(cls, path: str) -> CyreneModel:
        data = torch.load(path, map_location="cpu", weights_only=True)
        config = CyreneConfig(**data.get("config", {}))
        model = cls(config)
        if config.num_hidden_layers > 0:
            model.add_hidden_layer(config.hidden_neurons)
        model.warmup(min(config.warmup_steps, 20))
        return model

    @property
    def step_count(self) -> int:
        return self._step

    @property
    def h_front(self) -> int:
        return self.config.hidden_size


# ═══════════════════════════════════════════════════════════════════
# 工厂函数
# ═══════════════════════════════════════════════════════════════════


def create_cyrene(config: CyreneConfig | None = None) -> CyreneModel:
    """创建并初始化一个 Cyrene 模型."""
    config = config or CyreneConfig()
    model = CyreneModel(config)
    if config.num_hidden_layers > 0:
        model.add_hidden_layer(config.hidden_neurons)
    return model


def load_cyrene_checkpoint(path: str, config: CyreneConfig | None = None) -> CyreneModel:
    """从检查点加载 Cyrene 模型."""
    if os.path.exists(path):
        return CyreneModel.load(path)
    return create_cyrene(config)
