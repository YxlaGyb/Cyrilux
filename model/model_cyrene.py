"""Cyrene 模型
字节级 PC 持续学习模型定义。

这是整个项目中唯一的模型入口模块。
CyreneModel 是项目唯一的模型类，包含完整的架构定义和运行逻辑。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Callable

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
from model.pc.sparse_forward import (
    _to_fp16,
    batch_hebbian,
    emit_if_active,
    hebbian_step,
    predict_neuron,
    update_neuron,
)
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


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CyreneModel:
    """Cyrene 模型 — 字节级 PC 持续学习模型。

    架构:
        SensoryFrontend (GPU) → EventBridge → NeuronPool (CPU)
        ├─ temporal self-prediction (SparseTemporalSelf)
        ├─ topdown prediction (SparseTopdown)
        └─ lm_head 字节预测 (SparseLMHead)
    可塑性: Hebbian + Oja, 多巴胺/乙酰胆碱调制
    无 train/eval 模式: 启动即持续运行.
    提供 stage 方法将 step 拆分为可独立调用的阶段, 以及 hook 扩展点.

    Usage:
        config = CyreneConfig(hidden_size=64)
        model = CyreneModel(config)
        model.add_hidden_layer(256)
        model.hook('after_step', my_logger)
        stats = model.step(byte_seq)
        # 或显式分阶段:
        h = model.encode(byte_seq)
        model.ingest(h)
        model.process_sensory_events()
        ...
    """

    _EVENTS = [
        "before_step", "after_step",
        "before_encode", "after_encode",
        "before_sensory", "after_sensory",
        "before_predict", "after_predict",
        "before_modulate", "after_modulate",
        "before_hebbian", "after_hebbian",
        "before_homeostasis", "after_homeostasis",
    ]

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
        self._last_pred_byte: int = -1

        # warmup
        self.bridge.set_warmup(config.warmup_steps)

        # hook 系统
        self._hooks: dict[str, list[Callable]] = {e: [] for e in self._EVENTS}

    # ═══════════════════════════════════════════════════════════════
    # Hook 系统
    # ═══════════════════════════════════════════════════════════════

    def hook(self, event: str, fn: Callable):
        """注册 hook fn(model, **ctx) 到 event."""
        if event not in self._EVENTS:
            raise ValueError(f"Unknown hook event: {event}")
        self._hooks.setdefault(event, []).append(fn)

    def unhook(self, event: str, fn: Callable | None = None):
        """移除 hook. fn=None 清空该事件全部 hook."""
        if event not in self._EVENTS:
            raise ValueError(f"Unknown hook event: {event}")
        if fn is None:
            self._hooks[event].clear()
        else:
            self._hooks[event] = [h for h in self._hooks[event] if h is not fn]

    def _fire(self, event: str, **ctx):
        for fn in self._hooks.get(event, ()):
            fn(self, **ctx)

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
    # 显式阶段方法, 每一步可独立调用
    # ═══════════════════════════════════════════════════════════════

    @torch.inference_mode()
    def encode(self, byte_seq: torch.Tensor) -> list[torch.Tensor]:
        """Stage 1: GPU 感官编码."""
        self._fire("before_encode", byte_seq=byte_seq)
        h_list = self.frontend(byte_seq)
        self._fire("after_encode", h_list=h_list)
        return h_list

    def ingest(self, h_list: list[torch.Tensor], top_k: int = 0):
        """Stage 2: 特征 → 事件队列."""
        self.bridge.ingest_hlist(h_list, top_k=top_k)

    def process_sensory_events(self, max_events: int = 500) -> int:
        """Stage 3: 消费感官事件并更新神经元."""
        self._fire("before_sensory")
        events = self.bridge.sensory_queue.pop(max_events)
        n_processed = 0
        for ev in events:
            self._process_sensory_event(ev)
            n_processed += 1
        self._n_sensory_events += n_processed
        self._fire("after_sensory", n_processed=n_processed)
        return n_processed

    def process_network_events(self, max_events: int = 10) -> int:
        """Stage 4: 消费网络事件并传播."""
        events = self.bridge.network_queue.pop(max_events)
        n_processed = 0
        for ev in events:
            self._process_network_event(ev)
            n_processed += 1
        self._n_network_events += n_processed
        return n_processed

    @torch.inference_mode()
    def predict_pass(self):
        """Stage 5: 时序 + 自上而下预测."""
        self._fire("before_predict")
        self._temporal_topdown_pass()
        self._fire("after_predict")

    @torch.inference_mode()
    def compute_stats(self) -> tuple[float, float, int]:
        """Stage 6: 自由能 + LM loss."""
        free_energy = self._compute_free_energy()
        lm_loss, pred_byte = self._compute_lm()
        self._free_energy_history.append(free_energy)
        return free_energy, lm_loss, pred_byte

    @torch.inference_mode()
    def modulate(self, free_energy: float):
        """Stage 7: 神经调质 D/ACh 计算 (每步调用)."""
        self._fire("before_modulate", free_energy=free_energy)
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
        self._fire("after_modulate", D=D, ACh=ACh, modulation=modulation, uncertainty=uncertainty)

    @torch.inference_mode()
    def hebbian_pass(self, modulation: float):
        """Stage 8: Hebbian + Oja 更新 (非 warmup 时调用)."""
        self._fire("before_hebbian", modulation=modulation)
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
        self._fire("after_hebbian", n_active=len(active))

    @torch.inference_mode()
    def homeostasis_pass(self) -> dict:
        """Stage 8b: 稳态可塑性 (阈值/修剪/生长)."""
        self._fire("before_homeostasis")
        hs = homeostasis_step(
            self.pool,
            self._step,
            target_rate=0.01,
            prune_interval=self.config.prune_interval,
            grow_interval=self.config.grow_interval,
        )
        self._fire("after_homeostasis", hs_stats=hs)
        return hs

    @torch.inference_mode()
    def finalize_step(self):
        """Stage 9: 保存 z(t) 供下一轮使用."""
        self._store_prev_z()

    # ═══════════════════════════════════════════════════════════════
    # 事件处理 (私有 — 从 EventBridge 移入)
    # ═══════════════════════════════════════════════════════════════

    def _process_sensory_event(self, ev):
        """处理单条感官事件: 找/创建神经元 → 预测 → 更新 → Hebbian."""
        target_id = None
        for n in self.pool.get_neurons_by_layer(ev.layer):
            if n.position == ev.pos and n.channel == ev.channel:
                target_id = n.id
                break

        if target_id is None:
            n = self.pool.create_neuron(
                layer=ev.layer,
                position=ev.pos,
                channel=ev.channel,
                threshold=0.05,
            )
            target_id = n.id

        predict_neuron(self.pool, target_id)
        update_neuron(self.pool, target_id, z_new=ev.value)
        hebbian_step(self.pool, target_id, eta=3e-4)

    def _process_network_event(self, ev):
        """处理单条网络事件: 预测 → 更新 → Hebbian → 可选传播."""
        target = self.pool.neurons.get(ev.neuron_id)
        if target is None:
            return

        predict_neuron(self.pool, ev.neuron_id)
        update_neuron(self.pool, ev.neuron_id)
        hebbian_step(self.pool, ev.neuron_id, eta=3e-4)

        child_event = emit_if_active(
            self.pool,
            ev.neuron_id,
            current_time=self._step,
        )
        if child_event is not None:
            self.bridge.network_queue.push(child_event)

    # ═══════════════════════════════════════════════════════════════
    # 核心运行逻辑, 向后兼容的完整 step
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

        self._fire("before_step", byte_seq=byte_seq)

        h_list = self.encode(byte_seq)
        self.ingest(h_list, top_k=(0 if is_warmup else 4))
        self.process_sensory_events(max_events=500)
        self.process_network_events(max_events=10)
        self.predict_pass()
        free_energy, lm_loss, pred_byte = self.compute_stats()

        hs_stats: dict = {}
        if self._step % self.config.homeostasis_interval == 0:
            hs_stats = self.homeostasis_pass()

        if not is_warmup and free_energy > 1e-8:
            self.modulate(free_energy)
            self.hebbian_pass(self._last_modulation)

        uncertainty = (
            0.5
            if is_warmup or len(self._free_energy_history) < 3
            else compute_uncertainty(self._free_energy_history[-20:])
        )
        self.finalize_step()

        stats = {
            "step": self._step,
            "n_sensory_events": self._n_sensory_events,
            "n_network_events": self._n_network_events,
            "free_energy": free_energy,
            "lm_loss": lm_loss,
            "pred_byte": pred_byte,
            "n_neurons": self.pool.get_total_neurons(),
            "n_synapses": self.pool.get_total_synapses(),
            "firing_rate": self.pool.get_activity_stats()["avg_firing_rate"],
            "threshold": self.pool.get_activity_stats()["avg_threshold"],
            "warmup": is_warmup,
            "D": self._last_D,
            "ACh": self._last_ACh,
            "modulation": self._last_modulation,
            "uncertainty": uncertainty,
            **hs_stats,
        }

        self._fire("after_step", stats=stats)
        return stats

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
        """计算 LM head 预测, 返回 (0.0, pred_byte). CE loss 由 evaluation 模块独立计算."""
        last_pred = -1
        if self._top_layer > 0 and self.pool.layer_groups.get(self._top_layer):
            logits = self.lm_head.predict_logits(self.pool, self._top_layer)
            self._last_logits = logits
            last_pred = max(range(256), key=lambda i: logits[i])
            self._last_pred_byte = last_pred
        return 0.0, last_pred

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
