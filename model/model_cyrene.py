"""Cyrene 模型 — 全张量化 PC 持续学习.

基于 TensorNeuronPool + EventBridge 的 device-agnostic 实现.
可在 CPU 或 CUDA 上运行, 所有计算批量执行, 零逐元素传输.

架构:
    SensoryFrontend (Conv1D) → EventBridge (GPU top-k)
    → TensorNeuronPool (scatter_add 预测 + 批量 Hebbian)
    + 时序/topdown/LM 投影 + 神经调制 + 稳态
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Callable

import torch

from model.pc.homeostasis import homeostasis_step
from model.pc.neuromodulation import (
    combine_modulation,
    compute_ach,
    compute_dopamine,
    compute_precision_scales,
    compute_uncertainty,
)
from model.pc.tensor_pool import TensorNeuronPool
from pkg.device.cuda import setup_cuda_device
from pkg.device.event_bridge import EventBridge, SensoryEventBatch
from pkg.device.sensory_frontend import SensoryFrontend


@dataclass
class CyreneConfig:
    """Cyrene 模型配置."""

    hidden_size: int = 64
    num_hidden_layers: int = 1
    max_neurons: int = 65536
    max_synapses: int = 8_000_000
    K_fan: int = 128
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
    """Cyrene 模型 — 全张量化字节级 PC 持续学习.

    Usage:
        config = CyreneConfig(hidden_size=64)
        model = CyreneModel(config)
        model.add_hidden_layer(256)
        stats = model.step(byte_seq)
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
        self.device = setup_cuda_device()

        # GPU 感官前端
        self.frontend = SensoryFrontend(h_front=config.hidden_size)
        self.frontend.eval()
        self.frontend = self.frontend.to(self.device)

        # 全张量化神经元池
        self.pool = TensorNeuronPool(
            max_neurons=config.max_neurons,
            max_synapses=config.max_synapses,
            K=config.K_fan,
            device=self.device,
        )

        # 事件桥接 (tensor 版本)
        self.bridge = EventBridge(
            h_front=config.hidden_size,
            sensory_threshold=0.05,
            device=self.device,
        )

        # 运行状态
        self._step: int = 0
        self._top_layer: int = 0
        self._hidden_layer_created: bool = False
        self._free_energy_history: list[float] = []

        # 神经调制状态
        self._last_D: float = 0.5
        self._last_ACh: float = 0.5
        self._last_modulation: float = 0.5
        self._last_pred_byte: int = -1

        # warmup
        self.bridge.set_warmup(config.warmup_steps)

        # hook 系统
        self._hooks: dict[str, list[Callable]] = {e: [] for e in self._EVENTS}

    # ═══════════════════════════════════════════════════════════════
    # Hook 系统
    # ═══════════════════════════════════════════════════════════════

    def hook(self, event: str, fn: Callable):
        if event not in self._EVENTS:
            raise ValueError(f"Unknown hook event: {event}")
        self._hooks.setdefault(event, []).append(fn)

    def unhook(self, event: str, fn: Callable | None = None):
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
        """添加隐藏层: 创建神经元, 建立感官→隐藏连接, 分配时序投影."""
        nids = [self.pool.create_neuron(layer=to_layer) for _ in range(n_neurons)]
        self.pool.connect_layer(from_layer, to_layer, connection_density)
        self.pool.temporal_connect(nids)
        self._top_layer = max(self._top_layer, to_layer)
        self._hidden_layer_created = True

    # ═══════════════════════════════════════════════════════════════
    # 显式阶段方法
    # ═══════════════════════════════════════════════════════════════

    @torch.inference_mode()
    def encode(self, byte_seq: torch.Tensor) -> list[torch.Tensor]:
        """Stage 1: GPU 感官编码."""
        self._fire("before_encode", byte_seq=byte_seq)
        h_list = self.frontend(byte_seq)
        self._fire("after_encode", h_list=h_list)
        return h_list

    @torch.inference_mode()
    def ingest(self, h_list: list[torch.Tensor], top_k: int = 0) -> SensoryEventBatch | None:
        """Stage 2: 特征 → 事件 batch (GPU top-k, 零 .item())."""
        return self.bridge.ingest_hlist(h_list, top_k=top_k)

    @torch.inference_mode()
    def process_sensory_events(self, events: SensoryEventBatch | None) -> int:
        """Stage 3: 批量处理感官事件 — 匹配/创建神经元 + 预测 + 更新."""
        self._fire("before_sensory")
        if events is None or len(events) == 0:
            self._fire("after_sensory", n_processed=0)
            return 0

        E = len(events)
        max_ev = 500
        if E > max_ev:
            events = SensoryEventBatch(
                pos=events.pos[:max_ev],
                ch=events.ch[:max_ev],
                val=events.val[:max_ev],
                layer=events.layer[:max_ev],
                block_id=events.block_id[:max_ev],
            )
            E = max_ev

        # 一次 .cpu() 转移全部, 避免逐元素 CUDA 同步
        pos_cpu = events.pos.cpu()
        ch_cpu = events.ch.cpu()
        val_cpu = events.val.cpu()
        layer_cpu = events.layer.cpu()

        # O(1) hash 匹配 + 构建目标列表 (全部在 CPU 侧)
        matched_nids, unmatched_mask, matched_map = self.pool.match_sensory_events(
            pos_cpu, ch_cpu, layer_cpu, val_cpu
        )

        all_nids_list: list[int] = []
        z_new_list: list[float] = []

        for e in range(E):
            val_e = float(val_cpu[e].item())
            if unmatched_mask[e]:
                nid = self.pool.create_neuron(
                    layer=int(layer_cpu[e].item()),
                    position=int(pos_cpu[e].item()),
                    channel=int(ch_cpu[e].item()),
                    threshold=0.05,
                )
                all_nids_list.append(nid)
                z_new_list.append(val_e)
            else:
                idx = int(matched_map[e].item())
                nid = int(matched_nids[idx].item())
                all_nids_list.append(nid)
                z_new_list.append(val_e)

        if not all_nids_list:
            self._fire("after_sensory", n_processed=0)
            return 0

        all_nids = torch.tensor(all_nids_list, dtype=torch.int32, device=self.device)
        z_new = torch.tensor(z_new_list, dtype=torch.float16, device=self.device)
        self.pool.update_batch(all_nids, z_new)

        n_processed = len(all_nids_list)
        self._fire("after_sensory", n_processed=n_processed)
        return n_processed

    @torch.inference_mode()
    def process_network_events(self, max_events: int = 10) -> int:
        """Stage 4: 批量处理网络事件."""
        nids, eps = self.bridge.pop_network_events(max_events)
        if nids.shape[0] == 0:
            return 0

        # 更新 (使用当前 z, 重算预测)
        self.pool.update_batch(nids, self.pool.state[nids.long(), 0])
        return nids.shape[0]

    @torch.inference_mode()
    def predict_pass(self):
        """Stage 5: 时序 + 自上而下预测."""
        self._fire("before_predict")
        self.pool.temporal_topdown_pass(self._top_layer)
        self._fire("after_predict")

    @torch.inference_mode()
    def compute_stats(self) -> tuple[float, float, int]:
        """Stage 6: 自由能 + LM logits."""
        free_energy = float(self.pool.compute_free_energy().item())
        pred_byte = -1
        if self._top_layer > 0 and self.pool.get_layer_width(self._top_layer) > 0:
            logits = self.pool.compute_lm_logits(self._top_layer)
            pred_byte = int(logits.argmax().item())
            self._last_pred_byte = pred_byte
        self._free_energy_history.append(free_energy)
        return free_energy, 0.0, pred_byte

    @torch.inference_mode()
    def modulate(self, free_energy: float):
        """Stage 7: 神经调质 D/ACh 计算."""
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
        """Stage 8: Hebbian + Oja 更新."""
        self._fire("before_hebbian", modulation=modulation)
        active = self.pool.get_active_neurons()
        if active.shape[0] > 0:
            eta = self.config.hebbian_base_eta
            self.pool.hebbian_pass(
                active, eta=eta, oja_alpha=self.config.oja_alpha, dopamine=modulation
            )
            eta_t = eta * 0.3
            self.pool.hebbian_temporal(active, eta=eta_t, dopamine=modulation)
            self.pool.hebbian_topdown(active, eta=eta_t, dopamine=modulation)
        self._fire("after_hebbian", n_active=int(active.shape[0]))

    @torch.inference_mode()
    def homeostasis_pass(self) -> dict:
        """Stage 8b: 稳态可塑性."""
        self._fire("before_homeostasis")
        hs = homeostasis_step(
            self.pool,
            self._step,
            prune_interval=self.config.prune_interval,
            grow_interval=self.config.grow_interval,
        )
        self._fire("after_homeostasis", hs_stats=hs)
        return hs

    @torch.inference_mode()
    def finalize_step(self):
        """Stage 9: 保存 z_prev."""
        self.pool.finalize_step()

    # ═══════════════════════════════════════════════════════════════
    # 核心 step
    # ═══════════════════════════════════════════════════════════════

    @torch.inference_mode()
    def step(self, byte_seq: torch.Tensor) -> dict:
        """执行一个完整步.

        Args:
            byte_seq: [1, 2, S] fp16 双通道字节编码

        Returns:
            统计字典.
        """
        self._step += 1
        is_warmup = self._step <= self.config.warmup_steps

        self._fire("before_step", byte_seq=byte_seq)

        h_list = self.encode(byte_seq)
        events = self.ingest(h_list, top_k=(0 if is_warmup else 4))
        self.process_sensory_events(events)
        self.process_network_events(max_events=10)
        self.predict_pass()
        free_energy, lm_loss, pred_byte = self.compute_stats()

        # 发射活跃神经元作为网络事件 (传给下一步)
        active = self.pool.emit_active(self._step)
        if active.shape[0] > 0:
            self.bridge.push_network_events(active, self.pool.state[active.long(), 2])

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

        activity = self.pool.get_activity_stats()
        stats = {
            "step": self._step,
            "free_energy": free_energy,
            "lm_loss": lm_loss,
            "pred_byte": pred_byte,
            "n_neurons": activity["total_neurons"],
            "n_synapses": activity["total_synapses"],
            "firing_rate": activity["avg_firing_rate"],
            "threshold": activity["avg_threshold"],
            "warmup": is_warmup,
            "D": self._last_D,
            "ACh": self._last_ACh,
            "modulation": self._last_modulation,
            "uncertainty": uncertainty,
            **hs_stats,
        }

        self._fire("after_step", stats=stats)
        return stats

    # ═══════════════════════════════════════════════════════════════
    # 高级 API
    # ═══════════════════════════════════════════════════════════════

    def run(self, byte_seq: torch.Tensor, n_steps: int = 1) -> list[dict]:
        stats_list = []
        for _ in range(n_steps):
            stats_list.append(self.step(byte_seq))
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
        """建立 topdown 连接."""
        return self.pool.topdown_connect_active(upper_layer, lower_layer, max_per_upper)

    def get_state(self) -> dict:
        activity = self.pool.get_activity_stats()
        return {
            "step": self._step,
            "pool_stats": activity,
            "bridge_stats": self.bridge.get_stats(),
            "free_energy": (self._free_energy_history[-1] if self._free_energy_history else 0.0),
            "warmup_remaining": max(0, self.config.warmup_steps - self._step),
            "D": self._last_D,
            "ACh": self._last_ACh,
            "modulation": self._last_modulation,
            "last_pred_byte": self._last_pred_byte,
            "temporal_connections": int(self.pool.t_connected.sum().item()),
            "topdown_connections": int(self.pool.td_alive.sum().item()),
        }

    def warmup(self, n_steps: int = 20):
        dummy = torch.zeros(1, 2, self.config.hidden_size, dtype=torch.half, device=self.device)
        for _ in range(n_steps):
            self.step(dummy)

    def save(self, path: str):
        state = {
            "config": {f.name: getattr(self.config, f.name) for f in fields(CyreneConfig)},
            "pool": self.pool.state_dict(),
            "stats": self.get_state(),
            "step": self._step,
            "top_layer": self._top_layer,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(state, path)

    @classmethod
    def load(cls, path: str) -> CyreneModel:
        data = torch.load(path, map_location="cpu", weights_only=True)
        config = CyreneConfig(**data.get("config", {}))
        model = cls(config)
        if "pool" in data:
            model.pool.load_state_dict(data["pool"])
        model._step = data.get("step", 0)
        model._top_layer = data.get("top_layer", 0)
        if config.num_hidden_layers > 0:
            model._hidden_layer_created = True
        model.bridge.set_warmup(max(0, config.warmup_steps - model._step))
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
