"""Cyrene 模型

架构:
    TensorNeuronPool (scatter_add 预测 + 批量 Hebbian)
    + 时序/topdown/LM 投影 + 神经调制 + 稳态
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from dataclasses import dataclass, fields

import torch

from model.pc import (
    F_Z,
    TensorNeuronPool,
    combine_modulation,
    compute_ach,
    compute_dopamine,
    compute_precision_scales,
    compute_uncertainty,
)
from pkg.device.cuda import setup_cuda_device


@dataclass
class CyreneConfig:
    """Cyrene 模型配置."""

    hidden_size: int = 64
    num_hidden_layers: int = 1
    max_memory_bytes: int = 2_147_483_648  # 内存预算上限 (2GB)
    K_fan: int = 128
    warmup_steps: int = 50
    hebbian_base_eta: float = 3e-4
    oja_alpha: float = 0.05
    ach_beta_0: float = 0.0
    hidden_neurons: int = 256
    connection_density: float = 0.2
    bias_strength: float = 0.7  # connect_layer 偏好池权重 (0=随机, 1=纯本地)
    use_mu_lm: bool = True  # LM head 用 MU (有更强的输入选择性)
    prune_interval: int = 100
    grow_interval: int = 200
    homeostasis_interval: int = 50


class LMHead:
    """LM Head 薄封装 — 委托到 TensorNeuronPool 的 lm_weight 计算."""

    def __init__(self, pool):
        self._pool = pool

    def predict_logits(self, pool, top_layer: int, use_mu: bool = True) -> list[float]:
        """返回 256 个字节的 logits (list[float], CPU)."""
        logits = pool.compute_lm_logits(top_layer, use_mu=use_mu)
        return logits.cpu().tolist()

    def cross_entropy_loss(self, logits: list[float], target: int) -> float:
        """计算 CE loss."""
        import torch

        t = torch.tensor(logits, dtype=torch.float32).unsqueeze(0)
        target_t = torch.tensor([target])
        return float(torch.nn.functional.cross_entropy(t, target_t).item())


class CyreneModel:
    """Cyrene 模型 — 全张量化字节级 PC 持续学习.

    Usage:
        config = CyreneConfig(hidden_size=64)
        model = CyreneModel(config)
        model.add_hidden_layer()
        stats = model.step(byte_seq)
    """

    _EVENTS = [
        "before_step",
        "after_step",
        "before_encode",
        "after_encode",
        "before_sensory",
        "after_sensory",
        "before_predict",
        "after_predict",
        "before_modulate",
        "after_modulate",
        "before_hebbian",
        "after_hebbian",
        "before_homeostasis",
        "after_homeostasis",
    ]

    def __init__(self, config: CyreneConfig):
        self.config = config
        self.device = setup_cuda_device()

        # 页式动态神经元池 (自组织, 2GB 约束内生长/修剪)
        self.pool = TensorNeuronPool(
            K=config.K_fan,
            device=self.device,
            max_memory_bytes=config.max_memory_bytes,
        )

        # LM Head (委托到 pool)
        self.lm_head = LMHead(self.pool)

        # 运行状态
        self._step: int = 0
        self._top_layer: int = 0
        self._hidden_layer_created: bool = False
        self._pending_connects: list = []
        self._free_energy_history: list[float] = []

        # 神经调制状态
        self._last_D: float = 0.5
        self._last_ACh: float = 0.5
        self._last_modulation: float = 0.5
        self._last_pred_byte: int = -1

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

    def add_hidden_layer(self):
        """创建 5 层皮层架构:
        L4(感觉输入) → L2 → L3 → L5(输出) → L6(反馈调节).

        每个层有不同的惯性 (时间常数) 和 k-WTA 比例.
        连接延迟到 warmup 后 (感官神经元届时已存在).
        隐藏层神经元按 nid % 8 分组, 保留字节分组通路.
        """
        from model.pc import (
            CONN_FEEDBACK,
            CONN_FEEDFORWARD,
            LAYER_CONFIG,
            LAYER_L2,
            LAYER_L3,
            LAYER_L4,
            LAYER_L5,
            LAYER_L6,
            TOP_LAYER,
        )

        for layer, (n, alpha, kwta) in LAYER_CONFIG.items():
            nids = [self.pool.neuron.create_neuron(layer=layer) for _ in range(n)]
            self.pool.projections.temporal_connect(nids)
            # 隐藏层神经元分配信道标签, L4 按列分配 (0-255)
            if layer == LAYER_L4:
                per_ch = max(1, n // 256)
                for i, nid in enumerate(nids):
                    self.pool.channel[nid] = i // per_ch
            elif layer > 0:
                per_ch = n // 8
                for i, nid in enumerate(nids):
                    self.pool.channel[nid] = i // per_ch

        self._top_layer = TOP_LAYER
        self._hidden_layer_created = True

        # 延迟连接 (warmup 结束后触发)
        self._pending_connects = [
            (0, LAYER_L4, 0.25, CONN_FEEDFORWARD),
            (LAYER_L4, LAYER_L2, 0.25, CONN_FEEDFORWARD),
            (LAYER_L2, LAYER_L3, 0.30, CONN_FEEDFORWARD),
            (LAYER_L3, LAYER_L5, 0.30, CONN_FEEDFORWARD),
            (LAYER_L5, LAYER_L6, 0.30, CONN_FEEDFORWARD),
            (LAYER_L6, LAYER_L5, 0.20, CONN_FEEDBACK),
            (LAYER_L5, LAYER_L3, 0.20, CONN_FEEDBACK),
            (LAYER_L3, LAYER_L2, 0.15, CONN_FEEDBACK),
            (LAYER_L2, LAYER_L4, 0.15, CONN_FEEDBACK),
        ]

    # ═══════════════════════════════════════════════════════════════
    # 显式阶段方法
    # ═══════════════════════════════════════════════════════════════

    def encode(self, byte_ids: torch.Tensor) -> list[tuple[int, int]]:
        """
        Stage 1:
            字节ID → (position, byte_value) 对列表.
        byte_ids:
            [1, S] long (0..255).
        """
        vals = byte_ids.squeeze(0).tolist()  # [S]
        return [(pos, int(v)) for pos, v in enumerate(vals) if v != 0]

    def process_sensory_events(self, byte_events: list[tuple[int, int]]) -> int:
        """
        Stage 2+3: 按 (position, byte_value) 匹配/创建 L0 神经元.
        每个字节值在每个位置有专属神经元 — 纯离散, 零嵌入.
        """
        self._fire("before_sensory")
        if not byte_events:
            self._fire("after_sensory", n_processed=0)
            return 0

        is_warmup_now = self._step <= self.config.warmup_steps

        is_warmup_now = self._step <= self.config.warmup_steps
        max_ev = 2000 if is_warmup_now else 500
        if len(byte_events) > max_ev:
            byte_events = byte_events[:max_ev]

        positions_l = [p for p, _ in byte_events]
        byte_vals_l = [b for _, b in byte_events]

        # 匹配: key = (layer=0, position, byte_value)
        matched_nids, matched_idx, unmatched_idx = self.pool.query.match_sensory_events(
            [0] * len(byte_events), positions_l, byte_vals_l
        )

        n_total = len(matched_nids) + len(unmatched_idx)
        if n_total == 0:
            self._fire("after_sensory", n_processed=0)
            return 0

        # 创建未匹配的神经元
        if unmatched_idx:
            new_nids = self.pool.neuron.create_neurons_batch(
                layers=[0] * len(unmatched_idx),
                positions=[positions_l[i] for i in unmatched_idx],
                channels=[byte_vals_l[i] for i in unmatched_idx],
                thresholds=[0.05] * len(unmatched_idx),
            )
            if self._top_layer > 0 and not self._pending_connects:
                l4_nids = self.pool.query.get_neurons_by_layer(10)  # LAYER_L4
                if len(l4_nids) > 0:
                    hl = l4_nids.tolist()
                    pre_list, post_list = [], []
                    # 构建 L4 信道索引 (仅需一次, 纯 Python)
                    l4_ch = {h: int(self.pool.channel[h].item()) for h in hl}
                    for snid, bv in zip(new_nids, [byte_vals_l[i] for i in unmatched_idx]):
                        k = max(1, int(len(hl) * 0.25))
                        group = bv % 8
                        preferred = [h for h in hl if l4_ch[h] == group]
                        rest = [h for h in hl if l4_ch[h] != group]
                        ordered = preferred + rest
                        for hnid in ordered[:k]:
                            pre_list.append(snid)
                            post_list.append(hnid)
                    if pre_list:
                        self.pool.synapse.create_synapses_batch(pre_list, post_list, init_scale=7.5)
        else:
            new_nids = []

        all_nids = matched_nids + new_nids
        # z=1.0 纯离散 on/off: 神经元身份本身编码字节值, 不需要嵌入

        all_nids_t = torch.tensor(all_nids, dtype=torch.int32, device=self.device)
        z_new_t = torch.ones(len(all_nids), dtype=torch.float16, device=self.device)
        self.pool.forward.update_batch(all_nids_t, z_new_t)

        self._fire("after_sensory", n_processed=n_total)
        return n_total

    @torch.inference_mode()
    def process_network_events(self, max_events: int = 10) -> int:
        """
        Stage 4: 网络事件处理
        已简化, 不再使用 EventBridge.
        """
        return 0

    @torch.inference_mode()
    def predict_pass(self):
        """Stage 5: 前馈预测 + 时序/topdown."""
        self._fire("before_predict")
        self.pool.forward.predict_all()
        self.pool.forward.temporal_topdown_pass(self._top_layer)
        self._fire("after_predict")

    @torch.inference_mode()
    def compute_stats(self) -> tuple[float, float, int]:
        """Stage 6: 自由能 + LM logits."""
        free_energy = float(self.pool.forward.compute_free_energy().item())
        pred_byte = -1
        if self._top_layer > 0 and self.pool.query.get_layer_width(self._top_layer) > 0:
            logits = self.pool.forward.compute_lm_logits(self._top_layer)
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
            self._free_energy_history[-2] if len(self._free_energy_history) >= 2 else free_energy
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
        """Stage 8: Hebbian + Oja + 时序更新."""
        self._fire("before_hebbian", modulation=modulation)
        active = self.pool.query.get_active_neurons()
        eta = self.config.hebbian_base_eta
        if active.shape[0] > 0:
            # 全量 Hebbian (前馈+反馈共用 weight 数组, hebbian_pass 通过 in_ptrs 全覆盖)
            eta_ff = eta * 60.0
            self.pool.learning.hebbian_pass(
                active, eta=eta_ff, oja_alpha=self.config.oja_alpha, dopamine=modulation
            )
        # 时序学习: 覆盖所有隐藏层
        hidden_mask = (self.pool.layer > 0) & self.pool.alive
        hidden_active = torch.where(hidden_mask)[0]
        if hidden_active.shape[0] > 0:
            self.pool.learning.hebbian_temporal(hidden_active, eta=eta * 50.0, dopamine=modulation)
        self._fire("after_hebbian", n_active=int(active.shape[0]))

    @torch.inference_mode()
    def homeostasis_pass(self) -> dict:
        """Stage 8b: 稳态可塑性 + 隐藏层动态生长."""
        self._fire("before_homeostasis")
        hs = self.pool.learning.homeostasis_step(
            self._step,
            prune_interval=self.config.prune_interval,
            grow_interval=self.config.grow_interval,
        )

        # 隐藏层动态生长: 自动补充修剪损失, 维持层容量
        if self._step % max(1, self.config.grow_interval) == 0:
            from model.pc import HIDDEN_LAYERS, LAYER_CONFIG

            for layer in HIDDEN_LAYERS:
                base_n = LAYER_CONFIG[layer][0]
                alive_nids = self.pool.query.get_neurons_by_layer(layer)
                current_n = len(alive_nids)
                if current_n < base_n:
                    # 补充至 base_n
                    to_grow = base_n - current_n
                    new_nids = self.pool.neuron.grow_hidden_neurons(layer, to_grow)
                    # 将新神经元连入网络
                    self._wire_hidden_neurons(layer, new_nids)
                    hs.setdefault("grown_hidden", 0)
                    hs["grown_hidden"] += len(new_nids)
                # 长期: 允许层生长超出 base_n (1.5x 上限, 基于高误差)
                max_n = int(base_n * 1.5)
                if current_n < max_n and current_n >= base_n:
                    alive = self.pool.alive[alive_nids]
                    if alive.any():
                        high_err = self.pool.state[alive_nids[alive], 2].abs() > 0.3
                        n_extra = int(high_err.sum().item())
                        if n_extra > max(1, (max_n - current_n) // 2):
                            to_grow = min(n_extra // 2, max_n - current_n)
                            new_nids = self.pool.neuron.grow_hidden_neurons(layer, to_grow)
                            self._wire_hidden_neurons(layer, new_nids)
                            hs["grown_hidden"] = hs.get("grown_hidden", 0) + len(new_nids)

        self._fire("after_homeostasis", hs_stats=hs)
        return hs

    def _wire_hidden_neurons(self, layer: int, nids: list[int]):
        """将新神经元连入现有前馈网络.
        根据层的上下游关系创建对应连接.
        """
        from model.pc import (
            CONN_FEEDFORWARD,
            LAYER_L2,
            LAYER_L3,
            LAYER_L4,
            LAYER_L5,
            LAYER_L6,
            LAYER_SENSORY,
        )

        if not nids:
            return

        # 前馈: 前一层 → 本层
        feedforward_src = {
            LAYER_L4: LAYER_SENSORY,
            LAYER_L2: LAYER_L4,
            LAYER_L3: LAYER_L2,
            LAYER_L5: LAYER_L3,
            LAYER_L6: LAYER_L5,
        }
        src_layer = feedforward_src.get(layer)
        if src_layer is not None:
            src_nids = self.pool.query.get_neurons_by_layer(src_layer)
            if len(src_nids) > 0:
                src_list = src_nids.tolist()
                pre_list, post_list = [], []
                density = 0.25 if layer == LAYER_L4 else 0.30
                for dst in nids:
                    k = max(1, int(len(src_list) * density))
                    for pre in random.sample(src_list, k):
                        pre_list.append(pre)
                        post_list.append(dst)
                if pre_list:
                    is_sensory = src_layer == LAYER_SENSORY
                    self.pool.synapse.create_synapses_batch(
                        pre_list, post_list,
                        init_scale=7.5 if is_sensory else 3.0,
                        conn_type=CONN_FEEDFORWARD,
                    )

        # 反馈: 本层 → 前一层 (独立 td 通路)
        feedback_src = {
            LAYER_L6: LAYER_L5,
            LAYER_L5: LAYER_L3,
            LAYER_L3: LAYER_L2,
            LAYER_L2: LAYER_L4,
        }
        fb_src = feedback_src.get(layer)
        if fb_src is not None:
            self.pool.projections.topdown_connect_layer(layer, fb_src, density=0.2)

        # L4 → LM head (TOP_LAYER)
        if layer == self._top_layer:
            self.pool.projections.lm_ensure_top_connected(self._top_layer)

    @torch.inference_mode()
    def finalize_step(self):
        """Stage 9: 保存 z_prev."""
        self.pool.learning.finalize_step()

    @torch.inference_mode()
    def reset_hidden_state(self):
        """重置隐藏层状态 (layer>0) 为 0. 感官层不改动."""
        from model.pc import F_EPS, F_MU, F_Z, F_Z_PREV

        mask = (self.pool.layer > 0) & self.pool.alive
        if mask.any():
            for col in (F_Z, F_MU, F_EPS, F_Z_PREV):
                self.pool.state[mask, col] = 0.0

    # ═══════════════════════════════════════════════════════════════
    # 核心 step
    # ═══════════════════════════════════════════════════════════════

    @torch.inference_mode()
    def step(self, byte_seq: torch.Tensor, target_byte: int = -1) -> dict:
        """执行一个完整步.

        Args:
            byte_seq: [1, 2, S] fp16 双通道字节编码
            target_byte: 目标字节 (0..255), -1 = 无监督信号

        Returns:
            统计字典.
        """
        self._step += 1
        is_warmup = self._step <= self.config.warmup_steps

        # warmup 结束后触发延迟连接 (全密度双向, 共用 weight 数组)
        if not is_warmup and self._pending_connects:
            for from_l, to_l, density, conn_type in self._pending_connects:
                is_sensory = from_l == 0
                if conn_type == 1:
                    self.pool.projections.topdown_connect_layer(from_l, to_l, density=density)
                else:
                    self.pool.synapse.connect_layer(
                        from_l, to_l, density=density,
                        bias_strength=self.config.bias_strength,
                        conn_type=conn_type,
                        init_scale=7.5 if is_sensory else 3.0,
                    )
            self._pending_connects = []
            # 确保 LM head 连接
            if self._top_layer > 0:
                self.pool.projections.lm_ensure_top_connected(self._top_layer)

        self._fire("before_step", byte_seq=byte_seq)

        byte_events = self.encode(byte_seq)
        self.process_sensory_events(byte_events)

        # warmup 结束后触发延迟连接 (必须在 sensory 存在之后)
        if not is_warmup and self._pending_connects:
            for from_l, to_l, density, conn_type in self._pending_connects:
                is_sensory = from_l == 0
                if conn_type == 1:
                    self.pool.projections.topdown_connect_layer(from_l, to_l, density=density)
                else:
                    self.pool.synapse.connect_layer(
                        from_l, to_l, density=density,
                        bias_strength=self.config.bias_strength,
                        conn_type=conn_type,
                        init_scale=7.5 if is_sensory else 3.0,
                    )
            self._pending_connects = []
            if self._top_layer > 0:
                self.pool.projections.lm_ensure_top_connected(self._top_layer)

        self.process_network_events(max_events=10)
        self.predict_pass()
        free_energy, lm_loss, pred_byte = self.compute_stats()

        # 每步阈值调节 (不管 homeostasis 间隔, 强反馈压制过度发放)
        self.pool.learning.adjust_thresholds(target_rate=0.15, rate_eta=0.05)

        hs_stats: dict = {}
        if self._step % self.config.homeostasis_interval == 0:
            hs_stats = self.homeostasis_pass()
            # LM head 稳态: 已禁用, 误差门控 Hebbian 本身有自调节 (正确时 gate=0.1)
            # L2 归一化破坏条件权重不对称性, 导致退化为全局频率预测
            # if not is_warmup and self._top_layer > 0:
            #     self.pool.learning.homeostasis_lm_head(self._top_layer)

        if not is_warmup and free_energy > 1e-8:
            self.modulate(free_energy)
            self.hebbian_pass(self._last_modulation)
            # LM head: 纯 Hebbian (pre-post 共现, 无误差信号)
            if target_byte >= 0 and self._top_layer > 0:
                self.pool.learning.hebbian_lm_head(
                    self._top_layer,
                    target_byte,
                    eta=self.config.hebbian_base_eta * 5000.0,
                    dopamine=self._last_modulation,
                    use_mu=self.config.use_mu_lm,
                )

        uncertainty = (
            0.5
            if is_warmup or len(self._free_energy_history) < 3
            else compute_uncertainty(self._free_energy_history[-20:])
        )
        self.finalize_step()

        # 每步结尾重置感官层 z=0: 只保留当前步的 L0 信号, 旧 L0 不干扰前馈
        sensory_mask = (self.pool.layer == 0) & self.pool.alive
        if sensory_mask.any():
            self.pool.state[sensory_mask, F_Z] = 0.0

        activity = self.pool.query.get_activity_stats()
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
            byte_ids = torch.tensor(
                [b for b in chunk], dtype=torch.long, device=self.device
            ).unsqueeze(0)
            self.step(byte_ids)
            processed += 1
        return processed

    def connect_topdown(self, upper_layer: int, lower_layer: int, max_per_upper: int = 8):
        """建立 topdown 连接."""
        return self.pool.projections.topdown_connect_active(upper_layer, lower_layer, max_per_upper)

    def get_state(self) -> dict:
        activity = self.pool.query.get_activity_stats()
        return {
            "step": self._step,
            "pool_stats": activity,
            "bridge_stats": {},
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
        dummy = torch.zeros(1, self.config.hidden_size, dtype=torch.long, device=self.device)
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

    def encode_and_predict(self, byte_seq: torch.Tensor):
        """只做感官编码和前馈预测，跳过 Hebbian 学习（用于外部 train_step）."""
        self._step += 1
        is_warmup = self._step <= self.config.warmup_steps
        byte_events = self.encode(byte_seq)
        self.process_sensory_events(byte_events)
        if not is_warmup and self._pending_connects:
            for from_l, to_l, density, conn_type in self._pending_connects:
                if conn_type == 1:
                    self.pool.projections.topdown_connect_layer(from_l, to_l, density=density)
                else:
                    self.pool.synapse.connect_layer(from_l, to_l, density=density,
                        bias_strength=self.config.bias_strength, conn_type=conn_type,
                        init_scale=7.5 if from_l == 0 else 3.0)
            self._pending_connects = []
            if self._top_layer > 0:
                self.pool.projections.lm_ensure_top_connected(self._top_layer)
        self.process_network_events(max_events=10)
        self.predict_pass()
        sensory_mask = (self.pool.layer == 0) & self.pool.alive
        if sensory_mask.any():
            self.pool.state[sensory_mask, 0] = 0.0

    def encode_and_predict_l4_only(self, byte_seq: torch.Tensor):
        """只做感官创建 + L4 预测，跳过全量 predict_pass（30x 加速）."""
        from model.pc import F_EPS, F_MU, F_Z

        self._step += 1
        is_warmup = self._step <= self.config.warmup_steps
        byte_events = self.encode(byte_seq)
        self.process_sensory_events(byte_events)
        if not is_warmup and self._pending_connects:
            for from_l, to_l, density, conn_type in self._pending_connects:
                if conn_type == 1:
                    self.pool.projections.topdown_connect_layer(from_l, to_l, density=density)
                else:
                    self.pool.synapse.connect_layer(from_l, to_l, density=density,
                        bias_strength=self.config.bias_strength, conn_type=conn_type,
                        init_scale=7.5 if from_l == 0 else 3.0)
            self._pending_connects = []
            if self._top_layer > 0:
                self.pool.projections.lm_ensure_top_connected(self._top_layer)
        self.process_network_events(max_events=10)
        # 只对 L4 做预测
        l4_mask = (self.pool.layer == self._top_layer) & self.pool.alive
        if l4_mask.any():
            l4_nids = torch.where(l4_mask)[0]
            mu = self.pool.forward.predict_neurons(l4_nids)
            self.pool.state[l4_nids, F_MU] = mu
            self.pool.state[l4_nids, F_EPS] = self.pool.state[l4_nids, F_Z] - mu
        sensory_mask = (self.pool.layer == 0) & self.pool.alive
        if sensory_mask.any():
            self.pool.state[sensory_mask, 0] = 0.0

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
            if model.pool.get_layer_width(model._top_layer) == 0:
                # 隐藏层被修剪: 重建神经元和连接
                model.add_hidden_layer()
            else:
                model._hidden_layer_created = True
        # warmup 已在 step() 中自动处理 (is_warmup = _step <= warmup_steps)
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
        model.add_hidden_layer()
    return model


def load_cyrene_checkpoint(path: str, config: CyreneConfig | None = None) -> CyreneModel:
    """从检查点加载 Cyrene 模型."""
    if os.path.exists(path):
        return CyreneModel.load(path)
    return create_cyrene(config)
