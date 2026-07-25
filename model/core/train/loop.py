"""统一训练循环 — PC 时空推理 + 多巴胺调制 + fp16 原生训练.

Callback 架构:

  TrainingLoop 持有一组 CallbackBase 对象, 在关键节点调用 _emit() 分发.
  所有持续学习 / 检查点 / 日志 / ICM / Pipeline 等 side-effect 全部由 callback 处理.

用法:
    from model.core.train import TrainingLoop, TrainingConfig

    cfg = TrainingConfig(batch_size=48, lr=3e-4, T_infer=2)
    loop = TrainingLoop(cfg)
    loop.train(task_pipelines=[...])

架构 (每步 5 阶段):
  Phase 1: forward_with_ce()       ← 共享前向 (有梯度)
  Phase 2: spatiotemporal_infer()  ← T 步推理 (π 可调)
  Phase 3: compute_*               ← F_pred (π 加权) + CE_conv
  Phase 4: Dopamine.update(F) → D  ← 3 级调制 (precision / beta / lr)
  Phase 5: backward + step         ← lr 调制
"""

import math
import os
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.continual.hippocampus_buffer import HippocampusBuffer
from model.core.globals import DEVICE
from pkg.utils.trainer_utils import count_budget, setup_seed
from model.model_cyrene import CyreneConfig
from model.pc.local_updates import (
    BCMState,
    apply_hebbian_updates,
    compute_all_hebbian_updates,
    compute_lambda,
    compute_modulators,
    compute_precision_scales,
)
from model.pc.pc_core import DopamineSignal, compute_uncertainty
from model.pc.pc_layers import CyrenePC

from .callback_base import CallbackBase
from .callbacks import (
    CheckpointCallback,
    ContinualCallback,
    IntrinsicCallback,
    LoggingCallback,
    PipelineCallback,
    SleepCallback,
)
from .config import TrainingConfig

_LOG2 = math.log(2)


class TrainingLoop:
    """PC 统一训练循环 (单步 5 阶段 + 持续学习) — Callback 架构."""

    def __init__(self, config: TrainingConfig):
        """初始化训练循环: 配置、环境、状态."""
        self.cfg = config
        self.device = DEVICE
        self._setup_environment()

        # ── 延迟初始化 (在 train() 中创建) ──
        self.model: Optional[CyrenePC] = None
        self.dopamine: Optional[DopamineSignal] = None
        self._orig_forward = None

        # ── 内部状态 ──
        self.global_step = 0
        self._total_steps = 0
        self.prev_precision_scales = None
        self.ema_z = None
        self.forgetting_log: list[dict] = []
        self._last_world_surprise: float = 0.0
        self._last_world_loss: Optional[float] = None
        self._last_world_mode: str = "full"
        self._F_trend_buffer: list[float] = []
        self._surprise_buffer: list[float] = []
        self._current_task_id: str = ""
        self._trained_tasks: list[str] = []

        # 上一步 F_total (用于多巴胺调制)
        self._last_F_bp: float = float("inf")
        self._global_F_hist: list[float] = []
        # 上一步多巴胺 D (传递给下一轮 init_z/forward_with_ce 做门控)
        self._last_dopamine_D: float = 0.5

        # 7a: WM 滚动指标
        self._wm_metrics: dict[str, list[float]] = {
            "transition_error": [],
            "uncertainty": [],
            "fp_rate": [],
        }
        self._wm_fp_count: int = 0
        self._wm_high_surprise_count: int = 0
        self._last_ce_for_fp: float = 0.0

        # 8a: 新任务 novelty 加速
        self._novelty_boost_steps: int = 0
        self._novelty_surprise_injected: float = 0.0

        # ── BCM 滑动阈值 ──
        self.bcm_state = BCMState(n_layers=24, tau=0.01).to(self.device)

        # ── NaN 回退状态 ──
        self._fallback_state: Optional[dict] = None

        # ── 内部模块引用 (callback 通过 loop 访问) ──
        self.memory_bank = None  # 由 ContinualCallback 设置
        self.sniffer = None
        self.abstraction_bank = None
        self.abstraction_sniffer = None
        self.hippocampus = HippocampusBuffer(capacity=200, min_info_gain=0.03)
        self.icm = None
        self.concept_discovery = None
        self.memory_gate = None
        self._icm_output: Optional[dict] = None
        self._intrinsic_stats: dict[str, list] = {
            "pred_loss": [],
            "inverse_loss": [],
            "information_gain": [],
            "uncertainty": [],
            "n_concepts": [],
        }
        self.world_model = None
        self.landscape = None
        self.consolidation_pipeline = None
        self.sleep_engine = None

        # ── Callbacks ──
        self.callbacks: list[CallbackBase] = []

    # ═══════════════════════════════════════════════════════════════
    # Callback 调度
    # ═══════════════════════════════════════════════════════════════

    def _emit(self, event: str, **kwargs):
        """向所有 callback 广播事件. 捕获单个 callback 异常防止连锁崩溃."""
        for cb in self.callbacks:
            try:
                fn = getattr(cb, event, None)
                if fn is not None:
                    fn(loop=self, **kwargs)
            except Exception as e:
                self._log(f"[Callback] {type(cb).__name__}.{event} 异常: {e}")

    def _build_default_callbacks(self):
        """根据 config 标志构造默认 callback 列表."""
        cbs: list[CallbackBase] = []

        cbs.append(ContinualCallback())
        cbs.append(IntrinsicCallback())
        cbs.append(PipelineCallback())
        cbs.append(SleepCallback())
        cbs.append(CheckpointCallback())
        cbs.append(LoggingCallback())

        self.callbacks = cbs

    # ═══════════════════════════════════════════════════════════════
    # 环境初始化
    # ═══════════════════════════════════════════════════════════════

    def _setup_environment(self):
        setup_seed(self.cfg.seed)
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("medium")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True

    def _log(self, message: str):
        """通过回调转发日志."""
        if self.cfg.progress_callback:
            self.cfg.progress_callback({"type": "log", "message": message})

    # ═══════════════════════════════════════════════════════════════
    # 模型构建
    # ═══════════════════════════════════════════════════════════════

    def _build_model(self) -> CyrenePC:
        lm_cfg = CyreneConfig(
            hidden_size=self.cfg.hidden_size,
            num_hidden_layers=self.cfg.num_hidden_layers,
        )
        model = CyrenePC(lm_cfg).half()

        # ── Phase 1: 依赖阈值发放参数注入 ──
        model._act_ema_decay = self.cfg.act_ema_decay
        model._act_target_ratio = self.cfg.act_target_ratio
        model._act_homeo_rate = self.cfg.act_homeo_rate
        model._act_energy_cost = self.cfg.act_energy_cost
        init_th = (
            torch.ones(model.num_sub_layers, model.config.hidden_size) * self.cfg.act_threshold_init
        )
        model.register_buffer("_act_threshold", init_th)

        if hasattr(torch, "compile") and self.device.type == "cuda" and hasattr(torch, "triton"):
            self._orig_forward = model.forward_with_ce
            try:
                model.forward_with_ce = torch.compile(self._orig_forward, mode="reduce-overhead")
                self._log("torch.compile 启用 (mode=reduce-overhead)")
            except Exception as e:
                model.forward_with_ce = self._orig_forward
                self._log(f"torch.compile 失败 (已忽略): {e}")

        if hasattr(torch, "compile") and self.device.type == "cuda" and hasattr(torch, "triton"):
            try:
                model._spatiotemporal_infer_step = torch.compile(
                    model._spatiotemporal_infer_step,
                    mode="reduce-overhead",
                )
                self._log("torch.compile → _spatiotemporal_infer_step 启用")
            except Exception as e:
                self._log(f"torch.compile → _spatiotemporal_infer_step 失败: {e}")

        return model

    # ═══════════════════════════════════════════════════════════════
    # 预热
    # ═══════════════════════════════════════════════════════════════

    def warmup(self):
        """预热: 用 dummy 数据跑一次前向."""
        assert self.model is not None
        self.model.train()
        with torch.no_grad():
            dummy_byte = torch.randint(
                0, 256, (self.cfg.batch_size, self.cfg.max_seq_len), device=self.device
            )
            dummy = torch.stack(
                [
                    dummy_byte.half(),
                    torch.full_like(dummy_byte, 2.0, dtype=torch.float16, device=self.device),
                ],
                dim=1,
            )
            dummy_pos = self.model.get_position_embeddings(self.cfg.max_seq_len, self.device)
            try:
                _, _ = self.model.forward_with_ce(dummy, dummy_byte, dummy_pos)
            except Exception as e:
                if self._orig_forward is not None:
                    self.model.forward_with_ce = self._orig_forward
                    self._log(f"torch.compile 已回退 (首次调用失败: {e})")
                    _, _ = self.model.forward_with_ce(dummy, dummy_byte, dummy_pos)
                else:
                    raise

            if self.cfg.enable_world_model and self.world_model is not None:
                z_init, _ = self.model.forward_with_ce(dummy, dummy_byte, dummy_pos)
                state_tensor = (
                    z_init[-1].detach() if isinstance(z_init, (list, tuple)) else z_init.detach()
                )
                ctx = torch.zeros(
                    dummy.size(0),
                    self.cfg.world_model_context_dim,
                    device=self.device,
                    dtype=torch.float16,
                )
                _, _ = self.world_model(state_tensor, ctx)
                _ = self.world_model.loss(state_tensor, state_tensor, ctx)

        self._log("Warmup done (cudnn benchmark + world model ready)")

    # ═══════════════════════════════════════════════════════════════
    # 上下文构建
    # ═══════════════════════════════════════════════════════════════

    def _build_world_model_context(self, batch_size: int) -> torch.Tensor:
        step_progress = self.global_step / max(self._total_steps, 1)
        last_D = getattr(self, "_last_D", 0.0)
        buf = self._F_trend_buffer
        F_trend = sum(buf[-50:]) / max(len(buf[-50:]), 1) if buf else 0.0
        ratios = list(self.sniffer.last_ratios.values()) if self.sniffer else []
        forgetting_max = max(ratios) if ratios else 0.0
        task_novelty = 0.0 if self._current_task_id in self._trained_tasks else 1.0
        ctx_vals = torch.tensor(
            [step_progress, last_D, F_trend, forgetting_max, task_novelty],
            device=self.device,
            dtype=torch.float16,
        )
        return ctx_vals.unsqueeze(0).expand(batch_size, -1)

    def _build_icm_context(self, batch_size: int) -> torch.Tensor:
        step_progress = self.global_step / max(self._total_steps, 1)
        last_D = getattr(self, "_last_D", 0.0)
        buf = self._F_trend_buffer
        F_trend = sum(buf[-50:]) / max(len(buf[-50:]), 1) if buf else 0.0
        ratios = list(self.sniffer.last_ratios.values()) if self.sniffer else []
        forgetting_max = max(ratios) if ratios else 0.0
        task_novelty = 0.0 if self._current_task_id in self._trained_tasks else 1.0
        icm_signal = self._icm_output or {}
        info_gain_ema = float(icm_signal.get("information_gain", 0.0))
        n_concepts = self.concept_discovery.n_concepts if self.concept_discovery else 0
        n_concepts_norm = min(n_concepts / 20.0, 1.0)
        icm_uncertainty = float(icm_signal.get("uncertainty", 0.0))
        ctx_vals = torch.tensor(
            [
                step_progress,
                last_D,
                F_trend,
                forgetting_max,
                task_novelty,
                info_gain_ema,
                n_concepts_norm,
                icm_uncertainty,
            ],
            device=self.device,
            dtype=torch.float16,
        )
        return ctx_vals.unsqueeze(0).expand(batch_size, -1)

    # ═══════════════════════════════════════════════════════════════
    # 单步训练 (6 阶段 bp_free)
    # ═══════════════════════════════════════════════════════════════

    def train_step(self, byte_seq: torch.Tensor, labels: torch.Tensor) -> dict:
        """执行 6 阶段 bp_free 训练: 零 autograd, 纯局部 Hebbian.

        Phase 0: 数据准备
        Phase 1: init_z (no_grad)
        Phase 2: PC 推理 (T 步, no_grad) → ε_by_layer
        Phase 3: 计算调制信号 (D, ACh, π, λ)
        Phase 4: Decoder 目标计算
        Phase 5: Hebbian 更新 (W.data.add_)
        Phase 6: 日志 + 诊断

        Returns: dict 同 train_step 格式
        """
        assert self.model is not None
        # ── Phase 0: 数据准备 ──
        seq_len = byte_seq.size(-1)
        bsz = byte_seq.size(0)
        pos_emb = self.model.get_position_embeddings(seq_len, self.device)

        # ── Phase 1: init_z + 缓存 SwiGLU 前激活 ──
        #     传递上一步多巴胺 D 给 post-SwiGLU 门控归一化层
        self.model._hebbian_cache_enable()
        with torch.no_grad():
            z_init = self.model.init_z(byte_seq, dopamine_D=self._last_dopamine_D)

        # ── temp_loss 诊断 ──
        bp_temp_loss = 0.0
        bp_temp_by_layer = []
        if hasattr(self.model, "temporal_proj") and seq_len > 1:
            n_layers = len(self.model.temporal_proj)
            tl_acc = torch.tensor(0.0, device=self.device)
            tl_list: list[torch.Tensor] = []
            for layer_i in range(n_layers):
                z_ℓ = z_init[layer_i + 1]
                if z_ℓ.size(1) > 1:
                    z_proj = self.model.temporal_proj[layer_i](z_ℓ[:, :-1, :])
                    tl = 0.5 * (z_proj - z_ℓ[:, 1:, :]).pow(2).mean()
                    tl_clamped = tl.clamp(max=100.0)
                    tl_list.append(tl_clamped)
                    tl_acc = tl_acc + tl_clamped
            if n_layers > 0 and tl_list:
                bp_temp_loss = (tl_acc / n_layers).item()
                bp_temp_by_layer = [t.item() for t in tl_list]

        # ── Phase 2: PC 推理 ──
        uncertainty = compute_uncertainty(self._global_F_hist, window=10)
        ACh = float(torch.sigmoid(torch.tensor(-uncertainty + self.cfg.hebbian_ach_beta_0)).item())

        with torch.no_grad():
            z_conv, errors_hist, F_hist, _, ε_list = self.model.spatiotemporal_infer(  # type: ignore[assignment]
                z_init,
                pos_emb,
                gamma=self.cfg.gamma,
                T=self.cfg.hebbian_infer_T,
                return_errors=True,
                return_pred_loss=False,
                ach_value=ACh,
                return_ε=True,
                adaptive_T=self.cfg.infer_adaptive_T,
                convergence_threshold=self.cfg.infer_convergence_threshold,
                patience=self.cfg.infer_patience,
                min_T=self.cfg.infer_min_T,
                skip_bottom_up=True,
            )

        F_curr = F_hist[-1] if F_hist else 0.0

        # T=1 融合: 提取推理阶段缓存的 SwiGLU 预激活, 跳过 Phase 5 重建
        hebbian_cache = self.model._hebbian_cache_disable_and_collect()

        # ── Phase 3: 调制信号 ──
        D, ACh_val, modulation = compute_modulators(F_curr, self._last_F_bp, uncertainty, self.cfg)
        self._last_dopamine_D = D  # 保存供下一轮 init_z 门控
        λ = compute_lambda(
            self.global_step, self.cfg.hebbian_lambda_decay, self.cfg.hebbian_lambda_min
        )
        π_list = compute_precision_scales(ε_list, ACh_val, D, self.cfg)

        # ── Phase 4: Decoder 目标 ──
        if labels is not None and seq_len > 1:
            target_onehot = nn.functional.one_hot(
                labels[:, 1:].long().clamp(0, 255), num_classes=256
            ).half()
        else:
            target_onehot = None

        # ── Phase 5: Hebbian 更新 ──
        hebb_diag = {}
        with torch.no_grad():
            oja_alpha = getattr(self.cfg, "oja_alpha", 0.05)
            syn_norm = getattr(self.cfg, "synaptic_normalize", True)
            syn_target = getattr(self.cfg, "synaptic_target_norm", 0.0)
            updates = compute_all_hebbian_updates(
                ε_list,
                z_init,
                byte_seq,
                self.model,
                self.cfg,
                D=D,
                ACh=ACh_val,
                modulation=modulation,
                λ=λ,
                decoder=self.model.decoder,
                lm_head=self.model.model.lm_head,
                target_byte_embed=target_onehot,
                oja_alpha=oja_alpha,
                bcm_state=self.bcm_state,
                verbose=True,
                hebbian_cache=hebbian_cache,
            )
            apply_hebbian_updates(
                updates, self.model, synaptic_normalize=syn_norm, target_norm=syn_target
            )
            hebb_diag = {
                "avg_growth": updates.pop("_diag_avg_growth", None),
                "n_inf": updates.pop("_diag_n_inf", None),
                "n_params": updates.pop("_diag_n_params", None),
                "oja_alpha": updates.pop("_diag_oja_alpha", None),
            }

        # ── Phase 5.6: 神经发生 ──
        neuro_stats = {
            "n_pruned": 0,
            "n_resurrected": 0,
            "n_split": 0,
            "active_ratio": 1.0,
        }
        if (
            self.cfg.enable_neurogenesis
            and hasattr(self, "neurogenesis")
            and self.neurogenesis is not None
        ):
            neuro_stats = self.neurogenesis.step(
                model=self.model,
                ε_list=ε_list,
                global_step=self.global_step,
            )

        # ── Phase 5.5: Salience Gate 更新 ──
        if self.cfg.enable_salience_gating and hasattr(self.model, "salience_gates"):
            with torch.no_grad():
                β = self.cfg.salience_reg_weight
                η_gate = self.cfg.salience_gate_lr
                for gate in self.model.salience_gates:
                    gate_sig = torch.sigmoid(gate.logits)  # type: ignore[attr-defined]
                    grad = -2.0 * β * (1.0 - gate_sig) * gate_sig * (1.0 - gate_sig)
                    gate.logits -= η_gate * grad  # type: ignore[attr-defined]

        # ── Phase 6: 日志 + 诊断 ──
        self._last_F_bp = F_curr
        self._global_F_hist.append(F_curr)

        with torch.no_grad():
            ce_diag = self.model.compute_ce_loss(z_conv, labels).item()
            if target_onehot is not None:
                z_L = z_conv[-1]
                z_dec = z_L[:, :-1, :] if z_L.size(1) > 1 else z_L
                dec_pred = nn.functional.linear(z_dec, self.model.decoder.weight)
                dec_loss = nn.functional.mse_loss(dec_pred, target_onehot).item()
            else:
                dec_loss = 0.0

        # ── 世界模型 ──
        world_loss_val = None
        surprise = 0.0
        if self.cfg.enable_world_model and self.world_model is not None:
            if self.global_step % self.cfg.consolidation_pipeline_interval == 0:
                with torch.no_grad():
                    wm_state = z_init[-1].detach()
                    wm_next = z_conv[-1].detach()
                    wm_ctx = self._build_world_model_context(bsz)
                    _, wm_uncertainty = self.world_model(wm_state, wm_ctx)
                    surprise = wm_uncertainty.detach().mean().item()
                    wl = self.world_model.loss(wm_state, wm_next, wm_ctx)
                    world_loss_val = wl.item()
                self._last_world_surprise = surprise
                self._last_world_loss = world_loss_val
            else:
                surprise = self._last_world_surprise
                world_loss_val = getattr(self, "_last_world_loss", None)

        # ── ICM ──
        icm_loss_val = 0.0
        if self.cfg.enable_intrinsic_motivation and self.icm is not None:
            if self.global_step % self.cfg.consolidation_pipeline_interval == 0:
                with torch.no_grad():
                    z_curr = z_conv[-1].detach()
                    z_prev = z_init[-1].detach()
                    action_embed = (z_curr - z_prev).mean(dim=1)
                    if action_embed.size(-1) > self.cfg.icm_action_dim:
                        action_embed = action_embed[:, : self.cfg.icm_action_dim]
                    elif action_embed.size(-1) < self.cfg.icm_action_dim:
                        pad = torch.zeros(
                            bsz,
                            self.cfg.icm_action_dim - action_embed.size(-1),
                            device=self.device,
                        )
                        action_embed = torch.cat([action_embed, pad], dim=-1)
                    icm_output = self.icm.forward(z_prev, z_curr)
                    self._icm_output = {
                        k: (v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v)
                        for k, v in icm_output.items()
                    }
                    icm_loss_val = (
                        self.cfg.icm_forward_weight * self._icm_output.get("pred_loss", 0.0)
                        + self.cfg.icm_inverse_weight * self._icm_output.get("inverse_loss", 0.0)
                        + self.cfg.icm_contrastive_weight
                        * self._icm_output.get("contrastive_loss", 0.0)
                    )
                if self.concept_discovery is not None:
                    info_gain = self._icm_output.get("information_gain", 0.0)
                    self.concept_discovery.observe(z_curr[0:1].detach(), intrinsic_value=info_gain)
            else:
                self._icm_output = None
                icm_loss_val = 0.0

        # ── 持续巩固管道 ──
        if (
            self.consolidation_pipeline is not None
            and self.global_step % self.cfg.consolidation_pipeline_interval == 0
        ):
            sample_z = [z[0:1].detach() for z in z_conv]
            sample_byte = byte_seq[0].detach()
            sample_label = labels[0].detach()
            sample_task = self._current_task_id
            sample_concept = ""
            if self.concept_discovery is not None and len(self.concept_discovery.concept_ids) > 0:
                sample_concept = self.concept_discovery.concept_ids[-1]
            self.consolidation_pipeline.observe(
                z_states=sample_z,
                byte_tensor=sample_byte,
                label_tensor=sample_label,
                task_id=sample_task,
                concept_id=sample_concept,
                information_gain=self._icm_output.get("information_gain", 0.0)
                if self._icm_output
                else 0.0,
                dopamine_score=D,
                step=self.global_step,
            )

        # ── 结果 ──
        self._last_world_mode = "full"
        self._last_D = D
        lr_used = self.cfg.hebbian_base_eta * modulation

        L = self.model.num_sub_layers
        bpb_pred = (F_curr * L) / _LOG2
        bpb_total = (F_curr * L + max(ce_diag, 1e-8)) / _LOG2

        result = {
            "ce_val": ce_diag,
            "F_val": F_curr,
            "F_final": F_curr,
            "F_hist": F_hist,
            "errors_hist": errors_hist,
            "bpb": bpb_total,
            "bpb_pred": bpb_pred,
            "temp_loss_val": bp_temp_loss,
            "temp_by_layer": bp_temp_by_layer,
            "D": D,
            "lr": lr_used,
            "β_local": 0.0,
            "β_conv": 0.0,
            "scale_local": 1.0,
            "scale_conv": 1.0,
            "π": π_list,
            "phase": "bp_free",
            "world_surprise": surprise,
            "world_loss": world_loss_val,
            "update_mode": "full",
            "icm_loss": icm_loss_val,
            "ACh": ACh_val,
            "λ": λ,
            "uncertainty": uncertainty,
            "decoder_loss": dec_loss,
        }
        if self._icm_output:
            for k in ["pred_loss", "inverse_loss", "information_gain", "uncertainty"]:
                result[f"icm_{k}"] = self._icm_output.get(k, 0.0)
        result["error_ratio"] = getattr(self.model, "_last_error_ratio", 1.0)

        if self.cfg.enable_salience_gating and hasattr(self.model, "salience_gates"):
            gs = self.model.get_gate_stats()
            result["gate_active_ratio"] = gs["active_ratio"]
            result["gate_n_active"] = gs["n_active"]
            result["gate_n_total"] = gs["n_total"]
        if self.cfg.enable_neurogenesis:
            result["neuro_n_pruned"] = neuro_stats.get("n_pruned", 0)
            result["neuro_n_resurrected"] = neuro_stats.get("n_resurrected", 0)
            result["neuro_n_split"] = neuro_stats.get("n_split", 0)
            result["neuro_active_ratio"] = neuro_stats.get("active_ratio", 1.0)
        if hebb_diag.get("avg_growth") is not None:
            result["hebb_diag"] = hebb_diag

        # ── Phase 3c: 海马体缓冲写入 ──
        info_gain = self._icm_output.get("information_gain", 0.0) if self._icm_output else 0.0
        if info_gain > self.hippocampus.min_info_gain:
            self.hippocampus.add(
                z_states=z_conv,
                byte_tensor=byte_seq[0].detach(),
                label_tensor=labels[0].detach(),
                info_gain=info_gain,
                step=self.global_step,
            )

        return result

    # ═══════════════════════════════════════════════════════════════
    # Hebbian 辅助
    # ═══════════════════════════════════════════════════════════════

    def _hebbian_update_on_data(
        self,
        byte_seq: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        z_init=None,
        stride: int = 1,
    ):
        """对 (byte_seq, labels) 或预计算 z_init 执行纯 Hebbian 权重更新."""
        assert self.model is not None
        if z_init is not None:
            seq_len = z_init[0].size(1)
        elif byte_seq is not None:
            seq_len = byte_seq.size(-1)
        else:
            return

        if stride > 1 and byte_seq is not None:
            indices = torch.arange(0, seq_len, stride, device=byte_seq.device)
            byte_seq = byte_seq[:, :, indices]
            if labels is not None:
                labels = labels[:, indices]
            seq_len = byte_seq.size(-1)

        pos_emb = self.model.get_position_embeddings(seq_len, self.device)

        if z_init is None and byte_seq is not None:
            z_init, _ = self.model.forward_with_ce(
                byte_seq, labels, pos_emb, dopamine_D=self._last_dopamine_D
            )

        if byte_seq is None:
            byte_seq = torch.zeros(1, 2, seq_len, device=self.device, dtype=torch.long)

        uncertainty = compute_uncertainty(self._global_F_hist, window=10)
        ACh = float(torch.sigmoid(torch.tensor(-uncertainty + self.cfg.hebbian_ach_beta_0)).item())
        with torch.no_grad():
            z_conv, errors_hist, _, _, ε_list = self.model.spatiotemporal_infer(  # type: ignore[assignment]
                z_init,
                pos_emb,
                gamma=self.cfg.gamma,
                T=self.cfg.hebbian_infer_T,
                return_errors=True,
                return_pred_loss=False,
                ach_value=ACh,
                return_ε=True,
            )
        F_curr = errors_hist[-1][0][1] if errors_hist and errors_hist[-1] else 0.0
        hebbian_cache = getattr(self.model, "_hebbian_cache", None)
        D, ACh_val, modulation = compute_modulators(F_curr, self._last_F_bp, uncertainty, self.cfg)
        self._last_dopamine_D = D
        λ = compute_lambda(
            self.global_step, self.cfg.hebbian_lambda_decay, self.cfg.hebbian_lambda_min
        )
        with torch.no_grad():
            oja_alpha = getattr(self.cfg, "oja_alpha", 0.05)
            syn_norm = getattr(self.cfg, "synaptic_normalize", True)
            syn_target = getattr(self.cfg, "synaptic_target_norm", 0.0)
            updates = compute_all_hebbian_updates(
                ε_list,
                z_init,
                byte_seq,
                self.model,
                self.cfg,
                D=D,
                ACh=ACh_val,
                modulation=modulation,
                λ=λ,
                decoder=self.model.decoder,
                lm_head=self.model.model.lm_head,
                target_byte_embed=None,
                oja_alpha=oja_alpha,
                bcm_state=self.bcm_state,
                verbose=False,
                hebbian_cache=hebbian_cache,
            )
            apply_hebbian_updates(
                updates, self.model, synaptic_normalize=syn_norm, target_norm=syn_target
            )

    # ═══════════════════════════════════════════════════════════════
    # 主训练循环
    # ═══════════════════════════════════════════════════════════════

    def train(
        self,
        task_pipelines: list[tuple[str, torch.utils.data.Dataset, Optional[DataLoader]]],
    ):
        """主训练入口 — Callback 架构.

        Args:
            task_pipelines: [(task_id, dataset, loader_override?), ...]
        """
        out_dir = os.path.join(os.getcwd(), self.cfg.out_dir)
        os.makedirs(out_dir, exist_ok=True)

        # ── 构建模型 ──
        self.model = self._build_model().to(self.device)
        budget = count_budget(self.model)
        self._fallback_state = {
            k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()
        }
        self._log(f"NaN fallback snapshot saved ({len(self._fallback_state)} tensors on CPU)")
        self._log(
            "Model: capacity budget"
            f" {budget['trainable_M']:.2f}M"
            " — effective capacity evolves during training"
        )

        # ── 神经发生 ──
        if self.cfg.enable_neurogenesis:
            from model.continual.neurogenesis import NeurogenesisController

            self.neurogenesis = NeurogenesisController(
                hidden_size=self.cfg.hidden_size,
                prune_interval=self.cfg.neurogenesis_prune_interval,
                grow_interval=self.cfg.neurogenesis_grow_interval,
                prune_threshold_act=self.cfg.neurogenesis_prune_threshold_act,
                prune_threshold_gate=self.cfg.neurogenesis_prune_threshold_gate,
                grow_error_threshold=self.cfg.neurogenesis_grow_error_threshold,
                max_grow_per_step=self.cfg.neurogenesis_max_grow_per_step,
            )
            self._log("Neurogenesis controller enabled (prune+grow)")
        else:
            self.neurogenesis = None

        # ── 世界模型 ──
        if self.cfg.enable_world_model:
            from model.continual.world_model import LatentWorldModel

            self.world_model = LatentWorldModel(
                input_dim=self.cfg.hidden_size,
                hidden_dim=self.cfg.world_model_hidden_dim,
                context_dim=self.cfg.world_model_context_dim,
            ).to(self.device)
            self.world_model_optimizer = None
            self._log("Latent world model enabled (inference-only)")

        # ── ICM ──
        if self.cfg.enable_intrinsic_motivation:
            from model.continual.concept_discovery import ConceptDiscovery
            from model.continual.intrinsic_curiosity import IntrinsicCuriosityModule
            from model.continual.memory_gating import MemoryGate

            self.icm = IntrinsicCuriosityModule(
                input_dim=self.cfg.hidden_size,
                action_embed_dim=self.cfg.icm_action_dim,
                hidden_dim=self.cfg.icm_hidden_dim,
            ).to(self.device)
            self.icm_optimizer = None
            self.concept_discovery = ConceptDiscovery(
                initial_threshold=self.cfg.concept_threshold_init,
                min_threshold=self.cfg.concept_threshold_min,
            )
            self.memory_gate = MemoryGate(
                threshold_low=0.05,
                threshold_high=0.5,
                target_storage_ratio=self.cfg.gate_target_storage,
                target_high_value_ratio=self.cfg.gate_target_high,
            )
            self._log("Intrinsic motivation enabled (ICM + Concept + Gate)")

        # ── 吸引子景观 + 持续巩固管道 ──
        if self.cfg.enable_consolidation_pipeline:
            from model.continual.attractor_landscape import AttractorLandscape
            from model.continual.consolidation_pipeline import ConsolidationPipeline

            self.landscape = AttractorLandscape(
                num_sub_layers=12,
                n_noise_levels=5,
                n_variants_per_proto=5,
                T_infer=self.cfg.T_infer,
            )
            self.consolidation_pipeline = ConsolidationPipeline(
                buffer_capacity=self.cfg.pipeline_buffer_capacity,
                memory_write_interval=self.cfg.pipeline_memory_write_interval,
                abstraction_write_interval=self.cfg.pipeline_abstraction_write_interval,
                sleep_check_interval=self.cfg.pipeline_sleep_check_interval,
                min_info_gain_for_write=self.cfg.pipeline_min_info_gain,
                memory_batch_size=32,
                abstraction_batch_size=16,
                num_sub_layers=12,
            )
            self._log("Consolidation pipeline enabled")

        # ── 深度 SLEEP ──
        if self.cfg.enable_deep_sleep:
            from model.continual.deep_sleep import SleepEngine

            self.sleep_engine = SleepEngine(
                num_sub_layers=12,
                pattern_completion_steps=self.cfg.sleep_completion_steps,
                noise_broadening_steps=self.cfg.sleep_noise_steps,
                competitive_steps=self.cfg.sleep_competitive_steps,
                T_infer_sleep=max(1, self.cfg.T_infer // 2),
                gamma_sleep=self.cfg.gamma * 0.5,
                hebbian_base_eta=self.cfg.hebbian_base_eta,
                hebbian_lambda_min=self.cfg.hebbian_lambda_min,
                dopamine_gamma=self.cfg.dopamine_gamma,
            )
            self._log("Deep sleep engine enabled")

        # ── 从 checkpoint 恢复 ──
        if self.cfg.checkpoint_path:
            ckpt = torch.load(
                self.cfg.checkpoint_path, map_location=self.device, weights_only=False
            )
            if "model_state" in ckpt:
                self.model.load_state_dict(ckpt["model_state"])
                if (
                    self.cfg.enable_world_model
                    and self.world_model is not None
                    and "world_model_state" in ckpt
                ):
                    self.world_model.load_state_dict(ckpt["world_model_state"])
            else:
                self.model.load_state_dict(ckpt)
            self._log(f"Resumed from checkpoint: {self.cfg.checkpoint_path}")

        # ── 多巴胺 + Sniffer ──
        self.dopamine = DopamineSignal(η=self.cfg.dopamine_eta, threshold=0.0)

        # ── 构建 Callbacks ──
        self._build_default_callbacks()

        # ── Sniffer 绑定 (callback 初始化完成后) ──
        if self.sniffer is not None:
            self.sniffer.model = self.model
        if self.abstraction_sniffer is not None:
            self.abstraction_sniffer.model = self.model
            self.abstraction_sniffer.world_model = self.world_model

        # ── 预热 ──
        self.warmup()

        # ── 主循环开始 ──
        trained_tasks: list[str] = []
        self._trained_tasks = trained_tasks
        self.model.train()

        self._emit("on_fit_start")

        for task_id, task_ds, _ in task_pipelines:
            self._current_task_id = task_id

            if isinstance(task_ds, DataLoader):
                loader = task_ds
                task_ds = loader.dataset
            else:
                loader = DataLoader(
                    task_ds,
                    batch_size=self.cfg.batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=True,
                )

            # ── 8a+8b: novelty 加速 + WM reset ──
            if self.cfg.enable_world_model:
                self._novelty_boost_steps = int(len(task_ds) * 0.05)  # type: ignore[arg-type]
                self._novelty_surprise_injected = self.cfg.world_model_surprise_threshold * 2.0
                if hasattr(self, "world_model") and self.world_model is not None:
                    self.world_model.reset_state()
                    self._log(f"[Novelty/8b] WM state reset for task {task_id}")
            if self.cfg.enable_intrinsic_motivation and self.icm is not None:
                self.icm.reset_state()
                self._icm_output = None

            self._total_steps = len(loader) * self.cfg.epochs
            self.global_step = 0

            self._log(
                "\n" + "=" * 60 + f"\nStarting Task {task_id}:"
                f" {len(loader.dataset)} samples\n" + "=" * 60  # type: ignore[arg-type]
            )
            self._emit("on_task_start", task_id=task_id, task_dataset=task_ds)

            for epoch in range(self.cfg.epochs):
                pbar = tqdm(
                    loader,
                    desc=f"Task {task_id} Epoch {epoch + 1}/{self.cfg.epochs}",
                    unit="step",
                    dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"
                    " [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
                )

                for byte_seq, labels in pbar:
                    byte_seq = byte_seq.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    m = self.train_step(byte_seq, labels)

                    if m.get("skipped", False):
                        if "world_surprise" not in m:
                            m["world_surprise"] = 0.0
                        continue

                    self.global_step += 1

                    # Hebbian 诊断
                    hebb = m.get("hebb_diag")
                    if hebb is not None and self.global_step % 50 == 0:
                        pbar.write(
                            f"  [Hebb] oja_α={hebb['oja_alpha']:.4f} | "
                            f"mean|ΔW|={hebb['avg_growth']:.6f} | "
                            f"updates={hebb['n_params']}"
                            + (f" | ⚠ {hebb['n_inf']} inf跳过" if hebb["n_inf"] > 0 else "")
                        )

                    # 8a: novelty 注入
                    if self._novelty_boost_steps > 0 and self.cfg.enable_world_model:
                        m["world_surprise"] = max(
                            m.get("world_surprise", 0.0),
                            self._novelty_surprise_injected,
                        )
                        self._novelty_boost_steps -= 1
                        if self._novelty_boost_steps == 0:
                            self._novelty_surprise_injected = 0.0

                    # ── 状态更新 ──
                    self._last_D = m["D"]
                    self._F_trend_buffer.append(m.get("F_val", m.get("F_final", 0.0)))
                    if len(self._F_trend_buffer) > 100:
                        self._F_trend_buffer.pop(0)
                    if self.sniffer is not None:
                        self.sniffer.update_surprise(m.get("world_surprise", 0.0))
                    if self.global_step % 100 == 0:
                        self._surprise_buffer.append(m.get("world_surprise", 0.0))

                    # ── WM 滚动指标 ──
                    if self.cfg.enable_world_model:
                        wl = m.get("world_loss")
                        ws = m.get("world_surprise", 0.0)
                        if wl is not None:
                            self._wm_metrics["transition_error"].append(wl)
                        self._wm_metrics["uncertainty"].append(ws)
                        ce_now = m["ce_val"]
                        if (
                            m.get("update_mode") == "full"
                            and ws > self.cfg.world_model_surprise_threshold
                        ):
                            self._wm_high_surprise_count += 1
                            if ce_now <= self._last_ce_for_fp:
                                self._wm_fp_count += 1
                        if self.global_step % 100 == 0 and self._wm_high_surprise_count > 0:
                            fp_rate = self._wm_fp_count / max(self._wm_high_surprise_count, 1)
                            self._wm_metrics["fp_rate"].append(fp_rate)
                        self._last_ce_for_fp = ce_now

                    # ── Callback: 步后处理 ──
                    self._emit(
                        "on_step_end",
                        metrics=m,
                        task_id=task_id,
                        epoch=epoch,
                        pbar=pbar,
                        loader=loader,
                        out_dir=out_dir,
                    )

            # ── 任务完成 ──
            self._emit(
                "on_task_end",
                task_id=task_id,
                task_dataset=task_ds,
                task_pipelines=task_pipelines,
                trained_tasks=trained_tasks,
                out_dir=out_dir,
            )

        # ── 全部任务完成 ──
        self._emit("on_fit_end", task_pipelines=task_pipelines, out_dir=out_dir)

        # ── 保存 unified_final.pt ──
        self.model.cpu()
        fp = os.path.join(out_dir, "unified_final.pt")
        torch.save(self.model.state_dict(), fp)
        self._log(f"unified_final saved → {fp} ({os.path.getsize(fp) // 1024 // 1024}MB)")

        self._log("Training complete.")
        if self.cfg.progress_callback:
            self.cfg.progress_callback({"type": "done", "message": "Training complete"})
