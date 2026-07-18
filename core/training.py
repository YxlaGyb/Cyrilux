"""
统一训练循环 — PC 时空推理 + 多巴胺调制 + 4bit QAT + 持续学习。

用法:
    from core.training import TrainingLoop, TrainingConfig

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
import os, sys, json, math, warnings
from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.pc_layers import PCLocalDynamicMiniMind
from model.model_minimind import MiniMindConfig
from model.pc_core import DopamineSignal
from core.trainer_utils import get_lr, setup_seed, count_parameters
from core.dataset import DualChannelDataset
from core.globals import DEVICE

# 持续学习模块
from continual.memory_bank import MemoryBank
from continual.forgetting_sniffer import ForgettingSniffer
from continual.offline_replay import OfflineReplayer
from continual.abstraction_bank import (
    AbstractionBank,
    AbstractionSniffer,
    VariationalReplayer,
    compute_layer_importance,
)
from continual.world_model import LatentWorldModel

# 内在动机模块
from continual.intrinsic_curiosity import IntrinsicCuriosityModule
from continual.concept_discovery import ConceptDiscovery
from continual.memory_gating import MemoryGate

# ─── 回调类型 ─────────────────────────────────────────────────────
ProgressCallback = Callable[[dict], None]
"""训练进度回调.
    {
        'type': 'log' | 'progress' | 'phase' | 'checkpoint' | 'task_done' | 'done' | 'error',
        'message': str, 'step': int, 'total_steps': int,
        'ce_loss': float, 'F': float, 'D': float, 'lr': float,
        'task_id': str, 'checkpoint_path': str,
    }
"""


@dataclass
class TrainingConfig:
    """统一训练配置。"""
    # 模型
    hidden_size: int = 256
    num_hidden_layers: int = 4
    use_moe: bool = False
    checkpoint_path: Optional[str] = None  # 从 checkpoint 恢复

    # 训练
    batch_size: int = 48
    max_seq_len: int = 128
    lr: float = 3e-4
    epochs: int = 1
    subset: int = 0           # 0 = 全量
    seed: int = 42
    grad_clip: float = 1.0
    split_size: int = 0       # 0 = 不分块, >0 = 每块样本数 (GUI 传入)

    # PC 时空推理
    T_infer: int = 2
    gamma: float = 0.1
    max_beta: float = 2.0
    max_beta_conv: float = 1.0
    ema_lambda: float = 0.001

    # 多巴胺
    dopamine_eta: float = 1.0
    dopamine_beta: float = 0.5
    dopamine_gamma: float = 0.3

    # AMP
    enable_amp: bool = False                # bf16 自动混合精度 (默认关, 用 --amp 开启)

    # CUDA Graphs
    use_cuda_graphs: bool = False           # CUDA Graph 录制 (实验性, 需固定 seq_len)

    # QAT
    enable_qat: bool = True
    qat_groupsize: int = 64
    no_quantize_embed: bool = False

    # 持续学习
    replay_ratio: int = 5           # 每 N 步插入 1 步回放
    bank_size: int = 2000           # 每任务最大 exemplar 数
    sniff_interval: int = 200       # 遗忘嗅探检查间隔
    repair_threshold: float = 1.2   # 遗忘触发阈值
    repair_steps: int = 10          # 修复步数
    eval_samples: int = 100         # 跨任务遗忘评估样本数

    # AbstractionBank
    n_prototypes: int = 8
    abstraction_replay_interval: int = 200
    abstraction_sniff_interval: int = 300
    abstraction_drift_threshold: float = 0.7

    # 世界模型 / latent dynamics
    enable_world_model: bool = True
    world_model_hidden_dim: int = 128
    world_model_context_dim: int = 5
    world_model_surprise_threshold: float = 0.25
    world_model_loss_weight: float = 0.1

    # 内在动机 (ICM / Concept / Gate)
    enable_intrinsic_motivation: bool = True
    icm_forward_weight: float = 1.0
    icm_inverse_weight: float = 0.1
    icm_contrastive_weight: float = 0.05
    icm_hidden_dim: int = 64
    icm_action_dim: int = 8
    concept_threshold_init: float = 0.85
    concept_threshold_min: float = 0.65
    gate_target_storage: float = 0.30
    gate_target_high: float = 0.10

    # I/O
    out_dir: str = 'out_pc_unified'
    save_interval: int = 500

    # 回调
    progress_callback: Optional[ProgressCallback] = None

    # Sleep / Consolidation
    sleep_consolidation: bool = True          # 6a: 任务后 WM 驱动合并
    full_sleep_after_all: bool = True         # 6b: 全部任务后完整睡眠
    sleep_replay_tasks: int = 2               # 6b: 睡眠阶段回放最不确定的 N 个任务
    sleep_replay_samples: int = 100           # 6b: 每个任务生成样本数

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_') and k != 'progress_callback'}


# ─── 训练循环 ─────────────────────────────────────────────────────

class TrainingLoop:
    """PC 统一训练循环 (单步 5 阶段 + 持续学习)。"""

    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.device = DEVICE
        self._setup_environment()

        # 延迟初始化 (在 train() 中创建)
        self.model: Optional[PCLocalDynamicMiniMind] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.world_model_optimizer: Optional[torch.optim.Optimizer] = None
        self.dopamine: Optional[DopamineSignal] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self.quantizer = None
        self._orig_forward = None
        self.world_model: Optional[LatentWorldModel] = None

        # 持续学习
        self.memory_bank = MemoryBank(max_per_task=self.cfg.bank_size)
        self.sniffer = ForgettingSniffer(
            memory_bank=self.memory_bank, model=None,  # 后面 set_model
            check_interval=self.cfg.sniff_interval,
            threshold=self.cfg.repair_threshold,
            repair_steps=self.cfg.repair_steps,
        )
        self.abstraction_bank = AbstractionBank(
            max_entries_per_task=self.cfg.bank_size,
            n_prototypes=self.cfg.n_prototypes,
            consolidation_frequency=1,
        )
        self.abstraction_sniffer = AbstractionSniffer(
            bank=self.abstraction_bank, model=None,
            check_interval=self.cfg.abstraction_sniff_interval,
            drift_threshold=self.cfg.abstraction_drift_threshold,
            world_model=None,  # 后面在 train() 中绑定
        )

        # 内部状态
        self.global_step = 0
        self.prev_precision_scales = None
        self.ema_z = None
        self._graph_capture_mode: bool = False  # CUDA Graph 录制标志
        self.forgetting_log: list[dict] = []
        self._last_world_surprise: float = 0.0
        self._last_world_mode: str = 'full'
        self._F_trend_buffer: list[float] = []
        self._surprise_buffer: list[float] = []
        self._current_task_id: str = ''
        self._trained_tasks: list[str] = []

        # 7a: WM 滚动指标
        self._wm_metrics: dict[str, list[float]] = {
            'transition_error': [],   # 世界模型 MSE 损失
            'uncertainty': [],        # world_surprise
            'fp_rate': [],            # 假阳性率 (高 surprise 但 CE 未升)
        }
        self._wm_fp_count: int = 0
        self._wm_high_surprise_count: int = 0
        self._last_ce_for_fp: float = 0.0

        # 8a: 新任务 novelty 加速
        self._novelty_boost_steps: int = 0
        self._novelty_surprise_injected: float = 0.0

        # ── 内在动机模块 ──
        self.icm: Optional[IntrinsicCuriosityModule] = None
        self.icm_optimizer: Optional[torch.optim.Optimizer] = None
        self.concept_discovery: Optional[ConceptDiscovery] = None
        self.memory_gate: Optional[MemoryGate] = None
        self._icm_output: Optional[dict] = None
        self._intrinsic_stats: dict[str, list] = {
            'pred_loss': [],
            'inverse_loss': [],
            'information_gain': [],
            'uncertainty': [],
            'n_concepts': [],
        }

    # ── 环境初始化 ──

    def _setup_environment(self):
        setup_seed(self.cfg.seed)
        if self.device.type == 'cuda':
            torch.set_float32_matmul_precision('medium')
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True

    def _log(self, message: str):
        """通过回调转发日志到 GUI (GuiLogger 会捕获终端输出)。"""
        if self.cfg.progress_callback:
            self.cfg.progress_callback({'type': 'log', 'message': message})

    def _build_model(self) -> PCLocalDynamicMiniMind:
        lm_cfg = MiniMindConfig(
            hidden_size=self.cfg.hidden_size,
            num_hidden_layers=self.cfg.num_hidden_layers,
            use_moe=self.cfg.use_moe,
        )
        model = PCLocalDynamicMiniMind(lm_cfg)

        # QAT 准备 (CPU 上执行)
        if self.cfg.enable_qat:
            from torchao.quantization.qat import Int4WeightOnlyQATQuantizer
            self.quantizer = Int4WeightOnlyQATQuantizer(
                groupsize=self.cfg.qat_groupsize,
                inner_k_tiles=4,
                precision=torch.float16,
                scales_precision=torch.bfloat16,
            )
            model = self.quantizer.prepare(model)
            self._log(f'Int4WeightOnly QAT prepared (groupsize={self.cfg.qat_groupsize})')

        # 编译 (Windows 无 Triton, 跳过 torch.compile)
        if hasattr(torch, 'compile') and sys.platform != 'win32':
            self._orig_forward = model.forward_with_ce
            try:
                model.forward_with_ce = torch.compile(self._orig_forward, mode='reduce-overhead')
                self._log('torch.compile 启用 (mode=reduce-overhead)')
            except Exception as e:
                model.forward_with_ce = self._orig_forward
                self._log(f'torch.compile 失败 (已忽略): {e}')

        return model

    def _build_optimizer(self):
        return torch.optim.AdamW(
            list(self.model.temporal_proj.parameters()) +
            list(self.model.topdown_proj.parameters()) +
            [p for n, p in self.model.model.named_parameters() if p.requires_grad],
            lr=self.cfg.lr, betas=(0.9, 0.95), fused=True,
        )

    def warmup(self):
        """吸收 cudnn benchmark 首步算法搜索延迟 + 预热世界模型。"""
        self.model.train()
        with torch.no_grad():
            dummy_byte = torch.randint(0, 256, (self.cfg.batch_size, self.cfg.max_seq_len), device=self.device)
            dummy = torch.stack([
                dummy_byte.float(),
                torch.full_like(dummy_byte, 2.0, dtype=torch.float, device=self.device),
            ], dim=1)
            dummy_pos = self.model.get_position_embeddings(self.cfg.max_seq_len, self.device)
            try:
                _, _ = self.model.forward_with_ce(dummy, dummy_byte, dummy_pos)
            except Exception as e:
                if self._orig_forward is not None:
                    self.model.forward_with_ce = self._orig_forward
                    self._log(f'torch.compile 已回退 (首次调用失败: {e})')
                    _, _ = self.model.forward_with_ce(dummy, dummy_byte, dummy_pos)
                else:
                    raise

            # 预热世界模型，避免第一步 surprise 是冷启动噪声
            if self.cfg.enable_world_model and self.world_model is not None:
                z_init, _ = self.model.forward_with_ce(dummy, dummy_byte, dummy_pos)
                if isinstance(z_init, (list, tuple)):
                    state_tensor = z_init[-1].detach()
                else:
                    state_tensor = z_init.detach()
                ctx = torch.zeros(dummy.size(0), self.cfg.world_model_context_dim,
                                  device=self.device, dtype=torch.float32)
                _, _ = self.world_model(state_tensor, ctx)
                _ = self.world_model.loss(state_tensor, state_tensor, ctx)

        self._log('Warmup done (cudnn benchmark + world model ready)')

    # ── 世界模型上下文构建 ──────────────────────────────────────────

    def _build_world_model_context(self, batch_size: int) -> torch.Tensor:
        """构建 5 维世界模型上下文 [B, 5]。

        dims: [step_progress, D, F_trend, forgetting_ratio_max, task_novelty]
        """
        step_progress = self.global_step / max(self._total_steps, 1)
        last_D = getattr(self, '_last_D', 0.0)
        # F_trend: 过去 100 步 F_pred 均值
        buf = self._F_trend_buffer
        F_trend = sum(buf[-50:]) / max(len(buf[-50:]), 1) if buf else 0.0
        # forgetting_ratio_max
        ratios = list(self.sniffer.last_ratios.values())
        forgetting_max = max(ratios) if ratios else 0.0
        # task_novelty
        task_novelty = 0.0 if self._current_task_id in self._trained_tasks else 1.0
        ctx_vals = torch.tensor(
            [step_progress, last_D, F_trend, forgetting_max, task_novelty],
            device=self.device, dtype=torch.float32,
        )
        return ctx_vals.unsqueeze(0).expand(batch_size, -1)

    def _build_icm_context(self, batch_size: int) -> torch.Tensor:
        """构建 8 维 ICM 上下文向量 [B, 8]。

        dims: [step_progress, D, F_trend, forgetting_ratio_max, task_novelty,
               info_gain_ema, n_concepts_norm, icm_uncertainty]
        """
        step_progress = self.global_step / max(self._total_steps, 1)
        last_D = getattr(self, '_last_D', 0.0)
        buf = self._F_trend_buffer
        F_trend = sum(buf[-50:]) / max(len(buf[-50:]), 1) if buf else 0.0
        ratios = list(self.sniffer.last_ratios.values())
        forgetting_max = max(ratios) if ratios else 0.0
        task_novelty = 0.0 if self._current_task_id in self._trained_tasks else 1.0
        # ICM signals
        icm_signal = self._icm_output or {}
        info_gain_ema = float(icm_signal.get('information_gain', 0.0))
        n_concepts = len(self.concept_discovery.concepts) if self.concept_discovery else 0
        n_concepts_norm = min(n_concepts / 20.0, 1.0)
        icm_uncertainty = float(icm_signal.get('uncertainty', 0.0))
        ctx_vals = torch.tensor(
            [step_progress, last_D, F_trend, forgetting_max, task_novelty,
             info_gain_ema, n_concepts_norm, icm_uncertainty],
            device=self.device, dtype=torch.float32,
        )
        return ctx_vals.unsqueeze(0).expand(batch_size, -1)

    # ── 单步训练 ──

    def train_step(self, byte_seq: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        执行 5 阶段单步训练 (PC 完整路径 + bf16 AMP + 多巴胺三通道调制)。

        Phase 1:  forward_with_ce  (共享前向, 有梯度)
        Phase 2:  spatiotemporal_infer (PC 推理 T 步)
        Phase 3:  F_pred (重算预测误差, 多巴胺精度加权)
        Phase 4:  多巴胺调制 (D → π/β/lr)
        Phase 5:  backward + step (AMP scaler)

        Returns: {'ce_val', 'F_val', 'D', 'lr', 'π', ...}
        """
        bsz, _, seq_len = byte_seq.shape
        self.global_step += 1

        # ── 前向 ──
        pos_emb = self.model.get_position_embeddings(seq_len, self.device)

        # ═══ 完整 PC 路径: Phase 1-5 ═══
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.cfg.enable_amp):
            z_init, ce_loss = self.model.forward_with_ce(byte_seq, labels, pos_emb)
            z_init_det = [z.detach() for z in z_init]

            # World model can gate the next update into full-precision inference
            # or a lightweight update when the latent transition is already stable.
            world_loss = None
            surprise = 0.0
            update_mode = 'full'
            wm_state_tensor = None
            wm_ctx = None
            if self.cfg.enable_world_model and self.world_model is not None:
                wm_state_tensor = z_init_det[-1].detach()
                wm_ctx = self._build_world_model_context(bsz)
                _, uncertainty = self.world_model(wm_state_tensor, wm_ctx)
                surprise = uncertainty.detach().mean().item()
                if surprise < self.cfg.world_model_surprise_threshold:
                    update_mode = 'light'

            # Compute the predictive-coding loop first; the world model only short-circuits
            # when the latent transition is already known to be stable, while still feeding
            # into the same dopamine and replay machinery.
            if update_mode == 'full':
                z_converged, errors_hist, F_hist, F_pred = self.model.spatiotemporal_infer(
                    z_init_det, pos_emb, gamma=self.cfg.gamma, T=self.cfg.T_infer,
                    return_errors=True, return_pred_loss=True,
                    precision_scales=self.prev_precision_scales,
                )
                target_metric = F_pred
            else:
                z_converged = z_init_det
                errors_hist = []
                F_hist = []
                F_pred = ce_loss
                target_metric = ce_loss

            # 世界模型损失：预测 z_init → z_converged 的 dynamics，而非自编码
            if wm_state_tensor is not None and wm_ctx is not None:
                if update_mode == 'full':
                    next_target = z_converged[-1].detach()
                else:
                    next_target = z_init_det[-1].detach()
                world_loss = self.world_model.loss(wm_state_tensor, next_target, wm_ctx)

            # Phase 4: 多巴胺调制
            D = self.dopamine.update(target_metric.item() if hasattr(target_metric, 'item') else float(target_metric))
            β_local = min(self.cfg.max_beta,
                          0.1 + self.global_step / max(self._total_steps, 1) * (self.cfg.max_beta - 0.1))
            β_conv = min(self.cfg.max_beta_conv,
                         0.0 + self.global_step / max(self._total_steps, 1) * self.cfg.max_beta_conv)
            β_local = β_local * (1.0 + self.cfg.dopamine_gamma * D)
            β_conv = β_conv * (1.0 + self.cfg.dopamine_gamma * D)

            # 精度调制 π (基于最后一步误差)
            last_errors = errors_hist[-1] if errors_hist else []
            if last_errors:
                err_norms_t = torch.tensor([e[1] for e in last_errors], device=self.device)
                max_err = err_norms_t.max() + 1e-8
                π_list = 1.0 + self.cfg.dopamine_eta * D * (err_norms_t / max_err)
                self.prev_precision_scales = π_list.detach().cpu().tolist()
            else:
                self.prev_precision_scales = None

            # 三路合并
            ce_local_sum = ce_loss * (bsz * seq_len)
            ce_conv_sum = ce_loss * (bsz * seq_len)
            if update_mode == 'full':
                scale_local = (F_pred.detach() / (ce_local_sum.detach() + 1e-8)).clamp(0.1, 10.0)
                scale_conv = (F_pred.detach() / (ce_conv_sum.detach() + 1e-8)).clamp(0.1, 10.0)
                total_loss = F_pred + β_local * scale_local * ce_local_sum \
                                    + β_conv * scale_conv * ce_conv_sum
                if world_loss is not None:
                    total_loss = total_loss + self.cfg.world_model_loss_weight * world_loss
            else:
                total_loss = ce_loss
                if world_loss is not None:
                    total_loss = total_loss + self.cfg.world_model_loss_weight * world_loss

            # ── ICM 内在动机 ──
            icm_loss = torch.tensor(0.0, device=self.device)
            if self.cfg.enable_intrinsic_motivation and self.icm is not None:
                # 构建 action embedding (简单: z_t+1 - z_t)
                z_curr = z_converged[-1].detach()  # [B, seq, hidden]
                z_prev = z_init_det[-1].detach()
                action_embed = (z_curr - z_prev).mean(dim=1)  # [B, hidden] → [B, icm_action_dim]
                if action_embed.size(-1) > self.cfg.icm_action_dim:
                    action_embed = action_embed[:, :self.cfg.icm_action_dim]
                elif action_embed.size(-1) < self.cfg.icm_action_dim:
                    pad = torch.zeros(bsz, self.cfg.icm_action_dim - action_embed.size(-1), device=self.device)
                    action_embed = torch.cat([action_embed, pad], dim=-1)

                icm_output = self.icm.forward(z_prev.mean(dim=1), z_curr.mean(dim=1), action_embed)
                self._icm_output = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in icm_output.items()}

                # ICM 损失组合
                icm_loss = (self.cfg.icm_forward_weight * icm_output['pred_loss'] +
                            self.cfg.icm_inverse_weight * icm_output['inverse_loss'] +
                            self.cfg.icm_contrastive_weight * icm_output.get('contrastive_loss', 0.0))
                total_loss = total_loss + icm_loss

                # 概念发现 + 记忆门控
                if self.concept_discovery is not None:
                    info_gain = self._icm_output.get('information_gain', 0.0)
                    self.concept_discovery.observe(
                        z_curr.mean(dim=1).detach(),
                        info_gain=info_gain,
                    )

        # Phase 5: backward
        self.optimizer.zero_grad(set_to_none=True)
        if self.world_model_optimizer is not None:
            self.world_model_optimizer.zero_grad(set_to_none=True)
        if self.icm_optimizer is not None:
            self.icm_optimizer.zero_grad(set_to_none=True)
        if self.cfg.enable_amp:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        # ── 公共后处理 (梯度裁剪 + lr 调度 + 参数更新) ──
        trainable = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if trainable:
            if self.cfg.enable_amp:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, self.cfg.grad_clip)

        current_lr = get_lr(self.global_step, self._total_steps, self.cfg.lr)
        current_lr = current_lr * (1.0 + self.cfg.dopamine_beta * D)
        for pg in self.optimizer.param_groups:
            pg['lr'] = current_lr
        # 4c: WM optimizer lr 对齐主 lr (缩放 0.1x)
        if self.world_model_optimizer is not None:
            wm_lr = get_lr(self.global_step, self._total_steps, self.cfg.lr) * 0.1
            for pg in self.world_model_optimizer.param_groups:
                pg['lr'] = wm_lr
        # ICM optimizer lr (缩放 0.05x)
        if self.icm_optimizer is not None:
            icm_lr = max(get_lr(self.global_step, self._total_steps, self.cfg.lr) * 0.05, 5e-5)
            for pg in self.icm_optimizer.param_groups:
                pg['lr'] = icm_lr

        if self.cfg.enable_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.world_model_optimizer is not None:
            self.world_model_optimizer.step()
        if self.icm_optimizer is not None:
            self.icm_optimizer.step()

        self._last_world_surprise = surprise
        self._last_world_mode = update_mode
        self._last_D = D
        # 内在动机统计
        if self._icm_output:
            for k in ['pred_loss', 'inverse_loss', 'information_gain', 'uncertainty']:
                self._intrinsic_stats[k].append(self._icm_output.get(k, 0.0))
            if self.concept_discovery:
                self._intrinsic_stats['n_concepts'].append(len(self.concept_discovery.concepts))
        F_final = F_hist[-1] if F_hist else 0.0
        F_val = F_pred.item() if F_pred is not None else ce_loss.item()
        scale_local_val = scale_local.item() if 'scale_local' in locals() else 1.0
        scale_conv_val = scale_conv.item() if 'scale_conv' in locals() else 1.0
        icm_loss_val = icm_loss.item() if isinstance(icm_loss, torch.Tensor) else 0.0
        result = {
            'ce_val': ce_loss.item(),
            'F_val': F_val,
            'F_final': F_final,
            'F_hist': F_hist,
            'errors_hist': errors_hist,
            'D': D,
            'lr': current_lr,
            'β_local': β_local,
            'β_conv': β_conv,
            'scale_local': scale_local_val,
            'scale_conv': scale_conv_val,
            'π': self.prev_precision_scales,
            'phase': 'full',
            'world_surprise': surprise,
            'world_loss': world_loss.item() if world_loss is not None else None,
            'update_mode': update_mode,
            'icm_loss': icm_loss_val,
        }
        if self._icm_output:
            for k in ['pred_loss', 'inverse_loss', 'information_gain', 'uncertainty']:
                result[f'icm_{k}'] = self._icm_output.get(k, 0.0)
        return result

    # ── CUDA Graph 单步 (纯 GPU, 无 .item / CPU sync) ───────

    def _graph_train_step(self, byte_seq: torch.Tensor, labels: torch.Tensor,
                          precision_scales, world_model_context: torch.Tensor | None = None) -> dict:
        """GPU-only train_step, 适合 CUDA Graph capture/replay。

        与 train_step 的区别:
            - 无 .item() / CPU sync
            - 无 dopamine.update() (在 graph 外执行)
            - 无 optimizer.step() / scaler.step() (在 graph 外执行)
            - 无梯度裁剪 (clip_grad_norm 涉及 CPU sync)
            - 无 lr_schedule / π 计算
            - 无 F_hist / errors_hist 收集
            - precision_scales 直接传入 tensor

        Returns:
            dict: 'total_loss' (tensor), 'F_pred' (tensor), 'ce_loss' (tensor),
                  'world_loss' (tensor), 'world_surprise' (tensor)
        """
        bsz, _, seq_len = byte_seq.shape
        self.global_step += 1

        pos_emb = self.model.get_position_embeddings(seq_len, self.device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.cfg.enable_amp):
            # ── Phase 1: forward_with_ce ──
            z_init, ce_loss = self.model.forward_with_ce(byte_seq, labels, pos_emb)
            z_init_det = [z.detach() for z in z_init]

            # ── Phase 2-3: spatiotemporal_infer + F_pred ──
            z_converged, _, _, F_pred = self.model.spatiotemporal_infer(
                z_init_det, pos_emb, gamma=self.cfg.gamma, T=self.cfg.T_infer,
                return_errors=False, return_pred_loss=True,
                precision_scales=precision_scales,
            )

            # ── 世界模型 (图形内，纯张量操作) ──
            world_loss = torch.tensor(0.0, device=self.device)
            world_surprise = torch.tensor(0.0, device=self.device)
            if self.cfg.enable_world_model and self.world_model is not None and world_model_context is not None:
                state_tensor = z_init_det[-1].detach()
                next_target = z_converged[-1].detach()
                _, uncertainty = self.world_model(state_tensor, world_model_context)
                wl = self.world_model.loss(state_tensor, next_target, world_model_context)
                world_loss = self.cfg.world_model_loss_weight * wl
                world_surprise = uncertainty.detach().mean()

            # ── Phase 4 (简化): 无 D, 仅合并 loss ──
            total_loss = ce_loss + F_pred + world_loss

        # ── Phase 5: backward ──
        self.optimizer.zero_grad(set_to_none=True)
        if self.world_model_optimizer is not None:
            self.world_model_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        return {
            'total_loss': total_loss.detach(),
            'F_pred': F_pred.detach() if F_pred is not None else torch.tensor(0.0, device=self.device),
            'ce_loss': ce_loss.detach(),
            'world_loss': world_loss.detach(),
            'world_surprise': world_surprise.detach(),
        }

    # ── 持续学习: 记忆回放 ──

    def _maybe_replay(self):
        if self.memory_bank.total <= 0:
            return
        if self.global_step % self.cfg.replay_ratio != 0:
            return
        if self.sniffer.is_repairing:
            return

        if self.cfg.enable_intrinsic_motivation and self.icm is not None:
            strategy = 'intrinsic'
        elif self.cfg.enable_world_model and self.world_model is not None:
            strategy = 'world_model' if self._last_world_surprise >= self.cfg.world_model_surprise_threshold else 'dopamine'
        else:
            strategy = 'dopamine'
        replay_ex = self.memory_bank.sample(self.cfg.batch_size, strategy=strategy)
        if not replay_ex:
            return

        replay_byte = torch.stack([ex.byte_tensor for ex in replay_ex], dim=0).to(self.device)
        replay_label = torch.stack([ex.label_tensor for ex in replay_ex], dim=0).to(self.device)
        replay_pos = self.model.get_position_embeddings(replay_byte.size(-1), self.device)

        _, replay_loss = self.model.forward_with_ce(replay_byte, replay_label, replay_pos)
        self.optimizer.zero_grad(set_to_none=True)
        replay_loss.backward()
        trainable_rp = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if trainable_rp:
            torch.nn.utils.clip_grad_norm_(trainable_rp, self.cfg.grad_clip)
        self.optimizer.step()

        # 5a: 刷新被回放样本的 transition_surprise 和 replay_priority
        if self.cfg.enable_world_model and self.world_model is not None:
            with torch.no_grad():
                z_rp, _ = self.model.forward_with_ce(replay_byte, replay_label, replay_pos)
                z_top = z_rp[-1].detach()
                ctx = self._build_world_model_context(replay_byte.size(0))
                _, uncertainty = self.world_model(z_top, ctx)
                new_surprise = uncertainty.mean().item()
            for ex in replay_ex:
                ex.transition_surprise = new_surprise
                ex.replay_priority = max(ex.dopamine_score, 0.1) + max(new_surprise, 0.0) + ex.intrinsic_value
        elif self.cfg.enable_intrinsic_motivation and self.icm is not None:
            # ICM 刷新 replay_priority
            if self._icm_output:
                info_gain = self._icm_output.get('information_gain', 0.0)
                for ex in replay_ex:
                    ex.replay_priority = max(ex.dopamine_score, 0.1) + info_gain

    def _maybe_abstraction_replay(self):
        gs = self.global_step
        if gs % self.cfg.abstraction_replay_interval != 0:
            return

        r_loss = self.abstraction_bank.replay_loss(
            self.model, batch_size=16, device=self.device,
            pos_emb=(None, None),
        )
        if r_loss is not None:
            self.optimizer.zero_grad(set_to_none=True)
            r_loss.backward()
            trainable_rp = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
            if trainable_rp:
                torch.nn.utils.clip_grad_norm_(trainable_rp, self.cfg.grad_clip)
            self.optimizer.step()
            self._log(f'[AbstractionBank] Replay step {gs}: loss={r_loss.item():.4f}')

        # 定时 consolidate
        if gs % (self.cfg.abstraction_replay_interval * 5) == 0:
            for tid in self.abstraction_bank._store:
                self.abstraction_bank.consolidate(tid)
                n_p = self.abstraction_bank.get_num_prototypes(tid)
                self._log(f'[AbstractionBank] Consolidated {tid}: {n_p} prototypes')

    def _maybe_sniff_forgetting(self):
        forgotten = self.sniffer.check(self.global_step, self.device)
        if not forgotten:
            return

        current_lr = self.optimizer.param_groups[0]['lr']
        repair_lr = self.sniffer.repair_begin(self.optimizer, current_lr, self.device)
        self._log(f'[Sniffer] FORGOTTEN: {forgotten} — repair LR={repair_lr:.2e}')

        # 世界模型 surprise 调制修复强度
        wm_factor = 1.0
        strategy = 'dopamine'
        if self.cfg.enable_world_model and self.world_model is not None:
            wm_surprise = self._last_world_surprise
            if wm_surprise > self.cfg.world_model_surprise_threshold:
                wm_factor = 1.0 + min(wm_surprise, 1.0)  # max 2x
                strategy = 'world_model'
            else:
                wm_factor = max(0.5, 1.0 - wm_surprise * 2)  # min 0.5x

        effective_steps = max(1, int(self.cfg.repair_steps * wm_factor))
        self._log(f'[Sniffer]  wm_surprise={self._last_world_surprise:.3f} '
                  f'factor={wm_factor:.2f} steps={effective_steps} strategy={strategy}')

        for _ in range(effective_steps):
            replay_data = self.sniffer.get_replay_batch(
                self.cfg.batch_size, self.device, strategy=strategy)
            if replay_data is None:
                break
            rp_byte, rp_label = replay_data
            rp_pos = self.model.get_position_embeddings(rp_byte.size(-1), self.device)
            z_init, rp_loss = self.model.forward_with_ce(rp_byte, rp_label, rp_pos)

            # 世界模型损失加入修复梯度（帮助模型同时修复 latent dynamics）
            if self.cfg.enable_world_model and self.world_model is not None:
                with torch.no_grad():
                    z_lat = z_init[-1].detach()
                wm_ctx = self._build_world_model_context(rp_byte.size(0))
                wl = self.world_model.loss(z_lat, z_lat, wm_ctx)
                rp_loss = rp_loss + self.cfg.world_model_loss_weight * wl

            self.optimizer.zero_grad(set_to_none=True)
            rp_loss.backward()
            trainable_rp = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
            if trainable_rp:
                torch.nn.utils.clip_grad_norm_(trainable_rp, self.cfg.grad_clip)
            self.optimizer.step()

        self.sniffer.repair_end(self.optimizer, current_lr)
        self._log(f'[Sniffer] Repair complete — LR restored to {current_lr:.2e}')

    def _maybe_sniff_abstraction(self):
        drifted = self.abstraction_sniffer.check(self.global_step, self.device, pos_emb=(None, None))
        if not drifted:
            return

        current_lr = self.optimizer.param_groups[0]['lr']
        repair_lr = self.abstraction_sniffer.repair_begin(self.optimizer, current_lr)
        self._log(f'[AbstractionSniffer] DRIFT detected: {drifted} — repair LR={repair_lr:.2e}')

        # 世界模型 surprise 调制修复强度
        wm_factor = 1.0
        if self.cfg.enable_world_model and self.world_model is not None:
            wm_surprise = self._last_world_surprise
            if wm_surprise > self.cfg.world_model_surprise_threshold:
                wm_factor = 1.0 + min(wm_surprise, 1.0)
            else:
                wm_factor = max(0.5, 1.0 - wm_surprise * 2)

        effective_steps = max(1, int(self.abstraction_sniffer.repair_steps * wm_factor))
        self._log(f'[AbstractionSniffer] wm_surprise={self._last_world_surprise:.3f} '
                  f'factor={wm_factor:.2f} steps={effective_steps}')

        for _ in range(effective_steps):
            r_loss = self.abstraction_bank.replay_loss(
                self.model, batch_size=16, device=self.device,
                pos_emb=(None, None),
            )
            if r_loss is None:
                break
            # 世界模型损失加入抽象修复梯度
            if self.cfg.enable_world_model and self.world_model is not None:
                proto_batch = self.abstraction_sniffer.get_replay_batch(4, self.device)
                if proto_batch:
                    z_protos, imp = proto_batch
                    z_top = z_protos[-1].mean(dim=1, keepdim=True)
                    ctx = self._build_world_model_context(z_top.size(0))
                    wl = self.world_model.loss(z_top, z_top, ctx)
                    r_loss = r_loss + self.cfg.world_model_loss_weight * wl
            self.optimizer.zero_grad(set_to_none=True)
            r_loss.backward()
            trainable_rp = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
            if trainable_rp:
                torch.nn.utils.clip_grad_norm_(trainable_rp, self.cfg.grad_clip)
            self.optimizer.step()

        self.abstraction_sniffer.repair_end(self.optimizer, current_lr)
        self._log(f'[AbstractionSniffer] Repair complete — LR restored')

    # ── 任务完成处理 ──

    def finalize_task(self, task_id: str, task_dataset, show_progress: bool = True):
        """采样 exemplars → MemoryBank + AbstractionBank."""
        n_samples = min(200, len(task_dataset))
        idx = torch.randperm(len(task_dataset))[:n_samples].tolist()

        samples = []
        total_bl = 0.0
        with torch.no_grad():
            for i in idx:
                bt, lt = task_dataset[i]
                samples.append((bt, lt))
                x = bt.unsqueeze(0).to(self.device)
                y = lt.unsqueeze(0).to(self.device)
                p = self.model.get_position_embeddings(x.size(-1), self.device)
                _, bl = self.model.forward_with_ce(x, y, p)
                total_bl += bl.item()

        avg_bl = total_bl / max(len(idx), 1)
        D_score = self._last_D if hasattr(self, '_last_D') else 0.5

        # 内在动机: 获取 ICM 信息增益和概念 ID
        info_gain_val = 0.0
        concept_id_val = ''
        if self.cfg.enable_intrinsic_motivation and self._icm_output:
            info_gain_val = self._icm_output.get('information_gain', 0.0)

        self.memory_bank.add_samples(
            task_id,
            samples,
            D_score,
            avg_bl,
            transition_surprise=self._last_world_surprise,
            intrinsic_value=info_gain_val,
            concept_id=concept_id_val,
        )
        self._log(f'[Continual] Task {task_id}: {n_samples} exemplars → bank '
                  f'(D={D_score:.3f}, baseline_CE={avg_bl:.4f}) — bank total: {self.memory_bank.total}')

        # AbstractionBank
        z_collected = []
        for bt, lt in samples[:100]:
            x = bt.unsqueeze(0).to(self.device)
            with torch.no_grad():
                z_init = self.model.init_z(x)
            z_conv, *_ = self.model.spatiotemporal_infer(
                z_init, pos_emb=(None, None),
                gamma=self.cfg.gamma, T=4,
                return_errors=False, return_pred_loss=False,
            )
            z_collected.append(z_conv)

        if z_collected:
            layer_imp = compute_layer_importance(
                z_collected[0], self.model, (None, None),
                dopamine_D=D_score, eta=1.0,
            )
            self.abstraction_bank.add_z_samples(
                task_id, z_collected,
                layer_importance=layer_imp,
                dopamine_score=D_score,
                world_model_surprise=self._last_world_surprise,
                concept_id=concept_id_val,
                information_gain=info_gain_val,
                group_by_concept=False,
            )
            self.abstraction_bank.consolidate(task_id)
            n_protos = self.abstraction_bank.get_num_prototypes(task_id)
            self._log(f'[AbstractionBank] Task {task_id}: '
                      f'{len(z_collected)} z_states → {n_protos} prototypes')

        # 6a: WM 驱动的合并后巩固
        if self.cfg.enable_world_model and self.cfg.sleep_consolidation and self.world_model is not None:
            self._sleep_consolidation(task_id)

    # ── Phase 6a: Sleep Consolidation ─────────────────────────────────

    def _sleep_consolidation(self, task_id: str):
        """用世界模型 transition_error 调整 store retention_weights → 重新合并。"""
        if not hasattr(self, 'abstraction_bank') or self.abstraction_bank is None:
            return
        entries = self.abstraction_bank._store.get(task_id, [])
        if not entries:
            return

        device = self.device
        n_adjusted = 0
        with torch.no_grad():
            wm_ctx = self._build_world_model_context(1)
            for e in entries:
                z_top = e['z_states'][-1]  # [1, seq, hidden]
                # sample middle token as representative
                seq_len = z_top.size(1)
                mid_idx = seq_len // 2
                z_rep = z_top[:, mid_idx:mid_idx+1, :]  # [1, 1, hidden]
                z_pred, _ = self.world_model(z_rep, wm_ctx)
                t_err = torch.mean((z_pred - z_rep) ** 2).item()
                # decay: higher error => lower retention
                decay = math.exp(-t_err * 2.0)
                old_w = e.get('retention_weight', 1.0)
                new_w = old_w * (0.5 + 0.5 * decay)  # [0.25, 1.0] × old
                e['retention_weight'] = new_w
                n_adjusted += 1

        if n_adjusted:
            self.abstraction_bank.consolidate(task_id)
            self._log(f'[Sleep/6a] WM-driven reconsolidation Task {task_id}: '
                      f'adjusted {n_adjusted} entries')

    # ── Phase 6b: Full Sleep Phase ───────────────────────────────────

    def _full_sleep_phase(self, task_pipelines: list):
        """全部任务后: 按 WM 不确定性排序 → OfflineReplayer replay → WM filter."""
        if not self.cfg.enable_world_model or self.world_model is None:
            return
        self._log('[Sleep/6b] Full sleep phase — generating WM-filtered replay...')

        # 收集每个任务的 WM uncertainty
        task_uncertainties: list[tuple[str, float]] = []
        for task_id, _ in task_pipelines:
            protos = self.abstraction_bank.get_prototypes(task_id)
            if not protos:
                task_uncertainties.append((task_id, 0.5))
                continue
            errors = []
            device = self.device
            with torch.no_grad():
                wm_ctx = self._build_world_model_context(1)
                for z_proto in protos[:20]:  # 采样 20 个
                    z_t = torch.as_tensor(z_proto, device=device).unsqueeze(0)
                    z_pred, _ = self.world_model(z_t, wm_ctx)
                    errors.append(torch.mean((z_pred - z_t) ** 2).item())
            task_uncertainties.append((task_id, sum(errors) / max(len(errors), 1)))

        # 按 uncertainty 降序
        task_uncertainties.sort(key=lambda x: x[1], reverse=True)
        top_k = min(self.cfg.sleep_replay_tasks, len(task_uncertainties))

        # 获取 OfflineReplayer（已在 train() 中初始化）
        replayer = OfflineReplayer(
            model=self.model,
            tokenizer=None,
            memory_bank=self.memory_bank,
            abstraction_bank=self.abstraction_bank,
            for_token_free=True,
            world_model=self.world_model,
        )

        for task_id, unc in task_uncertainties[:top_k]:
            self._log(f'[Sleep/6b] Replaying task {task_id} '
                      f'(uncertainty={unc:.4f}), generating {self.cfg.sleep_replay_samples} samples...')
            replayer.generate_for_task(
                task_id,
                n_samples=self.cfg.sleep_replay_samples,
                max_length=64,
                temperature=0.8,
                enable_wm_filter=True,
                enable_wm_temperature=True,
            )
        self._log('[Sleep/6b] Done')

    # ── 检查点 ──

    def _build_ckpt(self, epoch: int, task_id: str = None, metrics: dict = None) -> dict:
        ckpt = {
            'epoch': epoch,
            'step': self.global_step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'lm_config': MiniMindConfig(
                hidden_size=self.cfg.hidden_size,
                num_hidden_layers=self.cfg.num_hidden_layers,
                use_moe=self.cfg.use_moe,
            ),
            'config': self.cfg.to_dict(),
        }
        if self.cfg.enable_world_model and self.world_model is not None:
            ckpt['world_model_state'] = self.world_model.state_dict()
        if self.cfg.enable_intrinsic_motivation and self.icm is not None:
            ckpt['icm_state'] = self.icm.state_dict()
            ckpt['icm_optimizer_state'] = self.icm_optimizer.state_dict() if self.icm_optimizer else None
        if metrics:
            ckpt.update(metrics)
        ckpt['memory_bank'] = self.memory_bank.state_dict()
        ckpt['abstraction_bank'] = self.abstraction_bank.state_dict()
        if task_id:
            ckpt['task_id'] = task_id
        return ckpt

    def save_checkpoint(self, path: str, epoch: int = 0, task_id: str = None):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(self._build_ckpt(epoch, task_id), path)
        self._log(f'Checkpoint saved → {path}')

    # ── 跨任务遗忘评估 ──

    @torch.no_grad()
    def evaluate_cross_tasks(self, task_ds_list: list, n_samples: int = None) -> dict:
        """评估模型在所有已学任务上的 CE/PPL。"""
        results = {}
        self.model.eval()
        n = n_samples or self.cfg.eval_samples
        for tid, ds in task_ds_list:
            n_eval = min(n, len(ds))
            total_ce = 0.0
            for i in range(n_eval):
                bt, lt = ds[i]
                x = bt.unsqueeze(0).to(self.device)
                y = lt.unsqueeze(0).to(self.device)
                p = self.model.get_position_embeddings(x.size(-1), self.device)
                _, ce = self.model.forward_with_ce(x, y, p)
                total_ce += ce.item()
            avg_ce = total_ce / max(n_eval, 1)
            results[tid] = {'ce': avg_ce, 'ppl': math.exp(min(avg_ce, 20))}
        self.model.train()
        return results

    # ── 主训练循环 ──

    def train(self, task_pipelines: list[tuple[str, torch.utils.data.Dataset, Optional[DataLoader]]]):
        """
        主训练入口。

        Args:
            task_pipelines: [(task_id, dataset, loader_override?), ...]
                            loader_override 可选, 默认自动创建 DataLoader
        """
        out_dir = os.path.join(os.getcwd(), self.cfg.out_dir)
        os.makedirs(out_dir, exist_ok=True)

        # 创建模型
        self.model = self._build_model().to(self.device)
        self._log(f'Model: {count_parameters(self.model)["trainable_M"]:.2f}M trainable params')

        # QAT 参数统计
        if self.cfg.enable_qat:
            self._log(f'QAT params: {count_parameters(self.model)["trainable_M"]:.2f}M (post-QAT)')

        # 世界模型必须在 checkpoint 加载之前初始化，否则 world_model_state 无处加载
        self.graph_trainer: Optional['CUDAGraphTrainer'] = None
        if self.cfg.enable_world_model:
            self.world_model = LatentWorldModel(
                input_dim=self.cfg.hidden_size,
                hidden_dim=self.cfg.world_model_hidden_dim,
                context_dim=self.cfg.world_model_context_dim,
            ).to(self.device)
            self.world_model_optimizer = torch.optim.AdamW(
                self.world_model.parameters(),
                lr=max(self.cfg.lr * 0.1, 1e-4),
                betas=(0.9, 0.95),
            )
            self._log('Latent world model enabled')

        # ── 内在动机模块 ──
        if self.cfg.enable_intrinsic_motivation:
            self.icm = IntrinsicCuriosityModule(
                state_dim=self.cfg.hidden_size,
                action_dim=self.cfg.icm_action_dim,
                hidden_dim=self.cfg.icm_hidden_dim,
                contrastive_negative=64,
            ).to(self.device)
            self.icm_optimizer = torch.optim.AdamW(
                self.icm.parameters(),
                lr=max(self.cfg.lr * 0.05, 5e-5),
                betas=(0.9, 0.95),
            )
            self.concept_discovery = ConceptDiscovery(
                threshold_init=self.cfg.concept_threshold_init,
                threshold_min=self.cfg.concept_threshold_min,
            )
            self.memory_gate = MemoryGate(
                threshold_low=0.05,
                threshold_high=0.5,
                target_storage_ratio=self.cfg.gate_target_storage,
                target_high_value_ratio=self.cfg.gate_target_high,
            )
            self._log(f'Intrinsic motivation enabled (ICM + Concept + Gate)')

        # 从 checkpoint 恢复
        if self.cfg.checkpoint_path:
            ckpt = torch.load(self.cfg.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state'])
            if self.cfg.enable_world_model and self.world_model is not None and 'world_model_state' in ckpt:
                self.world_model.load_state_dict(ckpt['world_model_state'])
            self._log(f'Resumed from checkpoint: {self.cfg.checkpoint_path}')

        # 优化器
        self.optimizer = self._build_optimizer()
        self.dopamine = DopamineSignal(η=self.cfg.dopamine_eta, threshold=0.0)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.cfg.enable_amp)

        # Sniffer 绑定 model
        self.sniffer.model = self.model
        self.abstraction_sniffer.model = self.model
        self.abstraction_sniffer.world_model = self.world_model

        # 预热
        self.warmup()

        # CUDA Graphs 录制 (需 warmup 后, 确保 cudnn 算法固定)
        if self.cfg.use_cuda_graphs and self.device.type == 'cuda':
            from core.cuda_graphs import CUDAGraphTrainer
            self.graph_trainer = CUDAGraphTrainer(self, warmup_steps=10)
            self.graph_trainer.capture()
            self._log('CUDA Graphs captured')

        # 训练循环
        trained_tasks: list[str] = []
        self._trained_tasks = trained_tasks
        self.model.train()

        for task_id, task_ds in task_pipelines:
            self._current_task_id = task_id

            # 确保 task_pipelines 简化为 (task_id, dataset)
            if isinstance(task_ds, DataLoader):
                loader = task_ds
                task_ds = loader.dataset
            else:
                loader = DataLoader(task_ds, batch_size=self.cfg.batch_size,
                                    shuffle=True, num_workers=0, pin_memory=True)

            # 8a+8b: 新任务 novelty 加速 + WM reset
            if self.cfg.enable_world_model:
                self._novelty_boost_steps = int(len(task_ds) * 0.05)  # 前 5% 步数
                self._novelty_surprise_injected = self.cfg.world_model_surprise_threshold * 2.0
                if hasattr(self, 'world_model') and self.world_model is not None:
                    self.world_model.reset_state()
                    self._log(f'[Novelty/8b] WM state reset for task {task_id}')
            # ICM reset on task switch
            if self.cfg.enable_intrinsic_motivation and self.icm is not None:
                self.icm.reset_state()
                self._icm_output = None

            self._total_steps = len(loader) * self.cfg.epochs
            self.global_step = 0

            self._log(f'\n{"=" * 60}\nStarting Task {task_id}: {len(loader.dataset)} samples\n{"=" * 60}')

            for epoch in range(self.cfg.epochs):
                pbar = tqdm(loader,
                            desc=f'Task {task_id} Epoch {epoch + 1}/{self.cfg.epochs}',
                            unit='step', dynamic_ncols=True, ascii=True)

                for byte_seq, labels in pbar:
                    byte_seq = byte_seq.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    if self.graph_trainer is not None and self.graph_trainer.is_captured():
                        # ── CUDA Graph 模式 ──
                        # 构建世界模型上下文
                        wm_context = None
                        if self.cfg.enable_world_model and self.world_model is not None:
                            wm_context = self._build_world_model_context(byte_seq.size(0))

                        g_out = self.graph_trainer.replay(
                            byte_seq, labels,
                            self.prev_precision_scales,
                            wm_context,
                        )
                        # Graph 外: optimizer.step + scaler + dopamine
                        trainable = [p for p in self.model.parameters()
                                     if p.requires_grad and p.grad is not None]
                        if trainable:
                            if self.cfg.enable_amp:
                                self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(trainable, self.cfg.grad_clip)

                        current_lr = get_lr(self.global_step, self._total_steps, self.cfg.lr)
                        D = self.dopamine.update(g_out['F_pred'].item())
                        current_lr = current_lr * (1.0 + self.cfg.dopamine_beta * D)
                        for pg in self.optimizer.param_groups:
                            pg['lr'] = current_lr

                        if self.cfg.enable_amp:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        # 世界模型优化器 (图形外)
                        if self.world_model_optimizer is not None:
                            # 世界模型的梯度来自 graph 中 total_loss.backward()
                            wm_trainable = [p for p in self.world_model.parameters()
                                            if p.grad is not None]
                            if wm_trainable:
                                torch.nn.utils.clip_grad_norm_(wm_trainable, self.cfg.grad_clip)
                            self.world_model_optimizer.step()

                        w_surprise = g_out['world_surprise'].item()
                        self._last_world_surprise = w_surprise
                        self._last_world_mode = 'full'  # graph mode 始终走完整 PC 推理
                        m = {
                            'ce_val': g_out['ce_loss'].item(),
                            'F_val': g_out['F_pred'].item(),
                            'F_final': g_out['F_pred'].item(),
                            'F_hist': [],
                            'errors_hist': [],
                            'D': D,
                            'lr': current_lr,
                            'β_local': 0.0,
                            'β_conv': 0.0,
                            'scale_local': 1.0,
                            'scale_conv': 1.0,
                            'π': None,
                            'phase': 'graph',
                            'world_surprise': w_surprise,
                        }
                    else:
                        # ── 标准模式 ──
                        m = self.train_step(byte_seq, labels)

                    # ── ICM / Concept / Gate 后处理 ──
                    if self.cfg.enable_intrinsic_motivation and self.icm is not None:
                        # 概念 consolidation (每 500 步)
                        if self.global_step % 500 == 0 and self.concept_discovery is not None:
                            self.concept_discovery.consolidate()
                        # 记忆门控自适应 (每 adaptation_window 步)
                        if self.memory_gate is not None:
                            self.memory_gate.adapt_thresholds()
                        # ICM reset (每任务, 在任务切换时处理)

                    # 8a: novelty 阶段注入高 surprise → 强制 full-update
                    if self._novelty_boost_steps > 0 and self.cfg.enable_world_model:
                        m['world_surprise'] = max(
                            m.get('world_surprise', 0.0),
                            self._novelty_surprise_injected,
                        )
                        self._novelty_boost_steps -= 1
                        if self._novelty_boost_steps == 0:
                            self._novelty_surprise_injected = 0.0

                    self._last_D = m['D']
                    self._F_trend_buffer.append(m.get('F_val', m.get('F_final', 0.0)))
                    if len(self._F_trend_buffer) > 100:
                        self._F_trend_buffer.pop(0)
                    self.sniffer.update_surprise(m.get('world_surprise', 0.0))
                    # 记录 surprise 采样 (每 100 步)
                    if self.global_step % 100 == 0:
                        self._surprise_buffer.append(m.get('world_surprise', 0.0))

                    # 7a: WM 滚动指标
                    if self.cfg.enable_world_model:
                        wl = m.get('world_loss')
                        ws = m.get('world_surprise', 0.0)
                        if wl is not None:
                            self._wm_metrics['transition_error'].append(wl)
                        self._wm_metrics['uncertainty'].append(ws)
                        # 假阳性检测: 高 surprise 但 CE 未升
                        ce_now = m['ce_val']
                        if m.get('update_mode') == 'full' and ws > self.cfg.world_model_surprise_threshold:
                            self._wm_high_surprise_count += 1
                            if ce_now <= self._last_ce_for_fp:
                                self._wm_fp_count += 1
                        # 每 100 步记录 FP rate
                        if self.global_step % 100 == 0 and self._wm_high_surprise_count > 0:
                            fp_rate = self._wm_fp_count / max(self._wm_high_surprise_count, 1)
                            self._wm_metrics['fp_rate'].append(fp_rate)
                        self._last_ce_for_fp = ce_now

                    # 持续学习: 记忆回放
                    self._maybe_replay()
                    # 持续学习: AbstractionBank 回放
                    self._maybe_abstraction_replay()
                    # 遗忘嗅探 + 修复
                    self._maybe_sniff_forgetting()
                    # 抽象漂移检测
                    self._maybe_sniff_abstraction()

                    # 进度条
                    postfix = {
                        'CE': f'{m["ce_val"]:.4f}',
                        'F': f'{m["F_final"]:.1f}',
                        'D': f'{m["D"]:.3f}',
                        'W': f'{m.get("world_surprise", 0.0):.3f}',
                    }
                    if self.cfg.enable_intrinsic_motivation and self._icm_output:
                        postfix['IG'] = f'{self._icm_output.get("information_gain", 0.0):.4f}'
                        if self.concept_discovery:
                            postfix['C'] = f'{len(self.concept_discovery.concepts)}'
                    pbar.set_postfix(**postfix)

                    # 进度回调 (每步通知 GUI)
                    callback_dict = {
                        'type': 'progress', 'step': self.global_step,
                        'total_steps': self._total_steps,
                        'ce_loss': m["ce_val"],
                        'F': m["F_final"],
                        'D': m["D"],
                        'lr': m["lr"],
                    }
                    if self.cfg.enable_world_model:
                        callback_dict.update({
                            'world_surprise': m.get('world_surprise', 0.0),
                            'world_loss': m.get('world_loss'),
                            'wm_metrics': {k: (v[-1] if v else 0.0)
                                           for k, v in self._wm_metrics.items()},
                        })
                    if self.cfg.enable_intrinsic_motivation and self._icm_output:
                        callback_dict.update({
                            'information_gain': self._icm_output.get('information_gain', 0.0),
                            'icm_pred_loss': self._icm_output.get('pred_loss', 0.0),
                            'n_concepts': len(self.concept_discovery.concepts) if self.concept_discovery else 0,
                        })
                    if self.cfg.progress_callback:
                        self.cfg.progress_callback(callback_dict)

                    # 日志 (每 100 步)
                    if self.global_step % 100 == 0 and self.global_step > 0:
                        log = (f'[Step {self.global_step}/{self._total_steps}] '
                               f'F={m["F_final"]:.1f} CE={m["ce_val"]:.4f} '
                               f'D={m["D"]:.3f} lr={m["lr"]:.2e} '
                               f'W={m.get("world_surprise", 0.0):.3f}')
                        if self.cfg.enable_intrinsic_motivation and self._icm_output:
                            log += (f' IG={self._icm_output.get("information_gain", 0.0):.4f}'
                                    f' ICML={m.get("icm_loss", 0.0):.4f}'
                                    f' C={len(self.concept_discovery.concepts) if self.concept_discovery else 0}')
                        if m['π']:
                            π_str = ','.join(f'{p:.2f}' for p in m['π'])
                            log += f' π=[{π_str}]'
                        # 7a: WM 指标
                        if self.cfg.enable_world_model and self.global_step % 500 == 0:
                            te = self._wm_metrics['transition_error']
                            uq = self._wm_metrics['uncertainty']
                            fp = self._wm_metrics['fp_rate']
                            log += (f' | WM: TE={(te[-1] if te else 0):.4f} '
                                    f'U={(sum(uq[-100:])/max(len(uq[-100:]),1)):.4f} '
                                    f'FP={(fp[-1] if fp else 0):.3f}')
                        self._log(log)

                    # 检查点
                    if self.global_step % self.cfg.save_interval == 0 or self.global_step == 1:
                        ckpt_path = os.path.join(out_dir, f'unified_ckpt_s{self.global_step}.pt')
                        self.save_checkpoint(ckpt_path, epoch, task_id)
                        if self.cfg.progress_callback:
                            self.cfg.progress_callback({
                                'type': 'checkpoint', 'step': self.global_step,
                                'checkpoint_path': ckpt_path,
                            })

            # ── 任务完成 ──
            self.finalize_task(task_id, task_ds if isinstance(task_ds, torch.utils.data.Dataset) else loader.dataset)

            # ── 内在动机: 概念 consolidation + 统计 ──
            if self.cfg.enable_intrinsic_motivation and self.concept_discovery is not None:
                self.concept_discovery.consolidate()
                n_concepts = len(self.concept_discovery.concepts)
                fragile = len(self.concept_discovery.get_fragile_concept_ids())
                self._log(f'[Intrinsic] After {task_id}: {n_concepts} concepts ({fragile} fragile)')
                # 打印概念摘要
                for c in list(self.concept_discovery.concepts.values())[:10]:
                    self._log(f'  Concept {c.concept_id}: support={c.support}, '
                              f'avg_IG={c.avg_intrinsic_value:.4f}')
                # 内在动机统计
                ig_mean = (sum(self._intrinsic_stats['information_gain'][-500:]) /
                           max(len(self._intrinsic_stats['information_gain'][-500:]), 1))
                self._log(f'[Intrinsic Stats] IG_mean_500={ig_mean:.4f}, '
                          f'n_concepts_peak={max(self._intrinsic_stats.get("n_concepts", [0]))}')
            if self.memory_gate is not None:
                gate_stats = self.memory_gate.get_stats()
                self._log(f'[MemoryGate] threshold_low={gate_stats["threshold_low"]:.4f}, '
                          f'threshold_high={gate_stats["threshold_high"]:.4f}, '
                          f'storage_ratio={gate_stats["storage_ratio"]:.3f}')

            # 遗忘评估
            trained_tasks.append(task_id)
            eval_ds_list = [(tid, ds) for tid, ds in task_pipelines if tid in trained_tasks]
            eval_results = self.evaluate_cross_tasks(eval_ds_list)
            self._log(f'[Eval] After Task {task_id}:')
            for tid, metrics in eval_results.items():
                marker = ' ← trained' if tid == task_id else ''
                self._log(f'  Task {tid}: CE={metrics["ce"]:.4f}, PPL={metrics["ppl"]:.2f}{marker}')

            # 5b: 世界模型 surprise 时序加入 forgetting_log
            surprise_timeline = []
            if self.cfg.enable_world_model and hasattr(self, '_surprise_buffer'):
                surprise_timeline = list(self._surprise_buffer)

            self.forgetting_log.append({
                'after_task': task_id,
                'results': eval_results,
                'avg_world_surprise': (
                    sum(surprise_timeline) / max(len(surprise_timeline), 1)
                    if surprise_timeline else None
                ),
                'surprise_timeline': surprise_timeline,
            })
            # 清空 surprise buffer 供下一任务使用
            self._surprise_buffer.clear()
            with open(os.path.join(out_dir, 'forgetting_log.json'), 'w') as f:
                json.dump(self.forgetting_log, f, indent=2)

            # 任务检查点
            self.save_checkpoint(
                os.path.join(out_dir, f'task_{task_id}_final.pt'),
                self.cfg.epochs - 1, task_id,
            )

        # ── Phase 6b: Full Sleep ──
        if self.cfg.enable_world_model and self.cfg.full_sleep_after_all:
            self._full_sleep_phase(task_pipelines)

        # ── 最终保存 ──
        self.save_checkpoint(os.path.join(out_dir, 'unified_final.pt'), self.cfg.epochs - 1)

        # QAT 转换 → int4
        if self.cfg.enable_qat and self.quantizer is not None:
            self._log('Converting to int4 inference format (CPU)...')
            model_cpu = self.model.cpu()
            try:
                model_cpu = self.quantizer.convert(model_cpu)
                torch.save(model_cpu.state_dict(), os.path.join(out_dir, 'int4_model.pt'))
                self._log(f'Int4 model saved → {out_dir}/int4_model.pt')
            except Exception as e:
                self._log(f'Int4 convert failed (non-critical): {e}')

        self._log('Training complete.')
        if self.cfg.progress_callback:
            self.cfg.progress_callback({'type': 'done', 'message': 'Training complete'})
