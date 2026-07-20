"""统一训练循环 — PC 时空推理 + 多巴胺调制 + fp16 原生训练。

用法:
    from model.core.training import TrainingLoop, TrainingConfig

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
import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Optional

# 常数
_LOG2 = math.log(2)

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.continual.abstraction_bank import (
    AbstractionBank,
    AbstractionSniffer,
    compute_layer_importance,
)

# 吸引子景观 + 持续巩固 + 深度睡眠
from model.continual.attractor_landscape import AttractorLandscape
from model.continual.concept_discovery import ConceptDiscovery
from model.continual.consolidation_pipeline import (
    ConsolidationPipeline,
)
from model.continual.deep_sleep import SleepEngine
from model.continual.forgetting_sniffer import ForgettingSniffer
from model.continual.hippocampus_buffer import HippocampusBuffer

# 内在动机模块
from model.continual.intrinsic_curiosity import IntrinsicCuriosityModule

# 持续学习模块
from model.continual.memory_bank import MemoryBank
from model.continual.memory_gating import MemoryGate
from model.continual.neurogenesis import NeurogenesisController
from model.continual.offline_replay import OfflineReplayer
from model.continual.world_model import LatentWorldModel
from model.core.globals import DEVICE
from model.core.trainer_utils import count_budget, setup_seed
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
    vocab_size: int = 256           # 字节级默认 256; 词元级可覆盖
    checkpoint_path: Optional[str] = None  # 从 checkpoint 恢复

    # 训练
    batch_size: int = 48
    max_seq_len: int = 128
    lr: float = 3e-4
    epochs: int = 1
    subset: int = 0           # 0 = 全量
    seed: int = 42
    split_size: int = 0       # 0 = 不分块, >0 = 每块样本数 (GUI 传入)

    # PC 时空推理
    T_infer: int = 2
    gamma: float = 0.1

    # 时间预测损失 (temp_loss): 驱动 backbone 产出自预测的 z
    enable_temp_loss: bool = True
    temp_loss_weight: float = 0.1

    # 多巴胺
    dopamine_eta: float = 1.0
    dopamine_beta: float = 0.5
    dopamine_gamma: float = 0.3

    # Hebbian 模式超参
    hebbian_base_eta: float = 3e-4             # 基础 Hebbian 学习率 (原 3e-6→3e-4, 提升 100×)
    hebbian_lambda_decay: int = 5000           # τ_λ, decoder 约束退火衰减常数
    hebbian_lambda_min: float = 0.01           # λ_min, decoder 约束永不归零
    hebbian_infer_T: int = 3                   # PC 推理步数 (原 1→3, 更好表示)
    hebbian_ach_beta_0: float = 0.0            # ACh 偏置 β₀

    # Oja 规则 (权重自组织)
    oja_alpha: float = 0.05                    # Oja 衰减系数 (per-sample 更新, 与 temporal/topdown 一致)
    oja_eta: float = 0.05                      # Oja 独立学习率 (不绑定 Hebbian η)
    oja_adaptive: bool = True                  # 自适应 oja (按层范数自动缩放)

    # 误差归一化 (生物发放率约束)
    ε_rms_target: float = 1.0                  # ε 目标 RMS (模拟发放率上限)

    # 突触归一化 (Synaptic Normalization)
    synaptic_normalize: bool = False           # 禁用 — 与增强 Hebbian 更新冲突
    synaptic_target_norm: float = 0.0          # 目标 L2 范数 (0=auto: sqrt(fan_in))

    # Salience Gating (结构自组织)
    enable_salience_gating: bool = True        # 启用门控
    salience_temperature: float = 0.1          # 门控温度 (越低越硬)
    salience_reg_weight: float = 0.001         # 稀疏正则权重 β
    salience_gate_lr: float = 1e-3             # 门控 logits 学习率 (直接 SGD)

    # 神经发生 (Neurogenesis / 结构自组织)
    enable_neurogenesis: bool = True             # 启用通道剪枝+生长
    neurogenesis_prune_interval: int = 100       # 剪枝间隔 (步)
    neurogenesis_grow_interval: int = 300        # 生长间隔 (步)
    neurogenesis_prune_threshold_act: float = 0.001  # 激活 EMA 剪枝阈值
    neurogenesis_prune_threshold_gate: float = 0.05  # Gate 值剪枝阈值
    neurogenesis_grow_error_threshold: float = 2.0   # 生长触发误差
    neurogenesis_max_grow_per_step: int = 8          # 单步最大生长通道数

    # Phase 7b: 降频 (consolidation/ICM/WM 不必每步执行)
    consolidation_pipeline_interval: int = 5          # 巩固管道 tick/force 检查间隔 (步)

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

    # Phase 1: 依赖阈值发放 (自然稀疏替代 hardcoded top-k)
    act_threshold_init: float = 0.3           # per-channel 阈值初始值
    act_target_ratio: float = 0.20            # 目标活跃率 ~20%
    act_ema_decay: float = 0.999              # 激活分布 EMA 衰减
    act_homeo_rate: float = 0.02              # 稳态可塑性调节步长
    act_energy_cost: float = 0.5              # β·N_active 能量代价 (Phase 3: F += β·active_ratio)

    # Phase 2: 突触竞争 — per-weight-row WTA (模拟 ~1% 突触增强率)
    synaptic_competition_k: int = 8           # 每行胜者数 (0=禁用, ≈hidden*0.01)
    synaptic_competition_use_abs: bool = False  # False=按代数值, True=按绝对值竞争

    # Phase 4: 稀疏外积 — 仅计算活跃突触前通道 (替代全矩阵再归零)
    sparse_outer_k: int = 32                  # 保留的活跃突触前通道数 (0=禁用, ≈hidden*0.04)
    hebbian_eps_gate: float = 0.0             # ε 门控阈值 (0=禁用, >0=跳过 ‖ε‖<阈值的层)

    # 推理控制
    infer_adaptive_T: bool = True             # 自适应推理终止 (收敛后提前结束)
    infer_convergence_threshold: float = 0.05 # F 相对变化阈值 (5%)
    infer_patience: int = 2                   # 连续收敛步数
    infer_min_T: int = 1                      # 最小推理步数 · 生物单次传播

    # Stride 回放
    replay_stride: int = 4                    # 回放时序列下采样步长 (1=无下采样, 4=减4倍)

    # Sleep / Consolidation
    sleep_consolidation: bool = True          # 6a: 任务后 WM 驱动合并
    full_sleep_after_all: bool = True         # 6b: 全部任务后完整睡眠
    sleep_replay_tasks: int = 2               # 6b: 睡眠阶段回放最不确定的 N 个任务
    sleep_replay_samples: int = 100           # 6b: 每个任务生成样本数

    # 吸引子景观 + 持续巩固管道 (Phase A-C)
    enable_consolidation_pipeline: bool = True
    pipeline_buffer_capacity: int = 500
    pipeline_memory_write_interval: int = 50   # 每 N 步写 MemoryBank
    pipeline_abstraction_write_interval: int = 200  # 每 N 步写 AbstractionBank
    pipeline_sleep_check_interval: int = 500
    pipeline_min_info_gain: float = 0.05

    # 深度 SLEEP (Phase F)
    enable_deep_sleep: bool = True
    sleep_completion_steps: int = 20
    sleep_noise_steps: int = 20
    sleep_competitive_steps: int = 10
    # (grad_clip and lr_scale removed — Hebbian updates have internal protection)

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
        self.model: Optional['CyrenePC'] = None
        self.dopamine: Optional[DopamineSignal] = None
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
        self.forgetting_log: list[dict] = []
        self._last_world_surprise: float = 0.0
        self._last_world_loss: Optional[float] = None
        self._last_world_mode: str = 'full'
        self._F_trend_buffer: list[float] = []
        self._surprise_buffer: list[float] = []
        self._current_task_id: str = ''
        self._trained_tasks: list[str] = []

        # 上一步 F_total (用于多巴胺调制)
        self._last_F_bp: float = float('inf')
        self._global_F_hist: list[float] = []         # 全局 F 历史

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

        # ── BCM 滑动阈值 (生物可塑性平衡) ──
        self.bcm_state = BCMState(n_layers=24, tau=0.01).to(self.device)

        # ── NaN 回退状态 (初始权重快照, 用于 NaN 恢复) ──
        self._fallback_state: Optional[dict] = None

        # ── 内在动机模块 ──
        self.icm: Optional[IntrinsicCuriosityModule] = None
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

        # ── 吸引子景观 + 持续巩固 (Phase A-C) ──
        self.landscape: Optional[AttractorLandscape] = None
        self.consolidation_pipeline: Optional[ConsolidationPipeline] = None
        self.sleep_engine: Optional[SleepEngine] = None

        # ── 海马体快速缓冲 (Phase 3b) ──
        self.hippocampus = HippocampusBuffer(capacity=200, min_info_gain=0.03)

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

    def _build_model(self) -> 'CyrenePC':
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
        # 覆盖初始阈值 (重填 buffer)
        init_th = torch.ones(model.num_sub_layers, model.config.hidden_size) * self.cfg.act_threshold_init
        model.register_buffer('_act_threshold', init_th)

        # 编译 (需要 CUDA + Triton; 否则退化为 CPU inductor 需 MSVC)
        if hasattr(torch, 'compile') and self.device.type == 'cuda' and hasattr(torch, 'triton'):
            self._orig_forward = model.forward_with_ce
            try:
                model.forward_with_ce = torch.compile(self._orig_forward, mode='reduce-overhead')
                self._log('torch.compile 启用 (mode=reduce-overhead)')
            except Exception as e:
                model.forward_with_ce = self._orig_forward
                self._log(f'torch.compile 失败 (已忽略): {e}')

        # Phase 7c: 编译 PC 推理热路径 (需要 CUDA + Triton; Windows 无 Triton 时会退化为 CPU inductor 需 MSVC)
        if hasattr(torch, 'compile') and self.device.type == 'cuda' and hasattr(torch, 'triton'):
            try:
                model._spatiotemporal_infer_step = torch.compile(
                    model._spatiotemporal_infer_step, mode='reduce-overhead',
                )
                self._log('torch.compile → _spatiotemporal_infer_step 启用')
            except Exception as e:
                self._log(f'torch.compile → _spatiotemporal_infer_step 失败: {e}')

        return model

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
        n_concepts = self.concept_discovery.n_concepts if self.concept_discovery else 0
        n_concepts_norm = min(n_concepts / 20.0, 1.0)
        icm_uncertainty = float(icm_signal.get('uncertainty', 0.0))
        ctx_vals = torch.tensor(
            [step_progress, last_D, F_trend, forgetting_max, task_novelty,
             info_gain_ema, n_concepts_norm, icm_uncertainty],
            device=self.device, dtype=torch.float32,
        )
        return ctx_vals.unsqueeze(0).expand(batch_size, -1)

    # ── 单步训练 ──

    # ── 单步训练: 零反向传播, 纯 Hebbian 更新 ──

    def train_step(self, byte_seq: torch.Tensor, labels: torch.Tensor) -> dict:
        """执行 6 阶段 bp_free 训练: 零 autograd, 纯局部 Hebbian.

        Phase 0: 数据准备
        Phase 1: init_z (no_grad)
        Phase 2: PC 推理 (T 步, no_grad) → ε_by_layer
        Phase 3: 计算调制信号 (D, ACh, π, λ)
        Phase 4: Decoder 目标计算
        Phase 5: Hebbian 更新 (W.data.add_)
        Phase 6: 日志 + 诊断

        Returns:
            dict: 同 train_step 格式, 兼容后续持续学习管道
        """
        from model.pc.local_updates import (
            apply_hebbian_updates,
            compute_all_hebbian_updates,
            compute_lambda,
            compute_modulators,
            compute_precision_scales,
        )

        # ── Phase 0: 数据准备 ──
        seq_len = byte_seq.size(-1)
        bsz = byte_seq.size(0)
        pos_emb = self.model.get_position_embeddings(seq_len, self.device)

        # ── Phase 1: init_z (no_grad) ──
        with torch.no_grad():
            z_init = self.model.init_z(byte_seq)  # list[tensor, L+1]

        # ── temp_loss 诊断 (no_grad, clamp=100 防 9e9 日志) ──
        bp_temp_loss = 0.0
        bp_temp_by_layer = []
        if hasattr(self.model, 'temporal_proj') and seq_len > 1:
            n_layers = len(self.model.temporal_proj)
            tl_acc = torch.tensor(0.0, device=self.device)
            tl_list: list[torch.Tensor] = []
            for ℓ in range(n_layers):
                z_ℓ = z_init[ℓ + 1]
                if z_ℓ.size(1) > 1:
                    tl = 0.5 * (self.model.temporal_proj[ℓ](z_ℓ[:, :-1, :]) - z_ℓ[:, 1:, :]).pow(2).mean()
                    tl_clamped = tl.clamp(max=100.0)
                    tl_list.append(tl_clamped)
                    tl_acc = tl_acc + tl_clamped
            if n_layers > 0 and tl_list:
                bp_temp_loss = (tl_acc / n_layers).item()
                bp_temp_by_layer = [t.item() for t in tl_list]

        # ── Phase 2: PC 推理 (no_grad, T 步) ──
        # 从 F 历史算 uncertainty → ACh
        uncertainty = compute_uncertainty(self._global_F_hist, window=10)
        ACh = float(torch.sigmoid(torch.tensor(-uncertainty + self.cfg.hebbian_ach_beta_0)).item())

        with torch.no_grad():
            z_conv, errors_hist, F_hist, _, ε_list = self.model.spatiotemporal_infer(
                z_init, pos_emb,
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
                skip_bottom_up=True,  # Phase 5: 复用 init_z 的预测值, 跳过冗余 predict()
            )

        F_curr = F_hist[-1] if F_hist else 0.0

        # ── Phase 3: 调制信号 ──
        D, ACh_val, modulation = compute_modulators(
            F_curr, self._last_F_bp, uncertainty, self.cfg)
        λ = compute_lambda(self.global_step, self.cfg.hebbian_lambda_decay,
                           self.cfg.hebbian_lambda_min)

        π_list = compute_precision_scales(ε_list, ACh_val, D, self.cfg)

        # ── Phase 4: Decoder 目标 ──
        # 使用 onehot 编码的 next_byte
        if labels is not None and seq_len > 1:
            target_onehot = nn.functional.one_hot(
                labels[:, 1:].long().clamp(0, 255), num_classes=256).float()
        else:
            target_onehot = None

        # ── Phase 5: Hebbian 更新 (零 autograd) ──
        hebb_diag = {}
        with torch.no_grad():
            oja_alpha = getattr(self.cfg, 'oja_alpha', 0.05)
            syn_norm = getattr(self.cfg, 'synaptic_normalize', True)
            syn_target = getattr(self.cfg, 'synaptic_target_norm', 0.0)
            updates = compute_all_hebbian_updates(
                ε_list, z_init, byte_seq, self.model, self.cfg,
                D=D, ACh=ACh_val, modulation=modulation, λ=λ,
                decoder=self.model.decoder,
                lm_head=self.model.model.lm_head,
                target_byte_embed=target_onehot,
                oja_alpha=oja_alpha, bcm_state=self.bcm_state, verbose=True,
            )
            apply_hebbian_updates(updates, self.model,
                                  synaptic_normalize=syn_norm,
                                  target_norm=syn_target)
            # 提取 Hebbian 诊断 (供上层用 tqdm.write 输出)
            hebb_diag = {
                'avg_growth': updates.pop('_diag_avg_growth', None),
                'n_inf': updates.pop('_diag_n_inf', None),
                'n_params': updates.pop('_diag_n_params', None),
                'oja_alpha': updates.pop('_diag_oja_alpha', None),
            }

        # ── Phase 5.6: 神经发生 (Neurogenesis Controller) ──
        neuro_stats = {'n_pruned': 0, 'n_resurrected': 0, 'n_split': 0, 'active_ratio': 1.0}
        if (self.cfg.enable_neurogenesis and hasattr(self, 'neurogenesis')
                and self.neurogenesis is not None):
            neuro_stats = self.neurogenesis.step(
                model=self.model, ε_list=ε_list, global_step=self.global_step,
            )

        # ── Phase 5.5: Salience Gate 更新 (直接 SGD, 零 autograd) ──
        if self.cfg.enable_salience_gating and hasattr(self.model, 'salience_gates'):
            with torch.no_grad():
                β = self.cfg.salience_reg_weight
                η_gate = self.cfg.salience_gate_lr
                for gate in self.model.salience_gates:
                    # L_gate = β · Σ(1 - σ(logits))²
                    gate_sig = torch.sigmoid(gate.logits)
                    sparsity_loss = β * ((1.0 - gate_sig) ** 2).sum()
                    # 直接 SGD: ∇L = -2β · (1 - σ) · σ · (1 - σ) ... 
                    # 简化: dL/d_logit = -2β · (1 - σ) · σ · (1 - σ)
                    # 但 sigmoid 的导数是 σ·(1-σ), 所以:
                    # dL/d_logit = -2β · (1-σ) · σ · (1-σ) 
                    grad = -2.0 * β * (1.0 - gate_sig) * gate_sig * (1.0 - gate_sig)
                    gate.logits -= η_gate * grad

        # ── Phase 6: 日志 + 诊断 ──
        self._last_F_bp = F_curr
        self._global_F_hist.append(F_curr)

        # CE 诊断 (no_grad, 仅用于日志)
        with torch.no_grad():
            ce_diag = self.model.compute_ce_loss(z_conv, labels).item()
            # 解码器损失诊断 (z_L[:-1] → next_byte)
            if target_onehot is not None:
                z_L = z_conv[-1]
                z_dec = z_L[:, :-1, :].float() if z_L.size(1) > 1 else z_L.float()
                dec_pred = nn.functional.linear(z_dec, self.model.decoder.weight.float())
                dec_loss = nn.functional.mse_loss(dec_pred, target_onehot).item()
            else:
                dec_loss = 0.0

        # 世界模型 (no_grad 推理, 零 backward)
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
                world_loss_val = getattr(self, '_last_world_loss', None)
            # 世界模型不再 backward, 仅保留推理评估 surprise

        # ICM 内在动机 (no_grad 推理, 零 backward)
        icm_loss_val = 0.0
        if self.cfg.enable_intrinsic_motivation and self.icm is not None:
            if self.global_step % self.cfg.consolidation_pipeline_interval == 0:
                with torch.no_grad():
                    z_curr = z_conv[-1].detach()
                    z_prev = z_init[-1].detach()
                    action_embed = (z_curr - z_prev).mean(dim=1)
                    if action_embed.size(-1) > self.cfg.icm_action_dim:
                        action_embed = action_embed[:, :self.cfg.icm_action_dim]
                    elif action_embed.size(-1) < self.cfg.icm_action_dim:
                        pad = torch.zeros(bsz, self.cfg.icm_action_dim - action_embed.size(-1), device=self.device)
                        action_embed = torch.cat([action_embed, pad], dim=-1)
                    icm_output = self.icm.forward(z_prev, z_curr)
                    self._icm_output = {k: (v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v)
                                       for k, v in icm_output.items()}
                    icm_loss_val = (self.cfg.icm_forward_weight * self._icm_output.get('pred_loss', 0.0) +
                                   self.cfg.icm_inverse_weight * self._icm_output.get('inverse_loss', 0.0) +
                                   self.cfg.icm_contrastive_weight * self._icm_output.get('contrastive_loss', 0.0))
                # 概念发现 (仅观察, 零 backward)
                if self.concept_discovery is not None:
                    info_gain = self._icm_output.get('information_gain', 0.0)
                    self.concept_discovery.observe(z_curr[0:1].detach(), intrinsic_value=info_gain)
            else:
                self._icm_output = None
                icm_loss_val = 0.0

        # ── 持续巩固管道 ──
        if self.consolidation_pipeline is not None and self.global_step % self.cfg.consolidation_pipeline_interval == 0:
            sample_z = [z[0:1].detach() for z in z_conv]
            sample_byte = byte_seq[0].detach()
            sample_label = labels[0].detach()
            sample_task = self._current_task_id
            sample_concept = ''
            if self.concept_discovery is not None and len(self.concept_discovery.concept_ids) > 0:
                sample_concept = self.concept_discovery.concept_ids[-1]
            self.consolidation_pipeline.observe(
                z_states=sample_z, byte_tensor=sample_byte, label_tensor=sample_label,
                task_id=sample_task, concept_id=sample_concept,
                information_gain=self._icm_output.get('information_gain', 0.0) if self._icm_output else 0.0,
                dopamine_score=D, step=self.global_step,
            )

        # ── 结果 ──
        self._last_world_mode = 'full'
        self._last_D = D
        lr_used = self.cfg.hebbian_base_eta * modulation

        # BPB 诊断
        L = self.model.num_sub_layers
        bpb_pred = (F_curr * L) / _LOG2
        bpb_total = (F_curr * L + max(ce_diag, 1e-8)) / _LOG2

        result = {
            'ce_val': ce_diag,
            'F_val': F_curr,
            'F_final': F_curr,
            'F_hist': F_hist,
            'errors_hist': errors_hist,
            'bpb': bpb_total,
            'bpb_pred': bpb_pred,
            'temp_loss_val': bp_temp_loss,
            'temp_by_layer': bp_temp_by_layer,
            'D': D,
            'lr': lr_used,
            'β_local': 0.0,
            'β_conv': 0.0,
            'scale_local': 1.0,
            'scale_conv': 1.0,
            'π': π_list,
            'phase': 'bp_free',
            'world_surprise': surprise,
            'world_loss': world_loss_val,
            'update_mode': 'full',
            'icm_loss': icm_loss_val,
            'ACh': ACh_val,
            'λ': λ,
            'uncertainty': uncertainty,
            'decoder_loss': dec_loss,
        }
        if self._icm_output:
            for k in ['pred_loss', 'inverse_loss', 'information_gain', 'uncertainty']:
                result[f'icm_{k}'] = self._icm_output.get(k, 0.0)
        result['error_ratio'] = getattr(self.model, '_last_error_ratio', 1.0)

        # 门控自组织统计
        if self.cfg.enable_salience_gating and hasattr(self.model, 'salience_gates'):
            gs = self.model.get_gate_stats()
            result['gate_active_ratio'] = gs['active_ratio']
            result['gate_n_active'] = gs['n_active']
            result['gate_n_total'] = gs['n_total']
        # 神经发生统计
        if self.cfg.enable_neurogenesis:
            result['neuro_n_pruned'] = neuro_stats.get('n_pruned', 0)
            result['neuro_n_resurrected'] = neuro_stats.get('n_resurrected', 0)
            result['neuro_n_split'] = neuro_stats.get('n_split', 0)
            result['neuro_active_ratio'] = neuro_stats.get('active_ratio', 1.0)

        # Hebbian 诊断 (由 compute_all_hebbian_updates 收集)
        if hebb_diag.get('avg_growth') is not None:
            result['hebb_diag'] = hebb_diag

        # ── Phase 3c: 海马体缓冲写入 ──
        info_gain = self._icm_output.get('information_gain', 0.0) if self._icm_output else 0.0
        if info_gain > self.hippocampus.min_info_gain:
            self.hippocampus.add(
                z_states=z_conv, byte_tensor=byte_seq[0].detach(),
                label_tensor=labels[0].detach(),
                info_gain=info_gain, step=self.global_step,
            )

        return result

    # ── Hebbian 辅助: 对任意数据运行 Hebbian 更新 ──

    def _hebbian_update_on_data(self, byte_seq: Optional[torch.Tensor] = None,
                                 labels: Optional[torch.Tensor] = None,
                                 z_init=None, stride: int = 1):
        """对 (byte_seq, labels) 或直接用预计算 z_init 执行一次纯 Hebbian 权重更新.

        Args:
            stride: 序列下采样步长 (>1 时沿时间维降采样, 减少计算量)
        """
        if z_init is not None:
            seq_len = z_init[0].size(1)
        elif byte_seq is not None:
            seq_len = byte_seq.size(-1)
        else:
            return

        # ── Stride 下采样: 沿时间维降采样以加速 ──
        if stride > 1 and byte_seq is not None:
            # byte_seq: [B, 2, S] → 沿 S 维下采样
            indices = torch.arange(0, seq_len, stride, device=byte_seq.device)
            byte_seq = byte_seq[:, :, indices]
            if labels is not None:
                labels = labels[:, indices]
            seq_len = byte_seq.size(-1)

        pos_emb = self.model.get_position_embeddings(seq_len, self.device)

        if z_init is None and byte_seq is not None:
            z_init, _ = self.model.forward_with_ce(byte_seq, labels, pos_emb)

        if byte_seq is None:
            byte_seq = torch.zeros(1, 2, seq_len, device=self.device, dtype=torch.long)

        uncertainty = compute_uncertainty(self._global_F_hist, window=10)
        ACh = float(torch.sigmoid(torch.tensor(-uncertainty + self.cfg.hebbian_ach_beta_0)).item())
        with torch.no_grad():
            z_conv, errors_hist, _, _, ε_list = self.model.spatiotemporal_infer(
                z_init, pos_emb, gamma=self.cfg.gamma, T=self.cfg.hebbian_infer_T,
                return_errors=True, return_pred_loss=False, ach_value=ACh, return_ε=True,
            )
        F_curr = errors_hist[-1][0][1] if errors_hist and errors_hist[-1] else 0.0
        D, ACh_val, modulation = compute_modulators(F_curr, self._last_F_bp, uncertainty, self.cfg)
        λ = compute_lambda(self.global_step, self.cfg.hebbian_lambda_decay, self.cfg.hebbian_lambda_min)
        π_list = compute_precision_scales(ε_list, ACh_val, D, self.cfg)
        with torch.no_grad():
            oja_alpha = getattr(self.cfg, 'oja_alpha', 0.05)
            syn_norm = getattr(self.cfg, 'synaptic_normalize', True)
            syn_target = getattr(self.cfg, 'synaptic_target_norm', 0.0)
            updates = compute_all_hebbian_updates(
                ε_list, z_init, byte_seq, self.model, self.cfg,
                D=D, ACh=ACh_val, modulation=modulation, λ=λ,
                decoder=self.model.decoder,
                lm_head=self.model.model.lm_head,
                target_byte_embed=None,
                oja_alpha=oja_alpha, bcm_state=self.bcm_state, verbose=False,
            )
            apply_hebbian_updates(updates, self.model,
                                  synaptic_normalize=syn_norm,
                                  target_norm=syn_target)

    # ── 持续学习: 记忆回放 ──

    def _maybe_replay(self):
        if self.memory_bank.total <= 0:
            return

        # ── 海马体快速回放 (Phase 3c) ──
        # 以 replay_ratio * 2 的间隔从 hippocampus 加权采样回放
        if (self.hippocampus.size > 0
                and self.global_step % (self.cfg.replay_ratio * 2) == 0
                and not (self.sniffer.is_repairing
                         if hasattr(self.sniffer, 'is_repairing') else False)):
            hc_batch = self.hippocampus.sample_for_replay(
                self.cfg.batch_size // 4, device=self.device)
            if hc_batch is not None:
                replay_byte_hc, replay_label_hc = hc_batch
                # 将 byte: [B, S] → [B, 2, S] 格式 (DualChannelDataset)
                replay_byte_hc = torch.stack([
                    replay_byte_hc.float(),
                    torch.full_like(replay_byte_hc, 2.0, dtype=torch.float,
                                    device=self.device),
                ], dim=1)
                self._hebbian_update_on_data(replay_byte_hc, replay_label_hc,
                                             stride=self.cfg.replay_stride)

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

        # Hebbian 回放: 正向 + 推理 → 局部 Hebbian 更新 (带 stride 加速)
        self._hebbian_update_on_data(replay_byte, replay_label,
                                     stride=self.cfg.replay_stride)

        # 刷新被回放样本的 transition_surprise 和 replay_priority
        if self.cfg.enable_world_model and self.world_model is not None:
            with torch.no_grad():
                pos_emb = self.model.get_position_embeddings(replay_byte.size(-1), self.device)
                z_rp, _ = self.model.forward_with_ce(replay_byte, replay_label, pos_emb)
                z_top = z_rp[-1].detach()
                ctx = self._build_world_model_context(replay_byte.size(0))
                _, uncertainty = self.world_model(z_top, ctx)
                new_surprise = uncertainty.mean().item()
            for ex in replay_ex:
                ex.transition_surprise = new_surprise
                ex.replay_priority = max(ex.dopamine_score, 0.1) + max(new_surprise, 0.0) + ex.intrinsic_value
        elif self.cfg.enable_intrinsic_motivation and self.icm is not None:
            if self._icm_output:
                info_gain = self._icm_output.get('information_gain', 0.0)
                for ex in replay_ex:
                    ex.replay_priority = max(ex.dopamine_score, 0.1) + info_gain

    def _maybe_abstraction_replay(self):
        gs = self.global_step
        if gs % self.cfg.abstraction_replay_interval != 0:
            return

        # 从 AbstractionBank 获取回放数据并执行 Hebbian 更新
        replay_data = self.abstraction_bank.sample_replay_batch(
            batch_size=16, device=self.device)
        if replay_data is not None:
            z_batch, _seqlen = replay_data
            self._hebbian_update_on_data(z_init=z_batch)
            self._log(f'[AbstractionBank] Hebbian replay step {gs}')

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

        self._log(f'[Sniffer] FORGOTTEN: {forgotten} — Hebbian repair')

        # 世界模型 surprise 调制修复强度
        wm_factor = 1.0
        strategy = 'dopamine'
        if self.cfg.enable_world_model and self.world_model is not None:
            wm_surprise = self._last_world_surprise
            if wm_surprise > self.cfg.world_model_surprise_threshold:
                wm_factor = 1.0 + min(wm_surprise, 1.0)
                strategy = 'world_model'
            else:
                wm_factor = max(0.5, 1.0 - wm_surprise * 2)

        effective_steps = max(1, int(self.cfg.repair_steps * wm_factor))
        self._log(f'[Sniffer]  wm_surprise={self._last_world_surprise:.3f} '
                  f'factor={wm_factor:.2f} steps={effective_steps} strategy={strategy}')

        for _ in range(effective_steps):
            replay_data = self.sniffer.get_replay_batch(
                self.cfg.batch_size, self.device, strategy=strategy)
            if replay_data is None:
                break
            rp_byte, rp_label = replay_data
            self._hebbian_update_on_data(rp_byte, rp_label)

        self._log('[Sniffer] Hebbian repair complete')

    def _maybe_sniff_abstraction(self):
        drifted = self.abstraction_sniffer.check(self.global_step, self.device, pos_emb=(None, None))
        if not drifted:
            return

        self._log(f'[AbstractionSniffer] DRIFT detected: {drifted} — Hebbian repair')

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
            replay_data = self.abstraction_bank.sample_replay_batch(
                batch_size=16, device=self.device)
            if replay_data is None:
                break
            z_batch, _seqlen = replay_data
            self._hebbian_update_on_data(z_init=z_batch)

        self._log('[AbstractionSniffer] Hebbian repair complete')

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
                # sample middle position as representative
                seq_len = z_top.size(1)
                mid_idx = seq_len // 2
                z_rep = z_top[:, mid_idx:mid_idx+1, :].to(device)  # [1, 1, hidden]
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
        """全部任务后: 深度 SLEEP — 吸引子景观维护 (使用 SleepEngine)。"""
        if not self.cfg.enable_deep_sleep or self.sleep_engine is None:
            # fallback: 旧版 WM filter replay
            self._log('[Sleep] Deep sleep disabled, fallback to standard replay')
            self._standard_sleep_phase(task_pipelines)
            return

        self._log('[Sleep/Deep] 开始深度睡眠 — 吸引子景观维护...')
        self.model.train()

        # 检查 landscape 状态
        if self.landscape is not None:
            report = self.landscape.full_landscape_report(
                self.model, self.abstraction_bank,
                pos_emb=(None, None), gamma=self.cfg.gamma,
                device=self.device,
            )
            self._log(f'[Sleep] 景观报告: {report["n_prototypes_total"]} prototypes, '
                      f'entropy={report["entropy_metrics"]["normalized_entropy"]:.3f}, '
                      f'collapse_ratio={report["collapse_ratio"]:.3f}')

        # 检查 bank 是否为空
        if self.memory_bank.total < 4:
            self._log('[Sleep] MemoryBank 样本不足, 跳过深度睡眠')
            return

        # 执行 3 阶段深度睡眠
        phases = ['completion', 'noise', 'competitive']
        results = self.sleep_engine.run(
            model=self.model,
            memory_bank=self.memory_bank,
            abstraction_bank=self.abstraction_bank,
            device=self.device,
            phases=phases,
        )

        for phase_name, avg_loss in results.items():
            self._log(f'[Sleep] Phase {phase_name}: avg_loss={avg_loss:.4f}')

        # 睡眠后更新 AbstractionBank
        for gk in list(self.abstraction_bank._store.keys()):
            self.abstraction_bank.consolidate(gk)
        for gk in list(self.abstraction_bank._store_by_concept.keys()):
            self.abstraction_bank.consolidate(gk)

        self._log('[Sleep/Deep] 深度睡眠完成')

    def _standard_sleep_phase(self, task_pipelines: list):
        """标准 SLEEP 回退: WM-filtered OfflineReplayer replay (原 _full_sleep_phase 逻辑)。"""
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

        # 获取 OfflineReplayer
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
            'lm_config': CyreneConfig(
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
        if metrics:
            ckpt.update(metrics)
        ckpt['memory_bank'] = self.memory_bank.state_dict()
        ckpt['abstraction_bank'] = self.abstraction_bank.state_dict()
        if self.sleep_engine is not None:
            ckpt['sleep_engine'] = self.sleep_engine.state_dict()
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
        """主训练入口。

        Args:
            task_pipelines: [(task_id, dataset, loader_override?), ...]
                            loader_override 可选, 默认自动创建 DataLoader
        """
        out_dir = os.path.join(os.getcwd(), self.cfg.out_dir)
        os.makedirs(out_dir, exist_ok=True)

        # 创建模型
        self.model = self._build_model().to(self.device)
        budget = count_budget(self.model)
        # 保存初始权重快照用于 NaN 恢复 (CPU, 仅保存一次)
        self._fallback_state = {
            k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()
        }
        self._log(f'NaN fallback snapshot saved ({len(self._fallback_state)} tensors on CPU)')
        self._log(f'Model: capacity budget {budget["trainable_M"]:.2f}M — effective capacity evolves during training')

        # ── 神经发生控制器 (结构自组织: 剪枝+生长) ──
        if self.cfg.enable_neurogenesis:
            self.neurogenesis = NeurogenesisController(
                hidden_size=self.cfg.hidden_size,
                prune_interval=self.cfg.neurogenesis_prune_interval,
                grow_interval=self.cfg.neurogenesis_grow_interval,
                prune_threshold_act=self.cfg.neurogenesis_prune_threshold_act,
                prune_threshold_gate=self.cfg.neurogenesis_prune_threshold_gate,
                grow_error_threshold=self.cfg.neurogenesis_grow_error_threshold,
                max_grow_per_step=self.cfg.neurogenesis_max_grow_per_step,
            )
            self._log('Neurogenesis controller enabled (prune+grow)')
        else:
            self.neurogenesis = None

        # 世界模型必须在 checkpoint 加载之前初始化，否则 world_model_state 无处加载
        # ── 世界模型 (推理评估用, 零 backward) ──
        if self.cfg.enable_world_model:
            self.world_model = LatentWorldModel(
                input_dim=self.cfg.hidden_size,
                hidden_dim=self.cfg.world_model_hidden_dim,
                context_dim=self.cfg.world_model_context_dim,
            ).to(self.device)
            # 零 backward: world_model 不再 Adam 训练
            self.world_model_optimizer = None
            self._log('Latent world model enabled (inference-only)')

        # ── 内在动机模块 (推理评估用, 零 backward) ──
        if self.cfg.enable_intrinsic_motivation:
            self.icm = IntrinsicCuriosityModule(
                input_dim=self.cfg.hidden_size,
                action_embed_dim=self.cfg.icm_action_dim,
                hidden_dim=self.cfg.icm_hidden_dim,
            ).to(self.device)
            # 零 backward: ICM 不再 Adam 训练
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
            self._log('Intrinsic motivation enabled (ICM + Concept + Gate)')

        # ── 吸引子景观 + 持续巩固管道 (Phase A-C) ──
        if self.cfg.enable_consolidation_pipeline:
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
            self._log('Consolidation pipeline enabled')

        # ── 深度 SLEEP 引擎 (Phase F) ──
        if self.cfg.enable_deep_sleep:
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
            self._log('Deep sleep engine enabled')

        # 从 checkpoint 恢复
        if self.cfg.checkpoint_path:
            ckpt = torch.load(self.cfg.checkpoint_path, map_location=self.device, weights_only=False)
            if 'model_state' in ckpt:
                self.model.load_state_dict(ckpt['model_state'])
                if self.cfg.enable_world_model and self.world_model is not None and 'world_model_state' in ckpt:
                    self.world_model.load_state_dict(ckpt['world_model_state'])
            else:
                # 纯 state_dict (如 unified_final.pt)
                self.model.load_state_dict(ckpt)
            self._log(f'Resumed from checkpoint: {self.cfg.checkpoint_path}')

        self.dopamine = DopamineSignal(η=self.cfg.dopamine_eta, threshold=0.0)

        # Sniffer 绑定 model
        self.sniffer.model = self.model
        self.abstraction_sniffer.model = self.model
        self.abstraction_sniffer.world_model = self.world_model

        # 预热
        self.warmup()

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
                                    shuffle=True, num_workers=4, pin_memory=True,
                                    persistent_workers=True)

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
                            unit='step', dynamic_ncols=True,
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')

                for byte_seq, labels in pbar:
                    byte_seq = byte_seq.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    # ── 纯局部学习模式 (零反向传播) ──
                    m = self.train_step(byte_seq, labels)

                    # ── 跳过因 NaN 跳过的步 ──
                    if m.get('skipped', False):
                        if 'world_surprise' not in m:
                            m['world_surprise'] = 0.0
                        continue

                    # ── 步数递增 (修复: 原来缺失) ──
                    self.global_step += 1

                    # ── Hebbian 诊断输出 (每 50 步, 不破坏进度条) ──
                    hebb = m.get('hebb_diag')
                    if hebb is not None and self.global_step % 50 == 0:
                        pbar.write(
                            f'  [Hebb] oja_α={hebb["oja_alpha"]:.4f} | '
                            f'mean|ΔW|={hebb["avg_growth"]:.6f} | '
                            f'updates={hebb["n_params"]}'
                            + (f' | ⚠ {hebb["n_inf"]} inf跳过' if hebb['n_inf'] > 0 else '')
                        )

                    # ── ICM / Concept / Gate 后处理 ──
                    if self.cfg.enable_intrinsic_motivation and self.icm is not None:
                        # 概念 consolidation (每 500 步)
                        if self.global_step % 500 == 0 and self.concept_discovery is not None:
                            self.concept_discovery.consolidate()
                        # 记忆门控自适应 (每 consolidation_pipeline_interval 步 Phase 7b)
                        if self.memory_gate is not None and self.global_step % self.cfg.consolidation_pipeline_interval == 0:
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
                    # 持续巩固管道: 调度 (Phase B)  — Phase 7b: interval 降频
                    if self.consolidation_pipeline is not None and self.global_step % self.cfg.consolidation_pipeline_interval == 0:
                        current_D = m.get('D', 0.0)
                        # nan 保护: 替换 nan 为 0.0 (保持中立)
                        if isinstance(current_D, float) and (current_D != current_D):
                            current_D = 0.0

                        # 累积多巴胺窗口 → 高 D 持续 → 强制巩固
                        self._dopamine_window = getattr(self, '_dopamine_window', [])
                        self._dopamine_window.append(current_D)
                        if len(self._dopamine_window) > 30:
                            self._dopamine_window.pop(0)

                        tick_result = {'triggered': None}
                        try:
                            tick_result = self.consolidation_pipeline.tick(
                                self.global_step, self.model,
                                self.memory_bank, self.abstraction_bank,
                                device=self.device,
                                dopamine_score=current_D,
                            )
                        except Exception as pipe_err:
                            self._log(f'[Pipeline] tick 忽略异常: {pipe_err}')
                        if tick_result.get('triggered') and self.global_step % 500 == 0:
                            self._log(f'[Pipeline] {tick_result["triggered"]}')

                        # 高 D 稳定状态 → 强制巩固
                        mean_D = sum(self._dopamine_window) / len(self._dopamine_window)
                        if mean_D > 0.70 and len(self._dopamine_window) >= 20:
                            try:
                                force_result = self.consolidation_pipeline.force_consolidate(
                                    self.global_step, self.model,
                                    self.memory_bank, self.abstraction_bank,
                                    device=self.device,
                                )
                                self._dopamine_window.clear()
                            except Exception as pipe_err:
                                self._log(f'[Pipeline] force_consolidate 忽略异常: {pipe_err}')
                            if force_result['triggered'] and self.global_step % 500 == 0:
                                self._log(f'[Pipeline] force:{force_result["triggered"]}')

                    # 进度条 (精简 postfix: 只留核心指标)
                    postfix = {
                        'CE': f'{m["ce_val"]:.4f}',
                        'F': f'{m["F_final"]:.1f}',
                        'D': f'{m["D"]:.3f}',
                        'W': f'{m.get("world_surprise", 0.0):.3f}',
                    }
                    if self.cfg.enable_intrinsic_motivation and self._icm_output:
                        postfix['IG'] = f'{self._icm_output.get("information_gain", 0.0):.4f}'
                        if self.concept_discovery:
                            postfix['C'] = f'{self.concept_discovery.n_concepts}'
                    pbar.set_postfix(**postfix)
                    callback_dict = {
                        'type': 'progress', 'step': self.global_step,
                        'total_steps': self._total_steps,
                        'ce_loss': m["ce_val"],
                        'F': m["F_final"],
                        'temp_loss': m.get('temp_loss_val', 0.0),
                        'BPB': m.get('bpb', 0.0),
                        'bpb_pred': m.get('bpb_pred', 0.0),
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
                            'n_concepts': self.concept_discovery.n_concepts if self.concept_discovery else 0,
                        })
                    if self.cfg.progress_callback:
                        self.cfg.progress_callback(callback_dict)

                    # 日志 (每 100 步)
                    if self.global_step % 100 == 0 and self.global_step > 0:
                        log = (f'[Step {self.global_step}/{self._total_steps}] '
                               f'F={m["F_final"]:.1f} CE={m["ce_val"]:.4f} '
                               f'TL={m.get("temp_loss_val", 0.0):.4f} '
                               f'BPB={m.get("bpb", 0.0):.2f} '
                               f'D={m["D"]:.3f} lr={m["lr"]:.2e} '
                               f'W={m.get("world_surprise", 0.0):.3f}')
                        if self.cfg.enable_intrinsic_motivation and self._icm_output:
                            log += (f' IG={self._icm_output.get("information_gain", 0.0):.4f}'
                                    f' ICML={m.get("icm_loss", 0.0):.4f}'
                                    f' C={self.concept_discovery.n_concepts if self.concept_discovery else 0}')
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
                        # CLI: 通过 tqdm.write 不破坏进度条; GUI: 走回调
                        if self.cfg.progress_callback:
                            self._log(log)
                        else:
                            pbar.write(log)

                    # 检查点 (save_interval=0 时不保存中间检查点)
                    if self.cfg.save_interval > 0 and (self.global_step % self.cfg.save_interval == 0 or self.global_step == 1):
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
                n_concepts = self.concept_discovery.n_concepts
                fragile = len(self.concept_discovery.get_fragile_concept_ids())
                self._log(f'[Intrinsic] After {task_id}: {n_concepts} concepts ({fragile} fragile)')
                # 打印概念摘要
                for cid, c in self.concept_discovery.alive_concepts[:10]:
                    self._log(f'  Concept {cid}: support={c.support}, '
                              f'avg_IG={c.avg_intrinsic_value:.4f}')
                # 内在动机统计
                ig_mean = (sum(self._intrinsic_stats['information_gain'][-500:]) /
                           max(len(self._intrinsic_stats['information_gain'][-500:]), 1))
                self._log(f'[Intrinsic Stats] IG_mean_500={ig_mean:.4f}, '
                          f'n_concepts_peak={max(self._intrinsic_stats.get("n_concepts") or [0])}')
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

        # ── 保存 unified_final.pt ──
        self.model.cpu()
        fp = os.path.join(out_dir, 'unified_final.pt')
        torch.save(self.model.state_dict(), fp)
        self._log(f'unified_final saved → {fp} ({os.path.getsize(fp) // 1024 // 1024}MB)')


        self._log('Training complete.')
        if self.cfg.progress_callback:
            self.cfg.progress_callback({'type': 'done', 'message': 'Training complete'})
