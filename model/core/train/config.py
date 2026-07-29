"""训练配置
TrainingConfig + ProgressCallback.
"""

from dataclasses import dataclass
from typing import Callable, Optional

ProgressCallback = Callable[[dict], None]
"""训练进度回调.
    {
        'type': 'log' | 'progress' | 'phase' | 'checkpoint'
        | 'task_done' | 'done' | 'error',
        'message': str, 'step': int, 'total_steps': int,
        'ce_loss': float, 'F': float, 'D': float, 'lr': float,
        'task_id': str, 'checkpoint_path': str,
    }
"""


@dataclass
class TrainingConfig:
    """统一训练配置."""

    # 模型
    hidden_size: int = 256
    num_hidden_layers: int = 4
    use_moe: bool = False
    vocab_size: int = 256  # 字节级默认 256; 词元级可覆盖
    checkpoint_path: Optional[str] = None  # 从 checkpoint 恢复

    # 训练
    batch_size: int = 48
    max_seq_len: int = 128
    lr: float = 3e-4
    epochs: int = 1
    subset: int = 0  # 0 = 全量
    seed: int = 42
    split_size: int = 0  # 0 = 不分块, >0 = 每块样本数 (GUI 传入)

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
    hebbian_base_eta: float = 3e-4
    hebbian_lambda_decay: int = 5000
    hebbian_lambda_min: float = 0.01
    hebbian_infer_T: int = 3
    hebbian_ach_beta_0: float = 0.0

    # Oja 规则
    oja_alpha: float = 0.05
    oja_eta: float = 0.05
    oja_adaptive: bool = True

    # 误差归一化
    ε_rms_target: float = 1.0

    # 位置子采样 (每步随机抽 1/6 的位置做 Hebbian 更新)
    hebbian_subsample_ratio: float = 0.1667

    # 突触归一化
    synaptic_normalize: bool = False
    synaptic_target_norm: float = 0.0

    # Salience Gating
    enable_salience_gating: bool = True
    salience_temperature: float = 0.1
    salience_reg_weight: float = 0.001
    salience_gate_lr: float = 1e-3

    # 稳态 (生长/修剪)
    prune_interval: int = 100
    grow_interval: int = 200
    homeostasis_interval: int = 50
    connection_density: float = 0.2
    bias_strength: float = 0.7

    # Phase 7b: 降频
    consolidation_pipeline_interval: int = 5

    # 持续学习
    replay_ratio: int = 5
    bank_size: int = 2000
    sniff_interval: int = 200
    repair_threshold: float = 1.2
    repair_steps: int = 10
    eval_samples: int = 100

    # AbstractionBank
    n_prototypes: int = 8
    abstraction_replay_interval: int = 200
    abstraction_sniff_interval: int = 300
    abstraction_drift_threshold: float = 0.7

    # 世界模型
    enable_world_model: bool = True
    world_model_hidden_dim: int = 128
    world_model_context_dim: int = 5
    world_model_surprise_threshold: float = 0.25
    world_model_loss_weight: float = 0.1

    # 内在动机
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
    out_dir: str = "out"
    save_interval: int = 10000

    # 回调
    progress_callback: Optional[ProgressCallback] = None

    # Phase 1: 依赖阈值发放
    act_threshold_init: float = 0.3
    act_target_ratio: float = 0.20
    act_ema_decay: float = 0.999
    act_homeo_rate: float = 0.02
    act_energy_cost: float = 0.5

    # Phase 2: 突触竞争
    synaptic_competition_k: int = 8
    synaptic_competition_use_abs: bool = False

    # Phase 4: 稀疏外积
    sparse_outer_k: int = 32
    hebbian_eps_gate: float = 0.0

    # 推理控制
    infer_adaptive_T: bool = True
    infer_convergence_threshold: float = 0.05
    infer_patience: int = 2
    infer_min_T: int = 1

    # Stride 回放
    replay_stride: int = 4

    # Sleep / Consolidation
    sleep_consolidation: bool = True
    full_sleep_after_all: bool = True
    sleep_replay_tasks: int = 2
    sleep_replay_samples: int = 100

    # 吸引子景观 + 持续巩固管道
    enable_consolidation_pipeline: bool = True
    pipeline_buffer_capacity: int = 500
    pipeline_memory_write_interval: int = 50
    pipeline_abstraction_write_interval: int = 200
    pipeline_sleep_check_interval: int = 500
    pipeline_min_info_gain: float = 0.05

    # 深度 SLEEP
    enable_deep_sleep: bool = True
    sleep_completion_steps: int = 20
    sleep_noise_steps: int = 20
    sleep_competitive_steps: int = 10

    def to_dict(self) -> dict:
        """将配置序列化为字典 (排除 progress_callback)."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and k != "progress_callback"
        }
