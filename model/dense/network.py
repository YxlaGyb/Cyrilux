"""DensePCNet

密集 PPA 闭环网络主类 (门面).

Perception-Prediction-Action 闭环, 全 fp16, 零反向传播, 纯 matmul, 无位置编码:
- 感知: L0(纯 one-hot) → L4 → L2 → L3 → L5 → L6 (自下而上)
- 时序: 每层学习的时间核 W_t 递归 z[t] 依赖 z[t-1], 时序轨迹在隐空间自然分化
- 生成: 时空差分共振 (L5_t → ΔL4_t), 未来相对现在的变化
- 精度: π_l = 1/(σ_εl + c); 高确定性(低惊喜)→ACh 记忆巩固, 低确定性→多巴胺大重构
- 学习: Hebbian dW = π·ε ⊗ z_pre, 零 autograd
- 监控: 自由能 free_energy = Σ_l ½·π_l·‖ε_l‖²  (不监控 PPL/Top-1)

职责边界 (门面模式, 参照 model_cyrene):
- 本文件: 配置 + 权重声明 + 序列化 + 门面方法 (一行委托)
- forward.py: ForwardEngine — 前馈/推理/生成/稀疏绑定
- learning.py: LearningEngine — Hebbian 学习
- pruning.py: PruningEngine — 动态神经元修剪
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DensePCConfig:
    """PPA 网络配置."""

    d_input: int = 256  # 输入字节维度 (固定 vocab_size, 无 PE)
    d_l4: int = 1024
    d_l2: int = 384
    d_l3: int = 384
    d_l5: int = 256
    d_l6: int = 128
    max_seq_len: int = 256
    # 时序双通道输入: W_04 输入 = [z0[t], z0[t-1]] 拼接 (词序信息进入表示层).
    # 单帧重建约束是词序盲区的根因 (dog/cat 互换 z5 恒同); 双通道纯线性路由, 合规
    input_history: bool = True

    # Hebbian 物理有效学习率 (1:1 映射到内部更新公式，无隐藏缩放)
    lr_hebbian: float = 0.003
    temporal_lr_ratio: float = 5.0
    oja_alpha: float = 0.05
    column_dropout: float = 0.25
    # 时间惯性 alpha (per-layer, per-neuron)
    inertia_alpha: float = 0.3

    # 修剪参数
    prune_interval: int = 1000
    prune_warmup: int = 5000
    prune_fraction: float = 0.05
    death_probation: int = 200
    death_threshold: float = 1e-4
    active_size_lower_bound: int = 128
    # L4 专属修剪下限 (预测主空间保底): L4 承载"表示+预测"双任务,
    # 维度坍缩 <600 时信息挤压成噪声 → NaN (20000 步实测); 512 是物理屏障
    l4_lower_bound: int = 512
    # 表示层冻结 + LM 头强化 (情况 A: 探测证明 z4 含字节信息后启用):
    # 表示层 lr=0, W_lm lr×lm_lr_boost, BCM 滑阈不冻结 (防漂移)
    freeze_backbone: bool = False
    lm_lr_boost: float = 1.0
    oja_elasticity: float = 0.05
    probation_decay: float = 0.5

    # 时序预测连接 (时空差分共振) 配置
    gen_precision: float = 1.0  # 差分共振强度
    # 微柱阵列: L5 拆成 n 个独立列块, 块间不共享 Hebbian/Oja
    l5_blocks: int = 4
    # 稀疏绑定层: L5 之上哈希式稀疏绑定 (k-WTA), 离散符元的种子
    bind_dim: int = 4096
    bind_k: int = 10
    # 实验参数: 绑定模式 (none/hard/soft) + 正交化开关
    bind_mode: str = "hard"
    bind_orth: bool = False

    def dims(self) -> dict[str, int]:
        return {
            "l4": self.d_l4,
            "l2": self.d_l2,
            "l3": self.d_l3,
            "l5": self.d_l5,
            "l6": self.d_l6,
        }

    def param_count(self) -> int:
        """预估参数量 (含生成连接与时间核)."""
        d = self.dims()
        n = 0
        # 前馈
        n += d["l4"] * self.d_input  # W_04
        n += d["l2"] * d["l4"]  # W_42
        n += d["l3"] * d["l2"]  # W_23
        n += d["l5"] * d["l3"]  # W_35
        n += d["l6"] * d["l5"]  # W_56
        # 时序预测连接
        n += d["l4"] * d["l5"]  # W_diff (L5_t → ΔL4_t)
        # 时间核 (方阵)
        for k in ("l4", "l2", "l3", "l5", "l6"):
            n += d[k] * d[k]
        # 偏置
        n += sum(d[k] for k in ("l4", "l2", "l3", "l5", "l6"))
        return n


class DensePCNet(nn.Module):
    """PPA 闭环网络 (门面: 权重 + 引擎委托).

    Args:
        config: 网络配置 (或 None 使用默认).
    """

    def __init__(self, config: DensePCConfig | None = None, max_seq_len: int = 256):
        super().__init__()
        self.cfg = config or DensePCConfig()
        d = self.cfg.dims()

        # ── 前馈权重 (自下而上感知) ──
        # W_04 输入维: 单帧 256 或双通道 512 (input_history=True 时拼接 z0[t-1])
        self._in_dim = self.cfg.d_input * (2 if self.cfg.input_history else 1)
        self.W_04 = nn.Parameter(torch.empty(d["l4"], self._in_dim, dtype=torch.float16))
        self.W_42 = nn.Parameter(torch.empty(d["l2"], d["l4"], dtype=torch.float16))
        self.W_23 = nn.Parameter(torch.empty(d["l3"], d["l2"], dtype=torch.float16))
        # ── 微柱阵列: L5 拆 4 独立列块, 块间不共享 Hebbian/Oja ──
        self.n_blocks = self.cfg.l5_blocks
        self.b5 = d["l5"] // self.n_blocks  # 每块维度
        self.W_35 = nn.ParameterList(
            [nn.Parameter(torch.empty(self.b5, d["l3"], dtype=torch.float16)) for _ in range(self.n_blocks)]
        )
        self.W_56 = nn.Parameter(torch.empty(d["l6"], d["l5"], dtype=torch.float16))

        # ── 世界模型 (下一状态预测): W_diff 在 L4 空间预测 Δz4 = z4[t] - z4[t-1] ──
        self.W_diff = nn.Parameter(torch.empty(d["l4"], d["l4"], dtype=torch.float16))
        self.b_diff = nn.Parameter(torch.zeros(d["l4"], dtype=torch.float16))
        # ── 状态预测矩阵 (预测编码融合): 独立于 W_diff, 专门给表示层提供预测误差 ──
        # eps_state = (z4[t] @ W_state_pred) - (z4[t+1] - z4[t]); 表示层最终误差
        # final_eps = eps_recon + 0.3 * eps_state — 迫使隐状态携带"未来往哪走"
        self.W_state_pred = nn.Parameter(torch.empty(d["l4"], d["l4"], dtype=torch.float16))

        # ── LM 头 (自监督赫布): z4 → 256 字节 logits, 独立于重建 W_04 ──
        # W_04 双向重建被证实解码死锁 (真实 delta 注入仍复读空格);
        # W_lm 唯一任务: 把状态映射到下一字节, dW_lm = z4^T @ (target - logits) 纯外积
        self.W_lm = nn.Parameter(torch.empty(d["l4"], self.cfg.d_input, dtype=torch.float16))
        self.bias_lm = nn.Parameter(torch.zeros(self.cfg.d_input, dtype=torch.float16))

        # ── 稀疏绑定层 (海马体式): z5 → W_bind → 4096 维, top-k WTA 硬稀疏 ──
        # 连续 L5 激活经高维竞争坍缩为 k 个"离散符元" (纯赫布, 只更新激活行);
        # 底层 L5 连续系统兜住信息流, 绑定层出问题不影响底层安全
        self.W_bind = nn.Parameter(torch.empty(d["l5"], self.cfg.bind_dim, dtype=torch.float16))

        # ── 多尺度软加权时间窗 (2/4/8 并行因果卷积) ──
        # 软权重按各尺度 EMA 误差自适应; 4 步时间窗环形缓冲保留误差记忆
        self.register_buffer("_w_soft", torch.tensor([0.1, 0.8, 0.1], dtype=torch.float16))
        self.register_buffer("_e_ema_2", torch.tensor(0.05, dtype=torch.float16))
        self.register_buffer("_e_ema_4", torch.tensor(0.05, dtype=torch.float16))
        self.register_buffer("_e_ema_8", torch.tensor(0.05, dtype=torch.float16))
        for i in range(4):
            self.register_buffer(f"_dw_buf_{i}", torch.zeros(d["l4"], d["l4"], dtype=torch.float16))
        self._buf_i = 0
        # W_diff 独立 BCM 滑阈 (防指数爆炸)
        self.register_buffer("_theta_w", torch.full((d["l4"],), 0.01, dtype=torch.float16))

        # ── 时序权重 (每层, Hebbian 学习, 非超参数) ──
        self.W_t4 = nn.Parameter(torch.empty(d["l4"], d["l4"], dtype=torch.float16))
        self.W_t2 = nn.Parameter(torch.empty(d["l2"], d["l2"], dtype=torch.float16))
        self.W_t3 = nn.Parameter(torch.empty(d["l3"], d["l3"], dtype=torch.float16))
        self.W_t5 = nn.Parameter(torch.empty(d["l5"], d["l5"], dtype=torch.float16))
        self.W_t6 = nn.Parameter(torch.empty(d["l6"], d["l6"], dtype=torch.float16))

        # ── 层偏置 ──
        self.bias_l4 = nn.Parameter(torch.zeros(d["l4"], dtype=torch.float16))
        self.bias_l2 = nn.Parameter(torch.zeros(d["l2"], dtype=torch.float16))
        self.bias_l3 = nn.Parameter(torch.zeros(d["l3"], dtype=torch.float16))
        self.bias_l5 = nn.Parameter(torch.zeros(d["l5"], dtype=torch.float16))
        self.bias_l6 = nn.Parameter(torch.zeros(d["l6"], dtype=torch.float16))

        # ── 动态生长状态 ──
        self.active_size = {"l4": d["l4"], "l2": d["l2"], "l3": d["l3"], "l5": d["l5"], "l6": d["l6"]}
        self._step_counter = 0
        self._death_row: dict[str, torch.Tensor | None] = {
            "l4": None,
            "l2": None,
            "l3": None,
            "l5": None,
            "l6": None,
        }
        self._probation_counter: dict[str, torch.Tensor | None] = {
            "l4": None,
            "l2": None,
            "l3": None,
            "l5": None,
            "l6": None,
        }

        # ── 神经调制与竞争机制 ──
        self.register_buffer("_surprise_buf", torch.tensor(1.0, dtype=torch.float16))  # 惊喜基线
        # BCM 滑阈 (替代 Oja): theta = EMA(eps²), phi = eps(eps-theta)
        for ln, dim in (("l4", d["l4"]), ("l2", d["l2"]), ("l3", d["l3"]), ("l5", d["l5"]), ("l6", d["l6"])):
            self.register_buffer(f"_theta_{ln}", torch.full((dim,), 0.01, dtype=torch.float16))
        self.register_buffer("_theta_diff", torch.full((d["l4"],), 0.01, dtype=torch.float16))
        # Foldiak 反赫布侧抑制 (L5 去相关): M 协方差, 零对角线
        self.M_l5 = nn.Parameter(torch.zeros(d["l5"], d["l5"], dtype=torch.float16))
        # 固定掩码 (只切 dW 更新路径): 10% 突触剪切 + [0.5,1.5] 随机增益播种
        for b in range(self.n_blocks):
            self.register_buffer(f"_syn_mask_{b}", (torch.rand(self.b5, d["l3"]) > 0.1).to(torch.float16))
        for b in range(self.n_blocks):
            self.register_buffer(f"_gain_mask_{b}", (0.5 + torch.rand(self.b5, d["l3"])).to(torch.float16))
        # L3 种子: W_23 固定随机增益掩码 (上游扰动级联到 L5)
        self.register_buffer("_gain_l3", (0.5 + torch.rand(d["l3"], d["l2"])).to(torch.float16))

        self._init_weights()

        # ── 引擎委托 (门面) ──
        from .forward import ForwardEngine
        from .learning import LearningEngine
        from .pruning import PruningEngine

        self.forward_engine = ForwardEngine(self)
        self.learning_engine = LearningEngine(self)
        self.pruner = PruningEngine(self)

    def _init_weights(self):
        """Kaiming 初始化所有权重 (行范数 ≈ 1.0, 配合 Oja 稳态)."""
        for name, p in self.named_parameters():
            if "bias" in name:
                continue
            nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(p.shape[-1]))

    # ── 门面: 一行委托 (逻辑在引擎模块) ──

    def forward(self, byte_ids: torch.Tensor) -> dict:
        """推理前馈: 返回未来预测偏差.  ACh 关闭, 确定性."""
        return self.forward_engine.forward(byte_ids)

    def generate(
        self, prompt: str, n_tokens: int = 40, temperature: float = 0.7, dev: torch.device | None = None
    ) -> bytes:
        """行动: L4 状态 + 预测差分 → W_lm 解码生成字节."""
        return self.forward_engine.generate(prompt, n_tokens=n_tokens, temperature=temperature, dev=dev)

    def _predict(self, byte_ids: torch.Tensor, store_state: bool = True, is_inference: bool = False) -> dict:
        """核心前馈: 感知 (L0→L6) + 微柱路由 + Foldiak 去相关 + 去中心化 + 增量预测."""
        return self.forward_engine._predict(byte_ids, store_state=store_state, is_inference=is_inference)

    def learn(self, byte_ids: torch.Tensor) -> dict:
        """Hebbian 学习 (零反传, 零误差回路). 不接收 targets."""
        return self.learning_engine.learn(byte_ids)

    def _prune(self):
        """拓扑重塑: 发育期内不剪 → 死缓二级判决 → 相对排名淘汰."""
        self.pruner._prune()

    def save(self, path: str):
        """保存模型权重."""
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, config: DensePCConfig | None = None) -> DensePCNet:
        """加载模型权重 (含修剪后的检查点: 按检查点形状对齐, 重设 active_size).

        未传 config 时按检查点 W_* 形状推断层维度 (修剪后检查点维度 < 默认值).
        """
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if config is None:
            config = DensePCConfig(
                d_l4=sd["W_04"].shape[0],
                d_l2=sd["W_42"].shape[0],
                d_l3=sd["W_23"].shape[0],
                d_l5=sd["W_56"].shape[1],
                d_l6=sd["W_56"].shape[0],
                # W_04 列数 512 = 双通道 [z0[t], z0[t-1]], 256 = 单帧
                input_history=sd["W_04"].shape[1] != 256,
            )
        net = cls(config)
        nsd = net.state_dict()
        for k, v in sd.items():
            if k not in nsd:
                continue
            if nsd[k].shape == v.shape:
                nsd[k] = v
            else:
                idx = tuple(slice(0, min(a, b)) for a, b in zip(nsd[k].shape, v.shape))
                nsd[k][idx] = v[idx]
        net.load_state_dict(nsd)
        net.active_size = {
            "l4": net.W_04.shape[0],
            "l2": net.W_42.shape[0],
            "l3": net.W_23.shape[0],
            "l5": net.W_56.shape[1],
            "l6": net.W_56.shape[0],
        }
        return net
