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
    d_l5: int = 1024  # 统一 L5 (撤销硬切块): 原 256 拆 4×64 微柱, 现单矩阵 [1024,384]
    d_l6: int = 128
    max_seq_len: int = 256
    # 时序双通道输入: W_04 输入 = [z0[t], z0[t-1]] 拼接 (词序信息进入表示层).
    # 单帧重建约束是词序盲区的根因 (dog/cat 互换 z5 恒同); 双通道纯线性路由, 合规
    input_history: bool = True

    # Hebbian 物理有效学习率 (1:1 映射到内部更新公式，无隐藏缩放)
    lr_hebbian: float = 0.003
    temporal_lr_ratio: float = 5.0  # 频率锚点方案: 恢复默认, 不靠降速治谱
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
    # 动态稳态竞争 (速率自适应): 按 W_lm 熵斜率调节表示层更新幅度 —
    # 熵加速下降 → 表示层放慢 (保护成果); 熵停滞 → 表示层放大 (强迫重组供新信息)
    adaptive_traction: bool = False
    lm_lr_boost: float = 1.0
    oja_elasticity: float = 0.05
    probation_decay: float = 0.5

    # 时序预测连接 (时空差分共振) 配置
    gen_precision: float = 1.0  # 差分共振强度
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
        # ── L5 统一矩阵 (撤销微柱硬切块): 单 W_35 [1024,384], 无块路由 ──
        self.W_35 = nn.Parameter(torch.empty(d["l5"], d["l3"], dtype=torch.float16))
        self.W_56 = nn.Parameter(torch.empty(d["l6"], d["l5"], dtype=torch.float16))

        # ── 世界模型 (下一状态预测): W_diff 在 L4 空间预测 Δz4 = z4[t] - z4[t-1] ──
        self.W_diff = nn.Parameter(torch.empty(d["l4"], d["l4"], dtype=torch.float16))
        self.b_diff = nn.Parameter(torch.zeros(d["l4"], dtype=torch.float16))
        # ── 状态预测矩阵 (预测编码融合): 独立于 W_diff, 专门给表示层提供预测误差 ──
        # eps_state = (z4[t] @ W_state_pred) - (z4[t+1] - z4[t]); 表示层最终误差
        # final_eps = eps_recon + 0.3 * eps_state — 迫使隐状态携带"未来往哪走"
        self.W_state_pred = nn.Parameter(torch.empty(d["l4"], d["l4"], dtype=torch.float16))

        # ── LM 头 (自监督赫布): [z4, m2, m8, m32] → 256 字节 logits, 独立于重建 W_04 ──
        # W_04 双向重建被证实解码死锁 (真实 delta 注入仍复读空格);
        # W_lm 唯一任务: 把状态映射到下一字节, 纯外积.
        # 多级记忆池拼接进 W_lm 输入 (输入维 = 4×d_l4): 3 级因果卷积核 (2/8/32 步)
        # 承载跨序列低分辨率信息; 池间竞争由各通路 BCM 滑阈自动调节 (注意力雏形)
        self.register_buffer("_m_pool", torch.zeros(3 * d["l4"], dtype=torch.float16))
        # W_lm 行 = 输入神经元 (z4 + 3 记忆池, 神经元对齐), 列 = 256 字节;
        # 修剪 perm 重排需同步 4 个区段 (见 pruning._sync_l4_aux)
        self.W_lm = nn.Parameter(torch.empty(d["l4"] * 4, self.cfg.d_input, dtype=torch.float16))
        # W_lm_2 独立子预测器 (Q3 解耦): 专责 t+2 预测, 共享 z4/记忆池输入,
        # 更新与 W_lm 完全独立 (dW 互不干扰) — 避免同一突触拟合双目标的信号冲突
        self.W_lm_2 = nn.Parameter(torch.empty(d["l4"] * 4, self.cfg.d_input, dtype=torch.float16))
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
        # 动态稳态竞争状态: 20 步熵窗口 (环形) + 最小二乘斜率拟合预分配 (t 中心化)
        # + 速率自适应缩放因子 (fp32 调度域, 非训练张量; scale ∈ (0,2) 数学有界)
        self.register_buffer("_ent_buf", torch.zeros(20, dtype=torch.float32))
        self.register_buffer("_t_center", torch.arange(20, dtype=torch.float32) - 9.5)
        self.register_buffer("_t_denom", (torch.arange(20, dtype=torch.float32) - 9.5).square().sum())
        self._ent_i = 0
        self.register_buffer("_traction_scale", torch.tensor(1.0, dtype=torch.float32))
        # BCM 滑阈 (替代 Oja): theta = EMA(eps²), phi = eps(eps-theta)
        for ln, dim in (("l4", d["l4"]), ("l2", d["l2"]), ("l3", d["l3"]), ("l5", d["l5"]), ("l6", d["l6"])):
            self.register_buffer(f"_theta_{ln}", torch.full((dim,), 0.01, dtype=torch.float16))
        self.register_buffer("_theta_diff", torch.full((d["l4"],), 0.01, dtype=torch.float16))
        # W_lm 专属 BCM 滑阈 (防输出过冲): theta = EMA(pred²), phi = pred(pred-theta),
        # W_lm 开始输出高频极值 (振荡源头) 时 theta 升高 → pred-theta<0 → 抑制过冲.
        # 纯机制剪刀, 线性, 无 BP; 与 W_diff 的 _theta_w 同模式
        self.register_buffer("_theta_wlm", torch.full((self.cfg.d_input,), 0.01, dtype=torch.float16))
        # W_lm_2 专属 BCM 滑阈 (同 W_lm 防过冲模式)
        self.register_buffer("_theta_wlm2", torch.full((self.cfg.d_input,), 0.01, dtype=torch.float16))
        # 池间侧抑制竞争滑阈 (Q4): 4 池各一个能量级 BCM theta (标量, 池级竞争)
        self.register_buffer("_theta_pool", torch.full((4,), 0.01, dtype=torch.float16))
        # Foldiak 反赫布权重去同质化矩阵 (每层一个): E 学权重行间**绝对相关**
        # (|cos| — 有符号 cos 被 ± 符号随机掩盖, 检测不到 Hebbian 行收敛到 ±w 对),
        # 作用于 dW/W: W -= 0.2·E_n@W, 打破行收敛 → 投影秩 1 (PR_eff 焊死根因).
        # E 零起步 = 无抑制; 零对角; 指数遗忘 ×0.99, 稳态 E_ij ≈ |corr_ij|
        self.E_l5 = nn.Parameter(torch.zeros(d["l5"], d["l5"], dtype=torch.float16))
        self.E_42 = nn.Parameter(torch.zeros(d["l2"], d["l2"], dtype=torch.float16))
        self.E_23 = nn.Parameter(torch.zeros(d["l3"], d["l3"], dtype=torch.float16))
        # 时间核去同质化矩阵: W_t 既有秩 1 结构 (top1 sv 11.4 固化, 旧检查点
        # 激活秩 1 的来源 — 锚点只防新增长, 不清除旧结构). 超量 E (β=1.2)
        # 直接清除 W_t 主方向
        self.E_t2 = nn.Parameter(torch.zeros(d["l2"], d["l2"], dtype=torch.float16))
        self.E_t3 = nn.Parameter(torch.zeros(d["l3"], d["l3"], dtype=torch.float16))
        self.E_t5 = nn.Parameter(torch.zeros(d["l5"], d["l5"], dtype=torch.float16))
        # Foldiak 反赫布侧抑制矩阵 (L5 激活去相关, 方案 D): M 零起步 = 前向恒等
        # (z5 - α·M@z5); 学 z_out 协方差 (白化本质), 零对角, 指数遗忘防爆炸
        self.M_l5 = nn.Parameter(torch.zeros(d["l5"], d["l5"], dtype=torch.float16))
        # 固定掩码 (只切 dW 更新路径): [0.5,1.5] 随机增益播种 (L5 无块, 无 10% 剪切)
        self.register_buffer("_gain_mask", (0.5 + torch.rand(d["l5"], d["l3"])).to(torch.float16))
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
        """Kaiming 初始化所有权重 (行范数 ≈ 1.0, 配合 Oja 稳态).

        例外: W_lm/W_lm_2 的记忆池段 (m2/m8/m32, 后 3/4) 初始化接近 0 (1e-4) —
        新池从"无贡献"开始学 (生物发育: 突触从杂乱到有效), 由池门控自然放大
        有用的池; z4 段保持 Kaiming (加载旧权重时被覆盖).
        """
        for name, p in self.named_parameters():
            if "bias" in name:
                continue
            if name in ("E_l5", "E_42", "E_23", "M_l5", "E_t2", "E_t3", "E_t5"):
                continue  # 去相关矩阵零起步 (M=0 → 前向恒等)
            if name in ("W_lm", "W_lm_2"):
                nn.init.normal_(p[: self.cfg.d_l4], mean=0.0, std=1.0 / math.sqrt(p.shape[-1]))
                nn.init.normal_(p[self.cfg.d_l4:], mean=0.0, std=1e-4)
            else:
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
        # _m_pool (多级记忆池) 对齐活性 L4 (旧检查点无此缓冲 → init 全量, 需裁 3 段)
        if net._m_pool.shape[0] != 3 * net.active_size["l4"]:
            old_m = net._m_pool.data
            net.register_buffer("_m_pool", old_m[: 3 * net.active_size["l4"]].contiguous())
            del old_m
        return net
