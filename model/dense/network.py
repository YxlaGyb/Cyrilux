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
    # 自适应塑性归一化 (第 75 轮): s_h = clip(0.03/ρ_hebb^raw, 0.005, 1.0) 缩放 W_35
    # Hebbian 外积. 单变量实验开关, 默认关保持旧行为
    adaptive_rho: bool = False
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

        # ── 自组织预测引擎 (第 50 轮): 层间局部预测矩阵 ──
        # W_pred_54: L4 → 预测 L5 (pred_L5 = z4 @ W_pred_54.T, [N,S,a5])
        # W_pred_43: L3 → 预测 L4 (pred_L4 = z3 @ W_pred_43.T, [N,S,a4])
        # 每层拥有自己的预测目标: local_err = z_cur - pred(上层), 惊喜度直接
        # 驱动中间层重组. 纯外积更新, 零 BP
        self.W_pred_54 = nn.Parameter(torch.empty(d["l5"], d["l4"], dtype=torch.float16))
        self.W_pred_43 = nn.Parameter(torch.empty(d["l4"], d["l3"], dtype=torch.float16))

        # ── 竞争性概念绑定层 (纯赫布非线性, 任务 4): z4 → W_bind → K=16 概念槽 ──
        # sim = z4 @ W_bind [N,S,16]; 软竞争 z_bind = sim / ‖sim‖ (L2 归一化产生
        # 竞争性非线性: 匹配槽被相对放大, 其余被抑制). 所有槽位向量参与 W_lm
        # 预测, 由 W_lm 端学习哪些槽组合预测哪字节. 槽位更新纯赫布外积 + 去均值
        # (Oja 式, 零 BP), 归一化防坍缩
        self.bind_slot_dim = 32  # 裁决 14: 16→32 槽扩容 (容量上限 16→32, 20+ 可达)
        self._lm_in = d["l4"] * 4 + self.bind_slot_dim
        self.W_bind = nn.Parameter(torch.empty(d["l4"], self.bind_slot_dim, dtype=torch.float16))
        # 概念槽 homeostatic 滑阈 (第 75 轮): θ_j 跟踪 z_bind_j² 槽能量,
        # z_bind_j /= (1+θ_j) — 高激活槽增益低 → 对称性自发破缺 → 软 WTA 涌现
        self.register_buffer("_theta_bind", torch.zeros(self.bind_slot_dim, dtype=torch.float16))
        # W_bind 列方向去同质化 (第 75 轮: 逐样本加权破对称后的防垄断安全网,
        # 槽维 16×16, 与行 decorr 正交)
        self.E_bind_col = nn.Parameter(torch.zeros(self.bind_slot_dim, self.bind_slot_dim, dtype=torch.float16))
        # ── 槽自循环 W_bind_self (第 76 轮, 海马 CA3 循环侧支) ──
        # z_bind[t] = f(z4[t]@W_bind + z_bind[t-1]@W_bind_self): 槽激活携带自身
        # 历史 → 动态检测器, 时序积分使槽可编码多字节模式 (无记忆状态机无法
        # 产生结构化输出). 初始 std=1/√16 小随机, z4 主导; 序列首步 prev=零.
        # 学习: 槽共现 Hebbian (i@t-1 激活 & j@t 激活 → 强化 i→j 转移),
        # soft_norm 行范数约束与主 W_bind 同步
        self.W_bind_self = nn.Parameter(torch.empty(self.bind_slot_dim, self.bind_slot_dim, dtype=torch.float16))
        # W_bind_self 列去同质化 (第 76 轮, 裁决: 防均匀转移坍缩): 16 列各对应
        # 一个源槽的转移权重向量 — 均匀转移时所有列收敛同向量 (列相似度 0.906
        # 实测), decorr 施斥力迫使不同源槽分化目标槽偏好. 与 W_bind 主矩阵列
        # decorr 同构, 零起步 (E=0 → 无抑制)
        self.E_bind_self = nn.Parameter(torch.zeros(self.bind_slot_dim, self.bind_slot_dim, dtype=torch.float16))
        # ── 自发活动发生器 (第 76 轮战略转向: 双重驱动) ──
        # 内部全局调控信号 intrinsic_drive: 独立于外部预测误差的内部驱动源.
        # 自发振荡 A = 0.5+0.5·sin(2π·cnt/20) — 纯 fp16: cnt 为 0-19 整数计数器
        # (fp16 精确表示), 正弦查表 _intr_sin 预计算 20 项 fp16 (周期 ~20 步,
        # 自发节律, 不依赖任何输入); 状态耦合 Ω = 槽激活功率 EMA (概念槽切换
        # 频率, 内部状态). intr = 0.5·A + 0.5·Ω ∈ [0,1], 广播至 W_bind_self
        # 与 W_act 更新 — 与多巴胺/乙酰胆碱并列的第三驱动 (内部, 非外部误差)
        self.register_buffer("_intr_cnt", torch.zeros(1, dtype=torch.float16))  # 相位计数器 0-19
        self.register_buffer(
            "_intr_sin",
            torch.tensor(
                [0.5 + 0.5 * math.sin(2.0 * math.pi * i / 20.0) for i in range(20)],
                dtype=torch.float16,
            ),
        )  # 正弦查表 [20] fp16
        self.register_buffer("_intr_omega", torch.tensor(0.3, dtype=torch.float16))  # 槽切换功率 EMA
        # ── 动作读出矩阵 W_act (无 Token 生成, 第 76 轮): [K=16 槽, 256 字节] ──
        # 概念槽 z_bind → 离散字节脉冲: potential = z_bind @ W_act; 推理 argmax 脉冲
        # 输出, 学习三因子赫布 (目标列强化 + softmax 稳态抑制 + dop_gain 门控).
        # 行 = 槽 (随 W_bind 同 perm 修剪), 列 = 256 字节 (soft_norm 列归一有界)
        self.W_act = nn.Parameter(torch.empty(self.bind_slot_dim, self.cfg.d_input, dtype=torch.float16))
        # W_act 输出端 BCM 滑阈 (防字节垄断): theta = EMA(potential²), 高电位字节
        # 抑制 → 稳态 (与 _theta_wlm 同款剪刀)
        self.register_buffer("_theta_act", torch.full((self.cfg.d_input,), 0.01, dtype=torch.float16))
        # 独立逐样本 surprise EMA (第 75 轮最终): 每样本独立历史, 初始 1.0
        # 让第一批 rel_n ≈ surprise_n (无共享统计, 纯局部)
        self.register_buffer("_s_ema_n", torch.ones(8, dtype=torch.float16))

        # ── LM 头 (自监督赫布): [z4, m2, m8, m32, bind] → 256 字节 logits, 独立于重建 W_04 ──
        # W_04 双向重建被证实解码死锁 (真实 delta 注入仍复读空格);
        # W_lm 唯一任务: 把状态映射到下一字节, 纯外积.
        # 多级记忆池拼接进 W_lm 输入 (输入维 = 4×d_l4 + bind): 3 级因果卷积核
        # (2/8/32 步) 承载跨序列低分辨率信息; 池间竞争由各通路 BCM 滑阈自动调节 (注意力雏形)
        self.register_buffer("_m_pool", torch.zeros(3 * d["l4"], dtype=torch.float16))
        # 内部 EMA 频率 (第 54 轮读出端去偏): 纯模型内部累计 target 分布,
        # 绝不依赖外部统计. 全 fp16 (0.01 级增量 fp16 可精确表示, 1/256 初值
        # 与增量同量级, 无精度损失)
        self.register_buffer("_freq", torch.full((self.cfg.d_input,), 1.0 / 256.0, dtype=torch.float16))
        # ── 非线性混合层 (第 57 轮): W1 [lm_in, d_h] + 三阶激活 + 转置误差传播 ──
        # h = zh @ W1; h = h·(1-0.5h²); logits = h @ W_lm.
        # W_lm 输入从 zh 变为混合特征 h (d_h 维), 横向交叉组合打散高频 e 列.
        # 纯赫布: dW_lm = h^T e; e_h = e @ W_lm.T · (1-1.5h²); dW1 = zh^T e_h. 零 BP
        self.d_h = 256
        self.W1 = nn.Parameter(torch.empty(self._lm_in, self.d_h, dtype=torch.float16))
        # W_lm 行 = 混合特征 h 维度 (d_h), 列 = 256 字节; 修剪 perm 重排只影响
        # z4/池段 (前 4 段), h 为折叠空间 (无神经元映射) → 恒等保持
        self.W_lm = nn.Parameter(torch.empty(self.d_h, self.cfg.d_input, dtype=torch.float16))
        # W_lm_2 独立子预测器 (Q3 解耦): 专责 t+2 预测, 共享混合特征 h 输入,
        # 更新与 W_lm 完全独立 (dW 互不干扰) — 避免同一突触拟合双目标的信号冲突
        self.W_lm_2 = nn.Parameter(torch.empty(self.d_h, self.cfg.d_input, dtype=torch.float16))
        self.bias_lm = nn.Parameter(torch.zeros(self.cfg.d_input, dtype=torch.float16))

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
        # W_04 输出端 homeostatic 滑阈 (第 75 轮): θ_j 慢速跟踪 mu4_j² 平均能量,
        # g_j = 1/(1+θ_j) 抑制高激活列权重更新 → 打破 σ₁ 奇异值垄断
        self.register_buffer("_theta_w04", torch.full((d["l4"],), 0.01, dtype=torch.float16))
        # W_t4 输出端 homeostatic 滑阈 (第 75 轮): 打破 W_t4 σ₁/σ₂≈4000:1 垄断,
        # rec4 从压制源转丰富源. θ_j 跟踪 rec4_j² = 时序差分每维能量
        self.register_buffer("_theta_wt4", torch.zeros(d["l4"], dtype=torch.float16))

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
        # 批级预测熵 EMA 基线 (动态自适应锚点): 当前熵高于基线 = 迷茫 (放 bias),
        # 低于基线 = 稳定 (衰减缩回). 初始 5.5 ≈ 均匀分布熵 (256 字节)
        self.register_buffer("_ent_ema", torch.tensor(5.5, dtype=torch.float16))
        # 动态稳态竞争状态: 20 步熵窗口 (环形) + 最小二乘斜率拟合预分配 (t 中心化)
        # + 速率自适应缩放因子 (全 fp16; scale ∈ (0,2) 数学有界)
        self.register_buffer("_ent_buf", torch.zeros(20, dtype=torch.float16))
        self.register_buffer("_t_center", torch.arange(20, dtype=torch.float16) - 9.5)
        self.register_buffer("_t_denom", (torch.arange(20, dtype=torch.float16) - 9.5).square().sum())
        self._ent_i = 0
        self.register_buffer("_traction_scale", torch.tensor(1.0, dtype=torch.float16))
        # 快慢时序散度 (第 69 轮): Z_slow = β·Z_slow + (1-β)·Z_fast (慢通道滑动
        # 记忆, fp16 纯量乘法); 新奇度 N = ‖Z_fast - Z_slow‖² — 死循环时状态
        # 不再转移, Z_fast 坍缩向 Z_slow, N → 0, 触发负向多巴胺 (主动遗忘 LTD)
        self.register_buffer("_z_slow", torch.zeros(d["l4"], dtype=torch.float16))
        # tau 初始 = z4 原始幅度 (std~0.06) 平方量级: 若初始 0.01 > nov 量级,
        # nov-tau 恒负 → D 恒负 → 全局 LTD 擦除权重 (第 69 轮实测熵 1.18→5.5)
        self.register_buffer("_theta_novelty", torch.full((1,), 0.001, dtype=torch.float16))
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
        self.E_t4 = nn.Parameter(torch.zeros(d["l4"], d["l4"], dtype=torch.float16))  # W_t4 漏挂修复
        self.E_t5 = nn.Parameter(torch.zeros(d["l5"], d["l5"], dtype=torch.float16))
        # W_bind 行去同质化矩阵 (与 E_l5/E_42/E_23 同款, 防秩 1 自锁)
        self.E_bind = nn.Parameter(torch.zeros(d["l4"], d["l4"], dtype=torch.float16))
        self.E_04 = nn.Parameter(torch.zeros(d["l4"], d["l4"], dtype=torch.float16))
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
            if name in ("E_l5", "E_42", "E_23", "M_l5", "E_t2", "E_t3", "E_t4", "E_t5", "E_bind", "E_bind_col", "E_04", "E_bind_self"):
                continue  # 去相关矩阵零起步 (M=0 → 前向恒等)
            if name == "W_act":
                nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(self.bind_slot_dim))
            if name == "W_bind_self":
                nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(self.bind_slot_dim))
            if name in ("W_lm", "W_lm_2"):
                nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(self.d_h))
            elif name == "W1":
                nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(p.shape[0]))
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

    def learn(self, byte_ids: torch.Tensor, closed_loop: bool = False) -> dict:
        """Hebbian 学习 (零反传, 零误差回路). 不接收 targets.

        closed_loop=True: 自回归闭环训练 (输入 = 真实前缀 + 自生成后缀).
        """
        return self.learning_engine.learn(byte_ids, closed_loop=closed_loop)

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
