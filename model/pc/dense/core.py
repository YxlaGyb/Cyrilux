"""DensePCNet-PPA — 感知-预测-行动 (Perception-Prediction-Action) 闭环网络.

全 fp16, 零反向传播, 零 .item(), 纯 matmul, 无位置编码:
- 感知: L0(纯 one-hot) → L4 → L2 → L3 → L5 → L6 (自下而上)
- 时序: 每层学习的时间核 W_t 递归 z[t] 依赖 z[t-1], 时序轨迹在隐空间自然分化
- 生成: 时空差分共振 (L5_t → ΔL4_t), 未来相对现在的变化
- 精度: π_l = 1/(σ_εl + c); 高确定性(低惊喜)→ACh 记忆巩固, 低确定性→多巴胺大重构
- 学习: Hebbian dW = π·ε ⊗ z_pre, 零 autograd
- 监控: 自由能 free_energy = Σ_l ½·π_l·‖ε_l‖²  (不监控 PPL/Top-1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
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
    oja_elasticity: float = 0.05
    probation_decay: float = 0.5

    # 时序预测连接 (时空差分共振) 配置
    gen_precision: float = 1.0  # 差分共振强度
    rpe_gate_max: float = 8.0  # 保留 (未用于误差回路)
    # 微柱阵列: L5 拆成 n 个独立列块, 块间不共享 Hebbian/Oja
    l5_blocks: int = 4

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
    """PPA 闭环网络.

    Args:
        config: 网络配置 (或 None 使用默认).
    """

    def __init__(self, config: DensePCConfig | None = None, max_seq_len: int = 256):
        super().__init__()
        self.cfg = config or DensePCConfig()
        d = self.cfg.dims()

        # ── 前馈权重 (自下而上感知) ──
        self.W_04 = nn.Parameter(torch.empty(d["l4"], self.cfg.d_input, dtype=torch.float16))
        self.W_42 = nn.Parameter(torch.empty(d["l2"], d["l4"], dtype=torch.float16))
        self.W_23 = nn.Parameter(torch.empty(d["l3"], d["l2"], dtype=torch.float16))
        # ── 微柱阵列: L5 拆成 l5_blocks 个独立列块 (每块 W_block[b] 形状 [b5, a3])
        # 块间不共享 Hebbian/Oja, 各自竞争独立学习 (皮层微柱 Minicolumn)
        self.n_blocks = self.cfg.l5_blocks
        self.b5 = d["l5"] // self.n_blocks  # 每块维度
        self.W_35 = nn.ParameterList([
            nn.Parameter(torch.empty(self.b5, d["l3"], dtype=torch.float16))
            for _ in range(self.n_blocks)
        ])
        self.W_56 = nn.Parameter(torch.empty(d["l6"], d["l5"], dtype=torch.float16))

        # ── 时空差分共振 (世界模型): L5_t 预测 ΔL4_t = L4_{t+1} - L4_t ──
        # 差分更新天然含正负, 撑开 L5 表示空间, 避开 ±1 反相锁定
        self.W_diff = nn.Parameter(torch.empty(d["l4"], d["l5"], dtype=torch.float16))
        # 镜像解耦: b_diff 可学习偏置 (L5 零均值 → 投影归零, 偏置补生路)
        self.b_diff = nn.Parameter(torch.zeros(d["l4"], dtype=torch.float16))
        # 反频率门控: rarity[byte] = 1/sqrt(freq+1), 由数据集统计后注入
        # 高频数字权重极低, 稀有字母权重高 (惊奇度二次修正)
        self.register_buffer("_rarity", torch.ones(256, dtype=torch.float16))
        # ── 第 22 轮: 多步时间窗积分 (LTP/LTD) — 4 步滑动平均环形缓冲 ──
        # dW_diff 用最近 4 步的平均外积, 保留"前几秒的字母闪光"记忆
        # 单纯空格常量预测不再最优, W_diff 必须保持对前几步误差的记忆
        for i in range(4):
            self.register_buffer(f"_dw_buf_{i}", torch.zeros(d["l4"], d["l5"], dtype=torch.float16))
        self._buf_i = 0

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
            "l4": None, "l2": None, "l3": None, "l5": None, "l6": None
        }
        self._probation_counter: dict[str, torch.Tensor | None] = {
            "l4": None, "l2": None, "l3": None, "l5": None, "l6": None
        }
        # 惊喜基线 (EMA): 多巴胺/ACh 精度调度的参照系
        self.register_buffer("_surprise_buf", torch.tensor(1.0, dtype=torch.float16))
        # ── BCM 滑阈 (替代 Oja): theta = EMA(eps²), phi = eps(eps-theta) ──
        for ln, dim in (("l4", d["l4"]), ("l2", d["l2"]), ("l3", d["l3"]),
                        ("l5", d["l5"]), ("l6", d["l6"])):
            self.register_buffer(f"_theta_{ln}", torch.full((dim,), 0.01, dtype=torch.float16))
        self.register_buffer("_theta_diff", torch.full((d["l4"],), 0.01, dtype=torch.float16))
        # ── Foldiak 反赫布侧抑制 (L5 去相关): M 协方差, 零对角线 ──
        self.M_l5 = nn.Parameter(torch.zeros(d["l5"], d["l5"], dtype=torch.float16))
        # ── 固定 Bernoulli 突触剪切 (p=0.1): 只切 dW 更新路径, 不断激活路径 ──
        # 预先计算永久不变的掩码: 10% 位置永远 0; 随机关联断裂催生局部吸引子
        for b in range(self.n_blocks):
            self.register_buffer(f"_syn_mask_{b}",
                                 (torch.rand(self.b5, d["l3"]) > 0.1).to(torch.float16))
        # ── 破局一: 静态随机增益软掩码 (播种对称破缺) ──
        # 每神经元永久不同的增益 [0.5,1.5], 第一轮后有效协方差 C_j 永久不同
        # 纯矩阵操作, GEMM 形状不变, Tensor Core 无损; 非调参
        for b in range(self.n_blocks):
            self.register_buffer(f"_gain_mask_{b}",
                                 (0.5 + torch.rand(self.b5, d["l3"])).to(torch.float16))
        # ── L3 种子 (全链条推广): W_23 固定随机增益掩码, 每行(神经元)不同 ──
        self.register_buffer("_gain_l3", (0.5 + torch.rand(d["l3"], d["l2"])).to(torch.float16))

        self._init_weights()

    def _init_weights(self):
        """Kaiming 初始化所有权重 (行范数 ≈ 1.0, 配合 Oja 稳态)."""
        for name, p in self.named_parameters():
            if "bias" in name:
                continue
            nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(p.shape[-1]))

    # ─────────────────────────────────────────────────────────────
    # 前馈
    # ─────────────────────────────────────────────────────────────

    def forward(self, byte_ids: torch.Tensor) -> dict:
        """推理前馈: 返回未来预测偏差.  ACh 关闭, 确定性."""
        return self._predict(byte_ids, store_state=True, is_inference=True)

    def generate(self, prompt: str, n_tokens: int = 40, temperature: float = 0.7,
                 dev: torch.device | None = None) -> bytes:
        """行动: 自顶向下重建生成字节序列 (L4 → W_04 解码回 L0 字节空间).

        生成端预测编码反馈注入 (不改训练): L4 每步输出前叠加 L5 差分预测
        z4_gen = z4 + α*(z5_raw @ W_diff + b_diff)  — 自上而下的预测注入
        打破 L4 高频数字死循环 (W_diff 学的是"下一时刻 L4 的变化量")
        """
        if dev is None:
            dev = next(self.parameters()).device
        gen = list(prompt.encode("utf-8"))
        for _ in range(n_tokens):
            bv = torch.tensor([gen[-64:]], dtype=torch.long, device=dev)
            _ = self._predict(bv, store_state=True, is_inference=True)
            a4 = self.active_size["l4"]
            a5 = self.active_size["l5"]
            W_diff_a = self.W_diff[:a4, :a5]
            # L5 差分预测 (原始幅度 z5_raw, 与训练通路一致)
            mu_diff = self._z5_raw @ W_diff_a.T + self.bias_l4[:a4] + self.b_diff[:a4].unsqueeze(0)
            z4_gen = self._z4 + 1.0 * mu_diff  # α=1.0 预测注入
            mu0_top = z4_gen @ self.W_04[:a4]  # [1,S,256]
            last = mu0_top[0, -1].float() / temperature
            topv, _ = torch.topk(last, min(15, 256))
            last[last < topv[-1]] = -float("inf")
            probs = torch.softmax(last, dim=-1)
            gen.append(int(torch.multinomial(probs, 1).item()))
        return bytes(gen)

    def _recurrent(self, mu: torch.Tensor, dev: torch.device, W_t: torch.Tensor, sweeps: int = 3) -> torch.Tensor:
        """时间核递归: z[t] 依赖 z[t-1] (经学习到的 W_t), 纯 matmul 全 fp16.

        因果移位 + 前向卷积式扫描, sweeps 次让时序信息向后传播更远.
        时序耦合强度由 W_t 的学习范数决定 (非超参数), pre-norm + 1/√d 防溢出.
        """
        N, S, d = mu.shape
        z = mu
        inv_d = 1.0 / math.sqrt(d)
        for _ in range(sweeps):
            shift = torch.cat([torch.zeros(N, 1, d, dtype=z.dtype, device=dev), z[:, :-1]], dim=1)
            s_rms = shift.square().mean(dim=-1, keepdim=True)
            shift_n = shift * torch.rsqrt(s_rms + 1e-6)
            z = z + (shift_n @ W_t.T) * inv_d
            rms = z.square().mean(dim=-1, keepdim=True)
            z = z * torch.rsqrt(rms + 1e-6)
        return z

    def _predict(self, byte_ids: torch.Tensor, store_state: bool = True, is_inference: bool = False) -> dict:
        """核心前馈 (感知→时序).  无误差抑制回路 — L5 不修正 L4, 只共振未来."""
        N, S = byte_ids.shape
        dev = byte_ids.device
        a4, a2, a3, a5, a6 = (self.active_size[k] for k in ("l4", "l2", "l3", "l5", "l6"))

        W_04_a = self.W_04[:a4]
        W_42_a = self.W_42[:a2]
        W_23_a = self.W_23[:a3]
        W_56_a = self.W_56[:a6]
        W_diff_a = self.W_diff[:a4, :a5]

        # ── 感知: L0 纯 one-hot (无位置编码, 时序全靠 W_t) ──
        z0 = F.one_hot(byte_ids, num_classes=256).to(torch.float16)  # [N,S,256]

        mu4 = z0 @ W_04_a.T + self.bias_l4[:a4]
        z4 = self._recurrent(mu4, dev, self.W_t4[:a4, :a4])
        mu2 = z4 @ W_42_a.T + self.bias_l2[:a2]
        z2 = self._recurrent(mu2, dev, self.W_t2[:a2, :a2])
        mu3 = z2 @ W_23_a.T + self.bias_l3[:a3]
        if not is_inference:
            # ACh 乙酰胆碱噪声: 底层 (L3 投影输入), 推理时关闭
            mu3 = mu3 + torch.sign(2.0 * (torch.rand_like(mu3) - 0.5)) * 0.03
        z3 = self._recurrent(mu3, dev, self.W_t3[:a3, :a3])
        # ── 微柱前馈: 4 个独立小矩阵, 时间步交错路由 ──
        # 每个微柱只看到"时间步交错后的局部特征", 不接触全批次公共统计
        # 微柱 b: 处理时间步 {b, b+4, ...}, 输出到 z5 的 [b*b5:(b+1)*b5] 特征切片
        z5 = torch.zeros(N, S, a5, dtype=torch.float16, device=dev)
        for b in range(self.n_blocks):
            z3_route = z3[:, b::self.n_blocks, :]  # [N, S//n, a3] 该微柱的时间步子集
            zb = z3_route @ self.W_35[b].T + self.bias_l5[b * self.b5:(b + 1) * self.b5]
            z5[:, b::self.n_blocks, b * self.b5:(b + 1) * self.b5] = zb
        # ── 第三步: Foldiak 反赫布去相关 (L5 输出协方差去相关) ──
        # z5 = z_raw - α*(M @ z_raw); α=0.2 维持档 (破局三降档: 0.4→0.2)
        # 先由随机增益播种差异, Foldiak 只放大维护已有差异, 不强行匀平
        z5_fd = z5 - 0.2 * (self.M_l5[:a5, :a5] @ z5.transpose(-2, -1)).transpose(-2, -1)
        # ── 路由分离 (第 18 轮): L5 内部保留两套输出 ──
        # z5_raw: Foldiak 后未去中心的原始绝对幅度 → 喂 W_diff 预测/更新
        # z5:      去中心化 → 喂 LM Head (维持 PR_eff, 保命脉)
        z5_raw = z5_fd
        z5 = z5_fd - z5_fd.mean(dim=-1, keepdim=True)
        # 稀疏性交给 BCM 滑阈 + Foldiak 自然竞争演化 (移除 k-WTA 硬截断)
        mu6 = z5 @ W_56_a.T + self.bias_l6[:a6]
        z6 = self._recurrent(mu6, dev, self.W_t6[:a6, :a6])

        # ── 时空差分共振: L5_t 预测 ΔL4_t = L4_{t+1} - L4_t (未来-现在, 含正负) ──
        # mu_diff[t] = W_diff @ z5_raw[t] + b_diff (原始幅度, 切断去中心化绑定)
        z5_raw_pad = torch.cat([z5_raw[:, 1:], torch.zeros(N, 1, a5, dtype=z5_raw.dtype, device=dev)], dim=1)
        bd = self.b_diff[:a4].unsqueeze(0).unsqueeze(0)
        mu_diff = z5_raw_pad @ W_diff_a.T + self.bias_l4[:a4] + bd  # [N,S,a4] 对齐 S
        # 差分共振偏差 (监控用, 不参与任何梯度/学习)
        diff_err = (z4 - mu_diff).square().mean()

        if store_state:
            self._z0 = z0
            self._z4 = z4
            self._z2 = z2
            self._z3 = z3
            self._z5 = z5
            self._z5_raw = z5_raw
            self._z6 = z6

        return {
            "mu_diff": mu_diff,
            "diff_err": diff_err,
            "free_energy": diff_err,
        }

    def _precise(self, eps: torch.Tensor) -> torch.Tensor:
        """精度加权: π_l = 1/(σ_εl + c), 归一化每层误差尺度."""
        s = eps.std() + 1e-3
        return eps / s

    def learn(self, byte_ids: torch.Tensor) -> dict:
        """Hebbian 学习 (时空相位锁定, 零反传, 零误差回路). 不接收 targets.

        唯一的"预测"学习: dW_future = L5_t ⊗ L4_{t+1} (纯共现共振, 无惩罚).
        其余前馈权重仍用逐层预测误差驱动 (标准 PC 自下而上学习).

        Args:
            byte_ids: [N, S] long 输入.

        Returns:
            stats dict (future_err, 各层误差范数).
        """
        _ = self._predict(byte_ids, store_state=True)
        N, S = byte_ids.shape
        dev = byte_ids.device
        a4, a2, a3, a5, a6 = (self.active_size[k] for k in ("l4", "l2", "l3", "l5", "l6"))
        a_sizes = [a4, a2, a3, a5, a6]

        W_23_a = self.W_23[:a3]
        W_56_a = self.W_56[:a6]
        W_diff_a = self.W_diff[:a4, :a5]

        z0 = self._z0
        z4, z2, z3, z5, z6 = self._z4, self._z2, self._z3, self._z5, self._z6

        # ── 逐层预测误差 (自下而上 PC, 驱动前馈权重) ──
        eps4 = z4 - (z0 @ self.W_04[:a4].T + self.bias_l4[:a4])  # L4 感知重建误差
        eps2 = z2 - (z4 @ self.W_42[:a2].T + self.bias_l2[:a2])  # L2 预测误差
        eps3 = z3 - (z2 @ W_23_a.T + self.bias_l3[:a3])  # L3 预测误差
        eps6 = z6 - (z5 @ W_56_a.T + self.bias_l6[:a6])  # L6 预测误差
        # L6 顶端: 时间自预测误差
        z6_pre = torch.cat([torch.zeros(N, 1, a6, dtype=z6.dtype, device=dev), z6[:, :-1]], dim=1)
        eps6 = z6 - z6_pre
        # ── L5 统一误差源: 时序差分 (移除静态前馈误差) ──
        # L5 的误差不再自我预测, 而是自身跨时刻的变化 z5[t]-z5[t-1]
        # W_35 与 W_diff 都竞争解释这个"变化特征" (对抗并行, L3 静态误差保留)
        eps5_td = z5[:, 1:] - z5[:, :-1]  # [N,S-1,a5]

        # ── 时空差分共振: 纯 Hebbian, 无误差, 差分含正负撑开表示空间 ──
        def _rms(x):
            rms = x.square().mean(dim=-1, keepdim=True)
            alive = (rms > 1e-8).to(x.dtype)
            return x * alive / (rms + 1e-4).sqrt()

        # ── 第 25 轮: 差动赫布学习 (Differential Hebbian) ──
        # dW_diff = (z5[t]-z5[t-1]).T @ (eps[t]-eps[t-1]) 纯时空差分外积
        # 连续空格/重复 e: z5 时域差分为 0 → dW 天然为 0 (数学必然, 非开关)
        # 无门控无阈值: 零变化→零更新, 网络被迫只学"发生变化的时刻"
        z4_fut = z4[:, 1:]  # [N,S-1,a4] 未来的 L4
        z4_cur = z4[:, :-1]  # [N,S-1,a4] 现在的 L4
        pred_d = z4_cur + self._z5_raw[:, :-1] @ W_diff_a.T + self.bias_l4[:a4] + self.b_diff[:a4]
        eps_t = z4_fut - pred_d  # [N,S-1,a4] 未来预测误差
        dz5 = self._z5_raw[:, 2:] - self._z5_raw[:, 1:-1]  # [N,S-2,a5] 输入变化
        deps = eps_t[:, 1:] - eps_t[:, :-1]  # [N,S-2,a4] 误差变化

        def _norm(x):  # 无阈值幅度归一: 0 保持 0 (无 alive 开关, 纯连续)
            r = x.square().mean(dim=-1, keepdim=True)
            return x / (r + 1e-4).sqrt()

        dW_diff_t = torch.bmm(
            _norm(deps).transpose(-2, -1), _norm(dz5)
        ).mean(dim=0) * (1.0 / (S - 2))
        # ── 第 22 轮: 4 步时间窗积分 — 环形缓冲池, 更新用最近 4 步平均外积 ──
        # dW = (dW_t + dW_t-1 + dW_t-2 + dW_t-3)/4; 保留"前几秒字母闪光"记忆
        # 单纯空格常量预测不再最优, W_diff 必须保持对前几步误差的记忆
        buf = getattr(self, f"_dw_buf_{self._buf_i}")
        buf.copy_(dW_diff_t)
        dW_avg = buf.clone()
        for i in range(1, 4):
            dW_avg = dW_avg + getattr(self, f"_dw_buf_{(self._buf_i - i) % 4}")
        dW_avg = dW_avg * 0.25
        self._buf_i = (self._buf_i + 1) % 4
        # b_diff 偏置误差 (在 eta 定义后更新, 见下方 W_diff 更新段)

        # ── 精度调度 (保留): 多巴胺/ACh 调制, 无动态 π 负反馈 (会中和) ──
        surprise = (eps4.square().mean() + eps2.square().mean() + eps3.square().mean()
                    + eps5_td.square().mean() + eps6.square().mean()) * 0.2
        rel = surprise / (self._surprise_buf + 1e-4)
        self._surprise_buf.data.mul_(0.95).add_(0.05 * surprise.data)
        dop_gain = rel.clamp(0.3, 5.0)
        ach_gain = (1.0 / (rel + 1e-4)).clamp(max=3.0)
        eta = self.cfg.lr_hebbian * dop_gain
        eta_t = eta * self.cfg.temporal_lr_ratio * ach_gain

        # ── Hebbian 外积 (逐层误差 ⊗ pre 活动) ──
        eps4_p, eps2_p, eps3_p, eps6_p = (
            self._precise(eps4), self._precise(eps2), self._precise(eps3),
            self._precise(eps6),
        )
        inv_s = 1.0 / S

        dW_list = [
            (eps4_p.transpose(-2, -1) @ _rms(z0)).mean(dim=0) * inv_s,
            (eps2_p.transpose(-2, -1) @ _rms(z4)).mean(dim=0) * inv_s,
            (eps6_p.transpose(-2, -1) @ _rms(z5)).mean(dim=0) * inv_s,
        ]
        W_list = [self.W_04[:a4], self.W_42[:a2], self.W_56[:a6]]
        for dW, W in zip(dW_list, W_list):
            col_mask = torch.rand(W.shape[0], 1, device=dev) < self.cfg.column_dropout
            W.data += (dW * (~col_mask).to(torch.float16)) * eta

        # ── L3 全链条种子: W_23 随机增益 + 误差门控 + 列 Oja (与 L5 同配置) ──
        # L3 权重非均匀扰动 → z3 批次内语义差异 → 上游级联到 L5 激活分散
        dW23 = (eps3_p.transpose(-2, -1) @ _rms(z2)).mean(dim=0) * inv_s
        err3_norm = eps3.pow(2).mean(dim=(0, 1)).sqrt() + 1e-8  # [a3] 每神经元
        gate3 = 0.1 + 0.9 * (err3_norm / err3_norm.max())
        dW23 = dW23 * self._gain_l3[:a3, :] * gate3.unsqueeze(1)
        c3_mask = torch.rand(a3, 1, device=dev) < self.cfg.column_dropout
        self.W_23[:a3].data += (dW23 * (~c3_mask).to(torch.float16)) * eta
        # L3: 无 Oja 归一化 (全面移除) — BCM 滑阈承担稳定性职责
        # 随机增益 + 误差门控的差异种子不再被任何归一化抹平

        # ── 第一步: 换轴 — 资格迹+样本竞争 (bmm 逐样本, 显著性加权, 不批平均) ──
        # dW_n = bmm(eps^T, z) [N, d_out, d_in]; gate_n = surprise_n / Σ surprise_n
        # 线性加权 (禁止 exp/非线性), 保留样本差异, 抹平效应消除
        n_sub = 4
        sub = max(1, N // n_sub)
        for b in range(self.n_blocks):
            b_s = slice(b * self.b5, (b + 1) * self.b5)
            z5_b = z5[:, b::self.n_blocks, b_s]  # [N,S//n,b5] 该块路由步的激活
            eps_b = z5_b[:, 1:] - z5_b[:, :-1]  # [N,S//n-1,b5] 块内时序差分误差
            z3_b = z3[:, b::self.n_blocks, :][:, :-1]  # [N,S//n-1,a3] 对齐 pre
            z3_bp = z3_b * (torch.rand_like(z3_b) > 0.3).to(torch.float16)
            # 第二步: BCM 滑阈 — theta = EMA(eps²), phi = eps(eps - theta)
            # 活跃度高的神经元 theta 升高 → 误差变号停止增长 → 逼迫他人接手
            th = self._theta_l5[b * self.b5:(b + 1) * self.b5]
            e2 = (eps_b * eps_b).mean(dim=(0, 1))
            # 破局三: BCM 降档为维持档 (强度减半: 0.05→0.025)
            th.mul_(0.975).add_(0.025 * e2)
            phi_b = eps_b * (eps_b - th)
            # 幅度稳定 (结构化, 非 clamp): RMSNorm 保 phi 尺度, 防 bmm fp16 累加溢出
            phi_rms = phi_b.square().mean(dim=-1, keepdim=True)
            phi_b = phi_b * torch.rsqrt(phi_rms + 1e-4)
            # 样本显著性: gate_n = surprise_n / Σ surprise_n (线性, 非门控函数)
            g_n = (phi_b * phi_b).mean(dim=(1, 2)) + 1e-8
            g_n = g_n / g_n.sum()
            Wb = self.W_35[b].data
            syn_mask = getattr(self, f"_syn_mask_{b}")  # 固定永久掩码, 只切 dW
            gain_mask = getattr(self, f"_gain_mask_{b}")  # 破局一: 静态随机增益
            # 误差幅度门控 (gate_floor): 高误差神经元主导更新, 低误差保 10% 下限
            # err_norm = eps 在 (样本,时间) 维度的 L2 幅度均值 → 每神经元 [b5]
            err_norm = eps_b.pow(2).mean(dim=(0, 1)).sqrt() + 1e-8  # [b5]
            upd_gate = 0.1 + 0.9 * (err_norm / err_norm.max())
            for s in range(n_sub):
                sl = slice(s * sub, (s + 1) * sub)
                dW_n = torch.bmm(phi_b[sl].transpose(-2, -1), _rms(z3_bp[sl]))
                dW_sub = (dW_n * g_n[sl, None, None]).sum(dim=0) * (1.0 / (S // self.n_blocks - 1))
                # 破局一: 静态随机增益播种 (乘 [0.5,1.5] 固定掩码)
                dW_sub = dW_sub * gain_mask
                # 固定突触剪切: dW 乘永久掩码 (10% 位置恒 0), 进 Oja 之前
                dW_sub = dW_sub * syn_mask
                # 破局二: 更新层门控 (0.1 下限), 最没用神经元也保 10% 更新
                dW_sub = dW_sub * upd_gate.unsqueeze(1)
                b_mask = torch.rand(self.b5, 1, device=dev) < self.cfg.column_dropout
                Wb += (dW_sub * (~b_mask).to(torch.float16)) * eta
                # 每子块后: 无 Oja 归一化 (全面移除) — BCM 滑阈承担稳定性
                # gain_mask 的神经元幅度差异不再被列归一化抹平
                # (BCM 有界性: 高激活→高 theta→误差变号→权重受限, 不会爆炸)

        # ── 第三步: Foldiak 反赫布 — 侧向矩阵 M 协方差去相关 (零对角线) ──
        # 破局三: Foldiak 降档为维持档 (强度减半: 0.01→0.005, 前向 α 0.4→0.2)
        z5_flat = z5.reshape(-1, a5)
        dM = (z5_flat.transpose(0, 1) @ z5_flat) / z5_flat.shape[0]
        eye_mask = (1.0 - torch.eye(a5, device=dev, dtype=torch.float16))
        self.M_l5[:a5, :a5].data += (dM * eye_mask) * (0.005 * ach_gain) * eta

        # ── 时空差分连接 Hebbian 更新 (纯共现, 无误差) ──
        # 用 4 步时间窗平均 dW_avg (第 22 轮)
        fut_mask = torch.rand(a4, 1, device=dev) < self.cfg.column_dropout
        W_diff_a.data += (dW_avg * (~fut_mask).to(torch.float16)) * eta
        # b_diff 偏置 Hebbian 更新: db = future_err (纯逐元素累加, 补零均值生路)
        future_e = (z4[:, 1:] - (z4[:, :-1] + self._z5_raw[:, :-1] @ W_diff_a.T + self.bias_l4[:a4] + self.b_diff[:a4])).mean(dim=(0, 1))
        self.b_diff[:a4].data += future_e * eta

        # ── 时序 Hebbian (W_t 学习, 高确定性时增强 → 记忆巩固) ──
        for (z_cur, W_t), a_sz in zip(
            [(z4, self.W_t4), (z2, self.W_t2), (z3, self.W_t3), (z5, self.W_t5), (z6, self.W_t6)],
            a_sizes
        ):
            pre = z_cur[:, :-1]
            post = z_cur[:, 1:]
            dW_t = (_rms(pre).transpose(-2, -1) @ _rms(post)).mean(dim=0) * (1.0 / (S - 1))
            W_t[:a_sz, :a_sz].data += dW_t * eta_t
            # 无软 Oja (全面移除): BCM 时序竞争承担稳定性

        # ── 无 Oja 归一化 (全面移除) ──
        # 前馈权重 W_04/W_42/W_56 与 W_diff 的稳定性由 BCM 滑阈承担

        # ── 拓扑重塑 ──
        self._step_counter += 1
        if self._step_counter > self.cfg.prune_warmup and self._step_counter % self.cfg.prune_interval == 0:
            self._prune()

        return {
            "free_energy": (eps4.square().mean() + eps2.square().mean() + eps3.square().mean()
                            + eps5_td.square().mean() + eps6.square().mean()),
            "future_err": (z4[:, 1:] - (z4[:, :-1] + z5[:, :-1] @ W_diff_a.T + self.bias_l4[:a4])).square().mean(),
            "surprise": surprise,
            "dop_gain": dop_gain,
            "ach_gain": ach_gain,
        }

    # ═══════════════════════════════════════════════════════════
    # 动态神经元修剪 (慢速循环)
    # ═══════════════════════════════════════════════════════════

    def _prune(self):
        """拓扑重塑: 发育期内不剪 → 死缓二级判决 → 相对排名淘汰."""
        layers = ["l4", "l2", "l3", "l6"]
        W_attr = {"l4": "W_04", "l2": "W_42", "l3": "W_23", "l6": "W_56"}
        t_attr = {"l4": "W_t4", "l2": "W_t2", "l3": "W_t3", "l6": "W_t6"}
        b_attr = {"l4": "bias_l4", "l2": "bias_l2", "l3": "bias_l3", "l6": "bias_l6"}
        src_layer = {"l4": "l0", "l2": "l4", "l3": "l2", "l6": "l5"}

        bound = self.cfg.active_size_lower_bound
        frac = self.cfg.prune_fraction
        dprob = self.cfg.death_probation

        expired_flags: dict[str, torch.Tensor | None] = {}
        for layer in layers:
            active = self.active_size[layer]
            dr = self._death_row.get(layer)
            pc = self._probation_counter.get(layer)
            if dr is None or pc is None:
                expired_flags[layer] = None
                continue
            dr_a = dr[:active]
            pc_a = pc[:active]
            in_death = dr_a.bool()
            expired = in_death & (pc_a >= dprob)
            W = getattr(self, W_attr[layer])
            rn = W[:active].data.norm(dim=1)
            revived = in_death & (rn > self.cfg.death_threshold) & ~expired
            if revived.any():
                dr_a[revived] = False
                pc_a[revived] = 0
            expired_flags[layer] = expired if expired.any() else None

        new_death: dict[str, torch.Tensor | None] = {}
        n_alive_map: dict[str, int] = {}

        for layer in layers:
            active = self.active_size[layer]
            if active <= bound:
                n_alive_map[layer] = active
                new_death[layer] = None
                continue

            W = getattr(self, W_attr[layer])
            rn = W[:active].data.norm(dim=1)

            n_candidate = max(1, int(active * frac))
            _, dead_ix = rn.topk(n_candidate, dim=-1, largest=False)
            candidate_mask = torch.zeros(active, dtype=torch.bool, device=rn.device)
            candidate_mask[dead_ix] = True

            expired = expired_flags.get(layer)
            if expired is not None:
                candidate_mask = candidate_mask & ~expired

            alive_mask = ~candidate_mask
            if expired is not None:
                alive_mask = alive_mask & ~expired

            n_alive = max(bound, (int(alive_mask.sum().item()) // 8) * 8)
            n_alive = max(n_alive, active - n_candidate)
            n_alive = min(n_alive, active)

            if n_alive >= active:
                n_alive_map[layer] = active
                new_death[layer] = None
                continue

            keep = torch.where(alive_mask)[0]
            probation = torch.where(candidate_mask)[0]
            if expired is not None:
                dead = torch.where(expired)[0]
            else:
                dead = torch.zeros(0, dtype=torch.long, device=rn.device)
            perm = torch.cat([keep, probation, dead])

            dr = self._death_row.get(layer)
            pc = self._probation_counter.get(layer)
            new_dr = torch.zeros(active, dtype=torch.int8, device=rn.device) if dr is None else dr[:active].clone()
            new_pc = torch.zeros(active, dtype=torch.int16, device=rn.device) if pc is None else pc[:active].clone()

            if expired is not None:
                new_dr[expired] = 0
                new_pc[expired] = 0
            new_dr[probation] = 1
            new_pc[probation] = 0

            W.data = W.data[perm].contiguous()
            W_t = getattr(self, t_attr[layer])
            W_t.data = W_t.data[perm][:, perm].contiguous()
            b = getattr(self, b_attr[layer])
            b.data = b.data[perm].contiguous()

            self._death_row[layer] = new_dr[perm]
            self._probation_counter[layer] = new_pc[perm]

            n_alive_map[layer] = n_alive
            new_death[layer] = None

        # -- 列同步 & 显存回收 --
        up_size = {"l0": self.cfg.d_input}
        for layer in layers:
            n_alive = n_alive_map.get(layer, self.active_size[layer])
            src = src_layer[layer]
            src_n = up_size.get(src, self.active_size.get(src, self.cfg.dims()[src]))
            if n_alive >= self.active_size[layer] and \
               getattr(self, W_attr[layer]).shape[1] == src_n:
                up_size[layer] = self.active_size[layer]
                continue

            W = getattr(self, W_attr[layer])
            old = W.data
            setattr(self, W_attr[layer], nn.Parameter(old[:n_alive, :src_n].contiguous()))
            del old

            W_t = getattr(self, t_attr[layer])
            old_t = W_t.data
            setattr(self, t_attr[layer], nn.Parameter(old_t[:n_alive, :n_alive].contiguous()))
            del old_t

            b = getattr(self, b_attr[layer])
            old_b = b.data
            setattr(self, b_attr[layer], nn.Parameter(old_b[:n_alive].contiguous()))
            del old_b

            if self._death_row[layer] is not None:
                self._death_row[layer] = self._death_row[layer][:n_alive]
                self._probation_counter[layer] = self._probation_counter[layer][:n_alive]

            self.active_size[layer] = n_alive
            up_size[layer] = n_alive

        # W_diff 列/行同步: 行 = L4, 列 = L5 (L5 微柱不修剪, 保持 n_blocks*b5 固定)
        old_fut = self.W_diff.data
        self.W_diff = nn.Parameter(old_fut[:self.active_size["l4"], :self.active_size["l5"]].contiguous())
        del old_fut

    def save(self, path: str):
        """保存模型权重."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, config: DensePCConfig | None = None) -> DensePCNet:
        """加载模型权重."""
        net = cls(config or DensePCConfig())
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return net
