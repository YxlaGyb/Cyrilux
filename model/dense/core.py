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

from ..modulation import compute_ach_gain, compute_dopamine_gain, soft_norm_preserve


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
        # ── 微柱阵列: L5 拆 4 独立列块, 块间不共享 Hebbian/Oja ──
        self.n_blocks = self.cfg.l5_blocks
        self.b5 = d["l5"] // self.n_blocks  # 每块维度
        self.W_35 = nn.ParameterList([
            nn.Parameter(torch.empty(self.b5, d["l3"], dtype=torch.float16))
            for _ in range(self.n_blocks)
        ])
        self.W_56 = nn.Parameter(torch.empty(d["l6"], d["l5"], dtype=torch.float16))

        # ── 世界模型 (增量预测): W_diff 在 a5 空间预测 dz5, b_diff 补零均值生路 ──
        self.W_diff = nn.Parameter(torch.empty(d["l5"], d["l5"], dtype=torch.float16))
        self.b_diff = nn.Parameter(torch.zeros(d["l5"], dtype=torch.float16))

        # ── 多尺度软加权时间窗 (2/4/8 并行因果卷积) ──
        # 软权重按各尺度 EMA 误差自适应; 4 步时间窗环形缓冲保留误差记忆
        self.register_buffer("_w_soft", torch.tensor([0.1, 0.8, 0.1], dtype=torch.float16))
        self.register_buffer("_e_ema_2", torch.tensor(0.05, dtype=torch.float16))
        self.register_buffer("_e_ema_4", torch.tensor(0.05, dtype=torch.float16))
        self.register_buffer("_e_ema_8", torch.tensor(0.05, dtype=torch.float16))
        for i in range(4):
            self.register_buffer(f"_dw_buf_{i}", torch.zeros(d["l5"], d["l5"], dtype=torch.float16))
        self._buf_i = 0
        # W_diff 独立 BCM 滑阈 (防指数爆炸)
        self.register_buffer("_theta_w", torch.full((d["l5"],), 0.01, dtype=torch.float16))

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

        # ── 神经调制与竞争机制 ──
        self.register_buffer("_surprise_buf", torch.tensor(1.0, dtype=torch.float16))  # 惊喜基线
        # BCM 滑阈 (替代 Oja): theta = EMA(eps²), phi = eps(eps-theta)
        for ln, dim in (("l4", d["l4"]), ("l2", d["l2"]), ("l3", d["l3"]),
                        ("l5", d["l5"]), ("l6", d["l6"])):
            self.register_buffer(f"_theta_{ln}", torch.full((dim,), 0.01, dtype=torch.float16))
        self.register_buffer("_theta_diff", torch.full((d["l4"],), 0.01, dtype=torch.float16))
        # Foldiak 反赫布侧抑制 (L5 去相关): M 协方差, 零对角线
        self.M_l5 = nn.Parameter(torch.zeros(d["l5"], d["l5"], dtype=torch.float16))
        # 固定掩码 (只切 dW 更新路径): 10% 突触剪切 + [0.5,1.5] 随机增益播种
        for b in range(self.n_blocks):
            self.register_buffer(f"_syn_mask_{b}",
                                 (torch.rand(self.b5, d["l3"]) > 0.1).to(torch.float16))
        for b in range(self.n_blocks):
            self.register_buffer(f"_gain_mask_{b}",
                                 (0.5 + torch.rand(self.b5, d["l3"])).to(torch.float16))
        # L3 种子: W_23 固定随机增益掩码 (上游扰动级联到 L5)
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
            a3 = self.active_size["l3"]
            a2 = self.active_size["l2"]
            W_diff_a = self.W_diff[:a5, :a5]
            # L5 增量预测 (变化量 dz5, a5 空间自洽)
            dz5_g = self._z5_raw[:, 1:] - self._z5_raw[:, :-1]
            dz5_g = dz5_g / (dz5_g.norm(dim=-1, keepdim=True) + 1e-3)
            mu_diff = dz5_g @ W_diff_a.T + self.b_diff[:a5].unsqueeze(0)  # [1,S-1,a5]
            # 自上而下生成路径: a5→a3→a2→a4 (微柱拼接的逆: 各微柱转置拼回 a5 列)
            z5_td = torch.cat([self.W_35[b][:self.b5].T for b in range(self.n_blocks)], dim=1)
            z5_td = z5_td.T  # [a5, a3]
            inj_a4 = mu_diff @ z5_td  # [1,S-1,a5]@[a5,a3] → [1,S-1,a3]
            inj_a4 = inj_a4 @ self.W_23[:a3].T  # a3→a2
            inj_a4 = inj_a4 @ self.W_42[:a2]  # a2→a4 (W_42 是 [a2,a4])
            z4_gen = self._z4[:, 1:] + 1.0 * inj_a4  # α=1.0 增量预测注入
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
        """核心前馈: 感知 (L0→L6) + 微柱路由 + Foldiak 去相关 + 去中心化 + 增量预测.

        机制:
        - 无位置编码, 时序全靠学习的时间核 W_t 递归
        - 微柱阵列 (L5 拆 4 块, 时间步交错路由), 输出经 Foldiak 去相关后去中心化
          (z5 去均值喂下游保持 PR; z5_raw 原始幅度喂 W_diff 增量预测, 路由分离)
        - ACh 噪声注入 L3 投影输入, is_inference=True 时关闭 (推理确定性)
        """
        N, S = byte_ids.shape
        dev = byte_ids.device
        a4, a2, a3, a5, a6 = (self.active_size[k] for k in ("l4", "l2", "l3", "l5", "l6"))

        W_04_a = self.W_04[:a4]
        W_42_a = self.W_42[:a2]
        W_23_a = self.W_23[:a3]
        W_56_a = self.W_56[:a6]
        W_diff_a = self.W_diff[:a5, :a5]

        # L0 纯 one-hot
        z0 = F.one_hot(byte_ids, num_classes=256).to(torch.float16)  # [N,S,256]

        mu4 = z0 @ W_04_a.T + self.bias_l4[:a4]
        z4 = self._recurrent(mu4, dev, self.W_t4[:a4, :a4])
        mu2 = z4 @ W_42_a.T + self.bias_l2[:a2]
        z2 = self._recurrent(mu2, dev, self.W_t2[:a2, :a2])
        mu3 = z2 @ W_23_a.T + self.bias_l3[:a3]
        if not is_inference:
            mu3 = mu3 + torch.sign(2.0 * (torch.rand_like(mu3) - 0.5)) * 0.03  # ACh 噪声
        z3 = self._recurrent(mu3, dev, self.W_t3[:a3, :a3])
        # 微柱路由: 微柱 b 处理时间步 {b, b+4, ...}, 输出到 z5 的 [b*b5:(b+1)*b5] 切片
        z5 = torch.zeros(N, S, a5, dtype=torch.float16, device=dev)
        for b in range(self.n_blocks):
            z3_route = z3[:, b::self.n_blocks, :]  # [N, S//n, a3] 该微柱的时间步子集
            zb = z3_route @ self.W_35[b].T + self.bias_l5[b * self.b5:(b + 1) * self.b5]
            z5[:, b::self.n_blocks, b * self.b5:(b + 1) * self.b5] = zb
        # Foldiak 去相关 + 路由分离: z5_raw 原始幅度喂 W_diff, z5 去中心化喂下游
        z5_fd = z5 - 0.2 * (self.M_l5[:a5, :a5] @ z5.transpose(-2, -1)).transpose(-2, -1)
        z5_raw = z5_fd
        z5 = z5_fd - z5_fd.mean(dim=-1, keepdim=True)
        mu6 = z5 @ W_56_a.T + self.bias_l6[:a6]
        z6 = self._recurrent(mu6, dev, self.W_t6[:a6, :a6])

        # 增量预测: W_diff 在 a5 空间预测 dz5 = z5_raw[t] - z5_raw[t-1]
        dz5_pred = z5_raw[:, 1:] - z5_raw[:, :-1]
        dz5_pred = dz5_pred / (dz5_pred.norm(dim=-1, keepdim=True) + 1e-3)  # RMS 归一化
        dz5_pad = torch.cat([dz5_pred, torch.zeros(N, 1, a5, dtype=z5_raw.dtype, device=dev)], dim=1)
        bd = self.b_diff[:a5].unsqueeze(0).unsqueeze(0)
        mu_diff = dz5_pad @ W_diff_a.T + bd
        diff_err = (dz5_pred - mu_diff[:, :-1]).square().mean()  # 监控用

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
        """Hebbian 学习 (零反传, 零误差回路). 不接收 targets.

        机制:
        - 前馈权重: 逐层预测误差驱动 (标准 PC 自下而上), L3 加随机增益+门控种子
        - 微柱 W_35: 块内 BCM 滑阈 + 样本显著性加权 + 增益/剪切/门控掩码
        - W_diff: 增量预测 (dz5 = z5[t]-z5[t-1]), 多尺度软窗 + 4 步时间窗 + 独立 BCM
        - 时序 W_t: 共现 Hebbian, ACh 调制; 无 Oja, BCM 承担稳定性
        - 精度调度: 多巴胺/ACh 调制全局学习率, 前 50 步 W_diff 减半

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
        W_diff_a = self.W_diff[:a5, :a5]

        z0 = self._z0
        z4, z2, z3, z5, z6 = self._z4, self._z2, self._z3, self._z5, self._z6

        # 逐层预测误差 (自下而上 PC); L5 用时序差分误差, L6 用时间自预测
        eps4 = z4 - (z0 @ self.W_04[:a4].T + self.bias_l4[:a4])
        eps2 = z2 - (z4 @ self.W_42[:a2].T + self.bias_l2[:a2])
        eps3 = z3 - (z2 @ W_23_a.T + self.bias_l3[:a3])
        z6_pre = torch.cat([torch.zeros(N, 1, a6, dtype=z6.dtype, device=dev), z6[:, :-1]], dim=1)
        eps6 = z6 - z6_pre
        eps5_td = z5[:, 1:] - z5[:, :-1]  # L5: 跨时刻变化 z5[t]-z5[t-1]

        def _rms(x):
            rms = x.square().mean(dim=-1, keepdim=True)
            alive = (rms > 1e-8).to(x.dtype)
            return x * alive / (rms + 1e-4).sqrt()

        # 增量预测: pred = W_diff @ dz5, e = dz5 - pred, dW = dz5.T @ e
        dz5 = self._z5_raw[:, 1:] - self._z5_raw[:, :-1]
        dz5 = dz5 / (dz5.norm(dim=-1, keepdim=True) + 1e-3)  # RMS 归一化
        # 多尺度软加权时间窗 (2/4/8 并行因果卷积): z5[t-k] 预测 dz5[t],
        # 各尺度误差 EMA 平滑后软加权; K=8 前 7 步零填充 Mask 掉防伪影
        z5r = self._z5_raw  # [N,S,a5]
        S_full = z5r.shape[1]
        masks = {}
        preds_k = {}
        errs_k = {}
        for k, kname in ((2, "2"), (4, "4"), (8, "8")):
            z_shift = torch.cat([torch.zeros(N, k, a5, dtype=z5r.dtype, device=dev), z5r[:, :-k]], dim=1)
            z_shift_n = z_shift / (z_shift.norm(dim=-1, keepdim=True) + 1e-3)
            pred_k = z_shift_n @ W_diff_a.T + self.b_diff[:a5]
            err_k = (dz5 - pred_k[:, :-1]).square().mean()
            if k == 8:
                valid = torch.arange(S_full - 1, device=dev) >= 7
                err_k = ((dz5[:, valid] - pred_k[:, :-1][:, valid]).square()).mean()
            masks[kname] = valid if k == 8 else None
            preds_k[kname] = pred_k
            errs_k[kname] = err_k
        # EMA 平滑各尺度误差 (α=0.9, 时间常数~10步)
        e2, e4, e8 = (errs_k["2"], errs_k["4"], errs_k["8"])
        ema2 = self._e_ema_2.mul_(0.1).add_(0.9 * e2)
        ema4 = self._e_ema_4.mul_(0.1).add_(0.9 * e4)
        ema8 = self._e_ema_8.mul_(0.1).add_(0.9 * e8)
        # 软权重: w_k = (1/(e_ema_k+1e-3)) / Σ (加性保护非 clamp)
        inv2, inv4, inv8 = (1.0 / (ema2 + 1e-3)), (1.0 / (ema4 + 1e-3)), (1.0 / (ema8 + 1e-3))
        sum_inv = inv2 + inv4 + inv8
        w2, w4, w8 = inv2 / sum_inv, inv4 / sum_inv, inv8 / sum_inv
        self._w_soft[0] = w2
        self._w_soft[1] = w4
        self._w_soft[2] = w8
        pred_d = (w2 * preds_k["2"][:, :-1] + w4 * preds_k["4"][:, :-1] + w8 * preds_k["8"][:, :-1])
        # 误差用掩码后的有效区 (K=8 前 7 步剔除)
        if masks["8"] is not None:
            e_t = (dz5[:, masks["8"]] - pred_d[:, masks["8"]]).detach()
            e_t_all = dz5 - pred_d
        else:
            e_t = dz5 - pred_d
            e_t_all = e_t
        # W_diff BCM 滑阈: theta_w = EMA(pred²), phi_w 衰减高活跃预测方向
        th_w = self._theta_w[:a5]
        th_w.mul_(0.975).add_(0.025 * (pred_d * pred_d).mean(dim=(0, 1)))
        phi_w = pred_d * (pred_d - th_w)
        # 差动赫布外积: dW = dz5.T @ (e - 0.1*phi_w), 零变化→零更新
        e_mod = e_t_all - 0.1 * phi_w
        dW_diff_t = torch.bmm(
            e_mod.transpose(-2, -1), dz5
        ).mean(dim=0) * (1.0 / (S - 1))
        # 4 步时间窗环形缓冲: 更新用最近 4 步平均外积 (保留误差记忆)
        buf = getattr(self, f"_dw_buf_{self._buf_i}")
        buf.copy_(dW_diff_t)
        dW_avg = buf.clone()
        for i in range(1, 4):
            dW_avg = dW_avg + getattr(self, f"_dw_buf_{(self._buf_i - i) % 4}")
        dW_avg = dW_avg * 0.25
        self._buf_i = (self._buf_i + 1) % 4

        # 精度调度: 多巴胺/ACh 调制学习率; 前 50 步 W_diff 学习率减半 (先稳后放)
        surprise = (eps4.square().mean() + eps2.square().mean() + eps3.square().mean()
                    + eps5_td.square().mean() + eps6.square().mean()) * 0.2
        rel = float((surprise / (self._surprise_buf + 1e-4)).detach())
        self._surprise_buf.data.mul_(0.95).add_(0.05 * surprise.data)
        dop_gain = torch.tensor(compute_dopamine_gain(rel, 0.3, 5.0), dtype=torch.float16, device=dev)
        ach_gain = torch.tensor(compute_ach_gain(rel, 3.0), dtype=torch.float16, device=dev)
        eta = self.cfg.lr_hebbian * dop_gain
        if self._step_counter < 50:
            eta = eta * 0.5
        eta_t = eta * self.cfg.temporal_lr_ratio * ach_gain

        # Hebbian 外积 (逐层误差 ⊗ pre 活动)
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

        # L3 种子: W_23 随机增益 + 误差门控 (上游扰动级联到 L5 分散)
        # _gain_l3 是固定 [384, 384] 种子, L3 修剪后行数收缩, 需按当前活性行切片
        gain_l3 = self._gain_l3[:a3, :a3] if a3 < 384 else self._gain_l3[:a3, :]
        dW23 = (eps3_p.transpose(-2, -1) @ _rms(z2)).mean(dim=0) * inv_s
        err3_norm = eps3.pow(2).mean(dim=(0, 1)).sqrt() + 1e-8  # [a3] 每神经元
        gate3 = 0.1 + 0.9 * (err3_norm / err3_norm.max())
        dW23 = dW23 * gain_l3 * gate3.unsqueeze(1)
        c3_mask = torch.rand(a3, 1, device=dev) < self.cfg.column_dropout
        self.W_23[:a3].data += (dW23 * (~c3_mask).to(torch.float16)) * eta
        # 软范数保持 (0.8-1.2): 增益种子幅度差异保留, 权重有界防溢出
        soft_norm_preserve(self.W_23[:a3].data)

        # ── 第一步: 换轴 — 资格迹+样本竞争 (bmm 逐样本, 显著性加权, 不批平均) ──
        # dW_n = bmm(eps^T, z) [N, d_out, d_in]; gate_n = surprise_n / Σ surprise_n
        # 线性加权 (禁止 exp/非线性), 保留样本差异, 抹平效应消除
        n_sub = 4
        sub = max(1, N // n_sub)
        for b in range(self.n_blocks):
            b_s = slice(b * self.b5, (b + 1) * self.b5)
            z5_b = z5[:, b::self.n_blocks, b_s]  # 该块路由步的激活
            eps_b = z5_b[:, 1:] - z5_b[:, :-1]  # 块内时序差分误差
            z3_b = z3[:, b::self.n_blocks, :][:, :-1]
            z3_bp = z3_b * (torch.rand_like(z3_b) > 0.3).to(torch.float16)
            # BCM 滑阈: theta = EMA(eps²), phi = eps(eps-theta); 先 RMS 归一化 eps,
            # 防 eps 平方级增长在 fp16 溢出 (16774 步首爆 W_35[2] 根因)
            eps_b = _rms(eps_b)
            th = self._theta_l5[b * self.b5:(b + 1) * self.b5]
            e2 = (eps_b * eps_b).mean(dim=(0, 1))
            th.mul_(0.975).add_(0.025 * e2)
            phi_b = eps_b * (eps_b - th)
            # 样本显著性: gate_n = surprise_n / Σ surprise_n (线性加权)
            g_n = (phi_b * phi_b).mean(dim=(1, 2)) + 1e-8
            g_n = g_n / g_n.sum()
            Wb = self.W_35[b].data
            # 固定 10% 突触剪切 / 静态随机增益 [0.5,1.5]; 随 L3 修剪裁列
            syn_mask = getattr(self, f"_syn_mask_{b}")[:, :a3]
            gain_mask = getattr(self, f"_gain_mask_{b}")[:, :a3]
            # 误差门控: 高误差神经元主导更新, 低误差保 10% 下限
            err_norm = eps_b.pow(2).mean(dim=(0, 1)).sqrt() + 1e-8
            upd_gate = 0.1 + 0.9 * (err_norm / err_norm.max())
            for s in range(n_sub):
                sl = slice(s * sub, (s + 1) * sub)
                dW_n = torch.bmm(phi_b[sl].transpose(-2, -1), _rms(z3_bp[sl]))
                dW_sub = (dW_n * g_n[sl, None, None]).sum(dim=0) * (1.0 / (S // self.n_blocks - 1))
                dW_sub = dW_sub * gain_mask * syn_mask * upd_gate.unsqueeze(1)
                b_mask = torch.rand(self.b5, 1, device=dev) < self.cfg.column_dropout
                Wb += (dW_sub * (~b_mask).to(torch.float16)) * eta
                # 软范数保持 (0.8-1.2): W_35 微柱无 Oja, 长训累积溢出 fp16 → NaN
                # (16774 步首爆 W_35[2]); 幅度差异保留 (结构化非 clamp)
                soft_norm_preserve(Wb)

        # Foldiak 反赫布: 侧向矩阵 M 协方差去相关 (零对角线)
        # dM 先除范数平方再平均: cov 元素 = z_i·z_j 内积, z5 幅度 ~13 时 256 项和可超
        # fp16 上限 65504 (trace 实测 z5~12 时 cov 已 209; z5 稍大即溢出). 归一化到
        # ~0(1) 再积分, 消除溢出路径; M 行范数受 0.8-1.2 保持约束不长爆
        z5_flat = z5.reshape(-1, a5)
        dM = (z5_flat.transpose(0, 1) @ z5_flat)
        dM = dM / (dM.norm() + 1e-3)
        eye_mask = (1.0 - torch.eye(a5, device=dev, dtype=torch.float16))
        self.M_l5[:a5, :a5].data += (dM * eye_mask) * (0.005 * ach_gain) * eta
        soft_norm_preserve(self.M_l5[:a5, :a5].data)

        # W_diff 增量预测更新 (4 步时间窗平均外积) + b_diff 偏置
        fut_mask = torch.rand(a5, 1, device=dev) < self.cfg.column_dropout
        W_diff_a.data += (dW_avg * (~fut_mask).to(torch.float16)) * eta
        future_e = (dz5 - pred_d).mean(dim=(0, 1))
        self.b_diff[:a5].data += future_e * eta

        # 时序 Hebbian (W_t 学习, 高确定性时增强 → 记忆巩固)
        for (z_cur, W_t), a_sz in zip(
            [(z4, self.W_t4), (z2, self.W_t2), (z3, self.W_t3), (z5, self.W_t5), (z6, self.W_t6)],
            a_sizes
        ):
            pre = z_cur[:, :-1]
            post = z_cur[:, 1:]
            dW_t = (_rms(pre).transpose(-2, -1) @ _rms(post)).mean(dim=0) * (1.0 / (S - 1))
            W_t[:a_sz, :a_sz].data += dW_t * eta_t
            # 软范数保持 (0.8-1.2): 权重有界防 fp16 溢出 (5万步 NaN 根因)
            soft_norm_preserve(W_t[:a_sz, :a_sz].data)

        # 前馈权重软范数保持 (0.8-1.2): W_04/W_42/W_56 无 BCM 约束,
        # 长训累积溢出 fp16 → NaN; 幅度差异保留 (结构化非 clamp)
        for W, a_sz in zip([self.W_04, self.W_42, self.W_56], [a4, a2, a6]):
            soft_norm_preserve(W[:a_sz].data)
        # W_diff 同款软范数保持 (行范数)
        soft_norm_preserve(W_diff_a.data)

        # ── 拓扑重塑 ──
        self._step_counter += 1
        if self._step_counter > self.cfg.prune_warmup and self._step_counter % self.cfg.prune_interval == 0:
            self._prune()

        stats = {
            "free_energy": (eps4.square().mean() + eps2.square().mean() + eps3.square().mean()
                            + eps5_td.square().mean() + eps6.square().mean()),
            "future_err": (dz5 - pred_d).square().mean(),
            "surprise": surprise,
            "dop_gain": dop_gain,
            "ach_gain": ach_gain,
        }
        # 释放每步状态引用 (显存按需): _z* 是 store_state 存的大张量,
        # 不释放则 caching allocator 无法复用, 4GB 卡上逐步累积到 OOM
        for k in ("_z0", "_z4", "_z2", "_z3", "_z5", "_z5_raw", "_z6"):
            if hasattr(self, k):
                delattr(self, k)
        return stats

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

            n_alive = max(bound, (int(alive_mask.sum().detach().item()) // 8) * 8)
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
            # l0 已在 up_size 定义; 其余源层在 active_size 中 (dims() 无 l0)
            src_n = up_size.get(src, self.active_size.get(src, self.cfg.d_input))
            if n_alive >= self.active_size[layer] and \
               getattr(self, W_attr[layer]).shape[1] == src_n:
                up_size[layer] = self.active_size[layer]
                continue

            W = getattr(self, W_attr[layer])
            old = W.data
            setattr(self, W_attr[layer], nn.Parameter(old[:n_alive, :src_n].contiguous()))
            del old

            # W_35 微柱输入维 = L3 活性维: L3 修剪后同步 W_35 列数, 否则路由 matmul 形状崩
            # (6000 步冒烟实测 z3_route[.., 365] @ W_35[b].T[384, 64] 形状不匹配)
            if layer == "l3":
                for b in range(self.n_blocks):
                    old_b = self.W_35[b].data
                    self.W_35[b] = nn.Parameter(old_b[:, :src_n].contiguous())
                    del old_b

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

        # W_diff 是 a5 空间方阵 (行=输出维 a5, 列=输入维 a5), 与 L4 活性无关;
        # 仅当 L5 被修剪时同步列 (W_diff 行数不随 L4 裁, 否则 matmul 输出维 256 配 bd 252 崩)
        if self.active_size["l5"] < self.cfg.d_l5:
            old_fut = self.W_diff.data
            self.W_diff = nn.Parameter(old_fut[:self.active_size["l5"], :self.active_size["l5"]].contiguous())
            del old_fut

        # W_56/W_t5 列同步: L5 修剪后 W_56 输入维 = W_t5 方阵维 = 活性 L5
        if self.active_size["l5"] < self.cfg.d_l5:
            old = self.W_56.data
            self.W_56 = nn.Parameter(old[:self.active_size["l6"], :self.active_size["l5"]].contiguous())
            del old
            old_t5 = self.W_t5.data
            self.W_t5 = nn.Parameter(old_t5[:self.active_size["l5"], :self.active_size["l5"]].contiguous())
            del old_t5

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
