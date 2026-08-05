"""ForwardEngine

密集 PPA 前馈/推理/生成/稀疏绑定.

感知: L0(纯 one-hot, 时序双通道) → L4 → L2 → L3 → L5(微柱阵列) → L6
时序: 每层时间核 W_t 递归 z[t] 依赖 z[t-1]
生成: L4 状态 + W_diff 预测差分 → W_lm 解码字节
绑定: z5 → W_bind → bind_dim, top-k WTA 稀疏化

全 fp16, 零 .float(), 零反向传播.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .network import DensePCNet


def _rms(x: torch.Tensor) -> torch.Tensor:
    """零向量保护 RMS 归一化 (fp16 下 1e-8 舍入为 0, 需掩码保护分母)."""
    rms = x.square().mean(dim=-1, keepdim=True)
    alive = (rms > 1e-8).to(x.dtype)
    denom = torch.where(alive > 0, (rms * 1.01).sqrt(), torch.ones_like(rms))
    return x * alive / denom


def _l2_norm(x: torch.Tensor) -> torch.Tensor:
    """零向量保护 L2 范数归一化 (前馈投影前的原语义, 与 _rms 量纲不同)."""
    nrm = x.norm(dim=-1, keepdim=True)
    alive = (nrm > 1e-8).to(x.dtype)
    denom = torch.where(alive > 0, nrm * 1.01, torch.ones_like(nrm))
    return x * alive / denom


class ForwardEngine:
    """前馈引擎: 持 net 引用, 状态经 net._z*/net.active_size 传递."""

    def __init__(self, net: DensePCNet):
        self.net = net

    def forward(self, byte_ids: torch.Tensor) -> dict:
        """推理前馈: 返回未来预测偏差.  ACh 关闭, 确定性."""
        return self._predict(byte_ids, store_state=True, is_inference=True)

    def generate(
        self, prompt: str, n_tokens: int = 40, temperature: float = 0.7, dev: torch.device | None = None
    ) -> bytes:
        """行动: L4 状态 + 预测差分 → W_lm 解码生成字节.

        生成端: z4_next = z4 + pred_delta (pred_delta = z4 @ W_diff),
        mu0 = z4_next @ W_lm → 256 字节 logits (W_lm 自监督赫布头, 非重建 W_04).
        """
        net = self.net
        if dev is None:
            dev = next(net.parameters()).device
        gen = list(prompt.encode("utf-8"))
        for _ in range(n_tokens):
            bv = torch.tensor([gen[-64:]], dtype=torch.long, device=dev)
            _ = self._predict(bv, store_state=True, is_inference=True)
            a4 = net.active_size["l4"]
            W_diff_a = net.W_diff[:a4, :a4]
            # 下一状态预测: pred_delta = z4 @ W_diff, z4_next = z4 + pred_delta
            z4_n = net._z4 / (net._z4.norm(dim=-1, keepdim=True) + 1e-3)
            pred_delta = z4_n @ W_diff_a.T + net.b_diff[:a4].unsqueeze(0).unsqueeze(0)
            z4_next = net._z4 + pred_delta
            zh_next = torch.cat([z4_next, net._h], dim=-1)  # 工作记忆拼接
            mu0_top = zh_next @ net.W_lm[:2 * a4] + net.bias_lm  # W_lm 解码
            last = mu0_top[0, -1].float() / temperature
            topv, _ = torch.topk(last, min(15, 256))
            last[last < topv[-1]] = -float("inf")
            probs = torch.softmax(last, dim=-1)
            gen.append(int(torch.multinomial(probs, 1).item()))
        return bytes(gen)

    def _recurrent(
        self, mu: torch.Tensor, dev: torch.device, W_t: torch.Tensor, sweeps: int = 1
    ) -> torch.Tensor:
        """时间核递归: z[t] 依赖 z[t-1] (经学习到的 W_t), 纯 matmul 全 fp16.

        因果移位 + 前向卷积式扫描. 只归一化递归项, 不归一化整体 z —
        保留 mu 携带的输入语义区分度 (mu4 跨上下文 cos=0.26, 旧实现递归后
        整体 RMS 归一化把方向抹到 0.985; sweeps=1 单次微扰防递归项累积污染方向).
        """
        N, S, d = mu.shape
        z = mu
        inv_d = 1.0 / math.sqrt(d)
        for _ in range(sweeps):
            shift = torch.cat([torch.zeros(N, 1, d, dtype=z.dtype, device=dev), z[:, :-1]], dim=1)
            shift_n = _rms(shift)
            rec = (shift_n @ W_t.T) * inv_d
            # 递归项单独归一化再乘 inv_d 小扰动: 保 mu 语义方向 (cos 0.69→0.98 抹平),
            # 同时幅度有界防逐层累积 (旧整体 RMS 压幅度但抹方向; 无约束则 58 步 NaN)
            rec = _rms(rec) * inv_d
            z = z + rec
        return z

    def _predict(self, byte_ids: torch.Tensor, store_state: bool = True, is_inference: bool = False) -> dict:
        """核心前馈: 感知 (L0→L6) + 微柱路由 + Foldiak 去相关 + 去中心化 + 增量预测.

        机制:
        - 无位置编码, 时序全靠学习的时间核 W_t 递归
        - 微柱阵列 (L5 拆 4 块, 时间步交错路由), 输出经 Foldiak 去相关后去中心化
          (z5 去均值喂下游保持 PR; z5_raw 原始幅度喂 W_diff 增量预测, 路由分离)
        - ACh 噪声注入 L3 投影输入, is_inference=True 时关闭 (推理确定性)
        """
        net = self.net
        N, S = byte_ids.shape
        dev = byte_ids.device
        a4, a2, a3, a5, a6 = (net.active_size[k] for k in ("l4", "l2", "l3", "l5", "l6"))

        W_04_a = net.W_04[:a4]
        W_42_a = net.W_42[:a2]
        W_23_a = net.W_23[:a3]
        W_56_a = net.W_56[:a6]
        W_diff_a = net.W_diff[:a4, :a4]

        # L0 纯 one-hot; 时序双通道: [z0[t], z0[t-1]] 拼接 (词序信息进入表示层)
        z0 = F.one_hot(byte_ids, num_classes=256).to(torch.float16)  # [N,S,256]
        if net.cfg.input_history:
            z0_prev = torch.cat([torch.zeros(N, 1, 256, dtype=z0.dtype, device=dev), z0[:, :-1]], dim=1)
            z0 = torch.cat([z0, z0_prev], dim=-1)  # [N,S,512]

        mu4 = z0 @ W_04_a.T + net.bias_l4[:a4]
        z4 = self._recurrent(mu4, dev, net.W_t4[:a4, :a4])
        z4_n = _l2_norm(z4)
        mu2 = z4_n @ W_42_a.T + net.bias_l2[:a2]
        z2 = self._recurrent(mu2, dev, net.W_t2[:a2, :a2])
        # 预投影 RMSNorm (CLAUDE.md 铁律: 投影前加 RMSNorm 防 fp16 溢出):
        # 只归一化输入, 保方向压尖峰; 不归一化输出 mu, 不抹平差异
        z2_n = _l2_norm(z2)
        mu3 = z2_n @ W_23_a.T + net.bias_l3[:a3]
        if not is_inference:
            mu3 = mu3 + torch.sign(2.0 * (torch.rand_like(mu3) - 0.5)) * 0.03  # ACh 噪声
        z3 = self._recurrent(mu3, dev, net.W_t3[:a3, :a3])
        # 微柱路由: 微柱 b 处理时间步 {b, b+4, ...}, 输出到 z5 的 [b*b5:(b+1)*b5] 切片
        z5 = torch.zeros(N, S, a5, dtype=torch.float16, device=dev)
        z3_n = _l2_norm(z3)
        for b in range(net.n_blocks):
            z3_route = z3_n[:, b :: net.n_blocks, :]  # [N, S//n, a3] 该微柱的时间步子集
            zb = z3_route @ net.W_35[b].T + net.bias_l5[b * net.b5 : (b + 1) * net.b5]
            z5[:, b :: net.n_blocks, b * net.b5 : (b + 1) * net.b5] = zb
        # Foldiak 去相关 + 路由分离: z5_raw 原始幅度喂 W_diff, z5 去中心化喂下游
        z5_fd = z5 - 0.2 * (net.M_l5[:a5, :a5] @ z5.transpose(-2, -1)).transpose(-2, -1)
        z5_raw = z5_fd
        # 空间软竞争 (微柱级): 各微柱独立算时序差分误差 (与 learn 同口径),
        # 误差软化倒数加权 — 误差小的柱权重放大, 误差大的柱被抑制但不清零.
        # w = 1/(1 + err·scale): 动态范围有限, 坏柱权重下限 >0, 防正反馈崩溃 NaN.
        # 打破 4 柱同权均质化; 与多尺度时间窗软加权同构.
        w_sp = torch.ones(net.n_blocks, dtype=torch.float16, device=dev)
        if not is_inference:
            for b in range(net.n_blocks):
                b_s = slice(b * net.b5, (b + 1) * net.b5)
                # 用缩放前的原始 z5 (Foldiak 后), 避免原地缩放污染后续块误差口径
                zb = z5[..., b_s][:, b :: net.n_blocks]
                epsb = zb[:, 1:] - zb[:, :-1]
                # L1 平均误差防平方溢出 (fp16), 误差大的柱权重小
                err_b = epsb.abs().mean() * 1.01
                w_sp[b] = 1.0 / (1.0 + err_b * 4.0)
            # 只抑制不放大 (w_sp ≤ 1): 坏柱缩小, 好柱保持, 幅度只减不增防溢出
            for b in range(net.n_blocks):
                b_s = slice(b * net.b5, (b + 1) * net.b5)
                z5_raw[..., b_s] = z5_raw[..., b_s] * w_sp[b]
        z5 = z5_raw - z5_raw.mean(dim=-1, keepdim=True)
        mu6 = z5 @ W_56_a.T + net.bias_l6[:a6]
        z6 = self._recurrent(mu6, dev, net.W_t6[:a6, :a6])

        # 下一状态预测 (显式预测目标): W_diff 在 L4 空间预测 Δz4 = z4[t] - z4[t-1]
        # target_delta = z4_next - z4; pred_delta = z4 @ W_diff;
        # eps_diff = pred_delta - target_delta (训练误差, 驱动 W_diff 学习动力学)
        dz4_pred = z4[:, 1:] - z4[:, :-1]
        dz4_pred = dz4_pred / (dz4_pred.norm(dim=-1, keepdim=True) + 1e-3)  # RMS 归一化
        z4_prev = z4[:, :-1] / (z4[:, :-1].norm(dim=-1, keepdim=True) + 1e-3)
        z4_prev_pad = torch.cat([torch.zeros(N, 1, a4, dtype=z4.dtype, device=dev), z4_prev], dim=1)
        bd = net.b_diff[:a4].unsqueeze(0).unsqueeze(0)
        mu_diff = z4_prev_pad @ W_diff_a.T + bd
        diff_err = (dz4_pred - mu_diff[:, :-1]).square().mean()  # eps_diff 训练误差

        if store_state:
            net._z0 = z0
            net._z4 = z4
            net._z2 = z2
            net._z3 = z3
            net._z5 = z5
            net._z5_raw = z5_raw
            net._z6 = z6
            # 工作记忆 (海马体式指数积分): h_t = 0.99·h_{t-1} + 0.01·z4_t,
            # batch 内按时间步递推, 跨序列累积低分辨率环境信息 (不改变物理输入层)
            h_t = net._h_mem.unsqueeze(0).unsqueeze(0).expand(N, 1, -1).clone()  # [N,1,a4] 第 0 步 = 上序列末态
            for t in range(1, S):
                h_t = torch.cat([h_t, net._h_alpha * h_t[:, -1:] + 0.01 * z4[:, t:t + 1]], dim=1)
            net._h_mem.copy_((net._h_alpha * h_t[:, -1] + 0.01 * z4[:, -1]).mean(dim=0))
            net._h = h_t  # [N, S, a4] 每步的积分记忆

        # 稀疏绑定 (k-WTA): 连续 z5 经 W_bind 映射到 bind_dim, 只留 top-k 激活
        # "离散符元" = 高维竞争坍缩出的 k 个神经元 ID, 非外部字典
        # bind_mode="none" (直连实验) 时跳过整个绑定层
        if not is_inference and net.cfg.bind_mode != "none":
            self._bind(z5)

        return {
            "mu_diff": mu_diff,
            "diff_err": diff_err,
            "free_energy": diff_err,
        }

    def _bind(self, z5: torch.Tensor) -> None:
        """稀疏绑定层前向: z5 → W_bind → bind_dim, 硬/软 top-k WTA.

        硬 VQ: top-k 置 1 其余 0 (离散符元 ID); 软 VQ: top-k 保留幅度.
        bind_orth=True 时 W_bind 列做 Gram-Schmidt 正交化 (容量实验).
        """
        net = self.net
        a5 = net.active_size["l5"]
        k = net.cfg.bind_k
        z5_n = _l2_norm(z5)
        pre = z5_n @ net.W_bind[:a5]  # [N,S,bind_dim]
        vals, idx = pre.topk(k, dim=-1)
        if net.cfg.bind_mode == "hard":
            sparse = torch.zeros_like(pre)
            sparse.scatter_(-1, idx, torch.ones_like(vals))
        else:  # soft VQ: 保留幅度
            sparse = torch.zeros_like(pre)
            sparse.scatter_(-1, idx, vals)
        net._bind_pre = pre
        net._bind_idx = idx  # [N,S,k] 激活神经元 ID (语义分离度指标)
        net._bind_sparse = sparse

    def _precise(self, eps: torch.Tensor) -> torch.Tensor:
        """精度加权: π_l = 1/(σ_εl + c), 归一化每层误差尺度."""
        s = eps.std() + 1e-3
        return eps / s
