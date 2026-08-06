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
            zh_next = torch.cat([z4_next, net._m2, net._m8, net._m32, net._bind_vec], dim=-1)  # 记忆池+绑定拼接
            inv_h = 1.0 / math.sqrt(4 * a4 + 3 * net.bind_slot_dim)
            mu0_top = (zh_next @ net.W_lm + net.bias_lm) * inv_h  # W_lm 解码 (输出缩放)
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
        z3_n = _l2_norm(z3)
        # L5 统一矩阵 (撤销微柱硬切块): 全量 z3 → 单 W_35 [a5, a3]
        z5 = z3_n @ net.W_35[:a5].T + net.bias_l5[:a5]
        # 路由分离: z5_raw 原始幅度喂 W_diff, z5 去中心化喂下游.
        # Foldiak 反赫布侧抑制 (方案 D): z5 -= 0.2·M@z5, M 零起步 = 恒等;
        # M 学 z_out 协方差 → 白化去相关 → 打破行收敛 ±w 的共线激活
        z5_fd = z5 - 0.2 * (net.M_l5[:a5, :a5] @ z5.transpose(-2, -1)).transpose(-2, -1)
        z5_raw = z5_fd
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
            # 多级记忆池 (3 级因果卷积核, 替代单变量 h): 即时 2 步 / 短时 8 步 /
            # 长时 32 步 — 每级 = 核窗口内 z4 的因果滑动平均 (纯机制, 无超参权重),
            # 位置信息由各级窗口保留, 跨序列经 _m_pool 延续 (零填充 = 接续上序列末态)
            m_prev = net._m_pool.unsqueeze(0).unsqueeze(0).expand(N, 1, -1).clone()  # [N,1,3a4]
            m2 = m_prev[:, :, :a4]
            m8 = m_prev[:, :, a4:2 * a4]
            m32 = m_prev[:, :, 2 * a4:]
            for t in range(1, S):
                zt = z4[:, t:t + 1]  # [N,1,a4]
                m2 = torch.cat([m2, 0.5 * m2[:, -1:] + 0.5 * zt], dim=1)
                m8 = torch.cat([m8, 0.125 * m8[:, -1:] + 0.875 * zt], dim=1)
                m32 = torch.cat([m32, 0.03125 * m32[:, -1:] + 0.96875 * zt], dim=1)
            net._m_pool = torch.cat([m2[:, -1], m8[:, -1], m32[:, -1]], dim=-1).mean(dim=0)  # [3a4]
            net._m2, net._m8, net._m32 = m2, m8, m32  # [N,S,a4] 每级记忆池

        # 稀疏绑定 (角色分离三槽): 连续 z4 → W_bind 三块 → 槽内 top-k WTA 硬稀疏
        # bind_vec = [实体(256) | 角色(256) | 谓语(256)] 拼进 W_lm 输入 (第 5 段);
        # 推理也计算 (生成/评估与训练同构)
        if net.cfg.bind_mode != "none":
            self._bind(z4)

        return {
            "mu_diff": mu_diff,
            "diff_err": diff_err,
            "free_energy": diff_err,
        }

    def _bind(self, z4: torch.Tensor) -> None:
        """角色分离绑定前向: z4 → W_bind 三块 (实体/角色/谓语) → 槽内 top-k WTA.

        每槽 256 维独立投影 (块对角 W_bind), 槽内 top-k 置 1 其余 0 (硬离散符元).
        bind_vec = [槽1|槽2|槽3] 拼接 (3*256), 与 W_lm 输入第 5 段对齐;
        全部 3*256 槽位始终参与计算 (不同于 bind_k 的 top-k 选择数, 那是槽内阈值).
        """
        net = self.net
        a4 = net.active_size["l4"]
        z4_n = _l2_norm(z4)
        pre = z4_n @ net.W_bind[:a4]  # [N,S,768]
        k = net.cfg.bind_k
        bd = net.bind_slot_dim
        sparse = torch.zeros_like(pre)
        for i in range(3):
            sl = slice(i * bd, (i + 1) * bd)
            vals, idx = pre[:, :, sl].topk(k, dim=-1)
            sparse[:, :, sl].scatter_(-1, idx, torch.ones_like(vals))
        net._bind_pre = pre
        net._bind_idx = idx  # 末槽 top-k ID (诊断)
        net._bind_vec = sparse

    def _precise(self, eps: torch.Tensor) -> torch.Tensor:
        """精度加权: π_l = 1/(σ_εl + c), 归一化每层误差尺度."""
        s = eps.std() + 1e-3
        return eps / s
