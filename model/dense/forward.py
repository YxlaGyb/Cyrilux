"""
ForwardEngine
前馈/推理/生成/稀疏绑定.

感知 L0 → L4 → L2 → L3 → L5 → L6; 时序经时间核 W_t 递归;
生成经 W_diff 预测差分 + W_lm 解码; 绑定 z4 → W_bind → 概念槽.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from model.model_cyrene import DensePCNet

from model.modulation import l2_norm, rms_norm


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
        """行动: L4 状态 + 预测差分 → W_lm 解码生成字节."""
        if dev is None:
            dev = next(self.net.parameters()).device
        ids = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=dev)
        out = self.continuation(ids, n_tokens, temperature=temperature, rep_backstop=True)
        return bytes(out[0].tolist())

    def continuation(
        self,
        byte_ids: torch.Tensor,
        n_gen: int,
        temperature: float = 0.7,
        rep_backstop: bool = False,
    ) -> torch.Tensor:
        """自回归续写: 基于 [N,S] 前缀批量并行生成, 返回 [N, S+n_gen].

        训练 (rollout) 与验证 (generate) 共用同一 logits 管线.
        rep_backstop: 屏蔽近期 2..24 字节周期末字节的 logits.
        UTF-8 续字节超长阻断: 连续 3 个续字节后屏蔽全部续字节.
        """
        net = self.net
        N, S = byte_ids.shape
        dev = byte_ids.device
        cur = byte_ids.clone()
        last_byte = torch.full((N,), -1, dtype=torch.long, device=dev)
        # UTF-8 语法状态机: 当前字符还需续字节数 (0 = 字符边界)
        expect_cont = torch.zeros(N, dtype=torch.long, device=dev)
        if not hasattr(net, "_block_stats"):
            net._block_stats = {"rep": 0, "utf8": 0, "gen": 0}
        stats = net._block_stats
        for _ in range(n_gen):
            bv = cur[:, -64:]  # 上下文窗口 64 (任务语义)
            _ = self._predict(bv, store_state=True, is_inference=True)
            # W_lm 读出为唯一声道; W_act 降格为意图调制器 (见下方注入)
            if True:
                dim_4 = net.active_size["l4"]
                W_diff_a = net.W_diff[:dim_4, :dim_4]
                z4_n = net._z4 / (net._z4.norm(dim=-1, keepdim=True) + 1e-3)
                pred_delta = z4_n @ W_diff_a.T + net.b_diff[:dim_4].unsqueeze(0).unsqueeze(0)
                z4_next = net._z4 + pred_delta
                # 与训练头同款: 分流抑制 → RMS → 三阶 → 记忆单元+绑定拼接 → W1 混合层
                z4_nl = z4_next / (1.0 + z4_next.abs())
                z4_nl = rms_norm(z4_nl)
                z4_nl = z4_nl * (1.0 - 0.5 * z4_nl.pow(2))
                z4_nl = z4_nl / (1.0 + z4_nl.abs())  # 与训练端同款分流抑制
                zh_next = torch.cat([z4_nl, net._bind_vec, net._mem_out], dim=-1)
                zh_next = rms_norm(zh_next)  # 与训练端同口径 pre-norm
                h_next = zh_next @ net.W1
                h_next = h_next / (1.0 + h_next.abs())  # 分流抑制 (与训练同款)
                h_next = h_next / (h_next.square().mean(dim=-1, keepdim=True).sqrt() * 1.01 + 1e-8)
                h_next = h_next * (1.0 - 0.5 * h_next.pow(2))
                inv_h = 1.0 / math.sqrt(net.d_h)
                mu0_top = (h_next @ net.W_lm + net.bias_lm) * inv_h
                logits_c = (mu0_top - mu0_top.mean(dim=-1, keepdim=True)) / (
                    mu0_top.std(dim=-1, keepdim=True) + 1e-4
                )
                mu0_top = logits_c / logits_c.abs().max(dim=-1, keepdim=True).values * 60.0
                mask_print = torch.zeros(256, dtype=torch.float16, device=dev)
                mask_print[32:] = 1.0
                mu0_top = mu0_top + (1.0 - mask_print) * -1e4
                last = mu0_top[:, -1]  # [N,256] fp16
                # W_act 意图电位裸注入, 幅度归其自身稳态承载
                last = last + (net._bind_vec[:, -1] @ net.W_act)
            if temperature > 0:
                last = last / temperature
            stats["gen"] += N
            if rep_backstop:
                # n-gram 周期检测 (2..24): 重复周期末字节置 -1e4
                for p in range(2, 25):
                    if cur.shape[1] >= 2 * p:
                        pat = cur[:, -p:]  # [N,p] 最近 p 字节模式
                        prev = cur[:, -2 * p : -p]  # [N,p] 再前 p 字节模式
                        period = (pat == prev).all(dim=-1)  # [N] 周期重复
                        n_block = int(period.sum().item())
                        if n_block:
                            cyc_vals = torch.unique(pat[period])
                            for bv_ in cyc_vals.tolist():
                                last[period, bv_] = -1e4
                            stats["rep"] += n_block
            # UTF-8 语法阻断 (恒生效): 字节流必须合法才可被读
            #   expect_cont>0 (字符中途): 下一字节须为续字节 0x80-0xBF
            #   expect_cont==0 (字符边界): 下一字节为 ASCII 或合法起始 0xC2-0xF4
            n_utf8 = int((expect_cont > 0).sum().item())
            if n_utf8:
                last[expect_cont > 0, 0x00:0x80] = -1e4  # 中途禁 ASCII
                last[expect_cont > 0, 0xC0:0x100] = -1e4  # 中途禁 lead/非法
                stats["utf8"] += n_utf8
            n_bad = int((expect_cont == 0).sum().item())
            if n_bad:
                # 边界禁续字节 0x80-0xBF + 非法起始 0xC0-0xC1 (overlong)
                last[expect_cont == 0, 0x80:0xC2] = -1e4
            topv, _ = torch.topk(last, min(15, 256), dim=-1)
            last[last < topv[:, -1:]] = -float("inf")
            probs = torch.softmax(last, dim=-1)
            if getattr(net, "_entropy_sample", False):
                # 熵探索: 频率偏置软采样 = 变异通道 (仅采样, 不进学习更新)
                beta = getattr(net, "_entropy_beta", 0.3)
                freq_bias = (beta * (1.0 - net._freq_act)).unsqueeze(0)  # [1,256]
                pot_b = last + freq_bias
                b = torch.multinomial(torch.softmax(pot_b, dim=-1), 1).squeeze(-1)
            elif temperature <= 0.0:
                b = probs.argmax(dim=-1)
            else:
                b = torch.multinomial(probs, 1).squeeze(-1)
            cur = torch.cat([cur, b.unsqueeze(-1)], dim=1)
            last_byte = b
            # 更新 UTF-8 字符状态: 边界发合法起始 → 记录续字节需求; 中途发续 → 递减
            is_l2 = (b >= 0xC2) & (b <= 0xDF)
            is_l3 = (b >= 0xE0) & (b <= 0xEF)
            is_l4 = (b >= 0xF0) & (b <= 0xF4)
            lead_len = torch.where(is_l2, 1, torch.where(is_l3, 2, torch.where(is_l4, 3, torch.zeros_like(b))))
            expect_cont = torch.where(expect_cont > 0, expect_cont - 1, lead_len)
        return cur

    def _recurrent(
        self,
        mu: torch.Tensor,
        dev: torch.device,
        W_t: torch.Tensor,
        sweeps: int = 1,
        stp: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        stp_tag: str | None = None,
    ) -> torch.Tensor:
        """时间核递归: z[t] 依赖 z[t-1] (经 W_t), 纯 matmul.

        只归一化递归项, 保留 mu 携带的输入语义方向.
        stp (r, tau, u): 逐帧递归项 ×r[t] (资源耗尽 → 递归减弱, 自发间歇), 仅自由运行启用.
        stp_tag: 层名, 用于末资源写回 net._stp_r_end.
        """
        N, S, d = mu.shape
        z = mu
        inv_d = 1.0 / math.sqrt(d)
        if stp is None:
            for _ in range(sweeps):
                shift = torch.cat([torch.zeros(N, 1, d, dtype=z.dtype, device=dev), z[:, :-1]], dim=1)
                shift_n = rms_norm(shift)
                rec = (shift_n @ W_t.T) * inv_d
                rec = rms_norm(rec) * inv_d  # 单独归一化保方向, 幅度有界防累积
                z = z + rec
            return z
        # STP 路径: 逐帧递归项 × 当前资源; 幅度由 W_t 行范数承载 (防发散靠能量而非范数锁)
        # 递推用 z_out (含 STP 调制的输出), 首帧 = mu 首帧 (无递归)
        r, tau, u = stp
        r = r.unsqueeze(0)  # [1,d]
        inv_tau = (1.0 / tau).unsqueeze(0)  # [1,d]
        u_b = u.unsqueeze(0)  # [1,d]
        z_out = z[:, :1].clone()  # 首帧无递归 (shift=0)
        for t in range(1, S):
            shift = z_out[:, t - 1 : t]  # [N,1,d] 用已调制的输出递推
            shift_n = rms_norm(shift)
            rec = (shift_n @ W_t.T) * inv_d  # 幅度 = W_t 行范数 (增益自由度)
            z_t = z[:, t : t + 1] + rec * r  # 递归项 × 可用资源
            z_out = torch.cat([z_out, z_t], dim=1)
            # 资源更新: r ← r + (1−r)/τ − U_eff·r·|z|; U_eff = U·act/(act+ema) 自参照, 高活动耗尽加速
            act = z_t.abs().mean(dim=1)
            ema = getattr(self.net, f"_stp_active_ema_{stp_tag}")[: r.shape[1]]  # 修剪后对齐活性尺寸
            ema.mul_(0.99).add_(0.01 * act.squeeze(0))
            # 局部 U 自适应: 活动 vs 自身慢基线 → 释放概率双向调节 (纯局部负反馈)
            if self.net.cfg.stp_u_adapt:
                rel = (act - ema.unsqueeze(0)) / (ema.unsqueeze(0) + 1e-6)
                u_b.add_(self.net.cfg.stp_u_adapt_rate * rel * u_b)
                u_b.clamp_(self.net.cfg.stp_u_min, self.net.cfg.stp_u_max)

            u_eff = u_b * act / (act + ema + 1e-6)
            r = r + (1.0 - r) * inv_tau - u_eff * r * act
            r = r.clamp(0.01, 1.0)  # 资源界 (有界防发散)
        if stp_tag is not None:
            self.net._stp_r_end[stp_tag] = r.squeeze(0).clone()  # 窗末写回 (诊断/更新用)
        return z_out

    def _predict(self, byte_ids: torch.Tensor, store_state: bool = True, is_inference: bool = False) -> dict:
        """核心前馈: 感知 (L0→L6) + Foldiak 去相关 + 去中心化 + 增量预测.

        无位置编码, 时序全靠 W_t 递归; z5 去均值喂下游, z5_raw 喂 W_diff;
        ACh 噪声注入 L3, is_inference=True 时关闭.
        """
        net = self.net
        free_run = byte_ids is None
        if free_run:
            # 自由运行: 外部输入恒零, 活动由内部递归 + 三尺度加性振荡器驱动
            N, S = 1, net.cfg.free_run_window
            dev = next(net.parameters()).device
            z0 = torch.zeros(N, S, net._in_dim, dtype=torch.float16, device=dev)
            # 相位向量每窗一次 (查表 index_select, 零运行时三角): 值 = 1-2·phase
            vf = net._osc_f_tab[
                (net._osc_f_cnt.long() + torch.arange(S, device=dev)) % 64
            ].unsqueeze(0).unsqueeze(-1)  # [1,S,1]
            vm = net._osc_m_tab[
                (net._osc_m_cnt.long() + torch.arange(S, device=dev)) % 256
            ].unsqueeze(0).unsqueeze(-1)
            vs = net._osc_s_tab[
                (net._osc_s_cnt.long() + torch.arange(S, device=dev)) % 1024
            ].unsqueeze(0).unsqueeze(-1)
            net._osc_f_cnt.add_(S).remainder_(64)
            net._osc_m_cnt.add_(S).remainder_(256)
            net._osc_s_cnt.add_(S).remainder_(1024)
        else:
            N, S = byte_ids.shape
            dev = byte_ids.device
        dim_4, dim_2, dim_3, dim_5, dim_6 = (net.active_size[k] for k in ("l4", "l2", "l3", "l5", "l6"))

        W_04_a = net.W_04[:dim_4]
        W_42_a = net.W_42[:dim_2]
        W_23_a = net.W_23[:dim_3]
        W_56_a = net.W_56[:dim_6]
        W_diff_a = net.W_diff[:dim_4, :dim_4]

        if not free_run:
            # L0 纯 one-hot; 时序双通道: [z0[t], z0[t-1]] 拼接 (词序信息进入表示层)
            z0 = F.one_hot(byte_ids, num_classes=256).to(torch.float16)  # [N,S,256]
            if net.cfg.input_history:
                z0_prev = torch.cat([torch.zeros(N, 1, 256, dtype=z0.dtype, device=dev), z0[:, :-1]], dim=1)
                z0 = torch.cat([z0, z0_prev], dim=-1)  # [N,S,512]

        mu4 = z0 @ W_04_a.T + net.bias_l4[:dim_4]
        if free_run:
            mu4 = mu4 + net.cfg.osc_amp_m * vm  # 中节律加性电流 → L4 预激活
        net._mu4_diag = mu4  # 诊断插桩: 采集 PR(mu4) 区分 W_04 vs W_t4 坍缩
        # 去窗口化: 移除跨窗 carry 拼接, 每窗首帧从零起步
        if free_run:
            stp4 = (net._stp_r_l4[:dim_4], net._stp_tau_l4[:dim_4], net._stp_u_l4[:dim_4])
        else:
            stp4 = None
        z4 = self._recurrent(mu4, dev, net.W_t4[:dim_4, :dim_4], stp=stp4, stp_tag="l4" if free_run else None)
        z4_n = l2_norm(z4)
        mu2 = z4_n @ W_42_a.T + net.bias_l2[:dim_2]
        if free_run:
            mu2 = mu2 + net.cfg.osc_amp_m * vm  # 中节律 → L2
        z2 = self._recurrent(
            mu2, dev, net.W_t2[:dim_2, :dim_2],
            stp=(net._stp_r_l2[:dim_2], net._stp_tau_l2[:dim_2], net._stp_u_l2[:dim_2]) if free_run else None,
            stp_tag="l2" if free_run else None,
        )
        z2_n = l2_norm(z2)  # 只归一化输入, 保方向压尖峰
        mu3 = z2_n @ W_23_a.T + net.bias_l3[:dim_3]
        if free_run:
            mu3 = mu3 + net.cfg.osc_amp_s * vs  # 慢节律 → L3
        if not is_inference:
            mu3 = mu3 + torch.sign(2.0 * (torch.rand_like(mu3) - 0.5)) * 0.03  # ACh 噪声
        z3 = self._recurrent(
            mu3, dev, net.W_t3[:dim_3, :dim_3],
            stp=(net._stp_r_l3[:dim_3], net._stp_tau_l3[:dim_3], net._stp_u_l3[:dim_3]) if free_run else None,
            stp_tag="l3" if free_run else None,
        )
        if free_run:
            z3 = z3 / (1.0 + z3.abs())  # 分流抑制: 断 z3 下游尖峰链, 仅自由运行
        z3_n = l2_norm(z3)
        # L5 统一单矩阵
        z5 = z3_n @ net.W_35[:dim_5].T + net.bias_l5[:dim_5]
        if free_run:
            z5 = z5 + net.cfg.osc_amp_f * vf  # 快节律 → L5 预激活
        # 路由分离: z5_raw 原始幅度喂 W_diff, z5 去中心化喂下游.
        # Foldiak 反赫布侧抑制: M 学协方差 → 白化去相关 (零起步 = 恒等).
        # z5 不做三阶: 稀疏尖峰经三阶翻转放大 → W_diff 差分 NaN, 非线性由 z3 承载
        z5_fd = z5 - 0.2 * (net.M_l5[:dim_5, :dim_5] @ z5.transpose(-2, -1)).transpose(-2, -1)
        z5_raw = z5_fd
        z5 = z5_raw - z5_raw.mean(dim=-1, keepdim=True)
        mu6 = z5 @ W_56_a.T + net.bias_l6[:dim_6]
        if free_run:
            mu6 = mu6 + net.cfg.osc_amp_s * vs  # 慢节律 → L6
        z6 = self._recurrent(
            mu6, dev, net.W_t6[:dim_6, :dim_6],
            stp=(net._stp_r_l6[:dim_6], net._stp_tau_l6[:dim_6], net._stp_u_l6[:dim_6]) if free_run else None,
            stp_tag="l6" if free_run else None,
        )

        # W_diff 在 L4 空间预测 Δz4, eps_diff 驱动 W_diff 学习动力学
        dz4_pred = z4[:, 1:] - z4[:, :-1]
        dz4_pred = dz4_pred / (dz4_pred.norm(dim=-1, keepdim=True) + 1e-3)  # RMS 归一化
        z4_prev = z4[:, :-1] / (z4[:, :-1].norm(dim=-1, keepdim=True) + 1e-3)
        z4_prev_pad = torch.cat([torch.zeros(N, 1, dim_4, dtype=z4.dtype, device=dev), z4_prev], dim=1)
        bd = net.b_diff[:dim_4].unsqueeze(0).unsqueeze(0)
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
            if free_run:
                # 去窗口化: 每窗独立起步, STP 资源状态跨窗延续
                for sln, sz in (("l4", z4), ("l2", z2), ("l3", z3), ("l6", z6)):
                    if sln in net._stp_r_end:
                        self._stp_update(sln, net._stp_r_end[sln], sz)
            # 竞争性记忆单元群: K 个泄漏积分单元, m_c[t] = (1-α_c)·m_c[t-1] + α_c·z4[t].
            # 保留循环版: 张量化在 fp16 下下溢/舍入不同, 数学不等价
            K = net._mem_m.shape[0]
            a4_ = net._mem_m.shape[1]
            m_prev = net._mem_m.unsqueeze(0).unsqueeze(0).expand(N, 1, K, a4_).clone()  # [N,1,K,dim_4]
            m = m_prev
            for t in range(1, S):
                zt = z4[:, t : t + 1].unsqueeze(2)  # [N,1,1,dim_4]
                mt = (1 - net._mem_a)[None, None, :, None] * m[:, -1:] + net._mem_a[None, None, :, None] * zt
                m = torch.cat([m, mt], dim=1)
            net._mem_out = m.reshape(N, S, K * a4_)  # [N,S,K·dim_4] 展平供 zh 拼接
            net._mem_m = m[:, -1].mean(dim=0).detach()  # [K,dim_4] 批均值写回
            # 新奇度 (快慢散度): 死循环→N→0 (LTD); 正常推进→N 正 (LTP). 保留循环版 (张量化 fp16 下溢)
            zslow = torch.zeros_like(z4)
            zslow[:, 0] = z4[:, 0]
            for t in range(1, S):
                zslow[:, t] = 0.99 * zslow[:, t - 1] + 0.01 * z4[:, t]
            net._novelty = ((z4 - zslow).square().mean(dim=-1)).detach()  # [N,S] 逐帧新奇度

        # 稀疏绑定: 连续 z4 → W_bind → 槽内 top-k WTA; 推理也计算 (与训练同构)
        if net.cfg.bind_mode != "none":
            net._fr_vf = vf if free_run else None  # 快节律向量供 _bind 注入
            self._bind(z4)
            net._fr_vf = None
        else:
            net._fr_vf = None

        return {
            "mu_diff": mu_diff,
            "diff_err": diff_err,
            "free_energy": diff_err,
        }

    def _stp_update(self, layer: str, r_end: torch.Tensor, z: torch.Tensor) -> None:
        """STP 资源写回 (窗末): r 末帧写回 buffer (跨窗延续)."""
        r_buf = getattr(self.net, f"_stp_r_{layer}")
        r_buf[: r_end.shape[0]].copy_(r_end)  # 修剪后 active 收缩: 只写头部切片

    def _bind(self, z4: torch.Tensor) -> None:
        """竞争性概念绑定: z4 → W_bind → 槽位 → 连续值稀疏激活向量.

        relu(raw - θ) 阈值竞争 + 分流抑制, 无 sum=1 约束, 输出为连续值稀疏向量.
        θ 复用 _theta_bind (跟踪槽能量).
        """
        net = self.net
        dim_4 = net.active_size["l4"]
        z4_n = l2_norm(z4)
        raw = z4_n @ net.W_bind[:dim_4]  # [N,S,K] 自下而上投影
        if getattr(net, "_bind_loop", True):
            # 自由运行快节律注入: 加性电流 → z_bind 预激活
            if getattr(net, "_fr_vf", None) is not None:
                raw = raw + net.cfg.osc_amp_f * net._fr_vf
            # STP 槽资源: 逐帧输出 ×r (资源耗尽 → 槽间歇切换)
            stp_bind = None
            if getattr(net, "_fr_vf", None) is not None:
                stp_bind = (net._stp_r_bind, net._stp_tau_bind, net._stp_u_bind)
            z_bind = self._bind_sparse(raw[:, :1] - net._theta_bind)  # t=0 无历史
            z_binds = [z_bind]
            r_bind = stp_bind[0].unsqueeze(0) if stp_bind else None
            inv_tau_b = (1.0 / stp_bind[1]).unsqueeze(0) if stp_bind else None
            u_bind_b = stp_bind[2].unsqueeze(0) if stp_bind else None
            for t in range(1, raw.shape[1]):
                raw_t = raw[:, t : t + 1] + 0.5 * (z_bind @ net.W_bind_self)
                if stp_bind:
                    raw_t = raw_t * r_bind  # 槽输入 × 可用资源
                z_bind = self._bind_sparse(raw_t - net._theta_bind)
                z_binds.append(z_bind)
                if stp_bind:
                    act = z_bind.abs().mean(dim=1)  # [1,K] (同递归层修正)
                    ema_b = getattr(net, "_stp_active_ema_bind")[: r_bind.shape[1]]
                    ema_b.mul_(0.99).add_(0.01 * act.squeeze(0))
                    # 槽位局部 U 自适应 (同递归层, 纯局部负反馈)
                    if self.net.cfg.stp_u_adapt:
                        rel_b = (act - ema_b.unsqueeze(0)) / (ema_b.unsqueeze(0) + 1e-6)
                        u_bind_b.add_(self.net.cfg.stp_u_adapt_rate * rel_b * u_bind_b)
                        u_bind_b.clamp_(self.net.cfg.stp_u_min, self.net.cfg.stp_u_max)

                    u_eff_b = u_bind_b * act / (act + ema_b + 1e-6)  # 自参照耗尽
                    r_bind = r_bind + (1.0 - r_bind) * inv_tau_b - u_eff_b * r_bind * act
                    r_bind = r_bind.clamp(0.01, 1.0)
            if stp_bind and r_bind is not None:
                net._stp_r_end["bind"] = r_bind.squeeze(0).clone()
            z_bind = torch.cat(z_binds, dim=1)
        else:
            z_bind = self._bind_sparse(raw - net._theta_bind)
        # 分流抑制: 上界 1.0, 范数保留为自由度
        th_b = net._theta_bind
        th_b.mul_(0.98).add_(0.02 * (z_bind * z_bind).mean(dim=(0, 1)))
        z_bind = z_bind / (1.0 + z_bind.abs())
        net._bind_pre = raw
        net._bind_vec = z_bind  # [N,S,K] 连续值稀疏向量
        if getattr(net, "_fr_vf", None) is not None:
            if "bind" in net._stp_r_end:
                self._stp_update("bind", net._stp_r_end["bind"], z_bind)  # 槽 STP 资源写回
        # 动作电位: 概念槽 → W_act → 字节脉冲潜能
        net._intent_pot = z_bind @ net.W_act  # [N,S,256] (soft_norm 列归一有界)

    def _bind_sparse(self, pre_act: torch.Tensor, k: int = 4) -> torch.Tensor:
        """槽稀疏竞争: 每位置只保留 top-k 强激活槽, 其余清零.

        保留连续值幅度 (relu 后值, 非二值), 范数仍是自由度; 阈值 θ 先弱激活过滤再 top-k.
        """
        x = torch.relu(pre_act)  # [...,K] 弱激活过滤
        if x.shape[-1] <= k:
            return x
        # 每位置 top-k 掩码: 只保留最大的 k 个值
        thr = x.topk(k, dim=-1).values[..., -1:]  # 第 k 大值
        mask = (x >= thr).to(x.dtype)
        return x * mask

    def _precise(self, eps: torch.Tensor) -> torch.Tensor:
        """精度加权: π_l = 1/(σ_εl + c), 归一化每层误差尺度."""
        s = eps.std() + 1e-3
        return eps / s
