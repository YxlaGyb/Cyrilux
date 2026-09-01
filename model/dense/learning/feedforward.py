"""
前馈链学习域
bias 稳态 + W_04/W_42/W_56/W_23/W_state_pred + W_35 + 收尾软范数.
"""

from __future__ import annotations

import torch

from ...modulation import soft_norm_preserve
from model.modulation import rms_norm
from ._common import _activity_baseline, _decorr_W, _elig_accum, _energy_constraint, _MixinBase, _rho_ctrl


class FeedforwardMixin(_MixinBase):
    """前馈链域 (方法挂载到 LearningEngine)."""

    def _update_bias(self, ctx, sh):
        """自由运行/自回声偏置稳态更新."""
        net = self.net
        free_run, echo_loop, echo_world_frozen = ctx.free_run, ctx.echo_loop, ctx.echo_world_frozen
        if (free_run or echo_loop) and not echo_world_frozen:
            z4, z2, z3, z5, z6 = net._z4, net._z2, net._z3, net._z5, net._z6
            _, ex4 = _activity_baseline(net, z4, "_active_ema_b4")
            _, ex2 = _activity_baseline(net, z2, "_active_ema_b2")
            _, ex3 = _activity_baseline(net, z3, "_active_ema_b3")
            _, ex5 = _activity_baseline(net, z5, "_active_ema_b5")
            _, ex6 = _activity_baseline(net, z6, "_active_ema_b6")
            beta = net.cfg.oja_elasticity
            # 增长门控 g_i = act²/(bias²+act²): bias 已主导局部信号时抑制其积分 (防焊进分流饱和区)
            b_gate = {}
            for b_par, zz, a_sz in (
                (net.bias_l4, z4, ctx.dim_4), (net.bias_l2, z2, ctx.dim_2),
                (net.bias_l3, z3, ctx.dim_3), (net.bias_l5, z5, ctx.dim_5),
                (net.bias_l6, z6, ctx.dim_6),
            ):
                b2 = b_par[:a_sz].data.square()
                act2 = (zz * zz).mean(dim=(0, 1))  # 窗内活动能量
                b_gate[id(b_par)] = act2 / (b2 + act2 + 1e-6)
            eta = ctx.eta
            eps4, eps2, eps3, eps5, eps6 = (
                sh.errs.eps4, sh.errs.eps2, sh.errs.eps3, sh.errs.eps5, sh.errs.eps6,
            )
            net.bias_l4[:ctx.dim_4].data += eta * (eps4.mean(dim=(0, 1)) / (1.0 + beta * ex4)) * b_gate[id(net.bias_l4)]
            net.bias_l2[:ctx.dim_2].data += eta * (eps2.mean(dim=(0, 1)) / (1.0 + beta * ex2)) * b_gate[id(net.bias_l2)]
            net.bias_l3[:ctx.dim_3].data += eta * (eps3.mean(dim=(0, 1)) / (1.0 + beta * ex3)) * b_gate[id(net.bias_l3)]
            net.bias_l5[:ctx.dim_5].data += eta * (eps5.mean(dim=(0, 1)) / (1.0 + beta * ex5)) * b_gate[id(net.bias_l5)]
            net.bias_l6[:ctx.dim_6].data += eta * (eps6.mean(dim=(0, 1)) / (1.0 + beta * ex6)) * b_gate[id(net.bias_l6)]
            # 自由运行泄漏随 bias 占比增强 (1+share 倍): 防长成定点支柱
            if free_run:
                for b_par, zz, a_sz in (
                    (net.bias_l4, z4, ctx.dim_4), (net.bias_l2, z2, ctx.dim_2),
                    (net.bias_l3, z3, ctx.dim_3), (net.bias_l5, z5, ctx.dim_5),
                    (net.bias_l6, z6, ctx.dim_6),
                ):
                    b2 = b_par[:a_sz].data.square()
                    act2 = (zz * zz).mean(dim=(0, 1))  # 窗内活动能量
                    share = b2 / (b2 + act2 + 1e-6)  # bias 占比 ∈ [0,1]
                    b_par[:a_sz].data.mul_(1.0 - net.cfg.bias_leak_rate * (1.0 + share))

    def _update_feed_ff(self, ctx, sh):
        """W_04/W_42/W_56/W_23 + eps_state 融合 + W_state_pred 自更新."""
        net = self.net
        dev, N = ctx.dev, ctx.N
        free_run, echo_world_frozen = ctx.free_run, ctx.echo_world_frozen
        inv_s, eta = ctx.inv_s, ctx.eta
        dim_4, dim_2, dim_3, dim_6 = ctx.dim_4, ctx.dim_2, ctx.dim_3, ctx.dim_6
        z0, z4, z2, z5 = net._z0, net._z4, net._z2, net._z5

        # 突触后增益控制: 用误差自身 std 归一化, 随修剪缩维自适应 (无静态系数)
        if not free_run and not echo_world_frozen:
            # W_04 主辅误差: 预测误差为主 + 0.2×重建误差 (重建不需要词序, 只稳定信号);
            # F 由 _update_metabolism 同式计算 (单一出处, 保证驱动 W_04 与代谢账本同 F)
            final_error = sh.metab_f

            # 幅度-方向解耦: dW 归一化单位向量, 显著性只选方向不放大幅度
            # → 单步最大幅度 = lr, 防极端样本单步爆 NaN
            z0_n = rms_norm(z0)
            dW_04n = torch.bmm(final_error.transpose(-2, -1), z0_n)  # [N,dim_4,in]
            nrm_04 = dW_04n.norm(dim=(-2, -1), keepdim=True)
            # 零范数行分母=1 (预测完美 → dW=0 → 0/0=NaN)
            alive_04 = (nrm_04 > 1e-8).to(dW_04n.dtype)
            denom_04 = torch.where(alive_04 > 0, nrm_04, torch.ones_like(nrm_04))
            dW_04n = dW_04n * alive_04 / denom_04  # 单位向量 (只保留方向)
            e_norm = final_error.norm(dim=(1, 2))
            # N=1 时 e_norm.std() 无意义, 用 0 等价 (分母 = max)
            e_std = e_norm.std() if e_norm.shape[0] > 1 else torch.zeros_like(e_norm.max())
            e_denom = e_norm.max() + e_std
            e_alive = (e_denom > 1e-8).to(e_norm.dtype)
            g_04 = e_norm / torch.where(e_alive > 0, e_denom, torch.ones_like(e_denom))
            g_04 = (g_04 * e_alive).unsqueeze(-1).unsqueeze(-1)
            g_sum = g_04.sum() + (1 - e_alive.sum()).to(g_04.dtype)  # 全零 → 分母 1
            dW_04 = (dW_04n * g_04).sum(dim=0) / g_sum
            # 输出端 homeostatic 抑制: g_j=1/(1+θ_j), θ 跟踪输出列活动², 破 σ₁ 垄断
            th_w04 = net._theta_w04[:dim_4]
            mu4_h = z0 @ net.W_04[:dim_4].T + net.bias_l4[:dim_4]  # 与 eps4 定义同源
            th_w04.mul_(0.98).add_(0.02 * (mu4_h * mu4_h).mean(dim=(0, 1)))
            g_homeo = (1.0 / (1.0 + th_w04)).unsqueeze(1)  # [dim_4,1] 每输出列增益
            net._theta_w04_dist = th_w04  # 诊断: θ_j 分布
            dW_04 = dW_04 * g_homeo
            # 资格迹接力: 更新 = η·R·E 的滑均形式 (R 为生存信号时全迹缩放)
            dW_04 = dW_04 + _elig_accum(net, "W_04", dW_04) * getattr(net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16))
            net.W_04[:dim_4].data += dW_04 * eta
            soft_norm_preserve(net.W_04[:dim_4].data)
            # 行去同质化: 斜坡渐进 (coef 200 步升到 1) + 范数信任域 (单步 ≤5%‖W‖_F) 防 z4 突变换层
            ramp = min(1.0, net._step_counter / 200.0)
            _decorr_W(net.W_04[:dim_4].data, net.E_04[:dim_4, :dim_4], coef=ramp, max_delta_ratio=0.05, learn_boost=ctx.learn_boost)

            dW42 = (sh.errs.eps2_precise.transpose(-2, -1) @ rms_norm(z4)).mean(dim=0) * inv_s
            dW42 = _energy_constraint(net, net.W_42[:dim_2].data, dW42, sh.errs.eps2_precise, "_active_ema_w42")
            dW42 = dW42 + _elig_accum(net, "W_42", dW42) * getattr(net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16))
            col_mask = torch.rand(dim_2, 1, device=dev) < net.cfg.column_dropout
            net.W_42[:dim_2].data += (dW42 * (~col_mask).to(torch.float16)) * eta
            _decorr_W(net.W_42[:dim_2].data, net.E_42[:dim_2, :dim_2], learn_boost=ctx.learn_boost)
        # 字节域块结束: free_run / echo_world_frozen 冻结 W_04/W_42 (感知链)
        # W_56 (L5→L6 内部动力学) 在自由运行同样学习
        dW_56 = (sh.errs.eps6_precise.transpose(-2, -1) @ rms_norm(z5)).mean(dim=0) * inv_s
        dW_56 = _energy_constraint(net, net.W_56[:dim_6].data, dW_56, sh.errs.eps6_precise, "_active_ema_w56")
        dW_56 = dW_56 + _elig_accum(net, "W_56", dW_56) * getattr(net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16))
        col_mask = torch.rand(net.W_56[:dim_6].shape[0], 1, device=dev) < net.cfg.column_dropout
        net.W_56[:dim_6].data += (dW_56 * (~col_mask).to(torch.float16)) * eta

        # 预测编码融合: eps_state = (z4 @ W_state_pred) - Δz4, 让表示层同时携带"未来往哪走"
        W_sp_a = net.W_state_pred[:dim_4, :dim_4]
        dz4_full = net._z4[:, 1:] - net._z4[:, :-1]
        eps_state = (net._z4[:, :-1] @ W_sp_a.T) - dz4_full  # [N,S-1,dim_4]
        eps_state = rms_norm(eps_state)
        eps_state_3 = eps_state @ net.W_42[:dim_2].T[:, :dim_3]  # dim_4 → dim_3 (经 W_42 逆映射)
        eps3_pc = sh.errs.eps3 + 0.3 * torch.cat(
            [eps_state_3, torch.zeros(N, 1, dim_3, dtype=eps_state_3.dtype, device=dev)], dim=1
        )
        eps3_pc = net.forward_engine._precise(eps3_pc)

        # L3 种子: 随机增益 + 误差门控; _gain_l3 固定 [384,384], 修剪后按活性行切片
        dim_3 = ctx.dim_3
        gain_l3 = net._gain_l3[:dim_3, :dim_3] if dim_3 < 384 else net._gain_l3[:dim_3, :]
        dW23 = (eps3_pc.transpose(-2, -1) @ rms_norm(z2)).mean(dim=0) * inv_s
        dW23 = _energy_constraint(net, net.W_23[:dim_3].data, dW23, eps3_pc, "_active_ema_w23")
        dW23 = dW23 + _elig_accum(net, "W_23", dW23) * getattr(net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16))
        err3_norm = sh.errs.eps3.pow(2).mean(dim=(0, 1)).sqrt() + 1e-8  # [dim_3] 每神经元
        gate3 = 0.1 + 0.9 * (err3_norm / err3_norm.max())
        dW23 = dW23 * gain_l3 * gate3.unsqueeze(1)
        c3_mask = torch.rand(dim_3, 1, device=dev) < net.cfg.column_dropout
        net.W_23[:dim_3].data += (dW23 * (~c3_mask).to(torch.float16)) * eta
        soft_norm_preserve(net.W_23[:dim_3].data)  # 0.8-1.2: 增益幅度保留, 权重有界
        _decorr_W(net.W_23[:dim_3].data, net.E_23[:dim_3, :dim_3], learn_boost=ctx.learn_boost)
        # 状态预测矩阵自更新 (纯赫布): dW_sp = z4^T @ eps_state; echo_world_frozen 时冻结 (世界模型)
        if not echo_world_frozen:
            dW_sp = (net._z4[:, :-1].transpose(-2, -1) @ eps_state).mean(dim=0)
            dW_sp = _energy_constraint(net, W_sp_a.data, dW_sp, net._z4, "_active_ema_wsp")
            dW_sp = dW_sp + _elig_accum(net, "W_state_pred", dW_sp) * getattr(net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16))
            W_sp_a.data += dW_sp * eta
            soft_norm_preserve(W_sp_a.data)

    def _update_W35(self, ctx, sh):
        """W_35 更新: 预测编码融合误差全量驱动."""
        net = self.net
        dev, N, S = ctx.dev, ctx.N, ctx.S
        free_run, echo_world_frozen = ctx.free_run, ctx.echo_world_frozen
        dim_3, dim_5 = ctx.dim_3, ctx.dim_5
        z3, z5 = net._z3, net._z5
        if not free_run and not echo_world_frozen:
            # 预测编码向下平移: W_lm 误差 → dim_3 (补零对齐 S, 经 W_42 逆映射)
            eps_lm_proj_pad = torch.cat(
                [sh.lm.eps_lm_proj, torch.zeros(N, 1, ctx.dim_4, dtype=sh.lm.eps_lm_proj.dtype, device=dev)], dim=1
            )
            eps_lm_3 = eps_lm_proj_pad @ net.W_42[:ctx.dim_2].T[:, :dim_3]
            eps_lm_3 = rms_norm(eps_lm_3)
        z5_b = z5
        eps_b = z5_b[:, 1:] - z5_b[:, :-1]  # 时序差分误差
        z3_b = z3[:, :-1]
        z3_bp = z3_b * (torch.rand_like(z3_b) > 0.3).to(torch.float16)
        if not free_run and not echo_world_frozen:
            # 融合: eps_b = 时序差分 + 0.5 × LM 误差投影回 L5
            eps_lm_b = eps_lm_3[:, :-1] @ net.W_35[:dim_5].T  # [N,S-1,dim_5]
            eps_lm_b = rms_norm(eps_lm_b)
            eps_b = eps_b + 0.5 * eps_lm_b
        # BCM 滑阈: theta=EMA(eps²), phi=eps(eps-theta); 先 RMS 归一化防 fp16 溢出
        eps_b = rms_norm(eps_b)
        th = net._theta_l5[:dim_5]
        ep2 = (eps_b * eps_b).mean(dim=(0, 1))
        th.mul_(0.975).add_(0.025 * ep2)
        phi_b = eps_b * (eps_b - th)
        # 标量 s=max|phi_b|: 平方前归一化防溢出 (phi_b² 可超 fp16 上限 → inf → NaN)
        s_phi = phi_b.abs().max() + 1e-6
        phi_b = phi_b / s_phi
        g_n = (phi_b * phi_b).mean(dim=(1, 2)) + 1e-8  # 样本显著性 (线性加权)
        # 零向量保护: phi_b≡0 → g_n=0 → 0/0=NaN, alive 掩码分母=1
        g_sum = g_n.sum()
        g_alive = (g_sum > 1e-8).to(g_n.dtype)
        g_n = g_n / torch.where(g_alive > 0, g_sum, torch.ones_like(g_sum))
        Wb = net.W_35[:dim_5].data
        gain_mask = net._gain_mask[:dim_5, :dim_3]
        eta = ctx.eta
        err_norm = eps_b.pow(2).mean(dim=(0, 1)).sqrt() + 1e-8  # 高误差神经元主导更新
        e_max = err_norm.max()
        e_alive = (e_max > 1e-8).to(err_norm.dtype)
        upd_gate = 0.1 + 0.9 * (err_norm / torch.where(e_alive > 0, e_max, torch.ones_like(e_max)))
        # 资格调制: q_i = 每行时间残差能量 (时间差分无法解释的行获得更高学习资格)
        z5_prev = torch.cat([torch.zeros(N, 1, dim_5, dtype=z5.dtype, device=dev), z5[:, :-1]], dim=1)
        z5_prev_n = rms_norm(z5_prev)
        z5_pred_t = z5_prev_n @ net.W_t5[:dim_5].T  # W_t5 时间预测
        # 残差平方前按 max|残差| 归一化: 直接平方溢出 → q_i=inf/inf=NaN; s² 在分子分母消去, 数学全等
        res_t = z5[:, 1:] - z5_pred_t[:, 1:]  # [N,S-1,dim_5] 残差
        s = res_t.abs().max() + 1e-6
        res_neuron = ((res_t / s) ** 2).mean(dim=(0, 1))  # [dim_5] 每行相对残差能量
        q_i = res_neuron / (res_neuron.max() + 1e-6)  # [dim_5] 资格 ∈ [0,1]
        n_sub = min(4, max(1, N))  # N=1 时防重复子集
        sub = max(1, N // n_sub)
        inv_sm1 = 1.0 / (S - 1)
        dW_h = None
        for s in range(n_sub):
            sl = slice(s * sub, (s + 1) * sub)
            dW_n = torch.bmm((phi_b[sl] * inv_sm1).transpose(-2, -1), rms_norm(z3_bp[sl]))  # 预除防 fp16 中间累加超界
            dW_sub = (dW_n * g_n[sl, None, None]).sum(dim=0)
            dW_sub = dW_sub * gain_mask * upd_gate.unsqueeze(1)
            dW_sub = dW_sub * q_i.unsqueeze(1)
            b_mask = torch.rand(dim_5, 1, device=dev) < net.cfg.column_dropout
            dW_sub = dW_sub * (~b_mask).to(torch.float16)
            dW_h = dW_sub if dW_h is None else dW_h + dW_sub
        dW_h = _energy_constraint(net, Wb, dW_h, phi_b, "_active_ema_w35")
        dW_h = dW_h + _elig_accum(net, "W_35", dW_h) * getattr(net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16))
        dW_h = dW_h * eta
        # 通道级塑性控制: ρ_i=‖ΔW_i‖/‖W_i‖, s_i=clip(0.03/ρ_i, 0.005, 1.0), 统一预测连接时间尺度
        nW = Wb.norm() + 1e-8  # decorr 前统一基准

        if net.cfg.adaptive_rho:
            net._rho_map = {}
        Wb.add_(_rho_ctrl(dW_h, Wb, "hebb", net))
        # 前馈链突触缩放 (自由运行): L5 入纤增益随自身活动双向调节, 与 W_t 同款; 训练模式保持 soft_norm
        if free_run and net.cfg.wt_syn_scaling:
            p2w5 = (z5 * z5).mean(dim=(0, 1))  # [dim_5] L5 窗内活动²
            emaw5 = net._active_ema_b5[:dim_5]  # L5 慢基线 (bias 泄漏段已更新)
            # 分母 1e-6 (非 1e-3): 低活分支不被地板淹没, scale→2 防死亡螺旋
            scale5 = 2.0 * emaw5 / (p2w5 + emaw5 + 1e-6)  # 双向突触缩放 (0,2]
            r835 = net.cfg.wt_syn_scaling_rate
            Wb.data.mul_((1.0 - r835) + r835 * scale5.unsqueeze(1))
        else:
            soft_norm_preserve(Wb)
        dW_corr = _decorr_W(Wb, net.E_l5[:dim_5, :dim_5], learn_boost=ctx.learn_boost)
        if net.cfg.adaptive_rho:
            rho_raw, rho_eff, s_h = net._rho_map["hebb"]
            cos_hc = (dW_h.flatten() @ dW_corr.flatten()) / (
                dW_h.norm() * dW_corr.norm() + 1e-8
            )
            net._rho_raw, net._rho_eff, net._s_h = rho_raw, rho_eff, s_h
            net._rho_corr = dW_corr.norm() / nW
            net._cos_hc = cos_hc

    def _final_softnorm(self, ctx, sh):
        """前馈权重软范数保持 (0.8-1.2): 无 BCM 约束的权重长训累积溢出 fp16.

        free_run / echo_world_frozen 时 W_04/W_42/W_diff 冻结不触碰; W_56 恒保持.
        """
        net = self.net
        free_run, echo_world_frozen = ctx.free_run, ctx.echo_world_frozen
        dim_4, dim_2, dim_6 = ctx.dim_4, ctx.dim_2, ctx.dim_6
        if not free_run and not echo_world_frozen:
            for W, a_sz in zip([net.W_04, net.W_42], [dim_4, dim_2]):
                soft_norm_preserve(W[:a_sz].data)
            soft_norm_preserve(net.W_diff[:dim_4, :dim_4].data)
        soft_norm_preserve(net.W_56[:dim_6].data)
