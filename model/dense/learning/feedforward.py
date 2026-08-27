"""前馈链学习域: bias 稳态 + W_04/W_42/W_56/W_23/W_state_pred + W_35 + 收尾软范数.

从原 learn() 单函数按块拆分, 块内语句顺序逐行保持 (数值逐位等价).
"""

from __future__ import annotations

import torch

from ...modulation import soft_norm_preserve
from ..forward import _rms
from ._common import _activity_baseline, _decorr_W, _elig_accum, _energy_constraint, _MixinBase, _rho_ctrl


class FeedforwardMixin(_MixinBase):
    """前馈链域 (方法挂载到 LearningEngine)."""

    def _update_bias(self, ctx, sh):
        """自由运行/自回声偏置稳态更新 (第 77/79/86/87 轮 原块)."""
        net = self.net
        free_run, echo_loop, echo_world_frozen = ctx.free_run, ctx.echo_loop, ctx.echo_world_frozen
        if (free_run or echo_loop) and not echo_world_frozen:
            z4, z2, z3, z5, z6 = net._z4, net._z2, net._z3, net._z5, net._z6
            _, ex4 = _activity_baseline(net, z4, "_act_ema_b4")
            _, ex2 = _activity_baseline(net, z2, "_act_ema_b2")
            _, ex3 = _activity_baseline(net, z3, "_act_ema_b3")
            _, ex5 = _activity_baseline(net, z5, "_act_ema_b5")
            _, ex6 = _activity_baseline(net, z6, "_act_ema_b6")
            beta = net.cfg.oja_elasticity
            # 第 86 轮 (G9): bias 增长门控 — exp85 实证 bias 成为支柱 (8→54) 把 z4
            # 焊进分流饱和区 (sat 54%, env_cv 0.011 间歇性死亡). 根因: bias 积分
            # eps.mean 持续为正 (bias→mu→z4→eps 正反馈). 门控: 当 bias 已主导局部
            # 信号 (share 高) 时抑制其增长 — 逐单元局部, rel_n 家族, 零新常数.
            # g_i = act²_i/(bias²_i+act²_i+1e-6) ∈ (0,1]: bias 主导 → g→0 停增长
            # (bias 只承担均值, 不承担驱动); 活动主导 → g→1 正常积分.
            # 第 87 轮: 门控覆盖 echo_loop (自回声输入) — 交互训练双相位走这里.
            b_gate = {}
            for b_par, zz, a_sz in (
                (net.bias_l4, z4, ctx.a4), (net.bias_l2, z2, ctx.a2),
                (net.bias_l3, z3, ctx.a3), (net.bias_l5, z5, ctx.a5),
                (net.bias_l6, z6, ctx.a6),
            ):
                b2 = b_par[:a_sz].data.square()
                act2 = (zz * zz).mean(dim=(0, 1))  # 窗内活动能量
                b_gate[id(b_par)] = act2 / (b2 + act2 + 1e-6)
            eta = ctx.eta
            e4, e2, e3, e5t, e6 = (
                sh.errs.eps4, sh.errs.eps2, sh.errs.eps3, sh.errs.eps5_td, sh.errs.eps6,
            )
            net.bias_l4[:ctx.a4].data += eta * (e4.mean(dim=(0, 1)) / (1.0 + beta * ex4)) * b_gate[id(net.bias_l4)]
            net.bias_l2[:ctx.a2].data += eta * (e2.mean(dim=(0, 1)) / (1.0 + beta * ex2)) * b_gate[id(net.bias_l2)]
            net.bias_l3[:ctx.a3].data += eta * (e3.mean(dim=(0, 1)) / (1.0 + beta * ex3)) * b_gate[id(net.bias_l3)]
            net.bias_l5[:ctx.a5].data += eta * (e5t.mean(dim=(0, 1)) / (1.0 + beta * ex5)) * b_gate[id(net.bias_l5)]
            net.bias_l6[:ctx.a6].data += eta * (e6.mean(dim=(0, 1)) / (1.0 + beta * ex6)) * b_gate[id(net.bias_l6)]
            # 第 82 轮: 自由运行 bias 泄漏 — 防止 bias 长成“定点支柱”.
            # 纯局部逐单元衰减, 只限 free_run, 不碰 echo/外部输入模式.
            # 第 86 轮 (G9b): 泄漏随"bias 在局部信号中的占比"增强 — 固定泄漏
            # 1e-4 被 bias 积分压过 (bias_l4 8→54). rel_n 家族 (零新常数):
            # share_i = bias²_i/(bias²_i + act²_i + 1e-6), 泄漏率 = rate·(1+share)
            # ∈ [rate, 2·rate] — bias 主导 → 泄漏翻倍 (拆存量), 活动主导 → 基线.
            if free_run:
                for b_par, zz, a_sz in (
                    (net.bias_l4, z4, ctx.a4), (net.bias_l2, z2, ctx.a2),
                    (net.bias_l3, z3, ctx.a3), (net.bias_l5, z5, ctx.a5),
                    (net.bias_l6, z6, ctx.a6),
                ):
                    b2 = b_par[:a_sz].data.square()
                    act2 = (zz * zz).mean(dim=(0, 1))  # 窗内活动能量
                    share = b2 / (b2 + act2 + 1e-6)  # bias 占比 ∈ [0,1]
                    b_par[:a_sz].data.mul_(1.0 - net.cfg.bias_leak_rate * (1.0 + share))

    def _update_feed_ff(self, ctx, sh):
        """W_04 / W_42 / W_56 / W_23 + eps_state 融合 + W_state_pred 自更新."""
        net = self.net
        dev, N = ctx.dev, ctx.N
        free_run, echo_world_frozen = ctx.free_run, ctx.echo_world_frozen
        inv_s, eta = ctx.inv_s, ctx.eta
        a4, a2, a3, a6 = ctx.a4, ctx.a2, ctx.a3, ctx.a6
        z0, z4, z2, z5 = net._z0, net._z4, net._z2, net._z5

        # 突触后增益控制: 归一化基于当前误差自身的 std (统计去耦), 不依赖维度 —
        # 修剪缩小 L4 时误差方差自然变小, 分母自动适应, 无 1/A4 静态系数
        if not free_run and not echo_world_frozen:
            # ── W_04 主辅误差交换: 预测误差为主, 重建为辅 ──
            # 重建任务不需要词序 (稳定信号拉权重回单一解); 预测误差才需要词序.
            # final_error = err_pred_norm + 0.2 * err_recon_norm (量级对齐)
            # 突触后增益控制 (std 归一化) + 相对地板: 地板 = 全局 std 的 0.1%,
            # 随信号缩放 (修剪缩维 → 全局 std 自动降), 防 fp16 精度极限下除零放大
            eps_lm_pad = sh.lm.eps_lm_pad
            eps4 = sh.errs.eps4
            eps_pred_scale = eps_lm_pad.std(dim=-1, keepdim=True)
            eps_pred_scale = eps_pred_scale * 1.01 + eps_lm_pad.std() * 1e-3
            err_pred_norm = eps_lm_pad / eps_pred_scale
            eps_recon_scale = eps4.std(dim=-1, keepdim=True)
            eps_recon_scale = eps_recon_scale * 1.01 + eps4.std() * 1e-3
            err_recon_norm = eps4 / eps_recon_scale
            final_error = err_pred_norm + 0.2 * err_recon_norm

            # 幅度-方向解耦 (单步更新上界锁死): dW_n 归一化为单位向量,
            # 显著性只选方向不放大幅度 — 原 g_n×dW_n 等效 e² 平方级放大,
            # 极端样本单步爆幅度 → NaN; 归一化后最大单步幅度 = lr, 纯机制保证
            z0_n = _rms(z0)
            dW_04n = torch.bmm(final_error.transpose(-2, -1), z0_n)  # [N,a4,in]
            nrm_04 = dW_04n.norm(dim=(-2, -1), keepdim=True)
            # 零向量保护: 模型对某 batch 预测完美 (final_error 全零) → dW_04n 全零
            # → 0/0 = NaN (fp16 下 1e-6 被舍入为 0, 装饰性). 掩码: 零范数行分母=1
            alive_04 = (nrm_04 > 1e-8).to(dW_04n.dtype)
            denom_04 = torch.where(alive_04 > 0, nrm_04, torch.ones_like(nrm_04))
            dW_04n = dW_04n * alive_04 / denom_04  # 单位向量 (只保留方向)
            e_norm = final_error.norm(dim=(1, 2))
            # 零向量保护: e_norm 全零 (perfect batch) → max+std=0 → 0/0=NaN.
            # torch.where 掩码 (与 _rms 同款): 全零行分母=1, 分子×0 → 零更新
            # 第 87 轮: N=1 (交互单样本) 时 e_norm.std() 自由度≤0 → UserWarning.
            # 恒等映射保护: N=1 时 std 无意义, 用 0 等价 (e_denom = max 不变).
            e_std = e_norm.std() if e_norm.shape[0] > 1 else torch.zeros_like(e_norm.max())
            e_denom = e_norm.max() + e_std
            e_alive = (e_denom > 1e-8).to(e_norm.dtype)
            g_04 = e_norm / torch.where(e_alive > 0, e_denom, torch.ones_like(e_denom))
            g_04 = (g_04 * e_alive).unsqueeze(-1).unsqueeze(-1)
            g_sum = g_04.sum() + (1 - e_alive.sum()).to(g_04.dtype)  # 全零 → 分母 1
            dW_04 = (dW_04n * g_04).sum(dim=0) / g_sum
            # W_04 输出端 homeostatic 抑制 (第 75 轮): 打破 σ₁ 奇异值垄断.
            # θ_j 慢速跟踪 mu4_j² (μ=0.02), g_j = 1/(1+θ_j) — 高激活输出列权重
            # 更新被抑制, 能量被迫向 σ₂/σ₃ 扩散. 纯局部 (每列只看自己), 连续无截断
            th_w04 = net._theta_w04[:a4]
            mu4_h = z0 @ net.W_04[:a4].T + net.bias_l4[:a4]  # 与 eps4 定义同源
            th_w04.mul_(0.98).add_(0.02 * (mu4_h * mu4_h).mean(dim=(0, 1)))
            g_homeo = (1.0 / (1.0 + th_w04)).unsqueeze(1)  # [a4,1] 每输出列增益
            net._theta_w04_dist = th_w04  # 诊断: θ_j 分布
            dW_04 = dW_04 * g_homeo
            # 第 105 轮 (资格迹): 迹接力 — E 记录机制链后的活动轨迹 (滑均),
            # 更新 = eta·R·E. R=1.0 → 现有更新的平滑版; R=生存信号 → 全迹缩放.
            dW_04 = dW_04 + _elig_accum(net, "W_04", dW_04) * getattr(net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16))
            net.W_04[:a4].data += dW_04 * eta
            soft_norm_preserve(net.W_04[:a4].data)
            # W_04 行去同质化 (与 W_42/W_23 同款, 上游输入侧秩 1 修复):
            # E_04 幂迭代主方向投影抑制, 破 W_04 行收敛 ±w 坍缩.
            # 双层安全: 斜坡渐进 (coef 200 步升到 1) + 范数信任域 (单步扰动 ≤5%‖W‖_F)
            # — 开环有界扰动, 防 z4 分布突变打穿混合层 |x|≤1.4 安全线 (step 9 NaN 根因)
            ramp = min(1.0, net._step_counter / 200.0)
            _decorr_W(net.W_04[:a4].data, net.E_04[:a4, :a4], coef=ramp, max_delta_ratio=0.05, learn_boost=ctx.learn_boost)

            dW42 = (sh.errs.eps2_p.transpose(-2, -1) @ _rms(z4)).mean(dim=0) * inv_s
            dW42 = _energy_constraint(net, net.W_42[:a2].data, dW42, sh.errs.eps2_p, "_act_ema_w42")
            dW42 = dW42 + _elig_accum(net, "W_42", dW42) * getattr(net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16))
            col_mask = torch.rand(a2, 1, device=dev) < net.cfg.column_dropout
            net.W_42[:a2].data += (dW42 * (~col_mask).to(torch.float16)) * eta
            # W_42 权重去同质化 (行收敛 ±w → 投影秩 1 根因)
            _decorr_W(net.W_42[:a2].data, net.E_42[:a2, :a2], learn_boost=ctx.learn_boost)
        # ── 字节域块结束 (free_run 跳过: W_lm 家族/W_04/W_42, 冻结) ──
        # 第 102 轮: echo_world_frozen 时 W_42 冻结 — 乱码输入流不配做感知结构
        # (W_04/W_42 是感知链, 回声相位只有行为域 W_act 学习)
        # W_56 保留更新 (L5→L6 内部动力学, 自由运行同样学习)
        dW_56 = (sh.errs.eps6_p.transpose(-2, -1) @ _rms(z5)).mean(dim=0) * inv_s
        dW_56 = _energy_constraint(net, net.W_56[:a6].data, dW_56, sh.errs.eps6_p, "_act_ema_w56")
        dW_56 = dW_56 + _elig_accum(net, "W_56", dW_56) * getattr(net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16))
        col_mask = torch.rand(net.W_56[:a6].shape[0], 1, device=dev) < net.cfg.column_dropout
        net.W_56[:a6].data += (dW_56 * (~col_mask).to(torch.float16)) * eta

        # 预测编码融合: eps_state = (z4 @ W_state_pred) - Δz4, 注入 W_23 表示层更新.
        # 底层不再只接收重构误差, 同时携带"未来往哪走"的预测误差 (纯线性叠加)
        W_sp_a = net.W_state_pred[:a4, :a4]
        dz4_full = net._z4[:, 1:] - net._z4[:, :-1]
        eps_state = (net._z4[:, :-1] @ W_sp_a.T) - dz4_full  # [N,S-1,a4]
        eps_state = _rms(eps_state)
        # a4 → a3 投影 (经 W_42 逆映射, 尺度匹配)
        eps_state_a3 = eps_state @ net.W_42[:a2].T[:, :a3]
        eps3_pc = sh.errs.eps3 + 0.3 * torch.cat(
            [eps_state_a3, torch.zeros(N, 1, a3, dtype=eps_state_a3.dtype, device=dev)], dim=1
        )
        eps3_pc = net.forward_engine._precise(eps3_pc)

        # L3 种子: W_23 随机增益 + 误差门控 (上游扰动级联到 L5 分散)
        # _gain_l3 是固定 [384, 384] 种子, L3 修剪后行数收缩, 需按当前活性行切片
        gain_l3 = net._gain_l3[:a3, :a3] if a3 < 384 else net._gain_l3[:a3, :]
        dW23 = (eps3_pc.transpose(-2, -1) @ _rms(z2)).mean(dim=0) * inv_s
        dW23 = _energy_constraint(net, net.W_23[:a3].data, dW23, eps3_pc, "_act_ema_w23")
        dW23 = dW23 + _elig_accum(net, "W_23", dW23) * getattr(net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16))
        err3_norm = sh.errs.eps3.pow(2).mean(dim=(0, 1)).sqrt() + 1e-8  # [a3] 每神经元
        gate3 = 0.1 + 0.9 * (err3_norm / err3_norm.max())
        dW23 = dW23 * gain_l3 * gate3.unsqueeze(1)
        c3_mask = torch.rand(a3, 1, device=dev) < net.cfg.column_dropout
        net.W_23[:a3].data += (dW23 * (~c3_mask).to(torch.float16)) * eta
        # 软范数保持 (0.8-1.2): 增益种子幅度差异保留, 权重有界防溢出
        soft_norm_preserve(net.W_23[:a3].data)
        # W_23 权重去同质化 (行收敛 ±w → 投影秩 1 根因)
        _decorr_W(net.W_23[:a3].data, net.E_23[:a3, :a3], learn_boost=ctx.learn_boost)
        # 状态预测矩阵自更新 (纯赫布): dW_sp = z4^T @ eps_state, 零 BP
        # 第 102 轮: echo_world_frozen 时冻结 — W_state_pred 是世界模型
        # (预测 z4 下一步), 乱码输入流不配塑造它
        # (原顺序在 W_lm 头之后; 此处域内原位前移 — 本步无任何块读它, 数值等价)
        if not echo_world_frozen:
            dW_sp = (net._z4[:, :-1].transpose(-2, -1) @ eps_state).mean(dim=0)
            dW_sp = _energy_constraint(net, W_sp_a.data, dW_sp, net._z4, "_act_ema_wsp")
            dW_sp = dW_sp + _elig_accum(net, "W_state_pred", dW_sp) * getattr(net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16))
            W_sp_a.data += dW_sp * eta
            soft_norm_preserve(W_sp_a.data)

    def _update_W35(self, ctx, sh):
        """W_35 统一矩阵更新 (撤销微柱硬切块): 预测编码融合误差全量驱动."""
        net = self.net
        dev, N, S = ctx.dev, ctx.N, ctx.S
        free_run, echo_world_frozen = ctx.free_run, ctx.echo_world_frozen
        a3, a5 = ctx.a3, ctx.a5
        z3, z5 = net._z3, net._z5
        if not free_run and not echo_world_frozen:
            # 预测编码向下平移: W_lm 预测误差 → a3 空间
            # eps_lm_proj 已算 [N,S-1,a4], 补零到 S 对齐完整序列, 经 W_42 逆映射到 a3
            eps_lm_proj_pad = torch.cat(
                [sh.lm.eps_lm_proj, torch.zeros(N, 1, ctx.a4, dtype=sh.lm.eps_lm_proj.dtype, device=dev)], dim=1
            )
            eps_lm_a3 = eps_lm_proj_pad @ net.W_42[:ctx.a2].T[:, :a3]
            eps_lm_a3 = _rms(eps_lm_a3)
        z5_b = z5
        eps_b = z5_b[:, 1:] - z5_b[:, :-1]  # 时序差分误差
        z3_b = z3[:, :-1]
        z3_bp = z3_b * (torch.rand_like(z3_b) > 0.3).to(torch.float16)
        if not free_run and not echo_world_frozen:
            # 预测编码融合: eps_b = 时序差分 + 0.5 × LM 预测误差投影回 L5 空间
            eps_lm_b = eps_lm_a3[:, :-1] @ net.W_35[:a5].T  # [N,S-1,a5]
            eps_lm_b = _rms(eps_lm_b)
            eps_b = eps_b + 0.5 * eps_lm_b
        # BCM 滑阈: theta = EMA(eps²), phi = eps(eps-theta); 先 RMS 归一化防 fp16 溢出
        eps_b = _rms(eps_b)
        th = net._theta_l5[:a5]
        e2 = (eps_b * eps_b).mean(dim=(0, 1))
        th.mul_(0.975).add_(0.025 * e2)
        phi_b = eps_b * (eps_b - th)
        # 第 84 轮: phi_b 按 max|phi_b| 标量归一化 — eps_b 稀疏尖峰 (墙移除后
        # z4 进非线性区) 使 phi_b 达 ~310, 其平方在 _energy_constraint 的 p2、
        # g_n 与 _rms 内部均溢出 fp16 (312²≈97500>65504) → inf → NaN
        # (b60 step 457 实证, phi_b_max=312.8). 标量 s=max|phi_b| 无平方绝不
        # 溢出; phi_b/s ≤1 后一切平方安全; s 是批内标量缩放 (同 _precise 的
        # std 归一化一族), 相对结构与 BCM 符号保留, rho_ctrl 已钳最终幅度.
        s_phi = phi_b.abs().max() + 1e-6  # 标量 (max 无平方, fp16 安全)
        phi_b = phi_b / s_phi
        # 样本显著性: g_n = surprise_n / Σ surprise_n (线性加权)
        g_n = (phi_b * phi_b).mean(dim=(1, 2)) + 1e-8
        # 第 84 轮: 零向量保护 — z5 坍缩到常数时 phi_b≡0 → g_n=0+1e-8 在 fp16 舍入为
        # 0 → 0/0 = NaN (b60 step 391 实证 g_n_max=nan). 与 _rms/g_04 同款 alive 掩码.
        g_sum = g_n.sum()
        g_alive = (g_sum > 1e-8).to(g_n.dtype)
        g_n = g_n / torch.where(g_alive > 0, g_sum, torch.ones_like(g_sum))
        Wb = net.W_35[:a5].data
        gain_mask = net._gain_mask[:a5, :a3]
        eta = ctx.eta
        # 误差门控: 高误差神经元主导更新, 低误差保 10% 下限
        err_norm = eps_b.pow(2).mean(dim=(0, 1)).sqrt() + 1e-8
        # 第 84 轮: 零向量保护 — eps_b≡0 时 err_norm≡0 → 0/0 = NaN (同 g_n 实证).
        e_max = err_norm.max()
        e_alive = (e_max > 1e-8).to(err_norm.dtype)
        upd_gate = 0.1 + 0.9 * (err_norm / torch.where(e_alive > 0, e_max, torch.ones_like(e_max)))
        # ── W35 资格调制 (第 73 轮): 单变量, 移除 Cos 惩罚 ──
        # q_i = 每行 (微柱) 的时间残差能量 — 时间差分无法解释的行获得更高
        # 学习资格: ΔW_i = q_i·η·ε·z3^T. 让"谁学习"由内部误差决定, 而非
        # 外部正交约束. 验证 Hebbian 能否通过内部塑性竞争产生稳定多方向生态
        z5_prev = torch.cat([torch.zeros(N, 1, a5, dtype=z5.dtype, device=dev), z5[:, :-1]], dim=1)
        z5_prev_n = _rms(z5_prev)
        z5_pred_t = z5_prev_n @ net.W_t5[:a5].T  # W_t5 时间预测
        # 第 84 轮: 残差平方前按 max|残差| 标量归一化 — 墙移除后 W_t5 增益自由游走
        # (实测 rho 23.5, 残差幅度 ~230), 直接 (z5 - z5_pred_t)² 在 fp16 下平方溢出
        # → inf → q_i = inf/inf = NaN → W_35/E_l5 NaN (b60_fix step 57 实证).
        # 标量 s = max|残差| 不涉及平方 (绝不溢出), 归一化后平方 ≤1; q_i 末尾
        # 本就除以 res_neuron.max(), 标量 s² 在分子分母中精确消去 → q_i 与原式
        # 数学全等, 零语义改变, 只阻断 inf→NaN 路径 (CLAUDE.md: pre-norm 非 clamp).
        res_t = z5[:, 1:] - z5_pred_t[:, 1:]  # [N,S-1,a5] 残差
        s = res_t.abs().max() + 1e-6  # 标量 (max 无平方, fp16 安全)
        res_neuron = ((res_t / s) ** 2).mean(dim=(0, 1))  # [a5] 每行相对残差能量
        q_i = res_neuron / (res_neuron.max() + 1e-6)  # [a5] 资格 ∈ [0,1], 分化
        n_sub = min(4, max(1, N))  # N=1 时防重复子集 (原代码 4× 更新潜伏 bug)
        sub = max(1, N // n_sub)
        inv_sm1 = 1.0 / (S - 1)
        dW_h = None
        for s in range(n_sub):
            sl = slice(s * sub, (s + 1) * sub)
            # 预除 (第 78 轮): phi_b 先 ×1/(S−1) 再 bmm — fp16 中间累加不超界, 数学等价
            dW_n = torch.bmm((phi_b[sl] * inv_sm1).transpose(-2, -1), _rms(z3_bp[sl]))
            dW_sub = (dW_n * g_n[sl, None, None]).sum(dim=0)
            dW_sub = dW_sub * gain_mask * upd_gate.unsqueeze(1)
            dW_sub = dW_sub * q_i.unsqueeze(1)  # 资格调制 (第 73 轮)
            b_mask = torch.rand(a5, 1, device=dev) < net.cfg.column_dropout
            dW_sub = dW_sub * (~b_mask).to(torch.float16)
            dW_h = dW_sub if dW_h is None else dW_h + dW_sub
        dW_h = _energy_constraint(net, Wb, dW_h, phi_b, "_act_ema_w35")
        dW_h = dW_h + _elig_accum(net, "W_35", dW_h) * getattr(net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16))
        dW_h = dW_h * eta
        # ── 通道级塑性控制 (第 75 轮): 预测连接时间尺度统一控制律 ──
        # ρ_i = ||ΔW_i||/||W_i||, s_i = clip(0.03/ρ_i, 0.005, 1.0).
        # 冻结 BCM/Hebbian/Cos 竞争: 唯一变量 = 外积/注入时间尺度.
        # 空间机制 (decorr/soft_norm) 不动, 应用顺序保持原样: hebb → soft_norm → decorr.
        nW = Wb.norm() + 1e-8  # decorr 前统一基准

        if net.cfg.adaptive_rho:
            net._rho_map = {}
        Wb.add_(_rho_ctrl(dW_h, Wb, "hebb", net))
        # 第 83 轮 (G8 v2): 前馈链突触缩放 — L5 神经元入纤增益随自身活动双向
        # 调节, 与 W_t 同款机制. 实证 (probe v4): 高墙下递归链被双向缩放+STP
        # 兜住 (rho 49, sat 6.9%), 但 W_35 更新路径无局部饱和 → fp16 溢出 NaN
        # (W_35/E_l5, step ~100). 此处把 G8 扩展到前馈链, 并豁免 soft_norm
        # (与 G1 对 W_t 的处理同构: 行范数 = 增益自由度归系统, soft_norm 是
        # 增益锁, 会每步抹平缩放造成的异质). 训练模式保持 soft_norm.
        if free_run and net.cfg.wt_syn_scaling:
            p2w5 = (z5 * z5).mean(dim=(0, 1))  # [a5] L5 窗内活动²
            emaw5 = net._act_ema_b5[:a5]  # L5 慢基线 (bias 泄漏段已更新)
            # 第 84 轮: 分母保护 1e-3 → 1e-6 — 低活动分支被 1e-3 地板淹没: z5 坍缩后
            # ema 衰减到 ~1e-4 时 scale=2·1e-4/(0+1e-4+1e-3)≈0.18 → 永久收缩 → W_35
            # 死亡螺旋 (b60 step 391 实证: Wb_max=0.0055, z5_max=0.0038). 1e-6 与
            # rel_n 家族 (STP U_eff) 同款保护, 让低活分支真正给出 scale→2 (防冻结).
            scale5 = 2.0 * emaw5 / (p2w5 + emaw5 + 1e-6)  # 双向突触缩放 (0,2]
            r835 = net.cfg.wt_syn_scaling_rate
            Wb.data.mul_((1.0 - r835) + r835 * scale5.unsqueeze(1))
        else:
            soft_norm_preserve(Wb)
        dW_corr = _decorr_W(Wb, net.E_l5[:a5, :a5], learn_boost=ctx.learn_boost)  # 空间机制 (原顺序: 最后), 返回修正量
        if net.cfg.adaptive_rho:
            rho_raw, rho_eff, s_h = net._rho_map["hebb"]
            cos_hc = (dW_h.flatten() @ dW_corr.flatten()) / (
                dW_h.norm() * dW_corr.norm() + 1e-8
            )
            net._rho_raw, net._rho_eff, net._s_h = rho_raw, rho_eff, s_h
            net._rho_corr = dW_corr.norm() / nW
            net._cos_hc = cos_hc

    def _final_softnorm(self, ctx, sh):
        """前馈权重软范数保持 (0.8-1.2): W_04/W_42/W_56 无 BCM 约束,
        长训累积溢出 fp16 → NaN; 幅度差异保留 (结构化非 clamp)
        (free_run: W_04/W_42 冻结不更新, 但保持范数约束仍安全 — 不动它们)
        第 102 轮: echo_world_frozen 时 W_04/W_42/W_diff 冻结, 不触碰
        """
        net = self.net
        free_run, echo_world_frozen = ctx.free_run, ctx.echo_world_frozen
        a4, a2, a6 = ctx.a4, ctx.a2, ctx.a6
        if not free_run and not echo_world_frozen:
            for W, a_sz in zip([net.W_04, net.W_42], [a4, a2]):
                soft_norm_preserve(W[:a_sz].data)
            # W_diff 同款软范数保持 (行范数)
            soft_norm_preserve(net.W_diff[:a4, :a4].data)
        soft_norm_preserve(net.W_56[:a6].data)
