"""时序学习域: 多尺度软时间窗误差 + W_diff 应用 + W_t 时间核家族 + 谱守卫.

从原 learn() 单函数按块拆分, 块内语句顺序逐行保持 (数值逐位等价).
"""

from __future__ import annotations

import torch

from ...modulation import soft_norm_preserve
from ..forward import _rms
from ._common import _decorr_W, _elig_accum, _energy_constraint, _MixinBase, _spectral_radius_guard


class TemporalMixin(_MixinBase):
    """时序域 (方法挂载到 LearningEngine)."""

    def _build_diff_window(self, ctx):
        """多尺度软窗 (2/4/8 并行因果卷积) 差分误差 + W_diff 目标构建 (原 L262-334).

        返回 DiffWindow | None (free_run / echo_world_frozen 时无 diff 目标).
        """
        net = self.net
        if ctx.free_run or ctx.echo_world_frozen:
            return None
        dev, N = ctx.dev, ctx.N
        a4 = ctx.a4
        W_diff_a = net.W_diff[:a4, :a4]
        dz4 = net._z4[:, 1:] - net._z4[:, :-1]
        dz4 = dz4 / (dz4.norm(dim=-1, keepdim=True) + 1e-3)  # RMS 归一化
        z4r = net._z4  # [N,S,a4]
        S_full = z4r.shape[1]
        masks = {}
        preds_k = {}
        errs_k = {}
        for k, kname in ((2, "2"), (4, "4"), (8, "8")):
            # 第 87 轮: 交互短输入 — 序列短于尺度时收缩历史深度, 否则
            # z4r[:, :-k] 为空 → pred 形状错位 (聊天 "你好"=6 字节, k=8 崩).
            # k_eff = min(k, S-1) 保证 z_shift 恒为 [N,S]; k=8 的 valid 掩码
            # 同步收缩 (仍非空, 防 .mean() 空选区 NaN).
            k_eff = min(k, S_full - 1)
            z_shift = torch.cat([torch.zeros(N, k_eff, a4, dtype=z4r.dtype, device=dev), z4r[:, :-k_eff]], dim=1)
            z_shift_n = z_shift / (z_shift.norm(dim=-1, keepdim=True) + 1e-3)
            pred_k = z_shift_n @ W_diff_a.T + net.b_diff[:a4]
            err_k = (dz4 - pred_k[:, :-1]).square().mean()
            if k == 8:
                valid = torch.arange(S_full - 1, device=dev) >= (k_eff - 1)
                err_k = ((dz4[:, valid] - pred_k[:, :-1][:, valid]).square()).mean()
            masks[kname] = valid if k == 8 else None
            preds_k[kname] = pred_k
            errs_k[kname] = err_k
        # EMA 平滑各尺度误差 (α=0.9, 时间常数~10步)
        e2, e4, e8 = (errs_k["2"], errs_k["4"], errs_k["8"])
        ema2 = net._e_ema_2.mul_(0.1).add_(0.9 * e2)
        ema4 = net._e_ema_4.mul_(0.1).add_(0.9 * e4)
        ema8 = net._e_ema_8.mul_(0.1).add_(0.9 * e8)
        # 软权重: w_k = (1/(e_ema_k+1e-3)) / Σ (加性保护非 clamp)
        inv2, inv4, inv8 = (1.0 / (ema2 + 1e-3)), (1.0 / (ema4 + 1e-3)), (1.0 / (ema8 + 1e-3))
        sum_inv = inv2 + inv4 + inv8
        w2, w4, w8 = inv2 / sum_inv, inv4 / sum_inv, inv8 / sum_inv
        net._w_soft[0] = w2
        net._w_soft[1] = w4
        net._w_soft[2] = w8
        pred_d = w2 * preds_k["2"][:, :-1] + w4 * preds_k["4"][:, :-1] + w8 * preds_k["8"][:, :-1]
        # 误差用掩码后的有效区 (K=8 前 7 步剔除)
        if masks["8"] is not None:
            e_t = (dz4[:, masks["8"]] - pred_d[:, masks["8"]]).detach()
            e_t_all = dz4 - pred_d
        else:
            e_t = dz4 - pred_d
            e_t_all = e_t
        # W_diff BCM 滑阈: theta_w = EMA(pred²), phi_w 衰减高活跃预测方向
        th_w = net._theta_w[:a4]
        th_w.mul_(0.975).add_(0.025 * (pred_d * pred_d).mean(dim=(0, 1)))
        phi_w = pred_d * (pred_d - th_w)
        # 差动赫布外积 (动力学映射): dW = z4_prev^T @ (e - 0.1*phi_w),
        # 输入是上下文 z4_prev 而非差分 dz4 — 学习"当前 L4 状态 → 移动方向"
        z4_prev_n = z4r[:, :-1] / (z4r[:, :-1].norm(dim=-1, keepdim=True) + 1e-3)
        e_mod = e_t_all - 0.1 * phi_w
        dW_diff_t = torch.bmm(e_mod.transpose(-2, -1), z4_prev_n).mean(dim=0) * (1.0 / (ctx.S - 1))
        # 4 步时间窗环形缓冲: 更新用最近 4 步平均外积 (保留误差记忆)
        buf = getattr(net, f"_dw_buf_{net._buf_i}")
        buf.copy_(dW_diff_t)
        dW_avg = buf.clone()
        for i in range(1, 4):
            dW_avg = dW_avg + getattr(net, f"_dw_buf_{(net._buf_i - i) % 4}")
        dW_avg = dW_avg * 0.25
        net._buf_i = (net._buf_i + 1) % 4

        from .engine import DiffWindow

        return DiffWindow(dz4=dz4, pred_d=pred_d, dW_avg=dW_avg, e_t_all=e_t_all)

    def _apply_diff(self, ctx, sh):
        """W_diff 下一状态预测更新 (4 步时间窗平均外积) + b_diff 偏置 (L4 空间).

        (free_run 跳过 — W_diff 冻结, 依赖外部字节序列; 回声相位同冻结)
        """
        net = self.net
        if ctx.free_run or ctx.echo_world_frozen:
            return
        dev = ctx.dev
        a4 = ctx.a4
        W_diff_a = net.W_diff[:a4, :a4]
        fut_mask = torch.rand(a4, 1, device=dev) < net.cfg.column_dropout
        dW_avg = sh.diff.dW_avg + _elig_accum(net, "W_diff", sh.diff.dW_avg) * getattr(net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16))
        W_diff_a.data += (dW_avg * (~fut_mask).to(torch.float16)) * ctx.eta
        future_e = (sh.diff.dz4 - sh.diff.pred_d).mean(dim=(0, 1))
        net.b_diff[:a4].data += future_e * ctx.eta

    def _update_wt_family(self, ctx, sh):
        """时序 Hebbian (W_t 学习, 高确定性时增强 → 记忆巩固) + 谱守卫 (原 L924-1011)."""
        net = self.net
        dev = ctx.dev
        free_run = ctx.free_run
        a_sizes = [ctx.a4, ctx.a2, ctx.a3, ctx.a5, ctx.a6]
        eta_t = ctx.eta_t
        for (z_cur, wt_name, W_t), a_sz, E_t in zip(
            [(net._z4, "wt4", net.W_t4), (net._z2, "wt2", net.W_t2), (net._z3, "wt3", net.W_t3), (net._z5, "wt5", net.W_t5), (net._z6, "wt6", net.W_t6)],
            a_sizes,
            [net.E_t4, net.E_t2, net.E_t3, net.E_t5, net.E_t6],
        ):
            pre = z_cur[:, :-1]
            post = z_cur[:, 1:]
            dz = post - pre  # Δz = z_t - z_{t-1}
            dz_n = dz.norm(dim=-1)  # [N, S-1]
            if free_run:
                s = torch.ones_like(dz_n)  # 第 81 轮: 无静止掩码 (见上)
            else:
                # 绝对阈值 = z 平均范数的 2% (闭环训练起放宽: 5%→2% 给长程记忆喘息,
                # 时序动力学稍作释放; 监视 W_t top1 秩防坍缩): 分位阈值在 z 塌缩后
                # 失效 (塌缩后"动态帧"也在主方向, 贡献仍秩 1); 绝对阈值让塌缩 →
                # 差分变小 → 静止占比暴增 → W_t 学习信号枯竭 → 自然平衡点 (频率锚点)
                th = (post.norm(dim=-1).mean() * 0.02).unsqueeze(0)
                s = (dz_n < th).to(dz_n.dtype)  # 静止帧 → 0, 动态帧 → 1
            dW_t = (_rms(pre).transpose(-2, -1) @ (_rms(post) * s.unsqueeze(-1))).mean(dim=0) * (
                1.0 / (ctx.S - 1)
            )
            # 内建能量约束 (第 78 轮): Oja + 活动依赖遗忘. 第 81 轮: free_run 豁免 —
            # 第 78 轮 Oja 是为"增益锁死"时代设计的; 增益解放后 Oja 成死亡吸引子
            # (bias 维持高活动 → Oja 力恒定 ~2.5%/步, Hebbian 力随 W_t 收缩 →
            # W_t 渐进归零, 实测 15k 步 12.4→1.0, 递归动力学死亡). 递归增益的
            # 物理耗散 = STP 资源耗尽 + 分流抑制 + 谱守卫 1.5 死亡保险 (三者均为
            # "偏离越大回拉越强"的力), Oja 冗余且致命. 训练模式保持约束
            if not free_run:
                dW_t = _energy_constraint(net, W_t[:a_sz, :a_sz].data, dW_t, z_cur, f"_act_ema_{wt_name}")
                # 第 105 轮 (资格迹): W_t 系滑均接力 (name 映射: wt4->W_t4)
                dW_t = dW_t + _elig_accum(net, f"{wt_name[0].upper()}_{wt_name[1:]}", dW_t) * getattr(net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16))
            # W_t4 输出端 homeostatic 抑制 (第 75 轮): 打破 σ₁/σ₂≈4000:1 垄断.
            # θ_j 跟踪 rec4 每维能量 (时序差分), g_j = 1/(1+θ_j) — 高激活列更新
            # 被压, rec4 从压制源转丰富源. 与 W_04 同构, 纯局部
            if W_t is net.W_t4:
                th_wt4 = net._theta_wt4[:a_sz]
                th_wt4.mul_(0.98).add_(0.02 * (dz * dz).mean(dim=(0, 1)))
                dW_t = dW_t * (1.0 / (1.0 + th_wt4)).unsqueeze(1)
            W_t[:a_sz, :a_sz].data += dW_t * eta_t
            # 第 83 轮 (G8): 双向突触缩放 — 权重级慢稳态增益, 仅自由运行.
            # 问题 (model81 实测): 增益自由度解开后, W_t 谱半径要么被 decorr 拖向
            # 0 (有序相冻结), 要么被 Hebbian 推到 1.5 守卫墙 (守卫成为增益设定者,
            # 公理 1 违规). 缺的是"系统自己设定操作点"的慢稳态机制.
            # 机制 (双向, Turrigiano 突触缩放): 神经元 i 的入纤递归增益 (W_t 行 i)
            # 随"自身活动 vs 自身慢基线"双向调节 — 高活下调 (遏制爆炸), 低活上调
            # (防止冻结/死亡). scale_i = 2·ema_i/(p2_i+ema_i): 基线=1, p2→∞→0,
            # p2→0→2, 天然有界 (0,2], 无 clamp. 与 STP U_eff (act/(act+ema)) 同族.
            # 以 wt_syn_scaling_rate (慢) 混合进权重: W_t[i,:] ← W_t[i,:]·((1−r)+r·scale_i)
            # 平衡点 = 动力学自洽不动点 (活动回到基线 = scale 1 = 不干预).
            # 纯局部, 自参照, 零新常数. 与 W_t4 θ 抑制同族, 推广到全部 5 层.
            # 与 U 自适应 (快) 分工: U = 单步释放概率负反馈, 突触缩放 = 权重级慢稳态.
            if free_run and net.cfg.wt_syn_scaling:
                p2w = (z_cur * z_cur).mean(dim=(0, 1))  # [a_sz] 窗内活动²
                emaw = getattr(net, f"_act_ema_b{wt_name[2]}")[:a_sz]  # 同层慢基线 (bias 泄漏段已更新)
                # 第 84 轮: 分母保护 1e-3 → 1e-6 (同 W_35 侧, 见该处注释) — 低活分支
                # 不被地板淹没, scale→2 防冻结.
                scale = 2.0 * emaw / (p2w + emaw + 1e-6)  # 双向突触缩放 (0,2]
                r83 = net.cfg.wt_syn_scaling_rate
                W_t[:a_sz, :a_sz].data.mul_((1.0 - r83) + r83 * scale.unsqueeze(1))
            # 软范数保持 (0.8-1.2): 权重有界防 fp16 溢出 (5万步 NaN 根因).
            # 第 81 轮: free_run 豁免 — 递归增益自由度归系统自己 (W_t 行范数
            # = 递归幅度, 是 E-I 平衡的物理变量). 平衡力已在 dW_t 内 (内建能量
            # 约束 = Oja + 活动依赖遗忘, 纯局部), 发散由 STP 资源耗尽/分流抑制/
            # 谱守卫 1.5 死亡保险兜底; 训练模式 (有外部输入) 保持软范数
            if not free_run:
                soft_norm_preserve(W_t[:a_sz, :a_sz].data)
            # 超量 E 清除 W_t 既有秩 1 结构 (top1 sv 11.4 固化 → 递归拉 z 秩 1)
            # 第 81 轮: free_run coef 降至 0.2 (M_l5 Foldiak 侧抑制的既有系数, 非
            # 新数值) — 1.0 超量抑制在无静止掩码下每步擦除主方向, 与 Oja 同向
            # 形成"修正压倒增长"的死亡合力 (15k 步实测 W_t 家族 12.4→1.0 渐进死亡)
            if E_t is not None:
                _decorr_W(W_t[:a_sz, :a_sz].data, E_t[:a_sz, :a_sz], coef=0.2 if free_run else 1.0, learn_boost=ctx.learn_boost)

        # 递归矩阵谱半径安全约束 (第 79 轮改判, 第 81 轮重定义): 死亡保险 ρ>bound
        # 才动作 — 正常动力学由能量耗散自然饱和 (递归增益自由度, 生命第一因).
        # 第 83 轮: bound 配置化 (spectral_guard_bound) — 实证 (exp83/probe) 表明
        # 1.5 嵌在自然增长区间内会每步钳制 → 守卫成控制器; bound 应高于自然
        # 工作区间, 区间内增益由突触缩放 (G8) + STP 耗尽设定.
        # 仅在修剪置换后子矩阵 ρ 发散或能量失效时兜底, 不挡临界点之下的游走
        for wt in (net.W_t4, net.W_t2, net.W_t3, net.W_t5, net.W_t6):
            _spectral_radius_guard(wt.data, bound=net.cfg.spectral_guard_bound)
