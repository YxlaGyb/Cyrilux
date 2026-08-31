"""
时序学习域
多尺度软时间窗差分误差 + W_diff 应用 + W_t 时间核家族 + 谱守卫.
"""

from __future__ import annotations

import torch

from ...modulation import soft_norm_preserve
from model.modulation import rms_norm
from ._common import _decorr_W, _elig_accum, _energy_constraint, _MixinBase, _spectral_radius_guard


class TemporalMixin(_MixinBase):
    """时序域 (方法挂载到 LearningEngine)."""

    def _build_diff_window(self, ctx):
        """多尺度软窗 (2/4/8 并行因果卷积) 差分误差 + W_diff 目标构建.

        返回 DiffWindow | None (free_run / echo_world_frozen 时无 diff 目标).
        """
        net = self.net
        if ctx.free_run or ctx.echo_world_frozen:
            return None
        dev, N = ctx.dev, ctx.N
        dim_4 = ctx.dim_4
        W_diff_a = net.W_diff[:dim_4, :dim_4]
        dz4 = net._z4[:, 1:] - net._z4[:, :-1]
        dz4 = dz4 / (dz4.norm(dim=-1, keepdim=True) + 1e-3)  # RMS 归一化
        z4r = net._z4  # [N,S,dim_4]
        S_full = z4r.shape[1]
        masks = {}
        preds_k = {}
        errs_k = {}
        for k, kname in ((2, "2"), (4, "4"), (8, "8")):
            # 交互短输入收缩历史深度, 否则 z4r[:, :-k] 为空 → pred 形状错位
            k_eff = min(k, S_full - 1)
            z_shift = torch.cat([torch.zeros(N, k_eff, dim_4, dtype=z4r.dtype, device=dev), z4r[:, :-k_eff]], dim=1)
            z_shift_n = z_shift / (z_shift.norm(dim=-1, keepdim=True) + 1e-3)
            pred_k = z_shift_n @ W_diff_a.T + net.b_diff[:dim_4]
            err_k = (dz4 - pred_k[:, :-1]).square().mean()
            if k == 8:
                valid = torch.arange(S_full - 1, device=dev) >= (k_eff - 1)
                err_k = ((dz4[:, valid] - pred_k[:, :-1][:, valid]).square()).mean()
            masks[kname] = valid if k == 8 else None
            preds_k[kname] = pred_k
            errs_k[kname] = err_k
        # EMA 平滑各尺度误差 (α=0.9, 时间常数~10步)
        err_2, err_4, err_8 = (errs_k["2"], errs_k["4"], errs_k["8"])
        ema_2 = net._e_ema_2.mul_(0.1).add_(0.9 * err_2)
        ema_4 = net._e_ema_4.mul_(0.1).add_(0.9 * err_4)
        ema_8 = net._e_ema_8.mul_(0.1).add_(0.9 * err_8)
        # 软权重: w_k = (1/(e_ema_k+1e-3)) / Σ (加性保护非 clamp)
        inv2, inv4, inv8 = (1.0 / (ema_2 + 1e-3)), (1.0 / (ema_4 + 1e-3)), (1.0 / (ema_8 + 1e-3))
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
        th_w = net._theta_w[:dim_4]
        th_w.mul_(0.975).add_(0.025 * (pred_d * pred_d).mean(dim=(0, 1)))
        phi_w = pred_d * (pred_d - th_w)
        # 差动赫布外积: dW = z4_prev^T @ (e - 0.1*phi_w), 学"当前 L4 状态 → 移动方向"
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
        """W_diff 下一状态预测更新 (4 步时间窗平均外积) + b_diff 偏置 (free_run 冻结)."""
        net = self.net
        if ctx.free_run or ctx.echo_world_frozen:
            return
        dev = ctx.dev
        dim_4 = ctx.dim_4
        W_diff_a = net.W_diff[:dim_4, :dim_4]
        fut_mask = torch.rand(dim_4, 1, device=dev) < net.cfg.column_dropout
        dW_avg = sh.diff.dW_avg + _elig_accum(net, "W_diff", sh.diff.dW_avg) * getattr(net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16))
        W_diff_a.data += (dW_avg * (~fut_mask).to(torch.float16)) * ctx.eta
        future_e = (sh.diff.dz4 - sh.diff.pred_d).mean(dim=(0, 1))
        net.b_diff[:dim_4].data += future_e * ctx.eta

    def _update_wt_family(self, ctx, sh):
        """时序 Hebbian (W_t 学习, 高确定性时增强 → 记忆巩固) + 谱守卫."""
        net = self.net
        dev = ctx.dev
        free_run = ctx.free_run
        a_sizes = [ctx.dim_4, ctx.dim_2, ctx.dim_3, ctx.dim_5, ctx.dim_6]
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
                s = torch.ones_like(dz_n)  # 无静止掩码
            else:
                # 绝对阈值 = z 平均范数的 2%: 动态帧 1, 静止帧 0 (分位阈值在塌缩后失效)
                th = (post.norm(dim=-1).mean() * 0.02).unsqueeze(0)
                s = (dz_n < th).to(dz_n.dtype)
            dW_t = (rms_norm(pre).transpose(-2, -1) @ (rms_norm(post) * s.unsqueeze(-1))).mean(dim=0) * (
                1.0 / (ctx.S - 1)
            )
            # free_run 豁免 Oja 能量约束: 增益解放后 Oja 成死亡吸引子, 递归耗散改由 STP/分流/谱守卫承担
            if not free_run:
                dW_t = _energy_constraint(net, W_t[:a_sz, :a_sz].data, dW_t, z_cur, f"_active_ema_{wt_name}")
                dW_t = dW_t + _elig_accum(net, f"{wt_name[0].upper()}_{wt_name[1:]}", dW_t) * getattr(net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16))
            # W_t4 输出端 homeostatic 抑制: g_j=1/(1+θ_j), 破 σ₁/σ₂ 垄断, 与 W_04 同构
            if W_t is net.W_t4:
                th_wt4 = net._theta_wt4[:a_sz]
                th_wt4.mul_(0.98).add_(0.02 * (dz * dz).mean(dim=(0, 1)))
                dW_t = dW_t * (1.0 / (1.0 + th_wt4)).unsqueeze(1)
            W_t[:a_sz, :a_sz].data += dW_t * eta_t
            # 双向突触缩放 (自由运行): 入纤增益随"自身活动 vs 慢基线"双向调节,
            # 高活下调/低活上调, scale=2·ema/(p2+ema) ∈ (0,2], 自设定操作点
            if free_run and net.cfg.wt_syn_scaling:
                p2w = (z_cur * z_cur).mean(dim=(0, 1))  # [a_sz] 窗内活动²
                emaw = getattr(net, f"_active_ema_b{wt_name[2]}")[:a_sz]  # 同层慢基线
                scale = 2.0 * emaw / (p2w + emaw + 1e-6)  # 分母 1e-6 防低活分支冻结
                r83 = net.cfg.wt_syn_scaling_rate
                W_t[:a_sz, :a_sz].data.mul_((1.0 - r83) + r83 * scale.unsqueeze(1))
            # 软范数保持 (训练模式): 权重有界防 fp16 溢出; free_run 豁免 (递归增益自由度归系统)
            if not free_run:
                soft_norm_preserve(W_t[:a_sz, :a_sz].data)
            # 去主成分清除秩 1 结构; free_run coef 降至 0.2 (与 Oja 同向会形成死亡合力)
            if E_t is not None:
                _decorr_W(W_t[:a_sz, :a_sz].data, E_t[:a_sz, :a_sz], coef=0.2 if free_run else 1.0, learn_boost=ctx.learn_boost)

        # 谱半径守卫: 只在修剪置换/能量失效后发散时兜底, 不挡临界点之下的游走
        for wt in (net.W_t4, net.W_t2, net.W_t3, net.W_t5, net.W_t6):
            _spectral_radius_guard(wt.data, bound=net.cfg.spectral_guard_bound)
