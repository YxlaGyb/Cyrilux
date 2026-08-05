"""LearningEngine

密集 PPA Hebbian 学习 (零反传, 零误差回路).

机制:
- 前馈权重: 逐层预测误差驱动 (标准 PC 自下而上), L3 加随机增益+门控种子
- 微柱 W_35: 块内 BCM 滑阈 + 样本显著性加权 + 增益/剪切/门控掩码
- W_diff: 增量预测 (dz5 = z5[t]-z5[t-1]), 多尺度软窗 + 4 步时间窗 + 独立 BCM
- 时序 W_t: 共现 Hebbian, ACh 调制; 无 Oja, BCM 承担稳定性
- 精度调度: 多巴胺/ACh 调制全局学习率, 前 50 步 W_diff 减半

全 fp16, 零 .float(), 零 autograd.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ..modulation import compute_ach_gain, compute_dopamine_gain, soft_norm_preserve
from .forward import _rms

if TYPE_CHECKING:
    from .network import DensePCNet


class LearningEngine:
    """学习引擎: 持 net 引用, 复用 _predict 存的 _z* 状态."""

    def __init__(self, net: DensePCNet):
        self.net = net

    def learn(self, byte_ids: torch.Tensor) -> dict:
        """Hebbian 学习 (零反传, 零误差回路). 不接收 targets.

        Args:
            byte_ids: [N, S] long 输入.

        Returns:
            stats dict (future_err, 各层误差范数).
        """
        net = self.net
        _ = net.forward_engine._predict(byte_ids, store_state=True)
        N, S = byte_ids.shape
        dev = byte_ids.device
        a4, a2, a3, a5, a6 = (net.active_size[k] for k in ("l4", "l2", "l3", "l5", "l6"))
        a_sizes = [a4, a2, a3, a5, a6]

        W_23_a = net.W_23[:a3]
        W_diff_a = net.W_diff[:a4, :a4]

        z0 = net._z0
        z4, z2, z3, z5, z6 = net._z4, net._z2, net._z3, net._z5, net._z6

        # 逐层预测误差 (自下而上 PC); L5 用时序差分误差, L6 用时间自预测
        eps4 = z4 - (z0 @ net.W_04[:a4].T + net.bias_l4[:a4])
        eps2 = z2 - (z4 @ net.W_42[:a2].T + net.bias_l2[:a2])
        eps3 = z3 - (z2 @ W_23_a.T + net.bias_l3[:a3])
        z6_pre = torch.cat([torch.zeros(N, 1, a6, dtype=z6.dtype, device=dev), z6[:, :-1]], dim=1)
        eps6 = z6 - z6_pre
        eps5_td = z5[:, 1:] - z5[:, :-1]  # L5: 跨时刻变化 z5[t]-z5[t-1]

        # 下一状态预测 (显式预测目标): pred_delta = z4 @ W_diff, target_delta = z4[t] - z4[t-1]
        # eps_diff = pred_delta - target_delta (训练误差); 多尺度软窗同构保留
        dz4 = net._z4[:, 1:] - net._z4[:, :-1]
        dz4 = dz4 / (dz4.norm(dim=-1, keepdim=True) + 1e-3)  # RMS 归一化
        z4r = net._z4  # [N,S,a4]
        S_full = z4r.shape[1]
        masks = {}
        preds_k = {}
        errs_k = {}
        for k, kname in ((2, "2"), (4, "4"), (8, "8")):
            z_shift = torch.cat([torch.zeros(N, k, a4, dtype=z4r.dtype, device=dev), z4r[:, :-k]], dim=1)
            z_shift_n = z_shift / (z_shift.norm(dim=-1, keepdim=True) + 1e-3)
            pred_k = z_shift_n @ W_diff_a.T + net.b_diff[:a4]
            err_k = (dz4 - pred_k[:, :-1]).square().mean()
            if k == 8:
                valid = torch.arange(S_full - 1, device=dev) >= 7
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
        dW_diff_t = torch.bmm(e_mod.transpose(-2, -1), z4_prev_n).mean(dim=0) * (1.0 / (S - 1))
        # 4 步时间窗环形缓冲: 更新用最近 4 步平均外积 (保留误差记忆)
        buf = getattr(net, f"_dw_buf_{net._buf_i}")
        buf.copy_(dW_diff_t)
        dW_avg = buf.clone()
        for i in range(1, 4):
            dW_avg = dW_avg + getattr(net, f"_dw_buf_{(net._buf_i - i) % 4}")
        dW_avg = dW_avg * 0.25
        net._buf_i = (net._buf_i + 1) % 4

        # 精度调度: 多巴胺/ACh 调制学习率; 前 50 步 W_diff 学习率减半 (先稳后放)
        surprise = (
            eps4.square().mean()
            + eps2.square().mean()
            + eps3.square().mean()
            + eps5_td.square().mean()
            + eps6.square().mean()
        ) * 0.2
        rel = float((surprise / (net._surprise_buf + 1e-4)).detach())
        net._surprise_buf.data.mul_(0.95).add_(0.05 * surprise.data)
        dop_gain = torch.tensor(compute_dopamine_gain(rel, 0.3, 5.0), dtype=torch.float16, device=dev)
        ach_gain = torch.tensor(compute_ach_gain(rel, 3.0), dtype=torch.float16, device=dev)
        eta = net.cfg.lr_hebbian * dop_gain
        if net._step_counter < 50:
            eta = eta * 0.5
        eta_t = eta * net.cfg.temporal_lr_ratio * ach_gain
        # 动态稳态竞争 (速率自适应): 按 W_lm 熵斜率调节表示层更新幅度 —
        # 熵加速下降 (W_lm 预测好) → scale→0 表示层放慢; 熵停滞/上升 → scale→2
        # 表示层放大 (强迫重组供新信息). scale 每 100 步更新, 用上一步值 (滞后一步无影响)
        if net.cfg.adaptive_traction:
            eta = eta * net._traction_scale.to(torch.float16)
            eta_t = eta_t * net._traction_scale.to(torch.float16)
        eta_lm = net.cfg.lm_lr_boost

        # Hebbian 外积 (逐层误差 ⊗ pre 活动)
        fe = net.forward_engine
        eps2_p, eps6_p = (
            fe._precise(eps2),
            fe._precise(eps6),
        )
        # ── 预测编码闭环: W_lm 预测误差投影回 z4, 作为表示层 top-down 误差 ──
        # eps_lm_proj = eps_lm @ W_lm.T: 表示层被迫为"预测下一字节"重组编码,
        # 而非只重构当前字节. 纯赫布, 零 BP (大脑皮层最核心的闭环)
        logits_lm = z4 @ net.W_lm[:a4] + net.bias_lm  # [N,S,256]
        target_lm = F.one_hot(byte_ids[:, 1:], num_classes=256).to(torch.float16)
        # 赫布版 softmax 误差: eps = target - softmax(logits) (概率尺度 0-1).
        # 原始 target - logits 的负信号被 logits 幅度主导 (熵 5.5 时 logit~0 但非目标位
        # 255 项累积淹没目标位); softmax 后目标位概率 1/256, 误差信号与概率匹配
        probs_lm = torch.softmax(logits_lm.float(), dim=-1).to(torch.float16)  # [N,S,256]
        eps_lm = (target_lm - probs_lm[:, :-1]).detach()  # [N,S-1,256]

        # 动态稳态竞争: 每步记录 batch 级 W_lm 熵 (fp32 调度域, log 精度需 fp32),
        # 连续负反馈: 20 步窗口最小二乘斜率 (线性拟合滤噪, 零超参),
        # scale = 2/(1+exp(-slope20/σ)): 熵降 (slope20<0) → scale→0 表示层放慢
        # 保护成果; 熵停滞/上升 → scale→2 表示层放大强迫重组. 有界无 clamp
        # slope20 = 20 步熵总变化 (nats), σ = 窗口熵波动 (nats), 比值无量纲
        if net.cfg.adaptive_traction:
            ent = -(probs_lm.float() * torch.log(probs_lm.float() + 1e-9)).sum(dim=-1).mean()
            net._ent_buf[net._ent_i % 20].copy_(ent.detach())
            net._ent_i += 1
            if net._ent_i >= 20:
                idx = (net._ent_i - 19 + torch.arange(20, device=dev)) % 20
                w = net._ent_buf[idx]  # 按时间正序重排
                slope20 = (net._t_center * w).sum() / net._t_denom * 20.0  # 20 步总变化
                sigma = w.std() + 1e-4
                net._traction_scale.copy_(2.0 / (1.0 + torch.exp(-slope20 / sigma)))
        eps_lm_proj = eps_lm @ net.W_lm[:a4].T  # [N,S-1,a4]
        eps_lm_pad = torch.cat(
            [eps_lm_proj, torch.zeros(N, 1, a4, dtype=eps_lm_proj.dtype, device=dev)], dim=1
        )

        # ── W_04 主辅误差交换: 预测误差为主, 重建为辅 ──
        # 重建任务不需要词序 (稳定信号拉权重回单一解); 预测误差才需要词序.
        # final_error = err_pred_norm + 0.2 * err_recon_norm (量级对齐)
        # 突触后增益控制: 归一化基于当前误差自身的 std (统计去耦), 不依赖维度 —
        # 修剪缩小 L4 时误差方差自然变小, 分母自动适应, 无 1/A4 静态系数
        inv_s = 1.0 / S
        # 突触后增益控制 (std 归一化) + 相对地板: 地板 = 全局 std 的 0.1%,
        # 随信号缩放 (修剪缩维 → 全局 std 自动降), 防 fp16 精度极限下除零放大
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
        dW_04n = dW_04n / (nrm_04 + 1e-6)  # 单位向量 (只保留方向)
        e_norm = final_error.norm(dim=(1, 2))
        g_04 = (e_norm / (e_norm.max() + e_norm.std() + 1e-6)).unsqueeze(-1).unsqueeze(-1)
        dW_04 = (dW_04n * g_04).sum(dim=0) / (g_04.sum() + 1e-6)
        net.W_04[:a4].data += dW_04 * eta
        soft_norm_preserve(net.W_04[:a4].data)

        dW_list = [
            (eps2_p.transpose(-2, -1) @ _rms(z4)).mean(dim=0) * inv_s,
            (eps6_p.transpose(-2, -1) @ _rms(z5)).mean(dim=0) * inv_s,
        ]
        W_list = [net.W_42[:a2], net.W_56[:a6]]
        for dW, W in zip(dW_list, W_list):
            col_mask = torch.rand(W.shape[0], 1, device=dev) < net.cfg.column_dropout
            W.data += (dW * (~col_mask).to(torch.float16)) * eta

        # 预测编码融合: eps_state = (z4 @ W_state_pred) - Δz4, 注入 W_23 表示层更新.
        # 底层不再只接收重构误差, 同时携带"未来往哪走"的预测误差 (纯线性叠加)
        W_sp_a = net.W_state_pred[:a4, :a4]
        dz4_full = net._z4[:, 1:] - net._z4[:, :-1]
        eps_state = (net._z4[:, :-1] @ W_sp_a.T) - dz4_full  # [N,S-1,a4]
        eps_state = _rms(eps_state)
        # a4 → a3 投影 (经 W_42 逆映射, 尺度匹配)
        eps_state_a3 = eps_state @ net.W_42[:a2].T[:, :a3]
        eps3_pc = eps3 + 0.3 * torch.cat(
            [eps_state_a3, torch.zeros(N, 1, a3, dtype=eps_state_a3.dtype, device=dev)], dim=1
        )
        eps3_pc = fe._precise(eps3_pc)

        # L3 种子: W_23 随机增益 + 误差门控 (上游扰动级联到 L5 分散)
        # _gain_l3 是固定 [384, 384] 种子, L3 修剪后行数收缩, 需按当前活性行切片
        gain_l3 = net._gain_l3[:a3, :a3] if a3 < 384 else net._gain_l3[:a3, :]
        dW23 = (eps3_pc.transpose(-2, -1) @ _rms(z2)).mean(dim=0) * inv_s
        err3_norm = eps3.pow(2).mean(dim=(0, 1)).sqrt() + 1e-8  # [a3] 每神经元
        gate3 = 0.1 + 0.9 * (err3_norm / err3_norm.max())
        dW23 = dW23 * gain_l3 * gate3.unsqueeze(1)
        c3_mask = torch.rand(a3, 1, device=dev) < net.cfg.column_dropout
        net.W_23[:a3].data += (dW23 * (~c3_mask).to(torch.float16)) * eta
        # 软范数保持 (0.8-1.2): 增益种子幅度差异保留, 权重有界防溢出
        soft_norm_preserve(net.W_23[:a3].data)

        # ── 第一步: 换轴 — 资格迹+样本竞争 (bmm 逐样本, 显著性加权, 不批平均) ──
        # dW_n = bmm(eps^T, z) [N, d_out, d_in]; gate_n = surprise_n / Σ surprise_n
        # 线性加权 (禁止 exp/非线性), 保留样本差异, 抹平效应消除
        n_sub = 4
        sub = max(1, N // n_sub)
        # 预测编码向下平移: W_lm 预测误差 → a3 空间
        # eps_lm_proj 已算 [N,S-1,a4], 补零到 S 对齐完整序列, 经 W_42 逆映射到 a3
        eps_lm_proj_pad = torch.cat(
            [eps_lm_proj, torch.zeros(N, 1, a4, dtype=eps_lm_proj.dtype, device=dev)], dim=1
        )
        eps_lm_a3 = eps_lm_proj_pad @ net.W_42[:a2].T[:, :a3]
        eps_lm_a3 = _rms(eps_lm_a3)
        # 空间软竞争 (微柱学习率差异化): 各块独立算时序预测误差, 误差大的柱更新慢.
        # 打破 W_35 块间收敛同步 (实测 cos 相似度 0.72→0.83 均质化) — 同误差信号同更新
        # 规则必然同步; 竞争让 3 高频柱 + 1 低频柱分化
        for b in range(net.n_blocks):
            b_s = slice(b * net.b5, (b + 1) * net.b5)
            z5_b = z5[:, b :: net.n_blocks, b_s]  # 该块路由步的激活
            eps_b = z5_b[:, 1:] - z5_b[:, :-1]  # 块内时序差分误差
            z3_b = z3[:, b :: net.n_blocks, :][:, :-1]
            z3_bp = z3_b * (torch.rand_like(z3_b) > 0.3).to(torch.float16)
            # 预测编码融合: eps_b = 时序差分 + 0.5 × LM 预测误差投影回微柱空间.
            # 高层 (LM 头) 预测错了 → 告诉 W_35 "你给的微柱特征缺词序信息, 需改"
            # eps_lm_b 先 RMS 归一化防投影幅度溢出 (特定 batch 长文本 → 超大误差)
            eps_lm_b = eps_lm_a3[:, b :: net.n_blocks, :] @ net.W_35[b].T  # [N, S//n, b5]
            eps_lm_b = eps_lm_b[:, 1:]  # 对齐 eps_b (差分后 S//n - 1)
            eps_lm_b = _rms(eps_lm_b)
            eps_b = eps_b + 0.5 * eps_lm_b
            # 空间竞争权重: 块内原始误差 (归一化前, L1 防平方溢出), 线性加权 — 误差大的柱更新慢
            err_b = eps_b.abs().mean() + 1e-8
            # BCM 滑阈: theta = EMA(eps²), phi = eps(eps-theta); 先 RMS 归一化 eps,
            # 防 eps 平方级增长在 fp16 溢出 (16774 步首爆 W_35[2] 根因)
            eps_b = _rms(eps_b)
            th = net._theta_l5[b * net.b5 : (b + 1) * net.b5]
            e2 = (eps_b * eps_b).mean(dim=(0, 1))
            th.mul_(0.975).add_(0.025 * e2)
            phi_b = eps_b * (eps_b - th)
            # 样本显著性: gate_n = surprise_n / Σ surprise_n (线性加权)
            g_n = (phi_b * phi_b).mean(dim=(1, 2)) + 1e-8
            g_n = g_n / g_n.sum()
            Wb = net.W_35[b].data
            # 固定 10% 突触剪切 / 静态随机增益 [0.5,1.5]; 随 L3 修剪裁列
            syn_mask = getattr(net, f"_syn_mask_{b}")[:, :a3]
            gain_mask = getattr(net, f"_gain_mask_{b}")[:, :a3]
            # 误差门控: 高误差神经元主导更新, 低误差保 10% 下限
            err_norm = eps_b.pow(2).mean(dim=(0, 1)).sqrt() + 1e-8
            upd_gate = 0.1 + 0.9 * (err_norm / err_norm.max())
            # 空间竞争学习率: eta_b = eta / (1 + err_b_rel), err 大的柱更新慢
            eta_b = eta / (1.0 + err_b.detach())
            for s in range(n_sub):
                sl = slice(s * sub, (s + 1) * sub)
                dW_n = torch.bmm(phi_b[sl].transpose(-2, -1), _rms(z3_bp[sl]))
                dW_sub = (dW_n * g_n[sl, None, None]).sum(dim=0) * (1.0 / (S // net.n_blocks - 1))
                dW_sub = dW_sub * gain_mask * syn_mask * upd_gate.unsqueeze(1)
                b_mask = torch.rand(net.b5, 1, device=dev) < net.cfg.column_dropout
                Wb += (dW_sub * (~b_mask).to(torch.float16)) * eta_b
                # 软范数保持 (0.8-1.2): W_35 微柱无 Oja, 长训累积溢出 fp16 → NaN
                # (16774 步首爆 W_35[2]); 幅度差异保留 (结构化非 clamp)
                soft_norm_preserve(Wb)

        # Foldiak 反赫布: 侧向矩阵 M 协方差去相关 (零对角线)
        # dM 先除范数平方再平均: cov 元素 = z_i·z_j 内积, z5 幅度 ~13 时 256 项和可超
        # fp16 上限 65504 (trace 实测 z5~12 时 cov 已 209; z5 稍大即溢出). 归一化到
        # ~0(1) 再积分, 消除溢出路径; M 行范数受 0.8-1.2 保持约束不长爆
        z5_flat = z5.reshape(-1, a5)
        dM = z5_flat.transpose(0, 1) @ z5_flat
        dM = dM / (dM.norm() + 1e-3)
        eye_mask = 1.0 - torch.eye(a5, device=dev, dtype=torch.float16)
        net.M_l5[:a5, :a5].data += (dM * eye_mask) * (0.005 * ach_gain) * eta
        soft_norm_preserve(net.M_l5[:a5, :a5].data)

        # W_diff 下一状态预测更新 (4 步时间窗平均外积) + b_diff 偏置 (L4 空间)
        fut_mask = torch.rand(a4, 1, device=dev) < net.cfg.column_dropout
        W_diff_a.data += (dW_avg * (~fut_mask).to(torch.float16)) * eta
        future_e = (dz4 - pred_d).mean(dim=(0, 1))
        net.b_diff[:a4].data += future_e * eta

        # 时序 Hebbian (W_t 学习, 高确定性时增强 → 记忆巩固)
        for (z_cur, W_t), a_sz in zip(
            [(z4, net.W_t4), (z2, net.W_t2), (z3, net.W_t3), (z5, net.W_t5), (z6, net.W_t6)], a_sizes
        ):
            pre = z_cur[:, :-1]
            post = z_cur[:, 1:]
            dW_t = (_rms(pre).transpose(-2, -1) @ _rms(post)).mean(dim=0) * (1.0 / (S - 1))
            W_t[:a_sz, :a_sz].data += dW_t * eta_t
            # 软范数保持 (0.8-1.2): 权重有界防 fp16 溢出 (5万步 NaN 根因)
            soft_norm_preserve(W_t[:a_sz, :a_sz].data)

        # LM 头自监督赫布 (复用闭环段已算的 logits_lm/target_lm/eps_lm):
        # 突触前增益控制: error 先 RMS 缩放到单位能量再外积 — 更新幅度完全由
        # 内部误差能量自适应决定, 不依赖外部 eta 缩放 (替代 W_04 解码死锁)
        # 冻结模式下 ×lm_lr_boost (情况 A: 表示层冻结后 W_lm 强化学习)
        # 指数遗忘 (0.999/步, 纯乘法): 单位能量外积在目标附近振荡 (无阻尼 LMS),
        # 遗忘项提供阻尼让权重收敛而非翻转
        err_norm = eps_lm.norm(dim=-1, keepdim=True) * 1.01
        # 零向量保护: 某位置预测完美 (softmax 概率 fp16 舍入到 1.0) 时 eps_lm
        # 全零 → 0/0 = NaN, 冻结长跑 ~2000 步偶发爆 W_lm. 掩码: 零范数行分母=1
        alive = (err_norm > 1e-8).to(eps_lm.dtype)
        denom = torch.where(alive > 0, err_norm, torch.ones_like(err_norm))
        err_scaled = eps_lm * alive / denom  # 单位能量, 方向保留

        # W_lm 专属 BCM 滑阈 (防输出过冲): theta = EMA(logits²) (快 0.99 响应),
        # phi_wlm = logits_n·(logits_n - theta) — W_lm 开始输出高频极值时 theta 快速
        # 升高 → logits_n-theta<0 → phi 变负 → dW 修正反向 → 抑制过冲.
        # 纯机制剪刀, 线性, 零 BP; 0.1 系数镜像 W_diff BCM (同模式)
        # logits 先 RMS 归一化再进 BCM: 原始 logits ~52, 平方超 fp16 上限 (W_diff 同款)
        logits_n = _rms(logits_lm.detach())
        th_wlm = net._theta_wlm
        th_wlm.mul_(0.01).add_(0.99 * (logits_n * logits_n).mean(dim=(0, 1)))
        phi_wlm = logits_n * (logits_n - th_wlm)
        phi_wlm = _rms(phi_wlm)
        dW_lm = (z4[:, :-1].transpose(-2, -1) @ (err_scaled - 0.1 * phi_wlm[:, :-1])).mean(dim=0)  # [a4,256]
        net.W_lm[:a4].data.mul_(0.999)
        net.W_lm[:a4].data += dW_lm * eta_lm
        net.bias_lm.data += (err_scaled - 0.1 * phi_wlm[:, :-1]).mean(dim=(0, 1)) * eta_lm
        soft_norm_preserve(net.W_lm[:a4].data)

        # 状态预测矩阵自更新 (纯赫布): dW_sp = z4^T @ eps_state, 零 BP
        W_sp_a.data += (net._z4[:, :-1].transpose(-2, -1) @ eps_state).mean(dim=0) * eta
        soft_norm_preserve(W_sp_a.data)

        # 稀疏绑定层赫布更新 (只更新 top-k 激活行): dW_bind = z5^T @ sparse
        # 激活神经元 = 竞争胜出的"离散符元", 其连接被强化, 其余行不更新
        if hasattr(net, "_bind_sparse"):
            z5_nrm = z5.norm(dim=-1, keepdim=True)
            z5_alive = (z5_nrm > 1e-8).to(z5.dtype)
            z5_n = z5 * z5_alive / (z5_nrm * 1.01 + 1e-8 * (1 - z5_alive))
            dW_bind = (z5_n.transpose(-2, -1) @ net._bind_sparse).mean(dim=0)
            net.W_bind[:a5].data += dW_bind * eta
            soft_norm_preserve(net.W_bind[:a5].data)

        # 前馈权重软范数保持 (0.8-1.2): W_04/W_42/W_56 无 BCM 约束,
        # 长训累积溢出 fp16 → NaN; 幅度差异保留 (结构化非 clamp)
        for W, a_sz in zip([net.W_04, net.W_42, net.W_56], [a4, a2, a6]):
            soft_norm_preserve(W[:a_sz].data)
        # W_diff 同款软范数保持 (行范数)
        soft_norm_preserve(W_diff_a.data)

        # ── 拓扑重塑 ──
        net._step_counter += 1
        if net._step_counter > net.cfg.prune_warmup and net._step_counter % net.cfg.prune_interval == 0:
            net.pruner._prune()

        stats = {
            "free_energy": (
                eps4.square().mean()
                + eps2.square().mean()
                + eps3.square().mean()
                + eps5_td.square().mean()
                + eps6.square().mean()
            ),
            "future_err": (dz4 - pred_d).square().mean(),
            "surprise": surprise,
            "dop_gain": dop_gain,
            "ach_gain": ach_gain,
        }
        # 释放每步状态引用 (显存按需): _z* 是 store_state 存的大张量,
        # 不释放则 caching allocator 无法复用, 4GB 卡上逐步累积到 OOM
        for k in ("_z0", "_z4", "_z2", "_z3", "_z5", "_z5_raw", "_z6"):
            if hasattr(net, k):
                delattr(net, k)
        return stats
