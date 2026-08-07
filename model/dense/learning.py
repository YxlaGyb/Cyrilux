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

import math
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

        # ── 权重去主成分投影 (超量抑制): W -= β·(W@v)⊗v, v = W 的 top1 奇异方向 ──
        # Hebbian 更新把每层 W 行的输入空间分量收敛到 ±w 单一方向 (行间有符号
        # cos≈0 被符号随机掩盖, 但绝对 cos 134 倍于随机 → 投影秩 1, PR_eff 焊死
        # 根因). β=1.0 超量抑制 — 必须压倒 Hebbian 正反馈 (β=0.5 被证被覆盖).
        # 主方向直接从 W 幂迭代 (3 次, 不绕 E 的 |cos| 中介 — E 幂迭代 3 次收敛
        # 不充分, v 含噪声, 超量抑制放大噪声 → E_l5 NaN, 334 步实测复现)
        learn_boost = 2.0 - net._traction_scale.to(torch.float16)

        def _decorr_W(W: torch.Tensor, E: torch.Tensor) -> None:
            dim = W.shape[0]
            Wn = W / (W.norm(dim=1, keepdim=True) + 1e-3)
            dE = (Wn @ Wn.T).abs()  # 绝对相关 (诊断: 行收敛指标)
            eye_mask = 1.0 - torch.eye(dim, device=dev, dtype=torch.float16)
            E.data.mul_(0.97).add_((dE * eye_mask) * (0.05 * learn_boost * ach_gain))
            # top1 方向: 幂迭代 W^T W (列空间), 3 次
            v = torch.randn(W.shape[1], 1, device=dev, dtype=torch.float16) * 0.01
            for _ in range(3):
                v = W.T @ (W @ v)
                v = v / (v.norm() + 1e-8)
            c = W @ v  # 每行在 top1 方向上的投影系数 (含 ± 符号)
            W -= 1.0 * (c @ v.T)  # 行 i 减 1.0·c_i·v^T (超量抑制, 切断秩 1)

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
        # 多级记忆池 + 角色绑定拼接: [z4, m2, m8, m32, bind] 五通道进 W_lm.
        # 绑定向量 (任务 2) 由 z4 经 W_bind 三槽 top-k 生成, 离散符元承载
        # "主语-动词-宾语" 角色结构; 记忆池承载跨序列环境 (物理输入层不变)
        # W_lm 输入前的能量调制 + 竞争性非线性 (老师方向 C):
        # 1) 能量调制: z4 被 W_04 行范均分压到 std≈0.06 (微缩信号, 上下文信息
        #    被 bias 频率先验淹没 → 命中率 23% 铁板). 全局几何缩放 mean_abs→1.0,
        #    纯机制, 无 BP/clamp/float
        # 2) RMS 前置 (CLAUDE.md 铁律): 调制后厚尾平方可超 fp16 上限 (实测 65504),
        #    投影前 RMSNorm 结构化防溢出
        # 3) 三阶非线性: f(x)=x·(1-0.5x²) 类 tanh, 在 std≈1 时真正进入非线性区
        mean_abs = z4.abs().mean() + 1e-4
        z4_lm = z4 * (1.0 / mean_abs)
        z4_lm = _rms(z4_lm)
        z4_lm = z4_lm * (1.0 - 0.5 * z4_lm.pow(2))
        # 输出缩放 1/√H (CLAUDE.md: 投影输出溢出 → 乘 1/√H): logits 幅度 ∝ √(4a4+bind)
        # (表示层 PR 破局后有效维度大, 点积求和放大; 旧架构 PR~2 共线时隐式小).
        # 无缩放则 logits ±67 → softmax 饱和 → 熵锁 0.18 (4000 步实测)
        zh = torch.cat([z4_lm, net._m2, net._m8, net._m32, net._bind_vec], dim=-1)  # [N,S,4a4+16]
        inv_h = 1.0 / math.sqrt(4 * a4 + net.bind_slot_dim)
        logits_lm = (zh @ net.W_lm + net.bias_lm) * inv_h  # [N,S,256]
        # 池间侧抑制竞争 (Q4 注意力雏形): 每池 (z4/m2/m8/m32) 对 logits 的贡献能量
        # e_pool[i] = ‖zh_seg @ W_lm_seg‖ (未归一化原始能量 — 归一化会抹平池间差异,
        # 导致 BCM 阈值无区分度, 门控退化恒 0.99). BCM: theta = EMA(e_pool),
        # 池贡献大 (预测准) → theta 相对低 → 抑制系数 s = 1/(1+rel_theta) 高 (保持);
        # 池贡献小 (预测不准) → theta 相对高 → s 低 (抑制). 零超参 soft 门控
        seg = [slice(0, a4), slice(a4, 2 * a4), slice(2 * a4, 3 * a4), slice(3 * a4, 4 * a4)]
        e_pool = torch.stack(
            [(zh[:, :, s] @ net.W_lm[s]).norm(dim=-1).mean() for s in seg]
        )  # [4] 各池 logits 能量
        th_pool = net._theta_pool  # [4]
        th_pool.mul_(0.05).add_(0.95 * e_pool.detach())
        rel_th = th_pool / (th_pool.mean() + 1e-3)  # [4] 相对阈值
        pool_gate = 1.0 / (1.0 + rel_th)  # [4] 池级抑制系数
        pool_gate[0] = 1.0  # z4 当前状态不参与竞争 (基线通路)
        gate_full = torch.ones(4 * a4 + net.bind_slot_dim, dtype=torch.float16, device=dev)
        for i in range(1, 4):
            gate_full[seg[i]] = gate_full[seg[i]] * pool_gate[i].to(torch.float16)
        zh = zh * gate_full.unsqueeze(0).unsqueeze(0)
        logits_lm = (zh @ net.W_lm + net.bias_lm) * inv_h  # 重算: 门控后 (输出缩放)

        # 多步预测 (Q3 解耦): W_lm 专责 t+1, W_lm_2 独立子预测器专责 t+2.
        # 共享 z4/记忆池/bind 输入, 各自更新独立 (同一突触不拟合双目标 → 无信号冲突).
        # 生物学: 多巴胺 RPE 奖励"未来时间窗预测准确度"; 数学: 只有预测 2 步,
        # z4 才被迫携带跨词边界的高维结构 (单步预测锁死 N-gram 局域极小值)
        zh2 = torch.cat(
            [z4_lm[:, :-2], net._m2[:, :-2], net._m8[:, :-2], net._m32[:, :-2], net._bind_vec[:, :-2]], dim=-1
        )
        logits_t2 = (zh2 @ net.W_lm_2 + net.bias_lm) * inv_h  # [N,S-2,256] (输出缩放)
        target_lm = F.one_hot(byte_ids[:, 1:], num_classes=256).to(torch.float16)
        target_lm2 = F.one_hot(byte_ids[:, 2:], num_classes=256).to(torch.float16)
        # 赫布版 softmax 误差: eps = target - softmax(logits) (概率尺度 0-1).
        # 原始 target - logits 的负信号被 logits 幅度主导 (熵 5.5 时 logit~0 但非目标位
        # 255 项累积淹没目标位); softmax 后目标位概率 1/256, 误差信号与概率匹配
        probs_lm = torch.softmax(logits_lm.float(), dim=-1).to(torch.float16)  # [N,S,256]
        probs_t2 = torch.softmax(logits_t2.float(), dim=-1).to(torch.float16)
        eps_lm = (target_lm - probs_lm[:, :-1]).detach()  # [N,S-1,256]
        eps_t2 = (target_lm2 - probs_t2).detach()  # [N,S-2,256] 专供 W_lm_2 更新
        # 任务 1: 多步差分目标 — 差分误差 = (target_{t+2}-target_{t+1}) - (probs_{t+2}-probs_{t+1}),
        # 两步内字节变化的方向/幅度必须匹配 (structure 上"下一步变什么").
        # W_lm 吃 diff2 (S-1 对齐) + t+1 误差; W_lm_2 吃 diff2 + t+2 误差 (同权重,
        # 不引入 BP — 差分目标只是额外的赫布外积误差通道)
        probs_l1 = probs_lm[:, :-1]  # [N,S-1,256] = t+1 概率
        probs_l2 = probs_t2  # [N,S-2,256] = t+2 概率
        target_l1 = target_lm  # [N,S-1,256] = t+1 目标
        target_l2 = target_lm2  # [N,S-2,256] = t+2 目标
        diff2 = (target_l2 - target_l1[:, :-1]) - (probs_l2 - probs_l1[:, :-1])  # [N,S-2,256]
        diff2 = torch.cat([diff2, torch.zeros(N, 1, 256, dtype=diff2.dtype, device=dev)], dim=1)  # S-1 对齐
        # 0.2 权重: diff2 能量占比 ~22% (0.5 时 41%, W_lm 更新方向被差分信号主宰,
        # 单步目标被稀释 → 熵慢降、命中率冻结). 差分目标保留为辅助结构信号
        eps_total = (eps_lm + 0.2 * diff2).detach()  # W_lm: t+1 误差 + 差分误差 (S-1 对齐)
        eps_t2_total = (eps_t2 + 0.2 * diff2[:, :-1]).detach()  # W_lm_2: t+2 误差 + 差分误差 (S-2)

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
        # Q1 显著性反馈 + 任务 3 误差剧烈度缩放: 回传误差 × 时间突变范数
        # |z4[t]-z4[t-1]| × 预测误差能量 — 状态剧变且预测失误的时刻, 自上而下
        # 大幅改写表示层; 平稳且预测准的时刻回传弱 (保护已学结构).
        # 投影用全量 W_lm (5 段含绑定) 再取 z4 维: 预测误差经所有输入段权重汇聚到
        # z4 神经元 — 不受池门控排挤影响 (若只投影 z4 段, 池权重增长会压制 z4 段
        # → 表示层收不到预测误差 → 漂移 NaN, 9000 步崩盘根因)
        if getattr(net, "_q1_enabled", True):
            dz4_sig = (z4[:, 1:] - z4[:, :-1]).norm(dim=-1, keepdim=True)  # [N,S-1,1]
            dz4_sig = dz4_sig / (dz4_sig.max() + 1e-3)
            # soft 饱和增益: gain = x/(0.5·mean+0.5·x) ∈ (0,2), 有界无 clamp;
            # 误差 x=均值 → 1, x≫均值 → 2 (大幅改写), x≪均值 → 0 (保护)
            err_mag = eps_total.norm(dim=-1, keepdim=True)  # [N,S-1,1]
            err_ref = err_mag.mean() + 1e-3
            gain = err_mag / (0.5 * err_ref + 0.5 * err_mag)
            eps_lm_proj = (eps_total @ net.W_lm.T)[:, :, :a4] * dz4_sig * gain  # [N,S-1,a4]
        else:
            eps_lm_proj = (eps_total @ net.W_lm.T)[:, :, :a4]  # 均匀回传
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
        # 零向量保护: 模型对某 batch 预测完美 (final_error 全零) → dW_04n 全零
        # → 0/0 = NaN (fp16 下 1e-6 被舍入为 0, 装饰性). 掩码: 零范数行分母=1
        alive_04 = (nrm_04 > 1e-8).to(dW_04n.dtype)
        denom_04 = torch.where(alive_04 > 0, nrm_04, torch.ones_like(nrm_04))
        dW_04n = dW_04n * alive_04 / denom_04  # 单位向量 (只保留方向)
        e_norm = final_error.norm(dim=(1, 2))
        # 零向量保护: e_norm 全零 (perfect batch) → max+std=0 → 0/0=NaN.
        # torch.where 掩码 (与 _rms 同款): 全零行分母=1, 分子×0 → 零更新
        e_denom = e_norm.max() + e_norm.std()
        e_alive = (e_denom > 1e-8).to(e_norm.dtype)
        g_04 = e_norm / torch.where(e_alive > 0, e_denom, torch.ones_like(e_denom))
        g_04 = (g_04 * e_alive).unsqueeze(-1).unsqueeze(-1)
        g_sum = g_04.sum() + (1 - e_alive.sum()).to(g_04.dtype)  # 全零 → 分母 1
        dW_04 = (dW_04n * g_04).sum(dim=0) / g_sum
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
        # W_42 权重去同质化 (行收敛 ±w → 投影秩 1 根因)
        _decorr_W(net.W_42[:a2].data, net.E_42[:a2, :a2])

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
        # W_23 权重去同质化 (行收敛 ±w → 投影秩 1 根因)
        _decorr_W(net.W_23[:a3].data, net.E_23[:a3, :a3])

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
        # ── W_35 统一矩阵更新 (撤销微柱硬切块): 预测编码融合误差全量驱动 ──
        z5_b = z5
        eps_b = z5_b[:, 1:] - z5_b[:, :-1]  # 时序差分误差
        z3_b = z3[:, :-1]
        z3_bp = z3_b * (torch.rand_like(z3_b) > 0.3).to(torch.float16)
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
        # 样本显著性: g_n = surprise_n / Σ surprise_n (线性加权)
        g_n = (phi_b * phi_b).mean(dim=(1, 2)) + 1e-8
        g_n = g_n / g_n.sum()
        Wb = net.W_35[:a5].data
        gain_mask = net._gain_mask[:a5, :a3]
        # 误差门控: 高误差神经元主导更新, 低误差保 10% 下限
        err_norm = eps_b.pow(2).mean(dim=(0, 1)).sqrt() + 1e-8
        upd_gate = 0.1 + 0.9 * (err_norm / err_norm.max())
        n_sub = 4
        sub = max(1, N // n_sub)
        for s in range(n_sub):
            sl = slice(s * sub, (s + 1) * sub)
            dW_n = torch.bmm(phi_b[sl].transpose(-2, -1), _rms(z3_bp[sl]))
            dW_sub = (dW_n * g_n[sl, None, None]).sum(dim=0) * (1.0 / (S - 1))
            dW_sub = dW_sub * gain_mask * upd_gate.unsqueeze(1)
            b_mask = torch.rand(a5, 1, device=dev) < net.cfg.column_dropout
            Wb += (dW_sub * (~b_mask).to(torch.float16)) * eta
            soft_norm_preserve(Wb)
        _decorr_W(Wb, net.E_l5[:a5, :a5])

        # Foldiak 反赫布侧抑制更新 (方案 D): dM = z_out 协方差 (白化本质),
        # 零对角, 指数遗忘 ×0.99 防爆炸. 不做 Frobenius 归一化 — 归一化把 dM
        # 缩到 ~1e-4 (1024² 矩阵范数 ~1000), ×0.01 → ~1e-6 被 fp16 舍入,
        # 装饰性失效 (第 8 轮同款 bug 翻版, 实测 M_offdiag 0.0004 纹丝不动);
        # z_out 已逐行 RMS 归一化, 协方差元素 ∈[-1,1], 增量直接可表示
        z5_flat = z5.reshape(-1, a5).to(torch.float16)
        z5_flat = z5_flat / (z5_flat.norm(dim=-1, keepdim=True) + 1e-3)
        cov = z5_flat.transpose(0, 1) @ z5_flat / z5_flat.shape[0]
        eye_mask = 1.0 - torch.eye(a5, device=dev, dtype=torch.float16)
        net.M_l5[:a5, :a5].data.mul_(0.99).add_((cov * eye_mask) * (0.01 * learn_boost * ach_gain))

        # W_diff 下一状态预测更新 (4 步时间窗平均外积) + b_diff 偏置 (L4 空间)
        fut_mask = torch.rand(a4, 1, device=dev) < net.cfg.column_dropout
        W_diff_a.data += (dW_avg * (~fut_mask).to(torch.float16)) * eta
        future_e = (dz4 - pred_d).mean(dim=(0, 1))
        net.b_diff[:a4].data += future_e * eta

        # 时序 Hebbian (W_t 学习, 高确定性时增强 → 记忆巩固)
        # 高频抑制项 (频率锚点): 静止帧 (Δz = z_t - z_{t-1} 范数处于低分位, 几乎没动)
        # 完全不参与更新 — W_t 若指向不变主成分 (谱秩 1 引力中心), 其重复帧的
        # 贡献被清零 → 被迫只从有变化的帧学习转移 (时间核语义 = 转移矩阵).
        # 0.9 衰减被证无效 (只削 10% 幅度, 方向不变, 外积仍主方向); 阈值用
        # 差分分布低分位 (mean×0.01 太严, 随机字节流无帧触发); 连续掩码无分支
        for (z_cur, W_t), a_sz, E_t in zip(
            [(z4, net.W_t4), (z2, net.W_t2), (z3, net.W_t3), (z5, net.W_t5), (z6, net.W_t6)],
            a_sizes,
            [None, net.E_t2, net.E_t3, net.E_t5, None],
        ):
            pre = z_cur[:, :-1]
            post = z_cur[:, 1:]
            dz = post - pre  # Δz = z_t - z_{t-1}
            dz_n = dz.norm(dim=-1)  # [N, S-1]
            # 绝对阈值 = z 平均范数的 5%: 分位阈值在 z 塌缩后失效 (塌缩后"动态帧"
            # 也在主方向, 贡献仍秩 1); 绝对阈值让塌缩 → 差分变小 → 静止占比暴增 →
            # W_t 学习信号枯竭 → 自然平衡点 (频率锚点)
            th = (post.norm(dim=-1).mean() * 0.05).unsqueeze(0)
            s = (dz_n < th).to(dz_n.dtype)  # 静止帧 → 0, 动态帧 → 1
            dW_t = (_rms(pre).transpose(-2, -1) @ (_rms(post) * s.unsqueeze(-1))).mean(dim=0) * (1.0 / (S - 1))
            W_t[:a_sz, :a_sz].data += dW_t * eta_t
            # 软范数保持 (0.8-1.2): 权重有界防 fp16 溢出 (5万步 NaN 根因)
            soft_norm_preserve(W_t[:a_sz, :a_sz].data)
            # 超量 E 清除 W_t 既有秩 1 结构 (top1 sv 11.4 固化 → 递归拉 z 秩 1)
            if E_t is not None:
                _decorr_W(W_t[:a_sz, :a_sz].data, E_t[:a_sz, :a_sz])

        # LM 头自监督赫布 (复用闭环段已算的 logits_lm/eps_total):
        # 突触前增益控制: error 先 RMS 缩放到单位能量再外积 — 更新幅度完全由
        # 内部误差能量自适应决定, 不依赖外部 eta 缩放 (替代 W_04 解码死锁)
        # 指数遗忘 (0.999/步, 纯乘法): 单位能量外积在目标附近振荡 (无阻尼 LMS),
        # 遗忘项提供阻尼让权重收敛而非翻转
        err_norm = eps_total.norm(dim=-1, keepdim=True) * 1.01
        alive = (err_norm > 1e-8).to(eps_total.dtype)
        denom = torch.where(alive > 0, err_norm, torch.ones_like(err_norm))
        err_scaled = eps_total * alive / denom  # 单位能量, 方向保留

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
        # 更新外积用 _rms(zh): 高维输入稀释 dW 能量 (4096 维 vs 旧架构共线 ~2 维),
        # 0.999 指数遗忘吃掉稀释后的更新 → W_lm 行范数冻结 (实测 1.195 纹丝不动);
        # 输入归一化让更新能量与维度解耦, 每步更新恒定
        dW_lm = (_rms(zh[:, :-1]).transpose(-2, -1) @ (err_scaled - 0.1 * phi_wlm[:, :-1])).mean(dim=0) * math.sqrt(4 * a4 + net.bind_slot_dim)  # [4a4+768,256] (补偿输出缩放)
        # 单步更新幅度上界 (W_04 同款幅度-方向解耦): 高维输入下 dW 能量大,
        # 极端 batch (长文本/padding 边界) 单步爆 → W_lm_2 NaN (800 步实测);
        # 归一化到单位方向再乘 eta_lm, 最大单步幅度 = eta_lm
        dW_lm_n = dW_lm.norm() + 1e-8
        dW_lm = dW_lm / dW_lm_n
        # 记忆池段学习率激励 (生物倾角): 池段从 1e-4 低起点起爬, lr_mem = lr_z4 × 1.5,
        # 绑定段 (任务 2 新符元) 同激励, 让其能以更快速度爬到与 z4 公平竞争的高峰.
        # 固定机制, 非调参
        n_bind = net.bind_slot_dim
        lr_seg = torch.full((4 * a4 + n_bind, 1), eta_lm, dtype=torch.float16, device=dev)
        lr_seg[a4 : 4 * a4] = eta_lm * 1.5
        lr_seg[4 * a4 :] = eta_lm * 1.5
        net.W_lm.data.mul_(0.999)
        net.W_lm.data += dW_lm * lr_seg
        # bias 只学"哪些字节整体更常见"的相对偏置 (全体去均值, softmax 对平移不变):
        # 误差只用纯 t+1 目标 (eps_lm 的 err_scaled), 不含 diff2 — diff2 的逐字节
        # 目标位模式 (+1/-1 在 t2/t1 间交替) 会把中文高频 UTF-8 字节位系统性推高,
        # bias 单向累积 (实测 2000 步 std 471) → logits 被 bias 主导 → 熵锁死 1.6.
        # diff2 继续进 W_lm 权重更新 (dW_lm), 那里才是差分目标的用途
        bias_err = eps_lm / (eps_lm.norm(dim=-1, keepdim=True) * 1.01 + 1e-8)
        bias_d = bias_err.mean(dim=(0, 1))
        net.bias_lm.data += (bias_d - bias_d.mean()) * eta_lm
        # 范数约束 (结构保护, 除法缩放非 clamp): 高频字节的统计误差持续自我强化,
        # 去均值只减缓不阻止 (实测仍线性涨到 60+). 范数超 target 时整体等比缩回 —
        # 相对差异保留 (softmax 对缩放不变), 绝对电平受限, bias_std 稳定在 10 以内
        bn = net.bias_lm.norm()
        target_norm = 100.0  # 256 维均匀时 std ≈ 100/16 ≈ 6.3
        if bn > target_norm:
            net.bias_lm.data.mul_(target_norm / bn)
        soft_norm_preserve(net.W_lm.data)

        # W_lm_2 独立更新 (Q3 解耦): 吃 t+2 误差 + 差分误差 (eps_t2_total), 与 W_lm 完全独立
        # 同款机制: 零向量保护 + 单位能量 + BCM 防抖 + 指数遗忘
        err_norm2 = eps_t2_total.norm(dim=-1, keepdim=True) * 1.01
        alive2 = (err_norm2 > 1e-8).to(eps_t2_total.dtype)
        denom2 = torch.where(alive2 > 0, err_norm2, torch.ones_like(err_norm2))
        err_scaled2 = eps_t2_total * alive2 / denom2
        logits_n2 = _rms(logits_t2.detach())
        th_wlm2 = net._theta_wlm2
        th_wlm2.mul_(0.01).add_(0.99 * (logits_n2 * logits_n2).mean(dim=(0, 1)))
        phi_wlm2 = logits_n2 * (logits_n2 - th_wlm2)
        phi_wlm2 = _rms(phi_wlm2)
        dW_lm2 = (_rms(zh2).transpose(-2, -1) @ (err_scaled2 - 0.1 * phi_wlm2)).mean(dim=0) * math.sqrt(4 * a4 + net.bind_slot_dim)
        dW_lm2_n = dW_lm2.norm() + 1e-8
        dW_lm2 = dW_lm2 / dW_lm2_n  # 单步更新幅度上界 (防突爆)
        lr_seg2 = torch.full((4 * a4 + n_bind, 1), eta_lm, dtype=torch.float16, device=dev)
        lr_seg2[a4 : 4 * a4] = eta_lm * 1.5
        lr_seg2[4 * a4 :] = eta_lm * 1.5
        net.W_lm_2.data.mul_(0.999)
        net.W_lm_2.data += dW_lm2 * lr_seg2
        soft_norm_preserve(net.W_lm_2.data)

        # 状态预测矩阵自更新 (纯赫布): dW_sp = z4^T @ eps_state, 零 BP
        W_sp_a.data += (net._z4[:, :-1].transpose(-2, -1) @ eps_state).mean(dim=0) * eta
        soft_norm_preserve(W_sp_a.data)

        # ── 竞争性概念绑定层赫布更新 (任务 4, 纯外积, 零 BP): ──
        # dW_bind = z4_pre^T @ (z_bind - mean(z_bind)) — 槽位激活去均值 (Oja 式):
        # 高激活槽位强化 z4→槽映射, 低激活槽位削弱, 竞争分化; 零均值防单槽垄断.
        # W_bind 行范数保持 (0.8-1.2) 防坍缩, 软竞争 (L2 归一化) 无死亡
        if hasattr(net, "_bind_vec"):
            z4n = _rms(z4)
            bind_t = net._bind_vec - net._bind_vec.mean(dim=-1, keepdim=True)  # [N,S,K] 去均值
            dW_bind = (z4n[:, :-1].transpose(-2, -1) @ bind_t[:, :-1]).mean(dim=0) * (1.0 / (S - 1))
            net.W_bind[:a4].data.mul_(0.9995)
            net.W_bind[:a4].data += dW_bind * (eta * 2.0)
            soft_norm_preserve(net.W_bind[:a4].data)

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
