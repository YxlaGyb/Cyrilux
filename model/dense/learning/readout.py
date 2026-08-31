"""
读出端学习域
W_lm 信号构建 + LM 头更新 (W_lm/W_lm_2/W1/bias_lm).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modulation import soft_norm_preserve
from model.modulation import l2_norm, rms_norm
from ._common import LM_TRUST_REGION, _elig_accum, _MixinBase


class ReadoutMixin(_MixinBase):
    """读出端域 (方法挂载到 LearningEngine)."""

    def _build_lm_signal(self, ctx):
        """LM 头前向信号: 输入调制 → 混合层 → logits → 误差 → 投影.

        回声相位同样构建 (W_act 需要 probs_lm 决策态), 无 echo_world_frozen 守卫.
        """
        net = self.net
        if ctx.free_run:
            return None
        dev, N = ctx.dev, ctx.N
        net = net
        dim_4 = ctx.dim_4
        z4 = net._z4

        # 输入 = z4 + W_diff 预测差 (对齐训练/生成管线), 能量调制 + 三阶非线性
        z4_n_ = z4 / (z4.norm(dim=-1, keepdim=True) + 1e-3)
        pred_delta_ = z4_n_ @ net.W_diff[:dim_4, :dim_4].T + net.b_diff[:dim_4].unsqueeze(0).unsqueeze(0)
        z4r = z4 + pred_delta_
        z4_lm = z4r / (1.0 + z4r.abs())
        z4_lm = rms_norm(z4_lm)
        z4_lm = z4_lm * (1.0 - 0.5 * z4_lm.pow(2))
        # 三阶输出分流 x/(1+|x|): |x|>1.4 放大区尾部压回 ≤1, 防 W1 logits 溢出 NaN
        z4_lm = z4_lm / (1.0 + z4_lm.abs())
        # 输出缩放 1/√H; 竞争性记忆单元群 K 个泄漏积分单元 → zh 尾段 (与 z4_lm 直连段互补)
        zh = torch.cat([z4_lm, net._bind_vec, net._mem_out], dim=-1)  # [N,S,dim_4+32+K·dim_4]
        zh = rms_norm(zh)  # 与更新侧 rms_norm(zh) 对称

        # 非线性混合层: h = zh @ W1 → 三阶激活 → logits = h @ W_lm (多项式激活 FP16 原生安全)
        d_h = net.d_h
        W1_a = net.W1  # [lm_in, d_h]
        h = zh @ W1_a  # [N,S,d_h]
        h = h / (1.0 + h.abs())  # 分流抑制: W1 列极化尖峰正反馈, 掐断厚尾
        h = rms_norm(h)  # 4112 维点积值域 ±37, 归一化后进饱和区防爆炸
        net._h_in_max = h.abs().max().detach()  # 诊断: max|h_in| 距放大区余量
        h_in = h
        h = h_in * (1.0 - 0.5 * h_in.pow(2))  # 多项式激活, 零 BP
        h_deriv = 1.0 - 1.5 * h_in.pow(2)  # 激活导数 (必须用激活输入 x 算)
        inv_h = 1.0 / math.sqrt(d_h)
        logits_lm = (h @ net.W_lm + net.bias_lm) * inv_h  # [N,S,256]
        # 能量调制: 中心化 + 归一化 + max_abs 缩放严格落在 [-60,60] (fp16 极值安全)
        logits_c = (logits_lm - logits_lm.mean(dim=-1, keepdim=True)) / (
            logits_lm.std(dim=-1, keepdim=True) + 1e-4
        )
        logits_lm = logits_c / logits_c.abs().max(dim=-1, keepdim=True).values * 60.0
        # 可打印物理掩码: 0x00-0x1F 强制 -1e4 (fp16 安全极弱值)
        if ctx.closed_loop:
            target_oh = F.one_hot(ctx.byte_ids[:, 1:], num_classes=256).to(torch.float16).mean(dim=(0, 1))
        else:
            target_oh = F.one_hot(ctx.byte_ids, num_classes=256).to(torch.float16).mean(dim=(0, 1))
        if not ctx.echo_world_frozen:
            net._freq.mul_(0.99).add_(0.01 * target_oh.detach())  # 回声相位冻结频率统计
        mask_print = torch.zeros(256, dtype=torch.float16, device=dev)
        mask_print[32:] = 1.0
        logits_lm = logits_lm + (1.0 - mask_print) * -1e4

        # 多步预测: W_lm 专责 t+1, W_lm_2 独立子预测器专责 t+2 (共享 h, 各自更新)
        zh2 = torch.cat([z4_lm[:, :-2], net._bind_vec[:, :-2], net._mem_out[:, :-2]], dim=-1)
        zh2 = rms_norm(zh2)
        h2 = zh2 @ W1_a
        h2 = h2 / (1.0 + h2.abs())
        h2 = rms_norm(h2)
        h2 = h2 * (1.0 - 0.5 * h2.pow(2))
        logits_t2 = (h2 @ net.W_lm_2 + net.bias_lm) * inv_h  # [N,S-2,256]
        target_lm = F.one_hot(ctx.byte_ids[:, 1:], num_classes=256).to(torch.float16)
        target_lm2 = F.one_hot(ctx.byte_ids[:, 2:], num_classes=256).to(torch.float16)
        # 赫布版 softmax 误差: eps = target - softmax(logits); logits 已归一化 [-60,60] 无溢出
        probs_lm = torch.softmax(logits_lm, dim=-1)  # [N,S,256] fp16
        probs_t2 = torch.softmax(logits_t2, dim=-1)
        eps_lm = (target_lm - probs_lm[:, :-1]).detach()  # [N,S-1,256]
        eps_t2 = (target_lm2 - probs_t2).detach()  # [N,S-2,256]
        # 多步差分目标: diff2 = (target_{t+2}-target_{t+1}) - (probs_{t+2}-probs_{t+1}), 额外的赫布误差通道
        probs_l1 = probs_lm[:, :-1]  # [N,S-1,256] = t+1 概率
        probs_l2 = probs_t2  # [N,S-2,256] = t+2 概率
        target_l1 = target_lm  # [N,S-1,256]
        target_l2 = target_lm2  # [N,S-2,256]
        diff2 = (target_l2 - target_l1[:, :-1]) - (probs_l2 - probs_l1[:, :-1])  # [N,S-2,256]
        diff2 = torch.cat([diff2, torch.zeros(N, 1, 256, dtype=diff2.dtype, device=dev)], dim=1)  # S-1 对齐
        if ctx.closed_loop:
            lm_mask = ctx.learn_mask.unsqueeze(0).unsqueeze(-1)
            eps_lm = eps_lm * lm_mask
            eps_t2 = eps_t2 * ctx.learn_mask[1:].unsqueeze(0).unsqueeze(-1)
            diff2 = diff2 * lm_mask
        # 0.2 权重: diff2 能量占比 ~22%, 保留为辅助结构信号
        eps_total = (eps_lm + 0.2 * diff2).detach()  # W_lm: t+1 + 差分 (S-1 对齐)
        eps_t2_total = (eps_t2 + 0.2 * diff2[:, :-1]).detach()  # W_lm_2: t+2 + 差分 (S-2)

        # 语言带自校准: 感知相位记录真实语言 ε 中心 (EMA) 与弥散 (窗间 MAD EMA),
        # 供 echo 相位带状 R 校准 (action.py); echo 相位冻结不更新
        if not ctx.echo_world_frozen:
            eps_lang = 1.0 - (probs_lm[:, :-1] * target_lm).sum(dim=-1).mean()
            _lang_ema = getattr(net, "_lang_eps_ema", None)
            if _lang_ema is None:
                net._lang_eps_ema = eps_lang.detach().clone()
                net._lang_eps_mad = torch.zeros_like(eps_lang.detach())
            else:
                net._lang_eps_ema.mul_(0.995).add_(0.005 * eps_lang)
                devi = (eps_lang - net._lang_eps_ema).abs()
                net._lang_eps_mad.mul_(0.95).add_(0.05 * devi)

        # 动态稳态竞争: 熵 20 步窗口最小二乘斜率 → traction_scale ∈ (0,2) 有界无 clamp;
        # 熵降放慢表示层保护成果, 熵升放大强迫重组
        if net.cfg.adaptive_traction:
            log_p = torch.where(probs_lm > 0, torch.log(probs_lm), torch.zeros_like(probs_lm))
            ent = -(probs_lm * log_p).sum(dim=-1).mean()
            net._ent_buf[net._ent_i % 20].copy_(ent.detach())
            net._ent_i += 1
            if net._ent_i >= 20:
                idx = (net._ent_i - 19 + torch.arange(20, device=dev)) % 20
                w = net._ent_buf[idx]  # 按时间正序重排
                slope20 = (net._t_center * w).sum() / net._t_denom * 20.0  # 20 步总变化
                sigma = w.std() + 1e-4
                net._traction_scale.copy_(2.0 / (1.0 + torch.exp(-slope20 / sigma)))
        # 显著性反馈: 回传误差 × 时间突变范数, 状态剧变且预测失误时大幅改写表示层.
        # 投影用全量 W_lm 取 z4 维, 不受池门控排挤, 防表示层收不到误差而漂移 NaN.
        if getattr(net, "_q1_enabled", True):
            dz4_sig = (z4[:, 1:] - z4[:, :-1]).norm(dim=-1, keepdim=True)  # [N,S-1,1]
            dz4_sig = dz4_sig / (dz4_sig.max() + 1e-3)
            # soft 饱和增益: gain = x/(0.5·mean+0.5·x) ∈ (0,2)
            err_mag = eps_total.norm(dim=-1, keepdim=True)  # [N,S-1,1]
            err_ref = err_mag.mean() + 1e-3
            gain = err_mag / (0.5 * err_ref + 0.5 * err_mag)
            # 混合层转置投影: e → W_lm.T → W1.T → z4 段; RMS 归一化防 W_04 更新爆
            eps_lm_proj = (eps_total @ net.W_lm.T @ W1_a.T)[:, :, :dim_4] * dz4_sig * gain  # [N,S-1,dim_4]
            eps_lm_proj = rms_norm(eps_lm_proj)
        else:
            eps_lm_proj = (eps_total @ net.W_lm.T @ W1_a.T)[:, :, :dim_4]  # 均匀回传
            eps_lm_proj = rms_norm(eps_lm_proj)
        eps_lm_pad = torch.cat(
            [eps_lm_proj, torch.zeros(N, 1, dim_4, dtype=eps_lm_proj.dtype, device=dev)], dim=1
        )

        from .engine import LmSignal

        return LmSignal(
            eps_total=eps_total,
            eps_t2_total=eps_t2_total,
            eps_lm=eps_lm,
            eps_lm_proj=eps_lm_proj,
            eps_lm_pad=eps_lm_pad,
            logits_lm=logits_lm,
            logits_t2=logits_t2,
            probs_lm=probs_lm,
            h=h,
            h2=h2,
            h_deriv=h_deriv,
            zh=zh,
        )

    def _update_lm_head(self, ctx, sh):
        """LM 头自监督赫布更新: W_lm/W_lm_2/W1/bias_lm. 返回 d_t."""
        net = self.net
        if ctx.free_run or ctx.echo_world_frozen:
            return torch.ones(ctx.N, ctx.S - 1, 1, dtype=torch.float16, device=ctx.dev)
        dev = ctx.dev
        d_h = net.d_h
        inv_h = 1.0 / math.sqrt(d_h)
        byte_ids = ctx.byte_ids
        lm = sh.lm

        # 突触前增益控制: error RMS 缩放单位能量后再外积 (幅度由内部误差能量决定, 不依赖 eta);
        # 指数遗忘 (0.999/步) 提供阻尼防权重翻转
        err_norm = lm.eps_total.norm(dim=-1, keepdim=True) * 1.01
        alive = (err_norm > 1e-8).to(lm.eps_total.dtype)
        denom = torch.where(alive > 0, err_norm, torch.ones_like(err_norm))
        err_scaled = lm.eps_total * alive / denom  # 单位能量, 方向保留

        # W_lm 专属 BCM 滑阈 (防输出过冲); logits 先 RMS 归一化防平方溢出
        logits_n = rms_norm(lm.logits_lm.detach())
        th_wlm = net._theta_wlm
        th_wlm.mul_(0.01).add_(0.99 * (logits_n * logits_n).mean(dim=(0, 1)))
        phi_wlm = logits_n * (logits_n - th_wlm)
        phi_wlm = rms_norm(phi_wlm)
        # 结构对比度惩罚: 非目标列按"与目标列 logits 距离"加权, 区分差的错误列放大排斥
        target_oh = F.one_hot(byte_ids[:, 1:], num_classes=256).to(torch.float16)  # [N,S-1,256]
        logits_d = (lm.h[:, :-1] @ net.W_lm) * inv_h  # 未去偏 logits (对比度基准)
        tgt_logits = (logits_d * target_oh).sum(dim=-1, keepdim=True)  # [N,S-1,1] 目标列值
        contrast = (logits_d - tgt_logits).abs()  # 与目标列的距离
        contrast_w = 1.0 / (1.0 + contrast * 0.1)  # 距离近 → 权重大 (排斥), 距离远 → 小
        contrast_w = contrast_w * (1.0 - target_oh) + target_oh  # 目标位权重保持 1
        if net.cfg.lm_no_contrast:
            err_contrast = err_scaled * (1.0 - target_oh) + target_oh * err_scaled
        else:
            err_contrast = err_scaled * contrast_w  # 对比度加权误差 (空间排斥)
        # 快慢散度学习窗口: η = sigmoid(N - τ) ∈ (0,1) 只调变化量, 不翻转方向
        # (翻转会让 W_lm 因一次异常生成被重解释)
        if hasattr(net, "_novelty"):
            nov = net._novelty  # [N,S] 逐帧新奇度
            tau = net._theta_novelty
            tau.mul_(0.99).add_(0.01 * nov.mean())
            d_t = torch.sigmoid((nov - tau) * 500.0).unsqueeze(-1)  # [N,S,1] η ∈ (0,1)
            d_t = d_t[:, :-1]  # 对齐 S-1 (t+1 目标)
        else:
            d_t = torch.ones(ctx.N, ctx.S - 1, 1, dtype=torch.float16, device=dev)
        lm_update_mask = ctx.learn_mask.to(torch.float16).unsqueeze(0).unsqueeze(-1)  # [1,S-1,1]
        bcm_term = torch.zeros_like(phi_wlm[:, :-1]) if net.cfg.lm_no_bcm else 0.1 * phi_wlm[:, :-1]
        dW_lm = (
            rms_norm(lm.h[:, :-1]).transpose(-2, -1) @ ((err_contrast - bcm_term) * lm_update_mask * d_t)
        ).mean(dim=0) * math.sqrt(d_h)  # [d_h,256] (补偿输出缩放)
        # 单位向量 (单步更新幅度上界), 信任域防从零初始化时信号被噪声淹没
        dW_lm_n = dW_lm.norm() + 1e-8
        dW_lm = dW_lm / dW_lm_n
        dW_lm = dW_lm + _elig_accum(net, "W_lm", dW_lm) * getattr(
            net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16)
        )
        net.W_lm.data += dW_lm * (ctx.eta_lm * net.W_lm.norm() * LM_TRUST_REGION)
        # bias 硬复位: 范数锁定 target=10, 更新降 20× (去均值只学相对偏置)
        bias_err = lm.eps_lm  # [N,S-1,256] 原始 target - probs, 无归一化
        bias_d = bias_err.mean(dim=(0, 1))
        net.bias_lm.data += (bias_d - bias_d.mean()) * (ctx.eta_lm / 20.0)
        bn = net.bias_lm.norm()
        target_norm = 10.0
        if bn > target_norm:
            net.bias_lm.data.mul_(target_norm / bn)
        # W_lm 豁免 soft_norm (类间幅度差异 = 表达载体); 保留整体等比帽 10 防 fp16 溢出
        rn_lm = net.W_lm.data.norm(dim=1)
        mx_lm = rn_lm.max()
        if mx_lm > 10.0:
            net.W_lm.data.mul_((10.0 / (mx_lm + 1e-6)).to(torch.float16))

        # W1 混合层更新: 转置误差传播 (纯赫布); e_h = e @ W_lm.T · h_deriv
        e_h = (err_scaled @ net.W_lm.T) * lm.h_deriv[:, :-1]  # [N,S-1,d_h]
        lm.e_h = e_h
        dW1 = (rms_norm(lm.zh[:, :-1]).transpose(-2, -1) @ rms_norm(e_h)).mean(dim=0)
        net._dW1_absmax_raw = dW1.abs().max().detach()  # 遥测: 缩放前条目绝对最大
        # 结构化预缩放: 除条目绝对最大再除 64 → 范数 ≤4 (防 fp16 平方和溢出 → inf → 死锁/NaN)
        mx1 = dW1.abs().max() + 1e-4
        dW1 = dW1 / mx1 / 64.0
        dW1 = dW1 / (dW1.norm() + 1e-8)
        if not net.cfg.lm_freeze_w1:
            dW1 = dW1 + _elig_accum(net, "W1", dW1) * getattr(
                net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16)
            )
            net.W1.data += dW1 * (ctx.eta_lm * net.W1.norm() * LM_TRUST_REGION)
            soft_norm_preserve(net.W1.data)

        # W_lm_2 独立更新: t+2 误差 + 差分, 与 W_lm 完全独立 (同款机制)
        err_norm2 = lm.eps_t2_total.norm(dim=-1, keepdim=True) * 1.01
        alive2 = (err_norm2 > 1e-8).to(lm.eps_t2_total.dtype)
        denom2 = torch.where(alive2 > 0, err_norm2, torch.ones_like(err_norm2))
        err_scaled2 = lm.eps_t2_total * alive2 / denom2
        logits_n2 = rms_norm(lm.logits_t2.detach())
        th_wlm2 = net._theta_wlm2
        th_wlm2.mul_(0.01).add_(0.99 * (logits_n2 * logits_n2).mean(dim=(0, 1)))
        phi_wlm2 = logits_n2 * (logits_n2 - th_wlm2)
        phi_wlm2 = rms_norm(phi_wlm2)
        dW_lm2 = (
            rms_norm(lm.h2).transpose(-2, -1)
            @ (
                (err_scaled2 - 0.1 * phi_wlm2)
                * ctx.learn_mask[1:].to(torch.float16).unsqueeze(0).unsqueeze(-1)
                * d_t[:, :-1]
            )
        ).mean(dim=0) * math.sqrt(d_h)
        dW_lm2_n = dW_lm2.norm() + 1e-8
        dW_lm2 = dW_lm2 / dW_lm2_n  # 单步更新幅度上界
        dW_lm2 = dW_lm2 + _elig_accum(net, "W_lm_2", dW_lm2) * getattr(
            net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16)
        )
        net.W_lm_2.data += dW_lm2 * (ctx.eta_lm * net.W_lm_2.norm() * LM_TRUST_REGION)
        soft_norm_preserve(net.W_lm_2.data)
        # 每步清除新奇度 (前向已更新, 防陈旧信号跨步复用)
        if hasattr(net, "_novelty"):
            del net._novelty
        return d_t

    def _update_mem_units(self, ctx, sh):
        """竞争性记忆单元群: 适应度 → 增益 → 出生/死亡 (全部纯局部机制).

        出生: 读取误差快 EMA > 长程慢 EMA × 阈值 → 增益最高单元二分出子.
        死亡: g < g_min 持续超限 → 删除 (K_min=1 保底).
        时机: 在 _update_lm_head 之后, K 变化只影响下一步 (zh 列与 W1 行永不错位).
        """
        net = self.net
        if ctx.free_run or ctx.echo_world_frozen or sh.lm is None:
            return
        cfg = net.cfg
        dim_4 = ctx.dim_4
        N = ctx.N
        K = net._mem_m.shape[0]
        m_out = net._mem_out
        if m_out is None or m_out.shape[2] != K * dim_4:
            return
        lm = sh.lm
        if lm.e_h is None:
            return

        # 1) 适应度: 误差投影回 zh 空间, 取单元段与单元记忆的余弦
        e_zh = rms_norm(lm.e_h) @ net.W1.T  # [N,S-1,lm_in]
        e_cells = e_zh[:, :, dim_4 + net.bind_slot_dim :].reshape(N, lm.e_h.shape[1], K, dim_4).detach()
        mc = m_out[:, :-1].reshape(N, lm.e_h.shape[1], K, dim_4).detach()
        cos = (l2_norm(mc) * l2_norm(e_cells)).sum(dim=-1).mean(dim=(0, 1))  # [K] ∈ [-1,1]
        net._mem_q.mul_(0.99).add_(0.01 * cos)

        # 2) 增益 (有界参数 clamp, 非梯度路径): q>0 上推, 乘性衰减让零贡献单元 g→0
        net._mem_g.mul_(1.0 - cfg.mem_g_decay).clamp_(min=0.0)
        net._mem_g.add_(cfg.mem_eta_g * net._mem_q)
        net._mem_g.clamp_(max=cfg.mem_g_max)

        # 3) 死亡: g 低 → 计数, 持续超限 → 删除; 索引 0 永不判死
        low = net._mem_g < cfg.mem_g_min
        net._mem_death_cnt = torch.where(low, net._mem_death_cnt + 1, torch.zeros_like(net._mem_death_cnt))
        dead_mask = (net._mem_death_cnt >= cfg.mem_death_steps) & low
        dead_mask[0] = False
        if dead_mask[1:].any():
            net._mem_death_cnt = net._mem_death_cnt.where(~dead_mask, torch.zeros_like(net._mem_death_cnt))
            self._mem_resize(~dead_mask)

        # 4) 出生: 误差快 EMA 超阈值, 冷却期满, K<上限 → 增益最高单元二分
        err_rms = lm.eps_total.square().mean().sqrt()
        net._mem_err_ema.mul_(0.99).add_(0.01 * err_rms)
        net._mem_err_long.mul_(0.999).add_(0.001 * err_rms)
        net._mem_birth_cd += 1
        if (
            net._mem_birth_cd >= cfg.mem_birth_cooldown
            and cfg.mem_k_max > K
            and net._mem_err_ema > net._mem_err_long * cfg.mem_birth_thresh
        ):
            parent = int(torch.argmax(net._mem_g))  # 每步至多一次出生, 同步可接受
            self._mem_birth(parent)
            net._mem_birth_cd = 0

    def _mem_resize(self, keep: torch.Tensor):
        """K 变化: 单元缓冲 + W1 行 + W1_elig 迹同步重注册."""
        net = self.net
        dim_4 = net._mem_m.shape[1]
        d_h = net.d_h
        if keep.dtype == torch.bool:
            keep = keep.nonzero(as_tuple=False).squeeze(1)
        old_m = net._mem_m.data[keep].contiguous()
        old_a = net._mem_a.data[keep].contiguous()
        old_g = net._mem_g.data[keep].contiguous()
        old_q = net._mem_q.data[keep].contiguous()
        net.register_buffer("_mem_m", old_m)
        net.register_buffer("_mem_a", old_a)
        net.register_buffer("_mem_g", old_g)
        net.register_buffer("_mem_q", old_q)
        net.register_buffer("_mem_death_cnt", net._mem_death_cnt[keep].contiguous())
        # W1 行同步 (单元块位于 [dim_4+32 : dim_4+32+K·dim_4])
        head = dim_4 + net.bind_slot_dim
        w1 = net.W1.data
        new_w1 = torch.cat(
            [w1[:head], w1[head:].reshape(-1, dim_4, d_h)[keep].reshape(keep.shape[0] * dim_4, d_h)],
            dim=0,
        ).contiguous()
        net.W1 = nn.Parameter(new_w1)
        # W1_elig 迹同形同步 (出生/死亡后迹必须与 W1 行对齐)
        elig = getattr(net, "W1_elig", None)
        if elig is not None:
            ne = torch.cat(
                [elig.data[:head], elig.data[head:].reshape(-1, dim_4, d_h)[keep].reshape(keep.shape[0] * dim_4, d_h)],
                dim=0,
            ).contiguous()
            net.register_buffer("W1_elig", ne)
        net._lm_in = head + keep.shape[0] * dim_4

    def _mem_birth(self, parent: int):
        """从父单元二分出生 (α×2 或 ÷2, _mem_alt 交替, 越界取另一侧)."""
        net = self.net
        cfg = net.cfg
        K = net._mem_m.shape[0]
        dim_4 = net._mem_m.shape[1]
        d_h = net.d_h
        a_par = float(net._mem_a[parent])
        alt = net._mem_alt % 2
        net._mem_alt += 1
        a_child = a_par * 2.0 if alt == 0 else a_par / 2.0
        if a_child > cfg.mem_alpha_max:
            a_child = a_par / 2.0
        elif a_child < cfg.mem_alpha_min:
            a_child = a_par * 2.0
        a_child = max(cfg.mem_alpha_min, min(cfg.mem_alpha_max, a_child))
        # 子单元: m 继承父, g 小值起步 (需证明自己), q 零, α 二分
        m_child = net._mem_m.data[parent : parent + 1].clone()
        new_m = torch.cat([net._mem_m.data, m_child], dim=0).contiguous()
        new_a = torch.cat([net._mem_a.data, torch.tensor([a_child], dtype=torch.float16, device=net._mem_m.device)], dim=0).contiguous()
        new_g = torch.cat([net._mem_g.data, torch.full((1,), cfg.mem_g_min * 2.0, dtype=torch.float16, device=net._mem_m.device)], dim=0).contiguous()
        new_q = torch.cat([net._mem_q.data, torch.zeros(1, dtype=torch.float16, device=net._mem_m.device)], dim=0).contiguous()
        net.register_buffer("_mem_m", new_m)
        net.register_buffer("_mem_a", new_a)
        net.register_buffer("_mem_g", new_g)
        net.register_buffer("_mem_q", new_q)
        net.register_buffer(
            "_mem_death_cnt",
            torch.cat(
                [net._mem_death_cnt, torch.zeros(1, dtype=torch.int32, device=net._mem_g.device)]
            ).contiguous(),
        )
        # W1 单元块尾插零行 (零起步: 新单元先"无贡献", 由 W1 学权重自然放大)
        head = dim_4 + net.bind_slot_dim
        w1 = net.W1.data
        new_w1 = torch.cat(
            [w1[:head], w1[head:], torch.zeros(dim_4, d_h, dtype=torch.float16, device=w1.device)],
            dim=0,
        ).contiguous()
        net.W1 = nn.Parameter(new_w1)
        elig = getattr(net, "W1_elig", None)
        if elig is not None:
            ne = torch.cat(
                [elig.data[:head], elig.data[head:], torch.zeros(dim_4, d_h, dtype=torch.float16, device=elig.device)],
                dim=0,
            ).contiguous()
            net.register_buffer("W1_elig", ne)
        net._lm_in = head + (K + 1) * dim_4
