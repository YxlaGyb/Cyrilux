"""
读出端学习域
W_lm 信号构建 + LM 头更新 (W_lm/W_lm_2/W1/bias_lm).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from crates.kernels import cubic, rms_norm, saturate, soft_norm_preserve
from crates.modal.text.geometry import ALPHABET, CONTINUATION_PREFIX
from crates.model.constants import F_UNIT_SCALE, LM_TRUST_REGION
from crates.rtr import rtr_apply, trace

from ._common import _MixinBase


class ReadoutMixin(_MixinBase):
    """读出端域 (方法挂载到 LearningEngine)."""

    def _w1v2_forward(self, zh: torch.Tensor) -> torch.Tensor:
        """
        w1v2 前向 (w1v2_prereg.md §1.8):
            h = RMSNorm(concat(zh@R, GELU(zh@A_i)...)) [N,S,256].
        """
        net = self.net
        parts = [zh @ net.w1v2_R]
        for i in range(net._w1v2_n_heads):
            parts.append(F.gelu(zh @ getattr(net, f"w1v2_A_{i}")))
        return rms_norm(torch.cat(parts, dim=-1))

    def _build_lm_signal(self, ctx):
        """
        LM 头前向信号:
            输入调制 → 混合层 → logits → 误差 → 投影

        回声相位同样构建，W_act 需要 probs_lm 决策态)，无 echo_world_frozen 守卫
        """
        net = self.net
        if ctx.free_run:
            return None
        net = net
        dim_4 = ctx.dim_4
        z4 = net._z4

        # 输入 = z4 + W_diff 预测差，对齐训练/生成管线, 能量调制 + 三阶非线性
        z4_n_ = z4 / (z4.norm(dim=-1, keepdim=True) + 1e-3)
        pred_delta_ = z4_n_ @ net.W_diff[:dim_4, :dim_4].T + net.b_diff[:dim_4].unsqueeze(0).unsqueeze(0)
        z4r = z4 + pred_delta_
        z4_lm = saturate(z4r)
        z4_lm = rms_norm(z4_lm)
        z4_lm = cubic(z4_lm)

        # 三阶输出分流 x/(1+|x|): |x|>1.4 放大区尾部压回 ≤1, 防 W1 logits 溢出 NaN
        z4_lm = saturate(z4_lm)

        # 输出缩放 1/√H; 竞争性记忆单元群 K 个泄漏积分单元 → zh 尾段 (与 z4_lm 直连段互补)
        zh = torch.cat([z4_lm, net._bind_vec, net._mem_out], dim=-1)  # [N,S,dim_4+32+K·dim_4]
        zh = rms_norm(zh)  # 与更新侧 rms_norm(zh) 对称

        # 非线性混合层: h = zh @ W1 → 三阶激活 → logits = h @ W_lm (多项式激活 FP16 原生安全)
        d_h = net.d_h
        W1_a = net.W1  # [lm_in, d_h] w1v2 时仅用于 eps_lm_proj 表示层反馈投影, 不更新

        if getattr(net, "_w1v2", False):
            h = self._w1v2_forward(zh)  # [N,S,256] concat(残差, 4 头)
            h_deriv = None  # w1v2 不用 h_deriv (W1 不更新, mem units 被 freeze-below 跳过)
        else:
            h = zh @ W1_a  # [N,S,d_h]
            h = saturate(h)  # 分流抑制: W1 列极化尖峰正反馈, 掐断厚尾
            h = rms_norm(h)  # 4112 维点积值域 ±37, 归一化后进饱和区防爆炸
            net._h_in_max = h.abs().max().detach()  # 诊断: max|h_in| 距放大区余量
            h_in = h
            h = cubic(h_in)  # 多项式激活, 零 BP
            h_deriv = 1.0 - 1.5 * h_in.pow(2)  # 激活导数 (必须用激活输入 x 算)

        # 迁移扩 h (migrate_expand_prereg.md §1): 新通道 h_new = zh @ W1_new (三阶激活), logits 加和
        h_new = None
        h_new_deriv = None
        if getattr(net, "_migrate_expand", False):
            h_new = zh @ net.W1_new  # [N,S,256]
            h_new = saturate(h_new)
            h_new = rms_norm(h_new)
            h_new_in = h_new
            h_new = cubic(h_new_in)
            h_new_deriv = 1.0 - 1.5 * h_new_in.pow(2)
        inv_h = 1.0 / math.sqrt(d_h)
        logits_lm = (h @ net.W_lm + net.bias_lm) * inv_h  # [N,S,ALPHABET]
        if h_new is not None:
            logits_lm = logits_lm + h_new @ net.W_lm_new  # 新通道贡献

        # 能量调制: 中心化 + 归一化 + max_abs 缩放严格落在 [-60,60] (fp16 极值安全)
        logits_c = (logits_lm - logits_lm.mean(dim=-1, keepdim=True)) / (
            logits_lm.std(dim=-1, keepdim=True) + 1e-4
        )
        logits_lm = logits_c / logits_c.abs().max(dim=-1, keepdim=True).values * 60.0

        # 可打印物理掩码: 低于可打印起点的符号强制 -1e4 (fp16 安全极弱值)
        logits_lm = logits_lm + (1.0 - net._mask_print) * -1e4  # tensor-guard: bound (打印屏蔽设计界)

        # 多步预测: W_lm 专责 t+1, W_lm_2 独立子预测器专责 t+2 (共享 h, 各自更新)
        zh2 = torch.cat([z4_lm[:, :-2], net._bind_vec[:, :-2], net._mem_out[:, :-2]], dim=-1)
        zh2 = rms_norm(zh2)
        if getattr(net, "_w1v2", False):
            h2 = self._w1v2_forward(zh2)
        else:
            h2 = zh2 @ W1_a
            h2 = saturate(h2)
            h2 = rms_norm(h2)
            h2 = cubic(h2)
        logits_t2 = (h2 @ net.W_lm_2 + net.bias_lm) * inv_h  # [N,S-2,ALPHABET]
        logits_t2 = logits_t2 + (1.0 - net._mask_print) * -1e4  # 与 logits_lm 同款打印掩码 (C3: probs 双域统一)
        target_lm = F.one_hot(ctx.byte_ids[:, 1:], num_classes=ALPHABET).to(torch.float16)
        target_lm2 = F.one_hot(ctx.byte_ids[:, 2:], num_classes=ALPHABET).to(torch.float16)

        # 局部 softmax 误差: eps = target - softmax(logits); logits 已归一化 [-60,60] 无溢出
        probs_lm = torch.softmax(logits_lm, dim=-1)  # [N,S,ALPHABET] fp16
        probs_t2 = torch.softmax(logits_t2, dim=-1)
        eps_lm = (target_lm - probs_lm[:, :-1]).detach()  # [N,S-1,ALPHABET]
        eps_t2 = (target_lm2 - probs_t2).detach()  # [N,S-2,ALPHABET]

        # 多步差分目标: diff2 = (target_{t+2}-target_{t+1}) - (probs_{t+2}-probs_{t+1}), 额外的局部误差通道
        probs_l1 = probs_lm[:, :-1]  # [N,S-1,ALPHABET] = t+1 概率
        probs_l2 = probs_t2  # [N,S-2,ALPHABET] = t+2 概率
        target_l1 = target_lm  # [N,S-1,ALPHABET]
        target_l2 = target_lm2  # [N,S-2,ALPHABET]
        diff2 = (target_l2 - target_l1[:, :-1]) - (probs_l2 - probs_l1[:, :-1])  # [N,S-2,ALPHABET]

        # 行掩码: t+1 与 t+2 目标均 ≥32 的行才保留 (C3: 幻影目标行整行清零)
        diff2 = diff2 * (ctx.learn_mask[:, :-1] & ctx.learn_mask[:, 1:]).to(torch.float16).unsqueeze(-1)
        diff2 = torch.cat([diff2, net._padmax[:, :, :ALPHABET]], dim=1)  # S-1 对齐

        # 学习位置掩码 (engine 端从目标字节推导): t+1 位 [N,S-1,1], t+2 位 [N,S-2,1]
        lm_mask = ctx.learn_mask.to(torch.float16).unsqueeze(-1)
        eps_lm = eps_lm * lm_mask

        # scaffold 实验通道 (wordstart_add_prereg.md; 默认 0 零变更): eps 行加权, 全程 fp16
        # A: 目标字节为 UTF-8 首字节的行 ×boundary_w; 词首加法: 词首行额外 +word_start_add (不减词内)
        _bw = float(getattr(net, "_boundary_w", 0.0) or 0.0)
        _wsa = float(getattr(net, "_word_start_add", 0.0) or 0.0)
        if _bw > 0.0 or _wsa > 0.0:
            _lead = (ctx.byte_ids[:, 1:] & CONTINUATION_PREFIX[0]) != CONTINUATION_PREFIX[1]  # [N,S-1] target = UTF-8 首字节
            if _bw > 0.0:
                _rw = 1.0 + (_bw - 1.0) * _lead.to(torch.float16)
            else:
                _rw = torch.ones_like(_lead, dtype=torch.float16)
            _wsm = getattr(net, "_word_start_mask", None)
            if _wsa > 0.0 and _wsm is not None:
                _rw = _rw + _wsa * _wsm.to(torch.float16)
            eps_lm = eps_lm * _rw.unsqueeze(-1)

        # 字符级 eps 聚合 (pre_2a_step2_prereg.md; 默认关零变更): 同字符内各字节共享该字符的平均误差,
        # 学习信号单元 = 字符而非字节. eps 形状 [N,S-1,ALPHABET], 按位置维段聚合; 全程 fp16
        if getattr(net, "_char_agg", False):
            _tgt = ctx.byte_ids[:, 1:]  # [N,S-1] 目标字节
            _lead_t = ((_tgt & CONTINUATION_PREFIX[0]) != CONTINUATION_PREFIX[1]).to(torch.float16)  # 字符首字节
            _seg = (torch.cumsum(_lead_t, dim=1) - _lead_t).long()  # [N,S-1] 每位置所属字符 id (0 起)
            _idx3 = _seg.unsqueeze(-1).expand_as(eps_lm)  # [N,S-1,ALPHABET]
            _sum = torch.zeros_like(eps_lm)
            _cnt = torch.zeros_like(_seg, dtype=torch.float16)
            _sum.scatter_add_(1, _idx3, eps_lm)
            _cnt.scatter_add_(1, _seg, lm_mask.squeeze(-1))
            _den = torch.where(_cnt > 0, _cnt, torch.ones_like(_cnt))  # 不用 clamp
            _mean = _sum / _den.unsqueeze(-1)
            eps_lm = _mean.gather(1, _idx3) * lm_mask
        eps_t2 = eps_t2 * ctx.learn_mask[:, 1:].to(torch.float16).unsqueeze(-1)

        # 0.2 权重: diff2 能量占比 ~22%, 保留为辅助结构信号
        eps_total = (eps_lm + 0.2 * diff2).detach()  # W_lm: t+1 + 差分 (S-1 对齐)
        eps_t2_total = (eps_t2 + 0.2 * diff2[:, :-1]).detach()  # W_lm_2: t+2 + 差分 (S-2)

        # ── self-state-prediction 信号 (self_state_layered_prereg.md §2.1; 默认关零变更) ──
        # 状态头: pred_z4 = h_t @ W_z4_head (零初始化, 惰性建在 engine 上 — 非 nn 参数/buffer,
        # 不进 _extra_tensors 扫描 → 检查点逐位零变更; 本臂单跑 500 步无续训, 不持久化).
        # eps_z4 = z4_{t+1}(实际) − pred_t; pred 位置 t 只用 h_{≤t}, 无未来泄漏
        # (teacher-forced 批内等价于"延迟一步": z4[:,1:] 是递归前向实际到达的下一状态).
        if getattr(net, "_self_state_pred", False):
            if getattr(self, "_ssp_w", None) is None:
                self._ssp_w = torch.zeros(d_h, dim_4, dtype=torch.float16, device=z4.device)
            _pred_z4 = h @ self._ssp_w  # [N,S,dim_4]
            _eps_z4 = (z4[:, 1:] - _pred_z4[:, :-1]) * lm_mask  # [N,S-1,dim_4]
            self._ssp_eps = _eps_z4.detach()
            self._ssp_mse = _eps_z4.square().mean().detach()  # 遥测: 报告步才 .item(), 主循环零同步

        # ── 接线通道信号 (w1_state_pred_wire_prereg.md §2; 默认关零变更) ──
        # eps_state 按 feedforward.py:135 同式纯前向重算 (z4 @ W_state_pred − Δz4), 无新状态、无新矩阵;
        # 单位能量 (同 feedforward.py:136) + 学习掩码; 挂 engine (非 net) 不进检查点, 单步消费.
        if getattr(net, "_wire_state_pred", False):
            _dz_full = z4[:, 1:] - z4[:, :-1]  # [N,S-1,dim_4]
            _eps_state = (z4[:, :-1] @ net.W_state_pred[:dim_4, :dim_4].T) - _dz_full
            _eps_state = rms_norm(_eps_state) * lm_mask
            self._eps_state = _eps_state.detach()

        # 语言带自校准锚: 感知相位记录 ε 中心 EMA (恒温器输入, action.py 读 _lang_eps_ema);
        # echo 相位冻结不更新
        if not ctx.echo_world_frozen:
            eps_lang = 1.0 - (probs_lm[:, :-1] * target_lm).sum(dim=-1).mean()
            if net._lang_eps_cold:
                net._lang_eps_ema.copy_(eps_lang.detach())  # 首次快照
                net._lang_eps_cold = False
            else:
                net._lang_eps_ema.mul_(0.995).add_(0.005 * eps_lang)

        # 动态稳态竞争: 熵 20 步窗口最小二乘斜率 → traction_scale ∈ (0,2) 有界无 clamp;
        # 熵降放慢表示层保护成果, 熵升放大强迫重组
        if net.cfg.adaptive_traction:
            log_p = torch.where(probs_lm > 0, torch.log(probs_lm), torch.zeros_like(probs_lm))
            ent = -(probs_lm * log_p).sum(dim=-1).mean()
            net._ent_buf[net._ent_i % 20].copy_(ent.detach())
            net._ent_i += 1
            if net._ent_i >= 20:
                idx = (net._ent_i - 19 + net._arange20[None, :]) % 20  # 按时间正序重排 (预分配 arange)
                w = net._ent_buf[idx]  # 按时间正序重排
                slope20 = (net._t_center * w).sum() / net._t_denom * 20.0  # 20 步总变化
                sigma = w.std() + 1e-4
                net._traction_scale.copy_(2.0 / (1.0 + torch.exp(-slope20 / sigma)))

        # 显著性反馈: 回传误差 × 时间突变范数, 状态剧变且预测失误时大幅改写表示层.
        # 投影用全量 W_lm 取 z4 维, 不受池门控排挤, 防表示层收不到误差而漂移 NaN.
        dz4_sig = (z4[:, 1:] - z4[:, :-1]).norm(dim=-1, keepdim=True)  # [N,S-1,1]
        dz4_sig = dz4_sig / (dz4_sig.max() + 1e-3)

        # soft 饱和增益: gain = x/(0.5·mean+0.5·x) ∈ (0,2)
        err_mag = eps_total.norm(dim=-1, keepdim=True)  # [N,S-1,1]
        err_ref = err_mag.mean() + 1e-3
        gain = err_mag / (0.5 * err_ref + 0.5 * err_mag)

        # 混合层转置投影: e → W_lm.T → W1.T → z4 段; RMS 归一化防 W_04 更新爆
        # 结构化预缩放 (A1): 矩阵乘在 d_h 维累加可冲 fp16 上限, 溢出发生在 rms_norm
        # 之前 (后置归一化救不了); ÷64 = 指数移位, rms_norm 尺度不变 → 下游语义零变
        eps_in = eps_total / 64.0
        eps_proj_raw = (eps_in @ net.W_lm.T @ W1_a.T)[:, :, :dim_4] * dz4_sig * gain  # [N,S-1,dim_4]

        # 计价项取归一化前的投影能量. rms_norm 逐活行恒定输出 1/1.01 均方 (幅值被抹平),
        # 拿它的输出计价会让 F 逐位退化成行存活计数 fp16(0.990099·alive/256): 死区 2.8e4 倍,
        # 任一位点的信任域步都比翻一格台阶小 333 倍 → 无位点可被计价. ×F_UNIT_SCALE 只是
        # 2 的幂换单位 (fp16 指数移位, 精确), 让 metab_c/metab_d/κ 的既有标定继续成立.
        eps_proj_ms = eps_proj_raw.square().mean() * F_UNIT_SCALE
        eps_lm_proj = rms_norm(eps_proj_raw)

        from .engine import LmSignal

        return LmSignal(
            eps_total=eps_total,
            eps_t2_total=eps_t2_total,
            eps_lm=eps_lm,
            eps_lm_proj=eps_lm_proj,
            eps_proj_raw=eps_proj_raw,
            eps_proj_ms=eps_proj_ms,
            logits_lm=logits_lm,
            probs_lm=probs_lm,
            h=h,
            h2=h2,
            h_deriv=h_deriv,
            zh=zh,
            h_new=h_new,
            h_new_deriv=h_new_deriv,
        )

    def _update_w1v2_heads(self, ctx, lm):
        """
        w1v2 每头局部重建 (w1v2_prereg.md §1.3-1.5): 预测编码式局部规则, 非共现非 BP.

        e_i = zh - GELU(zh@A_i)@B_i; 梯度只在头内:
          dA_i ∝ zhᵀ @ ((e_i @ B_iᵀ) ⊙ GELU'(zh@A_i));  dB_i ∝ GELU(zh@A_i)ᵀ @ e_i.
        头间隔离 (不跨头), 不接 next-byte 梯度, A_i/B_i 联合优化.
        """
        net = self.net
        zh_prev = lm.zh[:, :-1]  # [N,S-1,lm_in] 对齐 t+1 目标
        mask = ctx.learn_mask.to(torch.float16).unsqueeze(-1)  # [N,S-1,1]
        eta = net._w1v2_eta_recon * LM_TRUST_REGION
        recon = []
        for i in range(net._w1v2_n_heads):
            A = getattr(net, f"w1v2_A_{i}")
            B = getattr(net, f"w1v2_B_{i}")
            z = zh_prev @ A  # [N,S-1,d_head]
            h_i = F.gelu(z)  # [N,S-1,d_head]
            r = (zh_prev - h_i @ B) * mask  # [N,S-1,lm_in] 重建误差 (学习位置掩码)
            e = r
            if net._w1v2_wzh is not None:  # w1v2b: 加权重建 (组合方向额外惩罚 λ)
                r_w = (r @ net._w1v2_wzh).unsqueeze(-1)  # [N,S-1,1] 组合方向投影误差
                e = r + net._w1v2_wzh_lam * r_w * net._w1v2_wzh.unsqueeze(0).unsqueeze(0)
            recon.append(r.square().mean().detach())  # 遥测报纯重建 MSE (可与第一步对比)

            # GELU 导数 (sigmoid 近似: gelu(x)≈x·σ(1.702x))
            sig = torch.sigmoid(1.702 * z)
            g = sig + 1.702 * z * sig * (1.0 - sig)  # [N,S-1,d_head]
            dA = (zh_prev.transpose(-2, -1) @ ((e @ B.transpose(-2, -1)) * g)).mean(dim=0)  # [lm_in,d_head]
            dB = (h_i.transpose(-2, -1) @ e).mean(dim=0)  # [d_head,lm_in]

            # 单位范数 + 相对幅度 (同 W1 更新风格), 零更新掩码防 fp16 0/0=NaN
            for d, p in ((dA, A), (dB, B)):
                mx = d.abs().max() + 1e-4
                d = d / mx / 64.0
                n = d.norm()
                alive = (n > 0).to(d.dtype)
                d = d * alive / torch.where(alive > 0, n, torch.ones_like(n))
                p.data += d * (eta * p.norm())
                pn = p.data.norm()  # 等比帽防 fp16 溢出 (B_i 初始范数 ~43, 上限 50)
                p.data.mul_(torch.minimum(torch.ones_like(pn), 50.0 / (pn + 1e-6)))
        net._w1v2_recon_err = sum(recon) / len(recon)  # 遥测: 平均重建 MSE (step%50 报告)

    @staticmethod
    def _mig_unit(d: torch.Tensor) -> torch.Tensor:
        """结构化预缩放 + 单位范数 (同 W1 更新, 零更新掩码防 fp16 0/0)."""
        mx = d.abs().max() + 1e-4
        d = d / mx / 64.0
        n = d.norm()
        alive = (n > 0).to(d.dtype)
        return d * alive / torch.where(alive > 0, n, torch.ones_like(n))

    def _update_migrate_heads(self, ctx, lm, err_scaled, d_t):
        """迁移扩 h (migrate_expand_prereg.md §1): W1_new 双信号 (next-byte 主 + 加权重建辅),
        W_lm_new next-byte, B_new 重建. W1/W_lm/W_lm_2 已冻结 (旧通道保底)."""
        net = self.net
        zh_prev = lm.zh[:, :-1]  # [N,S-1,lm_in]
        mask = ctx.learn_mask.to(torch.float16).unsqueeze(-1)  # [N,S-1,1]
        d_h = net.d_h
        eta = ctx.eta_lm * LM_TRUST_REGION

        # 1) W_lm_new: next-byte 更新 (同 W_lm 机制, 但用 h_new)
        dW_lm_new = (rms_norm(lm.h_new[:, :-1]).transpose(-2, -1) @ (err_scaled * mask * d_t)).mean(dim=0) * math.sqrt(d_h)
        dW_lm_new = self._mig_unit(dW_lm_new)
        net.W_lm_new.data += dW_lm_new * (ctx.eta_lm * net.W_lm_new.norm() * LM_TRUST_REGION)
        rn = net.W_lm_new.data.norm(dim=1)
        mx = rn.max()
        net.W_lm_new.data.mul_(torch.minimum(torch.ones_like(mx), 10.0 / (mx + 1e-6)).to(torch.float16))

        # 2) W1_new 主信号: next-byte 转置误差传播 (同 W1)
        e_h_new = (err_scaled @ net.W_lm_new.T) * lm.h_new_deriv[:, :-1]  # [N,S-1,256]
        dW1_next = (rms_norm(zh_prev).transpose(-2, -1) @ rms_norm(e_h_new)).mean(dim=0)  # [lm_in,256]

        # 3) W1_new + B_new 辅信号: 加权重建 w_zh (λ=0.5, 三阶激活同前向)
        zh_hat = lm.h_new[:, :-1] @ net.B_new  # [N,S-1,lm_in]
        r = (zh_prev - zh_hat) * mask
        if net._w1v2_wzh is not None:
            r_w = (r @ net._w1v2_wzh).unsqueeze(-1)  # [N,S-1,1]
            e = r + net._w1v2_wzh_lam * r_w * net._w1v2_wzh.unsqueeze(0).unsqueeze(0)
        else:
            e = r
        dW1_recon = (zh_prev.transpose(-2, -1) @ ((e @ net.B_new.transpose(-2, -1)) * lm.h_new_deriv[:, :-1])).mean(dim=0)
        dB_new = (lm.h_new[:, :-1].transpose(-2, -1) @ e).mean(dim=0)  # [256,lm_in]
        net._w1v2_recon_err = r.square().mean().detach()  # 遥测: 纯重建 MSE

        # W1_new 双信号叠加 (next-byte 主 + 重建辅, 各单位化后等权)
        dW1 = self._mig_unit(dW1_next) + self._mig_unit(dW1_recon)
        net.W1_new.data += self._mig_unit(dW1) * (eta * net.W1_new.norm())
        net.B_new.data += self._mig_unit(dB_new) * (eta * net.B_new.norm())
        for p in (net.W1_new, net.B_new):
            pn = p.data.norm()
            p.data.mul_(torch.minimum(torch.ones_like(pn), 50.0 / (pn + 1e-6)))

    def _single_recon_w1(self, ctx, lm):
        """
        单通道加权重建辅信号 (single_recon_prereg.md §1.5): dW1_recon = zh.T @ (e_eff@B.T ⊙ h_deriv).
        返回单位范数 × η_recon_ratio 的 dW1_recon; 同时更新 B_recon (相对幅度 + 等比帽).
        """
        net = self.net
        zh_prev = lm.zh[:, :-1]  # [N,S-1,lm_in]
        mask = ctx.learn_mask.to(torch.float16).unsqueeze(-1)
        h_act = lm.h[:, :-1]  # [N,S-1,256] 三阶激活后 (前向已算, 已含非线性)
        zh_hat = h_act @ net.B_recon  # [N,S-1,lm_in]
        r = (zh_prev - zh_hat) * mask
        if net._w1v2_wzh is not None:
            r_w = (r @ net._w1v2_wzh).unsqueeze(-1)
            e = r + net._w1v2_wzh_lam * r_w * net._w1v2_wzh.unsqueeze(0).unsqueeze(0)
        else:
            e = r
        dW1_recon = (zh_prev.transpose(-2, -1) @ ((e @ net.B_recon.transpose(-2, -1)) * lm.h_deriv[:, :-1])).mean(dim=0)
        dB_recon = (h_act.transpose(-2, -1) @ e).mean(dim=0)  # [256,lm_in]
        net._w1v2_recon_err = r.square().mean().detach()  # 遥测: 纯重建 MSE
        eta = ctx.eta_lm * LM_TRUST_REGION
        net.B_recon.data += self._mig_unit(dB_recon) * (eta * net.B_recon.norm())
        pn = net.B_recon.data.norm()
        net.B_recon.data.mul_(torch.minimum(torch.ones_like(pn), 50.0 / (pn + 1e-6)))
        return self._mig_unit(dW1_recon) * net._w1v2_eta_recon  # 辅信号比主小 (η_recon_ratio)

    def _update_lm_head(self, ctx, lm):
        """LM 头自监督局部更新: W_lm/W_lm_2/W1/bias_lm. 返回 d_t."""
        net = self.net
        if ctx.free_run or ctx.echo_world_frozen:
            return net._one_s[: ctx.N, : ctx.S - 1]  # tensor-guard: bound (冻结步恒一 d_t, 预分配)
        d_h = net.d_h
        inv_h = 1.0 / math.sqrt(d_h)
        byte_ids = ctx.byte_ids
        _mig = getattr(net, "_migrate_expand", False)  # migrate_expand_prereg.md: W1/W_lm 冻结, 新通道可训练
        _realign = getattr(net, "_readout_realign", False)  # E8a: 覆写 _mig 冻结门 (默认关, 逐位不变)
        _rdfrz = getattr(net, "_readout_freeze", False)  # E8: 读出侧全冻结 (默认关, 逐位不变)

        # E9f 拆门: 旧读出 (W1/W_lm/bias_lm/W_lm_2) 与 migrate 新通道 (W1_new/W_lm_new/B_new) 各用一个门;
        # 未显式接线 _migrate_freeze 时跟随 _rdfrz, 保证 E8/E9/默认路径逐位不变.
        _migfrz = getattr(net, "_migrate_freeze", _rdfrz)

        # 字节头降权 (self_state_layered_prereg.md §2.2; 默认 1.0 逐位零变更): eps 单位能量
        # 归一 + dW 单位范数化对信号侧全局缩放不变 → 降权落更新幅度 (×_bhw, 仅 ssp 开启时生效)
        _bhw = float(getattr(net, "_byte_head_w", 1.0)) if getattr(net, "_self_state_pred", False) else 1.0

        # 突触前增益控制: error RMS 缩放单位能量后再外积 (幅度由内部误差能量决定, 不依赖 eta);
        # 指数遗忘 (0.999/步) 提供阻尼防权重翻转
        err_norm = lm.eps_total.norm(dim=-1, keepdim=True) * 1.01
        alive = (err_norm > 1e-8).to(lm.eps_total.dtype)
        denom = torch.where(alive > 0, err_norm, torch.ones_like(err_norm))
        err_scaled = lm.eps_total * alive / denom  # 单位能量, 方向保留

        # 结构对比度惩罚: 非目标列按"与目标列 logits 距离"加权, 区分差的错误列放大排斥
        target_oh = F.one_hot(byte_ids[:, 1:], num_classes=ALPHABET).to(torch.float16)  # [N,S-1,ALPHABET]
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
            d_t = net._one_s[:ctx.N, : ctx.S - 1]  # 无新奇度轨迹: 恒一 d_t (预分配)
        lm_update_mask = ctx.learn_mask.to(torch.float16).unsqueeze(-1)  # [N,S-1,1]
        dW_lm = (
            rms_norm(lm.h[:, :-1]).transpose(-2, -1) @ (err_contrast * lm_update_mask * d_t)
        ).mean(dim=0) * math.sqrt(d_h)  # [d_h,ALPHABET] (补偿输出缩放)

        # 单位向量 (单步更新幅度上界), 信任域防从零初始化时信号被噪声淹没
        # 零更新保护: 1e-8 在 fp16 下舍为 0, 零窗 dW≡0 → 0/0=NaN (步 3012 实证); 掩码式归零
        # 交底: 旧代码 +1e-8 在 fp16 下等于 +0——ε 从未被观测到; 本修复不是换更好的 ε
        # (finfo.eps 同量级, 在 +ε 加法中同样消失), 是承认 ε 不存在——归零用乘掩码, 不用加底.
        n_lm = dW_lm.norm()
        alive_n = (n_lm > 0).to(dW_lm.dtype)
        dW_lm = dW_lm * alive_n / torch.where(alive_n > 0, n_lm, torch.ones_like(n_lm))
        dW_lm = rtr_apply(trace(net, "W_lm", dW_lm), net._survival_signal)
        if (not _mig or _realign) and not _rdfrz:  # 迁移扩 h: W_lm 冻结; _readout_realign (E8a) 覆写; _readout_freeze (E8) 全冻结
            net.W_lm.data += dW_lm * (ctx.eta_lm * net.W_lm.norm() * LM_TRUST_REGION * _bhw)
        if not _rdfrz:
            # bias 硬复位: 范数锁定 target=10, 更新降 20× (去均值只学相对偏置)
            bias_err = lm.eps_lm  # [N,S-1,ALPHABET] 原始 target - probs, 无归一化
            bias_d = bias_err.mean(dim=(0, 1))
            net.bias_lm.data += (bias_d - bias_d.mean()) * (ctx.eta_lm / 20.0 * _bhw)
            bn = net.bias_lm.norm()
            target_norm = 10.0
            net.bias_lm.data.mul_(
                torch.minimum(torch.ones_like(bn), target_norm / (bn + 1e-6))  # 等比帽 (branchless min, 非 clamp)
            )
            # W_lm 豁免 soft_norm (类间幅度差异 = 表达载体); 保留整体等比帽 10 防 fp16 溢出
            rn_lm = net.W_lm.data.norm(dim=1)
            mx_lm = rn_lm.max()
            net.W_lm.data.mul_(
                torch.minimum(torch.ones_like(mx_lm), 10.0 / (mx_lm + 1e-6)).to(torch.float16)  # 等比帽 (branchless min)
            )

        # W1 混合层更新: 转置误差传播; e_h = e @ W_lm.T · h_deriv
        # w1v2 分支 (w1v2_prereg.md §1.3-1.5): A_i/B_i 纯重建驱动, 头间隔离, 不接 next-byte 梯度
        if getattr(net, "_w1v2", False):
            self._update_w1v2_heads(ctx, lm)
            lm.e_h = None
        else:
            e_h = (err_scaled @ net.W_lm.T) * lm.h_deriv[:, :-1]  # [N,S-1,d_h]
            lm.e_h = e_h
            dW1 = (rms_norm(lm.zh[:, :-1]).transpose(-2, -1) @ rms_norm(e_h)).mean(dim=0)
            net._dW1_absmax_raw = dW1.abs().max().detach()  # 遥测: 缩放前条目绝对最大

            # 结构化预缩放: 除条目绝对最大再除 64 → 范数 ≤4 (防 fp16 平方和溢出 → inf → 死锁/NaN)
            # 末级归一化同款零更新保护: 1e-8 fp16 舍为 0, 零窗 0/0=NaN → 掩码式归零
            mx1 = dW1.abs().max() + 1e-4
            dW1 = dW1 / mx1 / 64.0
            n_w1 = dW1.norm()
            alive_n1 = (n_w1 > 0).to(dW1.dtype)
            dW1 = dW1 * alive_n1 / torch.where(alive_n1 > 0, n_w1, torch.ones_like(n_w1))
            if not net.cfg.lm_freeze_w1 and (not _mig or _realign) and not _rdfrz:
                dW1 = rtr_apply(trace(net, "W1", dW1), net._survival_signal)
                if getattr(net, "_single_recon", False):  # 单通道迁移: 加权重建辅信号
                    dW1 = dW1 + self._single_recon_w1(ctx, lm)
                net.W1.data += dW1 * (ctx.eta_lm * net.W1.norm() * LM_TRUST_REGION * _bhw)
                soft_norm_preserve(net.W1.data)

        # W_lm_2 独立更新: t+2 误差 + 差分, 与 W_lm 完全独立 (同款机制)
        err_norm2 = lm.eps_t2_total.norm(dim=-1, keepdim=True) * 1.01
        alive2 = (err_norm2 > 1e-8).to(lm.eps_t2_total.dtype)
        denom2 = torch.where(alive2 > 0, err_norm2, torch.ones_like(err_norm2))
        err_scaled2 = lm.eps_t2_total * alive2 / denom2
        dW_lm2 = (
            rms_norm(lm.h2).transpose(-2, -1)
            @ (
                err_scaled2
                * ctx.learn_mask[:, 1:].to(torch.float16).unsqueeze(-1)
                * d_t[:, :-1]
            )
        ).mean(dim=0) * math.sqrt(d_h)
        dW_lm2_n = dW_lm2.norm()
        alive_n2 = (dW_lm2_n > 0).to(dW_lm2.dtype)
        dW_lm2 = dW_lm2 * alive_n2 / torch.where(alive_n2 > 0, dW_lm2_n, torch.ones_like(dW_lm2_n))  # 单步更新幅度上界 (零更新掩码: 1e-8 fp16 舍为 0)
        dW_lm2 = rtr_apply(trace(net, "W_lm_2", dW_lm2), net._survival_signal)
        if not _mig and not _rdfrz:  # 迁移扩 h: W_lm_2 冻结; _readout_freeze (E8) 全冻结
            net.W_lm_2.data += dW_lm2 * (ctx.eta_lm * net.W_lm_2.norm() * LM_TRUST_REGION * _bhw)
            soft_norm_preserve(net.W_lm_2.data)

        # ── 迁移扩 h (migrate_expand_prereg.md §1): W1_new 双信号 + W_lm_new + B_new ──
        if _mig and not _migfrz:  # E9f 拆门: migrate 新通道独立冻结门 (旧读出由 _rdfrz 守; 默认跟随 _rdfrz)
            self._update_migrate_heads(ctx, lm, err_scaled, d_t)

        # ── 状态头预测通道 (self_state_layered_prereg.md §2.1; 默认关零变更) ──
        # 位置: free_run/echo_world_frozen 守卫之后 → 仅感知相位更新 (C5 冻结合约同域).
        # 两条更新: 头本体 (单位范数方向, 步长 η_lm·TR — 无自身范数缩放, 零初始化 ‖W‖=0)
        # + W1 转置传播 (与字节 e_h→dW1 同构; 不走 elig 迹, 避免与字节通道双消费 W1 迹).
        _ez = getattr(self, "_ssp_eps", None)  # [N,S-1,dim_4] 已带 lm_mask (单步消费, 末尾置 None)
        if getattr(net, "_self_state_pred", False) and _ez is not None:
            _ez_norm = _ez.norm(dim=-1, keepdim=True) * 1.01
            _ez_alive = (_ez_norm > 1e-8).to(_ez.dtype)
            _ez_den = torch.where(_ez_alive > 0, _ez_norm, torch.ones_like(_ez_norm))
            _ez_unit = _ez * _ez_alive / _ez_den  # 逐行单位能量 (掩码式, 防 fp16 0/0)

            # 1) 头本体
            _dW_z4 = (
                rms_norm(lm.h[:, :-1]).transpose(-2, -1) @ (_ez_unit * lm_update_mask * d_t)
            ).mean(dim=0)  # [d_h, dim_4]
            _n_z4 = _dW_z4.norm()
            _alive_z4 = (_n_z4 > 0).to(_dW_z4.dtype)
            _dW_z4 = _dW_z4 * _alive_z4 / torch.where(_alive_z4 > 0, _n_z4, torch.ones_like(_n_z4))
            self._ssp_w += _dW_z4 * (ctx.eta_lm * LM_TRUST_REGION)

            # 2) W1 转置传播: e = eps @ W_z4_headᵀ ⊙ h_deriv; 头为零 → e≡0 → 掩码式零更新
            _e_z4h = (_ez_unit @ self._ssp_w.T) * lm.h_deriv[:, :-1]  # [N,S-1,d_h]
            _dW1_z4 = (rms_norm(lm.zh[:, :-1]).transpose(-2, -1) @ rms_norm(_e_z4h)).mean(dim=0)
            _mx_z4 = _dW1_z4.abs().max() + 1e-4
            _dW1_z4 = _dW1_z4 / _mx_z4 / 64.0
            _n1_z4 = _dW1_z4.norm()
            _alive1_z4 = (_n1_z4 > 0).to(_dW1_z4.dtype)
            _dW1_z4 = _dW1_z4 * _alive1_z4 / torch.where(_alive1_z4 > 0, _n1_z4, torch.ones_like(_n1_z4))
            net.W1.data += _dW1_z4 * (ctx.eta_lm * net.W1.norm() * LM_TRUST_REGION)
            soft_norm_preserve(net.W1.data)
            self._ssp_eps = None  # 单步消费, 防陈旧信号跨步复用

        # ── 接线通道更新 (w1_state_pred_wire_prereg.md §2; 默认关零变更) ──
        # 前向投影经 W1 的 z4 切片: e_h_state = eps_state @ W1[:a4,:] ⊙ h_deriv;
        # dW1_state 更新 W1[:a4,:] (组合特征所在输入段). 自指: W1[:a4,:] 前向被用又更新 (同字节通道
        # e_h = err @ W_lm.T 的转置用法); 不走 elig 迹 (避免与字节通道双消费 W1 迹); 仅感知相位.
        _es = getattr(self, "_eps_state", None)  # [N,S-1,a4] 已带 lm_mask (单步消费)
        if getattr(net, "_wire_state_pred", False) and _es is not None:
            _a4 = ctx.dim_4
            _e_h_state = (_es @ net.W1[:_a4, :]) * lm.h_deriv[:, :-1]  # [N,S-1,d_h]
            _dW1_state = (rms_norm(lm.zh[:, :-1, :_a4]).transpose(-2, -1) @ rms_norm(_e_h_state)).mean(dim=0)  # [a4,d_h]
            _mx_s = _dW1_state.abs().max() + 1e-4
            _dW1_state = _dW1_state / _mx_s / 64.0
            _n_s = _dW1_state.norm()
            _alive_s = (_n_s > 0).to(_dW1_state.dtype)
            _dW1_state = _dW1_state * _alive_s / torch.where(_alive_s > 0, _n_s, torch.ones_like(_n_s))
            net.W1.data[:_a4, :] += _dW1_state * (ctx.eta_lm * net.W1[:_a4, :].norm() * LM_TRUST_REGION)
            soft_norm_preserve(net.W1.data)
            self._eps_state = None  # 单步消费

        # 每步清除新奇度 (前向已更新, 防陈旧信号跨步复用)
        if hasattr(net, "_novelty"):
            del net._novelty
        return d_t
