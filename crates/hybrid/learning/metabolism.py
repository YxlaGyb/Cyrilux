"""
体内代谢域
F (自由能) 每步计价 → E 能量账本 (成本入账) → R 生存信号 (W_act 资格迹货币).
P3-b 节律内生: 何时看/何时说由行为门 (体内状态) 涌现 — 行为账本差 / 代谢应激 /
内源振荡器 A / 新奇度 组合; ΔF 按行为分账, 死亡时钟分相.
"""

from __future__ import annotations

import math

import torch

from ._common import _MixinBase

# E9 多尺度状态聚合预测误差 (e9_preregistration.md §三): F 加一项 λ·Σ_k w_k·mean_t(eps_k²).
# K={8,32} 为主判集; k=16 仅降级阶梯 (rung≥1) 启用. w_k=1/√k, λ_main=0.1 (预写, 不事后挑选).
_MS_W = {8: 1.0 / math.sqrt(8.0), 16: 0.25, 32: 1.0 / math.sqrt(32.0)}
_MS_LAMBDA = 0.1
_MS_MAXK = 32
_MS_EMA_KEEP = 0.9  # _eps_k_ema 诊断 EMA 保留系数
_MS_THETA_ABS_DEFAULT = 1e4  # §4.5 θ_abs 占位; §5.4 标定后以 10×A0 经 --multi-scale-theta-abs 覆写


class MetabolismMixin(_MixinBase):
    """代谢域 (方法挂载到 LearningEngine)."""

    def _update_metabolism(self, ctx, sh):
        """体内代谢: F 每步计价 (感知/回声同构) → E 账本 (收入−成本) → R 生存信号.

        双账本 (P3-c 判定): E 携带成本 (资本/应激/门控货币), R 吃纯 ΔF —
        符号铁律 (条件二, 写死): **F 下降 → ΔF 为正 → R > 0**.
        R 的输入永不得换成 F 水平、误差绝对值或任何"压平即胜"的可收割量
        (dark room 禁区: 奖励误差水平会让压平分布成为过关最便宜路径).
        成本只进 E 不进 R — 否则平台期 ΔE 常态负 → R 常态负 → 条件二点状失效.
        """
        net = self.net
        if ctx.free_run or sh.lm is None:
            # free_run 无 lm 信号 (_build_lm_signal 返回 None), 代谢不结算
            return
        # 1) F := 绝对误差能量: 归一化误差平方是尺度无关的 — 饥荒
        # (预测相对变差) 会被误差自身标准差吸收, E 对饥荒零判别力 (术前探针实证:
        # 随机字节/分布漂移/W_t 渐进病理下 E min 都停在 -0.008 平台噪声带内).
        # 绝对货币: 乱码输入 → 原始误差能量暴涨 → E 深潜 → 死亡线可达.
        # sh.metab_f 仍供学习链用归一化 final_error (feedforward 契约不变, 显式优于隐式).
        eps_lm_pad = sh.lm.eps_lm_pad
        eps4 = sh.errs.eps4
        # 零向量保护 (rms_norm 同族掩码式): 全零窗误差 std=0 → 0/0=NaN (挂针实证, 步 13
        # 全零窗 W_04 起爆). 分母归零 = 该误差通道本步零信息 → 信号归零, 不写 NaN.
        # 阈值取 finfo.tiny (fp16=6.1e-5): 除数低于最小正规数在 fp16 下才会失稳;
        # eps (2^-11) 过宽会误伤真实低误差窗, tiny 是"除法良定义"的精确边界.
        s_pred = eps_lm_pad.std(dim=-1, keepdim=True) * 1.01 + eps_lm_pad.std() * 1e-3
        alive_p = (s_pred > torch.finfo(s_pred.dtype).tiny).to(s_pred.dtype)
        err_pred_norm = eps_lm_pad * alive_p / torch.where(
            alive_p > 0, s_pred, torch.ones_like(s_pred)
        )
        s_rec = eps4.std(dim=-1, keepdim=True) * 1.01 + eps4.std() * 1e-3
        alive_r = (s_rec > torch.finfo(s_rec.dtype).tiny).to(s_rec.dtype)
        err_recon_norm = eps4 * alive_r / torch.where(
            alive_r > 0, s_rec, torch.ones_like(s_rec)
        )
        final_error = err_pred_norm + 0.2 * err_recon_norm
        f_now = eps_lm_pad.square().mean() + 0.2 * eps4.square().mean()  # 0-dim fp16 标量
        # E9b-0 剂量/路径审计插桩 (e9b0_audit_preregistration.md): 默认关 = 零变更. 全部为
        # 诊断 buffer 写入, 不改权重、不改计算图、不改 RNG; 基线向量与三分量落 buffer 供报告步取样.
        audit = getattr(net, "_f_ms_audit", False)
        if audit:
            net._audit_eps_lm_sq.copy_(eps_lm_pad.square().mean())
            net._audit_eps4_sq.copy_(eps4.square().mean())
            net._audit_vec_base_rms.copy_(final_error.square().mean().sqrt())
            sh.metab_f_base = final_error
            net._audit_priced.zero_()
        # E9 双出处同改 (§3.3): 标量 f_now 与向量 final_error 同步加多尺度项. flag 关 = 短路 (§4.1
        # 选项 A), _multiscale_pred_error 不被调用, 5 个新增持久 buffer 全程保持 __init__ 初值.
        # 相位门 (§3.5): 仅感知相 (sh.diff is not None) 计价; 回声相冻结 W_diff, 不计价.
        if getattr(net, "_f_multiscale", False) and sh.diff is not None:
            ms_term, e_ms = self._multiscale_pred_error(ctx)
            f_now = f_now + ms_term
            s_ms = e_ms.std(dim=-1, keepdim=True) * 1.01 + e_ms.std() * 1e-3
            alive_ms = (s_ms > torch.finfo(s_ms.dtype).tiny).to(s_ms.dtype)
            e_ms_norm = e_ms * alive_ms / torch.where(
                alive_ms > 0, s_ms, torch.ones_like(s_ms)
            )
            final_error = final_error + _MS_LAMBDA * e_ms_norm
            if audit:
                net._audit_ms_scalar.copy_(ms_term)
                net._audit_ems_rms.copy_(e_ms.square().mean().sqrt())
                net._audit_ems_norm_rms.copy_(e_ms_norm.square().mean().sqrt())
                net._audit_vec_ms_rms.copy_((_MS_LAMBDA * e_ms_norm).square().mean().sqrt())
                net._audit_priced.fill_(1)
                sh.metab_f_ms = _MS_LAMBDA * e_ms_norm
        if audit:
            net._audit_f_total.copy_(f_now)
            if ctx.echo_loop:
                net._audit_f_echo_ema.mul_(0.99).add_(0.01 * f_now)
                net._audit_echo_steps.add_(1)
            else:
                net._audit_f_perc_ema.mul_(0.99).add_(0.01 * f_now)
                net._audit_perc_steps.add_(1)
            net._audit_priced_steps.add_(net._audit_priced)
        sh.metab_f = final_error  # 供 _update_feed_ff 取用 (归一化, F 单一出处)
        # 2) P3-c 全行为计价 (判词 §2.4): 看/学/记/说全耗能. 成本全部来自既有信号
        # (f_now / W1 行数 / 步型标志), 零 .item() 零热路径新建; W1 行数是主机侧 int
        # 直乘 (禁止 torch.tensor — 免掉各 resize 点的缓冲同步陷阱).
        #   看: κ_p·f_now (输入处理 F 水平, 感知步)  学: κ_l·sqrt(f_now) (ΔW 幅度代理, 每步)
        #   记: κ_m·W1 行数 (生存质量费, 每步 — 死亡回合直接降此费: 死亡→降开销闭合)
        #   说: κ_s (每次发声 ATP 费, 回声步)
        net._metab_cost_learn.copy_(f_now).sqrt_().mul_(net.cfg.metab_cost_learn)
        net._metab_cost_mem.fill_(net.cfg.metab_cost_mem * net.W1.shape[0])
        net._metab_cost_say.fill_(net.cfg.metab_cost_say if ctx.echo_loop else 0.0)
        if ctx.echo_loop:
            net._metab_cost_perc.zero_()
        else:
            net._metab_cost_perc.copy_(f_now).mul_(net.cfg.metab_cost_perc)
        net._metab_cost_tot.copy_(net._metab_cost_perc).add_(
            net._metab_cost_learn
        ).add_(net._metab_cost_mem).add_(net._metab_cost_say)
        # 3) P3-b 行为内 ΔF: 感知/回声各一轨 F_prev — 跨行为 F 不同窗/不同流 (回声=自生成流),
        # 行为内差分才是苹果对苹果; 行为首跑只登记 (df=0: 无进步无退步, 成本照付)。
        # 登记在全局哨兵之前: 首步的 F 必须落轨 (防跨步空窗 + 下步 df 失真).
        if ctx.echo_loop:
            if not net._settled_eco:
                net._metab_F_prev_eco.copy_(f_now)
                net._settled_eco = True
                df = net._zero1
            else:
                df = net._metab_F_prev_eco - f_now  # 进步为正
                net._metab_F_prev_eco.copy_(f_now)
        else:
            if not net._settled_perc:
                net._metab_F_prev_perc.copy_(f_now)
                net._settled_perc = True
                df = net._zero1
            else:
                df = net._metab_F_prev_perc - f_now  # 进步为正
                net._metab_F_prev_perc.copy_(f_now)
        if not net._metab_settled:
            # 全局首步哨兵 (E 起账): 只登记不结算; python 标志代替 `if _metab_F_prev < 0:`
            # (张量布尔 = 每步一次隐式 .item() 同步排空 — 利用率主嫌之一)
            net._metab_R = torch.zeros(1, dtype=torch.float16, device=ctx.dev)  # tensor-guard: rare (哨兵一次性)
            net._metab_settled = True  # 下步起结算 (哨兵只一次)
            return
        # 4) 有符号收入 − 成本: E ← E·(1−d) + c·ΔF − Σcost; 退步=负收入, E 可短负=濒死记账
        net._metab_E.mul_(1.0 - net.cfg.metab_d).add_(net.cfg.metab_c * df).sub_(
            net._metab_cost_tot
        )
        # 死亡判据计数: 平台期 F 规律性下探 (低能耗时刻),
        # 饥荒 = F 钉高位零下探 (v3 实测: 平台 q25-q95 = 0.975-0.986 ≈ 天花板, 但每 ~14 步
        # 一下探; famine_rand 400 步零下探). dip = f_now < base·(1−metab_dip_margin); 每次
        # dip 清零, 无 dip 连续 metab_death_steps 步 → 死亡回合. (v1/v2/v3 三次证伪:
        # E 域/绝对域/高域余量均无判别力 — 平台均值已钉天花板, κ 阈值落在信号域之外.)
        # 分相: 只有感知相能下探清零 (F_base 慢基线也只在感知相更新 — 回声相 F 是
        # 自生成流, 尺度/语义不同轨); 回声相无条件 +1 — 纯回声锁 250 步即死
        # ("长时间不看世界 = 病态", 死亡语义在此保留).
        if ctx.echo_loop:
            dip = torch.zeros_like(net._metab_F_base, dtype=torch.bool)  # 回声相无下探判据 (恒 +1)
        else:
            cold = net._metab_F_base.le(0.0)
            base_upd = net._metab_F_base.mul_(1.0 - net.cfg.metab_base_rate).add_(
                net.cfg.metab_base_rate * f_now
            )
            net._metab_F_base.copy_(torch.where(cold, f_now, base_upd))
            dip = torch.where(
                cold,
                torch.zeros_like(net._metab_F_base, dtype=torch.bool),
                f_now < net._metab_F_base * (1.0 - net.cfg.metab_dip_margin),
            )
        net._metab_starve_cnt.copy_(
            torch.where(dip, torch.zeros_like(net._metab_starve_cnt), net._metab_starve_cnt + 1)
        )
        # 对称牙: 回声清零 / 感知 +1 (与 starve 镜像)
        if ctx.echo_loop:
            net._metab_silence_cnt.zero_()
        else:
            net._metab_silence_cnt.add_(1)
        # 应激 = 饥荒契约进度 (starve_cnt/metab_death_steps): 尺度即机体自身死线, 零设计者增益
        net._metab_E_ref.mul_(1.0 - net.cfg.metab_eref_rate).add_(
            net.cfg.metab_eref_rate * net._metab_E
        )
        net._metab_famine_prog.copy_(net._metab_starve_cnt).mul_(
            1.0 / float(net.cfg.metab_death_steps)
        )
        net._metab_stress = net._metab_famine_prog
        # 6) P3-b 新奇度账本 (门控输入): 仅感知步刷新 — _novelty 由 _update_lm_head 末删除,
        # 而代谢在其前运行 (定点域序), 此处是 learn() 链内唯一安全读点. rel_n 自归一化:
        # 首感知步 ema=1e-3 → nov≈1 (世界全新 → 多看), 熟世界衰减 (看旧世界 → 想说的选项).
        if not ctx.echo_loop and hasattr(net, "_novelty"):
            raw = net._novelty.mean()  # 0-dim fp16 逐帧新奇度均值
            net._metab_nov.copy_(raw / (net._metab_nov_ema + raw + 1e-6))
            net._metab_nov_ema.mul_(1.0 - net.cfg.metab_nov_rate).add_(
                net.cfg.metab_nov_rate * raw
            )
        # 5) R 原语 (R1 契约同构, P3-c 双账本): R 只吃 ΔF 不吃成本 — 条件二逐点成立.
        #    tanh(ΔF/(div·MAD_ΔF)) ∈ (−1,1); div=metab_tanh_div 标定值 (R1 契约 [0.3,1] 重标:
        #    0.95/2.0 下 median(leg)=0.26 偏饿 → 0.90/1.5 → 0.38). 旧货币 ΔE ≈ c·ΔF (E≈0)
        #    使 MAD 比例同构 — div 预计免重标, 以标定探针为准.
        #    除 ‖迹‖ 在消费端 (_update_w_act 于迹更新后执行, 被乘的迹就是除的那个迹 —
        #    R1 精确形态, 且首步迹非零, 不会出现 /0 爆炸). 全 GPU fp16 零 .item().
        # 每行为各一条 MAD (两行为 ΔF 尺度相差 ~15×; 共用会把 R_echo 压成 0)
        if ctx.echo_loop:
            mad, cold_flag = net._metab_df_mad_eco, "_metab_mad_cold_eco"
        else:
            mad, cold_flag = net._metab_df_mad_perc, "_metab_mad_cold_perc"
        mad.mul_(net.cfg.metab_mad_alpha).add_(
            (1.0 - net.cfg.metab_mad_alpha) * df.abs()
        )
        if getattr(net, cold_flag):
            # 冷启动: 该行为首个样本 = |ΔF|, 下限防 0/0 (绝对货币下
            # 首结算步 ΔF 可精确为 0 → 0/0=nan); python 标志代替张量布尔 (免每步排空)
            mad.copy_(torch.maximum(df.abs(), torch.tensor(1e-5, dtype=torch.float16)))  # tensor-guard: rare (冷启动一次性)
            setattr(net, cold_flag, False)
        net._metab_R = torch.tanh(df / (net.cfg.metab_tanh_div * mad))
        # 7) P3-b 行为账本: 仅运行行为记账 (|R| EMA — 该行为最近收益体验); 非运行行为
        # 不更新不衰减 (留档记忆 — 衰减方向留作标定网格备选, 默认不实现)
        if ctx.echo_loop:
            net._metab_gain_eco.mul_(1.0 - net.cfg.metab_gain_rate).add_(
                net.cfg.metab_gain_rate * net._metab_R.abs()
            )
        else:
            net._metab_gain_perc.mul_(1.0 - net.cfg.metab_gain_rate).add_(
                net.cfg.metab_gain_rate * net._metab_R.abs()
            )

    def _multiscale_pred_error(self, ctx):
        """E9 多尺度状态聚合预测误差 (§三/§4.3): 对 horizon k 用冻结 W_diff roll out k 步状态,
        与 teacher-forced 未来态均值比对; 返回 (标量项 λ·Σ_k w_k·mean_t eps_k², 向量项 Σ_k w_k·e_k).

        仅当 net._f_multiscale 为真时被调用 (flag 关 = 短路, §4.1 选项 A); 相位门由调用侧
        (sh.diff is not None) 保证. 未来态 z4(t+j) 直接取 net._z4 (§4.2: 因果递归已 teacher-forced,
        无需二次前向; 调用点在 learn() 释放 net._z4 之前). 一次 32 步 roll out 复用聚合缓冲取部分和:
        k=8 部分和在 j=8 就地取用, k=16 部分和快照入 _z4_ek16_buf, k=32 部分和在末态取用 (§3.2 逐 k
        部分和 = (1/k)Σ_{j≤k}). 全 GPU fp16 零同步 (.item() 一律禁止); 降级阶梯 (§4.5) 以 GPU 掩码
        实现, 绝不 clamp 数值本身.
        """
        net = self.net
        dev, N, S, dim_4 = ctx.dev, ctx.N, ctx.S, ctx.dim_4
        z4 = net._z4  # [N,S,dim_4] teacher-forced 未来态
        W_diff_a = net.W_diff[:dim_4, :dim_4]
        b_diff_a = net.b_diff[:dim_4]
        s = net._z4_roll_buf[:N, :S, :dim_4]        # roll out 状态 s_j (就地覆写)
        run = net._z4_agg_pred_buf[:N, :S, :dim_4]  # Σ_{j≤k} s_j
        fut = net._z4_agg_fut_buf[:N, :S, :dim_4]   # Σ_{j≤k} z4(t+j)
        ek16 = net._z4_ek16_buf[:N, :S, :dim_4]     # k=16 部分和误差快照 (rung==1 专用)
        ar = net._seq_arange[:S]
        theta = float(getattr(net, "_f_ms_theta_abs", _MS_THETA_ABS_DEFAULT))
        zero_s = net._zero1.new_zeros(())
        e_ms = torch.zeros(N, S, dim_4, dtype=torch.float16, device=dev)
        ms = zero_s.clone()
        m8 = net._ms_mask_k8[:S]
        m32 = net._ms_mask_k32[:S]
        m8.copy_(ar < (S - 8))
        m32.copy_(ar < (S - 32))
        # 粘性 rung 的保留因子 (rollout 前取证, 无同步): keep_k = (最终 rung == 该阶条件).
        # k=8 在 j=8 就地定稿, 此时最终 rung 未定; 但 final= max(prev, next), 而 rung≤2 ⟺ prev≤2 ∧ ¬c8,
        # 故 k=8 保留因子 = rung_le2 · ¬c8 (两项 j=8 前/时已知), 与 rung3 粘性置零一致.
        rung_le2 = (net._f_ms_rung <= 2).to(torch.float16)
        # 主 roll out j=1..32: 就地累加 run/fut, 顺带在 j=8/16/32 取证异常 (非有限 或 absmax>θ_abs)
        s.copy_(z4)
        run.zero_()
        fut.zero_()
        nf = torch.zeros((), dtype=torch.bool, device=dev)  # 0-dim, 无同步
        amax = net._f_ms_absmax
        amax.zero_()
        c8 = nf.clone()
        c16 = nf.clone()
        for j in range(1, _MS_MAXK + 1):
            u = s / (s.norm(dim=-1, keepdim=True) + 1e-3)  # CosineNorm (与生成/读出同构)
            s.add_(u @ W_diff_a.T + b_diff_a)              # s_j = s_{j-1} + Δ
            run.add_(s)
            if j < S:
                fut[:, : S - j].add_(z4[:, j:])            # 有效位 t ≤ S−j−1
            nf = nf | (~torch.isfinite(s).all())
            amax.copy_(torch.maximum(amax, s.abs().max()))
            if j == 8:
                c8 = nf | (amax > theta)
                # k=8 部分和就地取用 (e_ms 此步尚空, 作暂存): e_ms ← ek8 = fut − run
                wk8 = (rung_le2 * (~c8).to(torch.float16)) * _MS_W[8]
                e_ms.copy_(fut).sub_(run)
                e_ms.mul_(m8.unsqueeze(0).unsqueeze(-1).to(e_ms.dtype))
                # eps_k² = mean_t(mean_d(e_k²)) (§3.2): mean 按 float 累积不溢出, 等价于
                # sum_valid/(k²·N·(S-k)·dim_4) — 分母的 N·dim_4 由 mean 吸收, 余 S/(S-k) 回补有效位.
                term = e_ms.square().mean() * (S / float(S - 8)) / 64.0
                term = torch.where(torch.isfinite(term), term, zero_s)
                net._eps_k_sq[0].copy_(term)
                net._eps_k_ema[0].mul_(_MS_EMA_KEEP).add_((1.0 - _MS_EMA_KEEP) * term)
                ms = ms + (wk8 * term).reshape(())
                e_ms.mul_(wk8 / 8)  # 就地转成 k=8 向量贡献
            elif j == 16:
                c16 = nf | (amax > theta)
                ek16.copy_(fut).sub_(run)                  # k=16 部分和误差快照
        c32 = nf | (amax > theta)
        # 降级阶梯: 粘性单调 (rung = max(prev, next)); 条件为当步异常 (零同步)
        rung_next = torch.where(
            c8, torch.full_like(net._f_ms_rung, 3),
            torch.where(
                c16, torch.full_like(net._f_ms_rung, 2),
                torch.where(c32, torch.full_like(net._f_ms_rung, 1),
                            torch.zeros_like(net._f_ms_rung)),
            ),
        )
        net._f_ms_rung.copy_(torch.maximum(net._f_ms_rung, rung_next))
        net._f_ms_nonfinite.add_(nf.to(torch.int32))
        rung = net._f_ms_rung
        keep16 = (rung == 1).to(torch.float16)
        keep32 = (rung == 0).to(torch.float16)
        # k=32 项 (末态部分和; roll out 已毕, s 复用为暂存): ek32 = fut − run
        s.copy_(fut).sub_(run)
        s.mul_(m32.unsqueeze(0).unsqueeze(-1).to(s.dtype))
        term = s.square().mean() * (S / float(S - 32)) / 1024.0
        term = torch.where(torch.isfinite(term), term, zero_s)
        net._eps_k_sq[1].copy_(term)
        net._eps_k_ema[1].mul_(_MS_EMA_KEEP).add_((1.0 - _MS_EMA_KEEP) * term)
        ms = ms + (keep32 * _MS_W[32] * term).reshape(())
        s.mul_(keep32 * _MS_W[32] / 32)
        e_ms.add_(s)
        # k=16 项 (仅降级阶梯 rung==1 启用; keep16=0 时整项归零)
        ek16.mul_((ar < (S - 16)).unsqueeze(0).unsqueeze(-1).to(ek16.dtype))
        term = ek16.square().mean() * (S / float(S - 16)) / 256.0
        term = torch.where(torch.isfinite(term), term, zero_s)
        ms = ms + (keep16 * _MS_W[16] * term).reshape(())
        ek16.mul_(keep16 * _MS_W[16] / 16)
        e_ms.add_(ek16)
        # 降级 (权重=0) 遇 inf 会产生 0×inf=NaN: 终置有限化, 结构性兜底 (非 clamp 数值)
        ms = torch.where(torch.isfinite(ms), ms, zero_s)
        e_ms = torch.where(torch.isfinite(e_ms), e_ms, torch.zeros_like(e_ms))
        return _MS_LAMBDA * ms, e_ms  # 标量含 λ (§3.3); 向量未乘 λ 未归一化, 由调用侧处理

    def _update_gate(self, ctx, sh):
        """P3-b 行为门 (节律内生): 本步末算, 供下一步路由 — 全 GPU 零同步.

        门输入全部来自体内 (见 gate_psay 纯函数, 两侧公式一一对应): 行为账本差
        (g_eco−g_perc = 说 vs 看的近期收益), 代谢应激 (能量紧 → 少说), 内源振荡器 A
        (节律基因, bind 已每步推进), 新奇度 (世界新 → 多看). σ 软饱和 = 结构性有界
        (无 clamp). 采样用预分配 _gate_rand 就地 uniform_ (零热路径新建); 决策写
        _metab_gate_hit (随检查点持久化); CPU 路由镜像由 HybridPCNet.sync_gate() 取
        (借阀门既有每步同步, 零新增同步点). free_run 无 lm 不结算 (门冻结).
        """
        net = self.net
        if ctx.free_run or sh.lm is None:
            return
        A = net._intr_sin.index_select(0, net._intr_cnt.long().squeeze(0))  # [1] fp16
        _stress = getattr(net, "_metab_stress", net._zero1)
        logit = (
            net.cfg.metab_gate_kappa_g * (net._metab_gain_eco - net._metab_gain_perc)
            - net.cfg.metab_gate_kappa_s * _stress
            + net.cfg.metab_gate_kappa_a * (A - 0.5)
            - net.cfg.metab_gate_kappa_n * (net._metab_nov - 0.5)
        )
        net._metab_psay.copy_(torch.sigmoid(logit))
        net._gate_rand.uniform_()  # 就地重取样 (预分配缓冲)
        net._metab_gate_hit.copy_(
            (net._gate_rand < net._metab_psay).to(torch.float16)
        )
