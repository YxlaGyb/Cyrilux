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


def rel_norm(raw: float, ema: float, eps: float = 1e-6) -> float:
    """自归一化相对量 (rel_n 家族, bind.py 同款): raw/(ema+raw+eps) ∈ [0,1)."""
    return raw / (ema + raw + eps)


def gate_psay(g_eco: float, g_perc: float, stress: float, A: float, nov: float,
              k_g: float, k_s: float, k_a: float, k_n: float) -> float:
    """行为门纯函数 (测试锚; GPU 版 _update_gate 与之公式一一对应).

    p_say = σ(κ_g·(g_eco−g_perc) − κ_s·stress + κ_A·(A−0.5) − κ_n·(nov−0.5)).
    符号: 账本差 >0 (说比看近期更赚) → 想说; 应激高 (能量紧) → 省着少说;
    振荡器 A 高相位 → 说 (节律基因); 新奇高 → 多看少说. σ 软饱和 = 结构性有界.
    """
    logit = (k_g * (g_eco - g_perc) - k_s * stress
             + k_a * (A - 0.5) - k_n * (nov - 0.5))
    return 1.0 / (1.0 + math.exp(-logit))


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
        # 1) F := 绝对误差能量 (P2 货币修正): 归一化误差平方是尺度无关的 — 饥荒
        # (预测相对变差) 会被误差自身标准差吸收, E 对饥荒零判别力 (术前探针实证:
        # 随机字节/分布漂移/W_t 渐进病理下 E min 都停在 -0.008 平台噪声带内).
        # 绝对货币: 乱码输入 → 原始误差能量暴涨 → E 深潜 → 死亡线可达.
        # sh.metab_f 仍供学习链用归一化 final_error (feedforward 契约不变, 显式优于隐式).
        eps_lm_pad = sh.lm.eps_lm_pad
        eps4 = sh.errs.eps4
        s_pred = eps_lm_pad.std(dim=-1, keepdim=True) * 1.01 + eps_lm_pad.std() * 1e-3
        err_pred_norm = eps_lm_pad / s_pred
        s_rec = eps4.std(dim=-1, keepdim=True) * 1.01 + eps4.std() * 1e-3
        err_recon_norm = eps4 / s_rec
        final_error = err_pred_norm + 0.2 * err_recon_norm
        sh.metab_f = final_error  # 供 _update_feed_ff 取用 (归一化, F 单一出处)
        f_now = eps_lm_pad.square().mean() + 0.2 * eps4.square().mean()  # [1] fp16 标量
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
        # 死亡判据计数 (P2 终裁, 下探缺失; P3-b 分相): 平台期 F 规律性下探 (低能耗时刻),
        # 饥荒 = F 钉高位零下探 (v3 实测: 平台 q25-q95 = 0.975-0.986 ≈ 天花板, 但每 ~14 步
        # 一下探; famine_rand 400 步零下探). dip = f_now < base·(1−metab_dip_margin); 每次
        # dip 清零, 无 dip 连续 metab_death_steps 步 → 死亡回合. (v1/v2/v3 三次证伪:
        # E 域/绝对域/高域余量均无判别力 — 平台均值已钉天花板, κ 阈值落在信号域之外.)
        # 分相: 只有感知相能下探清零 (F_base 慢基线也只在感知相更新 — 回声相 F 是
        # 自生成流, 尺度/语义不同轨); 回声相无条件 +1 — 纯回声锁 250 步即死
        # ("长时间不看世界 = 病态", P2 死亡语义在此保留).
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

    def _update_gate(self, ctx, sh):
        """P3-b 行为门 (节律内生): 本步末算, 供下一步路由 — 全 GPU 零同步.

        门输入全部来自体内 (见 gate_psay 纯函数, 两侧公式一一对应): 行为账本差
        (g_eco−g_perc = 说 vs 看的近期收益), 代谢应激 (能量紧 → 少说), 内源振荡器 A
        (节律基因, bind 已每步推进), 新奇度 (世界新 → 多看). σ 软饱和 = 结构性有界
        (无 clamp). 采样用预分配 _gate_rand 就地 uniform_ (零热路径新建); 决策写
        _metab_gate_hit (随检查点持久化); CPU 路由镜像由 DensePCNet.sync_gate() 取
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
