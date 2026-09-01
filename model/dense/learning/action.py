"""表达端域"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ...modulation import soft_norm_preserve
from ._common import _elig_accum, _MixinBase


class ActionMixin(_MixinBase):
    """表达端域 (方法挂载到 LearningEngine)."""

    def _update_w_act(self, ctx, sh):
        """W_act 闭环自洽生成学习 (表达者范式): 切断外部目标驱动, 仅 closed_loop/自回声."""
        net = self.net
        dev = ctx.dev
        if (
            not (ctx.closed_loop or ctx.echo_loop)
            or not hasattr(net, "_intent_pot")
            or not hasattr(net, "_gen_bytes")
            or not getattr(net, "_act_enabled", True)
        ):
            return
        N = ctx.N
        pot = net._intent_pot  # [N,S,256] 动作电位
        zbind = net._bind_vec
        gb = net._gen_bytes  # [N,N_gen] 自生成字节 (表达)
        N_gen = gb.shape[1]
        # 决策态对齐: zb_g 与 gb_a/oh_g/probs_g 取同一决策时点 (字节进入前)
        n_a = N_gen - 1  # 决策态对齐长度
        zb_g = zbind[:, -N_gen:-1]  # [N,n_a,16] 决策时点槽状态
        # 学习进步信号 LP_t = ε_{t-1} - ε_t: 误差下降 → 保留行为, 上升 → 抑制 (打破复读)
        zb_prev = torch.cat([zb_g[:, :1], zb_g[:, :-1]], dim=1)  # [N,n_a,16]
        pred_self = torch.softmax((zb_prev @ net.W_bind_self) * 4.0, dim=-1)
        eps_t = (zb_g - pred_self).square().mean(dim=-1, keepdim=True)  # [N,n_a,1] ε_t
        eps_prev = torch.cat([eps_t[:, :1], eps_t[:, :-1]], dim=1)  # ε_{t-1}
        LP = (eps_prev - eps_t).detach()  # 学习进步 [N,n_a,1]
        alpha = 50.0  # tanh 缩放 (ε 量级 ~0.003, ×50 → LP 归一)
        phi = 0.5 * (1.0 - torch.tanh(alpha * LP))  # 增益 [0,1]
        # 决策态对齐: p_gen = P(g_t | 上下文不含 g_t), 复读惊喜 >0 → 不被固化 (否则复读自锁)
        gb_a = gb[:, 1:]  # [N,n_a] 对齐生成字节
        oh_g = F.one_hot(gb_a, num_classes=256).to(torch.float16)  # [N,n_a,256]
        gb_start = ctx.S - N_gen  # 生成段起始位置
        probs_g = sh.lm.probs_lm[:, gb_start : gb_start + n_a]  # [N,n_a,256] 决策态分布
        p_gen = (probs_g * oh_g).sum(dim=-1, keepdim=True)  # [N,n_a,1]
        wlm_err = (1.0 - p_gen).detach()  # [N,n_a,1] 外部弱约束
        # surprise = 世界模型主导 (wlm_err 0.9) + 内部转移弱约束 (phi 0.1)
        surprise = (0.1 * phi + 0.9 * wlm_err).detach()
        # 字节频率门控: 罕见字节抑制衰减
        fa = net._freq_act
        oh_gen = F.one_hot(gb_a, num_classes=256).to(torch.float16)  # [N,n_a,256]
        fa.mul_(0.99).add_(0.01 * oh_gen.mean(dim=(0, 1)))
        beta = 0.5  # 新颖偏好强度
        freq_gate = (1.0 - beta * (1.0 - fa)).unsqueeze(0).unsqueeze(0)  # [1,1,256]
        surprise = surprise * freq_gate  # 逐字节频率调制
        net._LP = LP  # 诊断: 学习进步分布
        # 软目标 dW_act = zb_g^T @ (8·probs_g - oh_g), probs 项 ×8 使强化主导
        dW_act = (zb_g.transpose(-2, -1) @ (8.0 * probs_g - oh_g)).mean(dim=0)
        # 稳态抑制保留 (防字节垄断)
        pot_sm = torch.softmax(pot[:, -N_gen:-1].detach(), dim=-1)  # 决策态对齐
        dW_act = dW_act - 0.1 * (zb_g.transpose(-2, -1) @ pot_sm).mean(dim=0)
        dW_act = dW_act / (dW_act.norm() + 1e-8)  # 单步幅度上界 (幅度-方向解耦)
        # 信任域: 单位化方向 × 平均列范数 × 5.0 = 真 5% 步幅
        intr_d = getattr(net, "_intr_drive", torch.tensor(0.5, device=dev, dtype=torch.float16))
        col_norm = net.W_act.data.norm(dim=0).mean()  # 平均列范数 (~1.0)
        # 生存信号 R 由体内代谢产原语 (_metab_R = tanh(ΔE/2MAD)); ε 维度降为诊断.
        # 除迹在消费端 (迹更新后): 注入幅 = tanh(ΔE/2MAD) ∈ (−1,1) (R1 契约, 无 /0 爆炸).
        eps_now = wlm_err.mean()
        net._lm_eps = eps_now  # 诊断: 本窗 ε_lm (fp16 张量, 零同步)
        # 恒温器: 内部自校准锚 (_lang_eps_ema, 感知相位统计) + 饥饿应激压缩变异 (代谢-行为耦合)
        _gt = getattr(net, "_gen_temp", None)
        if _gt is None:
            net._gen_temp = torch.tensor(4.0, dtype=torch.float16, device=dev)
        else:
            _anchor = getattr(net, "_lang_eps_ema", None)
            if _anchor is not None:
                _t2 = _gt * torch.exp(0.5 * (_anchor - eps_now))
                _we = getattr(net, "_metab_E", None)
                _er = getattr(net, "_metab_E_ref", None)
                if _we is not None and _er is not None:
                    _t2 = _t2 * torch.exp(-0.2 * torch.relu(_er - _we))
                # τ 硬上限 2.0: 展平骗裁判的廉价解在 τ≈9.5-10, 上限 2.0 使其物理不可达 (保留选择压)
                net._gen_temp = torch.minimum(torch.maximum(_t2, torch.tensor(1.0, dtype=torch.float16, device=dev)), torch.tensor(2.0, dtype=torch.float16, device=dev))
        # 资格迹: 迹输入 = 行为外积 (决策态槽 × 实际字节 one-hot), R 生存信号在迹更新后除迹调制
        zbg_ac = zb_g - zb_g.mean(dim=1, keepdim=True)
        dW_elig = (zbg_ac.transpose(-2, -1) @ oh_g).mean(dim=0)
        E_act = _elig_accum(net, "W_act", dW_elig)
        _wr = getattr(net, "_metab_R", None)
        net._survival_signal = (
            _wr / (E_act.norm() + 1e-6) if _wr is not None
            else torch.zeros(1, dtype=torch.float16, device=dev)
        )
        dW_act = dW_act + E_act * net._survival_signal
        net.W_act.data += dW_act * (net.cfg.lm_lr_boost * 0.2) * (0.5 + intr_d) * col_norm * 5.0
        # 复读检测跟踪 (随机扰动已升级为 forward.py 内部状态错误配对, 此处仅跟踪供诊断)
        gb_recent = gb[:, -10:]  # [N,10] 最近 10 字节
        oh_r = F.one_hot(gb_recent, num_classes=256).to(torch.float16)  # [N,10,256]
        n_uniq = (oh_r.sum(dim=1) > 0).sum(dim=-1).to(torch.float16)  # [N]
        rep_run = getattr(net, "_rep_run", torch.zeros(N, device=dev, dtype=torch.float16))
        in_rep = (n_uniq < 3).to(torch.float16)  # [N]
        rep_run = torch.where(in_rep > 0, rep_run + 1.0, torch.zeros_like(rep_run))
        net._rep_run = rep_run
        net._rep_frac = in_rep.mean().to(torch.float16)  # 诊断 (fp16 张量, 零同步)
        # 列范数保持 (0.8-1.2): 槽列有界防溢出; 转置视图 in-place 直接写回原参数
        soft_norm_preserve(net.W_act.data.T)
