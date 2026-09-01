"""
体内代谢域
F (自由能) 每步计价 → E 能量账本 → R 生存信号 (W_act 资格迹货币).
"""

from __future__ import annotations

import torch

from ._common import _MixinBase


class MetabolismMixin(_MixinBase):
    """代谢域 (方法挂载到 LearningEngine)."""

    def _update_metabolism(self, ctx, sh):
        """体内代谢: F 每步计价 (感知/回声同构) → E 账本 → R 生存信号.

        符号铁律 (条件二, 写死): **F 下降 → ΔF 为正 → ΔE 上升 → R > 0**.
        R 的输入永不得换成 F 水平、误差绝对值或任何"压平即胜"的可收割量
        (dark room 禁区: 奖励误差水平会让压平分布成为过关最便宜路径).
        """
        net = self.net
        if ctx.free_run or sh.lm is None:
            # free_run 无 lm 信号 (_build_lm_signal 返回 None), 代谢不结算
            return
        # 1) F := final_error (与 _update_feed_ff 同式; 感知+回声都算, 独立于"非冻结"守卫)
        eps_lm_pad = sh.lm.eps_lm_pad
        eps4 = sh.errs.eps4
        s_pred = eps_lm_pad.std(dim=-1, keepdim=True) * 1.01 + eps_lm_pad.std() * 1e-3
        err_pred_norm = eps_lm_pad / s_pred
        s_rec = eps4.std(dim=-1, keepdim=True) * 1.01 + eps4.std() * 1e-3
        err_recon_norm = eps4 / s_rec
        final_error = err_pred_norm + 0.2 * err_recon_norm
        sh.metab_f = final_error  # 供 _update_feed_ff 取用 (F 单一出处)
        f_now = final_error.square().mean()  # [1] fp16 标量
        if net._metab_F_prev < 0:
            # 首步哨兵 (F_prev=-1): 只登记不结算
            net._metab_F_prev.copy_(f_now)
            net._metab_R = torch.zeros(1, dtype=torch.float16, device=ctx.dev)
            return
        # 2) 有符号收入: E ← E·(1−d) + c·ΔF; 退步=负收入, E 可短负=濒死记账 (P2 死亡判据)
        df = net._metab_F_prev - f_now  # 进步为正
        net._metab_F_prev.copy_(f_now)
        e_old = net._metab_E.clone()
        net._metab_E.mul_(1.0 - net.cfg.metab_d).add_(net.cfg.metab_c * df)
        de = net._metab_E - e_old  # 有符号 ΔE
        net._metab_de_mad.mul_(net.cfg.metab_mad_alpha).add_(
            (1.0 - net.cfg.metab_mad_alpha) * de.abs()
        )
        if net._metab_de_mad <= 0:
            # 冷启动 (world_lang 同款): MAD 首个样本 = |ΔE|
            net._metab_de_mad.copy_(de.abs())
        # 3) R 原语 (R1 契约同构): tanh(ΔE/(div·MAD)) ∈ (−1,1); div=metab_tanh_div 标定值
        #    (R1 契约 [0.3,1] 重标: 0.95/2.0 下 median(leg)=0.26 偏饿 → 0.90/1.5 → 0.38).
        #    除 ‖迹‖ 在消费端 (_update_w_act 于迹更新后执行, 被乘的迹就是除的那个迹 —
        #    R1 精确形态, 且首步迹非零, 不会出现 /0 爆炸). 全 GPU fp16 零 .item().
        net._metab_R = torch.tanh(de / (net.cfg.metab_tanh_div * net._metab_de_mad))
