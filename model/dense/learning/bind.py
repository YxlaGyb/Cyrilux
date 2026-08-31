"""
绑定域
M_l5 Foldiak 反赫布 + W_bind/W_bind_self 概念槽 + 内在驱动振荡器.

从原 learn() 单函数按块拆分, 块内语句顺序逐行保持 (数值逐位等价).
_decorr_W(net.W_bind_self.data.T, ...) 必须传转置视图 (副本不写回原参数).
"""

from __future__ import annotations

import torch

from ...modulation import soft_norm_preserve
from model.modulation import rms_norm
from ._common import _decorr_W, _elig_accum, _MixinBase


class BindMixin(_MixinBase):
    """绑定域 (方法挂载到 LearningEngine)."""

    def _update_Ml5(self, ctx, sh):
        """Foldiak 反赫布侧抑制更新: dM = z_out 协方差 (白化本质), 零对角, 指数遗忘防爆炸.

        z_out 已逐行 RMS 归一化, 协方差元素 ∈[-1,1], 增量 fp16 直接可表示 (不做 Frobenius 归一化).
        """
        net = self.net
        dev = ctx.dev
        dim_5 = ctx.dim_5
        z5 = net._z5
        z5_flat = z5.reshape(-1, dim_5).to(torch.float16)
        z5_flat = z5_flat / (z5_flat.norm(dim=-1, keepdim=True) + 1e-3)
        cov = z5_flat.transpose(0, 1) @ z5_flat / z5_flat.shape[0]
        eye_mask = 1.0 - torch.eye(dim_5, device=dev, dtype=torch.float16)
        net.M_l5[:dim_5, :dim_5].data.mul_(0.99).add_((cov * eye_mask) * (0.01 * ctx.learn_boost))

    def _update_bind(self, ctx, sh):
        """竞争性概念绑定层更新 (纯外积) + 内在驱动 W_bind_self."""
        net = self.net
        dev, N, S = ctx.dev, ctx.N, ctx.S
        echo_world_frozen = ctx.echo_world_frozen
        dim_4 = ctx.dim_4
        z4 = net._z4
        if hasattr(net, "_bind_vec") and not echo_world_frozen:
            z4n = rms_norm(z4)
            bind_t = net._bind_vec  # 纯 z_bind: 保留幅度差作 Hebbian 分化种子, 分流抑制防垄断
            # 每样本独立 surprise EMA 门控 gain_n = s_n/ema_n (clamp 0.3..5), 高惊喜样本外积放大
            if ctx.free_run:
                gain_n = torch.ones(N, dtype=torch.float16, device=dev)
            else:
                s_n = sh.lm.eps_total.square().mean(dim=-1).mean(dim=-1)  # [N] 每样本均方误差
                # 缓冲按 batch 扩容 (新样本 EMA=1.0 起)
                if net._s_ema_n.shape[0] < N:
                    old = net._s_ema_n
                    net.register_buffer(
                        "_s_ema_n",
                        torch.cat([old, torch.ones(N - old.shape[0], dtype=torch.float16, device=dev)]),
                    )
                    del old
                ema_n = net._s_ema_n[:N]
                rel_n = s_n / (ema_n + s_n + 1e-6)  # 自归一化相对惊喜 (量级无关, O(1))
                gain_n = (rel_n * 5.0).clamp(0.3, 5.0)
                net._s_ema_n[:N].mul_(0.95).add_(0.05 * s_n)
            net._gain_n = gain_n  # 诊断: 逐样本增益分布
            z4n_w = z4n[:, :-1] * gain_n[:, None, None]  # [N, S-1, dim_4]
            dW_bind = (
                torch.einsum("nsd,nsq->dq", z4n_w, bind_t[:, :-1]) / (gain_n.sum() + 1e-8) * (1.0 / (S - 1))
            )
            dW_bind_a = (
                dW_bind
                + _elig_accum(net, "W_bind", dW_bind)
                * getattr(net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16))
            ) * (ctx.eta * 2.0)
            net._rho_bind = dW_bind_a.norm() / (net.W_bind[:dim_4].norm() + 1e-8)
            net.W_bind[:dim_4].data.mul_(0.9995)
            net.W_bind[:dim_4].data += dW_bind_a
            # 自发噪声破缺: 注入与列间相似度成比例的涨落, 破对称吸引子 → Hebbian 放大差异
            W_col = net.W_bind[:dim_4].data.T  # [16, dim_4]
            cov_col = W_col @ W_col.T  # [16,16] 列间协方差
            repulsion = cov_col @ W_col  # [16, dim_4] 各列受到的净共线拉力
            rel_rep = repulsion.norm(dim=1, keepdim=True) / (W_col.norm(dim=1, keepdim=True) + 1e-8)
            scale = (1e-4 * rel_rep.clamp(min=1e-4)).to(torch.float16)  # [16,1]
            noise = torch.randn_like(W_col) * scale
            net._noise_scale = scale.mean()  # 诊断: 平均噪声幅度
            net._col_cos = (
                (cov_col * (1.0 - torch.eye(net.bind_slot_dim, device=dev, dtype=torch.float16))).abs().mean()
            )
            W_col.add_(noise)
            net.W_bind[:dim_4].data = W_col.T.contiguous()
            soft_norm_preserve(net.W_bind[:dim_4].data)
            # 行/列 decorr 破秩 1 自锁
            _decorr_W(net.W_bind[:dim_4].data, net.E_bind[:dim_4, :dim_4], learn_boost=ctx.learn_boost)
            _decorr_W(net.W_bind[:dim_4].data.T.contiguous(), net.E_bind_col, learn_boost=ctx.learn_boost)
            # W_bind_self 内在驱动 Hebbian: intrinsic_drive 门控自发生长, 独立于外部误差
            if getattr(net, "_bind_loop", True):
                zb = net._bind_vec
                zb_pre = zb[:, :-1]  # [N,S-1,K]
                zb_post = zb[:, 1:]  # [N,S-1,K] 对齐 learn_mask (t+1)
                zb_post = zb_post * ctx.learn_mask.to(torch.float16).unsqueeze(0).unsqueeze(-1)
                # 自发活动发生器: 相位计数器 + 正弦查表 + 槽切换功率耦合 (纯 fp16, index_select 无同步)
                cnt = net._intr_cnt
                cnt.add_(1.0)
                cnt.remainder_(20.0)  # 整数取模, fp16 精确
                A = net._intr_sin.index_select(0, cnt.long().squeeze(0))  # [1] fp16 查表
                # 槽切换功率: z_bind 相邻步差能量 (内部状态, 非预测误差)
                sw = (zb[:, 1:] - zb[:, :-1]).square().mean(dim=(0, 2))  # [S-1]
                om = net._intr_omega
                om.mul_(0.98).add_(0.02 * sw.mean())
                omega = sw / (om + sw + 1e-6)  # 自归一化 [0,1] (同 rel_n 家族)
                intr = (0.5 * A + 0.5 * omega).unsqueeze(0).unsqueeze(-1)  # [1,S-1,1]
                # 诊断标量 (fp16 张量, 供监控; 训练不读)
                net._intr_drive = (0.5 * A + 0.5 * omega.mean()).to(torch.float16)
                dW_self = (
                    zb_pre.transpose(-2, -1) @ ((zb_post - zb_post.mean(dim=-1, keepdim=True)) * intr)
                ).mean(dim=0)
                dW_self = dW_self / (dW_self.norm() + 1e-8)
                dW_self = dW_self + _elig_accum(net, "W_bind_self", dW_self) * getattr(
                    net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16)
                )
                net.W_bind_self.data.mul_(0.9995)
                # intrinsic 调制学习率 η_self = η_base·(1+0.5·A): 高潮期探索, 低谷期固化
                eta_self = (ctx.eta * 2.0) * (1.0 + 0.5 * A)
                net.W_bind_self.data += dW_self * eta_self
                # 列方向 decorr: 传转置视图 (副本不写回原参数)
                _decorr_W(net.W_bind_self.data.T, net.E_bind_self, learn_boost=ctx.learn_boost)
                soft_norm_preserve(net.W_bind_self.data)
