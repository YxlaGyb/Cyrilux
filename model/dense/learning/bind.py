"""绑定域: M_l5 Foldiak 反赫布 + W_bind/W_bind_self 概念槽 + 内在驱动振荡器.

从原 learn() 单函数按块拆分, 块内语句顺序逐行保持 (数值逐位等价).
注意: _decorr_W(net.W_bind_self.data.T, ...) 必须传转置视图 — 副本不写回原参数.
"""

from __future__ import annotations

import torch

from ...modulation import soft_norm_preserve
from ..forward import _rms
from ._common import _decorr_W, _elig_accum, _MixinBase


class BindMixin(_MixinBase):
    """绑定域 (方法挂载到 LearningEngine)."""

    def _update_Ml5(self, ctx, sh):
        """Foldiak 反赫布侧抑制更新 (方案 D): dM = z_out 协方差 (白化本质),
        零对角, 指数遗忘 ×0.99 防爆炸. 不做 Frobenius 归一化 — 归一化把 dM
        缩到 ~1e-4 (1024² 矩阵范数 ~1000), ×0.01 → ~1e-6 被 fp16 舍入,
        装饰性失效 (第 8 轮同款 bug 翻版, 实测 M_offdiag 0.0004 纹丝不动);
        z_out 已逐行 RMS 归一化, 协方差元素 ∈[-1,1], 增量直接可表示
        """
        net = self.net
        dev = ctx.dev
        a5 = ctx.a5
        z5 = net._z5
        z5_flat = z5.reshape(-1, a5).to(torch.float16)
        z5_flat = z5_flat / (z5_flat.norm(dim=-1, keepdim=True) + 1e-3)
        cov = z5_flat.transpose(0, 1) @ z5_flat / z5_flat.shape[0]
        eye_mask = 1.0 - torch.eye(a5, device=dev, dtype=torch.float16)
        net.M_l5[:a5, :a5].data.mul_(0.99).add_((cov * eye_mask) * (0.01 * ctx.learn_boost))

    def _update_bind(self, ctx, sh):
        """竞争性概念绑定层赫布更新 (任务 4, 纯外积, 零 BP) + 内在驱动 W_bind_self."""
        net = self.net
        dev, N, S = ctx.dev, ctx.N, ctx.S
        echo_world_frozen = ctx.echo_world_frozen
        a4 = ctx.a4
        z4 = net._z4
        if hasattr(net, "_bind_vec") and not echo_world_frozen:
            z4n = _rms(z4)
            bind_t = net._bind_vec  # 纯 z_bind (第 75 轮: 去均值对称抹差异 → 无偏置积累,
            # 高激活槽保留幅度差 → Hebbian 正反馈放大 → 分化种子; 分流抑制防垄断)
            # 独立逐样本三因子 Hebbian (第 75 轮最终): 每样本独立 surprise EMA,
            # gain_n = clip(s_n/ema_n, 0.3, 5.0) — 样本自身历史决定自身门控,
            # 高惊喜样本外积单独放大 → 非对称注入. 纯局部 (每样本只看自己)
            # (free_run: 无字节误差 eps_total → 门控退化为 1, 样本竞争由 0.3-5.0
            # clamp 下限自然覆盖; N=1 单样本时均一化无意义)
            # (第 80 轮: 自回声有真实字节误差 eps_total, 走 else 分支正常门控)
            if ctx.free_run:
                gain_n = torch.ones(N, dtype=torch.float16, device=dev)
            else:
                s_n = sh.lm.eps_total.square().mean(dim=-1).mean(dim=-1)  # [N] 每样本均方误差
                # _s_ema_n 缓冲按 batch 自适应扩容 (固定 8 与 CLI batch=48 不匹配
                # → 形状错误; 扩容后新样本 EMA=1.0 起, 语义不变)
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
            z4n_w = z4n[:, :-1] * gain_n[:, None, None]  # [N, S-1, a4]
            dW_bind = (
                torch.einsum("nsd,nsq->dq", z4n_w, bind_t[:, :-1]) / (gain_n.sum() + 1e-8) * (1.0 / (S - 1))
            )
            dW_bind_a = (
                dW_bind
                + _elig_accum(net, "W_bind", dW_bind)
                * getattr(net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16))
            ) * (ctx.eta * 2.0)
            net._rho_bind = dW_bind_a.norm() / (net.W_bind[:a4].norm() + 1e-8)
            net.W_bind[:a4].data.mul_(0.9995)
            net.W_bind[:a4].data += dW_bind_a
            # 自发噪声破缺 (第 75 轮): 对称吸引子上注入与列间相似度成比例的
            # 涨落 — 高相似度 → 强噪声 → 方向被扰动 → Hebbian 放大差异 → 相变.
            # 噪声幅度自适应衰减 (列分离后 repulsion→0 → 噪声→基线), 模拟
            # 发育期自发发放. 纯局部 (每列只看自身与其他列内积), 与 decorr 同构
            W_col = net.W_bind[:a4].data.T  # [16, a4]
            cov_col = W_col @ W_col.T  # [16,16] 列间协方差
            repulsion = cov_col @ W_col  # [16, a4] 各列受到的净共线拉力
            rel_rep = repulsion.norm(dim=1, keepdim=True) / (W_col.norm(dim=1, keepdim=True) + 1e-8)
            scale = (1e-4 * rel_rep.clamp(min=1e-4)).to(torch.float16)  # [16,1]
            noise = torch.randn_like(W_col) * scale
            net._noise_scale = scale.mean()  # 诊断: 平均噪声幅度
            net._col_cos = (
                (cov_col * (1.0 - torch.eye(net.bind_slot_dim, device=dev, dtype=torch.float16))).abs().mean()
            )
            W_col.add_(noise)
            net.W_bind[:a4].data = W_col.T.contiguous()
            soft_norm_preserve(net.W_bind[:a4].data)
            # W_bind 行去同质化 (与 W_35 同款, 破秩 1 自锁): E_bind 幂迭代主方向
            # 投影抑制 + E_bind 相关统计累计 (规格书 3/4)
            _decorr_W(net.W_bind[:a4].data, net.E_bind[:a4, :a4], learn_boost=ctx.learn_boost)
            # W_bind 列方向 decorr (第 75 轮安全网): 逐样本加权破对称后防单槽垄断.
            # 列空间 = 槽维 16, 转置后行 decorr 同型 (E_bind_col 16×16)
            _decorr_W(net.W_bind[:a4].data.T.contiguous(), net.E_bind_col, learn_boost=ctx.learn_boost)
            # W_bind_self 内在驱动 Hebbian (第 76 轮战略转向: 双重驱动):
            # 原共现规则收敛均匀转移, 误差门控 (裁决 5) 被否决 — 仍是外部信号
            # 被动驱动. 新规则: intrinsic_drive 门控 — 自发振荡 + 内部状态耦合,
            # 独立于外部预测误差 (大脑无感官输入时中脑调质系统自发放电先例).
            # intr[t] ∈ [0,1] 乘在 z_bind 转移外积上: 自发活动高潮期学习转移,
            # 低潮期不学 — 由内部节律驱动分化, 而非外部误差波动
            if getattr(net, "_bind_loop", True):
                zb = net._bind_vec
                zb_pre = zb[:, :-1]  # [N,S-1,K]
                zb_post = zb[:, 1:]  # [N,S-1,K] 对齐 learn_mask (t+1)
                zb_post = zb_post * ctx.learn_mask.to(torch.float16).unsqueeze(0).unsqueeze(-1)
                # 自发活动发生器: 相位计数器 (0-19 整数, fp16 精确) + 正弦查表
                # (预计算 20 项 fp16, 周期 ~20 步) + 槽切换功率耦合. 全 fp16
                # 零 GPU→CPU 同步 (index_select 纯张量查表), 零 fp32
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
                # 裁决 10: intrinsic 调制 W_bind_self 学习率 — η_self = η_base·(1+0.5·sinφ)
                # 高潮期 (A→1) 学习率 1.5× → 自洽误差上升 → 系统探索新表达;
                # 低谷期 (A→0) 学习率 1.0× → 自洽误差回落 → 固化探索成果.
                # 内部节律驱动探索-固化循环, 无新计算路径, 零 NaN 风险
                eta_self = (ctx.eta * 2.0) * (1.0 + 0.5 * A)
                net.W_bind_self.data += dW_self * eta_self
                # 列方向 decorr (第 76 轮裁决): 转置后行 decorr 同型 — 16 列各为
                # 源槽转移向量, 均匀统计下收敛同向量 (实测列相似 0.906), 斥力
                # 迫使分化. 更新后、soft_norm 前挂载 (与 W_bind 主矩阵同序).
                # 注意: 必须传转置视图 (非 contiguous 副本) — 副本修改不写回
                # 原参数 (第 76 轮实测: 0.906→0.935 不降反升, 装饰性失效)
                _decorr_W(net.W_bind_self.data.T, net.E_bind_self, learn_boost=ctx.learn_boost)
                soft_norm_preserve(net.W_bind_self.data)
