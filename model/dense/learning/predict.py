"""
自组织预测引擎域
层间局部误差 + RPE 门控 + W_pred_* 注入.

从原 learn() 单函数按块拆分, 块内语句顺序逐行保持 (数值逐位等价).
"""

from __future__ import annotations

import torch

from ...modulation import soft_norm_preserve
from model.modulation import rms_norm
from ._common import _elig_accum, _energy_constraint, _MixinBase, _rho_ctrl

# 折扣多步预测目标 (未来引力): 目标 = 未来 K 步折扣和
K_FUT = 8
GAMMA = 0.9


class PredictMixin(_MixinBase):
    """自组织预测引擎域 (方法挂载到 LearningEngine)."""

    def _update_pred_engine(self, ctx, sh):
        """未来折扣目标 + W_pred_54/43 更新 + W_35/W_42 注入 + local_err 诊断."""
        net = self.net
        dev = ctx.dev
        free_run = ctx.free_run
        dim_3, dim_4, dim_5 = ctx.dim_3, ctx.dim_4, ctx.dim_5
        z4, z3, z5 = net._z4, net._z3, net._z5
        Wb = net.W_35[:dim_5].data

        # rms_norm 零向量保护: 原 std 归一化在 free_run 低幅度 z4 下放大 25×, 点积超 fp16 上限
        z4_pred_in = rms_norm(z4)
        z3_pred_in = rms_norm(z3)
        if free_run:
            global_rpe = torch.tensor(0.0, dtype=torch.float16, device=dev)  # 无字节误差 → rpe=1
        else:
            global_rpe = (sh.lm.eps_total.square().mean().sqrt() * 10.0).clamp(max=1.0)  # 全局误差幅度
        Wp54_a = net.W_pred_54[:dim_5, :dim_4]
        Wp43_a = net.W_pred_43[:dim_4, :dim_3]
        pred_l5 = z4_pred_in @ Wp54_a.T  # [N,S,dim_5]
        pred_l4 = z3_pred_in @ Wp43_a.T  # [N,S,dim_4]
        # 未来折扣目标 z_future[t] = Σ_k γ^k·z[t+k+1]: 向未来轨迹漂移的牵引力, 末尾 K 位 mask 掉
        z5_fut = torch.zeros_like(z5)
        z4_fut = torch.zeros_like(z4)
        mask5 = torch.zeros(ctx.S, dtype=torch.bool, device=dev)
        mask4 = torch.zeros(ctx.S, dtype=torch.bool, device=dev)
        g = 1.0
        for k in range(K_FUT):
            if k + 1 < ctx.S:
                z5_fut[:, : -k - 1] += g * z5[:, k + 1 :]
                z4_fut[:, : -k - 1] += g * z4[:, k + 1 :]
                mask5[: -k - 1] = True
                mask4[: -k - 1] = True
            g *= GAMMA
        local_err_l5 = z5_fut - pred_l5  # 只在 mask5 有效位有意义
        local_err_l4 = z4_fut - pred_l4
        rpe = (1.0 + global_rpe).to(torch.float16)
        # 注入路径误差预归一化: 递归增益高时裸误差外积范数超 fp16 上限 → inf·0 = NaN (pre-norm 非 clamp)
        err5_m = rms_norm(local_err_l5[:, mask5])  # [N,T5,dim_5]
        err4_m = rms_norm(local_err_l4[:, mask4])  # [N,T4,dim_4]
        z3_m = z3[:, mask5]
        dW_pred35 = (err5_m.transpose(-2, -1) @ rms_norm(z3_m)).mean(dim=0) / max(1, int(mask5.sum()))
        Wb.data += _rho_ctrl(dW_pred35 * ctx.eta * rpe * 1.5, Wb, "inj35", net)
        # free_run+突触缩放时豁免 soft_norm (行范数 = 增益自由度归系统, 训练模式保持)
        if not (free_run and net.cfg.wt_syn_scaling):
            soft_norm_preserve(Wb.data)
        if not free_run and not ctx.echo_world_frozen:
            # W_42 注入: local_err_L4 逆映射到 dim_2 (free_run/回声相位 W_42 冻结)
            z4_m = z4[:, mask4]
            local_err_l4_2 = rms_norm(err4_m @ net.W_42[:ctx.dim_2].T)
            dW_pred42 = (local_err_l4_2.transpose(-2, -1) @ rms_norm(z4_m)).mean(dim=0) / max(1, int(mask4.sum()))
            net.W_42[:ctx.dim_2].data += _rho_ctrl(dW_pred42 * ctx.eta * rpe * 1.5, net.W_42[:ctx.dim_2], "inj42", net)
            soft_norm_preserve(net.W_42[:ctx.dim_2].data)
        # W_pred_* 自更新; 回声相位冻结 (W_pred_* 是世界模型)
        if not ctx.echo_world_frozen:
            dWp54 = (err5_m.transpose(-2, -1) @ z4_pred_in[:, mask5]).mean(dim=0)
            dWp54 = _energy_constraint(net, Wp54_a.data, dWp54, err5_m, "_active_ema_wp54")
            dWp54 = dWp54 + _elig_accum(net, "W_pred_54", dWp54) * getattr(net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16))
            Wp54_a.data += _rho_ctrl(dWp54 * ctx.eta, Wp54_a, "wp54", net)
            dWp43 = (err4_m.transpose(-2, -1) @ z3_pred_in[:, mask4]).mean(dim=0)
            dWp43 = _energy_constraint(net, Wp43_a.data, dWp43, err4_m, "_active_ema_wp43")
            dWp43 = dWp43 + _elig_accum(net, "W_pred_43", dWp43) * getattr(net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16))
            Wp43_a.data += _rho_ctrl(dWp43 * ctx.eta, Wp43_a, "wp43", net)
            soft_norm_preserve(Wp54_a.data)
            soft_norm_preserve(Wp43_a.data)
        # local_err 相对能量诊断; 分母加性保护防除零放大 (非 clamp)
        z5_scale = z5.square().mean() + 0.01 * z5.square().mean() + 1e-4
        z4_scale = z4.square().mean() + 0.01 * z4.square().mean() + 1e-4
        net._local_err_l5 = (local_err_l5[:, mask5].square().mean() / z5_scale).detach()
        net._local_err_l4 = (local_err_l4[:, mask4].square().mean() / z4_scale).detach()
        sh.l5_local_err = local_err_l5[:, mask5].square().mean().detach()
