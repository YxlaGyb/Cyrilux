"""自组织预测引擎域 (第 50 轮): 层间局部误差 + 多巴胺 RPE 门控 + W_pred_* 注入.

从原 learn() 单函数按块拆分, 块内语句顺序逐行保持 (数值逐位等价).
"""

from __future__ import annotations

import torch

from ...modulation import soft_norm_preserve
from ..forward import _rms
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
        a3, a4, a5 = ctx.a3, ctx.a4, ctx.a5
        z4, z3, z5 = net._z4, net._z3, net._z5
        Wb = net.W_35[:a5].data

        # 输入能量调制到 std≈1 (与 W_lm 输入同款); 用 _rms 零向量保护 —
        # 原 std 归一化在自由运行 z4 幅度极小 (rms~0.04) 时放大 25 倍,
        # 1024 维点积超 fp16 上限 → pred_l5 inf (第 77 轮实测 step 1)
        z4_pred_in = _rms(z4)
        z3_pred_in = _rms(z3)
        if free_run:
            global_rpe = torch.tensor(0.0, dtype=torch.float16, device=dev)  # 无字节误差 → rpe=1
        else:
            global_rpe = (sh.lm.eps_total.square().mean().sqrt() * 10.0).clamp(max=1.0)  # 全局误差幅度
        # 第 67 轮内部相对误差门控已移除: 全错态下"正常/异常"无区分度,
        # 参照物本身是错的, 任何门控阈值失效 (第 67 轮实测, 见交接文档).
        # 复读对治移交第 68 轮动态自信门控 (teacher forcing 兜底, forward.py)
        Wp54_a = net.W_pred_54[:a5, :a4]
        Wp43_a = net.W_pred_43[:a4, :a3]
        pred_l5 = z4_pred_in @ Wp54_a.T  # [N,S,a5]
        pred_l4 = z3_pred_in @ Wp43_a.T  # [N,S,a4]
        # 未来折扣目标 (K=8, γ=0.9, 纯张量循环累加)
        # z_future[t] = Σ_k γ^k · z[t+k+1] — 预测未来轨迹的加权期望, 不硬拟合
        # 当前帧. 给 z4/z3 "向未来潜在结构漂移的牵引力": 自回归生成时即使
        # 第一步错, 引力拉着网络回到更可能的未来轨迹 (对治暴露偏差, 非降熵).
        # 边界: 序列末尾 K 位无完整未来 → mask 掉不参与误差/更新.
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
        # 第 84 轮: 前馈链注入路径误差预归一化 — 主 Hebbian 路径 (eps_b = _rms(eps_b))
        # 已归一化, 但注入路径 err5_m/err4_m 裸用 → 递归增益升高时 (墙移除后 rho 达
        # ~49, probe v4 实测) z5 爆发 → local_err_l5 量级暴涨 → bmm 外积 Frobenius
        # 范数超 fp16 上限 (65504) → inf → _rho_ctrl 计算 inf·0 = NaN (W_35/E_l5
        # step~278 实证, probe v2 b3.0_g8). 预归一化 (pre-norm, 非 clamp, CLAUDE.md
        # 合规): 与主路径同款 _rms, 只保方向压量级; rho_ctrl 已把最终更新幅度钳在
        # 3%‖W‖, 归一化零语义改变 (稳定区), 只阻断 inf→NaN 路径.
        err5_m = _rms(local_err_l5[:, mask5])  # [N,T5,a5] 逐位 RMS 归一化
        err4_m = _rms(local_err_l4[:, mask4])  # [N,T4,a4] (供 wp43 更新)
        z3_m = z3[:, mask5]
        dW_pred35 = (err5_m.transpose(-2, -1) @ _rms(z3_m)).mean(dim=0) / max(1, int(mask5.sum()))
        Wb.data += _rho_ctrl(dW_pred35 * ctx.eta * rpe * 1.5, Wb, "inj35", net)
        # 第 83 轮 (G8 v2): free_run + 突触缩放时豁免 soft_norm — 与上方主更新同款
        # (行范数 = 增益自由度归系统, soft_norm 会每步抹平缩放造成的异质; 训练模式保持)
        if not (free_run and net.cfg.wt_syn_scaling):
            soft_norm_preserve(Wb.data)
        if not free_run and not ctx.echo_world_frozen:
            # W_42 注入: local_err_L4 经 W_42 逆映射到 a2 (mask 有效位)
            # (free_run 跳过 — W_42 冻结; 回声相位同冻结 — 乱码流不配做感知结构)
            z4_m = z4[:, mask4]
            local_err_l4_a2 = _rms(err4_m @ net.W_42[:ctx.a2].T)
            dW_pred42 = (local_err_l4_a2.transpose(-2, -1) @ _rms(z4_m)).mean(dim=0) / max(1, int(mask4.sum()))
            net.W_42[:ctx.a2].data += _rho_ctrl(dW_pred42 * ctx.eta * rpe * 1.5, net.W_42[:ctx.a2], "inj42", net)
            soft_norm_preserve(net.W_42[:ctx.a2].data)
        # W_pred 矩阵自更新 (纯外积, 输入调制后 z4/z3 保持一致, 注入强度 1.0)
        # 第 102 轮: 回声相位冻结 — W_pred_* 是世界模型 (层间预测), 乱码输入流
        # 不配塑造它
        if not ctx.echo_world_frozen:
            dWp54 = (err5_m.transpose(-2, -1) @ z4_pred_in[:, mask5]).mean(dim=0)
            dWp54 = _energy_constraint(net, Wp54_a.data, dWp54, err5_m, "_act_ema_wp54")
            dWp54 = dWp54 + _elig_accum(net, "W_pred_54", dWp54) * getattr(net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16))
            Wp54_a.data += _rho_ctrl(dWp54 * ctx.eta, Wp54_a, "wp54", net)
            dWp43 = (err4_m.transpose(-2, -1) @ z3_pred_in[:, mask4]).mean(dim=0)
            dWp43 = _energy_constraint(net, Wp43_a.data, dWp43, err4_m, "_act_ema_wp43")
            dWp43 = dWp43 + _elig_accum(net, "W_pred_43", dWp43) * getattr(net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16))
            Wp43_a.data += _rho_ctrl(dWp43 * ctx.eta, Wp43_a, "wp43", net)
            soft_norm_preserve(Wp54_a.data)
            soft_norm_preserve(Wp43_a.data)
        # 诊断: local_err 相对能量 (红线指标), z5 加性保护分母 (修复二:
        # z5 去中心化后近 0 样本 → 直接除 z5 能量 → inf; scale = std +
        # 0.01·mean_std + 1e-4 加性保护, 非 clamp, 保留信号方向)
        z5_scale = z5.square().mean() + 0.01 * z5.square().mean() + 1e-4
        z4_scale = z4.square().mean() + 0.01 * z4.square().mean() + 1e-4
        net._local_err_l5 = (local_err_l5[:, mask5].square().mean() / z5_scale).detach()
        net._local_err_l4 = (local_err_l4[:, mask4].square().mean() / z4_scale).detach()
        sh.l5_local_err = local_err_l5[:, mask5].square().mean().detach()
