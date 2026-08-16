"""LearningEngine

密集 PPA Hebbian 学习 (零反传, 零误差回路).

机制:
- 前馈权重: 逐层预测误差驱动 (标准 PC 自下而上), L3 加随机增益+门控种子
- 微柱 W_35: 块内 BCM 滑阈 + 样本显著性加权 + 增益/剪切/门控掩码
- W_diff: 增量预测 (dz5 = z5[t]-z5[t-1]), 多尺度软窗 + 4 步时间窗 + 独立 BCM
- 时序 W_t: 共现 Hebbian, 静止帧掩码 + homeostatic 列增益
- 内建能量约束 (第 78 轮): 全部可塑性权重施加真 Oja + 活动依赖遗忘项
  (_energy_constraint, 逐输出单元, 纯局部, 零全局统计)
- 学习率: 恒基准值 (lr_hebbian), 无全局调制 (第 78 轮裁决: 无上帝之手)

全 fp16, 零 .float(), 零 autograd.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ..modulation import soft_norm_preserve
from .forward import _rms

if TYPE_CHECKING:
    from .network import DensePCNet


def _activity_baseline(net: DensePCNet, post: torch.Tensor, ema_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """活动² 慢速局部基线 (第 79 轮抽取, 与第 78 轮能量约束同源): 返回 (p2, excess),
    更新 EMA 缓冲 (0.99/0.01, 首窗延迟初始化: ema ← post², excess 从零起步). 纯局部.

    # ponytail: 修剪 permute/shrink 可塑参数但不重排 _act_ema_*, 0.99 速率数百步自愈;
    # 仅当修剪后异常再现才需在 pruning.py 同步 EMA 缓冲.
    """
    p2 = (post * post).mean(dim=(0, 1))  # [out] 窗内活动²
    ema = getattr(net, ema_name)[: p2.shape[0]]  # 修剪后 active 收缩: 只写头部切片
    if ema_name not in net._act_ema_init:
        ema.copy_(p2)
        net._act_ema_init.add(ema_name)
        excess = torch.zeros_like(p2)
    else:
        ema.mul_(0.99).add_(0.01 * p2)
        excess = torch.relu(p2 - ema)
    return p2, excess


def _energy_constraint(net: DensePCNet, W: torch.Tensor, dW: torch.Tensor, post: torch.Tensor, ema_name: str) -> torch.Tensor:
    """内建能量约束 (第 78 轮): 真 Oja + 活动依赖遗忘项, 逐输出单元, 纯局部.

    post: [N,S,out] 与该点赫布外积同源的输出侧因子; ema: [out] 活动² 慢速基线.
    dW ← dW − (α·post² + β·relu(post²−ema)) ⊙ W;  ema ← 0.99·ema + 0.01·post².
    首窗延迟初始化: ema ← post² (excess 从零起步, 防早期过度抑制).
    α = cfg.oja_alpha, β = cfg.oja_elasticity (零全局统计, 无目标值).
    """
    p2, excess = _activity_baseline(net, post, ema_name)
    coef = net.cfg.oja_alpha * p2 + net.cfg.oja_elasticity * excess
    return dW - coef.unsqueeze(1) * W


def _spectral_radius_guard(W: torch.Tensor, rescale: bool = True, bound: float = 1.5) -> torch.Tensor:
    """递归矩阵谱半径安全约束 (第 79 轮改判, 第 81 轮重定义): 10 次幂迭代估 ρ.

    rescale=True: ρ > bound 时 W ← W·(bound/ρ) — 死亡保险 (发散的物理硬界),
    非临界点之墙. 第 81 轮: bound 从 0.95 移至 1.5 — 正常动力学由能量耗散
    (STP 资源耗尽/分流抑制/Oja) 自然饱和, 守卫只在能量失效时兜底. 无全局统计;
    修剪置换后子矩阵 ρ 越界 → 下一步即被拉回.
    rescale=False: 仅观测 (observer 用). 返回 ρ (0 维张量, 可 .item()).
    # ponytail: 每步 6 个小 randn 分配 (~µs 级), perf 敏感再改持久缓冲
    """
    a = W.shape[0]
    v = torch.randn(a, 1, device=W.device, dtype=W.dtype)
    v = v / (v.norm() + 1e-4)
    for _ in range(10):
        v = W @ v
        v = v / (v.norm() + 1e-4)
    rho = (W @ v).norm()
    if rescale:
        W.mul_(torch.minimum(bound / rho, torch.ones_like(rho)))
    return rho


class LearningEngine:
    """学习引擎: 持 net 引用, 复用 _predict 存的 _z* 状态."""

    def __init__(self, net: DensePCNet):
        self.net = net

    def _closed_loop_input(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """自回归暴露输入: 前 k 位真实锚定 + 后 S-k 位模型自生成.

        前半是真实分布 (锚定, 防初始漂移), 后半是模型自己的输出 (暴露,
        对治复读自锁). 批量并行 rollout, 纯前向零梯度.
        第 69 轮: 纯暴露 + 验证端 rep_backstop (第 60 轮形态). 复读对治
        完全交给内生快慢散度多巴胺 (learning.py D 极性翻转), 无外部干预.
        """
        net = self.net
        k = byte_ids.shape[1] // 2
        n_gen = byte_ids.shape[1] - k
        out = net.forward_engine.continuation(byte_ids[:, :k], n_gen, temperature=0.0, rep_backstop=True)
        # 自生成字节 (第 76 轮表达者范式): 生成段 = 系统的"表达", 供 W_act
        # 闭环自洽学习 (surprise = 世界模型对自生成字节的预测误差)
        net._gen_bytes = out[:, k:].detach()
        return out

    def learn(self, byte_ids: torch.Tensor | None = None, closed_loop: bool = False, free_run: bool = False) -> dict:
        """Hebbian 学习 (零反传, 零误差回路). 不接收 targets.

        Args:
            byte_ids: [N, S] long 输入. free_run=True 时忽略 (恒零输入).
            closed_loop: 自回归暴露训练 — 输入 = 前 k 真实 + 后 S-k 模型自生成
            (k=S//2), targets 始终是真实 byte_ids. 只有生成段 (k:) 参与误差,
            锚定段 (k 之前) 只看不学: 网络从自己的错误输出中学习重新找回
            真实目标, 打破 (输出→z4→输出) 复读自锁.
            free_run: 自由运行 (第 77 轮, 生命第一因) — 外部输入恒零, 活动由
            内部递归 + 三尺度振荡器驱动. 字节域权重全部冻结 (W_04/W_42/W_diff/
            W_lm 家族/W_act), 只更新内部动力学权重 (W_t*/W_35/W_23/W_56/W_bind/
            W_bind_self/W_pred_*/W_state_pred/M_l5/bias), 学习率恒基准值 (第 78 轮:
            无全局调制), 目标: 系统活着 (不 NaN, 不冻结, 不发散).
            自回声 (第 80 轮): byte_ids=None 且 free_run=False — 非监督交互环境,
            输入 = 上一窗 W_act 生成字节 (他者 = 系统自己的生成), 字节域学习解冻
            (同普通训练路径), 无标签无奖励, 目标 = 输入流自身的预测结构.

        Returns:
            stats dict (future_err, 各层误差范数).
        """
        net = self.net
        if free_run:
            inp = None
            N, S = 1, net.cfg.free_run_window
        elif closed_loop:
            inp = self._closed_loop_input(byte_ids)
        elif byte_ids is None:
            # 第 80 轮: 自回声自由运行 — 无外部输入, 输入 = 上一窗 W_act 生成的
            # 字节 (非监督交互环境: 他者 = 系统自己的生成, 无标签无奖励).
            # 生成沿 W_act 动作回路 (与 closed_loop 同管线), 结果作为本窗输入,
            # 激活字节域学习 (W_04/W_42/W_diff/W_lm 家族解冻, 见下方同块开关)
            N, S = 1, net.cfg.free_run_window
            net._gen_bytes = net.forward_engine.continuation(
                torch.zeros(1, 1, dtype=torch.long, device=net._osc_f_cnt.device), S - 1,
                temperature=0.0, rep_backstop=True,
            )[:, 1:]  # 去掉种子字节 (0x00), 输入 = 纯生成流
            inp = net._gen_bytes
        else:
            inp = byte_ids
        _ = net.forward_engine._predict(inp, store_state=True)
        N, S = inp.shape if inp is not None else (1, net.cfg.free_run_window)
        dev = next(net.parameters()).device
        k = S // 2 if closed_loop else 0
        # 第 80 轮: 自回声模式 (free_run=False, byte_ids=None) 激活字节域学习 —
        # 输入是系统自己的生成流 (他者响应), W_04/W_42/W_diff/W_lm 家族随
        # 普通训练路径学习该输入流的结构 (无标签, 目标 = 输入自身)
        echo_loop = (not free_run) and (byte_ids is None)
        if echo_loop:
            byte_ids = inp  # 自回声: 目标 = 自身生成流 (他者响应), 下游同普通训练
        # 生成段掩码 (closed_loop 时只有后半参与误差, 前半锚定只看不学):
        # 对齐 S-1 (t+1 目标), 位置 i 对应目标 byte_ids[:, i+1]
        learn_mask = torch.ones(S - 1, dtype=torch.bool, device=dev)
        if closed_loop:
            learn_mask[: k - 1] = False
        a4, a2, a3, a5, a6 = (net.active_size[k] for k in ("l4", "l2", "l3", "l5", "l6"))
        a_sizes = [a4, a2, a3, a5, a6]

        W_23_a = net.W_23[:a3]
        W_diff_a = net.W_diff[:a4, :a4]

        z0 = net._z0
        z4, z2, z3, z5, z6 = net._z4, net._z2, net._z3, net._z5, net._z6

        # ── 权重去主成分投影 (超量抑制): W -= β·(W@v)⊗v, v = W 的 top1 奇异方向 ──
        # Hebbian 更新把每层 W 行的输入空间分量收敛到 ±w 单一方向 (行间有符号
        # cos≈0 被符号随机掩盖, 但绝对 cos 134 倍于随机 → 投影秩 1, PR_eff 焊死
        # 根因). β=1.0 超量抑制 — 必须压倒 Hebbian 正反馈 (β=0.5 被证被覆盖).
        # 主方向直接从 W 幂迭代 (3 次, 不绕 E 的 |cos| 中介 — E 幂迭代 3 次收敛
        # 不充分, v 含噪声, 超量抑制放大噪声 → E_l5 NaN, 334 步实测复现)
        learn_boost = 2.0 - net._traction_scale.to(torch.float16)

        def _decorr_W(
            W: torch.Tensor,
            E: torch.Tensor,
            coef: float = 1.0,
            max_delta_ratio: float | None = None,
        ) -> torch.Tensor:
            dim = W.shape[0]
            Wn = W / (W.norm(dim=1, keepdim=True) + 1e-3)
            dE = (Wn @ Wn.T).abs()  # 绝对相关 (诊断: 行收敛指标)
            eye_mask = 1.0 - torch.eye(dim, device=dev, dtype=torch.float16)
            E.data.mul_(0.97).add_((dE * eye_mask) * (0.05 * learn_boost))
            # top1 方向: 幂迭代 W^T W (列空间), 3 次. 第 81 轮: 迭代前把 W 缩放
            # 到单位 Frobenius 范数 — 幂迭代数学上尺度不变 (只估方向), 但 fp16
            # 下小矩阵 (范数 <~0.1, 元素进入非正规范围 <6e-5) 在乘加累积中
            # 下溢/FTZ 产生 NaN (W_t3 被 Oja 衰减到范数 0.011 后 410 步实测).
            # Wn 只用于方向; 修正量 c@v.T 仍用原 W (幅度按原尺度, W→0 时修正→0)
            Wn = W / (W.norm() + 1e-8)
            v = torch.randn(W.shape[1], 1, device=dev, dtype=torch.float16) * 0.01
            for _ in range(3):
                v = Wn.T @ (Wn @ v)
                v = v / (v.norm() + 1e-8)
            c = W @ v  # 每行在 top1 方向上的投影系数 (含 ± 符号)
            dW = c @ v.T  # 行 i 减 c_i·v^T (超量抑制, 切断秩 1)
            # 范数信任域 (闭环制动): 单步扰动上限 = max_delta_ratio·‖W‖_F,
            # 方向保持等比缩放 (信任域方法, 开环有界扰动防下游突变)
            if max_delta_ratio is not None:
                dmax = max_delta_ratio * W.norm()
                dn = dW.norm()
                if dn > dmax:
                    dW = dW * (dmax / dn)
            dW = dW * coef
            W -= dW
            return dW

        # 逐层预测误差 (自下而上 PC); L5 用时序差分误差, L6 用时间自预测
        eps4 = z4 - (z0 @ net.W_04[:a4].T + net.bias_l4[:a4])
        eps2 = z2 - (z4 @ net.W_42[:a2].T + net.bias_l2[:a2])
        eps3 = z3 - (z2 @ W_23_a.T + net.bias_l3[:a3])
        z6_pre = torch.cat([torch.zeros(N, 1, a6, dtype=z6.dtype, device=dev), z6[:, :-1]], dim=1)
        eps6 = z6 - z6_pre
        eps5_td = z5[:, 1:] - z5[:, :-1]  # L5: 跨时刻变化 z5[t]-z5[t-1]

        # 下一状态预测 (显式预测目标): pred_delta = z4 @ W_diff, target_delta = z4[t] - z4[t-1]
        # eps_diff = pred_delta - target_delta (训练误差); 多尺度软窗同构保留
        # (free_run: W_diff 冻结, 整块死计算 — 第 78 轮跳过)
        if not free_run:
            dz4 = net._z4[:, 1:] - net._z4[:, :-1]
            dz4 = dz4 / (dz4.norm(dim=-1, keepdim=True) + 1e-3)  # RMS 归一化
            z4r = net._z4  # [N,S,a4]
            S_full = z4r.shape[1]
            masks = {}
            preds_k = {}
            errs_k = {}
            for k, kname in ((2, "2"), (4, "4"), (8, "8")):
                z_shift = torch.cat([torch.zeros(N, k, a4, dtype=z4r.dtype, device=dev), z4r[:, :-k]], dim=1)
                z_shift_n = z_shift / (z_shift.norm(dim=-1, keepdim=True) + 1e-3)
                pred_k = z_shift_n @ W_diff_a.T + net.b_diff[:a4]
                err_k = (dz4 - pred_k[:, :-1]).square().mean()
                if k == 8:
                    valid = torch.arange(S_full - 1, device=dev) >= 7
                    err_k = ((dz4[:, valid] - pred_k[:, :-1][:, valid]).square()).mean()
                masks[kname] = valid if k == 8 else None
                preds_k[kname] = pred_k
                errs_k[kname] = err_k
            # EMA 平滑各尺度误差 (α=0.9, 时间常数~10步)
            e2, e4, e8 = (errs_k["2"], errs_k["4"], errs_k["8"])
            ema2 = net._e_ema_2.mul_(0.1).add_(0.9 * e2)
            ema4 = net._e_ema_4.mul_(0.1).add_(0.9 * e4)
            ema8 = net._e_ema_8.mul_(0.1).add_(0.9 * e8)
            # 软权重: w_k = (1/(e_ema_k+1e-3)) / Σ (加性保护非 clamp)
            inv2, inv4, inv8 = (1.0 / (ema2 + 1e-3)), (1.0 / (ema4 + 1e-3)), (1.0 / (ema8 + 1e-3))
            sum_inv = inv2 + inv4 + inv8
            w2, w4, w8 = inv2 / sum_inv, inv4 / sum_inv, inv8 / sum_inv
            net._w_soft[0] = w2
            net._w_soft[1] = w4
            net._w_soft[2] = w8
            pred_d = w2 * preds_k["2"][:, :-1] + w4 * preds_k["4"][:, :-1] + w8 * preds_k["8"][:, :-1]
            # 误差用掩码后的有效区 (K=8 前 7 步剔除)
            if masks["8"] is not None:
                e_t = (dz4[:, masks["8"]] - pred_d[:, masks["8"]]).detach()
                e_t_all = dz4 - pred_d
            else:
                e_t = dz4 - pred_d
                e_t_all = e_t
            # W_diff BCM 滑阈: theta_w = EMA(pred²), phi_w 衰减高活跃预测方向
            th_w = net._theta_w[:a4]
            th_w.mul_(0.975).add_(0.025 * (pred_d * pred_d).mean(dim=(0, 1)))
            phi_w = pred_d * (pred_d - th_w)
            # 差动赫布外积 (动力学映射): dW = z4_prev^T @ (e - 0.1*phi_w),
            # 输入是上下文 z4_prev 而非差分 dz4 — 学习"当前 L4 状态 → 移动方向"
            z4_prev_n = z4r[:, :-1] / (z4r[:, :-1].norm(dim=-1, keepdim=True) + 1e-3)
            e_mod = e_t_all - 0.1 * phi_w
            dW_diff_t = torch.bmm(e_mod.transpose(-2, -1), z4_prev_n).mean(dim=0) * (1.0 / (S - 1))
            # 4 步时间窗环形缓冲: 更新用最近 4 步平均外积 (保留误差记忆)
            buf = getattr(net, f"_dw_buf_{net._buf_i}")
            buf.copy_(dW_diff_t)
            dW_avg = buf.clone()
            for i in range(1, 4):
                dW_avg = dW_avg + getattr(net, f"_dw_buf_{(net._buf_i - i) % 4}")
            dW_avg = dW_avg * 0.25
            net._buf_i = (net._buf_i + 1) % 4

        # 学习率 (第 78 轮: 无全局调制 — 恒基准值, 无上帝之手); 前 50 步减半 (先稳后放)
        eta = net.cfg.lr_hebbian
        if net._step_counter < 50:
            eta = eta * 0.5
        eta_t = eta * net.cfg.temporal_lr_ratio
        # 动态稳态竞争 (速率自适应): 按 W_lm 熵斜率调节表示层更新幅度 —
        # 熵加速下降 (W_lm 预测好) → scale→0 表示层放慢; 熵停滞/上升 → scale→2
        # 表示层放大 (强迫重组供新信息). scale 每 100 步更新, 用上一步值 (滞后一步无影响)
        if net.cfg.adaptive_traction:
            eta = eta * net._traction_scale.to(torch.float16)
            eta_t = eta_t * net._traction_scale.to(torch.float16)
        # 自由运行偏置稳态更新 (第 77 轮): bias 跟踪各层预测误差均值 —
        # 纯局部积分 (每单元只看自身误差, 自限: bias 增长 → 误差均值降 → 减缓)
        # 第 79 轮改判: 巨 bias 是饱和稳定器 (低增益阻尼递归), 不衰减存量 —
        # 改增量抑制: Δbias ← Δbias / (1.0 + β·excess), β = oja_elasticity.
        # 只限高活动单元的 bias 继续增长, 不削已有存量.
        # fp16 下 1+β·excess 对 excess<~0.01 舍入为 1 (隐式门槛: 仅突发级超基线被抑制).
        if free_run or echo_loop:
            _, ex4 = _activity_baseline(net, z4, "_act_ema_b4")
            _, ex2 = _activity_baseline(net, z2, "_act_ema_b2")
            _, ex3 = _activity_baseline(net, z3, "_act_ema_b3")
            _, ex5 = _activity_baseline(net, z5, "_act_ema_b5")
            _, ex6 = _activity_baseline(net, z6, "_act_ema_b6")
            beta = net.cfg.oja_elasticity
            net.bias_l4[:a4].data += eta * (eps4.mean(dim=(0, 1)) / (1.0 + beta * ex4))
            net.bias_l2[:a2].data += eta * (eps2.mean(dim=(0, 1)) / (1.0 + beta * ex2))
            net.bias_l3[:a3].data += eta * (eps3.mean(dim=(0, 1)) / (1.0 + beta * ex3))
            net.bias_l5[:a5].data += eta * (eps5_td.mean(dim=(0, 1)) / (1.0 + beta * ex5))
            net.bias_l6[:a6].data += eta * (eps6.mean(dim=(0, 1)) / (1.0 + beta * ex6))
            # 第 82 轮: 自由运行 bias 泄漏 — 防止 bias 长成“定点支柱”.
            # 纯局部逐单元衰减, 只限 free_run, 不碰 echo/外部输入模式.
            if free_run:
                net.bias_l4[:a4].data.mul_(1.0 - net.cfg.bias_leak_rate)
                net.bias_l2[:a2].data.mul_(1.0 - net.cfg.bias_leak_rate)
                net.bias_l3[:a3].data.mul_(1.0 - net.cfg.bias_leak_rate)
                net.bias_l5[:a5].data.mul_(1.0 - net.cfg.bias_leak_rate)
                net.bias_l6[:a6].data.mul_(1.0 - net.cfg.bias_leak_rate)

        eta_lm = net.cfg.lm_lr_boost

        # Hebbian 外积 (逐层误差 ⊗ pre 活动)
        fe = net.forward_engine
        eps2_p, eps6_p = (
            fe._precise(eps2),
            fe._precise(eps6),
        )
        if not free_run:
            # ── 预测编码闭环: W_lm 预测误差投影回 z4, 作为表示层 top-down 误差 ──
            # eps_lm_proj = eps_lm @ W_lm.T: 表示层被迫为"预测下一字节"重组编码,
            # 而非只重构当前字节. 纯赫布, 零 BP (大脑皮层最核心的闭环)
            # 多级记忆池 + 角色绑定拼接: [z4, m2, m8, m32, bind] 五通道进 W_lm.
            # 绑定向量 (任务 2) 由 z4 经 W_bind 三槽 top-k 生成, 离散符元承载
            # "主语-动词-宾语" 角色结构; 记忆池承载跨序列环境 (物理输入层不变)
            # W_lm 输入前的能量调制 + 竞争性非线性 (老师方向 C):
            # 1) 能量调制: z4 被 W_04 行范均分压到 std≈0.06 (微缩信号, 上下文信息
            #    被 bias 频率先验淹没 → 命中率 23% 铁板). 分流抑制 x/(1+|x|):
            #    线性归一化 (mean/RMS/median) 与稀疏分布不兼容 — 尖峰被批内统计量
            #    拉小后除以小 D 放大穿 1.4 → NaN. 分流: 半饱和 τ=1.0, 输出渐近界
            #    =1.0 < 1.4 安全线, 处处可微, 导数 1/(1+|x|)² ≤1 自动降权 (Huber 式)
            # 2) RMS 前置 (CLAUDE.md 铁律): 调制后厚尾平方可超 fp16 上限 (实测 65504),
            #    投影前 RMSNorm 结构化防溢出
            # 3) 三阶非线性: f(x)=x·(1-0.5x²) 类 tanh, 在 std≈1 时真正进入非线性区
            z4_lm = z4 / (1.0 + z4.abs())
            z4_lm = _rms(z4_lm)
            z4_lm = z4_lm * (1.0 - 0.5 * z4_lm.pow(2))
            # 输出缩放 1/√H (CLAUDE.md: 投影输出溢出 → 乘 1/√H)
            zh = torch.cat([z4_lm, net._m2, net._m8, net._m32, net._bind_vec], dim=-1)  # [N,S,4a4+16]
            # 第 80 轮: zh 整体 RMS 前置 (与更新侧 _rms(zh) 对称) — z4_lm 段三阶在
            # z4 幅度大时进入放大区 (echo 模式 W_04 解冻, z4 比训练态大 3 倍),
            # 段间量级差异 → zh 尖峰 → h 厚尾 → h_deriv 爆炸 → dW1 fp16 累加溢出
            zh = _rms(zh)

            # ── 非线性混合层 (第 57 轮): zh → W1 → h → 三阶激活 → W_lm → logits ──
            # h = zh @ W1 [d_h=256]; h = h·(1-0.5h²) 多项式激活 (FP16 原生安全);
            # logits = h @ W_lm. W1 横向交叉组合 z4 信息 (非纵向几何缩放),
            # 把高频 e 列打散到不同子空间. 池门控 (旧机制) 随线性读出一并移除
            # 分流抑制 (第 75 轮裁定): h = zh@W1 投影产生 9.4σ 厚尾 (W1 列极化在
            # 训练中生长: 尖峰 → Hebbian 强化 → 更大尖峰 正反馈), 前向路径必须
            # 掐断 — 与 z4_lm 同款 x/(1+|x|), τ=1.0 输出渐近界 1.0 < 1.4 安全线,
            # 处处可微, 尖峰压缩保留 (非 clamp)
            d_h = net.d_h
            W1_a = net.W1  # [lm_in, d_h]
            h = zh @ W1_a  # [N,S,d_h]
            h = h / (1.0 + h.abs())  # 分流抑制 (止血, W1 稳态机制另行讨论)
            # h 前 RMS 归一化 (同 z4 调制模式): zh 4112 维点积 → h 值域 ~±37,
            # 直接三阶激活 f(h)=h·(1-0.5h²) 对 |h|>1.4 进入放大区 → 爆炸 NaN
            # (实测 e_h max 632 → dW1 inf, step 9). RMS 压到 std≈1 进饱和区
            h = _rms(h)
            # 安全监控插桩 (第 75 轮): max|h_in| 距 1.4 放大区余量 (学习器诊断用)
            net._h_in_max = h.abs().max().detach()
            # 三阶激活 f(x)=x·(1-0.5x²) 的输入 x = RMS 后 h (std≈1); 导数 1-1.5x² 必须
            # 用激活输入算 — 用激活输出 f(x) 算导数, |x|>1.4 放大区 |f(x)|>|x| 使导数
            # 无界 (-28000 级), e_h = err@W_lm.T · h_deriv 在 fp16 溢出 inf → dW1 NaN
            h_in = h
            h = h_in * (1.0 - 0.5 * h_in.pow(2))  # 多项式激活, 零 BP
            h_deriv = 1.0 - 1.5 * h_in.pow(2)  # 激活导数 (转置误差传播用, 纯张量)
            inv_h = 1.0 / math.sqrt(d_h)
            logits_lm = (h @ net.W_lm + net.bias_lm) * inv_h  # [N,S,256]
            # ── 读出端能量调制 + 频率去偏 + 可打印掩码 (第 54/55/56 轮) ──
            # 1) 能量调制 (完整标准化): 先中心化 (减均值 — raw mean +0.03 被 ×60
            #    放大成 +3.11 系统性偏移), 再归一化 (除 std); 然后 max_abs 归一化
            #    严格落在 [-60, +60] (fp16 黄金法则: 避免极值溢出 — std 缩放把
            #    raw 尖峰 5.0 放大到 459, softmax 饱和单字节主导, 实测)
            logits_c = (logits_lm - logits_lm.mean(dim=-1, keepdim=True)) / (
                logits_lm.std(dim=-1, keepdim=True) + 1e-4
            )
            logits_lm = logits_c / logits_c.abs().max(dim=-1, keepdim=True).values * 60.0
            # 2) 去偏: logits -= 6·log(freq), 高频字节主场优势被抹平, 低频结构字符
            #    浮出. freq = 模型内部 EMA of target 分布, 加性保护 1e-6
            # 3) 可打印物理掩码: 0x20-0xFF 合法, 0x00-0x1F 强制 -1e4 (fp16 安全极弱值)
            if closed_loop:
                target_oh = F.one_hot(byte_ids[:, 1:], num_classes=256).to(torch.float16).mean(dim=(0, 1))
            else:
                target_oh = F.one_hot(byte_ids, num_classes=256).to(torch.float16).mean(dim=(0, 1))
            net._freq.mul_(0.99).add_(0.01 * target_oh.detach())
            freq_safe = net._freq + 1e-2
            logits_lm = logits_lm - (6.0 * torch.log(freq_safe)).to(torch.float16).unsqueeze(0).unsqueeze(0)
            mask_print = torch.zeros(256, dtype=torch.float16, device=dev)
            mask_print[32:] = 1.0
            logits_lm = logits_lm + (1.0 - mask_print) * -1e4
            # 池间侧抑制 (旧线性读出机制) 随 W1 混合层移除 — 池门控依赖逐段
            # W_lm 行映射, 混合层 h 为折叠空间 (无段映射), 不再适用

            # 多步预测 (Q3 解耦): W_lm 专责 t+1, W_lm_2 独立子预测器专责 t+2.
            # 共享混合特征 h 输入, 各自更新独立 (同一突触不拟合双目标 → 无信号冲突).
            zh2 = torch.cat(
                [z4_lm[:, :-2], net._m2[:, :-2], net._m8[:, :-2], net._m32[:, :-2], net._bind_vec[:, :-2]], dim=-1
            )
            zh2 = _rms(zh2)  # 第 80 轮: 同 zh 整体 RMS 前置 (见上)
            h2 = zh2 @ W1_a
            h2 = h2 / (1.0 + h2.abs())  # 分流抑制 (与 h 同款)
            h2 = _rms(h2)  # 与 h 同款前置 RMS: 4112 维点积后值域 ±37, 直接激活进入放大区 → fp16 溢出
            h2 = h2 * (1.0 - 0.5 * h2.pow(2))
            logits_t2 = (h2 @ net.W_lm_2 + net.bias_lm) * inv_h  # [N,S-2,256] (输出缩放)
            target_lm = F.one_hot(byte_ids[:, 1:], num_classes=256).to(torch.float16)
            target_lm2 = F.one_hot(byte_ids[:, 2:], num_classes=256).to(torch.float16)
            # 赫布版 softmax 误差: eps = target - softmax(logits) (概率尺度 0-1).
            # 原始 target - logits 的负信号被 logits 幅度主导 (熵 5.5 时 logit~0 但非目标位
            # 255 项累积淹没目标位); softmax 后目标位概率 1/256, 误差信号与概率匹配.
            # 全 fp16: logits 已归一化到 [-60,60] 有界, exp 输入有界无溢出
            probs_lm = torch.softmax(logits_lm, dim=-1)  # [N,S,256] fp16
            probs_t2 = torch.softmax(logits_t2, dim=-1)
            eps_lm = (target_lm - probs_lm[:, :-1]).detach()  # [N,S-1,256]
            eps_t2 = (target_lm2 - probs_t2).detach()  # [N,S-2,256] 专供 W_lm_2 更新
            # 任务 1: 多步差分目标 — 差分误差 = (target_{t+2}-target_{t+1}) - (probs_{t+2}-probs_{t+1}),
            # 两步内字节变化的方向/幅度必须匹配 (structure 上"下一步变什么").
            # W_lm 吃 diff2 (S-1 对齐) + t+1 误差; W_lm_2 吃 diff2 + t+2 误差 (同权重,
            # 不引入 BP — 差分目标只是额外的赫布外积误差通道)
            probs_l1 = probs_lm[:, :-1]  # [N,S-1,256] = t+1 概率
            probs_l2 = probs_t2  # [N,S-2,256] = t+2 概率
            target_l1 = target_lm  # [N,S-1,256] = t+1 目标
            target_l2 = target_lm2  # [N,S-2,256] = t+2 目标
            diff2 = (target_l2 - target_l1[:, :-1]) - (probs_l2 - probs_l1[:, :-1])  # [N,S-2,256]
            diff2 = torch.cat([diff2, torch.zeros(N, 1, 256, dtype=diff2.dtype, device=dev)], dim=1)  # S-1 对齐
            if closed_loop:
                lm_mask = learn_mask.unsqueeze(0).unsqueeze(-1)
                eps_lm = eps_lm * lm_mask
                eps_t2 = eps_t2 * learn_mask[1:].unsqueeze(0).unsqueeze(-1)
                diff2 = diff2 * lm_mask
            # 0.2 权重: diff2 能量占比 ~22% (0.5 时 41%, W_lm 更新方向被差分信号主宰,
            # 单步目标被稀释 → 熵慢降、命中率冻结). 差分目标保留为辅助结构信号
            eps_total = (eps_lm + 0.2 * diff2).detach()  # W_lm: t+1 误差 + 差分误差 (S-1 对齐)
            eps_t2_total = (eps_t2 + 0.2 * diff2[:, :-1]).detach()  # W_lm_2: t+2 误差 + 差分误差 (S-2)

            # 动态稳态竞争: 每步记录 batch 级 W_lm 熵 (全 fp16, 零精度依赖:
            # 0·log(0)≡0 信息论定义, torch.where 屏蔽零概率项 — 不用 epsilon
            # 保护常数, fp16 下 1e-9 舍入为 0 → log(0)=-inf → 熵 NaN → 全链崩,
            # 第 76 轮 fp16 整改后 21 步实测; 大脑精度下不可能事件贡献为 0)
            # 连续负反馈: 20 步窗口最小二乘斜率 (线性拟合滤噪, 零超参),
            # scale = 2/(1+exp(-slope20/σ)): 熵降 (slope20<0) → scale→0 表示层放慢
            # 保护成果; 熵停滞/上升 → scale→2 表示层放大强迫重组. 有界无 clamp
            # slope20 = 20 步熵总变化 (nats), σ = 窗口熵波动 (nats), 比值无量纲
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
            # Q1 显著性反馈 + 任务 3 误差剧烈度缩放: 回传误差 × 时间突变范数
            # |z4[t]-z4[t-1]| × 预测误差能量 — 状态剧变且预测失误的时刻, 自上而下
            # 大幅改写表示层; 平稳且预测准的时刻回传弱 (保护已学结构).
            # 投影用全量 W_lm (5 段含绑定) 再取 z4 维: 预测误差经所有输入段权重汇聚到
            # z4 神经元 — 不受池门控排挤影响 (若只投影 z4 段, 池权重增长会压制 z4 段
            # → 表示层收不到预测误差 → 漂移 NaN, 9000 步崩盘根因)
            if getattr(net, "_q1_enabled", True):
                dz4_sig = (z4[:, 1:] - z4[:, :-1]).norm(dim=-1, keepdim=True)  # [N,S-1,1]
                dz4_sig = dz4_sig / (dz4_sig.max() + 1e-3)
                # soft 饱和增益: gain = x/(0.5·mean+0.5·x) ∈ (0,2), 有界无 clamp;
                # 误差 x=均值 → 1, x≫均值 → 2 (大幅改写), x≪均值 → 0 (保护)
                err_mag = eps_total.norm(dim=-1, keepdim=True)  # [N,S-1,1]
                err_ref = err_mag.mean() + 1e-3
                gain = err_mag / (0.5 * err_ref + 0.5 * err_mag)
                # 混合层转置投影: e → W_lm.T → W1.T → z4 段. 投影后每位置范数
                # ~√256×行范, RMS 归一化防 W_04 更新爆 (step 7 NaN 根因)
                eps_lm_proj = (eps_total @ net.W_lm.T @ W1_a.T)[:, :, :a4] * dz4_sig * gain  # [N,S-1,a4]
                eps_lm_proj = _rms(eps_lm_proj)
            else:
                eps_lm_proj = (eps_total @ net.W_lm.T @ W1_a.T)[:, :, :a4]  # 均匀回传
                eps_lm_proj = _rms(eps_lm_proj)
            eps_lm_pad = torch.cat(
                [eps_lm_proj, torch.zeros(N, 1, a4, dtype=eps_lm_proj.dtype, device=dev)], dim=1
            )

        # 突触后增益控制: 归一化基于当前误差自身的 std (统计去耦), 不依赖维度 —
        # 修剪缩小 L4 时误差方差自然变小, 分母自动适应, 无 1/A4 静态系数
        inv_s = 1.0 / S
        if not free_run:
            # ── W_04 主辅误差交换: 预测误差为主, 重建为辅 ──
            # 重建任务不需要词序 (稳定信号拉权重回单一解); 预测误差才需要词序.
            # final_error = err_pred_norm + 0.2 * err_recon_norm (量级对齐)
            # 突触后增益控制 (std 归一化) + 相对地板: 地板 = 全局 std 的 0.1%,
            # 随信号缩放 (修剪缩维 → 全局 std 自动降), 防 fp16 精度极限下除零放大
            eps_pred_scale = eps_lm_pad.std(dim=-1, keepdim=True)
            eps_pred_scale = eps_pred_scale * 1.01 + eps_lm_pad.std() * 1e-3
            err_pred_norm = eps_lm_pad / eps_pred_scale
            eps_recon_scale = eps4.std(dim=-1, keepdim=True)
            eps_recon_scale = eps_recon_scale * 1.01 + eps4.std() * 1e-3
            err_recon_norm = eps4 / eps_recon_scale
            final_error = err_pred_norm + 0.2 * err_recon_norm

            # 幅度-方向解耦 (单步更新上界锁死): dW_n 归一化为单位向量,
            # 显著性只选方向不放大幅度 — 原 g_n×dW_n 等效 e² 平方级放大,
            # 极端样本单步爆幅度 → NaN; 归一化后最大单步幅度 = lr, 纯机制保证
            z0_n = _rms(z0)
            dW_04n = torch.bmm(final_error.transpose(-2, -1), z0_n)  # [N,a4,in]
            nrm_04 = dW_04n.norm(dim=(-2, -1), keepdim=True)
            # 零向量保护: 模型对某 batch 预测完美 (final_error 全零) → dW_04n 全零
            # → 0/0 = NaN (fp16 下 1e-6 被舍入为 0, 装饰性). 掩码: 零范数行分母=1
            alive_04 = (nrm_04 > 1e-8).to(dW_04n.dtype)
            denom_04 = torch.where(alive_04 > 0, nrm_04, torch.ones_like(nrm_04))
            dW_04n = dW_04n * alive_04 / denom_04  # 单位向量 (只保留方向)
            e_norm = final_error.norm(dim=(1, 2))
            # 零向量保护: e_norm 全零 (perfect batch) → max+std=0 → 0/0=NaN.
            # torch.where 掩码 (与 _rms 同款): 全零行分母=1, 分子×0 → 零更新
            e_denom = e_norm.max() + e_norm.std()
            e_alive = (e_denom > 1e-8).to(e_norm.dtype)
            g_04 = e_norm / torch.where(e_alive > 0, e_denom, torch.ones_like(e_denom))
            g_04 = (g_04 * e_alive).unsqueeze(-1).unsqueeze(-1)
            g_sum = g_04.sum() + (1 - e_alive.sum()).to(g_04.dtype)  # 全零 → 分母 1
            dW_04 = (dW_04n * g_04).sum(dim=0) / g_sum
            # W_04 输出端 homeostatic 抑制 (第 75 轮): 打破 σ₁ 奇异值垄断.
            # θ_j 慢速跟踪 mu4_j² (μ=0.02), g_j = 1/(1+θ_j) — 高激活输出列权重
            # 更新被抑制, 能量被迫向 σ₂/σ₃ 扩散. 纯局部 (每列只看自己), 连续无截断
            th_w04 = net._theta_w04[:a4]
            mu4_h = z0 @ net.W_04[:a4].T + net.bias_l4[:a4]  # 与 eps4 定义同源
            th_w04.mul_(0.98).add_(0.02 * (mu4_h * mu4_h).mean(dim=(0, 1)))
            g_homeo = (1.0 / (1.0 + th_w04)).unsqueeze(1)  # [a4,1] 每输出列增益
            net._theta_w04_dist = th_w04  # 诊断: θ_j 分布
            dW_04 = dW_04 * g_homeo
            net.W_04[:a4].data += dW_04 * eta
            soft_norm_preserve(net.W_04[:a4].data)
            # W_04 行去同质化 (与 W_42/W_23 同款, 上游输入侧秩 1 修复):
            # E_04 幂迭代主方向投影抑制, 破 W_04 行收敛 ±w 坍缩.
            # 双层安全: 斜坡渐进 (coef 200 步升到 1) + 范数信任域 (单步扰动 ≤5%‖W‖_F)
            # — 开环有界扰动, 防 z4 分布突变打穿混合层 |x|≤1.4 安全线 (step 9 NaN 根因)
            ramp = min(1.0, net._step_counter / 200.0)
            _decorr_W(net.W_04[:a4].data, net.E_04[:a4, :a4], coef=ramp, max_delta_ratio=0.05)

            dW42 = (eps2_p.transpose(-2, -1) @ _rms(z4)).mean(dim=0) * inv_s
            dW42 = _energy_constraint(net, net.W_42[:a2].data, dW42, eps2_p, "_act_ema_w42")
            col_mask = torch.rand(a2, 1, device=dev) < net.cfg.column_dropout
            net.W_42[:a2].data += (dW42 * (~col_mask).to(torch.float16)) * eta
            # W_42 权重去同质化 (行收敛 ±w → 投影秩 1 根因)
            _decorr_W(net.W_42[:a2].data, net.E_42[:a2, :a2])
        # ── 字节域块结束 (free_run 跳过: W_lm 家族/W_04/W_42, 冻结) ──
        # W_56 保留更新 (L5→L6 内部动力学, 自由运行同样学习)
        dW_56 = (eps6_p.transpose(-2, -1) @ _rms(z5)).mean(dim=0) * inv_s
        dW_56 = _energy_constraint(net, net.W_56[:a6].data, dW_56, eps6_p, "_act_ema_w56")
        col_mask = torch.rand(net.W_56[:a6].shape[0], 1, device=dev) < net.cfg.column_dropout
        net.W_56[:a6].data += (dW_56 * (~col_mask).to(torch.float16)) * eta

        # 预测编码融合: eps_state = (z4 @ W_state_pred) - Δz4, 注入 W_23 表示层更新.
        # 底层不再只接收重构误差, 同时携带"未来往哪走"的预测误差 (纯线性叠加)
        W_sp_a = net.W_state_pred[:a4, :a4]
        dz4_full = net._z4[:, 1:] - net._z4[:, :-1]
        eps_state = (net._z4[:, :-1] @ W_sp_a.T) - dz4_full  # [N,S-1,a4]
        eps_state = _rms(eps_state)
        # a4 → a3 投影 (经 W_42 逆映射, 尺度匹配)
        eps_state_a3 = eps_state @ net.W_42[:a2].T[:, :a3]
        eps3_pc = eps3 + 0.3 * torch.cat(
            [eps_state_a3, torch.zeros(N, 1, a3, dtype=eps_state_a3.dtype, device=dev)], dim=1
        )
        eps3_pc = fe._precise(eps3_pc)

        # L3 种子: W_23 随机增益 + 误差门控 (上游扰动级联到 L5 分散)
        # _gain_l3 是固定 [384, 384] 种子, L3 修剪后行数收缩, 需按当前活性行切片
        gain_l3 = net._gain_l3[:a3, :a3] if a3 < 384 else net._gain_l3[:a3, :]
        dW23 = (eps3_pc.transpose(-2, -1) @ _rms(z2)).mean(dim=0) * inv_s
        dW23 = _energy_constraint(net, net.W_23[:a3].data, dW23, eps3_pc, "_act_ema_w23")
        err3_norm = eps3.pow(2).mean(dim=(0, 1)).sqrt() + 1e-8  # [a3] 每神经元
        gate3 = 0.1 + 0.9 * (err3_norm / err3_norm.max())
        dW23 = dW23 * gain_l3 * gate3.unsqueeze(1)
        c3_mask = torch.rand(a3, 1, device=dev) < net.cfg.column_dropout
        net.W_23[:a3].data += (dW23 * (~c3_mask).to(torch.float16)) * eta
        # 软范数保持 (0.8-1.2): 增益种子幅度差异保留, 权重有界防溢出
        soft_norm_preserve(net.W_23[:a3].data)
        # W_23 权重去同质化 (行收敛 ±w → 投影秩 1 根因)
        _decorr_W(net.W_23[:a3].data, net.E_23[:a3, :a3])

        # ── 第一步: 换轴 — 资格迹+样本竞争 (bmm 逐样本, 显著性加权, 不批平均) ──
        # dW_n = bmm(eps^T, z) [N, d_out, d_in]; gate_n = surprise_n / Σ surprise_n
        # 线性加权 (禁止 exp/非线性), 保留样本差异, 抹平效应消除
        n_sub = 4
        sub = max(1, N // n_sub)
        if not free_run:
            # 预测编码向下平移: W_lm 预测误差 → a3 空间
            # eps_lm_proj 已算 [N,S-1,a4], 补零到 S 对齐完整序列, 经 W_42 逆映射到 a3
            eps_lm_proj_pad = torch.cat(
                [eps_lm_proj, torch.zeros(N, 1, a4, dtype=eps_lm_proj.dtype, device=dev)], dim=1
            )
            eps_lm_a3 = eps_lm_proj_pad @ net.W_42[:a2].T[:, :a3]
            eps_lm_a3 = _rms(eps_lm_a3)
        # ── W_35 统一矩阵更新 (撤销微柱硬切块): 预测编码融合误差全量驱动 ──
        z5_b = z5
        eps_b = z5_b[:, 1:] - z5_b[:, :-1]  # 时序差分误差
        z3_b = z3[:, :-1]
        z3_bp = z3_b * (torch.rand_like(z3_b) > 0.3).to(torch.float16)
        if not free_run:
            # 预测编码融合: eps_b = 时序差分 + 0.5 × LM 预测误差投影回 L5 空间
            eps_lm_b = eps_lm_a3[:, :-1] @ net.W_35[:a5].T  # [N,S-1,a5]
            eps_lm_b = _rms(eps_lm_b)
            eps_b = eps_b + 0.5 * eps_lm_b
        # BCM 滑阈: theta = EMA(eps²), phi = eps(eps-theta); 先 RMS 归一化防 fp16 溢出
        eps_b = _rms(eps_b)
        th = net._theta_l5[:a5]
        e2 = (eps_b * eps_b).mean(dim=(0, 1))
        th.mul_(0.975).add_(0.025 * e2)
        phi_b = eps_b * (eps_b - th)
        # 样本显著性: g_n = surprise_n / Σ surprise_n (线性加权)
        g_n = (phi_b * phi_b).mean(dim=(1, 2)) + 1e-8
        g_n = g_n / g_n.sum()
        Wb = net.W_35[:a5].data
        gain_mask = net._gain_mask[:a5, :a3]
        # 误差门控: 高误差神经元主导更新, 低误差保 10% 下限
        err_norm = eps_b.pow(2).mean(dim=(0, 1)).sqrt() + 1e-8
        upd_gate = 0.1 + 0.9 * (err_norm / err_norm.max())
        # ── W35 资格调制 (第 73 轮): 单变量, 移除 Cos 惩罚 ──
        # q_i = 每行 (微柱) 的时间残差能量 — 时间差分无法解释的行获得更高
        # 学习资格: ΔW_i = q_i·η·ε·z3^T. 让"谁学习"由内部误差决定, 而非
        # 外部正交约束. 验证 Hebbian 能否通过内部塑性竞争产生稳定多方向生态
        z5_prev = torch.cat([torch.zeros(N, 1, a5, dtype=z5.dtype, device=dev), z5[:, :-1]], dim=1)
        z5_prev_n = _rms(z5_prev)
        z5_pred_t = z5_prev_n @ net.W_t5[:a5].T  # W_t5 时间预测
        res_neuron = (z5[:, 1:] - z5_pred_t[:, 1:]).square().mean(dim=(0, 1))  # [a5] 每行残差能量
        q_i = res_neuron / (res_neuron.max() + 1e-6)  # [a5] 资格 ∈ [0,1], 分化
        n_sub = min(4, max(1, N))  # N=1 时防重复子集 (原代码 4× 更新潜伏 bug)
        sub = max(1, N // n_sub)
        inv_sm1 = 1.0 / (S - 1)
        dW_h = None
        for s in range(n_sub):
            sl = slice(s * sub, (s + 1) * sub)
            # 预除 (第 78 轮): phi_b 先 ×1/(S−1) 再 bmm — fp16 中间累加不超界, 数学等价
            dW_n = torch.bmm((phi_b[sl] * inv_sm1).transpose(-2, -1), _rms(z3_bp[sl]))
            dW_sub = (dW_n * g_n[sl, None, None]).sum(dim=0)
            dW_sub = dW_sub * gain_mask * upd_gate.unsqueeze(1)
            dW_sub = dW_sub * q_i.unsqueeze(1)  # 资格调制 (第 73 轮)
            b_mask = torch.rand(a5, 1, device=dev) < net.cfg.column_dropout
            dW_sub = dW_sub * (~b_mask).to(torch.float16)
            dW_h = dW_sub if dW_h is None else dW_h + dW_sub
        dW_h = _energy_constraint(net, Wb, dW_h, phi_b, "_act_ema_w35")
        dW_h = dW_h * eta
        # ── 通道级塑性控制 (第 75 轮): 预测连接时间尺度统一控制律 ──
        # ρ_i = ||ΔW_i||/||W_i||, s_i = clip(0.03/ρ_i, 0.005, 1.0).
        # 冻结 BCM/Hebbian/Cos 竞争: 唯一变量 = 外积/注入时间尺度.
        # 空间机制 (decorr/soft_norm) 不动, 应用顺序保持原样: hebb → soft_norm → decorr.
        nW = Wb.norm() + 1e-8  # decorr 前统一基准

        def _rho_ctrl(dW: torch.Tensor, W_ref: torch.Tensor, tag: str) -> torch.Tensor:
            if not net.cfg.adaptive_rho:
                return dW
            nW_i = W_ref.norm() + 1e-8
            rho = dW.norm() / nW_i  # fp16 范数比 (无量纲, [0,~1e2] 内可表示)
            s_i = (0.03 / (rho + 1e-8)).clamp(0.005, 1.0)
            dW_s = dW * s_i
            net._rho_map[tag] = (rho, dW_s.norm() / nW_i, s_i)
            return dW_s

        if net.cfg.adaptive_rho:
            net._rho_map = {}
        Wb.add_(_rho_ctrl(dW_h, Wb, "hebb"))
        # 第 83 轮 (G8 v2): 前馈链突触缩放 — L5 神经元入纤增益随自身活动双向
        # 调节, 与 W_t 同款机制. 实证 (probe v4): 高墙下递归链被双向缩放+STP
        # 兜住 (rho 49, sat 6.9%), 但 W_35 更新路径无局部饱和 → fp16 溢出 NaN
        # (W_35/E_l5, step ~100). 此处把 G8 扩展到前馈链, 并豁免 soft_norm
        # (与 G1 对 W_t 的处理同构: 行范数 = 增益自由度归系统, soft_norm 是
        # 增益锁, 会每步抹平缩放造成的异质). 训练模式保持 soft_norm.
        if free_run and net.cfg.wt_syn_scaling:
            p2w5 = (z5 * z5).mean(dim=(0, 1))  # [a5] L5 窗内活动²
            emaw5 = net._act_ema_b5[:a5]  # L5 慢基线 (bias 泄漏段已更新)
            scale5 = 2.0 * emaw5 / (p2w5 + emaw5 + 1e-3)  # 双向突触缩放 (0,2]
            r835 = net.cfg.wt_syn_scaling_rate
            Wb.data.mul_((1.0 - r835) + r835 * scale5.unsqueeze(1))
        else:
            soft_norm_preserve(Wb)
        dW_corr = _decorr_W(Wb, net.E_l5[:a5, :a5])  # 空间机制 (原顺序: 最后), 返回修正量
        if net.cfg.adaptive_rho:
            rho_raw, rho_eff, s_h = net._rho_map["hebb"]
            cos_hc = (dW_h.flatten() @ dW_corr.flatten()) / (
                dW_h.norm() * dW_corr.norm() + 1e-8
            )
            net._rho_raw, net._rho_eff, net._s_h = rho_raw, rho_eff, s_h
            net._rho_corr = dW_corr.norm() / nW
            net._cos_hc = cos_hc

        # ── 自组织预测引擎 (第 50 轮): 层间局部误差 + 多巴胺 RPE 门控 ──
        # 折扣多步预测目标 (未来引力, 老师最新方案): 目标 = 未来 K 步折扣和
        # z_future[t] = Σ_k γ^k · z[t+k+1] — 预测未来轨迹的加权期望, 不硬拟合
        # 当前帧. 给 z4/z3 "向未来潜在结构漂移的牵引力": 自回归生成时即使
        # 第一步错, 引力拉着网络回到更可能的未来轨迹 (对治暴露偏差, 非降熵).
        # 边界: 序列末尾 K 位无完整未来 → mask 掉不参与误差/更新.
        # 输入能量调制: z4 std 0.06 未调制 → pred 比目标小 3500 倍, 平凡零预测.
        # 调制到 std≈1 (与 W_lm 输入同款) 后误差真实可学.
        K_FUT = 8
        GAMMA = 0.9
        # 输入能量调制到 std≈1 (与 W_lm 输入同款); 用 _rms 零向量保护 —
        # 原 std 归一化在自由运行 z4 幅度极小 (rms~0.04) 时放大 25 倍,
        # 1024 维点积超 fp16 上限 → pred_l5 inf (第 77 轮实测 step 1)
        z4_pred_in = _rms(z4)
        z3_pred_in = _rms(z3)
        if free_run:
            global_rpe = torch.tensor(0.0, dtype=torch.float16, device=dev)  # 无字节误差 → rpe=1
        else:
            global_rpe = (eps_total.square().mean().sqrt() * 10.0).clamp(max=1.0)  # 全局误差幅度
        # 第 67 轮内部相对误差门控已移除: 全错态下"正常/异常"无区分度,
        # 参照物本身是错的, 任何门控阈值失效 (第 67 轮实测, 见交接文档).
        # 复读对治移交第 68 轮动态自信门控 (teacher forcing 兜底, forward.py)
        Wp54_a = net.W_pred_54[:a5, :a4]
        Wp43_a = net.W_pred_43[:a4, :a3]
        pred_l5 = z4_pred_in @ Wp54_a.T  # [N,S,a5]
        pred_l4 = z3_pred_in @ Wp43_a.T  # [N,S,a4]
        # 未来折扣目标 (K=8, γ=0.9, 纯张量循环累加)
        z5_fut = torch.zeros_like(z5)
        z4_fut = torch.zeros_like(z4)
        mask5 = torch.zeros(S, dtype=torch.bool, device=dev)
        mask4 = torch.zeros(S, dtype=torch.bool, device=dev)
        g = 1.0
        for k in range(K_FUT):
            if k + 1 < S:
                z5_fut[:, : -k - 1] += g * z5[:, k + 1 :]
                z4_fut[:, : -k - 1] += g * z4[:, k + 1 :]
                mask5[: -k - 1] = True
                mask4[: -k - 1] = True
            g *= GAMMA
        local_err_l5 = z5_fut - pred_l5  # 只在 mask5 有效位有意义
        local_err_l4 = z4_fut - pred_l4
        rpe = (1.0 + global_rpe).to(torch.float16)
        # W_35 注入: dW = local_err_L5.T @ z3 (mask 有效位, 与 W_35 输入同源).
        # 注入强度 1.5×eta×RPE (老师指示维持: 局部误差与主误差同量级)
        err5_m = local_err_l5[:, mask5]  # [N,T5,a5]
        z3_m = z3[:, mask5]
        dW_pred35 = (err5_m.transpose(-2, -1) @ _rms(z3_m)).mean(dim=0) / max(1, int(mask5.sum()))
        Wb.data += _rho_ctrl(dW_pred35 * eta * rpe * 1.5, Wb, "inj35")
        # 第 83 轮 (G8 v2): free_run + 突触缩放时豁免 soft_norm — 与上方主更新同款
        # (行范数 = 增益自由度归系统, soft_norm 会每步抹平缩放造成的异质; 训练模式保持)
        if not (free_run and net.cfg.wt_syn_scaling):
            soft_norm_preserve(Wb.data)
        if not free_run:
            # W_42 注入: local_err_L4 经 W_42 逆映射到 a2 (mask 有效位)
            # (free_run 跳过 — W_42 冻结)
            err4_m = local_err_l4[:, mask4]  # [N,T4,a4]
            z4_m = z4[:, mask4]
            local_err_l4_a2 = _rms(err4_m @ net.W_42[:a2].T)
            dW_pred42 = (local_err_l4_a2.transpose(-2, -1) @ _rms(z4_m)).mean(dim=0) / max(1, int(mask4.sum()))
            net.W_42[:a2].data += _rho_ctrl(dW_pred42 * eta * rpe * 1.5, net.W_42[:a2], "inj42")
            soft_norm_preserve(net.W_42[:a2].data)
        # W_pred 矩阵自更新 (纯外积, 输入调制后 z4/z3 保持一致, 注入强度 1.0)
        err4_m = local_err_l4[:, mask4]  # [N,T4,a4] (读取, 供 wp43 更新)
        dWp54 = (err5_m.transpose(-2, -1) @ z4_pred_in[:, mask5]).mean(dim=0)
        dWp54 = _energy_constraint(net, Wp54_a.data, dWp54, err5_m, "_act_ema_wp54")
        Wp54_a.data += _rho_ctrl(dWp54 * eta, Wp54_a, "wp54")
        dWp43 = (err4_m.transpose(-2, -1) @ z3_pred_in[:, mask4]).mean(dim=0)
        dWp43 = _energy_constraint(net, Wp43_a.data, dWp43, err4_m, "_act_ema_wp43")
        Wp43_a.data += _rho_ctrl(dWp43 * eta, Wp43_a, "wp43")
        soft_norm_preserve(Wp54_a.data)
        soft_norm_preserve(Wp43_a.data)
        # 诊断: local_err 相对能量 (红线指标), z5 加性保护分母 (修复二:
        # z5 去中心化后近 0 样本 → 直接除 z5 能量 → inf; scale = std +
        # 0.01·mean_std + 1e-4 加性保护, 非 clamp, 保留信号方向)
        z5_scale = z5.square().mean() + 0.01 * z5.square().mean() + 1e-4
        z4_scale = z4.square().mean() + 0.01 * z4.square().mean() + 1e-4
        net._local_err_l5 = (local_err_l5[:, mask5].square().mean() / z5_scale).detach()
        net._local_err_l4 = (local_err_l4[:, mask4].square().mean() / z4_scale).detach()

        # Foldiak 反赫布侧抑制更新 (方案 D): dM = z_out 协方差 (白化本质),
        # 零对角, 指数遗忘 ×0.99 防爆炸. 不做 Frobenius 归一化 — 归一化把 dM
        # 缩到 ~1e-4 (1024² 矩阵范数 ~1000), ×0.01 → ~1e-6 被 fp16 舍入,
        # 装饰性失效 (第 8 轮同款 bug 翻版, 实测 M_offdiag 0.0004 纹丝不动);
        # z_out 已逐行 RMS 归一化, 协方差元素 ∈[-1,1], 增量直接可表示
        z5_flat = z5.reshape(-1, a5).to(torch.float16)
        z5_flat = z5_flat / (z5_flat.norm(dim=-1, keepdim=True) + 1e-3)
        cov = z5_flat.transpose(0, 1) @ z5_flat / z5_flat.shape[0]
        eye_mask = 1.0 - torch.eye(a5, device=dev, dtype=torch.float16)
        net.M_l5[:a5, :a5].data.mul_(0.99).add_((cov * eye_mask) * (0.01 * learn_boost))

        if not free_run:
            # W_diff 下一状态预测更新 (4 步时间窗平均外积) + b_diff 偏置 (L4 空间)
            # (free_run 跳过 — W_diff 冻结, 依赖外部字节序列)
            fut_mask = torch.rand(a4, 1, device=dev) < net.cfg.column_dropout
            W_diff_a.data += (dW_avg * (~fut_mask).to(torch.float16)) * eta
            future_e = (dz4 - pred_d).mean(dim=(0, 1))
            net.b_diff[:a4].data += future_e * eta

        # 时序 Hebbian (W_t 学习, 高确定性时增强 → 记忆巩固)
        # 静止帧掩码 (频率锚点, 第 74 轮): 静止帧 (Δz 范数低于绝对阈值) 不参与
        # 更新 — W_t 语义 = 转移矩阵. 第 81 轮: free_run 豁免 — 系统停在
        # bias 定点的有序相时静止帧占比 ~98% (实测), 掩码把 Hebbian 增长力
        # 清零只剩 Oja 衰减单向拖向零 (410 步 W_t3 死亡) — 冻结态自锁循环.
        # 秩 1 坍塌的防护已由 decorr (第 70-74 轮机制) 承担, 掩码冗余.
        # 训练模式 (有外部输入) 保持掩码
        for (z_cur, wt_name, W_t), a_sz, E_t in zip(
            [(z4, "wt4", net.W_t4), (z2, "wt2", net.W_t2), (z3, "wt3", net.W_t3), (z5, "wt5", net.W_t5), (z6, "wt6", net.W_t6)],
            a_sizes,
            [net.E_t4, net.E_t2, net.E_t3, net.E_t5, None],
        ):
            pre = z_cur[:, :-1]
            post = z_cur[:, 1:]
            dz = post - pre  # Δz = z_t - z_{t-1}
            dz_n = dz.norm(dim=-1)  # [N, S-1]
            if free_run:
                s = torch.ones_like(dz_n)  # 第 81 轮: 无静止掩码 (见上)
            else:
                # 绝对阈值 = z 平均范数的 2% (闭环训练起放宽: 5%→2% 给长程记忆喘息,
                # 时序动力学稍作释放; 监视 W_t top1 秩防坍缩): 分位阈值在 z 塌缩后
                # 失效 (塌缩后"动态帧"也在主方向, 贡献仍秩 1); 绝对阈值让塌缩 →
                # 差分变小 → 静止占比暴增 → W_t 学习信号枯竭 → 自然平衡点 (频率锚点)
                th = (post.norm(dim=-1).mean() * 0.02).unsqueeze(0)
                s = (dz_n < th).to(dz_n.dtype)  # 静止帧 → 0, 动态帧 → 1
            dW_t = (_rms(pre).transpose(-2, -1) @ (_rms(post) * s.unsqueeze(-1))).mean(dim=0) * (
                1.0 / (S - 1)
            )
            # 内建能量约束 (第 78 轮): Oja + 活动依赖遗忘. 第 81 轮: free_run 豁免 —
            # 第 78 轮 Oja 是为"增益锁死"时代设计的; 增益解放后 Oja 成死亡吸引子
            # (bias 维持高活动 → Oja 力恒定 ~2.5%/步, Hebbian 力随 W_t 收缩 →
            # W_t 渐进归零, 实测 15k 步 12.4→1.0, 递归动力学死亡). 递归增益的
            # 物理耗散 = STP 资源耗尽 + 分流抑制 + 谱守卫 1.5 死亡保险 (三者均为
            # "偏离越大回拉越强"的力), Oja 冗余且致命. 训练模式保持约束
            if not free_run:
                dW_t = _energy_constraint(net, W_t[:a_sz, :a_sz].data, dW_t, z_cur, f"_act_ema_{wt_name}")
            # W_t4 输出端 homeostatic 抑制 (第 75 轮): 打破 σ₁/σ₂≈4000:1 垄断.
            # θ_j 跟踪 rec4 每维能量 (时序差分), g_j = 1/(1+θ_j) — 高激活列更新
            # 被压, rec4 从压制源转丰富源. 与 W_04 同构, 纯局部
            if W_t is net.W_t4:
                th_wt4 = net._theta_wt4[:a_sz]
                th_wt4.mul_(0.98).add_(0.02 * (dz * dz).mean(dim=(0, 1)))
                dW_t = dW_t * (1.0 / (1.0 + th_wt4)).unsqueeze(1)
            W_t[:a_sz, :a_sz].data += dW_t * eta_t
            # 第 83 轮 (G8): 双向突触缩放 — 权重级慢稳态增益, 仅自由运行.
            # 问题 (model81 实测): 增益自由度解开后, W_t 谱半径要么被 decorr 拖向
            # 0 (有序相冻结), 要么被 Hebbian 推到 1.5 守卫墙 (守卫成为增益设定者,
            # 公理 1 违规). 缺的是"系统自己设定操作点"的慢稳态机制.
            # 机制 (双向, Turrigiano 突触缩放): 神经元 i 的入纤递归增益 (W_t 行 i)
            # 随"自身活动 vs 自身慢基线"双向调节 — 高活下调 (遏制爆炸), 低活上调
            # (防止冻结/死亡). scale_i = 2·ema_i/(p2_i+ema_i): 基线=1, p2→∞→0,
            # p2→0→2, 天然有界 (0,2], 无 clamp. 与 STP U_eff (act/(act+ema)) 同族.
            # 以 wt_syn_scaling_rate (慢) 混合进权重: W_t[i,:] ← W_t[i,:]·((1−r)+r·scale_i)
            # 平衡点 = 动力学自洽不动点 (活动回到基线 = scale 1 = 不干预).
            # 纯局部, 自参照, 零新常数. 与 W_t4 θ 抑制同族, 推广到全部 5 层.
            # 与 U 自适应 (快) 分工: U = 单步释放概率负反馈, 突触缩放 = 权重级慢稳态.
            if free_run and net.cfg.wt_syn_scaling:
                p2w = (z_cur * z_cur).mean(dim=(0, 1))  # [a_sz] 窗内活动²
                emaw = getattr(net, f"_act_ema_b{wt_name[2]}")[:a_sz]  # 同层慢基线 (bias 泄漏段已更新)
                scale = 2.0 * emaw / (p2w + emaw + 1e-3)  # 双向突触缩放 (0,2]
                r83 = net.cfg.wt_syn_scaling_rate
                W_t[:a_sz, :a_sz].data.mul_((1.0 - r83) + r83 * scale.unsqueeze(1))
            # 软范数保持 (0.8-1.2): 权重有界防 fp16 溢出 (5万步 NaN 根因).
            # 第 81 轮: free_run 豁免 — 递归增益自由度归系统自己 (W_t 行范数
            # = 递归幅度, 是 E-I 平衡的物理变量). 平衡力已在 dW_t 内 (内建能量
            # 约束 = Oja + 活动依赖遗忘, 纯局部), 发散由 STP 资源耗尽/分流抑制/
            # 谱守卫 1.5 死亡保险兜底; 训练模式 (有外部输入) 保持软范数
            if not free_run:
                soft_norm_preserve(W_t[:a_sz, :a_sz].data)
            # 超量 E 清除 W_t 既有秩 1 结构 (top1 sv 11.4 固化 → 递归拉 z 秩 1)
            # 第 81 轮: free_run coef 降至 0.2 (M_l5 Foldiak 侧抑制的既有系数, 非
            # 新数值) — 1.0 超量抑制在无静止掩码下每步擦除主方向, 与 Oja 同向
            # 形成"修正压倒增长"的死亡合力 (15k 步实测 W_t 家族 12.4→1.0 渐进死亡)
            if E_t is not None:
                _decorr_W(W_t[:a_sz, :a_sz].data, E_t[:a_sz, :a_sz], coef=0.2 if free_run else 1.0)

        # 递归矩阵谱半径安全约束 (第 79 轮改判, 第 81 轮重定义): 死亡保险 ρ>bound
        # 才动作 — 正常动力学由能量耗散自然饱和 (递归增益自由度, 生命第一因).
        # 第 83 轮: bound 配置化 (spectral_guard_bound) — 实证 (exp83/probe) 表明
        # 1.5 嵌在自然增长区间内会每步钳制 → 守卫成控制器; bound 应高于自然
        # 工作区间, 区间内增益由突触缩放 (G8) + STP 耗尽设定.
        # 仅在修剪置换后子矩阵 ρ 发散或能量失效时兜底, 不挡临界点之下的游走
        for wt in (net.W_t4, net.W_t2, net.W_t3, net.W_t5, net.W_t6):
            _spectral_radius_guard(wt.data, bound=net.cfg.spectral_guard_bound)

        if not free_run:
            # LM 头自监督赫布 (复用闭环段已算的 logits_lm/eps_total):
            # 突触前增益控制: error 先 RMS 缩放到单位能量再外积 — 更新幅度完全由
            # 内部误差能量自适应决定, 不依赖外部 eta 缩放 (替代 W_04 解码死锁)
            # 指数遗忘 (0.999/步, 纯乘法): 单位能量外积在目标附近振荡 (无阻尼 LMS),
            # 遗忘项提供阻尼让权重收敛而非翻转
            err_norm = eps_total.norm(dim=-1, keepdim=True) * 1.01
            alive = (err_norm > 1e-8).to(eps_total.dtype)
            denom = torch.where(alive > 0, err_norm, torch.ones_like(err_norm))
            err_scaled = eps_total * alive / denom  # 单位能量, 方向保留

            # W_lm 专属 BCM 滑阈 (防输出过冲): theta = EMA(logits²) (快 0.99 响应),
            # phi_wlm = logits_n·(logits_n - theta) — W_lm 开始输出高频极值时 theta 快速
            # 升高 → logits_n-theta<0 → phi 变负 → dW 修正反向 → 抑制过冲.
            # 纯机制剪刀, 线性, 零 BP; 0.1 系数镜像 W_diff BCM (同模式)
            # logits 先 RMS 归一化再进 BCM: 原始 logits ~52, 平方超 fp16 上限 (W_diff 同款)
            logits_n = _rms(logits_lm.detach())
            th_wlm = net._theta_wlm
            th_wlm.mul_(0.01).add_(0.99 * (logits_n * logits_n).mean(dim=(0, 1)))
            phi_wlm = logits_n * (logits_n - th_wlm)
            phi_wlm = _rms(phi_wlm)
            # ── 结构对比度惩罚 (第 58 轮, 训练"区分力"非"生成力") ──
            # 核心: 强制 z4/h 空间拉大不同字符的距离. 正确列 logits 应显著高于
            # 错误列. 实现: 对误差做几何加权 — 目标位 (one-hot) 误差权重 ×1,
            # 非目标位按"与目标列的区分度"加权: 区分度低 (logits 接近目标) 的
            # 错误列被放大惩罚 (空间排斥), 区分度高 (已被压远) 的列权重衰减.
            # 纯数学空间排斥, 不改变 W_lm 更新公式结构 (仍是 h^T @ err 外积)
            target_oh = F.one_hot(byte_ids[:, 1:], num_classes=256).to(torch.float16)  # [N,S-1,256]
            logits_d = (h[:, :-1] @ net.W_lm) * inv_h  # 未去偏 logits (对比度基准)
            # 对比度: 目标列 logits vs 其他列 — 目标列 logits 高则区分好
            tgt_logits = (logits_d * target_oh).sum(dim=-1, keepdim=True)  # [N,S-1,1] 目标列值
            # 惩罚权重: 非目标列中 logits 接近目标的列 (区分差) 放大, 远离的衰减
            contrast = (logits_d - tgt_logits).abs()  # 与目标列的距离
            contrast_w = 1.0 / (1.0 + contrast * 0.1)  # 距离近 → 权重大 (排斥), 距离远 → 小
            contrast_w = contrast_w * (1.0 - target_oh) + target_oh  # 目标位权重保持 1
            err_contrast = err_scaled * contrast_w  # 对比度加权误差 (空间排斥)
            # ── 快慢散度学习窗口 (第 70 轮): 逐帧新奇度 N[t] = ‖Z_fast[t]-Z_slow[t]‖²
            # (前向已算 [N,S]). τ = EMA(N) 自适应 (内部参照物, 零外部统计).
            # 生成头不承担探索压力: 极性翻转 (LTD 反向重写) 会让 W_lm 因一次异常
            # 生成被重解释 (第 69 轮实测: 熵 0.078→4.57, 出口被推入错误吸引子).
            # 改为学习窗口缩放: η = sigmoid(N - τ) ∈ (0,1) — 只调"允许多少变化":
            # 低新奇度 (死循环, N → 0 < τ) → η → 0 关闭输出学习窗口 (保护已形成
            # 的输出映射); 正常推进 (N > τ) → η → 1 正常学习. 不翻转方向.
            if hasattr(net, "_novelty"):
                nov = net._novelty  # [N,S] 逐帧新奇度
                tau = net._theta_novelty
                tau.mul_(0.99).add_(0.01 * nov.mean())
                d_t = torch.sigmoid((nov - tau) * 500.0).unsqueeze(-1)  # [N,S,1] η ∈ (0,1)
                d_t = d_t[:, :-1]  # 对齐 S-1 (t+1 目标)
            else:
                d_t = torch.ones(N, S - 1, 1, dtype=torch.float16, device=dev)
            # 第 62 轮周期惩罚已按最终裁定移除 (训练期干预误伤 nn 词干, 见交接文档).
            # 第 70 轮: 学习窗口 η (sigmoid 有界, 零极性翻转 — 只调变化量, 不重写方向)
            lm_update_mask = learn_mask.to(torch.float16).unsqueeze(0).unsqueeze(-1)  # [1,S-1,1]
            dW_lm = (
                _rms(h[:, :-1]).transpose(-2, -1)
                @ ((err_contrast - 0.1 * phi_wlm[:, :-1]) * lm_update_mask * d_t)
            ).mean(dim=0) * math.sqrt(d_h)  # [d_h,256] (补偿输出缩放)
            # 单步更新幅度上界 (W_04 同款幅度-方向解耦): 防极端 batch 单步爆
            dW_lm_n = dW_lm.norm() + 1e-8
            dW_lm = dW_lm / dW_lm_n
            net.W_lm.data.mul_(0.999)
            net.W_lm.data += dW_lm * eta_lm
            # bias 硬复位: 范数锁定 (target=100, 除法缩放非 clamp) — 切断自由生长,
            # 死守 bias_std≈6.26. 去均值只学相对偏置.
            # 误差用原始概率误差 (不经单位能量归一化): 混合层下去偏项 ±45 让
            # softmax 尖锐, 单位能量归一化把稀疏误差放大 10 倍 (bias_d 0.03→0.58,
            # 单步增量 5.8 → 10 步爆, 实测 step 7)
            bias_err = eps_lm  # [N,S-1,256] 原始 target - probs, 无归一化
            bias_d = bias_err.mean(dim=(0, 1))
            net.bias_lm.data += (bias_d - bias_d.mean()) * eta_lm
            bn = net.bias_lm.norm()
            target_norm = 100.0
            if bn > target_norm:
                net.bias_lm.data.mul_(target_norm / bn)
            soft_norm_preserve(net.W_lm.data)

            # ── W1 混合层更新 (第 57 轮核心: 转置误差传播, 纯赫布零 BP) ──
            # e_h = e @ W_lm.T · (1 - 1.5·h²): 读出误差经 W_lm 转置投影回混合空间,
            # 乘激活导数 (多项式激活 f=h-0.5h³ 的导数 1-1.5h²) — 告诉 W1 如何把
            # 高频 e 的权重通过组合打散分配到 n/u 的特征上. 真实神经网络折叠,
            # 非几何缩放. lr1 = lr2 = eta_lm (自然竞争)
            e_h = (err_scaled @ net.W_lm.T) * h_deriv[:, :-1]  # [N,S-1,d_h] (err_scaled 已是 S-1)
            dW1 = (_rms(zh[:, :-1]).transpose(-2, -1) @ _rms(e_h)).mean(dim=0) * math.sqrt(d_h)
            dW1_n = dW1.norm() + 1e-8
            dW1 = dW1 / dW1_n
            W1_a.data.mul_(0.999)
            W1_a.data += dW1 * eta_lm
            soft_norm_preserve(W1_a.data)

            # W_lm_2 独立更新 (Q3 解耦): 吃 t+2 误差 + 差分误差 (eps_t2_total), 与 W_lm 完全独立
            # 同款机制: 零向量保护 + 单位能量 + BCM 防抖 + 指数遗忘
            err_norm2 = eps_t2_total.norm(dim=-1, keepdim=True) * 1.01
            alive2 = (err_norm2 > 1e-8).to(eps_t2_total.dtype)
            denom2 = torch.where(alive2 > 0, err_norm2, torch.ones_like(err_norm2))
            err_scaled2 = eps_t2_total * alive2 / denom2
            logits_n2 = _rms(logits_t2.detach())
            th_wlm2 = net._theta_wlm2
            th_wlm2.mul_(0.01).add_(0.99 * (logits_n2 * logits_n2).mean(dim=(0, 1)))
            phi_wlm2 = logits_n2 * (logits_n2 - th_wlm2)
            phi_wlm2 = _rms(phi_wlm2)
            dW_lm2 = (
                _rms(h2).transpose(-2, -1)
                @ (
                    (err_scaled2 - 0.1 * phi_wlm2)
                    * learn_mask[1:].to(torch.float16).unsqueeze(0).unsqueeze(-1)
                    * d_t[:, :-1]
                )
            ).mean(dim=0) * math.sqrt(d_h)
            dW_lm2_n = dW_lm2.norm() + 1e-8
            dW_lm2 = dW_lm2 / dW_lm2_n  # 单步更新幅度上界 (防突爆)
            net.W_lm_2.data.mul_(0.999)
            net.W_lm_2.data += dW_lm2 * eta_lm
            soft_norm_preserve(net.W_lm_2.data)
            # 每步清除新奇度 (前向已更新, 防陈旧信号跨步复用)
            if hasattr(net, "_novelty"):
                del net._novelty

        # 状态预测矩阵自更新 (纯赫布): dW_sp = z4^T @ eps_state, 零 BP
        dW_sp = (net._z4[:, :-1].transpose(-2, -1) @ eps_state).mean(dim=0)
        dW_sp = _energy_constraint(net, W_sp_a.data, dW_sp, net._z4, "_act_ema_wsp")
        W_sp_a.data += dW_sp * eta
        soft_norm_preserve(W_sp_a.data)

        # ── 竞争性概念绑定层赫布更新 (任务 4, 纯外积, 零 BP): ──
        # dW_bind = z4_pre^T @ (z_bind - mean(z_bind)) — 槽位激活去均值 (Oja 式):
        # 高激活槽位强化 z4→槽映射, 低激活槽位削弱, 竞争分化; 零均值防单槽垄断.
        # W_bind 行范数保持 (0.8-1.2) 防坍缩, 软竞争 (L2 归一化) 无死亡
        if hasattr(net, "_bind_vec"):
            z4n = _rms(z4)
            bind_t = net._bind_vec  # 纯 z_bind (第 75 轮: 去均值对称抹差异 → 无偏置积累,
            # 高激活槽保留幅度差 → Hebbian 正反馈放大 → 分化种子; 分流抑制防垄断)
            # 独立逐样本三因子 Hebbian (第 75 轮最终): 每样本独立 surprise EMA,
            # gain_n = clip(s_n/ema_n, 0.3, 5.0) — 样本自身历史决定自身门控,
            # 高惊喜样本外积单独放大 → 非对称注入. 纯局部 (每样本只看自己)
            # (free_run: 无字节误差 eps_total → 门控退化为 1, 样本竞争由 0.3-5.0
            # clamp 下限自然覆盖; N=1 单样本时均一化无意义)
            # (第 80 轮: 自回声有真实字节误差 eps_total, 走 else 分支正常门控)
            if free_run:
                gain_n = torch.ones(N, dtype=torch.float16, device=dev)
            else:
                s_n = (eps_total.square().mean(dim=-1).mean(dim=-1))  # [N] 每样本均方误差
                # _s_ema_n 缓冲按 batch 自适应扩容 (固定 8 与 CLI batch=48 不匹配
                # → 形状错误; 扩容后新样本 EMA=1.0 起, 语义不变)
                if net._s_ema_n.shape[0] < N:
                    old = net._s_ema_n
                    net.register_buffer("_s_ema_n", torch.cat([old, torch.ones(N - old.shape[0], dtype=torch.float16, device=dev)]))
                    del old
                ema_n = net._s_ema_n[:N]
                rel_n = s_n / (ema_n + s_n + 1e-6)  # 自归一化相对惊喜 (量级无关, O(1))
                gain_n = (rel_n * 5.0).clamp(0.3, 5.0)
                net._s_ema_n[:N].mul_(0.95).add_(0.05 * s_n)
            net._gain_n = gain_n  # 诊断: 逐样本增益分布
            z4n_w = z4n[:, :-1] * gain_n[:, None, None]  # [N, S-1, a4]
            dW_bind = torch.einsum("nsd,nsq->dq", z4n_w, bind_t[:, :-1]) / (gain_n.sum() + 1e-8) * (1.0 / (S - 1))
            dW_bind_a = dW_bind * (eta * 2.0)
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
            net._col_cos = (cov_col * (1.0 - torch.eye(net.bind_slot_dim, device=dev, dtype=torch.float16))).abs().mean()
            W_col.add_(noise)
            net.W_bind[:a4].data = W_col.T.contiguous()
            soft_norm_preserve(net.W_bind[:a4].data)
            # W_bind 行去同质化 (与 W_35 同款, 破秩 1 自锁): E_bind 幂迭代主方向
            # 投影抑制 + E_bind 相关统计累计 (规格书 3/4)
            _decorr_W(net.W_bind[:a4].data, net.E_bind[:a4, :a4])
            # W_bind 列方向 decorr (第 75 轮安全网): 逐样本加权破对称后防单槽垄断.
            # 列空间 = 槽维 16, 转置后行 decorr 同型 (E_bind_col 16×16)
            _decorr_W(net.W_bind[:a4].data.T.contiguous(), net.E_bind_col)
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
                zb_post = zb_post * learn_mask.to(torch.float16).unsqueeze(0).unsqueeze(-1)
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
                    zb_pre.transpose(-2, -1)
                    @ ((zb_post - zb_post.mean(dim=-1, keepdim=True)) * intr)
                ).mean(dim=0)
                dW_self = dW_self / (dW_self.norm() + 1e-8)
                net.W_bind_self.data.mul_(0.9995)
                # 裁决 10: intrinsic 调制 W_bind_self 学习率 — η_self = η_base·(1+0.5·sinφ)
                # 高潮期 (A→1) 学习率 1.5× → 自洽误差上升 → 系统探索新表达;
                # 低谷期 (A→0) 学习率 1.0× → 自洽误差回落 → 固化探索成果.
                # 内部节律驱动探索-固化循环, 无新计算路径, 零 NaN 风险
                eta_self = (eta * 2.0) * (1.0 + 0.5 * A)
                net.W_bind_self.data += dW_self * eta_self
                # 列方向 decorr (第 76 轮裁决): 转置后行 decorr 同型 — 16 列各为
                # 源槽转移向量, 均匀统计下收敛同向量 (实测列相似 0.906), 斥力
                # 迫使分化. 更新后、soft_norm 前挂载 (与 W_bind 主矩阵同序).
                # 注意: 必须传转置视图 (非 contiguous 副本) — 副本修改不写回
                # 原参数 (第 76 轮实测: 0.906→0.935 不降反升, 装饰性失效)
                _decorr_W(net.W_bind_self.data.T, net.E_bind_self)
                soft_norm_preserve(net.W_bind_self.data)

            # ── 闭环自洽生成 (第 76 轮最终裁决: 表达者范式) ──
            # W_act 从"预测器"转"行为生成器": 切断一切外部目标字节驱动.
            # 核心: 系统生成字节 → 内部世界模型 (W_lm) 重感知 → 自洽性惊喜
            # = 1 - probs_lm[gen_byte] → 三因子赫布. 高惊喜 (内部模型意外) →
            # 负向更新抑制该行为; 低惊喜 (内部自洽) → 弱负向保留. 复读是零
            # 惊喜稳定基态 → 自然强化为起点, 从稳定中诞生复杂 (牙牙学语).
            # 哲学: 系统不再匹配外部, 而是维持内部自洽 — 主动表达, 非被动回答.
            # 实现: closed_loop 或自回声 (第 80 轮) 时学习 — 自回声的生成段
            # 就是本窗输入流, 同一批 _gen_bytes 复用 (W_act 表达 → 重感知 →
            # 自洽惊喜 → 三因子赫布, 无标签无奖励)
            if (
                (closed_loop or echo_loop)
                and hasattr(net, "_act_pot")
                and hasattr(net, "_gen_bytes")
                and getattr(net, "_act_enabled", True)
            ):
                pot = net._act_pot  # [N,S,256] 动作电位
                zbind = net._bind_vec
                gb = net._gen_bytes  # [N,N_gen] 自生成字节 (表达)
                N_gen = gb.shape[1]
                zb_g = zbind[:, -N_gen:]  # [N,N_gen,16] 生成段槽状态
                # ── 学习进步信号 (裁决 16): 从"最小化预测误差"到"最大化学习进步" ──
                # 复读是"最小化当前预测误差"的全局最优解 — 必须放弃该目标.
                # LP_t = ε_{t-1} - ε_t (预测误差的负时间差分):
                #   LP>0: 误差下降, 系统"学会了" → 正强化 (保留该行为)
                #   LP<0: 误差上升, 系统"变笨了" → 负强化 (抑制该行为)
                # 复读: ε_{t-1}≈ε_t → LP≈0 → 行为不被强化 → 逐渐遗忘 (打破复读)
                # 探索新字节若提升可预测性 (LP>0) → 强化 → 表达库向复杂演化.
                # 多巴胺真实角色: 奖励预测误差 (比预期更好 = 学习进步), 非奖赏本身
                # 实现: φ = 0.5·(1 - tanh(α·LP)) ∈ [0,1] — LP 大 → φ→0 (弱抑制=
                # 保留), LP 小 → φ→1 (强抑制). 符号保持, 幅度受限, 纯局部.
                # (裁决公式 ΔW=-η·tanh(LP)·... 与"LP>0 正强化"矛盾, 按物理
                # 意图实现: LP>0 保留, LP<0 抑制)
                zb_prev = torch.cat([zb_g[:, :1], zb_g[:, :-1]], dim=1)  # [N,N_gen,16]
                pred_self = torch.softmax((zb_prev @ net.W_bind_self) * 4.0, dim=-1)
                eps_t = (zb_g - pred_self).square().mean(dim=-1, keepdim=True)  # [N,N_gen,1] ε_t
                eps_prev = torch.cat([eps_t[:, :1], eps_t[:, :-1]], dim=1)  # ε_{t-1}
                LP = (eps_prev - eps_t).detach()  # 学习进步 [N,N_gen,1]
                alpha = 50.0  # tanh 缩放 (ε 量级 ~0.003, ×50 → LP 归一)
                phi = 0.5 * (1.0 - torch.tanh(alpha * LP))  # 增益 [0,1]
                probs_g = probs_lm[:, -N_gen:]  # [N,N_gen,256] 生成段概率
                # (第 80 轮: 自回声时生成段 = 整个输入流 S 长, 对齐 S 无需偏移;
                # closed_loop 时生成段是后半, 偏移 S-N_gen-1)
                oh_g = F.one_hot(gb, num_classes=256).to(torch.float16)  # [N,N_gen,256]
                p_gen = (probs_g * oh_g).sum(dim=-1, keepdim=True)  # [N,N_gen,1]
                wlm_err = (1.0 - p_gen).detach()  # [N,N_gen,1] 外部弱约束
                surprise = (0.9 * phi + 0.1 * wlm_err).detach()  # 学习进步增益 + W_lm 弱约束
                # ── 字节频率门控 (裁决 15): 罕见字节抑制衰减 (辅助调制) ──
                fa = net._freq_act
                oh_gen = F.one_hot(gb, num_classes=256).to(torch.float16)  # [N,N_gen,256]
                fa.mul_(0.99).add_(0.01 * oh_gen.mean(dim=(0, 1)))
                beta = 0.5  # 新颖偏好强度
                freq_gate = (1.0 - beta * (1.0 - fa)).unsqueeze(0).unsqueeze(0)  # [1,1,256]
                surprise = surprise * freq_gate  # 逐字节频率调制
                net._LP = LP  # 诊断: 学习进步分布
                # 三因子赫布 (表达者版): dW_act = -η·(z_bind^T ⊗ one_hot(gen_byte))·surprise
                # 负号是裁决核心: 高惊喜 (内部模型意外) → 负向更新抑制该行为;
                # 低惊喜 (内部自洽) → 弱负向, 通路保留. 复读是低惊喜稳定基态
                # → 自然强化为起点. (第 76 轮修正: 旧实现正号把高惊喜表达
                # 反向强化 → 越表达越陌生→越强化→表达库钉死 2 字节自锁)
                dW_act = -(zb_g.transpose(-2, -1) @ (oh_g * surprise)).mean(dim=0)
                # 稳态抑制保留 (防字节垄断): 全列 -= 0.1·z_bind^T @ softmax(potential)
                pot_sm = torch.softmax(pot[:, -N_gen:].detach(), dim=-1)
                dW_act = dW_act - 0.1 * (zb_g.transpose(-2, -1) @ pot_sm).mean(dim=0)
                # 单步幅度上界 (W_lm 同款幅度-方向解耦)
                dW_act = dW_act / (dW_act.norm() + 1e-8)
                # 学习率: W_lm 读出端量级 × 0.2 × intrinsic 并列调制 (纯张量,
                # 零 GPU→CPU 同步 — _intr_drive 已是 fp16 张量, 直接张量乘法)
                intr_d = getattr(net, "_intr_drive", torch.tensor(0.5, device=dev, dtype=torch.float16))
                net.W_act.data += dW_act * (net.cfg.lm_lr_boost * 0.2) * (0.5 + intr_d)
                # ── 复读检测跟踪 (裁决 12): 随机扰动已升级为内部状态错误配对
                # (forward.py 生成路径, W_bind_self 定向扰动替代各向同性噪声).
                # 此处保留复读状态跟踪供诊断/监控
                gb_recent = gb[:, -10:]  # [N,10] 最近 10 字节
                oh_r = F.one_hot(gb_recent, num_classes=256).to(torch.float16)  # [N,10,256]
                n_uniq = (oh_r.sum(dim=1) > 0).sum(dim=-1).to(torch.float16)  # [N]
                rep_run = getattr(net, "_rep_run", torch.zeros(N, device=dev, dtype=torch.float16))
                in_rep = (n_uniq < 3).to(torch.float16)  # [N]
                rep_run = torch.where(in_rep > 0, rep_run + 1.0, torch.zeros_like(rep_run))
                net._rep_run = rep_run
                net._rep_frac = in_rep.mean().to(torch.float16)  # 诊断 (fp16 张量, 零同步)
                # 列范数保持 (0.8-1.2): 槽列有界防溢出; 转置视图 in-place 直接
                # 写回原参数 (第 76 轮修复: 旧 .contiguous() 副本装饰性失效)
                soft_norm_preserve(net.W_act.data.T)

        # 前馈权重软范数保持 (0.8-1.2): W_04/W_42/W_56 无 BCM 约束,
        # 长训累积溢出 fp16 → NaN; 幅度差异保留 (结构化非 clamp)
        # (free_run: W_04/W_42 冻结不更新, 但保持范数约束仍安全 — 不动它们)
        if not free_run:
            for W, a_sz in zip([net.W_04, net.W_42], [a4, a2]):
                soft_norm_preserve(W[:a_sz].data)
            # W_diff 同款软范数保持 (行范数)
            soft_norm_preserve(W_diff_a.data)
        soft_norm_preserve(net.W_56[:a6].data)

        # ── 拓扑重塑 ──
        net._step_counter += 1
        if net._step_counter > net.cfg.prune_warmup and net._step_counter % net.cfg.prune_interval == 0:
            net.pruner._prune()

        stats = {
            "free_energy": (
                eps4.square().mean()
                + eps2.square().mean()
                + eps3.square().mean()
                + eps5_td.square().mean()
                + eps6.square().mean()
            ),
            "future_err": (dz4 - pred_d).square().mean() if not free_run else 0.0,
            "d_polarity": float(d_t.mean()) if (not free_run and hasattr(net, "_novelty")) else 1.0,
            # 观测器 (第 70 轮 v2): 原始 L5 局部误差能量 (同量纲, 供贡献率计算).
            # 诊断用, 不参与任何更新
            "l5_local_err": local_err_l5[:, mask5].square().mean().detach(),
        }
        # 释放每步状态引用 (显存按需): _z* 是 store_state 存的大张量,
        # 不释放则 caching allocator 无法复用, 4GB 卡上逐步累积到 OOM
        # 诊断插桩 (第 77 轮): 观测器用, 释放前保留 z4/z5 引用
        net._last_z4 = net._z4.detach()
        net._last_z5 = net._z5.detach()
        net._last_z3 = net._z3.detach()  # 第 78 轮: 观测 z3 幅度 (NaN 前哨)
        net._last_z2 = net._z2.detach()  # 第 83 轮: 观测 l2 活动 (G8 exc)
        net._last_z6 = net._z6.detach()  # 第 83 轮: 观测 l6 活动 (G8 exc)
        for k in ("_z0", "_z4", "_z2", "_z3", "_z5", "_z5_raw", "_z6"):
            if hasattr(net, k):
                delattr(net, k)
        return stats
