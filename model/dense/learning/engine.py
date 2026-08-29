"""LearningEngine 编排: learn() 主循环 + 共享上下文字段.

学习机制按域拆分在六个 mixin 模块 (类在 __init__.py 组合):
- feedforward.py: W_04/W_42/W_23/W_35/W_56/bias 稳态/W_state_pred
- temporal.py: 多尺度时间窗 + W_diff + W_t 家族 + 谱守卫
- readout.py: LM 信号构建 + W_lm/W_lm_2/W1/bias_lm
- predict.py: 自组织预测引擎 (W_pred_*)
- bind.py: M_l5 + W_bind/W_bind_self
- action.py: W_act 表达通路

域间调用顺序与原始 learn() 单函数逐行一致 (数值逐位等价).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .action import ActionMixin
from .bind import BindMixin
from .feedforward import FeedforwardMixin
from .predict import PredictMixin
from .readout import ReadoutMixin
from .temporal import TemporalMixin

if TYPE_CHECKING:
    from ..network import DensePCNet


@dataclass
class LearnCtx:
    """learn() 步内共享量 (标量/掩码/形状)."""

    net: DensePCNet
    N: int
    S: int
    k: int
    dev: torch.device
    inp: torch.Tensor | None
    byte_ids: torch.Tensor  # 目标字节 (echo 时为自身生成流)
    closed_loop: bool
    free_run: bool
    echo_loop: bool
    echo_world_frozen: bool
    learn_mask: torch.Tensor
    a4: int
    a2: int
    a3: int
    a5: int
    a6: int
    eta: float | torch.Tensor
    eta_t: float | torch.Tensor
    eta_lm: float
    inv_s: float
    learn_boost: torch.Tensor


@dataclass
class DiffWindow:
    """多尺度软时间窗产物 (W_diff 误差/更新)."""

    dz4: torch.Tensor
    pred_d: torch.Tensor
    dW_avg: torch.Tensor
    e_t_all: torch.Tensor


@dataclass
class LmSignal:
    """LM 头前向信号 (含误差投影回表示层)."""

    eps_total: torch.Tensor
    eps_t2_total: torch.Tensor
    eps_lm: torch.Tensor
    eps_lm_proj: torch.Tensor
    eps_lm_pad: torch.Tensor
    logits_lm: torch.Tensor
    logits_t2: torch.Tensor
    probs_lm: torch.Tensor
    h: torch.Tensor
    h2: torch.Tensor
    h_deriv: torch.Tensor
    zh: torch.Tensor
    e_h: torch.Tensor | None = None


@dataclass
class LayerErrs:
    """逐层预测误差 (+ 精度加权版)."""

    eps4: torch.Tensor
    eps2: torch.Tensor
    eps3: torch.Tensor
    eps5_td: torch.Tensor
    eps6: torch.Tensor
    eps2_p: torch.Tensor
    eps6_p: torch.Tensor


@dataclass
class Shared:
    """域间传递: 层误差 / diff 窗 / LM 信号 (free_run 时 diff/lm 为 None)."""

    errs: LayerErrs
    diff: DiffWindow | None = None
    lm: LmSignal | None = None
    l5_local_err: torch.Tensor | None = None
    d_t: torch.Tensor | None = None


class EngineCore(
    FeedforwardMixin,
    TemporalMixin,
    ReadoutMixin,
    PredictMixin,
    BindMixin,
    ActionMixin,
):
    """学习引擎基类: 持 net 引用, 组装 learn() 编排 (域逻辑在 mixin).

    继承关系即组合 (LearningEngine = EngineCore + mixin); mixin 模块不
    import engine (readout 仅在函数内延迟导入 LmSignal), 无循环依赖.
    """

    def __init__(self, net: DensePCNet):
        self.net = net

    def _closed_loop_input(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """自回归暴露输入: 前 k 位真实锚定 + 后 S-k 位模型自生成.

        前半是真实分布 (锚定, 防初始漂移), 后半是模型自己的输出 (暴露,对治复读自锁)
        完全交给内生快慢散度多巴胺 (learning.py D 极性翻转), 无外部干预.
        """
        net = self.net
        k = byte_ids.shape[1] // 2
        n_gen = byte_ids.shape[1] - k
        out = net.forward_engine.continuation(byte_ids[:, :k], n_gen, temperature=0.0, rep_backstop=True)
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
        elif closed_loop:
            inp = self._closed_loop_input(byte_ids)
        elif byte_ids is None:
            # 108轮 自回声自由运行 — 无外部输入, 输入=上一窗生成字节.
            # 第 110 轮 D1 (声道合一): W_lm 读出为唯一声道 — use_w_act 发射
            # 分支已删除 (108 实验: W_act 发射=乱码 + ε_lm 极性倒挂双证伪),
            # W_act 降格为意图调制器 (continuation 内注入 logits, ≤15%).
            # 生成流 = 冻结世界模型在内部状态调制下的自续写.
            s_frw = net.cfg.free_run_window
            seed = getattr(net, "_echo_seed", None)
            if seed is None or seed.numel() == 0:
                seed = torch.zeros(1, 1, dtype=torch.long, device=net._osc_f_cnt.device)
            # 第 110 轮 D5 (声道探索): 温度采样为变异通道 — probe110b 扫描
            # 实测 ε(τ) 单调: 贪心 0.55 (循环带) → τ*=4.0 落语言带心 0.88 →
            # τ≥16 滑乱码带. 温度由恒温器负反馈自调 (action.py: ε 自测偏差
            # → 温度修正, 纯内部量), 默认 4.0 = 物理校准值 (107 κ 先例).
            # 110c: rep_backstop 默认关 — 反循环外部物理门从她的声音移除,
            # 循环由她自己的裁判 (语言带 R 惩罚循环带) + 恒温器处置.
            saved_entropy = getattr(net, "_entropy_sample", False)
            net._entropy_sample = getattr(net, "_echo_entropy", False)
            saved_rep = getattr(net, "_echo_rep", False)
            _gt = getattr(net, "_gen_temp", None)
            _temp = float(_gt.item()) if _gt is not None else 4.0
            try:
                out = net.forward_engine.continuation(
                    seed, s_frw - 1, temperature=_temp, rep_backstop=saved_rep
                )
            finally:
                net._entropy_sample = saved_entropy
            net._gen_bytes = out[:, -(s_frw - 1):]  # 去掉种子, 输入 = 纯生成流
            inp = net._gen_bytes
        else:
            inp = byte_ids
        _ = net.forward_engine._predict(inp, store_state=True)
        N, S = inp.shape if inp is not None else (1, net.cfg.free_run_window)
        dev = next(net.parameters()).device
        k = S // 2 if closed_loop else 0
        # 第80轮: 自回声模式激活字节域学习, 输入=自身生成流.
        # 第102轮修正: 回声相位冻结世界模型 (W_04/W_42/W_diff/W_lm 等), 仅更新表达端 W_act.
        # 原因: 从零初始化时 W_lm 未学到 UTF-8 结构, 回声输入=乱码, 世界模型在乱码上学习会污染感知.
        # 世界模型只在感知相位 (真实输入) 更新, 表达端照常学习.
        echo_loop = (not free_run) and (byte_ids is None)
        if echo_loop:
            byte_ids = inp  # 自回声: 目标 = 自身生成流 (他者响应), 下游同普通训练
        echo_world_frozen = echo_loop
        # 生成段掩码 (closed_loop 时只有后半参与误差, 前半锚定只看不学):
        # 对齐 S-1 (t+1 目标), 位置 i 对应目标 byte_ids[:, i+1]
        learn_mask = torch.ones(S - 1, dtype=torch.bool, device=dev)
        if closed_loop:
            learn_mask[: k - 1] = False
        a4, a2, a3, a5, a6 = (net.active_size[k] for k in ("l4", "l2", "l3", "l5", "l6"))

        W_23_a = net.W_23[:a3]
        z0 = net._z0
        z4, z2, z3, z5, z6 = net._z4, net._z2, net._z3, net._z5, net._z6

        # 逐层预测误差 (自下而上 PC); L5 用时序差分误差, L6 用时间自预测
        eps4 = z4 - (z0 @ net.W_04[:a4].T + net.bias_l4[:a4])
        eps2 = z2 - (z4 @ net.W_42[:a2].T + net.bias_l2[:a2])
        eps3 = z3 - (z2 @ W_23_a.T + net.bias_l3[:a3])
        z6_pre = torch.cat([torch.zeros(N, 1, a6, dtype=z6.dtype, device=dev), z6[:, :-1]], dim=1)
        eps6 = z6 - z6_pre
        eps5_td = z5[:, 1:] - z5[:, :-1]  # L5: 跨时刻变化 z5[t]-z5[t-1]

        fe = net.forward_engine
        eps2_p, eps6_p = (
            fe._precise(eps2),
            fe._precise(eps6),
        )

        # 权重去主成分投影的 learn_boost (超量抑制系数, 第 70-74 轮机制):
        # 在熵竞争更新 _traction_scale 之前取样 (原 learn() 顺序, 数值等价)
        learn_boost = 2.0 - net._traction_scale.to(torch.float16)

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
        eta_lm = net.cfg.lm_lr_boost
        inv_s = 1.0 / S

        ctx = LearnCtx(
            net=net, N=N, S=S, k=k, dev=dev, inp=inp, byte_ids=byte_ids,
            closed_loop=closed_loop, free_run=free_run, echo_loop=echo_loop,
            echo_world_frozen=echo_world_frozen, learn_mask=learn_mask,
            a4=a4, a2=a2, a3=a3, a5=a5, a6=a6,
            eta=eta, eta_t=eta_t, eta_lm=eta_lm, inv_s=inv_s,
            learn_boost=learn_boost,
        )
        sh = Shared(errs=LayerErrs(eps4, eps2, eps3, eps5_td, eps6, eps2_p, eps6_p))

        # ── 域调用 (顺序 = 原 learn() 逐块顺序, 数值逐位等价) ──
        sh.diff = self._build_diff_window(ctx)
        self._update_bias(ctx, sh)
        sh.lm = self._build_lm_signal(ctx)
        self._update_feed_ff(ctx, sh)
        self._update_W35(ctx, sh)
        self._update_pred_engine(ctx, sh)
        self._update_Ml5(ctx, sh)
        self._apply_diff(ctx, sh)
        self._update_wt_family(ctx, sh)
        sh.d_t = self._update_lm_head(ctx, sh)
        self._update_mem_units(ctx, sh)
        self._update_bind(ctx, sh)
        self._update_w_act(ctx, sh)
        self._final_softnorm(ctx, sh)

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
            "future_err": (sh.diff.dz4 - sh.diff.pred_d).square().mean()
            if (not free_run and not echo_world_frozen)
            else 0.0,
            "d_polarity": float(sh.d_t.mean())
            if (not free_run and not echo_world_frozen and sh.d_t is not None and hasattr(net, "_novelty"))
            else 1.0,
            # 观测器 (第 70 轮 v2): 原始 L5 局部误差能量 (同量纲, 供贡献率计算).
            # 诊断用, 不参与任何更新
            "l5_local_err": sh.l5_local_err,
        }
        # 释放每步状态引用 (显存按需): _z* 是 store_state 存的大张量,
        # 不释放则 caching allocator 无法复用, 4GB 卡上逐步累积到 OOM
        # 诊断插桩 (第 77 轮): 观测器用, 释放前保留 z4/z5 引用
        net._last_z4 = net._z4.detach()
        net._last_z5 = net._z5.detach()
        net._last_z3 = net._z3.detach()  # 第 78 轮: 观测 z3 幅度 (NaN 前哨)
        net._last_z2 = net._z2.detach()  # 第 83 轮: 观测 l2 活动 (G8 exc)
        net._last_z6 = net._z6.detach()  # 第 83 轮: 观测 l6 活动 (G8 exc)
        for k_ in ("_z0", "_z4", "_z2", "_z3", "_z5", "_z5_raw", "_z6"):
            if hasattr(net, k_):
                delattr(net, k_)
        return stats
