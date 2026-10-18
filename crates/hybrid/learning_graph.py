"""
学习路径静态图
一个算子 = learn() 原本的一次域调用, 域间值全部升为显式名字.

算子本体是既有 mixin 方法 (绑定方法, 模块级函数身份不变)
图只持结构:
    算子序列、名字接线、变体键. 张量一律经 env 传入, 图不持 ctx;
权重写是域块副作用, 不经写回契约.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from crates.device import CaptureBank, capture_available
from crates.graphs import CAPTURE, Graph, Op, backend, run_ref

from .learning._common import layer_errs

if TYPE_CHECKING:
    from crates.model.model_cyrene import HybridPCNet

    from .learning.engine import LearnCtx

NAME = "learn"
_MX_LAYERS = ("l4", "l2", "l3", "l6")
# 冷启动前缀: 学习族首两帧走解释求值, 让哨兵标志/EMA 冷启动/惰性缓冲在入图前定型.
# 捕获 warmup 会真实执行算子, 若这些宿主状态仍在图内改变, 回放就会跳过首帧语义.
_CAPTURE_AFTER = 2


def corebrain_learn(ctx: LearnCtx) -> dict[str, Any]:
    """中心脑一步: 图里以 ctx 作接线锚, 计算只依赖中心脑自身状态."""
    return ctx.net._core_brain.learn()


def bound_ops(net: HybridPCNet, engine: Any) -> dict[str, Callable[..., Any]]:
    """算子表: 每个域一次绑定, 名字 = 图里的算子身份 (声明期一次, 不随步重建)."""
    return {
        "layer_errs": layer_errs,
        "diff_window": engine._build_diff_window,
        "bias": engine._update_bias,
        "lm_signal": engine._build_lm_signal,
        "metabolism": engine._update_metabolism,
        "feed_ff": engine._update_feed_ff,
        "w35": engine._update_W35,
        "pred_engine": engine._update_pred_engine,
        "mx": engine._update_Mx,
        "ml5": engine._update_Ml5,
        "apply_diff": engine._apply_diff,
        "wt_family": engine._update_wt_family,
        "lm_head": engine._update_lm_head,
        "mem_account": net.mem_pool.account,
        "bind": engine._update_bind,
        "w_act": engine._update_w_act,
        "softnorm": engine._final_softnorm,
        "gate": engine._update_gate,
        "corebrain": corebrain_learn,
    }


def declare(
    net: HybridPCNet,
    ops: dict[str, Callable[..., Any]],
    *,
    ctx: LearnCtx,
    variant: tuple[tuple[str, Any], ...],
) -> Graph:
    """按变体声明本步的域表 (顺序即 learn() 原顺序)."""
    freeze = getattr(net, "_freeze_below", False)
    steps: list[Op] = []

    def add(name: str, inputs: tuple[str, ...], outputs: tuple[str, ...] = (), key: str = "") -> None:
        steps.append(Op(name, ops[key or name], inputs, outputs))

    add("layer_errs", ("ctx",), ("errs",))
    add("diff_window", ("ctx",), ("diff",))
    if not freeze:
        add("bias", ("ctx", "errs"))
    add("lm_signal", ("ctx",), ("lm",))
    add("metabolism", ("ctx", "errs", "diff", "lm"), ("metab_f", "metab_f_base", "metab_f_ms"))
    if not freeze:
        add("feed_ff", ("ctx", "errs", "metab_f", "metab_f_base", "metab_f_ms", "lm"))
        add("w35", ("ctx", "lm"))
        add("pred_engine", ("ctx", "lm"), ("l5_local_err",))
        if net.cfg.l5_inhibition == "ml":
            add("mx_l5", ("ctx", "layer_l5"), key="mx")
        else:
            add("ml5", ("ctx",))
        add("apply_diff", ("ctx", "diff"))
        add("wt_family", ("ctx",))
    add("lm_head", ("ctx", "lm"), ("d_t",))
    if not freeze:
        add("mem_account", ("ctx", "lm"), ("mem_dead",))
        add("bind", ("ctx", "lm"))
        add("w_act", ("ctx", "lm"))
        for layer in _MX_LAYERS:
            add(f"mx_{layer}", ("ctx", f"layer_{layer}"), key="mx")
    add("softnorm", ("ctx",))
    add("gate", ("ctx", "lm"))
    if not ctx.free_run and not ctx.echo_world_frozen:
        add("corebrain", ("ctx",), ("core_stats",))

    return Graph(
        name=NAME,
        ops=tuple(steps),
        inputs=("ctx",) + tuple(f"layer_{layer}" for layer in ("l4", "l2", "l3", "l5", "l6")),
        outputs=("errs", "diff", "lm", "d_t"),
        variant=variant,
    )


class LearningGraphs:
    """学习图缓存: (变体键, 结构键) → 图, 逐键指纹.

    结构键是参数与 buffer 的名字序列, 形状归变体键, 值不参与.
    形状一变变体键即不同, 旧图由 _drop_stale 丢弃; clear() 是显式全丢.
    """

    def __init__(self, net: HybridPCNet, engine: Any):
        self.net = net
        self.engine = engine
        self._ops: dict[str, Callable[..., Any]] | None = None
        self._cache: dict[tuple[Any, ...], Graph] = {}
        self._bank = CaptureBank(NAME, net)
        self._env: dict[str, Any] = {
            "layer_l4": "l4",
            "layer_l2": "l2",
            "layer_l3": "l3",
            "layer_l5": "l5",
            "layer_l6": "l6",
        }
        self.hits = 0
        self.misses = 0
        self._seen: dict[tuple[Any, ...], int] = {}
        self.fingerprints: dict[tuple[Any, ...], str] = {}

    def clear(self) -> None:
        """结构事件后丢弃全部图与指纹 (下一帧重新声明)."""
        self._cache.clear()
        self.fingerprints.clear()
        self._seen.clear()
        self._bank.drop()

    def _captures(self) -> bool:
        """学习族入图的判据: 只有后端与设备, 无逐步形状项 (形状归变体键)."""
        return backend() == CAPTURE and capture_available(self.net)

    def step(self, ctx: LearnCtx) -> tuple[Graph, dict[str, Any]]:
        """取图 (声明或复用) → 绑定环境 → 求值, 返回图与求值环境."""
        if self._ops is None:
            self._ops = bound_ops(self.net, self.engine)
        variant = self._variant(ctx)
        key = (variant, self._identity())
        graph = self._cache.get(key)
        if graph is None:
            self._drop_stale(variant)
            self.misses += 1
            graph = declare(self.net, self._ops, ctx=ctx, variant=variant)
            self._cache[key] = graph
        else:
            self.hits += 1
        env = self._env
        env["ctx"] = ctx
        warm = self._seen.get(key, 0)
        self._seen[key] = warm + 1
        if self._captures() and warm >= _CAPTURE_AFTER:
            env = self._bank.step(key, graph, env)
        else:
            run_ref(graph, env)
        if key not in self.fingerprints:
            self.fingerprints[key] = graph.fingerprint(env)
        return graph, env

    def _drop_stale(self, variant: tuple[tuple[str, Any], ...]) -> None:
        """形状项一变, 旧图绑定的形状不再成立."""
        live = dict(variant)
        for key in list(self._cache):
            stale = dict(key[0])
            if any(stale.get(name) != live.get(name) for name in ("params", "buffers")):
                self._cache.pop(key, None)

    def _identity(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """结构键: 参数与 buffer 的名字序列 (形状归变体键, 值不参与)."""
        return (
            tuple(name for name, _ in self.net.named_parameters()),
            tuple(name for name, _ in self.net.named_buffers()),
        )

    def _variant(self, ctx: LearnCtx) -> tuple[tuple[str, Any], ...]:
        net = self.net
        k_mem, a_mem = net._mem_m.shape
        return (
            ("S", ctx.S),
            ("N", ctx.N),
            ("freeze_below", getattr(net, "_freeze_below", False)),
            ("l5_inhibition", net.cfg.l5_inhibition),
            ("free_run", ctx.free_run),
            ("echo_loop", ctx.echo_loop),
            ("echo_world_frozen", ctx.echo_world_frozen),
            ("closed_loop", ctx.closed_loop),
            ("echo_learn", getattr(net, "_echo_learn", False)),
            ("f_multiscale", getattr(net, "_f_multiscale", False)),
            ("f_ms_audit", getattr(net, "_f_ms_audit", False)),
            ("w1v2", getattr(net, "_w1v2", False)),
            ("dims", tuple(net.active_size[k] for k in ("l4", "l2", "l3", "l5", "l6"))),
            ("K", k_mem),
            ("a_mem", a_mem),
            ("act_enabled", getattr(net, "_act_enabled", True)),
            ("params", tuple((name, tuple(p.shape)) for name, p in net.named_parameters())),
            ("buffers", tuple((name, tuple(b.shape)) for name, b in net.named_buffers())),
        )
