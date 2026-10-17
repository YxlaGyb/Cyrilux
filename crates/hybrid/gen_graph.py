"""
ContinuationGraphs
生成环静态声明层: 一步解码 = 一张图.

continuation 的逐步张量路径按变体声明为算子序列, 图只持结构, 张量经 env 传入;
步与步之间的依赖来自取样字节回写, 属自回归语义, 环留宿主.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from crates.graphs import Graph, Op, apply_writes, defined_names, run_ref, validate_graph
from crates.kernels import (
    act_inject,
    add_scaled,
    bind_hold,
    buf_seed,
    buf_write,
    clear_counts,
    clear_expect,
    count_add0,
    ctx_window,
    cubic,
    evo_logits,
    expect_step,
    gen_count,
    hold_step,
    hold_zero,
    join3,
    last_row,
    lead_len_of,
    logit_cat,
    logit_head,
    logits_std,
    mask_print,
    mix,
    out_view,
    pred_delta,
    predict_state,
    rep_block,
    rms_norm,
    sample_entropy,
    sample_greedy,
    sample_temp,
    saturate,
    softmax16,
    temp_scale,
    topk_floor,
    unit_norm,
    utf8_mask,
    win_buf,
)
from crates.modal.text.geometry import ALPHABET

_WIN = 64
_P_MAX = 24
_K_TOP = 15


def declare(
    net: Any,
    engine: Any,
    *,
    variant: tuple[tuple[str, Any], ...],
    s_len: int,
    n_gen: int,
    seed: bool,
    last_step: bool,
    commit: bool,
    hold_present: bool,
    temperature_kind: str,
    rep_backstop: bool,
    entropy: bool,
    gen_mode: str,
    migrate_expand: bool,
) -> Graph:
    """按变体声明一次解码步的算子序列与写回契约."""
    ops: list[Op] = []

    def add(name: str, fn: Any, inputs: tuple[str, ...], outputs: tuple[str, ...] = ()) -> None:
        ops.append(Op(name=name, fn=fn, inputs=inputs, outputs=outputs))

    if commit and not hold_present:
        add("hold_init", hold_zero, ("bind_vec",), ("hold",))
    if seed:
        add("win_buf", win_buf, ("cont_cur", "n_total"), ("cur_buf",))
        add("seed_buf", buf_seed, ("cur_buf", "byte_ids", "s_len"))
        add("clear_expect", clear_expect, ("expect_cont",))
        add("clear_counts", clear_counts, ("counts",))
    mark = len(ops)
    add("ctx", ctx_window, ("cur_buf", "cur_len", "win"), ("bv",))
    add("predict", predict_state, ("engine", "bv"), ("z4", "mem_out", "bind_vec_post"))
    add("z4_n", unit_norm, ("z4",), ("z4_n",))
    add("pred_delta", pred_delta, ("z4_n", "W_diff", "b_diff", "dim_4"), ("z4_delta",))
    add("z4_next", add_scaled, ("z4", "z4_delta", "one"), ("z4_next",))
    add("z4_sat", saturate, ("z4_next",), ("z4_sat",))
    add("z4_rms", rms_norm, ("z4_sat",), ("z4_rms",))
    add("z4_cub", cubic, ("z4_rms",), ("z4_cub",))
    add("z4_nl", saturate, ("z4_cub",), ("z4_nl",))
    add("bind", bind_hold, ("bind_vec_post", "hold", "expect_cont", "commit"), ("zh_bind",))
    add("zh_cat", join3, ("z4_nl", "zh_bind", "mem_out"), ("zh_cat",))
    add("zh_rms", rms_norm, ("zh_cat",), ("zh",))
    add("h_old_mix", mix, ("zh", "W1"), ("h_old_m",))
    add("h_old_sat", saturate, ("h_old_m",), ("h_old_sat",))
    add("h_old_rms", rms_norm, ("h_old_sat",), ("h_old_rms",))
    add("h_old", cubic, ("h_old_rms",), ("h_old",))

    expand = gen_mode in ("new", "cat") and migrate_expand
    if expand:
        add("h_new_mix", mix, ("zh", "W1_new"), ("h_new_m",))
        add("h_new_sat", saturate, ("h_new_m",), ("h_new_sat",))
        add("h_new_rms", rms_norm, ("h_new_sat",), ("h_new_rms",))
        add("h_new_cub", cubic, ("h_new_rms",), ("h_new",))

    if gen_mode == "new" and expand:
        add("logits", mix, ("h_new", "W_lm_new"), ("mu0",))
    elif gen_mode == "cat" and expand:
        add("logits", logit_cat, ("h_old", "W_lm", "bias_lm", "h_new", "W_lm_new", "inv_h", "alpha"), ("mu0",))
    else:
        add("logits", logit_head, ("h_old", "W_lm", "bias_lm", "inv_h"), ("mu0",))

    add("lstd", logits_std, ("mu0", "eps"), ("mu0_s",))
    add("lmask", mask_print, ("mu0_s", "mask_print", "floor"), ("mu0_m",))
    add("last", last_row, ("mu0_m",), ("last",))

    if commit:
        add("evo", evo_logits, ("z4_nl", "bind_vec_post", "mem_out", "W1", "W_lm", "bias_lm", "inv_h"), ("lg_evo",))
        add("count", gen_count, ("counts", "last", "lg_evo", "expect_cont"))

    add("inject", act_inject, ("last", "bind_vec_post", "W_act"), ("last_a",))

    tail = "last_a"
    if temperature_kind != "none":
        add("temp", temp_scale, (tail, "temperature"), ("last_t",))
        tail = "last_t"
    if rep_backstop:
        add("rep", rep_block, ("cur_buf", "cur_len", tail, "p_max"), ("last_r", "n_block"))
        add("rep_acc", count_add0, ("counts", "n_block"))
        tail = "last_r"
    add("utf8", utf8_mask, (tail, "expect_cont", "block_cnt", "block_start"), ("last_u",))
    add("topk", topk_floor, ("last_u", "k_top"), ("last_k",))
    add("probs", softmax16, ("last_k",), ("probs",))

    if entropy:
        add("pick", sample_entropy, ("last_k", "freq_act", "beta"), ("b",))
    elif temperature_kind == "none":
        add("pick", sample_greedy, ("probs",), ("b",))
    else:
        add("pick", sample_temp, ("probs",), ("b",))

    add("write", buf_write, ("cur_buf", "b", "cur_len"), ("cur_len_w",))
    add("lead", lead_len_of, ("b",), ("lead",))
    if commit:
        add("hold", hold_step, ("hold", "bind_vec_post", "expect_cont", "lead"), ("hold_new",))
    add("expect", expect_step, ("expect_cont", "lead"))

    if n_gen <= 0:
        del ops[mark:]

    outputs: tuple[str, ...] = ()
    if last_step:
        cursor = "cur_len_w" if n_gen > 0 else "cur_len"
        add("out", out_view, ("cur_buf", cursor), ("out",))
        outputs = ("out",)

    writes: list[tuple[str, str]] = []
    if n_gen > 0:
        writes = [
            ("z4", "_z4"),
            ("mem_out", "_mem_out"),
            ("bind_vec_post", "_bind_vec"),
        ]
        if commit:
            writes.append(("hold_new", "_cont_bind_hold"))
    elif commit and not hold_present:
        writes.append(("hold", "_cont_bind_hold"))

    produced = {name for op in ops for name in op.outputs}
    inputs = tuple(dict.fromkeys(n for op in ops for n in op.inputs if n not in produced))
    graph = Graph(
        name="continuation_step",
        ops=tuple(ops),
        inputs=inputs,
        outputs=outputs,
        writes=tuple(writes),
        variant=variant,
    )
    validate_graph(graph)
    return graph


class ContinuationGraphs:
    """生成步图的声明缓存与喂料层"""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.net = engine.net
        self._cache: dict[tuple[Any, ...], Graph] = {}
        self._bufs: dict[int, torch.Tensor] = {}
        self._buf_src: torch.Tensor | None = None
        self.fingerprints: dict[tuple[Any, ...], str] = {}
        self.hits = 0
        self.misses = 0

    def step(
        self,
        byte_ids: torch.Tensor,
        *,
        n_gen: int,
        cur_len: int,
        seed: bool,
        last_step: bool,
        temperature: float | torch.Tensor | None,
        rep_backstop: bool,
    ) -> tuple[Graph, dict[str, Any]]:
        net = self.net
        s_len = byte_ids.shape[1]
        commit = net._bind_commit
        hold_present = getattr(net, "_cont_bind_hold", None) is not None
        temperature_kind = _temperature_kind(temperature)
        entropy = getattr(net, "_entropy_sample", False)
        gen_mode = getattr(net, "_gen_logit_mode", "old")
        migrate_expand = getattr(net, "_migrate_expand", False)
        variant = self._variant(
            byte_ids,
            n_gen=n_gen,
            cur_len=cur_len,
            seed=seed,
            last_step=last_step,
            commit=commit,
            hold_present=hold_present,
            temperature_kind=temperature_kind,
            rep_backstop=rep_backstop,
            entropy=entropy,
            gen_mode=gen_mode,
            migrate_expand=migrate_expand,
        )
        key = (variant, self._identity())
        graph = self._cache.get(key)
        if graph is None:
            self.misses += 1
            graph = declare(
                net,
                self.engine,
                variant=variant,
                s_len=s_len,
                n_gen=n_gen,
                seed=seed,
                last_step=last_step,
                commit=commit,
                hold_present=hold_present,
                temperature_kind=temperature_kind,
                rep_backstop=rep_backstop,
                entropy=entropy,
                gen_mode=gen_mode,
                migrate_expand=migrate_expand,
            )
            self._cache[key] = graph
        else:
            self.hits += 1
        env = self._feed(
            graph,
            byte_ids=byte_ids,
            n_gen=n_gen,
            cur_len=cur_len,
            temperature=temperature,
            commit=commit,
        )
        run_ref(graph, env)
        if "cur_buf" in env:
            self._buf_src = net._cont_cur
            self._bufs[s_len + n_gen] = env["cur_buf"]
        apply_writes(graph, env, net)
        if key not in self.fingerprints:
            self.fingerprints[key] = graph.fingerprint(env)
        return graph, env

    def _identity(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """结构键: 参数与 buffer 的名字序列, 名字增删即换键."""
        return (
            tuple(n for n, _ in self.net.named_parameters()),
            tuple(n for n, _ in self.net.named_buffers()),
        )

    def _variant(
        self,
        byte_ids: torch.Tensor,
        *,
        n_gen: int,
        cur_len: int,
        seed: bool,
        last_step: bool,
        commit: bool,
        hold_present: bool,
        temperature_kind: str,
        rep_backstop: bool,
        entropy: bool,
        gen_mode: str,
        migrate_expand: bool,
    ) -> tuple[tuple[str, Any], ...]:
        net = self.net
        n_batch, s_len = byte_ids.shape
        dim_4 = net.active_size["l4"]
        bind_dim = net.bind_slot_dim
        dims = (
            dim_4,
            bind_dim,
            net._lm_in - dim_4 - bind_dim,
            net.d_h,
            net.W1_new.shape[-1] if hasattr(net, "W1_new") else 0,
            ALPHABET,
            net._in_dim,
        )
        return (
            ("S", s_len),
            ("N", n_batch),
            ("n_gen", n_gen),
            ("cur_len", cur_len),
            ("seed", seed),
            ("last", last_step),
            ("commit", commit),
            ("hold_present", hold_present),
            ("temperature_kind", temperature_kind),
            ("rep_backstop", rep_backstop),
            ("entropy", entropy),
            ("gen_mode", gen_mode),
            ("alpha", getattr(net, "_gen_logit_alpha", None)),
            ("migrate_expand", migrate_expand),
            ("generator_variant", None),
            ("dims", dims),
            ("params", tuple((n, tuple(p.shape)) for n, p in net.named_parameters())),
            ("buffers", tuple((n, tuple(b.shape)) for n, b in net.named_buffers())),
        )

    def _feed(
        self,
        graph: Graph,
        *,
        byte_ids: torch.Tensor,
        n_gen: int,
        cur_len: int,
        temperature: float | torch.Tensor | None,
        commit: bool,
    ) -> dict[str, Any]:
        """一次解码步的外部名字绑定 (宿主装配阶段)."""
        net = self.net
        n_batch, s_len = byte_ids.shape
        alpha = getattr(net, "_gen_logit_alpha", None)
        values: dict[str, Any] = {
            "engine": self.engine,
            "cont_cur": net._cont_cur,
            "byte_ids": byte_ids,
            "s_len": s_len,
            "n_gen": n_gen,
            "n_total": s_len + n_gen,
            "cur_len": cur_len,
            "win": min(cur_len, _WIN),
            "p_max": min(_P_MAX, cur_len // 2),
            "expect_cont": net._cont_expect,
            "counts": net._gen_counts,
            "hold": getattr(net, "_cont_bind_hold", None),
            "bind_vec": getattr(net, "_bind_vec", None),
            "commit": commit,
            "temperature": temperature,
            "beta": getattr(net, "_entropy_beta", 0.3),
            "alpha": 1.0 if alpha is None else alpha,
            "freq_act": net._freq_act,
            "mask_print": net._mask_print,
            "floor": -1e4,
            "k_top": min(_K_TOP, ALPHABET),
            "eps": 1e-4,
            "inv_h": 1.0 / math.sqrt(net.d_h),
            "one": 1.0,
            "dim_4": net.active_size["l4"],
            "W_diff": net.W_diff,
            "b_diff": net.b_diff,
            "W1": net.W1,
            "W_lm": net.W_lm,
            "bias_lm": net.bias_lm,
            "W_act": net.W_act,
            "W1_new": getattr(net, "W1_new", None),
            "W_lm_new": getattr(net, "W_lm_new", None),
            "block_cnt": net._utf8_block_cnt,
            "block_start": net._utf8_block_start,
        }
        if "cur_buf" in graph.inputs:
            if self._buf_src is not net._cont_cur:
                self._bufs.clear()
                self._buf_src = net._cont_cur
            buf = self._bufs.get(s_len + n_gen)
            values["cur_buf"] = net._cont_cur[:, : s_len + n_gen] if buf is None else buf
        return {name: values[name] for name in graph.inputs}


def _temperature_kind(temperature: float | torch.Tensor | None) -> str:
    """温度形态: None 与 float<=0 同图 (贪心), 其余一律做除法 (含 int 与张量)."""
    if temperature is None:
        return "none"
    if isinstance(temperature, float) and temperature <= 0.0:
        return "none"
    if isinstance(temperature, torch.Tensor):
        return "tensor"
    return "float_pos"


def defined(graph: Graph) -> frozenset[str]:
    """图中所有已定义名"""
    return defined_names(graph)
