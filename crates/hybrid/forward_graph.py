"""
感知逐步路径的静态图

声明:
    算子序列、名字接线、写回契约
图只持结构, 张量每次调用重建

按 (结构键, 变体键) 缓存, 结构事件 (出生/死亡/剪枝/压缩/init_* 迁移) 由宿主清缓存
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from crates.graphs import Graph, Op, apply_writes, defined_names, run_ref, validate_graph
from crates.hybrid.mask import bind, compete, g_rate, rate_ema
from crates.kernels import (
    ach_noise,
    add_scaled,
    add_unsqueezed,
    align_mem_window,
    align_rows,
    align_square,
    bias_3d,
    center_last,
    drop_last,
    foldiak_gate,
    history_join,
    join_last,
    l2_norm,
    mem_advance,
    mem_last,
    mem_read,
    mem_reset,
    nonzero_mask,
    nov_advance,
    nov_read,
    nov_reset,
    one_hot_input,
    osc_advance,
    osc_phase,
    pack3,
    pred_err,
    prepend_pad,
    project,
    saturate,
    seq_diff,
    unit_norm,
    zero_input,
)
from crates.modal.text.geometry import ALPHABET
from crates.model.constants import RHO_TARGET
from crates.neuron import survival_mask

if TYPE_CHECKING:
    from crates.hybrid.forward import ForwardEngine
    from crates.model.model_cyrene import HybridPCNet

_LAYERS = ("l4", "l2", "l3", "l5", "l6")
_STP_LAYERS = ("l4", "l2", "l3", "l6")

_ALIGN_ROWS = (
    ("W_04", "W_04_a", "dim_4"),
    ("W_42", "W_42_a", "dim_2"),
    ("W_23", "W_23_a", "dim_3"),
    ("W_35", "W_35_a", "dim_5"),
    ("W_56", "W_56_a", "dim_6"),
    ("bias_l4", "b_l4", "dim_4"),
    ("bias_l2", "b_l2", "dim_2"),
    ("bias_l3", "b_l3", "dim_3"),
    ("bias_l5", "b_l5", "dim_5"),
    ("bias_l6", "b_l6", "dim_6"),
    ("b_diff", "b_diff_a", "dim_4"),
    ("r_l4", "r_l4_a", "dim_4"),
    ("r_l2", "r_l2_a", "dim_2"),
    ("r_l3", "r_l3_a", "dim_3"),
    ("r_l5", "r_l5_a", "dim_5"),
    ("r_l6", "r_l6_a", "dim_6"),
)

_ALIGN_SQUARE = (
    ("W_diff", "W_diff_a", "dim_4"),
    ("W_t4", "W_t4_a", "dim_4"),
    ("W_t2", "W_t2_a", "dim_2"),
    ("W_t3", "W_t3_a", "dim_3"),
    ("W_t6", "W_t6_a", "dim_6"),
    ("M_x4", "M_x4_a", "dim_4"),
    ("M_x2", "M_x2_a", "dim_2"),
    ("M_x3", "M_x3_a", "dim_3"),
    ("M_x5", "M_x5_a", "dim_5"),
    ("M_x6", "M_x6_a", "dim_6"),
    ("M_l5", "M_l5_a", "dim_5"),
)

_STP_ALIGN = (("l4", "dim_4", "stp4"), ("l2", "dim_2", "stp2"), ("l3", "dim_3", "stp3"), ("l6", "dim_6", "stp6"))


def _k(layer: str, width: int) -> int:
    """k-WTA 胜者数: 由 LAYER_CONFIG 比例导出, 不新增常数"""
    return max(1, int(RHO_TARGET[layer] * width))


def declare(
    net: HybridPCNet,
    engine: ForwardEngine,
    *,
    free_run: bool,
    store_state: bool,
    is_inference: bool,
    s_len: int,
    variant: tuple[tuple[str, Any], ...],
) -> Graph:
    """按变体声明 _predict 的算子序列与写回契约."""
    ops: list[Op] = []

    def add(name: str, fn: Any, inputs: tuple[str, ...], outputs: tuple[str, ...]) -> None:
        ops.append(Op(name=name, fn=fn, inputs=inputs, outputs=outputs))

    for src, out, width in _ALIGN_ROWS:
        add(f"align_{src}", align_rows, (src, width), (out,))
    for src, out, width in _ALIGN_SQUARE:
        add(f"align_{src}", align_square, (src, width), (out,))
    add("align_mem_window", align_mem_window, ("mem_m_seq", "n_batch", "s_len", "k_mem", "a_mem"), ("mem_seq",))
    if free_run:
        for layer, dim, stp in _STP_ALIGN:
            add(f"{stp}_r", align_rows, (f"stp_r_{layer}", dim), (f"{stp}_r_a",))
            add(f"{stp}_tau", align_rows, (f"stp_tau_{layer}", dim), (f"{stp}_tau_a",))
            add(f"{stp}_u", align_rows, (f"stp_u_{layer}", dim), (f"{stp}_u_a",))
            add(f"{stp}_pack", pack3, (f"{stp}_r_a", f"{stp}_tau_a", f"{stp}_u_a"), (stp,))

    if free_run:
        add("input_zero", zero_input, ("dev", "n_batch", "s_len", "in_dim"), ("z0",))
        add("osc_f", osc_phase, ("tab_f", "cnt_f", "span_f", "s_len"), ("vf",))
        add("osc_m", osc_phase, ("tab_m", "cnt_m", "span_m", "s_len"), ("vm",))
        add("osc_s", osc_phase, ("tab_s", "cnt_s", "span_s", "s_len"), ("vs",))
        add("osc_f_step", osc_advance, ("cnt_f", "s_len", "span_f"), ("osc_f_done",))
        add("osc_m_step", osc_advance, ("cnt_m", "s_len", "span_m"), ("osc_m_done",))
        add("osc_s_step", osc_advance, ("cnt_s", "s_len", "span_s"), ("osc_s_done",))
    elif net.cfg.input_history:
        add("input_one_hot", one_hot_input, ("byte_ids", "alphabet"), ("z0_now",))
        add("input_history", history_join, ("padmax", "z0_now", "alphabet"), ("z0_prev",))
        add("input_join", join_last, ("z0_now", "z0_prev"), ("z0",))
    else:
        add("input_one_hot", one_hot_input, ("byte_ids", "alphabet"), ("z0",))

    add("mu4_project", project, ("z0", "W_04_a", "b_l4"), ("mu4",))
    mu4_used = "mu4"
    if free_run:
        add("mu4_osc", add_scaled, ("mu4", "vm", "amp_m"), ("mu4_osc",))
        mu4_used = "mu4_osc"
    add("l4_recur", engine._recurrent, (mu4_used, "dev", "W_t4_a", "sweeps", "stp4", "tag_l4"), ("z4_r",))
    add("l4_mask", survival_mask, ("net", "layer_l4"), ("mask_l4",))
    add("l4_gate", g_rate, ("r_l4_a", "rho_l4"), ("g_l4",))
    add("l4_compete", compete, ("z4_r", "mask_l4", "k_l4", "M_x4_a", "lam_l4", "g_l4"), ("z4", "C4"))
    add("l4_rate", rate_ema, ("r_l4_a", "C4", "rate_decay"), ("r_l4_step",))
    add("l4_bind", bind, ("C4", "mask_l4"), ("P_l4",))
    z4_used = "z4"
    if not free_run:
        add("valid_pos", nonzero_mask, ("byte_ids",), ("valid_pos",))
        add("core_brain", net._core_brain.forward, ("z4", "valid_pos"), ("core_delta",))
        add("z4_core", add_unsqueezed, ("z4", "core_delta"), ("z4_core",))
        z4_used = "z4_core"
    add("z4_norm", l2_norm, (z4_used,), ("z4_n",))

    add("mu2_project", project, ("z4_n", "W_42_a", "b_l2"), ("mu2",))
    mu2_used = "mu2"
    if free_run:
        add("mu2_osc", add_scaled, ("mu2", "vm", "amp_m"), ("mu2_osc",))
        mu2_used = "mu2_osc"
    add("l2_recur", engine._recurrent, (mu2_used, "dev", "W_t2_a", "sweeps", "stp2", "tag_l2"), ("z2_r",))
    add("l2_mask", survival_mask, ("net", "layer_l2"), ("mask_l2",))
    add("l2_gate", g_rate, ("r_l2_a", "rho_l2"), ("g_l2",))
    add("l2_compete", compete, ("z2_r", "mask_l2", "k_l2", "M_x2_a", "lam_l2", "g_l2"), ("z2", "C2"))
    add("l2_rate", rate_ema, ("r_l2_a", "C2", "rate_decay"), ("r_l2_step",))
    add("l2_bind", bind, ("C2", "mask_l2"), ("P_l2",))
    add("z2_norm", l2_norm, ("z2",), ("z2_n",))

    add("mu3_project", project, ("z2_n", "W_23_a", "b_l3"), ("mu3",))
    mu3_used = "mu3"
    if free_run:
        add("mu3_osc", add_scaled, ("mu3", "vs", "amp_s"), ("mu3_osc",))
        mu3_used = "mu3_osc"
    if not is_inference:
        add("mu3_noise", ach_noise, (mu3_used, "nov"), ("mu3_noise",))
        mu3_used = "mu3_noise"
    add("l3_recur", engine._recurrent, (mu3_used, "dev", "W_t3_a", "sweeps", "stp3", "tag_l3"), ("z3_r",))
    z3_used = "z3_r"
    if free_run:
        add("z3_saturate", saturate, ("z3_r",), ("z3_s",))
        z3_used = "z3_s"
    add("l3_mask", survival_mask, ("net", "layer_l3"), ("mask_l3",))
    add("l3_gate", g_rate, ("r_l3_a", "rho_l3"), ("g_l3",))
    add("l3_compete", compete, (z3_used, "mask_l3", "k_l3", "M_x3_a", "lam_l3", "g_l3"), ("z3", "C3"))
    add("l3_rate", rate_ema, ("r_l3_a", "C3", "rate_decay"), ("r_l3_step",))
    add("l3_bind", bind, ("C3", "mask_l3"), ("P_l3",))
    add("z3_norm", l2_norm, ("z3",), ("z3_n",))

    add("z5_project", project, ("z3_n", "W_35_a", "b_l5"), ("z5",))
    z5_used = "z5"
    if free_run:
        add("z5_osc", add_scaled, ("z5", "vf", "amp_f"), ("z5_osc",))
        z5_used = "z5_osc"
    add("l5_mask", survival_mask, ("net", "layer_l5"), ("mask_l5",))
    add("l5_gate", g_rate, ("r_l5_a", "rho_l5"), ("g_l5",))
    if net.cfg.l5_inhibition == "ml":
        add("l5_compete", compete, (z5_used, "mask_l5", "k_l5", "M_x5_a", "lam_l5", "g_l5"), ("z5_raw", "C5"))
    else:
        add("l5_foldiak", foldiak_gate, (z5_used, "M_l5_a"), ("z5_fd",))
        add("l5_compete", compete, ("z5_fd", "mask_l5", "k_l5", "none_l5", "lam_zero", "g_l5"), ("z5_raw", "C5"))
    add("l5_rate", rate_ema, ("r_l5_a", "C5", "rate_decay"), ("r_l5_step",))
    add("l5_bind", bind, ("C5", "mask_l5"), ("P_l5",))
    add("z5_center", center_last, ("z5_raw",), ("z5_c",))

    add("mu6_project", project, ("z5_c", "W_56_a", "b_l6"), ("mu6",))
    mu6_used = "mu6"
    if free_run:
        add("mu6_osc", add_scaled, ("mu6", "vs", "amp_s"), ("mu6_osc",))
        mu6_used = "mu6_osc"
    add("l6_recur", engine._recurrent, (mu6_used, "dev", "W_t6_a", "sweeps", "stp6", "tag_l6"), ("z6_r",))
    add("l6_mask", survival_mask, ("net", "layer_l6"), ("mask_l6",))
    add("l6_gate", g_rate, ("r_l6_a", "rho_l6"), ("g_l6",))
    add("l6_compete", compete, ("z6_r", "mask_l6", "k_l6", "M_x6_a", "lam_l6", "g_l6"), ("z6", "C6"))
    add("l6_rate", rate_ema, ("r_l6_a", "C6", "rate_decay"), ("r_l6_step",))
    add("l6_bind", bind, ("C6", "mask_l6"), ("P_l6",))

    add("dz4_diff", seq_diff, ("z4",), ("dz4",))
    add("dz4_norm", unit_norm, ("dz4",), ("dz4_n",))
    add("z4_prev", drop_last, ("z4",), ("z4_prev",))
    add("z4_prev_norm", unit_norm, ("z4_prev",), ("z4_prev_n",))
    add("z4_prev_pad", prepend_pad, ("z4_prev_n", "padmax", "dim_4"), ("z4_prev_pad",))
    add("b_diff_3d", bias_3d, ("b_diff_a",), ("b_diff_3d",))
    add("mu_diff", project, ("z4_prev_pad", "W_diff_a", "b_diff_3d"), ("mu_diff",))
    add("diff_err", pred_err, ("dz4_n", "mu_diff"), ("diff_err",))

    writes: list[tuple[str, str]] = [
        (mu4_used, "_mu4_diag"),
        ("P_l4", "_P_l4"),
        ("P_l2", "_P_l2"),
        ("P_l3", "_P_l3"),
        ("P_l5", "_P_l5"),
        ("P_l6", "_P_l6"),
        ("bind_raw", "_bind_pre"),
        ("bind_vec", "_bind_vec"),
        ("intent_pot", "_intent_pot"),
    ]

    if store_state:
        if free_run:
            add("stp_writeback", engine._stp_writeback, ("stp_end", "z4", "z2", "z3", "z6"), ("stp_step",))
        add("mem_reset", mem_reset, ("mem_seq", "mem_m", "n_batch"), ("mem_step_0",))
        for t in range(1, s_len):
            add(f"mem_t{t}", mem_advance, ("mem_seq", "z4", f"t_{t}", "mem_a"), (f"mem_step_{t}",))
        add("mem_read", mem_read, ("mem_seq", "z4", "kk"), ("mem_out",))
        add("mem_last", mem_last, ("mem_seq",), ("mem_last",))
        add("nov_reset", nov_reset, ("z4",), ("zslow",))
        for t in range(1, s_len):
            add(f"nov_t{t}", nov_advance, ("zslow", "z4", f"t_{t}"), (f"nov_step_{t}",))
        add("nov_read", nov_read, ("z4", "zslow"), ("nov_out",))
        writes += [
            ("z0", "_z0"),
            ("z4", "_z4"),
            ("z2", "_z2"),
            ("z3", "_z3"),
            ("z5_c", "_z5"),
            ("z5_raw", "_z5_raw"),
            ("z6", "_z6"),
            ("mem_out", "_mem_out"),
            ("mem_last", "_mem_m"),
            ("nov_out", "_novelty"),
        ]

    add("bind_slots", engine._bind, ("z4", "vf" if free_run else "none_vf"), ("bind_raw", "bind_vec", "intent_pot"))

    graph = Graph(
        name="predict",
        ops=tuple(ops),
        inputs=_inputs(s_len, free_run=free_run),
        outputs=("mu_diff", "diff_err"),
        writes=tuple(writes),
        variant=variant,
    )
    validate_graph(graph)
    return graph


def _inputs(s_len: int, *, free_run: bool) -> tuple[str, ...]:
    """外部名字清单: 全量权重与缓冲、配置标量、状态张量与逐步常数."""
    names = [
        "dev", "n_batch", "s_len", "in_dim", "alphabet", "sweeps",
        "byte_ids", "net", "padmax",
        "dim_4", "dim_2", "dim_3", "dim_5", "dim_6",
        "layer_l4", "layer_l2", "layer_l3", "layer_l5", "layer_l6",
        "W_04", "W_42", "W_23", "W_35", "W_56", "W_diff",
        "W_t4", "W_t2", "W_t3", "W_t6",
        "M_x4", "M_x2", "M_x3", "M_x5", "M_x6", "M_l5",
        "bias_l4", "bias_l2", "bias_l3", "bias_l5", "bias_l6", "b_diff",
        "r_l4", "r_l2", "r_l3", "r_l5", "r_l6",
        "rho_l4", "rho_l2", "rho_l3", "rho_l5", "rho_l6",
        "lam_l4", "lam_l2", "lam_l3", "lam_l5", "lam_l6",
        "k_l4", "k_l2", "k_l3", "k_l5", "k_l6", "rate_decay",
        "tag_l4", "tag_l2", "tag_l3", "tag_l6", "stp_end",
        "tab_f", "cnt_f", "span_f", "tab_m", "cnt_m", "span_m", "tab_s", "cnt_s", "span_s",
        "amp_f", "amp_m", "amp_s",
        "mem_m", "mem_a", "mem_m_seq", "kk", "k_mem", "a_mem",
        "nov", "none_vf", "none_l5", "lam_zero",
    ]
    if free_run:
        for layer in _STP_LAYERS:
            names += [f"stp_r_{layer}", f"stp_tau_{layer}", f"stp_u_{layer}"]
    else:
        names += ["stp4", "stp2", "stp3", "stp6"]
    return tuple(names) + tuple(f"t_{t}" for t in range(1, s_len))


class ForwardGraphs:
    """_predict 的图缓存与喂料: 图按 (结构键, 变体键) 复用, 张量每调用重建."""

    def __init__(self, engine: ForwardEngine) -> None:
        self.engine = engine
        self.net = engine.net
        self.hits = 0
        self.misses = 0
        self.fingerprints: dict[tuple[Any, ...], str] = {}
        self._cache: dict[tuple[Any, ...], Graph] = {}

    def clear(self) -> None:
        """结构事件 (出生/死亡/剪枝/压缩/init_* 迁移) 后由宿主调用."""
        self._cache.clear()
        self.fingerprints.clear()

    def step(
        self, byte_ids: torch.Tensor | None, *, store_state: bool, is_inference: bool
    ) -> tuple[Graph, dict[str, Any]]:
        net = self.net
        free_run = byte_ids is None
        if free_run:
            n_batch, s_len = 1, net.cfg.free_run_window
            dev = next(net.parameters()).device
        else:
            n_batch, s_len = byte_ids.shape
            dev = byte_ids.device
        variant = self._variant(free_run, store_state, is_inference, s_len, n_batch)
        key = (variant, self._identity())
        graph = self._cache.get(key)
        if graph is None:
            self.misses += 1
            graph = declare(
                net,
                self.engine,
                free_run=free_run,
                store_state=store_state,
                is_inference=is_inference,
                s_len=s_len,
                variant=variant,
            )
            self._cache[key] = graph
        else:
            self.hits += 1
        env = self._feed(graph, byte_ids=byte_ids, dev=dev, s_len=s_len, n_batch=n_batch, free_run=free_run)
        run_ref(graph, env)
        apply_writes(graph, env, net)
        if key not in self.fingerprints:
            self.fingerprints[key] = graph.fingerprint(env)
        return graph, env

    def _identity(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """结构键: 参数与 buffer 的名字序列, 名字增删即换键.

        形状归变体键, 值不参与; 写回会把 buffer 重绑为新对象 (如 `_mem_m`),
        身份键若持对象 id 则每次调用都换键, 复用永久失效.
        """
        return (
            tuple(n for n, _ in self.net.named_parameters()),
            tuple(n for n, _ in self.net.named_buffers()),
        )

    def _variant(
        self, free_run: bool, store_state: bool, is_inference: bool, s_len: int, n_batch: int
    ) -> tuple[tuple[str, Any], ...]:
        net = self.net
        k_mem, a_mem = net._mem_m.shape
        return (
            ("S", s_len),
            ("N", n_batch),
            ("is_inference", is_inference),
            ("store_state", store_state),
            ("free_run", free_run),
            ("l5_inhibition", net.cfg.l5_inhibition),
            ("input_history", net.cfg.input_history),
            ("generator_variant", None),
            ("dims", tuple(net.active_size[k] for k in _LAYERS)),
            ("K", k_mem),
            ("a_mem", a_mem),
            ("params", tuple((n, tuple(p.shape)) for n, p in net.named_parameters())),
            ("buffers", tuple((n, tuple(b.shape)) for n, b in net.named_buffers())),
        )

    def _feed(
        self,
        graph: Graph,
        *,
        byte_ids: torch.Tensor | None,
        dev: torch.device,
        s_len: int,
        n_batch: int,
        free_run: bool,
    ) -> dict[str, Any]:
        net = self.net
        dim_4, dim_2, dim_3, dim_5, dim_6 = (net.active_size[k] for k in _LAYERS)
        k_mem, a_mem = net._mem_m.shape
        values: dict[str, Any] = {
            "dev": dev,
            "n_batch": n_batch,
            "s_len": s_len,
            "in_dim": net._in_dim,
            "alphabet": ALPHABET,
            "sweeps": 1,
            "byte_ids": byte_ids,
            "net": net,
            "padmax": net._padmax,
            "dim_4": dim_4,
            "dim_2": dim_2,
            "dim_3": dim_3,
            "dim_5": dim_5,
            "dim_6": dim_6,
            "layer_l4": "l4",
            "layer_l2": "l2",
            "layer_l3": "l3",
            "layer_l5": "l5",
            "layer_l6": "l6",
            "W_04": net.W_04,
            "W_42": net.W_42,
            "W_23": net.W_23,
            "W_35": net.W_35,
            "W_56": net.W_56,
            "W_diff": net.W_diff,
            "W_t4": net.W_t4,
            "W_t2": net.W_t2,
            "W_t3": net.W_t3,
            "W_t6": net.W_t6,
            "M_x4": net.M_x4,
            "M_x2": net.M_x2,
            "M_x3": net.M_x3,
            "M_x5": net.M_x5,
            "M_x6": net.M_x6,
            "M_l5": net.M_l5,
            "bias_l4": net.bias_l4,
            "bias_l2": net.bias_l2,
            "bias_l3": net.bias_l3,
            "bias_l5": net.bias_l5,
            "bias_l6": net.bias_l6,
            "b_diff": net.b_diff,
            "r_l4": net._r_l4,
            "r_l2": net._r_l2,
            "r_l3": net._r_l3,
            "r_l5": net._r_l5,
            "r_l6": net._r_l6,
            "rho_l4": RHO_TARGET["l4"],
            "rho_l2": RHO_TARGET["l2"],
            "rho_l3": RHO_TARGET["l3"],
            "rho_l5": RHO_TARGET["l5"],
            "rho_l6": RHO_TARGET["l6"],
            "lam_l4": net._lam["l4"],
            "lam_l2": net._lam["l2"],
            "lam_l3": net._lam["l3"],
            "lam_l5": net._lam["l5"],
            "lam_l6": net._lam["l6"],
            "k_l4": _k("l4", dim_4),
            "k_l2": _k("l2", dim_2),
            "k_l3": _k("l3", dim_3),
            "k_l5": _k("l5", dim_5),
            "k_l6": _k("l6", dim_6),
            "rate_decay": net._rate_decay,
            "tag_l4": "l4" if free_run else None,
            "tag_l2": "l2" if free_run else None,
            "tag_l3": "l3" if free_run else None,
            "tag_l6": "l6" if free_run else None,
            "stp_end": net._stp_r_end,
            "tab_f": net._osc_f_tab,
            "cnt_f": net._osc_f_cnt,
            "span_f": net._osc_f_tab.shape[0],
            "tab_m": net._osc_m_tab,
            "cnt_m": net._osc_m_cnt,
            "span_m": net._osc_m_tab.shape[0],
            "tab_s": net._osc_s_tab,
            "cnt_s": net._osc_s_cnt,
            "span_s": net._osc_s_tab.shape[0],
            "amp_f": net.cfg.osc_amp_f,
            "amp_m": net.cfg.osc_amp_m,
            "amp_s": net.cfg.osc_amp_s,
            "mem_m": net._mem_m,
            "mem_a": net._mem_a,
            "mem_m_seq": net._mem_m_seq,
            "kk": float(k_mem),
            "k_mem": k_mem,
            "a_mem": a_mem,
            "nov": getattr(net, "_metab_nov", net._zero1),
            "none_vf": None,
            "none_l5": None,
            "lam_zero": 0.0,
        }
        for layer in _STP_LAYERS:
            for ch in ("r", "tau", "u"):
                values[f"stp_{ch}_{layer}"] = getattr(net, f"_stp_{ch}_{layer}")
        for stp in ("stp4", "stp2", "stp3", "stp6"):
            values[stp] = None
        for t in range(1, s_len):
            values[f"t_{t}"] = t
        return {name: values[name] for name in graph.inputs}


def defined(graph: Graph) -> frozenset[str]:
    """图中所有已定义名"""
    return defined_names(graph)
