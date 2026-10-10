"""
行索引状态登记表.

按神经元行索引的张量 → (层, 形态). 形态 vec=t[perm] / sq=t[perm][:,perm] / row=只排行.
名录须完整: 漏项会让状态挂到别的神经元上 (test_prune_row_state_audit 校验覆盖度).
"""

from __future__ import annotations

# (层, 主权重, 时间核, Foldiak, 时间核去相关) — l6 无 Foldiak
_LAYERS = (
    ("l4", "W_04", "W_t4", "E_04", "E_t4"),
    ("l2", "W_42", "W_t2", "E_42", "E_t2"),
    ("l3", "W_23", "W_t3", "E_23", "E_t3"),
    ("l5", "W_35", "W_t5", "E_l5", "E_t5"),
    ("l6", "W_56", "W_t6", "", "E_t6"),
)
_BY_LAYER = (  # (名字模板, 形态, 处理者)
    ("{w}", "row", "done"),
    ("{w}_elig", "row", "aux"),
    ("bias_{l}", "vec", "done"),
    ("{t}", "sq", "done"),
    ("{t}_elig", "sq", "aux"),
    ("{e}", "sq", "aux"),
    ("{et}", "sq", "aux"),
    ("_death_row_{l}", "vec", "done"),
    ("_probation_counter_{l}", "vec", "done"),
    ("_S_{l}", "vec", "done"),
    ("_stp_r_{l}", "vec", "aux"),
    ("_stp_tau_{l}", "vec", "aux"),
    ("_stp_u_{l}", "vec", "aux"),
    ("_stp_active_ema_{l}", "vec", "aux"),
)
_L4_ONLY = (  # 行 = L4
    ("b_diff", "vec", "aux"),
    ("E_bind", "sq", "aux"),
    ("_theta_w", "vec", "aux"),
    ("_theta_w04", "vec", "aux"),
    ("_theta_wt4", "vec", "aux"),
    ("W_diff", "sq", "done"),
    ("W_diff_elig", "sq", "aux"),
    ("W_state_pred", "sq", "done"),
    ("W_state_pred_elig", "sq", "aux"),
    ("W_pred_43", "row", "done"),
    ("W_pred_43_elig", "row", "aux"),
    ("W_bind", "row", "done"),
    ("W_bind_elig", "row", "aux"),
)
_L5_ONLY = (
    ("M_l5", "sq", "aux"),
    ("_theta_l5", "vec", "aux"),
    ("W_pred_54", "row", "done"),
    ("W_pred_54_elig", "row", "aux"),
)
_SPECIAL = ("W1", "W1_elig", "_dw_buf", "_mem_m", "_gain_mask", "_gain_l3")
_EXEMPT = (
    "W_bind_self_elig",
    "W_act_elig",
    "W_lm_elig",
    "W_lm_2_elig",
    "_core_brain.projector.P_short",
    "_core_brain.projector.P_long",
    "_core_brain.projector.P_long_ema",
    "_core_brain._delta_prev",
    "_core_brain.predictor.Q",
    "_core_brain._cov_s",
    "_core_brain._cov_l",
    "_core_brain.attention._a_long_ema",
)


def _row_state() -> dict[str, tuple[str, str, str]]:
    t: dict[str, tuple[str, str, str]] = {}
    for lyr, w, wt, e, et in _LAYERS:
        for tmpl, form, handler in _BY_LAYER:
            name = tmpl.format(w=w, t=wt, e=e, et=et, l=lyr)
            if name:
                t[name] = (lyr, form, handler)
    for lyr, spec in (("l4", _L4_ONLY), ("l5", _L5_ONLY)):
        t.update({n: (lyr, f, h) for n, f, h in spec})
    t.update(dict.fromkeys(_SPECIAL, ("l4", "special", "done")))
    t.update(dict.fromkeys(_EXEMPT, ("-", "exempt", "exempt")))
    from crates.model.model_cyrene import _ACTIVE_EMA_LAYER

    t.update({f"_active_ema_{a}": (lyr, "vec", "aux") for a, lyr in _ACTIVE_EMA_LAYER.items()})
    return t


ROW_STATE = _row_state()
