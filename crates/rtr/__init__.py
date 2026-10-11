"""
RTR 迹调制的资格可塑性 (trace-modulated eligibility plasticity).

三阶统计量 m = r − V[state] 调制资格迹, 取代无信用分配的共现主项.
"""

from crates.rtr.kernel import elig_accum, rtr_apply, rtr_residual
from crates.rtr.signals import update_survival_signal
from crates.rtr.trace import trace

__all__ = [
    "elig_accum",
    "rtr_apply",
    "rtr_residual",
    "trace",
    "update_survival_signal",
]
