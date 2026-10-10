"""
kernels Cyrilux 的叶层算子库
"""

from crates.kernels.activation import cubic, saturate
from crates.kernels.estimator import ste_delta
from crates.kernels.modulation import (
    combine_modulation,
    compute_ach,
    compute_dopamine,
    compute_free_energy,
    compute_uncertainty,
    gate_psay,
)
from crates.kernels.normalize import l2_norm, rel_norm, rms_norm, soft_norm_preserve
from crates.kernels.plasticity import (
    activity_baseline,
    elig_accum,
    energy_constraint,
    rho_control,
)
from crates.kernels.regularize import (
    foldiak_decorrelate,
    participation_ratio,
    variance_floor,
)
from crates.kernels.sparse import topk_wta
from crates.kernels.spectral import decorr_W, spectral_radius_guard

__all__ = [
    "activity_baseline",
    "combine_modulation",
    "compute_ach",
    "compute_dopamine",
    "compute_free_energy",
    "compute_uncertainty",
    "cubic",
    "decorr_W",
    "elig_accum",
    "energy_constraint",
    "foldiak_decorrelate",
    "gate_psay",
    "l2_norm",
    "participation_ratio",
    "rel_norm",
    "rho_control",
    "rms_norm",
    "saturate",
    "soft_norm_preserve",
    "spectral_radius_guard",
    "ste_delta",
    "topk_wta",
    "variance_floor",
]
