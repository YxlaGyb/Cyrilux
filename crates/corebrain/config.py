from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CoreBrainConfig:
    d_in: int = 1024
    n_slots: int = 32
    top_k: int = 4
    short_frac: float = 0.25

    eta_proj: float = 5.0e-3
    eta_q: float = 5.0e-3
    ema_tau: float = 0.99
    trust_region: float = 0.02

    vicreg_cov: float = 0.01
    std_floor: float = 0.05

    delta_scale: float = 0.05

    attention_ema_tau: float = 0.95
    attention_temperature: float = 1.0
    attention_floor: float = 0.05
