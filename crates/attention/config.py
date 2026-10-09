from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionConfig:
    ema_tau: float = 0.95
    temperature: float = 1.0
    floor: float = 0.05
    top_k: int = 4

    def __post_init__(self) -> None:
        if not 0.0 < self.ema_tau < 1.0:
            raise ValueError("attention ema_tau must be in (0, 1)")
        if self.temperature <= 0.0:
            raise ValueError("attention temperature must be positive")
        if not 0.0 <= self.floor < 1.0:
            raise ValueError("attention floor must be in [0, 1)")
        if self.top_k < 1:
            raise ValueError("attention top_k must be positive")
