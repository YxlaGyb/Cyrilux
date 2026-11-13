"""
单细胞前向

细胞内宽度: a = sigmoid(x @ W1.T), y = (a * alive) @ W2.T;
W1: (h, d), W2: (n, h), alive: (h,).
"""

from __future__ import annotations

import torch


class Cell:
    """单细胞: 细胞内宽度取代整细胞单阈值, alive 逐隐藏单元门控输出."""

    def __init__(
        self,
        d: int,
        h: int,
        n: int,
        device: torch.device | str | None = None,
        income_floor: float = 0.2,
        death_steps: int = 200,
        income_decay: float = 0.99,
    ) -> None:
        assert h % 128 == 0, "h must be a multiple of 128"
        self.W1 = torch.randn(h, d, dtype=torch.float16, device=device)
        self.W2 = torch.zeros(n, h, dtype=torch.float16, device=device)
        self.alive = torch.ones(h, dtype=torch.float16, device=device)
        self._income = torch.zeros(h, dtype=torch.float16, device=device)
        self._death_cnt = torch.zeros(h, dtype=torch.int16, device=device)
        self.income_floor = income_floor
        self.death_steps = death_steps
        self.income_decay = income_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d) → y: (B, n); 全程 fp16."""
        a = torch.sigmoid(x @ self.W1.T)
        return (a * self.alive) @ self.W2.T

    def update_alive(self, x: torch.Tensor, m: torch.Tensor) -> None:
        """收入 = |m|·Σ|W2|·存活隐藏均值; 连续低于种群均值 income_floor 倍 death_steps 步后永久置 0."""
        a = torch.sigmoid(x @ self.W1.T)
        am = a * self.alive
        income = m.abs() * self.W2.abs().sum(dim=0) * am.mean(dim=0)
        self._income.mul_(self.income_decay).add_((1.0 - self.income_decay) * income)
        low = self._income < self.income_floor * self._income.mean()
        self._death_cnt.copy_(torch.where(low, self._death_cnt + 1, torch.zeros_like(self._death_cnt)))
        self.alive.mul_((self._death_cnt < self.death_steps).to(self.alive.dtype))
