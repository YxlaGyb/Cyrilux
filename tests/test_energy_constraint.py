"""_energy_constraint 纯逻辑测试 (CPU fp16): 延迟初始化 / excess 激活 / 高活动衰减.

第 78 轮内建能量约束: dW ← dW − (α·post² + β·relu(post²−ema)) ⊙ W.
"""

import torch

from model.dense.learning import _energy_constraint


class _StubNet:
    def __init__(self, ema_dim: int):
        self.cfg = type("C", (), {"oja_alpha": 0.05, "oja_elasticity": 0.05})()
        self._act_ema_init: set[str] = set()
        self._act_ema_w35 = torch.zeros(ema_dim, dtype=torch.float16)


def test_first_window_lazy_init_no_excess():
    net = _StubNet(4)
    W = torch.full((4, 3), 0.5, dtype=torch.float16)
    dW = torch.zeros(4, 3, dtype=torch.float16)
    post = torch.tensor([[[2.0, 1.0, 0.5, 0.1]]], dtype=torch.float16)  # [1,1,4]
    out = _energy_constraint(net, W, dW, post, "_act_ema_w35")
    p2 = torch.tensor([4.0, 1.0, 0.25, 0.01], dtype=torch.float16)
    # 首窗: ema ← post², excess = 0 → 只有 Oja 项
    assert torch.allclose(net._act_ema_w35, p2, atol=1e-3)
    assert torch.allclose(out, -(0.05 * p2).unsqueeze(1) * W, atol=1e-3)
    assert "_act_ema_w35" in net._act_ema_init


def test_excess_activates_only_above_ema():
    net = _StubNet(2)
    net._act_ema_w35.copy_(torch.tensor([1.0, 1.0], dtype=torch.float16))
    net._act_ema_init.add("_act_ema_w35")
    W = torch.full((2, 2), 0.5, dtype=torch.float16)
    dW = torch.zeros(2, 2, dtype=torch.float16)
    post = torch.tensor([[[4.0, 1.0]]], dtype=torch.float16)  # 单元0 超基线, 单元1 持平
    out = _energy_constraint(net, W, dW, post, "_act_ema_w35")
    p2 = torch.tensor([16.0, 1.0], dtype=torch.float16)
    excess = torch.tensor([15.0, 0.0], dtype=torch.float16)
    expect = -(0.05 * p2 + 0.05 * excess).unsqueeze(1) * W
    assert torch.allclose(out, expect, atol=1e-2)


def test_high_activity_gets_stronger_decay():
    net = _StubNet(3)
    net._act_ema_w35.copy_(torch.ones(3, dtype=torch.float16))
    net._act_ema_init.add("_act_ema_w35")
    W = torch.full((3, 2), 0.5, dtype=torch.float16)
    dW = torch.ones(3, 2, dtype=torch.float16)
    out_hi = _energy_constraint(
        net, W, dW, torch.tensor([[[3.0, 3.0, 3.0]]], dtype=torch.float16), "_act_ema_w35"
    )
    out_lo = _energy_constraint(
        net, W, dW, torch.tensor([[[1.0, 1.0, 1.0]]], dtype=torch.float16), "_act_ema_w35"
    )
    assert out_hi.shape == (3, 2) and out_lo.shape == (3, 2)
    assert out_hi.abs().min() < out_lo.abs().min()  # 高活动单元被减更多
