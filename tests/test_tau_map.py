"""P3-a τ 有界映射测试: 软带范围/饥荒近下限/冷启动中点/无 clamp 源 (纯 CPU 逻辑)."""

import inspect

import torch

from model import CyreneModel, DensePCNet
from model.dense.learning.action import ActionMixin

TAU_LO = 0.9
TAU_HI = 2.5


def _make_net():
    torch.manual_seed(0)
    cfg = CyreneModel(
        d_l4=64, d_l2=32, d_l3=32, d_l5=64, d_l6=16,
        max_seq_len=16, free_run_window=16, mem_k0=2,
    )
    return DensePCNet(cfg)


def _echo(net):
    net._echo_seed = torch.zeros(1, 1, dtype=torch.long)
    net.learn(None, free_run=False)


def test_tau_bounded_and_finite():
    net = _make_net()
    for _ in range(3):
        net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))
        _echo(net)
    t = float(net._gen_temp.item())
    assert torch.isfinite(net._gen_temp)
    assert TAU_LO - 1e-3 <= t <= TAU_HI + 1e-3  # 软带由映射余域给出, 无 clamp 也守住


def test_tau_forced_famine_near_floor():
    """强制深度饥荒 (应激≈1) + 差分压底 → τ 逼近软带下限但不穿."""
    net = _make_net()
    net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))
    _echo(net)  # 建立锚/低通
    net._metab_E.fill_(-5.0)
    net._metab_E_ref.fill_(-0.05)
    net._metab_tau_d.fill_(-1.0)
    _echo(net)
    st = float(getattr(net, "_metab_stress", torch.tensor(0.0)).item())
    assert st > 0.99  # 应激饱和
    t = float(net._gen_temp.item())
    assert TAU_LO - 1e-3 <= t <= 1.0  # 近下限, 不穿


def test_tau_cold_start_midpoint():
    """从未感知直接回声: 锚缺失 → 映射中点 (4.0 硬退场)."""
    net = _make_net()
    _echo(net)
    t = float(net._gen_temp.item())
    assert abs(t - (TAU_LO + TAU_HI) / 2.0) < 1e-3


def test_tau_no_clamp_source():
    src = inspect.getsource(ActionMixin._update_w_act)
    assert "minimum(" not in src
    assert "maximum(" not in src
