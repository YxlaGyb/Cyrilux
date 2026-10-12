"""
RTR 叶层算子测试 (crates/rtr).

覆盖 资格迹累加 / 逐状态价值残差 / 迹调制更新 / 生存信号写点.
"""

from types import SimpleNamespace

import torch

from crates.model.constants import ELIG_GAMMA, RTR_ETA_V
from crates.rtr import elig_accum, rtr_apply, rtr_residual, trace, update_survival_signal


def test_elig_accum_is_exponential_accumulation():
    E = torch.zeros(3, 4, dtype=torch.float16)
    dW = torch.ones(3, 4, dtype=torch.float16)
    out = elig_accum(E, dW, 0.5)
    assert torch.allclose(out, torch.ones(3, 4, dtype=torch.float16), atol=1e-3)
    assert torch.allclose(E, torch.ones(3, 4, dtype=torch.float16), atol=1e-3)  # 缓冲就地推进
    out2 = elig_accum(E, dW, 0.5)
    assert torch.allclose(out2, torch.full((3, 4), 1.5, dtype=torch.float16), atol=1e-3)


def test_elig_accum_front_aligns_pruned_slice():
    # 缓冲随修剪存活 (全形), dW 只覆盖前缘活性切片 → 返回前缘视图, 尾块保留在缓冲里
    E = torch.zeros(6, 4, dtype=torch.float16)
    E[4:, :] = 9.0
    dW = torch.ones(4, 4, dtype=torch.float16)
    out = elig_accum(E, dW, 0.0)
    assert out.shape == (4, 4)
    assert torch.allclose(out, torch.ones(4, 4, dtype=torch.float16), atol=1e-3)
    assert torch.allclose(E[4:], torch.full((2, 4), 9.0, dtype=torch.float16), atol=1e-3)


def test_rtr_residual_tracks_value_in_place():
    w = torch.zeros(4, dtype=torch.float16)
    z = torch.ones(4, dtype=torch.float16)  # ẑ = z/‖z‖ = 0.5
    r = torch.tensor(1.0, dtype=torch.float16)
    m = rtr_residual(w, z, r, 0.5)
    assert m.shape == (1,)
    assert torch.allclose(m, torch.ones(1, dtype=torch.float16), atol=1e-3)  # 冷启动 w=0 → m = r
    assert torch.allclose(w, torch.full((4,), 0.25, dtype=torch.float16), atol=1e-3)
    m2 = rtr_residual(w, z, r, 0.5)
    assert torch.allclose(m2, torch.full((1,), 0.5, dtype=torch.float16), atol=1e-3)  # 残差递减


def test_rtr_residual_covers_nearby_states():
    # 无桶边界: 账本落在权向量上, 邻近状态与已记账状态共享同一批分量
    torch.manual_seed(0)
    z = torch.randn(16, dtype=torch.float16)
    z_near = z + 0.05 * torch.randn(16, dtype=torch.float16)
    r = torch.tensor(1.0, dtype=torch.float16)
    cold = torch.zeros(16, dtype=torch.float16)
    assert float(rtr_residual(cold, z_near, r, 0.2)[0]) > 0.9  # 对照: 空账本无覆盖

    w = torch.zeros(16, dtype=torch.float16)
    for _ in range(20):
        rtr_residual(w, z, r, 0.2)
    assert float(rtr_residual(w, z_near, r, 0.2)[0]) < 0.2  # 已记账状态的邻近状态被覆盖


def test_rtr_apply_is_self_normalised():
    E = torch.full((4, 5), 2.0, dtype=torch.float16)
    m_zero = torch.zeros(1, dtype=torch.float16)
    assert rtr_apply(E, m_zero).abs().sum() == 0  # 零残差 → 零更新
    m_pos = torch.ones(1, dtype=torch.float16)
    m_neg = -m_pos
    up = rtr_apply(E, m_pos)
    down = rtr_apply(E, m_neg)
    assert torch.allclose(up, -down, atol=1e-3)  # 符号随残差翻转
    assert up.shape == E.shape
    assert torch.allclose(up.norm(), torch.ones((), dtype=torch.float16), atol=1e-2)  # 逐域自归一


def test_trace_reads_and_advances_named_buffer():
    dW = torch.ones(3, 3, dtype=torch.float16)
    net = SimpleNamespace(W_x_elig=torch.zeros(3, 3, dtype=torch.float16))
    E = trace(net, "W_x", dW)
    assert torch.allclose(E, dW, atol=1e-3)
    assert torch.allclose(net.W_x_elig, dW, atol=1e-3)  # 命名缓冲就地累加
    trace(net, "W_x", dW)
    assert torch.allclose(net.W_x_elig, dW * (1.0 + ELIG_GAMMA), atol=1e-2)


def _signal_stub(echo_loop: bool, d_l4: int = 8, channel: int = 3):
    z4 = torch.zeros(2, 1, d_l4, dtype=torch.float16)
    z4[:, :, channel] = 1.0
    net = SimpleNamespace(
        _z4=z4,
        _rtr_w=torch.zeros(d_l4, 2, dtype=torch.float16),
        _metab_R=torch.tensor(1.0, dtype=torch.float16),
    )
    return net, SimpleNamespace(echo_loop=echo_loop)


def test_survival_signal_writes_phase_column():
    net, ctx = _signal_stub(echo_loop=False)
    update_survival_signal(net, ctx)
    assert net._survival_signal.shape == (1,)
    assert int(net._survival_signal[0]) == 1  # 冷启动 w=0 → m = r
    assert abs(float(net._rtr_w[3, 0]) - RTR_ETA_V) < 1e-4  # 感知列被跟踪
    assert net._rtr_w[:, 1].abs().sum() == 0  # 回声列未触及

    net2, ctx2 = _signal_stub(echo_loop=True)
    update_survival_signal(net2, ctx2)
    assert abs(float(net2._rtr_w[3, 1]) - RTR_ETA_V) < 1e-4
    assert net2._rtr_w[:, 0].abs().sum() == 0


def test_survival_signal_columns_are_independent():
    # 双轨互不干扰: 同状态在两相各自累积, 残差独立
    net, perc_ctx = _signal_stub(echo_loop=False)
    update_survival_signal(net, perc_ctx)
    perc_track = net._rtr_w[3, 0].clone()
    update_survival_signal(net, SimpleNamespace(echo_loop=True))
    assert net._rtr_w[3, 0] == perc_track
    assert net._rtr_w[3, 1] > 0
