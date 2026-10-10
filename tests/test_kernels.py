"""
kernels 叶层算子纯逻辑测试

覆盖激活 / 归一化 / 稀疏 / 直通估计 / 正则 / 塑性 / 谱 / 调制各算子.
"""

import torch

from crates.kernels import (
    activity_baseline,
    combine_modulation,
    compute_ach,
    compute_dopamine,
    compute_free_energy,
    compute_uncertainty,
    cubic,
    decorr_W,
    elig_accum,
    energy_constraint,
    foldiak_decorrelate,
    l2_norm,
    participation_ratio,
    rel_norm,
    rho_control,
    rms_norm,
    saturate,
    soft_norm_preserve,
    spectral_radius_guard,
    ste_delta,
    topk_wta,
    variance_floor,
)

# ---------- 激活 ----------

def test_saturate_bounds():
    x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0], dtype=torch.float16)
    y = saturate(x)
    assert torch.allclose(y, x / (1.0 + x.abs()), atol=1e-2)
    assert y.abs().max() < 1.0
    assert y[2] == 0.0


def test_cubic_polynomial():
    x = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float16)
    y = cubic(x)
    assert y[1] == 0.0
    assert torch.allclose(y, x * (1.0 - 0.5 * x.pow(2)), atol=1e-3)


# ---------- 归一化 ----------

def test_rms_norm_zero_vector_guard():
    z = torch.zeros(2, 4, dtype=torch.float16)
    out = rms_norm(z)
    assert torch.isfinite(out).all()
    assert out.abs().sum() == 0.0


def test_l2_norm_unit_norm():
    x = torch.tensor([[3.0, 4.0]], dtype=torch.float16)
    out = l2_norm(x)
    assert torch.isfinite(out).all()
    assert out.norm().item() < 1.1


def test_l2_norm_zero_vector_guard():
    z = torch.zeros(1, 3, dtype=torch.float16)
    assert l2_norm(z).abs().sum() == 0.0


def test_rel_norm_range():
    assert rel_norm(0.0, 1.0) == 0.0
    assert 0.0 <= rel_norm(5.0, 1e-3) < 1.0
    assert rel_norm(1.0, 1e-3) > 0.9


def test_soft_norm_preserve_bounds_rows():
    W = torch.randn(8, 16, dtype=torch.float16)
    W.mul_(3.0)
    soft_norm_preserve(W)
    rn = W.norm(dim=1)
    assert torch.isfinite(rn).all()
    assert rn.max().item() < 1.25


# ---------- 稀疏 ----------

def test_topk_wta_keeps_k_per_row():
    x = torch.tensor([[0.1, 0.9, 0.4, 0.7, 0.2]], dtype=torch.float16)
    a, mask = topk_wta(x, 2)
    assert int(mask.sum().item()) == 2
    assert int((a != 0).sum().item()) == 2


def test_topk_wta_k_ge_dim_identity():
    x = torch.randn(2, 3, dtype=torch.float16)
    a, mask = topk_wta(x, 3)
    assert bool(mask.all())
    assert torch.allclose(a, x)


# ---------- 直通估计 ----------

def test_ste_delta_zeroes_non_winner_columns():
    view = torch.randn(4, 8, dtype=torch.float16)
    err = torch.randn(4, 16, dtype=torch.float16)
    win = torch.zeros(4, 16, dtype=torch.bool)
    win[0, 2] = True
    win[3, 7] = True
    delta = ste_delta(view, err, win)
    assert delta.shape == (8, 16)
    for slot in range(16):
        if slot not in (2, 7):
            assert delta[:, slot].abs().sum() == 0.0


# ---------- 正则 ----------

def test_foldiak_single_sample_skip():
    a = torch.randn(1, 4, dtype=torch.float16)
    cov = torch.zeros(4, 4, dtype=torch.float16)
    out = foldiak_decorrelate(a, cov, 0.1)
    assert torch.allclose(out, a)


def test_foldiak_reduces_offdiagonal():
    torch.manual_seed(0)
    a = torch.randn(32, 4, dtype=torch.float16)
    cov = torch.zeros(4, 4, dtype=torch.float16)
    out = foldiak_decorrelate(a, cov, 0.2)
    assert out.shape == a.shape
    assert torch.isfinite(out).all()


def test_variance_floor_gain_at_least_one():
    a = torch.randn(16, 4, dtype=torch.float16)
    a[:, 0] = 0.0
    out = variance_floor(a, 0.1)
    assert out.shape == a.shape
    assert out[:, 0].abs().sum() == 0.0


def test_participation_ratio_bounds():
    a = torch.randn(32, 8, dtype=torch.float16)
    pr = participation_ratio(a)
    assert 1.0 <= float(pr) <= a.shape[0] + 1.0


# ---------- 塑性 ----------

def test_activity_baseline_lazy_init_then_excess():
    post = torch.tensor([[[2.0, 1.0]]], dtype=torch.float16)
    ema = torch.zeros(2, dtype=torch.float16)
    p2, excess = activity_baseline(post, ema, initialized=False)
    assert torch.allclose(ema, p2, atol=1e-3)
    assert excess.abs().sum() == 0.0
    post2 = torch.tensor([[[4.0, 1.0]]], dtype=torch.float16)
    p2b, ex2 = activity_baseline(post2, ema, initialized=True)
    assert ex2[0] > 0.0 and ex2[1] == 0.0


def test_energy_constraint_matches_formula():
    W = torch.full((2, 3), 0.5, dtype=torch.float16)
    dW = torch.zeros(2, 3, dtype=torch.float16)
    ema = torch.ones(2, dtype=torch.float16)
    post = torch.tensor([[[4.0, 1.0]]], dtype=torch.float16)
    out = energy_constraint(W, dW, post, ema, True, 0.05, 0.05)
    p2 = torch.tensor([16.0, 1.0], dtype=torch.float16)
    excess = torch.tensor([15.0, 0.0], dtype=torch.float16)
    expect = -(0.05 * p2 + 0.05 * excess).unsqueeze(1) * W
    assert torch.allclose(out, expect, atol=1e-2)


def test_elig_accum_trace():
    E = torch.zeros(2, 3, dtype=torch.float16)
    dW = torch.ones(2, 3, dtype=torch.float16)
    out = elig_accum(E, dW, 0.9)
    assert torch.allclose(out, torch.ones(2, 3, dtype=torch.float16), atol=1e-3)
    first = float(out[0, 0])
    out2 = elig_accum(E, dW, 0.9)
    assert float(out2[0, 0]) > first


def test_rho_control_disabled_passthrough():
    dW = torch.randn(3, 3, dtype=torch.float16)
    W_ref = torch.randn(3, 3, dtype=torch.float16)
    out, rho, s = rho_control(dW, W_ref, False)
    assert rho is None and s is None
    assert torch.allclose(out, dW)


def test_rho_control_scales():
    dW = torch.randn(3, 3, dtype=torch.float16)
    W_ref = torch.randn(3, 3, dtype=torch.float16)
    out, rho, s = rho_control(dW, W_ref, True)
    assert rho is not None and s is not None
    assert 0.005 <= float(s) <= 1.0
    assert out.shape == dW.shape


# ---------- 谱 ----------

def test_spectral_radius_guard_observes_only():
    W = torch.eye(4, dtype=torch.float16) * 2.0
    before = W.clone()
    rho = spectral_radius_guard(W, rescale=False, bound=1.5)
    assert float(rho) > 1.0
    assert torch.allclose(W, before)


def test_spectral_radius_guard_rescales():
    W = torch.eye(4, dtype=torch.float16) * 2.0
    before = W.norm().item()
    spectral_radius_guard(W, rescale=True, bound=1.5)
    assert W.norm().item() < before


def test_decorr_W_shapes_and_updates_E():
    torch.manual_seed(0)
    W = torch.randn(4, 5, dtype=torch.float16)
    E = torch.zeros(4, 4, dtype=torch.float16)
    dW = decorr_W(W, E, coef=1.0, max_delta_ratio=0.05, learn_boost=1.0)
    assert dW.shape == W.shape
    assert E.abs().sum() > 0.0


# ---------- 调制 ----------

def test_dopamine_sign():
    assert compute_dopamine(2.0, 1.0) > 0.5  # F 上升
    assert compute_dopamine(1.0, 2.0) < 0.5  # F 下降
    assert compute_dopamine(1.0, 1.0) == 0.5


def test_dopamine_nonfinite_returns_half():
    assert compute_dopamine(float("nan"), 1.0) == 0.5


def test_ach_and_combine():
    assert compute_ach(0.0) == 0.5
    assert compute_ach(-1.0) > 0.5
    assert combine_modulation(1.0, 1.0) == 1.0
    assert combine_modulation(0.0, 0.0) == 0.0


def test_compute_free_energy_sum_of_means():
    eps = [torch.full((1, 2), 2.0, dtype=torch.float16), torch.ones(1, 2, dtype=torch.float16)]
    assert torch.allclose(compute_free_energy(eps), torch.tensor(5.0, dtype=torch.float16))


def test_compute_uncertainty_short_history():
    assert compute_uncertainty([1.0, 2.0]) == 0.5


def test_compute_uncertainty_zero_mean():
    assert compute_uncertainty([0.0, 0.0, 0.0, 0.0]) == 0.0
