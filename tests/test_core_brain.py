from __future__ import annotations

import copy
from pathlib import Path

import torch

from crates.corebrain import CoreBrain, CoreBrainConfig
from crates.kernels import ste_delta

torch.set_grad_enabled(False)


def test_forward_shape_and_dtype():
    cb = CoreBrain(CoreBrainConfig(d_in=32, n_slots=16, top_k=4))
    z4 = torch.randn(6, 32, 32, dtype=torch.float16)
    delta = cb.forward(z4)
    assert delta.shape == (6, 32)
    assert delta.dtype == torch.float16
    assert torch.isfinite(delta).all()


def test_observe_readouts_finite_and_not_collapsed():
    cb = CoreBrain(CoreBrainConfig(d_in=32, n_slots=16, top_k=4))
    for _ in range(20):
        stats = cb.observe(torch.randn(8, 32, 32, dtype=torch.float16))
    for value in stats.values():
        assert torch.isfinite(value).all()
    assert float(stats["effective_rank"]) > 1.0
    assert 0.0 <= float(stats["slot_overlap"]) <= 1.0


def test_views_full_mask_is_bitwise_unchanged():
    """无填充 ⇒ 掩码均值 ≡ 普通均值: 整窗逐位不变 (含 CUDA 路径)."""
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        torch.manual_seed(0)
        cb = CoreBrain(
            CoreBrainConfig(d_in=32, n_slots=16, top_k=4, window_len=64, short_frac=0.5)
        ).to(device)
        z4 = torch.randn(4, 64, 32, dtype=torch.float16, device=device)
        q = max(1, int(cb.cfg.window_len * cb.cfg.short_frac))
        valid = torch.ones(4, 64, dtype=torch.bool, device=device)
        short, long = cb._views(z4, valid)
        assert torch.equal(short, z4[:, 64 - q:, :].mean(dim=1))
        assert torch.equal(long, z4.mean(dim=1))
        short_none, long_none = cb._views(z4)
        assert torch.equal(short, short_none)
        assert torch.equal(long, long_none)


def test_views_exclude_fill_content():
    """填充位不参与窗内容均值: 改填充字节, 两个视图与窗级 Δ 逐位不变."""
    torch.manual_seed(0)
    cb = CoreBrain(CoreBrainConfig(d_in=32, n_slots=16, top_k=4, window_len=64, short_frac=0.5))
    z4 = torch.randn(3, 64, 32, dtype=torch.float16)
    valid = torch.ones(3, 64, dtype=torch.bool)
    valid[0, 50:] = False
    valid[1, 60:] = False
    filled = z4.clone()
    filled[0, 50:] = torch.randn(14, 32, dtype=torch.float16) * 8.0
    filled[1, 60:] = torch.randn(4, 32, dtype=torch.float16) * 8.0
    short_a, long_a = cb._views(z4, valid)
    short_b, long_b = cb._views(filled, valid)
    assert torch.equal(short_a, short_b)
    assert torch.equal(long_a, long_b)
    other = copy.deepcopy(cb)
    assert torch.equal(cb.forward(z4, valid), other.forward(filled, valid))


def test_short_view_anchors_on_content_not_position():
    """短窗锚在内容末尾: 尾窗左对齐时按位置切会整窗取填充 (短视图 ≡ 0)."""
    torch.manual_seed(0)
    cb = CoreBrain(CoreBrainConfig(d_in=32, n_slots=16, top_k=4, window_len=64, short_frac=0.25))
    q = 16
    z4 = torch.randn(3, 64, 32, dtype=torch.float16)
    valid = torch.ones(3, 64, dtype=torch.bool)
    for i, n in enumerate((64, 32, 8)):
        valid[i, n:] = False
    short, long = cb._views(z4, valid)
    assert torch.equal(short[0], z4[0, 48:64].mean(dim=0))
    assert torch.equal(short[1], z4[1, 16:32].mean(dim=0))
    assert torch.equal(short[2], z4[2, :8].mean(dim=0))
    assert torch.equal(long[1], z4[1, :32].mean(dim=0))
    assert torch.count_nonzero(short[1:]) > 0


def test_empty_window_view_is_zero_and_finite():
    """全填充窗: 两视图 ≡ 0, 不产生 NaN."""
    cb = CoreBrain(CoreBrainConfig(d_in=32, n_slots=16, top_k=4, window_len=64, short_frac=0.25))
    z4 = torch.randn(2, 64, 32, dtype=torch.float16) * 8.0
    valid = torch.zeros(2, 64, dtype=torch.bool)
    short, long = cb._views(z4, valid)
    assert torch.count_nonzero(short) == 0
    assert torch.count_nonzero(long) == 0
    assert torch.isfinite(cb.forward(z4, valid)).all()


def test_core_brain_update_ignores_fill_length():
    """生产前向路径自带填充掩码: 真实内容相同、填充长度不同 → 中心脑权重逐位不变."""
    from crates.model import CyreneModel, HybridPCNet

    torch.manual_seed(7)
    cfg = CyreneModel(d_l4=64, d_l2=32, d_l3=32, d_l5=32, d_l6=32, max_seq_len=32)
    net_a = HybridPCNet(cfg)
    net_b = copy.deepcopy(net_a)
    before = [p.clone() for p in net_a._core_brain.parameters()]
    real = torch.randint(65, 91, (1, 12), dtype=torch.long)
    fill = torch.zeros(1, 1, dtype=torch.long)
    net_a.forward(torch.cat([real, fill.repeat(1, 4)], dim=1))
    net_a._core_brain.learn()
    net_b.forward(torch.cat([real, fill.repeat(1, 8)], dim=1))
    net_b._core_brain.learn()
    changed = False
    for (name_a, p_a), (name_b, p_b), p_0 in zip(
        net_a._core_brain.named_parameters(), net_b._core_brain.named_parameters(), before
    ):
        assert name_a == name_b
        assert torch.equal(p_a, p_b), name_a
        changed = changed or not torch.equal(p_a, p_0)
    assert changed


def test_ste_delta_zeroes_non_winner_columns():
    view = torch.randn(4, 8, dtype=torch.float16)
    err = torch.randn(4, 16, dtype=torch.float16)
    win = torch.zeros(4, 16, dtype=torch.bool)
    win[0, 2] = True
    win[1, 2] = True
    win[3, 7] = True
    delta = ste_delta(view, err, win)
    assert delta[:, 2].abs().sum() > 0
    for slot in range(16):
        if slot not in (2, 7):
            assert delta[:, slot].abs().sum() == 0


def test_cross_view_error_decreases_on_structured_input():
    torch.manual_seed(0)
    cb = CoreBrain(CoreBrainConfig(d_in=32, n_slots=16, top_k=8, eta_proj=0.05, eta_q=0.05))
    base = torch.randn(8, 1, 32, dtype=torch.float16) * 2.0

    def batch() -> torch.Tensor:
        return base + 0.1 * torch.randn(8, 32, 32, dtype=torch.float16)

    first = cb.observe(batch())["xv_energy"]
    last = first
    for _ in range(200):
        last = cb.observe(batch())["xv_energy"]
    assert float(last) < float(first)


def test_core_brain_and_attention_are_mandatory():
    from crates.model import CyreneModel, HybridPCNet

    cfg = CyreneModel(d_l4=64, d_l2=32, d_l3=32, d_l5=32, d_l6=32, max_seq_len=16)
    net = HybridPCNet(cfg)
    assert any(key.startswith("_core_brain.") for key in net.state_dict())
    assert any(key.startswith("_core_brain.attention.") for key in net.state_dict())
    assert net._core_brain.attention is not None
    stats = net.learn(torch.randint(0, 256, (1, 16), dtype=torch.long))
    assert torch.isfinite(stats["core_xv_energy"]).all()
    assert torch.isfinite(stats["attention_gain_mean"]).all()
    assert torch.isfinite(stats["attention_gain_min"]).all()


def test_source_carries_no_reference_marker():
    marker = "je" + "pa"
    root = Path(__file__).resolve().parent.parent / "crates" / "corebrain"
    for path in root.glob("*.py"):
        assert marker not in path.read_text(encoding="utf-8").lower()
