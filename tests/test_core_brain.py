from __future__ import annotations

from pathlib import Path

import torch

from crates.corebrain import CoreBrain, CoreBrainConfig
from crates.corebrain.project import ste_delta

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
    cb = CoreBrain(CoreBrainConfig(d_in=32, n_slots=16, top_k=8, eta_proj=0.05, eta_q=0.05))
    torch.manual_seed(0)
    base = torch.randn(8, 1, 32, dtype=torch.float16) * 2.0

    def batch() -> torch.Tensor:
        return base + 0.1 * torch.randn(8, 32, 32, dtype=torch.float16)

    first = cb.observe(batch())["xv_energy"]
    last = first
    for _ in range(200):
        last = cb.observe(batch())["xv_energy"]
    assert float(last) < float(first)


def test_default_off_adds_no_parameters():
    from crates.model import CyreneModel, DensePCNet

    cfg = CyreneModel(d_l4=64, d_l2=32, d_l3=32, d_l5=32, d_l6=32, max_seq_len=16)
    net = DensePCNet(cfg)
    assert not any(key.startswith("_core_brain") for key in net.state_dict())
    before = sum(p.numel() for p in net.parameters())
    net.enable_core_brain(n_slots=8, top_k=2)
    assert any(key.startswith("_core_brain.") for key in net.state_dict())
    assert sum(p.numel() for p in net.parameters()) > before
    stats = net.learn(torch.randint(0, 256, (1, 16), dtype=torch.long))
    assert torch.isfinite(stats["core_xv_energy"]).all()


def test_source_carries_no_reference_marker():
    """层源码不得携带被参考外部框架的名称 (本项目只做自研机制)."""
    marker = "je" + "pa"
    root = Path(__file__).resolve().parent.parent / "crates" / "corebrain"
    for path in root.glob("*.py"):
        assert marker not in path.read_text(encoding="utf-8").lower()
