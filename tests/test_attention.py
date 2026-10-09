from __future__ import annotations

import pytest
import torch
from click.testing import CliRunner

from crates.attention import AttentionConfig, ErrorAttention
from crates.cli.commands.train_cmd import train

torch.set_grad_enabled(False)


def test_gain_mean_floor_and_shape():
    attn = ErrorAttention(
        8,
        AttentionConfig(ema_tau=0.95, temperature=1.0, floor=0.1, top_k=2),
    )
    a_short = torch.randn(4, 8, dtype=torch.float16)
    a_long = torch.randn(4, 8, dtype=torch.float16) * 0.5
    delta = torch.randn(4, 8, dtype=torch.float16)
    out = attn(a_short, a_long, delta)

    assert out.gain.shape == (4, 8)
    assert out.effective_error.shape == delta.shape
    assert out.effective_error.dtype == torch.float16
    assert torch.allclose(
        out.gain.mean(dim=-1),
        torch.ones(4, dtype=torch.float16),
        atol=2e-2,
    )
    assert bool((out.gain >= 0.1 - 1e-3).all())


def test_ema_tracks_batch_mean():
    attn = ErrorAttention(
        4,
        AttentionConfig(ema_tau=0.5, temperature=1.0, floor=0.0, top_k=2),
    )
    a_short = torch.zeros(2, 4, dtype=torch.float16)
    a_long = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float16)
    delta = torch.zeros_like(a_short)

    attn(a_short, a_long, delta)

    expected = torch.tensor([1.5, 2.0, 2.5, 3.0], dtype=torch.float16)
    assert torch.allclose(attn._a_long_ema, expected, atol=1e-2)


def test_top_down_is_nonnegative_and_sparse():
    attn = ErrorAttention(
        8,
        AttentionConfig(ema_tau=0.95, temperature=1.0, floor=0.05, top_k=2),
    )
    a_short = torch.zeros(1, 8, dtype=torch.float16)
    a_long = torch.tensor([[-1.0, -4.0, 2.0, 3.0, -2.0, 1.0, 0.5, -0.25]], dtype=torch.float16)
    delta = torch.zeros_like(a_short)

    out = attn(a_short, a_long, delta)

    assert bool((out.top_down >= 0).all())
    assert int((out.top_down > 0).sum()) == 2


def test_module_has_no_trainable_parameters():
    attn = ErrorAttention(8, AttentionConfig(top_k=2))
    assert list(attn.parameters()) == []
    assert "_a_long_ema" in attn.state_dict()


@pytest.mark.parametrize(
    "kwargs",
    (
        {"ema_tau": 0.0},
        {"ema_tau": 1.0},
        {"temperature": 0.0},
        {"floor": -0.1},
        {"floor": 1.0},
        {"top_k": 0},
    ),
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        AttentionConfig(**kwargs)


def test_cli_has_no_core_brain_switch():
    result = CliRunner().invoke(train, ["--help"])
    assert result.exit_code == 0
    assert "--core-brain" not in result.output
