"""A1 结构化预缩放测试: eps_lm_proj 投影前 ÷64 指数移位, rms_norm 后语义零变 (纯 CPU)."""

import torch

from model import CyreneModel, DensePCNet
from model.modulation import rms_norm


def _make_net():
    torch.manual_seed(0)
    cfg = CyreneModel(
        d_l4=64, d_l2=32, d_l3=32, d_l5=64, d_l6=16,
        max_seq_len=16, free_run_window=16, mem_k0=2,
    )
    return DensePCNet(cfg)


def test_rms_norm_scale_invariance_fp16():
    """rms_norm(x @ W) ≡ rms_norm((x/64) @ W): 预缩放被后置归一化解析消去."""
    torch.manual_seed(0)
    x = torch.randn(4, 16, 128).to(torch.float16) * 3.0
    W = torch.randn(128, 64).to(torch.float16) * 0.05
    a = rms_norm(x @ W)
    b = rms_norm((x / 64.0) @ W)
    rel = float((a - b).norm() / (a.norm() + 1e-6))
    assert rel < 1e-2


def test_four_step_learn_all_finite():
    """预缩放后 4 步感知 learn 全 finite (投影链无溢出 → 无 NaN)."""
    net = _make_net()
    for _ in range(4):
        stats = net.learn(torch.randint(32, 256, (1, 8), dtype=torch.long))
        assert torch.isfinite(stats["free_energy"])
    assert torch.isfinite(net._last_z3).all()  # NaN 前哨
    assert torch.isfinite(net.W_lm.data).all()
    assert torch.isfinite(net.W_35.data).all()
