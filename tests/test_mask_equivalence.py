"""
P3 稀疏码门控逐位等价测试.

判据: P_l ≡ 1 (S 全存活 + 竞争退化全通过) 时, 学习端 dW⊙P_l 必须是恒等 —
全管线与"关闭掩码"路径逐位一致; 掩码开启时死/不活跃神经元不得进任意 Hebbian 权重.
"""

import torch

from crates.hybrid import forward as fwd
from crates.hybrid.learning import _common
from crates.hybrid.learning import bind as bindmod
from crates.hybrid.learning import engine as engmod
from crates.hybrid.learning import feedforward as ffmod
from crates.hybrid.learning import predict as predmod
from crates.hybrid.learning import temporal as tempmod
from crates.model import CyreneModel, HybridPCNet

_PLASTIC_MODULES = (engmod, ffmod, predmod, tempmod, bindmod)
_LAYERS = ("l4", "l2", "l3", "l5", "l6")


def _make_net(seed: int = 0) -> HybridPCNet:
    torch.manual_seed(seed)
    cfg = CyreneModel(
        d_l4=64, d_l2=32, d_l3=32, d_l5=64, d_l6=16,
        max_seq_len=16, free_run_window=16, mem_k0=2,
    )
    return HybridPCNet(cfg)


def _full_compete(h, S, k):
    """竞争退化: 全通过 (等价掩码关闭, S 全一时与原 topk 恒等路径对齐)."""
    return h, torch.ones_like(h, dtype=torch.bool)


def _inputs(n: int, seed: int = 123):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(32, 256, (1, 12), generator=g, dtype=torch.long).clone() for _ in range(n)]


def _snap_params(net: HybridPCNet):
    return {n: p.data.clone() for n, p in net.named_parameters()}


def test_plastic_identity_when_mask_all_true():
    """_plastic 在 P_l ≡ 1 时对所有调用点形状逐位恒等 (P_l 形状 [N,S,dim])."""
    net = _make_net()
    net.learn(torch.randint(32, 256, (1, 12), dtype=torch.long))
    for layer in _LAYERS:
        P = getattr(net, f"_P_{layer}")
        assert P.dim() == 3, f"{layer} P_l 形状应 [N,S,dim], 实为 {tuple(P.shape)}"
        P.fill_(True)
        a = net.active_size[layer]
        S = P.shape[1]
        post = torch.randn(2, S, a, dtype=torch.float16)
        assert torch.equal(_common._plastic(post, net, layer), post)
        assert torch.equal(
            _common._plastic(post[:, 1:], net, layer, pos=slice(1, None)), post[:, 1:]
        )
        assert torch.equal(
            _common._plastic(post[:, :-1], net, layer, pos=slice(None, -1)), post[:, :-1]
        )
        mask = torch.tensor([True, False] + [True] * (S - 2))
        sub = post[:, mask]
        assert torch.equal(_common._plastic(sub, net, layer, pos=mask), sub)


def test_pipeline_bitwise_equivalence_when_mask_all_true(monkeypatch):
    """P_l ≡ 1 时全管道 30 步与关闭门控逐位一致 (逐位锚)."""
    monkeypatch.setattr(fwd, "compete", _full_compete)
    xin = _inputs(30)

    net_ref = _make_net(seed=7)
    torch.manual_seed(123)
    for x in xin:
        net_ref.learn(x)
    ref = _snap_params(net_ref)

    def _identity(post, net, layer, pos=None):
        return post

    for m in _PLASTIC_MODULES:
        monkeypatch.setattr(m, "_plastic", _identity)

    net_new = _make_net(seed=7)
    torch.manual_seed(123)
    for x in xin:
        net_new.learn(x)
    for n, p in net_new.named_parameters():
        assert torch.equal(p.data, ref[n]), f"参数 {n} 在 P≡1 下与关闭门控不一致"


def test_mask_on_changes_trajectory_and_is_finite(monkeypatch):
    """掩码真开 (真竞争): 轨迹与关闭不同, 且全程有限 (无 NaN/inf)."""
    xin = _inputs(20, seed=55)

    net_on = _make_net(seed=9)
    torch.manual_seed(321)
    for x in xin:
        net_on.learn(x)
        assert torch.isfinite(net_on._z4).all()
    on = _snap_params(net_on)

    monkeypatch.setattr(fwd, "compete", _full_compete)
    net_off = _make_net(seed=9)
    torch.manual_seed(321)
    for x in xin:
        net_off.learn(x)
    off = _snap_params(net_off)

    assert any(not torch.equal(on[n], off[n]) for n in on), "掩码开启却与关闭逐位一致 (空测试)"


def test_dead_units_gated_out_of_plastic_code():
    """S_l 关位后死神经元不进可塑码 P_l, 活神经元仍进."""
    net = _make_net(seed=3)
    net.learn(torch.randint(32, 256, (1, 12), dtype=torch.long))
    net._S_l4[:8] = False
    net.learn(torch.randint(32, 256, (1, 12), dtype=torch.long))
    assert not net._P_l4[..., :8].any(), "死神经元进入可塑码"
    assert net._P_l4[..., 8:].any(), "活神经元被误杀"
