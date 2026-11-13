"""
P3 稀疏码门控测试.

判据: P_l ≡ 1 时 _plastic 对所有调用点逐位恒等; 掩码开启时死/不活跃神经元不得进
任意局部更新权重; 压实后 S_l 物理宽度与 active_size 对齐.
"""

import torch

from crates.hybrid.learning import _common
from crates.model import CyreneModel, HybridPCNet

_LAYERS = ("l4", "l2", "l3", "l5", "l6")


def _make_net(seed: int = 0) -> HybridPCNet:
    torch.manual_seed(seed)
    cfg = CyreneModel(
        d_l4=64,
        d_l2=32,
        d_l3=32,
        d_l5=64,
        d_l6=16,
        max_seq_len=16,
        free_run_window=16,
        mem_k0=2,
    )
    return HybridPCNet(cfg)


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
        assert torch.equal(_common._plastic(post[:, 1:], net, layer, pos=slice(1, None)), post[:, 1:])
        assert torch.equal(_common._plastic(post[:, :-1], net, layer, pos=slice(None, -1)), post[:, :-1])
        mask = torch.tensor([True, False] + [True] * (S - 2))
        sub = post[:, mask]
        assert torch.equal(_common._plastic(sub, net, layer, pos=mask), sub)


def test_dead_units_gated_out_of_plastic_code():
    """S_l 关位后死神经元不进可塑码 P_l, 活神经元仍进."""
    net = _make_net(seed=3)
    net.learn(torch.randint(32, 256, (1, 12), dtype=torch.long))
    net._S_l4[:8] = False
    net.learn(torch.randint(32, 256, (1, 12), dtype=torch.long))
    assert not net._P_l4[..., :8].any(), "死神经元进入可塑码"
    assert net._P_l4[..., 8:].any(), "活神经元被误杀"


def _tiny_prune_net() -> HybridPCNet:
    torch.manual_seed(0)
    cfg = CyreneModel(
        d_l4=512,
        d_l2=192,
        d_l3=192,
        d_l5=192,
        d_l6=128,
        prune_warmup=0,
        prune_fraction=0.6,
        death_probation=1,
        active_size_lower_bound=128,
        l4_lower_bound=512,
        mem_k0=1,
        mem_k_max=1,
        input_history=True,
        max_seq_len=8,
    )
    return HybridPCNet(cfg)


def test_survival_mask_tracks_physical_width_through_pruning():
    """压实 (perm/shrink) 后 _S_l 物理宽度必须与 active_size 对齐, 且前向不崩."""
    net = _tiny_prune_net()
    net.learn(torch.randint(0, 256, (1, 4), dtype=torch.long))
    with torch.no_grad():
        w35 = net.W_35.data.clone()
        a5, n_low = w35.shape[0], 115
        top_a5 = a5 - n_low
        w35[:top_a5] = w35[:top_a5] / (w35[:top_a5].norm(dim=1, keepdim=True) + 1e-6) * 3.0
        w35[top_a5:] = w35[top_a5:] / (w35[top_a5:].norm(dim=1, keepdim=True) + 1e-6) * 1e-6
        net.W_35.copy_(w35.to(torch.float16))
        for name in ("W_04", "W_42", "W_23", "W_56"):
            p = getattr(net, name).data
            p.copy_((p / (p.norm(dim=1, keepdim=True) + 1e-6)).to(torch.float16))

    net.pruner.decide()
    net.pruner.decide()
    net.pruner.compact()
    assert net.active_size["l5"] < 192, "L5 未被剪, 测试前提失效"
    for layer in _LAYERS:
        assert getattr(net, f"_S_{layer}").shape[0] == net.active_size[layer], (
            f"{layer} 压实后 S 宽度与物理宽度错位"
        )
    net.learn(torch.randint(0, 256, (1, 4), dtype=torch.long))
