"""exp127: compact() 必须同步迁移扩 h 通道 (W1_new / B_new).

回归背景: topology.compact 在 L4 收缩时按 [z4 | bind | 单元×K] 布局 gather W1,
但迁移扩 h 通道的 W1_new (行 = W1 行) 与 B_new (列 = W1 行) 此前漏同步 →
与 W1 布局错位 (静默语义错位, 形状自洽不产 NaN)。本测试在 _migrate_expand=True
下触发 L4 收缩, 断言 W1_new 行 / B_new 列与 W1 行逐位对齐。

全部 CPU / fp16 / deterministic, 与 test_prune_l5 同口径。
"""

from __future__ import annotations

import torch

from crates.model import CyreneModel, HybridPCNet

D_INPUT = 256


def _tiny_migrate_cfg() -> CyreneModel:
    return CyreneModel(
        d_l4=768, d_l2=128, d_l3=128, d_l5=128, d_l6=128,
        prune_warmup=0, prune_fraction=0.05, death_probation=1,
        active_size_lower_bound=128, l4_lower_bound=512,
        mem_k0=3, mem_k_max=3, max_seq_len=8,
    )


def _decide_until_dead(net, rounds: int = 2) -> None:
    for _ in range(rounds):
        net.pruner.decide()


def test_l4_shrink_migrate_channel_aligned():
    """L4 收缩时 W1_new 行 / B_new 列 必须按 W1 布局 gather, 与 W1 逐位对齐。"""
    cfg = _tiny_migrate_cfg()
    net = HybridPCNet(cfg)
    net.init_migrate_expand()  # 加 W1_new / W_lm_new / B_new; _migrate_expand=True
    assert getattr(net, "_migrate_expand", False) is True
    n_bind = net.bind_slot_dim  # 32
    K = net._mem_m.shape[0]  # 3
    a4_full = net.active_size["l4"]  # 768
    head = a4_full + n_bind

    with torch.no_grad():
        # 身份标记: 第 i 行 (W1/W1_new) 或第 i 列 (B_new) 在 [:,0] / [0,:] 写入 i
        w1 = net.W1.data
        w1n = net.W1_new.data
        bn = net.B_new.data
        w1.zero_(); w1n.zero_(); bn.zero_()
        for i in range(a4_full):
            w1[i, 0] = i
            w1n[i, 0] = i
            bn[0, i] = i
        # W_04 行范数严格递减 → 最弱尾行触发 L4 排名淘汰
        r = (torch.arange(a4_full, dtype=torch.float16).flip(0) + 1).unsqueeze(1) / a4_full
        net.W_04.data.copy_(r.expand(-1, net.W_04.shape[1]).to(torch.float16))

    # 死亡回合同序: mem 饥荒击杀 (K 3→2) → 拓扑修剪
    net._mem_g.copy_(torch.tensor([0.9, 0.5, 0.1], dtype=torch.float16))
    net.mem_pool.famine_kill()
    assert net._mem_m.shape[0] == K - 1
    _decide_until_dead(net)
    s_l4_pre = net._S_l4.detach().clone()
    try:
        net.pruner.compact()
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"compact 抛异常 {type(e).__name__}: {e}")

    a4 = net.active_size["l4"]
    assert a4 < a4_full and a4 >= 512, f"L4 未剪/触底异常 a4={a4}"

    e_w1 = net.W1[:, 0]
    e_w1n = net.W1_new[:, 0]
    e_bn = net.B_new[0, :]

    # 形状同步: W1_new 行数 = W1 行数; B_new 列数 = W1 行数
    assert net.W1_new.shape[0] == net.W1.shape[0], (
        f"W1_new 行数 {net.W1_new.shape[0]} ≠ W1 {net.W1.shape[0]}"
    )
    assert net.B_new.shape[1] == net.W1.shape[0], (
        f"B_new 列数 {net.B_new.shape[1]} ≠ W1 行数 {net.W1.shape[0]}"
    )

    # 逐位对齐: W1_new 行 / B_new 列 与 W1 行同一 gather
    assert torch.equal(e_w1n, e_w1), "W1_new 行与 W1 行错位 (gather 不一致)"
    assert torch.equal(e_bn, e_w1), "B_new 列与 W1 行错位 (gather 不一致)"

    # 正确性断言 (审查修正3): 仅 torch.equal(e_w1n, e_w1) 证"同步", 不证"选对行".
    # 身份标记使第 i 行在 [:,0] = i; compact 把存活行 (_S_l4==1) gather 到 W1 头部 [:a4].
    # 保留行必须恰好等于压缩前 _S_l4 标记存活的原始行索引 (而非错位/任意行).
    kept_idx = net.W1[:a4, 0].round().long()
    live_idx = (s_l4_pre.flatten() > 0.5).nonzero().flatten()
    assert live_idx.numel() >= a4, f"存活行数 {live_idx.numel()} < 压缩后 a4 {a4}"
    # 物理压实按 8 对齐 floor (n_alive=(n_live//8)*8), 保留 = 存活行前段 live_idx[:a4]
    assert torch.equal(kept_idx.sort().values, live_idx[:a4].sort().values), (
        f"保留行非存活行前段 (gather 选错行): kept={kept_idx.tolist()} live={live_idx[:a4].tolist()}")

    # W_lm_new 两维均固定 (d_h / d_input), 收缩不受影响
    assert tuple(net.W_lm_new.shape) == (net.d_h, net.cfg.d_input), (
        f"W_lm_new 形状异常 {tuple(net.W_lm_new.shape)}"
    )

    # 收缩后 forward 有限 (迁移通道参与 logits, 错位会静默污染)
    out = net.forward(torch.randint(0, D_INPUT, (1, 8), dtype=torch.long))
    mu = out["mu_diff"]
    if isinstance(mu, torch.Tensor):
        assert not torch.isnan(mu).any(), "forward mu_diff NaN (迁移通道错位)"


def test_compact_no_migrate_untouched():
    """对照: 无迁移通道时 compact 不得触碰任何 _migrate 属性 (存在性保持不变)。"""
    cfg = _tiny_migrate_cfg()
    net = HybridPCNet(cfg)
    assert not getattr(net, "_migrate_expand", False)
    net._mem_g.copy_(torch.tensor([0.9, 0.5, 0.1], dtype=torch.float16))
    net.mem_pool.famine_kill()
    _decide_until_dead(net)
    net.pruner.compact()
    # 默认模型无迁移通道参数, 不应被 compact 意外创建
    assert not hasattr(net, "W1_new")
    assert not hasattr(net, "B_new")
