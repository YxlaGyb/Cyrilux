"""
记忆单元法庭契约测试: 收入(生存货币) + 持有费(时间租金), 纯 CPU.

信用 = |m|·cos(单元内容, 账本状态方向 ẑ), m = _survival_signal (生存账本残差);
增益 = (1−d)g(1−d·s) + η·q, s = log-α 谱上的归一化持久度 ∈ [0,1] (慢单元租金 ×2).

判庭入参 (ctx/sh) 由替身给出: update 只读 ctx.free_run / ctx.echo_world_frozen /
sh.lm.{e_h, eps_total}; _z4 与 _survival_signal 在前向里是瞬态, 测试自备.
"""

from types import SimpleNamespace

import torch

from crates.model import CyreneModel, HybridPCNet


def _tiny_cfg(**over) -> CyreneModel:
    kw = {
        "d_l4": 64, "d_l2": 32, "d_l3": 32, "d_l5": 64, "d_l6": 16,
        "max_seq_len": 16, "free_run_window": 16, "mem_k0": 2, "mem_k_max": 2,
    }
    kw.update(over)
    return CyreneModel(**kw)


def _stub(e_h: float = 1.0, eps: float = 0.5):
    ctx = SimpleNamespace(free_run=False, echo_world_frozen=False)
    sh = SimpleNamespace(
        lm=SimpleNamespace(
            e_h=torch.full((1, 8, 1), e_h, dtype=torch.float16),
            eps_total=torch.full((1,), eps, dtype=torch.float16),
        )
    )
    return ctx, sh


def _seed(net, m: float) -> None:
    """手置 ẑ = e0 (池化 z4 第 0 通道) 与生存货币; 单元内容由各测试自行写."""
    with torch.no_grad():
        z4 = torch.zeros(1, 8, net._mem_m.shape[1], dtype=torch.float16)
        z4[:, :, 0] = 1.0
        net._z4 = z4
        net._survival_signal = torch.full((1,), m, dtype=torch.float16)
        net._mem_m.zero_()
        net._mem_q.zero_()


def _run(net, ctx, sh):
    """跑一步判庭, 返回 (Δ_mem_q, Δ_mem_g, Δ_mem_err_ema), 并把账本复位到步前."""
    with torch.no_grad():
        q0 = net._mem_q.clone()
        g0 = net._mem_g.clone()
        e0 = net._mem_err_ema.clone()
        net.mem_pool.decide(net.mem_pool.account(ctx, sh.lm))
        dq = net._mem_q - q0
        dg = net._mem_g - g0
        de = net._mem_err_ema - e0
        net._mem_q.copy_(q0)
        net._mem_g.copy_(g0)
        net._mem_err_ema.copy_(e0)
    return dq, dg, de


def test_currency_prices_alignment():
    """信用按内容与 ẑ 的对齐定价: +ẑ 得正, −ẑ 得负, 零内容得零 (对称可检验)."""
    net = HybridPCNet(_tiny_cfg(mem_k0=3, mem_k_max=3))
    _seed(net, m=0.5)
    with torch.no_grad():
        net._mem_m[0, 0] = 1.0
        net._mem_m[1, 0] = -1.0
    ctx, sh = _stub()
    dq, _, _ = _run(net, ctx, sh)
    assert dq[0] > 0
    assert dq[1] < 0
    assert dq[2] == 0
    assert abs(dq[0] + dq[1]) < 1e-6


def test_credit_ignores_byte_error():
    """换货币的判据: 字节误差 (e_h / eps_total) 不再移动信用, 生存货币才移动 (2× 货币 → 2× 信用)."""
    net = HybridPCNet(_tiny_cfg())
    _seed(net, m=0.5)
    with torch.no_grad():
        net._mem_m[:, 0] = 1.0
        ctx, sh = _stub(e_h=1.0, eps=0.5)
        dq_a, dg_a, de_a = _run(net, ctx, sh)
        ctx_b, sh_b = _stub(e_h=-7.0, eps=50.0)
        dq_b, dg_b, de_b = _run(net, ctx_b, sh_b)
        assert torch.equal(dq_a, dq_b)
        assert torch.equal(dg_a, dg_b)
        assert not torch.equal(de_a, de_b)  # 字节误差确实被读到 (否则本测试空转)
        net._survival_signal = torch.full((1,), 1.0, dtype=torch.float16)
        dq_c, _, _ = _run(net, ctx_b, sh_b)
    assert torch.allclose(dq_c, dq_a * 2, rtol=1e-2, atol=1e-5)
    assert float(dq_a[0]) > 0.001


def test_w1v2_court_disabled():
    """_w1v2 模式下判庭整段早退: 账本与出生误差通道零变化."""
    net = HybridPCNet(_tiny_cfg())
    _seed(net, m=0.5)
    with torch.no_grad():
        net._mem_m[:, 0] = 1.0
        net._w1v2 = True
        ctx, sh = _stub()
        before = (net._mem_q.clone(), net._mem_g.clone(), net._mem_death_cnt.clone())
        net.mem_pool.decide(net.mem_pool.account(ctx, sh.lm))
        assert torch.equal(net._mem_q, before[0])
        assert torch.equal(net._mem_g, before[1])
        assert torch.equal(net._mem_death_cnt, before[2])
        assert float(net._mem_err_ema.item()) == 0.0
