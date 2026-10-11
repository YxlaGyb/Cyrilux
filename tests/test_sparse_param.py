"""P4 稀疏参数化 + 全层侧抑制 (M_x) 永久测试.

五条判据:
1. 出生态位双臂逐位等价 (foldiak vs ml, M_x=0 且 _r=0 → g≡1.0, 管道退化到旧路径);
2. M_x* fp16 稳定界 (更新元 cov ∈ [-1,1], 0.99 EMA 遗忘 → 有界);
3. M_x5 占位符随修剪压实同步 (ROW_STATE 覆盖);
4. g_rate 单向门性质 (r≤ρ 精确 1.0, 静默单元不被压进 topk);
5. EMA 治理梯度零污染 (_r_* 是 buffer 不进 named_parameters, 无 autograd 路径).
"""

from __future__ import annotations

import torch

from crates.hybrid.mask import g_rate, rate_ema
from crates.model import CyreneModel, HybridPCNet
from crates.model.constants import RHO_TARGET
from crates.neuron import MX_PLACEHOLDER, ROW_STATE

_LAYERS = ("l4", "l2", "l3", "l5", "l6")
_MX = {"l4": "M_x4", "l2": "M_x2", "l3": "M_x3", "l5": "M_x5", "l6": "M_x6"}


def _net(arm: str, seed: int = 7) -> HybridPCNet:
    torch.manual_seed(seed)
    cfg = CyreneModel(
        d_l4=64, d_l2=32, d_l3=32, d_l5=64, d_l6=16,
        max_seq_len=16, free_run_window=16, mem_k0=2, l5_inhibition=arm,
    )
    return HybridPCNet(cfg)


def _birth_snapshot(a: HybridPCNet, b: HybridPCNet, x: torch.Tensor) -> None:
    """双网出生态同状态前向: 输出与全部层级中间量逐位一致."""
    with torch.no_grad():
        oa, ob = a(x), b(x)
    ta, tb = (oa if torch.is_tensor(oa) else next(iter(oa.values())),
              ob if torch.is_tensor(ob) else next(iter(ob.values())))
    assert torch.equal(ta, tb)
    for name in ("_z0", "_z4", "_z2", "_z3", "_z5", "_z5_raw", "_z6"):
        ta, tb = getattr(a, name, None), getattr(b, name, None)
        if torch.is_tensor(ta) and torch.is_tensor(tb):
            assert torch.equal(ta, tb), f"{name} 出生态位双臂不等"


def test_birth_bitwise_foldiak_vs_ml():
    """出生态 (M_x=0, _r=0 → g≡1.0) 下 foldiak 与 ml 两臂逐位一致.

    这是新架构的退化锚: 零参数时 compete 与旧 topk 路径算术同构,
    -0.0 与额外乘 1.0 的差异都不允许 (torch.equal 含符号位).
    """
    a, b = _net("foldiak"), _net("ml")
    b.load_state_dict(a.state_dict())
    assert torch.equal(a.M_x5, b.M_x5) and float(a.M_x5.detach().abs().sum()) == 0.0
    x = torch.randint(32, 256, (1, 12), dtype=torch.long)
    _birth_snapshot(a, b, x)


def test_birth_bitwise_ml_vs_mask_off_path():
    """ml 臂出生态须与『M_x=0 恒等注入 + g=1』即旧路径逐位一致: _r 出生恒零 → g 精确 1.0."""
    net = _net("ml")
    assert all(float(getattr(net, f"_r_{l}").abs().sum()) == 0.0 for l in _LAYERS)
    assert all(float(getattr(net, _MX[l]).abs().sum()) == 0.0 for l in _LAYERS)
    g = g_rate(net._r_l4, RHO_TARGET["l4"])
    assert g.dtype == torch.float16 and torch.all(g == 1.0)


def test_g_rate_one_way_gate():
    """单向门: r≤ρ → 恰 1.0 (静默单元永不被压); r>ρ → <1.0 且随超额单调递减.

    fp16 在 0.1 附近 ulp≈6e-5, 超额样本取 0.01 起步保证 relu 严格可分辨.
    """
    rho = RHO_TARGET["l4"]
    quiet = g_rate(torch.tensor([0.0, rho * 0.5, rho], dtype=torch.float16), rho)
    assert torch.all(quiet == 1.0)
    hot = torch.tensor([rho + 0.01, rho * 2, rho * 5], dtype=torch.float16)
    gh = g_rate(hot, rho)
    assert torch.all(gh < 1.0) and gh[0] > gh[1] > gh[2]
    assert torch.isfinite(gh).all() and gh.dtype == torch.float16


def test_rate_ema_governance_direction():
    """EMA 治理方向: 高胜率 → r 升过 ρ → g<1.0; 静默 → r 降回 ρ → g 回 1.0."""
    r = torch.zeros(4, dtype=torch.float16)
    win = torch.ones(2, 3, 4, dtype=torch.bool)
    lose = ~win
    decay = 0.01
    for _ in range(200):
        rate_ema(r, win, decay)
    assert torch.all(r > RHO_TARGET["l4"]), f"全胜 200 步后 r 未越过 ρ: {r.tolist()}"
    assert torch.all(g_rate(r, RHO_TARGET["l4"]) < 1.0)
    for _ in range(400):
        rate_ema(r, lose, decay)
    assert torch.all(r <= RHO_TARGET["l4"]), f"全败后 r 未回落 ρ: {r.tolist()}"
    assert torch.all(g_rate(r, RHO_TARGET["l4"]) == 1.0)


def test_rate_buffers_gradient_free():
    """治理字段不进梯度: _r_* 是 buffer (不在 named_parameters), 学习后值变化非 autograd."""
    net = _net("ml")
    pnames = {n for n, _ in net.named_parameters()}
    for l in _LAYERS:
        assert f"_r_{l}" not in pnames, f"_r_{l} 混入参数 (治理先验进权重 = 最高风险违规)"
        assert getattr(net, f"_r_{l}").requires_grad is False
    assert net._r_l4.dtype == torch.float16
    # ROW_STATE 登记在案 → 修剪 perm 同步与审计覆盖
    for l in _LAYERS:
        assert ROW_STATE[f"_r_{l}"] == (l, "vec", "aux")


def test_mx_update_bounded_fp16():
    """M_x 更新元有界: cov 元 ∈ [-1,1] (c 归一化外积), EMA 遗忘 → M 不发散、无 NaN."""
    net = _net("ml", seed=3)
    torch.manual_seed(11)
    for i in range(12):
        net.learn(torch.randint(32, 256, (1, 12), dtype=torch.long))
    for l in _LAYERS:
        m = getattr(net, _MX[l]).data
        assert torch.isfinite(m).all(), f"{_MX[l]} 出现 NaN/Inf"
        assert m.abs().max() <= 1.0 + 1e-3, f"{_MX[l]} 越界: max={m.abs().max()}"
        assert m.dtype == torch.float16


def test_mx5_placeholder_syncs_after_compact():
    """foldiak 臂 M_x5 恒零占位随修剪 perm 重排 + 物理收缩 (ROW_STATE sq/aux 覆盖).

    契约与 M_x4 等同: 方阵按 active_size 裁剪, forward 以 [:dim,:dim] 切片消费,
    出生维度由 cfg.dims() 保存 → load 按检查点形状重建.
    """
    cfg = CyreneModel(
        d_l4=512, d_l2=192, d_l3=192, d_l5=192, d_l6=128,
        prune_warmup=0, prune_fraction=0.6, death_probation=1,
        active_size_lower_bound=128, l4_lower_bound=512,
        mem_k0=1, mem_k_max=1, max_seq_len=8, l5_inhibition="foldiak",
    )
    torch.manual_seed(0)
    net = HybridPCNet(cfg)
    net.learn(torch.randint(0, 256, (1, 4), dtype=torch.long))
    with torch.no_grad():
        w35 = net.W_35.data
        n_low = 115
        top = w35.shape[0] - n_low
        w35[top:] = w35[top:] / (w35[top:].norm(dim=1, keepdim=True) + 1e-6) * 1e-6
    for _ in range(2):
        net.pruner.decide()
    net.pruner.compact()
    a5 = net.active_size["l5"]
    assert a5 < 192, "本测试需要 L5 真收缩"
    assert tuple(net.M_x5.shape) == (a5, a5), f"M_x5 未随压实收缩: {tuple(net.M_x5.shape)}≠({a5},{a5})"
    assert torch.isfinite(net.M_x5).all() and float(net.M_x5.detach().abs().sum()) == 0.0
    assert "M_x5" in MX_PLACEHOLDER


def test_load_missing_mx_and_r_keys_keep_zero(tmp_path):
    """旧口径检查点缺 M_x*/​_r_* 键 → 保持零出生 (迁移不注入行为变化)."""
    net = _net("ml", seed=5)
    for i in range(4):
        torch.manual_seed(21 + i)
        net.learn(torch.randint(32, 256, (1, 12), dtype=torch.long))
    net._r_l4.fill_(0.04)
    net.M_x4.data.normal_(0.0, 0.01)
    p = tmp_path / "strip.safetensors"
    from safetensors.torch import load_file, save_file

    from crates.model.model_cyrene import read_checkpoint_metadata

    net.save(str(p))
    blob = {k: v.clone() for k, v in load_file(str(p), device="cpu").items()}
    dropped = {k: v for k, v in blob.items() if not (k.startswith("M_x") or k.startswith("_r_"))}
    assert len(dropped) < len(blob), "本测试需要剥离出 M_x*/_r_* 键"
    meta = dict(read_checkpoint_metadata(str(p)))
    p2 = tmp_path / "strip2.safetensors"
    save_file(dropped, str(p2), metadata=meta)
    net2 = HybridPCNet.load(str(p2), net.cfg)
    assert torch.equal(net2._r_l4, torch.zeros_like(net2._r_l4))
    assert float(net2.M_x4.abs().sum()) == 0.0


def test_ckpt_load_smoke_both_arms():
    """v201_mig 检查点双臂加载冒烟: foldiak (默认) 与 ml 都能载入且前向有限."""
    import glob
    import os

    from crates.model.model_cyrene import HybridPCNet as _N
    pats = (
        "out/*/migrated/exp115_say.safetensors",
        "out/*/migrated/*.safetensors",
        "out/*/*.safetensors",
    )
    ckpt = None
    for pat in pats:
        hits = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
        for h in hits:
            try:
                _N.load(h)
                ckpt = h
                break
            except Exception:
                continue
        if ckpt:
            break
    if ckpt is None:
        import pytest

        pytest.skip("out/ 无当前口径可加载检查点")
    for arm in ("foldiak", "ml"):
        net = _N.load(ckpt)
        net.cfg.l5_inhibition = arm
        with torch.no_grad():
            out = net(torch.randint(0, 256, (1, 8), dtype=torch.long))
        t = out if torch.is_tensor(out) else next(iter(out.values()))
        assert torch.isfinite(t).all(), f"{arm} 臂加载后前向出现 NaN"
