"""让 L5 参与修剪
RED 测试 → shape 断言 + 一步 learn forward 不崩.
"""

from __future__ import annotations

import pytest
import torch

from model import CyreneModel, DensePCNet

# 字节域默认 256 (与 CyreneModel.d_input 默认一致,动作域 d_act 同)
D_INPUT = 256
D_ACT = 256


# 所有测试 CPU,fp16,deterministic
torch.manual_seed(0)


def _tiny_cfg(**over) -> CyreneModel:
    kw = {
        # 让 L4 精确等 l4_lower_bound → L4 不剪,专注观察 L5
        "d_l4": 512,
        "d_l2": 192,   # > bound 128 → 允许剪
        "d_l3": 192,   # > bound 128
        "d_l5": 192,   # > bound 128 ← 本轮主角
        "d_l6": 128,   # = bound 128 (L6 不剪,测 L5→L6 列同步)
        "prune_warmup": 0,
        "prune_fraction": 0.6,  # 每步最多砍 60%
        "death_probation": 1,   # 死缓 1 步直接过期 (本轮首步没 expired 就行)
        "active_size_lower_bound": 128,
        "l4_lower_bound": 512,
        "mem_k0": 1,
        "mem_k_max": 1,        # 禁用单元出生/死亡,避免 W1 形状非修剪因素干扰
        "input_history": True, # W_04 列=512 (默认)
        "max_seq_len": 8,
    }
    kw.update(over)
    return CyreneModel(**kw)


def test_l5_enters_layers_list():
    """RED-1:layers 列表现在含 l5,且顺序保证依赖链 l3→l5→l6."""
    cfg = _tiny_cfg()
    net = DensePCNet(cfg)
    pruner = net.pruner
    # 触发一次以填充内部字典
    pruner._W_attr = None  # 强制重置状态
    # 直接进 _prune 入口, 但不执行 (patch _step_counter 直接 return 的做法不好)
    # 改用:读取主入口内部 hardcoded 顺序 — 我们 spy 通过手动调用 _prune 前先
    #   检查 pruner._prune.__code__ 里的常量字符串太脆弱.
    # 更直接: 手工构造 layers 字典 (按 _prune 内的字面量)
    import inspect
    src = inspect.getsource(pruner._prune)
    # 断言顺序: l4,l2,l3,l5,l6
    assert 'layers = ["l4", "l2", "l3", "l5", "l6"]' in src, (
        "layers 未按依赖链登记 L5,或顺序不对"
    )
    assert '"l5": "W_35"' in src, "W_attr 漏登记 L5→W_35"
    assert '"l5": "bias_l5"' in src, "b_attr 漏登记 L5→bias_l5"
    assert '"l5": "l3"' in src, "src_layer 漏登记 L5→L3"
    del net


def test_l5_prune_shapes_all_synced():
    """RED-2: 真正剪一次 L5,断言 16 项张量形状同步 a5_new == active_size[l5].

    准备:把 W_35 末尾 60 行手工设成极小 (触发排名淘汰), 其他层所有权重保持 0.1×
    随机 (保证不触发它们剪或剪也不影响 L5 观测).
    """
    cfg = _tiny_cfg()
    net = DensePCNet(cfg)
    # 初始形状
    assert net.active_size["l5"] == cfg.d_l5 == 192, f"init bad a5={net.active_size['l5']}"

    # 手工制造 L5 低活:W_35 后 115 行 (约60%) 行范数压到几乎 0,其余行保持中等
    with torch.no_grad():
        w35 = net.W_35.data.clone()
        a5 = w35.shape[0]
        n_low = 115
        top_a5 = a5 - n_low  # 77
        # 前 top_a5 行放大 (给一个明显大的 norm)
        w35[:top_a5] = w35[:top_a5] / (w35[:top_a5].norm(dim=1, keepdim=True) + 1e-6) * 3.0
        # 后 n_low 行压到 1e-3 级别
        w35[top_a5:] = w35[top_a5:] / (w35[top_a5:].norm(dim=1, keepdim=True) + 1e-6) * 0.001
        net.W_35.copy_(w35.to(torch.float16))
        # 也给 L4/L2/L3/L6 的 W_* 保持合理范围,避免别的层触发过度修剪
        for name in ("W_04", "W_42", "W_23", "W_56"):
            p = getattr(net, name).data
            p.copy_((p / (p.norm(dim=1, keepdim=True) + 1e-6) * 1.0).to(torch.float16))

    # 用极短的字节序列跑 learn,推进内部状态后直调 _prune
    byte_ids = torch.randint(0, D_INPUT, (1, 4), dtype=torch.long)
    # 不关心 loss — 只要 shape 对就好
    try:
        net.learn(byte_ids, closed_loop=False, free_run=False)
        net.pruner._prune()
    except Exception as e:
        pytest.fail(f"首步 learn 直接抛异常 {type(e).__name__}: {e}")

    a5_new = net.active_size["l5"]
    assert a5_new < 192, f"L5 没被剪! a5_new={a5_new}"
    assert a5_new >= 128, f"L5 触底保护失效 a5_new={a5_new}"

    # ── 断言所有 L5 同源张量形状 ──
    # 1. 核心权重 W_35 / W_t5 / bias_l5
    assert tuple(net.W_35.shape) == (a5_new, net.active_size["l3"]), (
        f"W_35 shape {tuple(net.W_35.shape)} ≠ {(a5_new, net.active_size['l3'])}"
    )
    assert tuple(net.W_t5.shape) == (a5_new, a5_new)
    assert tuple(net.bias_l5.shape) == (a5_new,)

    # 2. L5 方阵三张: M_l5 / E_l5 / E_t5
    for attr in ("M_l5", "E_l5", "E_t5"):
        shape = tuple(getattr(net, attr).shape)
        assert shape == (a5_new, a5_new), f"{attr} shape {shape}≠({a5_new},{a5_new})"

    # 3. W_pred_54 行 / _gain_mask 行
    a4_new = net.active_size["l4"]
    a3_new = net.active_size["l3"]
    assert tuple(net.W_pred_54.shape) == (a5_new, a4_new), (
        f"W_pred_54 {tuple(net.W_pred_54.shape)}≠({a5_new},{a4_new})"
    )
    assert tuple(net._gain_mask.shape) == (a5_new, a3_new), (
        f"_gain_mask {tuple(net._gain_mask.shape)}≠({a5_new},{a3_new})"
    )

    # 4. BCM 滑阈 _theta_l5
    assert tuple(net._theta_l5.shape) == (a5_new,)

    # 5. STP l5 四兄弟
    for attr in ("_stp_r_l5", "_stp_tau_l5", "_stp_u_l5", "_stp_active_ema_l5"):
        shape = tuple(getattr(net, attr).shape)
        assert shape == (a5_new,), f"{attr} shape {shape}≠({a5_new},)"

    # 6. act_ema L5 四兄弟
    for attr in ("_active_ema_w35", "_active_ema_wt5", "_active_ema_wp54", "_active_ema_b5"):
        shape = tuple(getattr(net, attr).shape)
        assert shape == (a5_new,), f"{attr} shape {shape}≠({a5_new},)"

    # 7. 资格迹:L5 涉及的 Hebbian 权重名
    for (attr, exp_shape) in [
        ("W_35_elig", (a5_new, a3_new)),
        ("W_t5_elig", (a5_new, a5_new)),
        ("W_pred_54_elig", (a5_new, a4_new)),
    ]:
        buf = getattr(net, attr)
        shape = tuple(buf.shape)
        assert shape == exp_shape, f"{attr} shape {shape}≠{exp_shape}"

    # 8. 死亡行/死缓计数器
    assert net._death_row_l5.numel() == a5_new
    assert net._probation_counter_l5.numel() == a5_new

    # 9. 下游: L6 的 W_56 列 = a5_new (因为 L6 的 active_size_lower_bound=128=init,
    #    d_l6=128,L6 不剪,所以只要看列就行了)
    a6_shape = tuple(net.W_56.shape)
    assert a6_shape[1] == a5_new, f"W_56 列={a6_shape[1]}≠a5_new={a5_new}"
    # W_56_elig 列 = a5_new
    assert tuple(net.W_56_elig.shape)[1] == a5_new


def test_l5_prune_then_forward_no_crash():
    """RED-3:修剪后再跑一次 forward + learn,确保 matmul 形状全对齐,无 NaN/shape 崩。"""
    cfg = _tiny_cfg()
    net = DensePCNet(cfg)
    # 准备 L5 低活
    with torch.no_grad():
        a5 = net.W_35.shape[0]
        n_low = 115
        top_a5 = a5 - n_low
        net.W_35.data[:top_a5].mul_(3.0)
        net.W_35.data[top_a5:].mul_(0.001)
    byte_ids_a = torch.randint(0, D_INPUT, (1, 4), dtype=torch.long)
    byte_ids_b = torch.randint(0, D_INPUT, (1, 4), dtype=torch.long)

    # Step1:修剪发生
    net.learn(byte_ids_a, closed_loop=False, free_run=False)
    net.pruner._prune()
    # 断言修剪真的发生
    assert net.active_size["l5"] < cfg.d_l5, "L5 未剪,测试前提失效"

    # Step2:修剪后 forward + learn 必须形状 OK
    try:
        out = net.forward(byte_ids_b)  # type: ignore
        s2 = net.learn(byte_ids_b, closed_loop=False, free_run=False)
    except Exception as e:
        pytest.fail(f"剪后 forward/learn 抛 {type(e).__name__}: {e}")
    # forward 返回 dict{mu_diff, diff_err, free_energy}
    assert isinstance(out, dict), f"forward 回 {type(out)} 不是 dict"
    for k in ("mu_diff", "diff_err", "free_energy"):
        assert k in out, f"forward dict 缺键 {k}"
        v = out[k]
        if isinstance(v, torch.Tensor):
            assert not torch.isnan(v).any(), f"forward[{k}] NaN"
    # 观测: mu_diff 的形状应和预测一致 (形状不定,但至少有限元素)
    mu = out["mu_diff"]
    if isinstance(mu, torch.Tensor):
        assert mu.numel() > 0, "mu_diff 空张量"
    # learn 返回 dict, free_energy 键存在, 且无 NaN (若为 tensor)
    assert "free_energy" in s2
    fe = s2["free_energy"]
    if isinstance(fe, torch.Tensor):
        assert not torch.isnan(fe).any()


def test_l4_shrink_w1_markers_aligned():
    """缺陷 A/B 回归: L4 收缩时 W1/W1_elig 按 [z4 | bind | 单元×K] 布局 gather.

    旧实现是头切片 — bind 段会命中 z4 probation 行 (静默语义错位: 形状自洽,
    不产 NaN, 因此"剪后 forward 零 NaN"型测试永假阳性). 现有 _tiny_cfg 的
    d_l4=512=bound 使 L4 永不 permute, 盲区原样. 本测试: d_l4=768 > bound,
    行标记唯一化, 修剪后断言 bind 段完整保序、单元块只含本块标记、迹同形同步.
    """
    cfg = CyreneModel(
        d_l4=768, d_l2=128, d_l3=128, d_l5=128, d_l6=128,
        prune_warmup=0, prune_fraction=0.05,
        active_size_lower_bound=128, l4_lower_bound=512,
        mem_k0=3, mem_k_max=3, max_seq_len=8,
    )
    net = DensePCNet(cfg)
    n_bind = net.bind_slot_dim  # 32
    K = 3
    head = 768 + n_bind
    with torch.no_grad():
        w1 = net.W1.data
        w1.zero_()
        for i in range(768):
            w1[i, 0] = i  # z4 块: 0..767
        for i in range(n_bind):
            w1[768 + i, 0] = 1000 + i  # bind 块: 1000..1031
        for k in range(K):
            for i in range(768):
                w1[head + k * 768 + i, 0] = 2000 + k * 1000 + i  # 单元块 k
        net.W1_elig.copy_(net.W1.data)
        # W_04 行范数严格递减 → 最弱 = 尾行 (topk 稳定; 断言本身不依赖具体 keep 集)
        r = (torch.arange(768, dtype=torch.float32).flip(0) + 1).unsqueeze(1) / 768.0
        net.W_04.data.copy_(
            r.expand(-1, net.W_04.shape[1]).to(torch.float16)
        )
    # 死亡回合同序: mem 饥荒击杀 (K 3→2) → 拓扑修剪
    net._mem_g.copy_(torch.tensor([0.9, 0.5, 0.1], dtype=torch.float16))
    net.learning_engine._mem_famine_kill()
    assert net._mem_m.shape[0] == K - 1
    net.pruner._prune()
    a4 = net.active_size["l4"]
    assert a4 < 768 and a4 >= 512, f"L4 未剪/触底异常 a4={a4}"

    e_w1 = net.W1[:, 0]
    assert net.W1.shape[0] == a4 + n_bind + (K - 1) * a4, (
        f"W1 行数 {net.W1.shape[0]} ≠ {a4 + n_bind + (K - 1) * a4}"
    )
    # 缺陷 B: 迹同形同步, 且标记逐位一致 (旧代码 pruning 路径从不同步 W1_elig)
    assert net.W1_elig.shape == net.W1.shape
    assert torch.equal(net.W1_elig[:, 0], e_w1)
    assert net._lm_in == net.W1.shape[0]
    # bind 段完整保序 (绑定行从不 permute — 旧头切片会把 z4 probation 行放这里)
    bind_seg = e_w1[a4 : a4 + n_bind]
    assert torch.equal(bind_seg, torch.arange(1000, 1000 + n_bind, device=bind_seg.device))
    # 单元块只含本块标记 (z4/bind 泄漏即语义错位)
    for k in range(K - 1):
        blk = e_w1[a4 + n_bind + k * a4 : a4 + n_bind + (k + 1) * a4]
        lo, hi = 2000 + k * 1000, 2000 + k * 1000 + 768
        assert bool(((blk >= lo) & (blk < hi)).all()), f"单元块 {k} 标记泄漏"
    # z4 块不含 bind/单元标记
    z4 = e_w1[:a4]
    assert bool(((z4 < 768) | (z4 == 0)).all()), "z4 块混入 bind/单元标记"


def test_l4_shrink_then_learn_no_shape_crash():
    """L4 收缩后紧接 learn (感知+回声) 不崩 — 回归: 活性缓冲漏同步 (曾漏 _dw_slot 等)."""
    cfg = _tiny_cfg(d_l4=576)  # > l4_lower_bound=512 → L4 真收缩
    net = DensePCNet(cfg)
    net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))
    net.pruner._prune()
    a4 = net.active_size["l4"]
    assert a4 < net.cfg.d_l4
    assert net._dw_slot.shape == (1, a4, a4)
    assert net._z4_fut_buf.shape[-1] == a4
    assert net._theta_w04.shape == (a4,)
    net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))
