"""P1-P3 代谢域测试: 体内 E/R 账本 + 成本计价 + W_diff 预缩放 (纯 CPU 逻辑, 无 GPU)."""

import math

import pytest
import torch

from model import CyreneModel, DensePCNet


def _make_net():
    torch.manual_seed(0)
    cfg = CyreneModel(
        d_l4=64, d_l2=32, d_l3=32, d_l5=64, d_l6=16,
        max_seq_len=16, free_run_window=16, mem_k0=2,
    )
    return DensePCNet(cfg)


def test_metab_buffers_fp16_persistent():
    net = _make_net()
    for name in (
        "_metab_E", "_metab_E_ref", "_metab_df_mad_perc", "_metab_df_mad_eco",
        "_metab_famine_prog", "_metab_F_prev_perc", "_metab_F_prev_eco",
        "_metab_gain_perc", "_metab_gain_eco",
        "_metab_nov_ema", "_metab_nov", "_metab_psay", "_metab_gate_hit", "_gate_rand",
        "_metab_cost_perc", "_metab_cost_learn", "_metab_cost_mem",
        "_metab_cost_say", "_metab_cost_tot",
    ):
        buf = getattr(net, name)
        assert buf.dtype == torch.float16
        assert buf.shape == (1,)
    assert float(net._metab_E_ref) == 0.0  # E_ref = E 慢 EMA 冷启 0


def test_metabolism_first_step_register_then_settle():
    net = _make_net()
    x = torch.randint(0, 256, (1, 8), dtype=torch.long)
    net.learn(x)  # 首步: 只登记 F_prev(感知轨), R=0
    assert float(net._metab_F_prev_perc) >= 0.0
    assert float(net._metab_F_prev_eco) < 0.0  # 回声轨未动
    assert float(net._metab_R.item()) == 0.0
    net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))  # 次步: 结算
    r = float(net._metab_R.item())
    assert torch.isfinite(torch.tensor(r))
    assert abs(r) <= 1.0  # R 原语 = tanh(ΔF/div·MAD) 天然有界


def test_metabolism_ledger_replay():
    """账本重放 (P3-c 含成本 + P3-b 行为内 ΔF): E_t = (1-d)·E_{t-1} + c·ΔF_b − Σcost,
    其中 ΔF_b = 同一行为上一 F − 本 F (感知/回声各轨), fp16 累计容差内一致."""
    net = _make_net()
    net._echo_seed = torch.zeros(1, 1, dtype=torch.long)
    kind = ["perc", "echo", "perc", "echo", "perc"]  # 交错: 感知/回声交替, 两轨各有二次采样
    f_prev = {"perc": None, "echo": None}
    e = 0.0
    for i, k in enumerate(kind):
        if k == "perc":
            net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))
            fb = net._metab_F_prev_perc
            f_now = float(fb.item())
            cost = (net.cfg.metab_cost_perc * f_now + net.cfg.metab_cost_learn * math.sqrt(f_now)
                    + net.cfg.metab_cost_mem * net.W1.shape[0])
        else:
            net.learn(None, free_run=False)
            fb = net._metab_F_prev_eco
            f_now = float(fb.item())
            cost = (net.cfg.metab_cost_learn * math.sqrt(f_now)
                    + net.cfg.metab_cost_mem * net.W1.shape[0]
                    + net.cfg.metab_cost_say)
        prev = f_prev[k]  # 本行为上一次 F (None = 行为首跑)
        f_prev[k] = f_now
        if i == 0:
            continue  # 全局哨兵步: 只登记不结算
        df = (prev - f_now) if prev is not None else 0.0  # 行为首跑 df=0
        e = e * (1.0 - net.cfg.metab_d) + net.cfg.metab_c * df - cost
    assert abs(float(net._metab_E.item()) - e) < 5e-3


def test_metab_cost_structure():
    """P3-c 成本结构: 看/说按步型计价, 记=κ_m·W1 行数, 总支出=分量和; 双账本: 成本只进 E."""
    net = _make_net()
    net._echo_seed = torch.zeros(1, 1, dtype=torch.long)
    cfg = net.cfg
    # 感知步: 看>0, 说=0
    net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))
    assert float(net._metab_cost_perc) > 0.0
    assert float(net._metab_cost_say) == 0.0
    # 回声步: 看=0, 说=κ_s
    net.learn(None, free_run=False)
    assert float(net._metab_cost_perc) == 0.0
    assert float(net._metab_cost_say) == pytest.approx(cfg.metab_cost_say, rel=1e-2)
    # 记: 恒 = κ_m·W1 行数 (fp16 亚正常数精度用绝对容差); 总 = 分量和
    assert float(net._metab_cost_mem) == pytest.approx(
        cfg.metab_cost_mem * net.W1.shape[0], abs=1e-6
    )
    tot = (
        float(net._metab_cost_learn) + float(net._metab_cost_mem) + float(net._metab_cost_say)
    )
    # cost_tot 是 fp16 四分量顺序累加 — 与 python float 和差 ≤ 2 ULP (0.003 量级 ULP≈1.9e-6)
    assert float(net._metab_cost_tot) == pytest.approx(tot, abs=4e-6)
    assert torch.isfinite(net._metab_stress if hasattr(net, "_metab_stress") else torch.tensor(0.0))


def test_metabolism_echo_path_and_wdiff_prescale():
    """回声步代谢照常结算; W_diff 预缩放后磁盘真动 (遥测有限)."""
    net = _make_net()
    net._echo_seed = torch.zeros(1, 1, dtype=torch.long)
    w0 = net.W_diff.detach().clone()
    net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))  # 感知: 登记 + W_diff 学习
    assert (net.W_diff != w0).any()  # 预缩放前死锁: 磁盘条目不动
    assert torch.isfinite(net._dW_diff_absmax_raw)
    net.learn(None, free_run=False)  # 回声: 代谢结算, 无 NaN
    for p in net.parameters():
        assert torch.isfinite(p).all()
    assert torch.isfinite(net._metab_E)
