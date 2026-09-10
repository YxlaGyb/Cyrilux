"""P3-b 行为门测试: 门公式纯函数 + learn() 内部路由 + 每行为 F 分账 + 死亡时钟分相
+ 行为账本 EMA + ver4 迁移 + sync_gate 镜像 (纯 CPU, 无 GPU)."""


import pytest
import torch

from model import CyreneModel, DensePCNet
from model.dense.learning.metabolism import gate_psay, rel_norm


def _tiny(**over) -> CyreneModel:
    kw = {
        "d_l4": 64, "d_l2": 32, "d_l3": 32, "d_l5": 64, "d_l6": 16,
        # 窗 32: 回声种子 (≤16) + 15 步续写 ≤ max_seq_len (mem 序列缓冲按它切)
        "max_seq_len": 32, "free_run_window": 16, "mem_k0": 2, "mem_k_max": 2,
    }
    kw.update(over)
    return CyreneModel(**kw)


def _make(**over):
    torch.manual_seed(0)
    return DensePCNet(_tiny(**over))


def _echo_net(**over):
    net = _make(**over)
    net._echo_seed = torch.zeros(1, 1, dtype=torch.long)
    return net


# ---------- 门公式纯函数 ----------


def test_gate_psay_pure():
    # 中性点: 全部输入中性 (g 差=0, stress=0, A=0.5, nov=0.5) → σ(0)=0.5
    assert gate_psay(0.0, 0.0, 0.0, 0.5, 0.5, 3, 2, 1, 1.5) == pytest.approx(0.5)
    # 账本差单调升: 说比看更赚 → 想说
    p0 = gate_psay(0.0, 0.0, 0.0, 0.5, 0.5, 3, 2, 1, 1.5)
    p1 = gate_psay(0.3, 0.0, 0.0, 0.5, 0.5, 3, 2, 1, 1.5)
    p2 = gate_psay(0.0, 0.3, 0.0, 0.5, 0.5, 3, 2, 1, 1.5)
    assert p1 > p0 and p2 < p0
    # 应激单调降 (能量紧 → 少说)
    assert gate_psay(0.0, 0.0, 0.7, 0.5, 0.5, 3, 2, 1, 1.5) < p0
    # 振荡器单调升
    assert gate_psay(0.0, 0.0, 0.0, 1.0, 0.5, 3, 2, 1, 1.5) > p0
    # 新奇度单调降 (世界新 → 多看)
    assert gate_psay(0.0, 0.0, 0.0, 0.5, 0.9, 3, 2, 1, 1.5) < p0
    # 值域 (0,1) 与对称性: 输入全体翻转符号 → p' = 1−p (σ 对称)
    lo = gate_psay(-3.0, 0.0, 0.9, 0.0, 0.95, 3, 2, 1, 1.5)
    hi = gate_psay(3.0, 0.0, -0.9, 1.0, 0.05, 3, 2, 1, 1.5)
    assert 0.0 < lo < 0.5 < hi < 1.0


def test_rel_norm_family():
    assert rel_norm(1.0, 1e-3) == pytest.approx(1.0 / 1.001)  # 首样本: 新奇≈1
    assert rel_norm(0.1, 0.1) == pytest.approx(0.1 / 0.200001)  # eps 内并入
    assert rel_norm(0.0, 1.0) == 0.0
    assert rel_norm(1.0, 1.0) == pytest.approx(1.0 / 2.000001)
    assert 0.0 <= rel_norm(5.0, 1e-3) < 1.0  # 值域 [0,1)


def test_gate_psay_consistent_with_gpu():
    """_metab_psay 与纯函数用同批缓冲值重算, fp16 容差内一致."""
    net = _echo_net()
    for _ in range(3):
        net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))
    A = float(net._intr_sin.index_select(0, net._intr_cnt.long().squeeze(0)).item())
    s = float(getattr(net, "_metab_stress", torch.tensor(0.0)).item())
    p = gate_psay(
        float(net._metab_gain_eco.item()), float(net._metab_gain_perc.item()),
        s, A, float(net._metab_nov.item()),
        net.cfg.metab_gate_kappa_g, net.cfg.metab_gate_kappa_s,
        net.cfg.metab_gate_kappa_a, net.cfg.metab_gate_kappa_n,
    )
    assert abs(p - float(net._metab_psay.item())) < 5e-3  # fp16 σ 容差


# ---------- learn() 内部路由 ----------


def test_learn_none_forced_echo():
    """learn(None) 恒回声: cost_say>0, cost_perc=0 (兼容语义锁定)."""
    net = _echo_net()
    net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))  # 感知: 哨兵
    net.learn(None, free_run=False)
    assert float(net._metab_cost_say) > 0.0
    assert float(net._metab_cost_perc) == 0.0
    assert hasattr(net, "_gen_bytes")


def test_routing_gate_echo():
    """门选回声: byte_ids 非 None 也走回声分支 — cost 分型、感知轨 F_prev 未动、世界域冻结."""
    net = _echo_net()
    net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))  # 感知: 哨兵
    w0 = net.W_diff.detach().clone()
    p0 = net._metab_F_prev_perc.clone()
    net._behavior_py = True  # 门镜像手工置位 (等价 sync_gate 读到 1)
    net.learn(torch.randint(0, 256, (1, 9), dtype=torch.long))
    assert float(net._metab_cost_say) > 0.0
    assert float(net._metab_cost_perc) == 0.0
    assert torch.equal(net._metab_F_prev_perc, p0)  # 感知轨未动 (分账)
    assert torch.equal(net.W_diff, w0)  # 世界域冻结 (回声相位)
    assert hasattr(net, "_gen_bytes")


def test_routing_force_perception_overrides():
    """force_perception=True 且门选回声 → 仍走感知."""
    net = _echo_net()
    net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))  # 哨兵
    net._behavior_py = True
    net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long), force_perception=True)
    assert float(net._metab_cost_perc) > 0.0
    assert float(net._metab_cost_say) == 0.0


def test_per_behavior_f_prev():
    """learn(x) 只动感知轨; learn(None) 只动回声轨."""
    net = _echo_net()
    net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))  # 哨兵: perc 登记
    assert float(net._metab_F_prev_perc) >= 0.0
    assert float(net._metab_F_prev_eco) < 0.0
    net.learn(None, free_run=False)  # 回声首跑: eco 登记, perc 不动
    assert float(net._metab_F_prev_eco) >= 0.0
    assert float(net._metab_F_prev_perc) >= 0.0


# ---------- 死亡时钟分相 ----------


def test_echo_starve_unconditional_plus_one():
    """回声相无 dip 判据恒 +1; 感知相 dip 清零只有感知步 (dip_margin=-10 恒 dip 技巧).
    注意冷启动步 (base==0 → 登记基线) 按无下探计数 +1 — P2 语义, 与 margin 无关."""
    net = DensePCNet(_tiny(metab_dip_margin=-10.0))
    net._echo_seed = torch.zeros(1, 1, dtype=torch.long)
    net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))  # 哨兵 (不计数)
    net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))  # 感知冷启动: +1
    net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))  # 感知恒 dip → 0
    assert float(net._metab_starve_cnt.item()) == 0.0
    net.learn(None, free_run=False)  # 回声: 无条件 +1 (F 多低都加)
    assert float(net._metab_starve_cnt.item()) == 1.0
    net.learn(None, free_run=False)
    assert float(net._metab_starve_cnt.item()) == 2.0


def test_perception_dip_resets_irrespective_of_echo_runs():
    """感知下探中断回声累计: 回声 +2 后感知 dip → 回 0 (冷启动步先行)."""
    net = DensePCNet(_tiny(metab_dip_margin=-10.0))
    net._echo_seed = torch.zeros(1, 1, dtype=torch.long)
    net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))  # 哨兵
    net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))  # 感知冷启动: +1
    net.learn(None, free_run=False)
    net.learn(None, free_run=False)
    assert float(net._metab_starve_cnt.item()) == 3.0
    net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))  # 感知恒 dip → 清零
    assert float(net._metab_starve_cnt.item()) == 0.0


# ---------- 行为账本 EMA ----------


def test_gain_ledger_ema_only_running_behavior():
    """仅运行行为记账; 非运行行为不更新不衰减."""
    net = _echo_net()
    net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))  # 哨兵
    net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))  # 感知: gain_perc 记账
    g_p = float(net._metab_gain_perc.item())
    assert g_p >= 0.0 and g_p < 1.0
    assert float(net._metab_gain_eco.item()) == 0.0  # 回声从未运行
    net.learn(None, free_run=False)  # 回声首跑 (df=0 → R=0)
    net.learn(None, free_run=False)  # 回声二次 (df 真实)
    assert float(net._metab_gain_eco.item()) >= 0.0
    assert float(net._metab_gain_perc.item()) == g_p  # 感知账本冻结


# ---------- sync_gate 镜像 ----------


def test_sync_gate_mirror():
    """sync_gate() 后 _behavior_py 与 _metab_gate_hit 一致."""
    net = _echo_net()
    net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))
    hit = float(net._metab_gate_hit.item())
    net.sync_gate()
    assert net._behavior_py == (hit > 0.0)


def test_load_resets_behavior_mirror(tmp_path):
    """load 后 _behavior_py 默认 False (感知), 未调 sync_gate 前路由确定."""
    net = _echo_net()
    net._behavior_py = True
    net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))
    p = str(tmp_path / "_p3b_mirror.safetensors")
    net.save(p)
    loaded = DensePCNet.load(p, _tiny())
    assert loaded._behavior_py is False
