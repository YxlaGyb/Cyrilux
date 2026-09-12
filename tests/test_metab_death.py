"""
死亡契约测试: 下探缺失判据 — F 无下探计数 → 死亡回合 (mem 饥荒击杀 + 拓扑修剪), 纯 CPU.

确定性技巧: metab_dip_margin=1.0 → dip 阈值 = base·(1−δ) = 0 → f<0 永不 → 恒计数;
metab_dip_margin=-10 → 阈值 = 11·base → f<11·base 恒真 → 恒清零.
冷启动 (base==0): 登记基线, 该步按无下探计数 +1.
"""

import pytest
import torch
from safetensors.torch import save_file

from model import CyreneModel, DensePCNet


def _tiny_cfg(**over) -> CyreneModel:
    kw = {
        "d_l4": 64, "d_l2": 32, "d_l3": 32, "d_l5": 64, "d_l6": 16,
        "max_seq_len": 16, "free_run_window": 16, "mem_k0": 2, "mem_k_max": 2,
    }
    kw.update(over)
    return CyreneModel(**kw)


def _learn(net, n: int = 1):
    for _ in range(n):
        net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long))


def test_buffers_int32_persistent():
    net = DensePCNet(_tiny_cfg())
    for name in ("_metab_starve_cnt", "_metab_silence_cnt", "_metab_death_round_cnt"):
        buf = getattr(net, name)
        assert buf.dtype == torch.int32
        assert buf.shape == (1,)
    assert float(net._metab_F_base.item()) == 0.0  # 冷启动哨兵


def test_starve_counter_accumulates_and_resets():
    """无下探 → 连续计数; 下探 → 清零."""
    cfg = _tiny_cfg(metab_dip_margin=1.0)  # dip 阈值 0 → 恒无下探
    net = DensePCNet(cfg)
    _learn(net, 1)  # 哨兵: 只登记 F_prev, 不计数
    assert float(net._metab_starve_cnt.item()) == 0.0
    _learn(net, 1)  # 冷启动: base=f_now, 无下探计数 +1
    assert float(net._metab_starve_cnt.item()) == 1.0
    _learn(net, 1)
    assert float(net._metab_starve_cnt.item()) == 2.0
    net.cfg.metab_dip_margin = -10.0  # 阈值=11·base → 恒下探 → 清零
    _learn(net, 1)
    assert float(net._metab_starve_cnt.item()) == 0.0


def test_no_death_at_default_margin():
    """默认 δ=0.10: 短随机段无死亡回合 (无下探期 ≤ 12 < N=250)."""
    cfg = _tiny_cfg(prune_warmup=0)
    net = DensePCNet(cfg)
    _learn(net, 12)
    net.maybe_prune(16)  # 轮询点
    assert float(net._metab_death_round_cnt.item()) == 0.0
    assert net.active_size["l4"] == 64


def test_warmup_shields_rounds_but_counts():
    """warmup 期回合被挡, 计数继续积累 (发育期免回合, 不免登记)."""
    cfg = _tiny_cfg(metab_dip_margin=1.0, metab_death_steps=2, prune_warmup=1000)
    net = DensePCNet(cfg)
    _learn(net, 3)  # 哨兵 + 冷启动(1) + 无下探(2) → 计数 2 ≥ 2
    net.maybe_prune(8)  # 8 ≤ warmup → 回合被挡
    assert float(net._metab_starve_cnt.item()) >= 2.0
    assert float(net._metab_death_round_cnt.item()) == 0.0


def test_free_run_does_not_count():
    """free_run 代谢守卫提前返回 → 生成期不计数."""
    net = DensePCNet(_tiny_cfg(metab_dip_margin=1.0))
    net._metab_starve_cnt.fill_(7)
    net.learn(None, free_run=True)
    assert float(net._metab_starve_cnt.item()) == 7.0


def test_famine_kill_semantics():
    """K≥2 杀最低 g 且索引 0 免疫 (g0 最低也不杀); W1/_lm_in/W1_elig 按 K−1 块同步."""
    cfg = _tiny_cfg(mem_k0=3, mem_k_max=3)
    net = DensePCNet(cfg)
    with torch.no_grad():
        net._mem_g.copy_(torch.tensor([0.01, 0.9, 0.5], dtype=torch.float16))
        net._mem_m[:, 0] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
    head = 64 + net.bind_slot_dim
    assert net.W1.shape[0] == head + 3 * 64
    net.learning_engine._mem_famine_kill()
    assert net._mem_m.shape[0] == 2
    assert torch.equal(
        net._mem_m[:, 0], torch.tensor([1.0, 2.0], dtype=torch.float16)
    )  # 杀 g=0.5 (索引2); 索引0 免疫
    assert net.W1.shape[0] == head + 2 * 64
    assert net.W1_elig.shape == net.W1.shape
    assert net._lm_in == net.W1.shape[0]
    net.learning_engine._mem_famine_kill()  # K=2 → 1
    assert net._mem_m.shape[0] == 1
    net.learning_engine._mem_famine_kill()  # K=1 → no-op
    assert net._mem_m.shape[0] == 1
    assert net.W1.shape[0] == head + 1 * 64


def test_death_round_triggers_and_resets():
    """死亡回合: 计数达标 → 回合 (mem 击杀) → 计数清零; 慢性饥荒反复回合, K=1 后 no-op."""
    cfg = _tiny_cfg(metab_dip_margin=1.0, metab_death_steps=1, prune_warmup=0, mem_k0=2)
    net = DensePCNet(cfg)
    _learn(net, 2)  # 哨兵 + 冷启动(计数 1) ≥ 1
    net.maybe_prune(8)
    assert float(net._metab_death_round_cnt.item()) == 1.0
    assert net._mem_m.shape[0] == 1
    assert float(net._metab_starve_cnt.item()) == 0.0
    _learn(net, 1)  # 慢性饥荒: 再积累 → 再回合 (K=1 后回合 no-op)
    net.maybe_prune(16)
    assert float(net._metab_death_round_cnt.item()) == 2.0
    assert net._mem_m.shape[0] == 1
    _learn(net, 1)
    assert torch.isfinite(net.W1).all()
    assert torch.isfinite(net._metab_E)


def test_floor_round_noop_and_roundtrip(tmp_path):
    """兜底 no-op: K=1 且各层 active=bound → 回合零结构变化, 计数照 +1; save/load 保计数."""
    cfg = _tiny_cfg(metab_dip_margin=1.0, metab_death_steps=1, prune_warmup=0,
                    mem_k0=1, mem_k_max=1)
    net = DensePCNet(cfg)
    _learn(net, 2)  # 哨兵 + 冷启动(计数 1)
    sz = dict(net.active_size)
    w1_shape = net.W1.shape
    net.maybe_prune(8)
    assert float(net._metab_death_round_cnt.item()) == 1.0
    assert net.active_size == sz
    assert net._mem_m.shape[0] == 1
    assert net.W1.shape == w1_shape
    p = tmp_path / "floor.safetensors"
    net.save(str(p))
    loaded = DensePCNet.load(str(p), cfg)
    assert int(loaded._metab_death_round_cnt.item()) == 1
    assert int(loaded._metab_starve_cnt.item()) == 0  # 回合后已清零
    assert loaded._mem_m.shape[0] == 1


def test_weights_only_checkpoint_refused(tmp_path):
    """无 state_ver 的 weights-only 快照 → load 拒收 (不留「缺键即默认」的后门)."""
    cfg = _tiny_cfg(metab_dip_margin=1.0, metab_death_steps=1, prune_warmup=0)
    net = DensePCNet(cfg)
    p = tmp_path / "weights_only.safetensors"
    save_file({k: v.contiguous() for k, v in net.state_dict().items()}, str(p))
    with pytest.raises(ValueError, match="weights-only"):
        DensePCNet.load(str(p), cfg)


def test_mem_birth_rebound_after_famine():
    """饥荒击杀后 K 可由出生回弹 (新单元克隆自存活父) — mem_k 是瞬态应激指标,
    不可逆判据是 active_size (只减不增)."""
    cfg = _tiny_cfg(mem_k0=2, mem_k_max=3)
    net = DensePCNet(cfg)
    net.learning_engine._mem_famine_kill()
    assert net._mem_m.shape[0] == 1
    net.learning_engine._mem_birth(0)
    assert net._mem_m.shape[0] == 2
    head = 64 + net.bind_slot_dim
    assert net.W1.shape[0] == head + 2 * 64
    assert net._lm_in == net.W1.shape[0]


# ---------- P3-b 对称牙: 长期只看不说 ----------


def test_silence_tooth_mirrors_starve():
    """强制感知 → 沉默计数累积 → 回合; 触发后归零 (dip_margin=-10 屏蔽 starve 干扰)."""
    cfg = _tiny_cfg(metab_dip_margin=-10.0, metab_silence_steps=3, prune_warmup=0)
    net = DensePCNet(cfg)
    for _ in range(4):  # 首步哨兵不计数 → 4 步累积 3
        net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long), force_perception=True)
    assert float(net._metab_silence_cnt.item()) == 3.0
    assert float(net._metab_starve_cnt.item()) == 0.0  # dip 恒真 → starve 未参与
    assert net.maybe_prune(8) == "silence"
    assert float(net._metab_death_round_cnt.item()) == 1.0
    assert float(net._metab_silence_cnt.item()) == 0.0


def test_silence_reset_by_echo():
    """回声 = 表达发生 → 沉默计数清零 (镜像: 感知 dip 清零 starve)."""
    net = DensePCNet(_tiny_cfg(metab_dip_margin=-10.0, metab_silence_steps=250))
    for _ in range(3):
        net.learn(torch.randint(0, 256, (1, 8), dtype=torch.long), force_perception=True)
    assert float(net._metab_silence_cnt.item()) == 2.0
    net.learn(None, free_run=False)  # 回声
    assert float(net._metab_silence_cnt.item()) == 0.0
    assert float(net._metab_starve_cnt.item()) == 1.0  # 回声相 starve 恒 +1 (镜像证据)
