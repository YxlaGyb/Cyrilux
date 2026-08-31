"""world_lang 世界语言物理单元测试: 评分 / 新颖税 / E 代谢 / 认证门.

world_lang.py 在 scripts/ (非包), 用 importlib 加载 (与 probe 脚本同法).
"""
import importlib.util
import json
from collections import deque

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("world_lang", "scripts/world_lang.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
WorldLangPhysics = _mod.WorldLangPhysics

LINES = [
    "你好，今天天气很好，我们一起去公园散步吧。",
    "学习知识需要耐心，每天进步一点点就会成功。",
    "这辆汽车的速度很快，安全性能也很好。",
    "科学家发现了新的规律，这是重要的发现。",
    "生活就像一场旅行，重要的是沿途的风景。",
]


@pytest.fixture
def world(tmp_path):
    p = tmp_path / "mini.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for t in LINES:
            f.write(json.dumps({"text": t}) + "\n")
    w = WorldLangPhysics(str(p), n_char_lines=10, n_trigram_lines=None, top_n=100)
    w.q_ref = 0.5  # 手动覆盖 (mini 语料 heldout 空 → 自校准 q_base=0)
    return w


def test_L_coverage_common_chars(world):
    # 全部常用字 (语料内) → L = 1.0
    s = world.score("你好天气很好我们".encode("utf-8"))
    assert s["L"] == 1.0
    # 全生僻字 (不在 top-100) → L = 0.0
    s2 = world.score("龘靐齉爩鱻".encode("utf-8"))
    assert s2["L"] == 0.0


def test_S_structure_hit(world):
    # 语料内字节 3-gram → S 高 (位图已含)
    s = world.score(LINES[0].encode("utf-8"))
    assert s["S"] > 0.9
    # 随机字节 (非语料 3-gram) → S 低
    rng = np.random.default_rng(7)
    noise = bytes(rng.integers(0, 256, size=64, dtype=np.int64).tolist())
    assert world.score(noise)["S"] < 0.2


def test_X_novelty_tax(world):
    a = "你好天气很好我们出去散步".encode("utf-8")
    b = "abcdEFGH1234!@#$".encode("utf-8")  # 与 A 无共同字节 3-gram
    world._hist.clear()
    world.record(a)
    # 复读同一发声 → 与最近历史并集完全重叠 → X → 1
    assert world.score(a)["X"] > 0.95
    # 全新发声 (与 A 无共同 3-gram) → X = 0
    assert world.score(b)["X"] == 0.0


def test_E_metabolism_monotone(world):
    e0 = world.E
    # 高 q → E 上升
    world.step_E(0.9, trace_norm=12.0)
    assert world.E > e0
    # 零 q → E 单调衰减 (只消耗)
    for _ in range(20):
        world.step_E(0.0, trace_norm=12.0)
    assert world.E < e0


def test_R1_gain_two_sided_calibration(world):
    # R = tanh(ΔE/(2·MAD))/‖迹‖ (113 R1). 消费侧: 同流水, 迹范数翻倍
    # → R 精确减半 (写入 = 单位方向 × 幅度, 音量随系统自己的行为尺度).
    world.E = 1.0
    world.de_mad = None
    r_a = world.step_E(0.9, trace_norm=10.0)
    world.E = 1.0
    world.de_mad = None
    r_b = world.step_E(0.9, trace_norm=20.0)
    assert r_a == pytest.approx(2.0 * r_b)
    # tanh 界: |R| ≤ 1/‖迹‖ (迹 ≥ 1 时 ≤ 1, 契约 0-1 级)
    assert abs(r_a) <= 0.1 + 1e-12
    assert abs(r_a) > 0.0
    # 信号侧: 同流水, 世界噪声地板 (MAD) 更大 → |R| 更小
    world.E = 1.0
    world.de_mad = 0.5
    r_noisy = world.step_E(0.9, trace_norm=10.0)
    assert abs(r_noisy) < abs(r_a)
    # de_mad EMA 收敛: 多步后为正有限
    assert 0.0 < world.de_mad < float("inf")


def test_R1_mad_cold_start(world):
    # 冷启动首笔: de_mad = |ΔE| → |tanh| = tanh(0.5) ≈ 0.462 (平滑入契约区)
    world.E = 1.0
    world.de_mad = None
    world.step_E(0.9, trace_norm=1.0)
    de = 1.0 * (1 - 0.05) + 0.1 * 0.9 - 1.0
    assert world.de_mad == pytest.approx(abs(de))


def test_certify_gate(world):
    assert not world.certify(0.1, 0.9)  # 低于 q_ref → 不认证
    assert world.n_certified == 0
    assert world.certify(0.6, 0.55)  # 高于 q_ref → 认证
    assert world.n_certified == 1
    assert world.world_eps_ema == pytest.approx(0.55)


def test_certify_anchor_ema(world):
    for v in (0.50, 0.52, 0.54):
        world.certify(0.6, v)
    # EMA 收敛到最近值附近 (α=0.995 慢, 三次后 ≈ 加权均值)
    assert 0.50 <= world.world_eps_ema <= 0.54
    assert world.world_eps_mad >= 0.0


def test_state_roundtrip(tmp_path, world):
    # 演化出非平凡动态状态: E 代谢 + 认证锚 + 新颖度历史 + de_mad
    world.step_E(0.7, trace_norm=12.0)
    world.step_E(0.5, trace_norm=11.8)
    for v in (0.50, 0.52):
        world.certify(0.6, v)
    world.record("你好天气很好我们".encode("utf-8"))
    world.record("学习知识需要耐心".encode("utf-8"))
    assert world.de_mad is not None

    st = world.save_state(step=1500, gen_temp=1.0)
    # JSON 侧车可序列化 (exp113 断点续跑的存储形态)
    p = tmp_path / "sidecar.json"
    p.write_text(json.dumps(st), encoding="utf-8")

    # 新世界 (同语料重建) 恢复 → 全部动态状态逐位等价
    p2 = tmp_path / "mini.jsonl"
    w2 = WorldLangPhysics(str(p2), n_char_lines=10, n_trigram_lines=None, top_n=100)
    step, gt = w2.load_state(json.loads(p.read_text(encoding="utf-8")))
    assert (step, gt) == (1500, 1.0)
    assert w2.E == world.E
    assert w2.de_mad == world.de_mad
    assert w2.world_eps_ema == world.world_eps_ema
    assert w2.world_eps_mad == world.world_eps_mad
    assert w2.n_certified == world.n_certified
    assert w2._hist == world._hist  # deque[frozenset] 深等价

    # 恢复后行为连续: 同一发声的新颖税 = 恢复前 (历史已接续)
    dup = "你好天气很好我们".encode("utf-8")
    assert w2.score(dup)["X"] == world.score(dup)["X"] > 0.9


def test_state_roundtrip_old_sidecar_no_de_mad(tmp_path, world):
    # 旧侧车 (111/112 轮, 无 de_mad 键) → 恢复为 None → R1 冷启动
    world.step_E(0.7, trace_norm=12.0)
    st = world.save_state(step=1500, gen_temp=1.0)
    del st["de_mad"]
    p = tmp_path / "sidecar_old.json"
    p.write_text(json.dumps(st), encoding="utf-8")
    p2 = tmp_path / "mini.jsonl"
    w2 = WorldLangPhysics(str(p2), n_char_lines=10, n_trigram_lines=None, top_n=100)
    w2.load_state(json.loads(p.read_text(encoding="utf-8")))
    assert w2.de_mad is None


def test_state_roundtrip_null_anchor(tmp_path, world):
    # 锚未初始化 (从未认证) → JSON null → 恢复仍为 None
    st = world.save_state(step=0, gen_temp=4.0)
    p = tmp_path / "sidecar0.json"
    p.write_text(json.dumps(st), encoding="utf-8")
    p2 = tmp_path / "mini.jsonl"
    w2 = WorldLangPhysics(str(p2), n_char_lines=10, n_trigram_lines=None, top_n=100)
    _, gt = w2.load_state(json.loads(p.read_text(encoding="utf-8")))
    assert gt == 4.0
    assert w2.world_eps_ema is None
    assert w2._hist == world._hist == deque([], maxlen=world.k_history)
