"""world_lang 世界语言物理单元测试: 评分 / 新颖税 / E 代谢 / 认证门.

world_lang.py 在 scripts/ (非包), 用 importlib 加载 (与 probe 脚本同法).
"""
import importlib.util
import json

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
    world.step_E(0.9)
    assert world.E > e0
    # 零 q → E 单调衰减 (只消耗)
    for _ in range(20):
        world.step_E(0.0)
    assert world.E < e0


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
