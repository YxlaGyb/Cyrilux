"""
world_lang 世界语言物理仪表单元测试
评分 / 新颖税 / 认证遥测 / 侧车 round-trip / 旧键兼容.
"""
import importlib.util
import json
from collections import deque

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("world_lang", "scripts/world_lang.py")
assert _spec is not None and _spec.loader is not None
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
        f.writelines(json.dumps({"text": t}) + "\n" for t in LINES)
    w = WorldLangPhysics(str(p), n_char_lines=10, n_trigram_lines=None, top_n=100)
    w.q_ref = 0.5  # 手动覆盖 (mini 语料 heldout 空 → 自校准 q_base=0)
    return w


def test_L_coverage_common_chars(world):
    # 全部常用字 (语料内) → L = 1.0
    s = world.score("你好天气很好我们".encode())
    assert s["L"] == 1.0
    # 全生僻字 (不在 top-100) → L = 0.0
    s2 = world.score("龘靐齉爩鱻".encode())
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
    a = "你好天气很好我们出去散步".encode()
    b = b"abcdEFGH1234!@#$"  # 与 A 无共同字节 3-gram
    world._hist.clear()
    world.record(a)
    # 复读同一发声 → 与最近历史并集完全重叠 → X → 1
    assert world.score(a)["X"] > 0.95
    # 全新发声 (与 A 无共同 3-gram) → X = 0
    assert world.score(b)["X"] == 0.0


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


def test_step_E_retired(world):
    # 红线卫生锁: 体外代谢账本已删除 (P1 断链), 不得复生
    assert not hasattr(world, "step_E")
    assert not hasattr(world, "E")
    assert not hasattr(world, "de_mad")


def test_state_roundtrip(tmp_path, world):
    # 演化出非平凡动态状态: 认证锚 + 新颖度历史
    for v in (0.50, 0.52):
        world.certify(0.6, v)
    world.record("你好天气很好我们".encode())
    world.record("学习知识需要耐心".encode())

    st = world.save_state(step=1500, gen_temp=1.0)
    # JSON 侧车可序列化 (断点续跑的存储形态)
    p = tmp_path / "sidecar.json"
    p.write_text(json.dumps(st), encoding="utf-8")

    # 新世界 (同语料重建) 恢复 → 全部动态状态逐位等价
    p2 = tmp_path / "mini.jsonl"
    w2 = WorldLangPhysics(str(p2), n_char_lines=10, n_trigram_lines=None, top_n=100)
    step, gt = w2.load_state(json.loads(p.read_text(encoding="utf-8")))
    assert (step, gt) == (1500, 1.0)
    assert w2.world_eps_ema == world.world_eps_ema
    assert w2.world_eps_mad == world.world_eps_mad
    assert w2.n_certified == world.n_certified
    assert w2._hist == world._hist  # deque[frozenset] 深等价

    # 恢复后行为连续: 同一发声的新颖税 = 恢复前 (历史已接续)
    dup = "你好天气很好我们".encode()
    assert w2.score(dup)["X"] == world.score(dup)["X"] > 0.9


def test_load_state_tolerates_legacy_keys(tmp_path, world):
    # 旧侧车 (P1 前) 含体外 E/de_mad 残留键 → 恢复忽略之, 仪表不复活死状态
    st = world.save_state(step=1500, gen_temp=1.0)
    st["E"] = 0.7
    st["de_mad"] = 0.004
    p = tmp_path / "sidecar_legacy.json"
    p.write_text(json.dumps(st), encoding="utf-8")
    p2 = tmp_path / "mini.jsonl"
    w2 = WorldLangPhysics(str(p2), n_char_lines=10, n_trigram_lines=None, top_n=100)
    step, gt = w2.load_state(json.loads(p.read_text(encoding="utf-8")))
    assert (step, gt) == (1500, 1.0)
    assert not hasattr(w2, "E")
    assert not hasattr(w2, "de_mad")
    assert w2._hist == world._hist


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
