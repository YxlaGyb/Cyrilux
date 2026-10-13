"""文本模态几何预设测试: 预设是信号容器的唯一定义处.

判据 = 与 HybridPCNet 注册的 `_mask_print` / `_utf8_block_cnt` / `_utf8_block_start` 逐位相等
(运行期掩码由预设构造, 二者不得漂移), 以及预设只含形状声明、不含任何语义表.

外加三条守卫: 器官是叶层且不含 torch; 字节域字面量只许留在器官与两把报告尺;
新增的调用点不自行造张量、不做主机同步.
"""

import re
from pathlib import Path

import torch

from crates.modal.text import geometry as g
from crates.model import CyreneModel, HybridPCNet

ROOT = Path(__file__).resolve().parents[1]
ORGAN_DIR = ROOT / "crates" / "modal" / "text"
RULERS = {ROOT / "scripts" / "audit_char_coverage.py", ROOT / "scripts" / "world_lang.py"}
BYTE_DOMAIN_LITERALS = re.compile(r"\b(0x80|0xBF|0xC0|0xC1|0xC2|0xDF|0xE0|0xEF|0xF0|0xF4|0xF5)\b|num_classes=256")
CALL_SITES = {
    "crates/hybrid/forward.py": ("ALPHABET", "LEAD_RANGES", "LEAD_CONTINUATIONS"),
    "crates/hybrid/learning/readout.py": ("ALPHABET", "CONTINUATION_PREFIX"),
    "crates/hybrid/learning/action.py": ("ALPHABET",),
    "crates/hybrid/learning/engine.py": ("PRINTABLE_START",),
}
ALLOC_OR_SYNC = ("torch.tensor(", "torch.as_tensor(", "torch.full(", ".item()", ".cpu()")


def _make_net():
    torch.manual_seed(0)
    cfg = CyreneModel(
        d_l4=64, d_l2=32, d_l3=32, d_l5=64, d_l6=16,
        max_seq_len=16, free_run_window=16, mem_k0=2,
    )
    return HybridPCNet(cfg)


def test_model_masks_are_the_presets():
    """运行期掩码与预设逐位相等 (可打印域 + UTF-8 两张表)."""
    net = _make_net()
    assert net._mask_print.bool().tolist() == [b >= g.PRINTABLE_START for b in range(g.ALPHABET)]
    assert net._utf8_block_cnt.tolist() == list(g.UTF8_BLOCK_CNT)
    assert net._utf8_block_start.tolist() == list(g.UTF8_BLOCK_START)


def test_byte_classes_cover_the_container():
    """续字节与合法起始互斥; 起始域 = ASCII + 2/3/4 字节起始."""
    cont = [b for b in range(g.ALPHABET) if g.is_continuation(b)]
    start = [b for b in range(g.ALPHABET) if g.is_start(b)]
    assert cont == list(range(0x80, 0xC0))
    assert start == list(range(0x00, 0x80)) + list(range(0xC2, 0xF5))
    assert not set(cont) & set(start)
    # 0xC0-0xC1 (overlong) 与 0xF5-0xFF: 两侧都不认 — 冻结口径, 不在此改写
    assert not any(g.is_start(b) or g.is_continuation(b) for b in [0xC0, 0xC1, *range(0xF5, 0x100)])


def test_masks_are_the_forbidden_sets():
    """掩码语义: 字符中途禁非续字节, 字符边界禁续字节与 overlong 起始."""
    for b in range(g.ALPHABET):
        assert bool(g.UTF8_BLOCK_CNT[b]) != g.is_continuation(b)
        assert bool(g.UTF8_BLOCK_START[b]) == (g.is_continuation(b) or b in g.OVERLONG_LEADS)


def test_pad_outside_printable_domain():
    """填充位在可打印域之外: ljust 填充位落进 learn 的目标掩码 (目标字节 <32)."""
    assert (g.ALPHABET, g.PAD_ID, g.PRINTABLE_START) == (256, 0x00, 0x20)
    assert g.PAD_ID < g.PRINTABLE_START


def test_presets_carry_no_semantics():
    """预设只有形状声明: 无张量、无查表容器 (dict/list/str)、无可学习参数."""
    values = vars(g)
    names = [n for n in values if not n.startswith("_") and n != "annotations"]
    assert names
    kinds = {type(values[n]).__name__ for n in names}
    assert kinds <= {"int", "bool", "tuple", "function"}, kinds
    assert not any(isinstance(values[n], torch.Tensor) for n in names)


def test_organ_is_a_leaf_without_torch():
    """器官只声明形状, 不造张量: 不 import torch, 也不 import 网络与工具."""
    files = sorted(ORGAN_DIR.glob("*.py"))
    assert files
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert "torch." not in text, f"{f} 使用 torch: 缓冲构造属 crates.model"
        for line in text.splitlines():
            stmt = line.strip()
            if not (stmt.startswith("import ") or stmt.startswith("from ")):
                continue
            for forbidden in ("torch", "crates.model", "crates.hybrid", "crates.corebrain", "crates.utils", "crates.kernels"):
                assert forbidden not in stmt, f"{f} import {forbidden}"


def test_lead_and_prefix_presets_match_the_frozen_byte_domain():
    """续字节需求表与位前缀判据逐位等价于冻结口径 (0xC2-0xDF→1, 0xE0-0xEF→2, 0xF0-0xF4→3)."""
    for b in range(g.ALPHABET):
        old = 1 if 0xC2 <= b <= 0xDF else 2 if 0xE0 <= b <= 0xEF else 3 if 0xF0 <= b <= 0xF4 else 0
        assert g.lead_length(b) == old, b
        assert (b & 0xC0 != 0x80) == ((b & g.CONTINUATION_PREFIX[0]) != g.CONTINUATION_PREFIX[1]), b
    assert len(g.PRINT_MASK) == g.ALPHABET
    assert list(g.PRINT_MASK) == [b >= g.PRINTABLE_START for b in range(g.ALPHABET)]


def test_byte_domain_literals_live_only_in_the_organ_and_the_rulers():
    """字节域事实只有一个定义处: 器官; 报告尺可自查, 其余生产代码必须读器官."""
    for base in ("crates", "scripts"):
        for f in sorted((ROOT / base).rglob("*.py")):
            if ORGAN_DIR in f.parents or f in RULERS or "tmp" in f.parts:
                continue
            hit = BYTE_DOMAIN_LITERALS.search(f.read_text(encoding="utf-8"))
            assert hit is None, f"{f.relative_to(ROOT)} 重述字节域: {hit.group(0)}"


def test_call_sites_do_not_allocate_or_sync():
    """调用点读器官常量, 不因此新增张量构造与主机同步 (热路径单步成本不变)."""
    for rel, needles in CALL_SITES.items():
        for n, line in enumerate((ROOT / rel).read_text(encoding="utf-8").splitlines(), 1):
            if not any(x in line for x in needles):
                continue
            for bad in ALLOC_OR_SYNC:
                assert bad not in line, f"{rel}:{n} 调用点引入 {bad}"
