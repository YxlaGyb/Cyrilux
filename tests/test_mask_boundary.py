"""
稀疏掩码边界守卫.

锁三件事:
1. crates/hybrid/mask.py 零越界 token (无访存路径, 无精度逃逸);
2. mask.py 只依赖 torch 与 crates.kernels (不反指编排层);
3. 扫描器必须能发现植入的越界 token (防扫描器失活).
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("mask_boundary", "scripts/mask_boundary.py")
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FORBIDDEN_TOKENS = _mod.FORBIDDEN_TOKENS
scan_mask_sources = _mod.scan_mask_sources


def test_mask_sources_zero_boundary_tokens():
    hits = scan_mask_sources()
    assert hits == [], f"掩码越界 token 命中 (访存路径/精度逃逸): {hits}"


def test_mask_boundary_tokens_nonempty():
    assert "gather(" in FORBIDDEN_TOKENS
    assert "to_sparse" in FORBIDDEN_TOKENS
    assert ".float(" in FORBIDDEN_TOKENS


def test_mask_only_depends_on_torch_and_kernels():
    mask = Path(__file__).resolve().parent.parent / "crates" / "hybrid" / "mask.py"
    for line in mask.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        assert (
            "torch" in stripped or "crates.kernels" in stripped or "__future__" in stripped
        ), f"mask.py 越界依赖: {stripped}"


def test_scanner_detects_planted_token(tmp_path):
    (tmp_path / "mask.py").write_text("x = y.gather(0, idx)\n", encoding="utf-8")
    assert scan_mask_sources(tmp_path) == ["mask.py:1: gather("]


def test_scanner_ignores_token_in_docstring_and_comment(tmp_path):
    (tmp_path / "mask.py").write_text(
        '"""禁 gather/scatter/reshape/nonzero."""\n# 也不要 to_sparse\nx = 1\n',
        encoding="utf-8",
    )
    assert scan_mask_sources(tmp_path) == []
