"""
稀疏掩码边界静态扫描.

铁律: mask.py 只做 bool 乘法 / EMA / topk. 禁任何访存路径 (gather/scatter/索引/reshape)
与任何精度逃逸 (fp32/fp64/bf16), 违者即稀疏机制越界.

仅扫真实代码 token: 文档串与注释里出现这些词是说明, 不是越界.
单一事实源 (tests/test_mask_boundary.py 复用).
"""
from __future__ import annotations

import io
import tokenize
from pathlib import Path

FORBIDDEN_TOKENS = (
    "gather(",
    "scatter",
    "index_select",
    "index_add",
    "index_put",
    "index_fill",
    "index_copy",
    "nonzero",
    "to_sparse",
    "sparse_coo",
    "sparse_csr",
    "reshape",
    "as_strided",
    ".float(",
    ".double(",
    ".bfloat16(",
    ".item(",
)


_SKIP = {tokenize.COMMENT, tokenize.STRING} | {
    getattr(tokenize, n) for n in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END") if hasattr(tokenize, n)
}


def code_only(text: str) -> str:
    """把注释与字符串字面量抹成空白, 保留行号与其余代码原样."""
    starts, pos = [], 0
    for line in text.splitlines(keepends=True):
        starts.append(pos)
        pos += len(line)
    buf = list(text)
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type not in _SKIP:
                continue
            (srow, scol), (erow, ecol) = tok.start, tok.end
            a, b = starts[srow - 1] + scol, starts[erow - 1] + ecol
            buf[a:b] = " " * (b - a)
    except (tokenize.TokenError, IndexError):
        pass
    return "".join(buf)


def scan_mask_sources(root: str | Path | None = None) -> list[str]:
    """扫描 mask.py, 返回命中清单 "相对路径:行号: token" (空 = 干净)."""
    base = Path(root) if root is not None else Path(__file__).resolve().parent.parent / "crates" / "hybrid"
    hits: list[str] = []
    for p in sorted(base.rglob("mask.py")):
        for i, line in enumerate(code_only(p.read_text(encoding="utf-8")).splitlines(), 1):
            for tok in FORBIDDEN_TOKENS:
                if tok in line:
                    hits.append(f"{p.relative_to(base)}:{i}: {tok}")
    return hits
