"""
语言通用性红线静态扫描
单一事实源 (tests/test_redline.py 与 arch_guard 共用).
"""
from __future__ import annotations

from pathlib import Path

REDLINE_TOKENS = (
    "step_E",        # 体外代谢账本
    "inject_world",  # 体外注入接口
    "_world_R",      # 体外生存信号字段
    "_world_E",      # 体外能量账本字段 (含 _world_E_ref 等家族)
    "world_lang",    # 报告仪表禁止被核心库 import (分层解耦 + 红线)
)


def scan_model_sources(root: str | Path | None = None) -> list[str]:
    """扫描 model/ 活动源码, 返回命中清单 "相对路径:行号: token" (空 = 干净)."""
    base = Path(root) if root is not None else Path(__file__).resolve().parent.parent / "model"
    hits: list[str] = []
    for p in sorted(base.rglob("*.py")):
        if "_archived_sparse" in p.parts:
            continue
        text = p.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            for tok in REDLINE_TOKENS:
                if tok in line:
                    hits.append(f"{p.relative_to(base)}:{i}: {tok}")
    return hits
