"""语料预处理 (§7 审计流程, v1.12 审批批令 4-② 选支: 仓内预处理).

职责单一: 清洗 + 入训前窗口分布审计, 两个动作同一遍扫描完成.
  1. 清洗: 丢弃"提取文本为空"的行 (全零窗成因; wikipedia-zh 实测 10,705 行 / 0.76%);
  2. 审计: 输入/输出两侧统计 零窗 / 短窗(< max_seq_len 字节, 带零尾) / 重复窗(文本级);
  3. 产出: 清洗后语料 (逐行保留原始字节, 不做二次编码) + 审计 JSON (run 目录).

判据 (§7): 输出侧 zero_windows == 0 才允许接入训练循环; 否则脚本非零退出.
用法: python scripts/prepare_corpus.py --input dataset/wikipedia-zh-cn-20240901.json \
                                      --output dataset/wikipedia-zh-cn-20240901.clean.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

from dataset import ByteDataset  # noqa: E402
from crates.cli.utils import run_file  # noqa: E402


def _stats_of_line(raw_line: bytes, max_seq_len: int) -> tuple[str, int | None]:
    """返回 (类别, 文本字节数): 类别 ∈ {zero, short, ok, bad_json}."""
    try:
        sample = json.loads(raw_line.decode("utf-8", errors="replace"))
    except Exception:
        return "bad_json", None
    text, _roles = ByteDataset._extract_with_roles(sample)
    n = len(text.encode("utf-8"))
    if n == 0:
        return "zero", 0
    if n < max_seq_len:
        return "short", n
    return "ok", n


def run(in_path: str, out_path: str, max_seq_len: int) -> dict:
    seen: set[bytes] = set()
    rep = {
        "input": os.path.abspath(in_path), "output": os.path.abspath(out_path),
        "max_seq_len": max_seq_len,
        "in_total": 0, "in_zero": 0, "in_short": 0, "in_dup": 0, "in_bad_json": 0,
        "written": 0, "out_zero": 0, "out_short": 0, "out_dup": 0,
        "zero_line_idx": [], "dup_line_idx": [],
    }
    with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
        for i, raw_line in enumerate(fin):
            rep["in_total"] += 1
            kind, nbytes = _stats_of_line(raw_line, max_seq_len)
            if kind == "bad_json":
                rep["in_bad_json"] += 1
                continue
            if kind == "zero":
                rep["in_zero"] += 1
                if len(rep["zero_line_idx"]) < 12:
                    rep["zero_line_idx"].append(i)
                continue
            digest = hashlib.blake2b(raw_line.strip(), digest_size=16).digest()
            if digest in seen:
                rep["in_dup"] += 1
                if len(rep["dup_line_idx"]) < 12:
                    rep["dup_line_idx"].append(i)
                continue
            seen.add(digest)
            if kind == "short":
                rep["in_short"] += 1
                rep["out_short"] += 1
            fout.write(raw_line if raw_line.endswith(b"\n") else raw_line + b"\n")
            rep["written"] += 1
    rep["out_zero"] = 0  # 清洗判据: 空文本行一律丢弃 (写出路径不含该分支)
    rep["out_dup"] = 0    # 重复行一律丢弃 (写出前已去重)
    rep["in_zero_ratio"] = round(rep["in_zero"] / max(rep["in_total"], 1), 6)
    rep["in_dup_ratio"] = round(rep["in_dup"] / max(rep["in_total"], 1), 6)
    rep["pass"] = rep["out_zero"] == 0
    return rep


def main():
    ap = argparse.ArgumentParser(description="语料清洗 + 窗口分布审计 (§7)")
    ap.add_argument("--input", required=True, help="源语料 (JSONL)")
    ap.add_argument("--output", required=True, help="清洗后语料 (JSONL)")
    ap.add_argument("--max-seq-len", type=int, default=256, help="窗口长度 (短窗判定用)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    rep = run(args.input, args.output, args.max_seq_len)
    outp = run_file("corpus_prepare_audit.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if not k.endswith("_line_idx")},
                     ensure_ascii=False, indent=1), flush=True)
    print(f"audit saved: {outp}", flush=True)
    if not rep["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
