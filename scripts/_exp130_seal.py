"""exp130 回执辅助: 计算 exp129_result.md 的『规范内容』哈希与全部证据表条目。

规范内容定义: 删去 SELF-REF-BEGIN / SELF-REF-END 两个标记行之间的全部内容,
保留两个标记行本身。此定义自洽且可复现 (md 不能把自身哈希写进自身)。
"""
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def canonical_bytes(path):
    raw = open(path, "rb").read().decode("utf-8")
    lines = raw.splitlines(keepends=True)
    out, skip = [], False
    for ln in lines:
        if "SELF-REF-BEGIN" in ln:
            out.append(ln)
            skip = True
            continue
        if "SELF-REF-END" in ln:
            skip = False
            out.append(ln)
            continue
        if not skip:
            out.append(ln)
    return "".join(out).encode("utf-8")


def main():
    L = []
    p = os.path.join(ROOT, "docs", "plan", "exp129_result.md")
    cb = canonical_bytes(p)
    L.append(("docs/plan/exp129_result.md (v2 规范内容)", len(cb),
              hashlib.sha256(cb).hexdigest()))
    p2 = os.path.join(ROOT, "docs", "plan", "exp130_result.md")
    cb2 = canonical_bytes(p2)
    L.append(("docs/plan/exp130_result.md (规范内容)", len(cb2),
              hashlib.sha256(cb2).hexdigest()))
    for rel in (sys.argv[1:] or (
            "scripts/exp130_coherence_offline.py",
            "scripts/exp130_coherence_instr.py",
            "out/exp130_A_offline/exp130_A_coherence.json",
            "out/exp130_A_offline/exp130_A_firstline.txt",
            "out/exp130_A_offline/exp130_report.txt",
            "out/v463-20261022-185708/exp130.json",
            "out/v463-20261022-185708/exp130_telemetry.jsonl",
            "out/v463-20261022-185708/exp130_stdout.log",
            "docs/plan/exp129_result.md",
            "docs/plan/exp130_result.md")):
        fp = os.path.join(ROOT, rel.replace("/", os.sep))
        if not os.path.exists(fp):
            L.append((rel, None, "MISSING"))
            continue
        L.append((rel, os.path.getsize(fp),
                  hashlib.sha256(open(fp, "rb").read()).hexdigest()))
        # 若该文件含自引用标记, 同时给规范内容哈希 (对自引用块的改动免疫)
        if "SELF-REF-BEGIN" in open(fp, encoding="utf-8").read():
            cb = canonical_bytes(fp)
            L.append((rel + " (规范内容)", len(cb),
                      hashlib.sha256(cb).hexdigest()))
    for name, b, h in L:
        print(f"{name}\t{b}\t{h}")


if __name__ == "__main__":
    main()
