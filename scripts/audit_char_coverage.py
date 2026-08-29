"""audit_char_coverage: 字符覆盖率质证审计 — "输出了任何中文吗?" 的硬测量 (只读).

第 110 轮确立 (用户质询触发): (1) 输出里有没有常见字? (2) 为什么把
\xe9\xa0\xb7 一类字节当中文? (3) 字节模型怎么学中文, 又为何能出 "英文"?
本脚本不解释, 只测量:

  A. 训练日志 40 条原始样本 (exp110c_world_stdout.log) 的字级统计
  B. 终检查点重生成 (种子=语料文本尾 16B, 同 echo 语义, N=24×63B):
     温度 0 (贪心) / 1.0 (自然采样) / 9.5 (实际运行温度, jsonl 末值)
  C. 语料基线 (200 行感知文本, 同统计)
  D. 纯物理下限: UTF-8 状态机 + 均匀随机合法字节, 零学习参照 —
     回答 "合法 CJK 字符里有多少是常见字, 如果什么都不学"

统计口径 (全部按解码后的字符, 非字节):
  CJK 字数 / top-100 / top-1000 / top-3000 语料高频字覆盖率 / ASCII 占比
  常见二字词命中 (我们/可以/他们/… 共 15 词)

全部样本完整解码落盘 out/audit_char_coverage.txt (证据文件, 无截断).
用法: uv run python scripts/audit_char_coverage.py
"""
import ast
import importlib.util
import json
import random
import re
import sys
from collections import Counter

import torch

torch.set_grad_enabled(False)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from model.dense import DensePCConfig, DensePCNet

_spec = importlib.util.spec_from_file_location("probe110", "scripts/probe110_langnoise.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["probe110"] = _mod
_spec.loader.exec_module(_mod)
read_lines = _mod.read_lines
snapshot_state = _mod.snapshot_state
restore_state = _mod.restore_state
DEV = _mod.DEV
DATA = _mod.DATA

N_SEQ = 24
COMMON_WORDS = ["我们", "可以", "他们", "什么", "自己", "知道", "这样", "没有",
                "就是", "时间", "一个", "因为", "所以", "如果", "现在"]


def corpus_char_stats(n_lines=20000):
    cnt = Counter()
    with open(DATA, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            cnt.update(json.loads(line)["text"])
    return cnt


def coverage(chars, top_sets):
    cjk = [c for c in chars if ord(c) > 127]
    ascii_n = len(chars) - len(cjk)
    res = {"total": len(chars), "cjk": len(cjk),
           "ascii_frac": ascii_n / max(1, len(chars))}
    for name, s in top_sets.items():
        res[name] = sum(1 for c in cjk if c in s) / max(1, len(cjk))
    res["word_hits"] = sum(chars.count(w) for w in COMMON_WORDS)
    return res


def physics_null(n_chars=8000, rng=random.Random(11)):
    """UTF-8 状态机 + 均匀随机合法字节: 零学习的覆盖率下限."""
    boundary = list(range(32, 127)) + list(range(0xC2, 0xF5))
    cont = list(range(0x80, 0xC0))
    out = []
    while len(out) < n_chars:
        b0 = rng.choice(boundary)
        if b0 < 0x80:
            out.append(chr(b0))
            continue
        n_cont = 1 if b0 <= 0xDF else 2 if b0 <= 0xEF else 3
        seq = bytes([b0] + [rng.choice(cont) for _ in range(n_cont)])
        out.append(seq.decode("utf-8", errors="ignore"))
    return "".join(out)


def gen_at(net, snap, seeds, temp):
    res = []
    torch.manual_seed(123)
    for s in seeds:
        restore_state(net, snap)
        seed_t = torch.tensor([list(s)], dtype=torch.long, device=DEV)
        out = net.forward_engine.continuation(seed_t, 63, temperature=temp, rep_backstop=False)
        res.append(bytes(int(v) for v in out[0, len(s):].tolist()))
    restore_state(net, snap)
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/exp110c_world.pt")
    ap.add_argument("--log", default="out/exp110c_world_stdout.log")
    ap.add_argument("--out", default="out/audit_char_coverage.txt")
    args = ap.parse_args()

    audit = open(args.out, "w", encoding="utf-8", buffering=1)

    cnt = corpus_char_stats()
    top_sets = {
        "top100": set(c for c, _ in cnt.most_common(100)),
        "top1000": set(c for c, _ in cnt.most_common(1000)),
        "top3000": set(c for c, _ in cnt.most_common(3000)),
    }

    conds = {}

    # A. 训练日志原始样本
    log = open(args.log, encoding="utf-8", errors="replace").read()
    raws = [ast.literal_eval(m) for m in re.findall(r"gen=(b'[^']*')", log)]
    logged = "".join(r.decode("utf-8", errors="ignore") for r in raws)
    conds["A 日志40条(实际运行)"] = (logged, raws)

    # B. 终检查点重生成
    cfg = DensePCConfig(d_input=256, d_act=256, max_seq_len=256)
    net = DensePCNet.load(args.ckpt, cfg).to(DEV)
    net._entropy_sample = False
    snap = snapshot_state(net)
    texts = read_lines(DATA, 5000, 300, 160)
    seeds = [t.encode("utf-8")[:16] for t in texts[:N_SEQ]]
    for temp, name in ((0.0, "B1 贪心 τ=0"), (1.0, "B2 自然 τ=1.0"), (9.5, "B3 实际 τ=9.5")):
        streams = gen_at(net, snap, seeds, temp)
        conds[name] = ("".join(s.decode("utf-8", errors="ignore") for s in streams), streams)

    # C. 语料基线
    corpus_text = "".join(texts[:N_SEQ])
    conds["C 语料基线"] = (corpus_text, None)

    # D. 纯物理下限
    conds["D 纯物理(零学习)"] = (physics_null(), None)

    # ── 统计表 ──
    print(f"{'条件':<20} {'字符数':>6} {'CJK':>6} {'top100':>8} {'top1000':>8} "
          f"{'top3000':>8} {'ASCII%':>7} {'词命中':>5}", flush=True)
    audit.write(f"{'条件':<20} {'字符数':>6} {'CJK':>6} {'top100':>8} {'top1000':>8} "
                f"{'top3000':>8} {'ASCII%':>7} {'词命中':>5}\n")
    for name, (text, _) in conds.items():
        st = coverage(text, top_sets)
        line = (f"{name:<20} {st['total']:>6} {st['cjk']:>6} {st['top100']:>8.3f} "
                f"{st['top1000']:>8.3f} {st['top3000']:>8.3f} "
                f"{st['ascii_frac']:>7.3f} {st['word_hits']:>5}")
        print(line, flush=True)
        audit.write(line + "\n")

    # ── 全样本完整解码 (证据) ──
    audit.write("\n" + "=" * 70 + "\n全样本完整解码 (无截断, 逐条)\n" + "=" * 70 + "\n")
    for name, (text, streams) in conds.items():
        audit.write(f"\n──── {name} ────\n")
        if streams:
            for i, s in enumerate(streams):
                audit.write(f"[{i:02d}] {s.decode('utf-8', errors='replace')!r}\n")
        else:
            audit.write(text[:600] + " …\n")

    audit.close()
    print("\n完整证据: out/audit_char_coverage.txt", flush=True)


if __name__ == "__main__":
    main()
