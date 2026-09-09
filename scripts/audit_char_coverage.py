"""
audit_char_coverage 字符覆盖率质证审计
用法: uv run python scripts/audit_char_coverage.py --ckpt out/exp113_say.pt \
        --log out/exp113_say.log --temps 0.0,1.8,9.5
--temps: B 系列重生成温度列表 (B1 = 首个恒贪心, 末位恒对照; 中间位
113 起传实际末段 τ — 标签随参数自描述).
"""
import ast
import importlib.util
import json
import random
import re
import sys
from collections import Counter

import torch

from pkg.cli.utils import run_file

torch.set_grad_enabled(False)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from model import CyreneModel, DensePCNet

# 原 import probe110_langnoise (110 波已删): 内联等价小工具 (功能不变)
DATA = "dataset/pretrain_t2t_mini.jsonl"
DEV = torch.device("cuda")


def read_lines(data_path, n_lines=5000, max_len=300, min_len=160):
    """语料行读取: 前 n_lines 行中长度 [min_len, max_len] 的文本."""
    texts = []
    with open(data_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            t = json.loads(line)["text"]
            if min_len <= len(t) <= max_len:
                texts.append(t)
    return texts


def snapshot_state(net):
    """全状态快照 (生成重放前冻结, 温度系重生成须同一状态)."""
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def restore_state(net, snap):
    for k, v in snap.items():
        b = getattr(net, k, None)
        if b is not None and b.shape == v.shape:
            b.copy_(v)

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
    ap.add_argument("--out", default=None)
    ap.add_argument("--temps", default="0.0,1.0,9.5",
                    help="B 系列重生成温度 (逗号分隔; B1 首位恒贪心, 末位恒对照)")
    args = ap.parse_args()
    args.out = args.out or run_file("audit_char_coverage.txt")

    audit = open(args.out, "w", encoding="utf-8", buffering=1)

    cnt = corpus_char_stats()
    top_sets = {
        "top100": set(c for c, _ in cnt.most_common(100)),
        "top1000": set(c for c, _ in cnt.most_common(1000)),
        "top3000": set(c for c, _ in cnt.most_common(3000)),
    }

    conds = {}

    # A. 训练日志原始样本 (113 起日志格式 = 解码文本 repr, 111 及以前 = 字节 repr)
    log = open(args.log, encoding="utf-8", errors="replace").read()
    raws = [ast.literal_eval(m) for m in re.findall(r"gen=(b'[^']*')", log)]
    logged = "".join(r.decode("utf-8", errors="ignore") for r in raws)
    strs = [ast.literal_eval(m) for m in re.findall(r"gen=('[^']*')", log)]
    logged += "".join(strs)
    conds["A 日志40条(实际运行)"] = (logged, raws)

    # B. 终检查点重生成
    # config=None: 按检查点形状派生 (修剪后检查点兼容路径 — 显式 cfg 会走切片补齐,
    # 产生 _mem_m 列数与 active_size 错位的 Frankenstein)
    net = DensePCNet.load(args.ckpt).to(DEV)
    net._entropy_sample = False
    snap = snapshot_state(net)
    texts = read_lines(DATA, 5000, 300, 160)
    seeds = [t.encode("utf-8")[:16] for t in texts[:N_SEQ]]
    temps = [float(v) for v in args.temps.split(",") if v.strip()]
    for j, temp in enumerate(temps):
        if j == 0:
            name = f"B1 贪心 τ={temp:g}"
        elif j == len(temps) - 1:
            name = f"B{j + 1} 对照 τ={temp:g}"
        else:
            name = f"B{j + 1} τ={temp:g}"
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
    print(f"\n完整证据: {args.out}", flush=True)


if __name__ == "__main__":
    main()
