"""训练曲线自动生成 v2 (2026-09-16 审批令: 累积口径 + 换面板).

v1 四面板 (瞬时 echo bigram / q / cost_tot / CJK 比) 三个被判"不服务问题":
瞬时 63B echo 是单点噪声、q 有界抖动、cost_tot 随 batch 抖。v2 改为:
  1. bigram 累积均值 (逐报告点累积; 叠画原始点灰线);
  2. CJK 比累积均值 (同口径; 叠画原始点);
  3. W1 disp (stdout 日志 `disp=` 解析; 权重位移, 非有界量);
  4. C cond (stdout 日志 `cond=` 解析; 白化器条件数).
日志来源: --log (exp115_say stdout 重定向文件); q/cost_tot 仍写入 CSV 备查.
用法: uv run python scripts/plot_train_curves.py --jsonl out/vXX-*/x.jsonl [--log stdout.log] [--out-dir DIR]
"""

import argparse
import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.tmp.f5_baseline import bigram_set, sample_metrics  # noqa: E402


def cum_mean(xs, ys):
    """累积均值 (跳过缺失); 返回与输入等长的累积均值序列."""
    out, s, n = [], 0.0, 0
    for y in ys:
        if y is not None:
            s += y
            n += 1
        out.append(s / n if n else None)
    return out


def parse_log(path):
    """从 stdout 日志解析 W1 disp 与 C cond (ASCII 正则, 编码无关)."""
    raw = open(path, "rb").read()
    t = raw.decode("utf-16-le", errors="replace") if raw[:2] == b"\xff\xfe" else raw.decode("utf-8", errors="replace")
    disp, cond, last_step = [], [], None
    for ln in t.splitlines():
        ms = re.search(r"step (\d+)", ln)
        if ms:
            last_step = int(ms.group(1))
        md = re.search(r"disp=([\d.]+)", ln)
        if md and last_step is not None:
            disp.append((last_step, float(md.group(1))))
        mc = re.search(r"cond=(\d+)", ln)
        if mc and last_step is not None:
            cond.append((last_step, int(mc.group(1))))
    return disp, cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--log", default=None, help="exp115_say stdout 重定向文件 (解析 disp/cond)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--suffix", default="v2")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # 纪律 15

    bgs = bigram_set()
    rows = []
    for line in open(args.jsonl, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "step" not in d:
            continue
        row = {"step": d["step"], "q": d.get("q"), "cost_tot": d.get("cost_tot"),
               "df_mad": d.get("df_mad"), "nov": d.get("nov")}
        gr = d.get("gen_raw")
        if gr:
            m = sample_metrics(bytes.fromhex(gr), bgs)
            row["bigram"] = m["bigram_density"]
            row["cjk_ratio"] = m["cjk_ratio"]
            row["word_hits"] = m["word_hits"]
            row["punct_ratio"] = m["punct_ratio"]
        rows.append(row)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.jsonl))
    tag = os.path.splitext(os.path.basename(args.jsonl))[0]
    csvp = os.path.join(out_dir, f"curves_{tag}.csv")
    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(csvp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # 累积均值 (审批令: 瞬时 63B echo 是单点噪声, 累积口径才看得出方向)
    big_raw = [(r["step"], r["bigram"]) for r in rows if r.get("bigram") is not None]
    cjk_raw = [(r["step"], r["cjk_ratio"]) for r in rows if r.get("cjk_ratio") is not None]
    big_cum = cum_mean(*zip(*big_raw)) if big_raw else []
    cjk_cum = cum_mean(*zip(*cjk_raw)) if cjk_raw else []
    disp, cond = ([], [])
    if args.log:
        disp, cond = parse_log(args.log)

    panels = [
        (big_raw, big_cum, "bigram density: cumulative mean (+ raw pts)"),
        (cjk_raw, cjk_cum, "CJK ratio: cumulative mean (+ raw pts)"),
        (disp, None, "W1 disp (weight displacement, stdout log)"),
        (cond, None, "C condition number (whitener, stdout log)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"training curves v2 (cumulative): {os.path.basename(args.jsonl)}", fontsize=12)
    summary = {}
    for ax, (raw_pts, cum_ys, label) in zip(axes.flat, panels):
        if raw_pts:
            xs = [p[0] for p in raw_pts]
            ys = [p[1] for p in raw_pts]
            if cum_ys is not None:
                ax.plot(xs, ys, marker=".", linewidth=0.8, color="0.75", label="raw")
                ax.plot(xs, cum_ys, linewidth=2, color="C0", label="cumulative mean")
                ax.legend(fontsize=8)
                summary[label.split(":")[0]] = {"raw_first": ys[0], "raw_last": ys[-1],
                                                "cum_first": cum_ys[0], "cum_last": cum_ys[-1]}
            else:
                ax.plot(xs, ys, marker=".", linewidth=1.2, color="C1")
                summary[label.split(" (")[0]] = {"first": ys[0], "last": ys[-1]}
            ax.set_xlabel("step")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    pngp = os.path.join(out_dir, f"curves_{tag}_{args.suffix}.png")
    fig.savefig(pngp, dpi=130)
    plt.close(fig)
    print(json.dumps({"csv": csvp, "png": pngp, "n_rows": len(rows),
                      "n_disp": len(disp), "n_cond": len(cond), "summary": summary},
                     ensure_ascii=True, indent=1), flush=True)


if __name__ == "__main__":
    main()
