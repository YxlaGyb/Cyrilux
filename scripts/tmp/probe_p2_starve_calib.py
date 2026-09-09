"""P2 术前标定探针: 平台期 vs 诱导饥荒的 E 簇长分布.

只读检查点 (不落盘), 绝不调用 maybe_prune (探针不得修改模型).
输出: out/probe_p2_starve_calib.json — 候选死线 {−0.005, −0.01, −0.02, −0.04}
的低于线簇长分布 (max/p999/p50) + E 分位数, 供标定 metab_death_line/metab_death_steps.
"""

import argparse
import json
import random
import sys
import os

import torch

torch.set_grad_enabled(False)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from model import CyreneModel, DensePCNet  # noqa: E402
from dataset import DualChannelDataset  # noqa: E402

S_MAX = 256
SEED_N = 16


def _seg_learn(net, ds, dev, steps, mode, rng, drift_tensor=None):
    """感知/回声交替 (exp115 同款节奏), 返回每步 E 序列."""
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    last_tail = None
    es = []
    fs = []
    for i in range(1, steps + 1):
        if i % 2 == 1:
            if mode == "normal":
                b, _ = ds[idxs[(i // 2) % len(idxs)]]
                x = b.unsqueeze(0).to(dev)
            elif mode == "famine_rand":
                x = torch.randint(0, 256, (1, S_MAX), dtype=torch.long, device=dev)
            elif mode == "famine_shift":
                b, _ = ds[idxs[(i // 2) % len(idxs)]]
                x = ((b + 1) % 256).unsqueeze(0).to(dev)
            else:  # famine_drift: 时间核渐进放大 → F 持续退化
                b, _ = ds[idxs[(i // 2) % len(idxs)]]
                x = b.unsqueeze(0).to(dev)
                with torch.no_grad():
                    drift_tensor.mul_(1.005)
            net.learn(x)
            last_tail = x[0, -SEED_N:]
        else:
            net._echo_seed = (
                last_tail.unsqueeze(0)
                if last_tail is not None
                else torch.zeros(1, 1, dtype=torch.long, device=dev)
            )
            net.learn(None, free_run=False)
        es.append(float(net._metab_E.item()))
        # P3-b 分账: 取刚运行行为的 F 轨 (odd=感知, even=回声)
        if i % 2 == 1:
            fs.append(float(net._metab_F_prev_perc.item()))
        else:
            fs.append(float(net._metab_F_prev_eco.item()))
    return es, fs


def clusters(seq, line):
    runs = []
    cur = 0
    for e in seq:
        if e < line:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/exp115_say.pt")
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--normal-steps", type=int, default=1000)
    ap.add_argument("--famine-steps", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/probe_p2_starve_calib.json")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=S_MAX, lm_freeze_w1=False)
    net = DensePCNet.load(args.ckpt, cfg).to(dev)
    ds = DualChannelDataset(args.data, max_length=S_MAX, max_samples=1270000, lazy=True)
    rng = random.Random(args.seed)

    lines = (-0.005, -0.01, -0.02, -0.04)
    report = {"ckpt": args.ckpt, "segments": {}}

    # 顺序: 平台 → 三种诱导饥荒 (同一机体, 饥荒都是"打在其平台期上")
    segs = {}
    segs["normal"] = _seg_learn(net, ds, dev, args.normal_steps, "normal", rng)
    segs["famine_rand"] = _seg_learn(net, ds, dev, args.famine_steps, "famine_rand", rng)
    segs["famine_shift"] = _seg_learn(net, ds, dev, args.famine_steps, "famine_shift", rng)
    segs["famine_drift"] = _seg_learn(net, ds, dev, args.famine_steps, "famine_drift",
                                      rng, drift_tensor=net.W_t4.data)

    for mode, (es, fs) in segs.items():
        q = sorted(es)

        def pct(p):
            return q[min(len(q) - 1, int(p * len(q)))]

        entry = {
            "n": len(es),
            "mean": round(sum(es) / len(es), 6),
            "min": round(min(es), 6),
            "max": round(max(es), 6),
            "p2": round(pct(0.02), 6),
            "p50": round(pct(0.5), 6),
            "p98": round(pct(0.98), 6),
            "f_max": round(max(fs), 6),
            "f_min": round(min(fs), 6),
            "clusters": {},
        }
        for line in lines:
            cl = sorted(clusters(es, line))
            entry["clusters"][str(line)] = {
                "n_clusters": len(cl),
                "max_run": cl[-1] if cl else 0,
                "p999": cl[min(len(cl) - 1, int(0.999 * len(cl)))] if cl else 0,
                "p50": cl[len(cl) // 2] if cl else 0,
                "longest_20": cl[-20:],
            }
        report["segments"][mode] = entry
        print(f"[{mode}] n={entry['n']} mean={entry['mean']:.5f} "
              f"min={entry['min']:.5f} p98={entry['p98']:.5f} "
              f"F∈[{entry['f_min']:.4g},{entry['f_max']:.4g}]")
        for line in lines:
            c = entry["clusters"][str(line)]
            print(f"  line {line:>7}: n_clusters={c['n_clusters']:>3} "
                  f"max_run={c['max_run']:>3} p999={c['p999']:>3} p50={c['p50']:>3}")

    # 基线锚死亡判据重放 (CPU, 镜像模型语义): base 起 0; 哨兵步跳过; cold (base≤0):
    # base=f 不计数; 否则 base ← (1−β)base+βf 后判 high = f > base·(1+κ).
    # 簇长按分段切分 — 平台不触发而饥荒触发即契约成立.
    margin_grid = (0.1, 0.2, 0.5, 1.0)
    beta = 0.002  # metab_base_rate 默认 (probe 前端重放; 留档 f_series 供换 β 免重跑)
    full_f = [f for (_, fs) in segs.values() for f in fs]
    seg_lens = [len(fs) for (_, fs) in segs.values()]
    seg_modes = list(segs.keys())
    report["base_replay"] = {"beta": beta, "margins": {str(k): {} for k in margin_grid}}
    for k in margin_grid:
        base = 0.0
        hi = []
        for i, f in enumerate(full_f):
            if i == 0:  # 首步哨兵 (只登记 F_prev)
                hi.append(False)
                continue
            if base <= 0:
                base = f
                hi.append(False)
                continue
            base = base * (1.0 - beta) + beta * f
            hi.append(f > base * (1.0 + k))
        off = 0
        for mode, n in zip(seg_modes, seg_lens):
            seg_hi = hi[off : off + n]
            off += n
            runs = []
            cur = 0
            for h in seg_hi:  # True 连续段
                if h:
                    cur += 1
                elif cur:
                    runs.append(cur)
                    cur = 0
            if cur:
                runs.append(cur)
            cl = sorted(runs)
            report["base_replay"]["margins"][str(k)][mode] = {
                "n_high": int(sum(seg_hi)),
                "n_clusters": len(cl),
                "max_run": cl[-1] if cl else 0,
                "p50": cl[len(cl) // 2] if cl else 0,
                "longest_15": cl[-15:],
            }
            print(f"  κ={k:.1f} [{mode:<12}] n_high={sum(seg_hi):>4}/ {n:<4} "
                  f"clusters={len(cl):>3} max_run={cl[-1] if cl else 0:>3} "
                  f"p50={cl[len(cl) // 2] if cl else 0:>3}")
    # F 序列留档 (供后续换 β/κ 免重跑)
    report["f_series"] = {m: [round(v, 6) for v in segs[m][1]] for m in seg_modes}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
