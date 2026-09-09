"""P3-a 术后探针: τ 时序列 (健康段 → 诱导饥荒段).

只读检查点 (不落盘, 绝不调 maybe_prune).
判据: ① 健康段 τ 落 [1.8, 2.5] (β_d=4 标定: d≈+0.1-0.3 → τ≈2.0-2.4, 复现历史"粘 2.00"
操作点 — 映射兑现了想升温的意图); ② 饥荒开端 τ 下探 (应激≈1 + ε_diff 转负) 至 <1.2 但
止于 ≥0.9 软带下限; ③ 全程零 NaN.
输出: out/probe_p3a_tau.json.
"""

import argparse
import json
import os
import random
import sys

import torch

torch.set_grad_enabled(False)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from model import DensePCNet  # noqa: E402
from dataset import DualChannelDataset  # noqa: E402

S_MAX = 256
SEED_N = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/exp115_p3c_ev.pt")
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--normal-steps", type=int, default=400)
    ap.add_argument("--famine-steps", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/probe_p3a_tau.json")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    net = DensePCNet.load(args.ckpt).to(dev)
    print(f"P3a: ckpt={args.ckpt} (τ 纯函数, 无存量状态)", flush=True)
    ds = DualChannelDataset(args.data, max_length=S_MAX, max_samples=1270000, lazy=True)
    idxs = list(range(len(ds)))
    random.Random(args.seed).shuffle(idxs)

    def run(n, mode):
        last_tail = None
        seq = []
        for i in range(1, n + 1):
            if i % 2 == 1:
                if mode == "normal":
                    b, _ = ds[idxs[(i // 2) % len(idxs)]]
                    x = b.unsqueeze(0).to(dev)
                else:
                    x = torch.randint(0, 256, (1, S_MAX), dtype=torch.long, device=dev)
                net.learn(x)
                last_tail = x[0, -SEED_N:]
            else:
                net._echo_seed = last_tail.unsqueeze(0)
                net.learn(None, free_run=False)
                gt = float(net._gen_temp.item())
                st = float(getattr(net, "_metab_stress", torch.tensor(0.0)).item())
                lm = float(getattr(net, "_lm_eps", torch.tensor(0.0)).item())
                seq.append({"step": i, "tau": gt, "stress": st, "eps_lm": lm})
        return seq

    normal = run(args.normal_steps, "normal")
    print(f"正常段完成: n_echo={sum(1 for r in normal if r), }", flush=True)
    famine = run(args.famine_steps, "famine_rand")
    print(f"饥荒段完成", flush=True)

    def stats(seq):
        if not seq:
            return {"n": 0, "tau_mean": float("nan"), "tau_min": float("nan"),
                    "tau_max": float("nan"), "tau_p10": float("nan"),
                    "tau_p90": float("nan"), "stress_mean": float("nan"),
                    "stress_max": float("nan")}
        ts = [r["tau"] for r in seq]
        ss = [r["stress"] for r in seq]
        ts_s = sorted(ts)
        return {
            "n": len(ts), "tau_mean": round(sum(ts) / len(ts), 4),
            "tau_min": round(min(ts), 4), "tau_max": round(max(ts), 4),
            "tau_p10": round(ts_s[int(0.1 * len(ts_s))], 4),
            "tau_p90": round(ts_s[int(0.9 * len(ts_s))], 4),
            "stress_mean": round(sum(ss) / len(ss), 4),
            "stress_max": round(max(ss), 4),
        }

    ns, fs = stats(normal), stats(famine)
    # 饥荒开端冲击段 (前 80 回声步): 应激抬升 + τ 下探
    onset = stats(famine[:80])
    report = {
        "ckpt": args.ckpt,
        "normal": ns, "famine": fs, "famine_onset": onset,
        "tau_series": normal + famine,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"normal": ns, "famine": fs, "famine_onset": onset}, indent=1))
    print(f"written {args.out}")
    # 判据自检
    ok = (
        ns["n"] > 0 and ns["tau_mean"] <= 2.5 + 1e-3 and ns["tau_min"] >= 0.9 - 1e-3
        and fs["tau_min"] >= 0.9 - 1e-3 and fs["tau_max"] <= 2.5 + 1e-3
    )
    print(f"判据自检: 软带 (min≥0.9, max≤2.5) {'PASS' if ok else 'FAIL'}")
    print(f"健康段 τ_mean={ns['tau_mean']} (目标 [1.8,2.5]) "
          f"{'PASS' if 1.8 <= ns['tau_mean'] <= 2.5 else 'CHECK'}")
    print(f"饥荒开端 τ_min={onset['tau_min']} (目标 <1.2 但 ≥0.9) "
          f"{'PASS' if 0.9 - 1e-3 <= onset['tau_min'] < 1.2 else 'CHECK'}")


if __name__ == "__main__":
    main()
