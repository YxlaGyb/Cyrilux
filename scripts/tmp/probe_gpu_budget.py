"""GPU 利用预算实测 v2: CUDA event 无 profiler 计时 (profiler 每核记账开销污染墙钟).

只读检查点; 12 步热身 + 8 步计时 (感知/回声交替); 每步记录 wall (event) 与步型;
另跑 1 步轻量 profiler 采核数 (不参与墙钟). 输出 out/probe_gpu_budget.json.
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
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="out/probe_gpu_budget.json")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(0)
    random.seed(0)
    net = DensePCNet.load(args.ckpt).to(dev)
    ds = DualChannelDataset(args.data, max_length=S_MAX, max_samples=1270000, lazy=True)
    idxs = list(range(len(ds)))
    random.Random(0).shuffle(idxs)
    last_tail = None

    def step(i):
        nonlocal last_tail
        if i % 2 == 1:
            b, _ = ds[idxs[(i // 2) % len(idxs)]]
            x = b.unsqueeze(0).to(dev)
            net.learn(x)
            last_tail = x[0, -SEED_N:]
        else:
            net._echo_seed = (
                last_tail.unsqueeze(0)
                if last_tail is not None
                else torch.zeros(1, 1, dtype=torch.long, device=dev)
            )
            net.learn(None, free_run=False)

    for i in range(1, args.warmup + 1):
        step(i)
    torch.cuda.synchronize()

    rows = []
    for i in range(args.warmup + 1, args.warmup + args.steps + 1):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        step(i)
        t1.record()
        torch.cuda.synchronize()
        wall = t0.elapsed_time(t1)
        rows.append({"step": i, "kind": "echo" if i % 2 == 0 else "perc",
                     "wall_ms": round(wall, 1)})
        print(f"step {i} [{rows[-1]['kind']}]: wall={wall:.0f}ms", flush=True)

    # 单步轻量 profiler (核数口径, 不进墙钟统计)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        step(args.warmup + args.steps + 1)
    ka = prof.key_averages()
    n_k = sum(x.count for x in ka if x.self_device_time_total > 0)
    gpu_one = sum(x.self_device_time_total for x in ka if x.self_device_time_total > 0)

    echo_ms = [r["wall_ms"] for r in rows if r["kind"] == "echo"]
    perc_ms = [r["wall_ms"] for r in rows if r["kind"] == "perc"]
    rep = {
        "ckpt": args.ckpt, "rows": rows,
        "echo_mean_ms": round(sum(echo_ms) / len(echo_ms), 1) if echo_ms else None,
        "perc_mean_ms": round(sum(perc_ms) / len(perc_ms), 1) if perc_ms else None,
        "one_step_kernels": n_k, "one_step_gpu_us": round(gpu_one, 1),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"echo_mean={rep['echo_mean_ms']}ms perc_mean={rep['perc_mean_ms']}ms "
          f"kernels(单步)={n_k} gpu(单步)={gpu_one:.0f}µs")
    print(f"单步墙钟/GPU 比: {'—' if not echo_ms else round((sum(echo_ms)+sum(perc_ms))/(len(echo_ms)+len(perc_ms)), 1)}ms/wall")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
