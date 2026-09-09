"""数据加载占比实测: ds[i] 取数 / .to(dev) 拷贝 / 模型本身 三者分开计时."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from _probe_meta import config_meta

from dataset import DualChannelDataset  # noqa: E402
from model import DensePCNet  # noqa: E402
from pkg.cli.utils import run_file  # noqa: E402

SEED_N = 16


def _t(fn, n: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/exp115_p3c_ev.pt")
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or run_file("bench_data_load.json")

    torch.set_grad_enabled(False)
    dev = "cuda"
    net = DensePCNet.load(args.ckpt).to(dev)
    ds = DualChannelDataset(args.data, max_length=args.max_length, max_samples=1270000, lazy=True)

    # 预热
    b, _ = ds[1]
    x = b.unsqueeze(0).to(dev)
    for _ in range(2):
        net.learn(x)

    fetch = _t(lambda: ds[3], args.n)
    move = _t(lambda: ds[3][0].unsqueeze(0).to(dev), args.n)
    model_only = _t(lambda: net.learn(x), args.n)
    full = _t(lambda: net.learn(ds[3][0].unsqueeze(0).to(dev)), args.n)

    rep = {
        "n_reps": args.n,
        "fetch_cpu_s": round(fetch, 5),
        "fetch_plus_h2d_s": round(move, 5),
        "h2d_only_s": round(move - fetch, 5),
        "model_only_s": round(model_only, 5),
        "full_train_step_s": round(full, 5),
        "fetch_share": round(fetch / full, 4),
        "h2d_share": round((move - fetch) / full, 4),
        "model_share": round(model_only / full, 4),
        "config": config_meta(probe="bench_data_load.py", args=vars(args)),
    }
    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "config"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
