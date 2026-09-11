"""分步型计时: 学习步 (learn(x)) vs 自回归步 (learn(None)) —— 不开 profiler."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from _probe_meta import config_meta

from dataset import ByteDataset  # noqa: E402
from model import DensePCNet  # noqa: E402
from pkg.cli.utils import run_file  # noqa: E402

SEED_N = 16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/exp115_p3c_ev.safetensors")
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or run_file("bench_split_steps.json")

    torch.set_grad_enabled(False)
    dev = "cuda"
    net = DensePCNet.load(args.ckpt).to(dev)
    ds = ByteDataset(args.data, max_length=args.max_length, max_samples=1270000, lazy=True)

    def train_step(i: int, tail: torch.Tensor) -> torch.Tensor:
        b = ds[i]
        x = b.unsqueeze(0).to(dev)
        net.learn(x)
        return x[0, -SEED_N:]

    def echo_step(i: int, tail: torch.Tensor) -> torch.Tensor:
        net._echo_seed = tail.unsqueeze(0)
        net.learn(None, free_run=False)
        return tail

    rep: dict[str, object] = {"config": config_meta(probe="bench_split_steps.py", args=vars(args))}
    tail = torch.zeros(SEED_N, dtype=torch.long, device=dev)
    for i in (1, 2, 3, 4):  # 预热
        tail = train_step(i, tail) if i % 2 else echo_step(i, tail)

    for kind, fn, base in (("train", train_step, 101), ("echo", echo_step, 200)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for k in range(args.steps):
            tail = fn(base + k * 2, tail)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / args.steps
        rep[f"{kind}_step_s"] = round(dt, 4)
        rep[f"{kind}_s_per_frame"] = round(dt / 256, 5)
        print(f"{kind}: {dt:.4f} s/step", flush=True)

    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
