"""单步训练 profiler: 只看两个指标.

1) cudaLaunchKernel 的 CPU 耗时  → host 提交 GPU 任务的开销
2) aten:: 系列算子的 CPU 耗时   → 算子本身的开销

默认只采 CPU 活动 (kineto 仍记录 CUDA API 调用), 事件量小、跑得完;
需要设备侧核函数时间时加 --cuda。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.profiler as profiler
from _probe_meta import config_meta

from dataset import DualChannelDataset  # noqa: E402
from model import DensePCNet  # noqa: E402
from pkg.cli.utils import run_dir, run_file  # noqa: E402

SEED_N = 16
LAUNCH_KEYS = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cudaMemcpyAsync", "cudaStreamSynchronize")


def _one_step(net, ds, dev, i: int, last_tail: torch.Tensor) -> torch.Tensor:
    if i % 2 == 1:
        b, _ = ds[i]
        x = b.unsqueeze(0).to(dev)
        net.learn(x)
        return x[0, -SEED_N:]
    net._echo_seed = last_tail.unsqueeze(0)
    net.learn(None, free_run=False)
    return last_tail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/exp115_p3c_ev.safetensors")
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--which", default="echo", choices=["echo", "train"], help="echo=自回归步, train=学习步")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--frw", type=int, default=0, help="覆盖 free_run_window (0=不动)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or run_file("prof_summary.json")

    torch.set_grad_enabled(False)
    dev = "cuda"
    net = DensePCNet.load(args.ckpt).to(dev)
    ds = DualChannelDataset(args.data, max_length=args.max_length, max_samples=1270000, lazy=True)
    if args.frw:
        try:
            net.cfg.free_run_window = args.frw
        except Exception:
            object.__setattr__(net.cfg, "free_run_window", args.frw)

    tail = torch.zeros(SEED_N, dtype=torch.long, device=dev)
    for i in range(1, 3):
        tail = _one_step(net, ds, dev, i, tail)

    acts = [profiler.ProfilerActivity.CPU]
    if args.cuda:
        acts.append(profiler.ProfilerActivity.CUDA)
    cb = None
    if args.trace:
        prof_dir = Path(run_dir()) / "prof"
        prof_dir.mkdir(parents=True, exist_ok=True)
        cb = profiler.tensorboard_trace_handler(str(prof_dir))

    base = 20 if args.which == "echo" else 21  # 20 偶=echo, 21 奇=train
    torch.cuda.synchronize()
    wall0 = time.perf_counter()
    with profiler.profile(activities=acts, on_trace_ready=cb, record_shapes=False, with_stack=False) as prof:
        for i in range(base, base + args.steps):
            tail = _one_step(net, ds, dev, i, tail)
            prof.step()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - wall0) / args.steps

    ka = prof.key_averages()
    st = args.steps

    launch = [e for e in ka if e.key in LAUNCH_KEYS]
    launch_ms = sum(e.self_cpu_time_total for e in launch) / 1000.0 / st
    launch_n = sum(e.count for e in launch) / st
    launch_us = (launch_ms * 1000.0 / launch_n) if launch_n else 0.0

    dev_ms = sum(getattr(e, "self_device_time_total", 0.0) for e in ka) / 1000.0 / st
    dev_top = sorted(ka, key=lambda e: -getattr(e, "self_device_time_total", 0.0))[:10]

    aten = sorted((e for e in ka if e.key.startswith("aten::")), key=lambda e: -e.self_cpu_time_total)
    aten_ms = sum(e.self_cpu_time_total for e in aten) / 1000.0 / st
    other_ms = sum(e.self_cpu_time_total for e in ka) / 1000.0 / st - launch_ms - aten_ms

    rows = [
        {
            "op": e.key,
            "count_per_step": round(e.count / st, 1),
            "self_cpu_ms_per_step": round(e.self_cpu_time_total / 1000.0 / st, 3),
            "avg_us": round(e.self_cpu_time_total / e.count, 2) if e.count else 0.0,
        }
        for e in aten[:15]
    ]

    summary = {
        "which": args.which,
        "steps": st,
        "wall_step_s": round(wall, 4),
        "launch": {
            "keys": [e.key for e in launch],
            "count_per_step": round(launch_n, 1),
            "cpu_ms_per_step": round(launch_ms, 1),
            "avg_us_per_call": round(launch_us, 2),
            "share_of_wall": round(launch_ms / (wall * 1000), 3),
        },
        "device_kernel_ms_per_step": round(dev_ms, 1),
        "gpu_busy_ratio": round(dev_ms / (wall * 1000), 3),
        "top_device": [
            {"op": e.key, "count_per_step": round(e.count / st, 1), "dev_ms": round(getattr(e, "self_device_time_total", 0.0) / 1000.0 / st, 3)}
            for e in dev_top
        ],
        "aten_self_cpu_ms_per_step": round(aten_ms, 1),
        "aten_share_of_wall": round(aten_ms / (wall * 1000), 3),
        "other_host_ms_per_step": round(other_ms, 1),
        "top_aten": rows,
        "config": config_meta(probe="probe_profile.py", args=vars(args)),
    }
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
