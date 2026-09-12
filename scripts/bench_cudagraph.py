"""
CUDA Graph 实测: _predict 能否被图捕获, 捕获代价, 回放多快.

逐档 S 各建一张图 (真实 continuation 的 S 是 1..64 递增的).
计时口径: 同一静态输入, eager 单次调用 vs graph.replay().
注意: _predict 会改内部状态, replay 是"重复同一段计算", 仅用于计时/可行性判定.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from _probe_meta import config_meta

from model import DensePCNet
from pkg.cli.utils import run_file


def _timeit(fn, reps: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def _finite(out) -> bool:
    if isinstance(out, dict):
        return all(torch.isfinite(v).all().item() for v in out.values() if torch.is_tensor(v))
    return bool(torch.isfinite(out).all().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/exp115_p3c_ev.safetensors")
    ap.add_argument("--s-list", default="16,32,64")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or run_file("bench_cudagraph.json")

    torch.set_grad_enabled(False)
    net = DensePCNet.load(args.ckpt).to("cuda")

    def f(t: torch.Tensor):
        return net.forward_engine._predict(t, store_state=True, is_inference=True)

    rows = []
    for s in [int(v) for v in args.s_list.split(",")]:
        x = torch.randint(0, 256, (1, s), device="cuda", dtype=torch.long)
        row: dict[str, object] = {"s": s}
        f(x)
        row["eager_call_ms"] = round(_timeit(lambda: f(x), args.reps) * 1000, 3)

        static_in = x.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                f(static_in)
        torch.cuda.current_stream().wait_stream(side)

        try:
            g = torch.cuda.CUDAGraph()
            t0 = time.perf_counter()
            with torch.cuda.graph(g):
                static_out = f(static_in)
            row["capture_wall_s"] = round(time.perf_counter() - t0, 3)
            row["capture_ok"] = True
            row["replay_ms"] = round(_timeit(lambda: g.replay(), args.reps) * 1000, 3)
            row["speedup_vs_eager"] = round(float(row["eager_call_ms"]) / float(row["replay_ms"]), 2)
            # 活性检查: 换输入后回放, 输出应随之变化
            static_in.copy_(torch.randint(0, 256, (1, s), device="cuda", dtype=torch.long))
            g.replay()
            row["out_finite"] = _finite(static_out)
        except Exception as e:
            row["capture_ok"] = False
            row["err"] = f"{type(e).__name__}: {str(e)[:400]}"
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    rep = {"rows": rows, "config": config_meta(probe="bench_cudagraph.py", args=vars(args))}
    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
