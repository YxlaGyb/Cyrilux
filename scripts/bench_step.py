"""
A/B 基准: eager vs torch.compile(mode=reduce-overhead) 全步耗时 + host 派发数

只记录实测值
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from _probe_meta import config_meta
from torch.utils._python_dispatch import TorchDispatchMode

from dataset import ByteDataset
from model import DensePCNet
from pkg.cli.utils import run_file

SEED_N = 16


def _opname(func: Any) -> str:
    try:
        return str(func._schema.name)
    except Exception:
        return str(func)


class OpCounter(TorchDispatchMode):
    def __init__(self) -> None:
        self.n = 0
        self.kinds: Counter[str] = Counter()

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        self.n += 1
        self.kinds[_opname(func)] += 1
        return func(*args, **(kwargs or {}))


def _one_step(net, ds, dev, i: int, last_tail: torch.Tensor) -> torch.Tensor:
    if i % 2 == 1:
        b = ds[i]
        x = b.unsqueeze(0).to(dev)
        net.learn(x)
        return x[0, -SEED_N:]
    net._echo_seed = last_tail.unsqueeze(0)
    net.learn(None, free_run=False)
    return last_tail


def _run(fn, net, ds, dev, steps: int, start: int, tail: torch.Tensor) -> torch.Tensor:
    for i in range(start, start + steps):
        tail = fn(net, ds, dev, i, tail)
    return tail


def _timeit(fn, net, ds, dev, steps: int, start: int, tail: torch.Tensor) -> float:
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _run(fn, net, ds, dev, steps, start, tail)
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps


def _count(fn, net, ds, dev, steps: int, start: int, tail: torch.Tensor) -> tuple[float, list]:
    c = OpCounter()
    with c:
        _run(fn, net, ds, dev, steps, start, tail)
    return c.n / steps, c.kinds.most_common(12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/exp115_p3c_ev.safetensors")
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="eager")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()
    args.out = args.out or run_file("bench_step.json")

    torch.set_grad_enabled(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = DensePCNet.load(args.ckpt).to(dev)
    ds = ByteDataset(args.data, max_length=args.max_length, max_samples=1270000, lazy=True)

    def fresh() -> torch.Tensor:
        return torch.zeros(SEED_N, dtype=torch.long, device=dev)

    rep: dict[str, object] = {"tag": args.tag, "steps": args.steps}

    # --- eager ---
    tail = fresh()
    _run(_one_step, net, ds, dev, 2, 1, tail)  # 预热
    tail = fresh()
    tail = _run(_one_step, net, ds, dev, 2, 1, tail)
    rep["eager_step_s"] = round(_timeit(_one_step, net, ds, dev, args.steps, 20, tail), 4)
    n, kinds = _count(_one_step, net, ds, dev, args.steps, 20, tail)
    rep["eager_dispatch_per_step"] = round(n, 1)
    rep["eager_top_ops"] = kinds

    # --- torch.compile ---
    if args.no_compile:
        rep["compile_ok"] = None
    else:
      try:
        cf = torch.compile(_one_step, mode="reduce-overhead", fullgraph=False)
        tail = fresh()
        _run(cf, net, ds, dev, 3, 1, tail)  # 触发编译 + 预热
        tail = fresh()
        tail = _run(cf, net, ds, dev, 2, 1, tail)
        rep["compiled_step_s"] = round(_timeit(cf, net, ds, dev, args.steps, 20, tail), 4)
        try:
            cn, ckinds = _count(cf, net, ds, dev, args.steps, 20, tail)
            rep["compiled_dispatch_per_step"] = round(cn, 1)
            rep["compiled_top_ops"] = ckinds
        except Exception as e:
            rep["compiled_count_err"] = f"{type(e).__name__}: {str(e)[:300]}"
        rep["compile_ok"] = True
        if "compiled_step_s" in rep:
            rep["speedup"] = round(float(rep["eager_step_s"]) / float(rep["compiled_step_s"]), 3)
      except Exception as e:
        rep["compile_ok"] = False
        rep["compile_err"] = f"{type(e).__name__}: {str(e)[:600]}"

    rep["config"] = config_meta(probe="bench_step.py", args=vars(args))
    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("config", "eager_top_ops", "compiled_top_ops")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
