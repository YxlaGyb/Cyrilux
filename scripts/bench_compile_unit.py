"""
单点编译基准: _predict 单次调用 eager vs torch.compile(mode=reduce-overhead).

continuation 每步调用 _predict 63 次 (free_run_window=64), 故本单点值 × 63 = 自回归段实测换算.
产物增量写入, 超时也能取到已完成阶段.
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

from model import DensePCNet
from pkg.cli.utils import run_file


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


def _timeit(fn, x, reps: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/exp115_p3c_ev.safetensors")
    ap.add_argument("--s", type=int, default=64)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--mode", default="reduce-overhead")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or run_file("bench_compile_unit.json")

    torch.set_grad_enabled(False)
    net = DensePCNet.load(args.ckpt).to("cuda")
    x = torch.randint(0, 256, (1, args.s), device="cuda", dtype=torch.long)

    def f(t: torch.Tensor):
        return net.forward_engine._predict(t, store_state=True, is_inference=True)

    rep: dict[str, object] = {"s": args.s, "reps": args.reps, "mode": args.mode}
    out = Path(args.out)

    # --- eager ---
    f(x)
    rep["eager_call_ms"] = round(_timeit(f, x, args.reps) * 1000, 3)
    c = OpCounter()
    with c:
        f(x)
    rep["eager_dispatch"] = c.n
    rep["eager_top_ops"] = c.kinds.most_common(10)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print("eager done", flush=True)

    # --- compiled ---
    try:
        cf = torch.compile(f, mode=args.mode, fullgraph=False)
        t0 = time.perf_counter()
        cf(x)
        cf(x)
        rep["compile_wall_s"] = round(time.perf_counter() - t0, 2)
        rep["compiled_call_ms"] = round(_timeit(cf, x, args.reps) * 1000, 3)
        rep["speedup"] = round(float(rep["eager_call_ms"]) / float(rep["compiled_call_ms"]), 3)
        try:
            cc = OpCounter()
            with cc:
                cf(x)
            rep["compiled_dispatch"] = cc.n
        except Exception as e:
            rep["compiled_count_err"] = f"{type(e).__name__}: {str(e)[:200]}"
        rep["ok"] = True
    except Exception as e:
        rep["ok"] = False
        rep["err"] = f"{type(e).__name__}: {str(e)[:600]}"

    rep["config"] = config_meta(probe="bench_compile_unit.py", args=vars(args))
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("config", "eager_top_ops")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
