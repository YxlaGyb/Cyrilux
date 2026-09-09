"""D1 发射开销测量 + torch.compile 评估 (P3-2).

先量准每步 learn 的 Python→算子派发次数, 再判断 compile / CUDA graph 是否可行.
所有失败都记录到产物, 不静默.
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

from dataset import DualChannelDataset  # noqa: E402
from model import DensePCNet  # noqa: E402
from pkg.cli.utils import run_file  # noqa: E402

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


def _one_step(net, ds, dev, i: int, last_tail: torch.Tensor):
    """交替 learn(x) / learn(None) 一步, 返回新的 tail."""
    if i % 2 == 1:
        b, _ = ds[i]
        x = b.unsqueeze(0).to(dev)
        net.learn(x)
        return x[0, -SEED_N:]
    net._echo_seed = last_tail.unsqueeze(0)
    net.learn(None, free_run=False)
    return last_tail


def _measure(net, ds, dev, steps: int, start: int, last_tail):
    counter = OpCounter()
    with counter:
        for i in range(start, start + steps):
            last_tail = _one_step(net, ds, dev, i, last_tail)
    return counter, last_tail


def _timeit(net, ds, dev, steps: int, start: int, last_tail) -> float:
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(start, start + steps):
        last_tail = _one_step(net, ds, dev, i, last_tail)
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/exp115_p3c_ev.pt")
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or run_file("dispatch_probe.json")

    torch.set_grad_enabled(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = DensePCNet.load(args.ckpt).to(dev)
    ds = DualChannelDataset(args.data, max_length=args.max_length, max_samples=1270000, lazy=True)
    last_tail = torch.zeros(SEED_N, dtype=torch.long, device=dev)

    for i in range(1, 3):  # 预热
        last_tail = _one_step(net, ds, dev, i, last_tail)

    counter, last_tail = _measure(net, ds, dev, args.steps, 1, last_tail)
    base_s = _timeit(net, ds, dev, args.steps, 20, last_tail)

    report: dict[str, object] = {
        "dispatch_per_step": round(counter.n / args.steps, 1),
        "top_ops": counter.kinds.most_common(20),
        "n_distinct_ops": len(counter.kinds),
        "baseline_step_s": round(base_s, 4),
        "steps": args.steps,
    }

    # compile 可行性评估: 对纯前向子图做 try/except, 结果如实记录
    compile_info: dict[str, object] = {}
    try:
        b, _ = ds[1]
        x = b.unsqueeze(0).to(dev)
        fwd = net.forward_engine.forward
        eager = OpCounter()
        with eager:
            fwd(x)
        report["forward_dispatch"] = eager.n

        def pure_fwd(t: torch.Tensor):
            return fwd(t)

        cfwd = torch.compile(pure_fwd, mode="reduce-overhead", fullgraph=False)
        cfwd(x)  # 触发编译
        cfwd(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        compiled = OpCounter()
        with compiled:
            cfwd(x)
        compile_info = {
            "eager_fwd_dispatch": eager.n,
            "compiled_fwd_dispatch": compiled.n,
            "ratio": round(compiled.n / max(eager.n, 1), 3),
            "ok": True,
        }
    except Exception as e:
        compile_info = {"ok": False, "err": f"{type(e).__name__}: {str(e)[:300]}"}
    report["compile_forward"] = compile_info

    report["config"] = config_meta(probe="probe_dispatch.py", args=vars(args))
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "config"},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
