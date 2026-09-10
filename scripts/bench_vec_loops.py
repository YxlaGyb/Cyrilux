"""两个线性递推循环: 原 Python 循环 vs 前缀扫描 (fp32).

实测: 耗时 / 派发数 / 相对 fp64 真值的 max|Δ|.
m[t]     = (1-α)·m[t-1] + α·z4[t]        α = net._mem_a
zslow[t] = 0.99·zslow[t-1] + 0.01·z4[t]

闭式 (逐块, 块长 C): x[t] = d^u·s + c·Σ_v d^{u-v}·u_v,  d=1-c, u=t-t0
分块原因: α=0.5 时 d^-S 在 fp32 溢出 (S=256 → Inf), 块内指数 ≤ C 即可。
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

from model import DensePCNet  # noqa: E402
from pkg.cli.utils import run_file  # noqa: E402

CHUNK = 32


def _opname(func: Any) -> str:
    try:
        return str(func._schema.name)
    except Exception:
        return str(func)


class OpCounter(TorchDispatchMode):
    def __init__(self) -> None:
        self.n = 0

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        self.n += 1
        return func(*args, **(kwargs or {}))


# ---------- m_seq ----------
def mem_loop(z4, m0, a, buf, dtype=None):
    N, S = z4.shape[0], z4.shape[1]
    K, a4 = m0.shape
    if dtype is not None:
        z4, m0, a, buf = z4.to(dtype), m0.to(dtype), a.to(dtype), buf.to(dtype)
    buf[:, :1] = m0.unsqueeze(0).unsqueeze(0).expand(N, 1, K, a4)
    for t in range(1, S):
        zt = z4[:, t : t + 1].unsqueeze(2)
        buf[:, t : t + 1] = (1 - a)[None, None, :, None] * buf[:, t - 1 : t] + a[None, None, :, None] * zt
    return buf


def mem_scan(z4, m0, a, out_dtype, chunk: int = CHUNK):
    N, S = z4.shape[0], z4.shape[1]
    K, a4 = m0.shape
    c = a.float()
    d = 1.0 - c
    pad = (-S) % chunk
    Sp = S + pad
    z = z4.float().unsqueeze(2)                                   # [N,S,1,a4]
    if pad:
        z = torch.cat([z, torch.zeros(N, pad, 1, a4, device=z.device)], dim=1)
    zc = z.reshape(N, Sp // chunk, chunk, 1, a4)
    w = torch.cumprod(d.unsqueeze(0).expand(chunk, K), dim=0)     # d^u  [C,K]
    inv = c.unsqueeze(0) / w                                      # c/d^u
    acc = torch.cumsum(zc * inv[None, None, :, :, None], dim=2)   # [N,nb,C,1,a4]
    loc = w[None, None, :, :, None] * acc                         # 局部 (state=0)
    dec = w[None, None, :, :, None]                                # d^u
    out = torch.empty(N, Sp // chunk, chunk, K, a4, device=z.device, dtype=torch.float32)
    state = m0.float().unsqueeze(0).expand(N, K, a4)              # [N,K,a4]
    for b in range(Sp // chunk):                                  # 跨块串行: S/C 次
        out[:, b] = loc[:, b] + dec[:, b] * state.unsqueeze(1)
        state = out[:, b, chunk - 1]
    return out.reshape(N, Sp, K, a4)[:, :S].to(out_dtype)


# ---------- zslow ----------
def zslow_loop(z4, dtype=None):
    N, S = z4.shape[0], z4.shape[1]
    if dtype is not None:
        z4 = z4.to(dtype)
    out = torch.zeros_like(z4)
    out[:, 0] = z4[:, 0]
    for t in range(1, S):
        out[:, t] = 0.99 * out[:, t - 1] + 0.01 * z4[:, t]
    return out


def zslow_scan(z4, out_dtype, chunk: int = CHUNK):
    N, S = z4.shape[0], z4.shape[1]
    d = torch.tensor(0.99, dtype=torch.float32, device=z4.device)
    pad = (-S) % chunk
    Sp = S + pad
    z = z4.float()
    if pad:
        z = torch.cat([z, torch.zeros(N, pad, z.shape[2], device=z.device)], dim=1)
    zc = z.reshape(N, Sp // chunk, chunk, z.shape[2])
    w = torch.cumprod(d.expand(chunk), dim=0)                     # [C]
    inv = 0.01 / w
    acc = torch.cumsum(zc * inv[None, None, :, None], dim=2)
    loc = w[None, None, :, None] * acc
    dec = w[None, None, :, None]
    out = torch.empty(N, Sp // chunk, chunk, z.shape[2], device=z.device, dtype=torch.float32)
    state = torch.zeros(N, z.shape[2], device=z.device, dtype=torch.float32)
    for b in range(Sp // chunk):
        out[:, b] = loc[:, b] + dec[:, b] * state.unsqueeze(1)
        state = out[:, b, chunk - 1]
    return out.reshape(N, Sp, z.shape[2])[:, :S].to(out_dtype)


def _timeit(fn, reps: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0


def _mad(x, y) -> float:
    return round(float((x.float() - y.float()).abs().max()), 7)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/exp115_p3c_ev.safetensors")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or run_file("bench_vec_loops.json")

    torch.set_grad_enabled(False)
    net = DensePCNet.load(args.ckpt).to("cuda")
    x = torch.randint(0, 256, (1, 256), device="cuda", dtype=torch.long)
    net.forward_engine.forward(x)

    a = net._mem_a
    K, a4 = net._mem_m.shape
    rep: dict[str, object] = {
        "K": int(K), "a4": int(a4), "alpha": round(float(a.mean()), 4),
        "dtype_path": "z4/m_seq fp16", "chunk": CHUNK,
    }

    for S in (64, 256):
        z4 = net._z4[:, :S].contiguous()
        N = z4.shape[0]
        m0 = net._mem_m
        buf = torch.empty(N, S, K, a4, device="cuda", dtype=net._mem_m_seq.dtype)
        buf64 = torch.empty(N, S, K, a4, device="cuda", dtype=torch.float64)

        truth_m = mem_loop(z4, m0, a, buf64, dtype=torch.float64).float()
        truth_z = zslow_loop(z4, dtype=torch.float64).float()
        loop_m = mem_loop(z4, m0, a, buf.clone()).float()
        scan_m = mem_scan(z4, m0, a, buf.dtype).float()
        loop_z = zslow_loop(z4).float()
        scan_z = zslow_scan(z4, z4.dtype).float()

        rep[f"S{S}"] = {
            "mem_loop_ms": round(_timeit(lambda: mem_loop(z4, m0, a, buf), args.reps), 3),
            "mem_scan_ms": round(_timeit(lambda: mem_scan(z4, m0, a, buf.dtype), args.reps), 3),
            "zslow_loop_ms": round(_timeit(lambda: zslow_loop(z4), args.reps), 3),
            "zslow_scan_ms": round(_timeit(lambda: zslow_scan(z4, z4.dtype), args.reps), 3),
            "mem_err_loop_vs_fp64": _mad(loop_m, truth_m),
            "mem_err_scan_vs_fp64": _mad(scan_m, truth_m),
            "zslow_err_loop_vs_fp64": _mad(loop_z, truth_z),
            "zslow_err_scan_vs_fp64": _mad(scan_z, truth_z),
            "mem_scale": round(float(truth_m.abs().max()), 5),
            "zslow_scale": round(float(truth_z.abs().max()), 5),
        }

        c = {}
        for name, fn in (
            ("mem_loop", lambda: mem_loop(z4, m0, a, buf)),
            ("mem_scan", lambda: mem_scan(z4, m0, a, buf.dtype)),
            ("zslow_loop", lambda: zslow_loop(z4)),
            ("zslow_scan", lambda: zslow_scan(z4, z4.dtype)),
        ):
            cc = OpCounter()
            with cc:
                fn()
            c[f"{name}_dispatch"] = cc.n
        rep[f"S{S}"].update(c)

    rep["config"] = config_meta(probe="bench_vec_loops.py", args=vars(args))
    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "config"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
