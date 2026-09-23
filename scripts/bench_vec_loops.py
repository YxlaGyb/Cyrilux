"""两个线性递推循环: 原 Python 循环 vs 分块前缀扫描 — 全 fp16 (FP16 死令, 无 fp32/fp64 对照路).

实测: 耗时 / 派发数 / 两 fp16 实现间 max|Δ| 与非有限计数 (互为参照).
m[t]     = (1-α)·m[t-1] + α·z4[t]        α = net._mem_a
zslow[t] = 0.99·zslow[t-1] + 0.01·z4[t]

分块前缀扫描: x[t] = d^u·s + c·Σ_v d^{u-v}·u_v, d=1-c, u=t-t0.
块内 d^{-u} (u<C) 须落在 fp16 值域: α=0.5 时 C=8 → d^-8=256 安全; α 越大 d 越小,
d^-C 越易溢出 — 扫描路出现 Inf/NaN 时基准如实报告非有限计数, 那是 fp16 的真读数.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from _probe_meta import config_meta
from torch.utils._python_dispatch import TorchDispatchMode

from crates.model import DensePCNet
from crates.cli.utils import run_file

CHUNK = 8  # 块长: d^{-C} 须落在 fp16 值域 (α=0.5 → 256; α=0.75 → 65536 临界)


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
def mem_loop(z4, m0, a, buf):
    """原 Python 循环 (fp16): buf[:, t] = (1-α)·buf[:, t-1] + α·z4[:, t]."""
    N, S = z4.shape[0], z4.shape[1]
    K, a4 = m0.shape
    buf[:, :1] = m0.unsqueeze(0).unsqueeze(0).expand(N, 1, K, a4)
    for t in range(1, S):
        zt = z4[:, t : t + 1].unsqueeze(2)
        buf[:, t : t + 1] = (1 - a)[None, None, :, None] * buf[:, t - 1 : t] + a[None, None, :, None] * zt
    return buf


def mem_scan(z4, m0, a, chunk: int = CHUNK):
    """分块前缀扫描 (fp16): 块内 d^{-u} 闭式, 跨块串行传状态."""
    N, S = z4.shape[0], z4.shape[1]
    K, a4 = m0.shape
    c = a
    d = 1.0 - c
    pad = (-S) % chunk
    Sp = S + pad
    z = z4.unsqueeze(2)  # [N,S,1,a4] fp16
    if pad:
        z = torch.cat([z, torch.zeros(N, pad, 1, a4, device=z.device, dtype=z.dtype)], dim=1)
    zc = z.reshape(N, Sp // chunk, chunk, 1, a4)
    w = torch.cumprod(d.unsqueeze(0).expand(chunk, K), dim=0)  # d^u [C,K] fp16
    inv = c.unsqueeze(0) / w  # c/d^u [C,K] fp16 — u<C, α 小则安全, 溢出如实呈现
    acc = torch.cumsum(zc * inv[None, None, :, :, None], dim=2)  # [N,nb,C,1,a4]
    loc = w[None, None, :, :, None] * acc  # [N,nb,C,K,a4] 局部 (state=0)
    out = torch.empty(N, Sp // chunk, chunk, K, a4, device=z.device, dtype=torch.float16)
    state = m0.unsqueeze(0).expand(N, K, a4)  # [N,K,a4] fp16
    nb = Sp // chunk
    for b in range(nb):  # 跨块串行: S/C 次
        # x[t0+v] = d^v·s + c·Σ_{j≤v} d^{v-j} z_j = w[v]·s + loc[:, b, v] (v = 块内位置)
        out[:, b] = loc[:, b] + w.unsqueeze(-1) * state.unsqueeze(1)
        state = out[:, b, chunk - 1]
    return out.reshape(N, Sp, K, a4)[:, :S]


# ---------- zslow ----------
def zslow_loop(z4):
    """原 Python 循环 (fp16)."""
    S = z4.shape[1]
    out = torch.zeros_like(z4)
    out[:, 0] = z4[:, 0]
    for t in range(1, S):
        out[:, t] = 0.99 * out[:, t - 1] + 0.01 * z4[:, t]
    return out


def zslow_scan(z4, chunk: int = CHUNK):
    """分块前缀扫描 (fp16): d=0.99, d^-C ≈ 1.086 (C=8), fp16 全程安全."""
    N, S = z4.shape[0], z4.shape[1]
    d = 0.99
    pad = (-S) % chunk
    Sp = S + pad
    z = z4
    if pad:
        z = torch.cat([z, torch.zeros(N, pad, z.shape[2], device=z.device, dtype=z.dtype)], dim=1)
    zc = z.reshape(N, Sp // chunk, chunk, z.shape[2])
    w = torch.cumprod(torch.full((chunk,), d, device=z.device, dtype=torch.float16), dim=0)  # [C]
    inv = 0.01 / w  # [C] fp16
    acc = torch.cumsum(zc * inv[None, None, :, None], dim=2)
    loc = w[None, None, :, None] * acc  # [N,nb,C,d_z]
    out = torch.empty(N, Sp // chunk, chunk, z.shape[2], device=z.device, dtype=torch.float16)
    state = torch.zeros(N, z.shape[2], device=z.device, dtype=torch.float16)
    nb = Sp // chunk
    for b in range(nb):
        out[:, b] = loc[:, b] + w[None, :, None] * state.unsqueeze(1)
        state = out[:, b, chunk - 1]
    return out.reshape(N, Sp, z.shape[2])[:, :S]


def _timeit(fn, reps: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0


def _mad(x, y):
    """max|Δ| 与非有限计数 (标量落盘: 单次 .cpu() 传输, 不用 .item())."""
    diff = (x - y).abs()
    bad = int(torch.isnan(diff).sum() + torch.isinf(diff).sum())
    finite = diff[~torch.isnan(diff) & ~torch.isinf(diff)]
    mx = float(finite.max().cpu()) if finite.numel() else float("inf")
    return round(mx, 6), bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/exp115_p3c_ev.safetensors")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=CHUNK)
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
        "K": int(K), "a4": int(a4), "alpha": round(float(a.mean().cpu()), 4),
        "dtype_path": "all fp16 (FP16 死令)", "chunk": args.chunk,
    }

    for S in (64, 256):
        z4 = net._z4[:, :S].contiguous()
        N = z4.shape[0]
        m0 = net._mem_m
        buf = torch.empty(N, S, K, a4, device="cuda", dtype=net._mem_m_seq.dtype)

        loop_m = mem_loop(z4, m0, a, buf.clone())
        scan_m = mem_scan(z4, m0, a, args.chunk)
        loop_z = zslow_loop(z4)
        scan_z = zslow_scan(z4, args.chunk)

        mad_m, bad_m = _mad(scan_m, loop_m)
        mad_z, bad_z = _mad(scan_z, loop_z)

        section = {
            "mem_loop_ms": round(_timeit(lambda: mem_loop(z4, m0, a, buf), args.reps), 3),
            "mem_scan_ms": round(_timeit(lambda: mem_scan(z4, m0, a, args.chunk), args.reps), 3),
            "zslow_loop_ms": round(_timeit(lambda: zslow_loop(z4), args.reps), 3),
            "zslow_scan_ms": round(_timeit(lambda: zslow_scan(z4, args.chunk), args.reps), 3),
            "mem_mad_scan_vs_loop": mad_m, "mem_nonfinite_scan": bad_m,
            "zslow_mad_scan_vs_loop": mad_z, "zslow_nonfinite_scan": bad_z,
            "mem_scale": round(float(loop_m.abs().max().cpu()), 5),
            "zslow_scale": round(float(loop_z.abs().max().cpu()), 5),
        }
        rep[f"S{S}"] = section

        c: dict = {}
        for name, fn in (
            ("mem_loop", lambda: mem_loop(z4, m0, a, buf)),
            ("mem_scan", lambda: mem_scan(z4, m0, a, args.chunk)),
            ("zslow_loop", lambda: zslow_loop(z4)),
            ("zslow_scan", lambda: zslow_scan(z4, args.chunk)),
        ):
            cc = OpCounter()
            with cc:
                fn()
            c[f"{name}_dispatch"] = cc.n
        section.update(c)

    rep["config"] = config_meta(probe="bench_vec_loops.py", args=vars(args))
    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "config"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
