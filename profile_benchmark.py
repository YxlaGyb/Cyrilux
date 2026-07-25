"""CyreneModel 性能基准测试 — 字节级指标。

指标是字节级, No token:
  - steps/s  (step 吞吐)
  - bytes/s  (字节吞吐 = steps/s * seq_len)
  - BPB      (bits-per-byte = CE_loss / ln(2))
  - free_energy 收敛曲线
  - 延迟分布 (ms/step)
  - 不同神经元规模下的显存/内存占用

用法:
    uv run python profile_benchmark.py
    uv run python profile_benchmark.py --layers 2 --neurons 2048 --steps 200
"""

from __future__ import annotations

import math
import time
import tracemalloc
from argparse import ArgumentParser

import torch

from model.model_cyrene import CyreneConfig, CyreneModel, create_cyrene


def _make_seq(n_bytes: int = 512) -> torch.Tensor:
    """生成随机字节序列 [1, 2, S] fp16."""
    raw = torch.randint(0, 256, (n_bytes,), dtype=torch.uint8)
    vals = raw.to(torch.half).div_(128.0).sub_(1.0).unsqueeze(0).unsqueeze(0)
    mask = torch.ones_like(vals)
    return torch.cat([vals, mask], dim=1)


def benchmark_throughput(model: CyreneModel, seq: torch.Tensor, n_steps: int) -> dict:
    """测步吞吐 (steps/s) 和 字节吞吐 (bytes/s)."""
    seq_len = seq.shape[-1]
    # warmup
    for _ in range(10):
        model.step(seq)
    # benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(n_steps):
        model.step(seq)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0
    steps_s = n_steps / elapsed
    bytes_s = steps_s * seq_len
    return {"steps_s": steps_s, "bytes_s": bytes_s, "elapsed_s": elapsed}


def benchmark_latency(model: CyreneModel, seq: torch.Tensor, n_steps: int = 200) -> dict:
    """测每步延迟分布 (ms)."""
    times = []
    for _ in range(n_steps):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter_ns()
        model.step(seq)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append((time.perf_counter_ns() - t0) / 1e6)
    times = sorted(times)
    return {
        "mean_ms": sum(times) / len(times),
        "median_ms": times[len(times) // 2],
        "p90_ms": times[int(len(times) * 0.9)],
        "p99_ms": times[int(len(times) * 0.99)],
        "min_ms": times[0],
        "max_ms": times[-1],
    }


def benchmark_memory(
    hidden_sizes: list[int] = None, seq_len: int = 128
) -> list[dict]:
    """测不同规模下的 CPU/GPU 内存."""
    hidden_sizes = hidden_sizes or [64, 128, 256]
    results = []
    for hs in hidden_sizes:
        tracemalloc.start()
        cfg = CyreneConfig(hidden_size=hs, hidden_neurons=hs * 2)
        model = create_cyrene(cfg)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem = {"hidden_size": hs, "cpu_mb": peak / 1e6}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            seq = _make_seq(seq_len)
            for _ in range(10):
                model.step(seq)
            mem["gpu_mb"] = torch.cuda.max_memory_allocated() / 1e6
        results.append(mem)
    return results


def main():
    parser = ArgumentParser(description="CyreneModel 字节级性能基准")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--neurons", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--all", action="store_true", help="运行全部测试")
    args = parser.parse_args()

    print("=" * 60)
    print(f"CyreneModel 基准: h_size={args.hidden_size}, layers={args.layers}, "
          f"neurons={args.neurons}, seq_len={args.seq_len}")
    print("=" * 60)

    # ── 创建模型 ──
    cfg = CyreneConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        hidden_neurons=args.neurons,
        warmup_steps=10,
    )
    model = create_cyrene(cfg)
    model.warmup(10)

    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name()}  "
              f"Mem: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB")

    seq = _make_seq(args.seq_len)
    n_steps = args.steps

    # 1. 吞吐
    print("\n--- 1. 吞吐 ---")
    tput = benchmark_throughput(model, seq, n_steps)
    print(f"  {n_steps} steps in {tput['elapsed_s']:.3f}s")
    print(f"  Throughput:  {tput['steps_s']:.1f} steps/s")
    print(f"               {tput['bytes_s']:.0f} bytes/s")
    print(f"               {tput['bytes_s'] * 8:.0f} bits/s")

    # 2. BPB (用随机序列测 CE loss)
    print("\n--- 2. bits-per-byte (随机序列) ---")
    total_ce = 0.0
    n_pos = 0
    for pos in range(1, args.seq_len):
        ctx = seq[:, :, :pos].contiguous()
        if ctx.shape[-1] < 13:
            continue
        stats = model.step(ctx)
        if not stats.get("warmup", True):
            total_ce += stats.get("lm_loss", 0.0)
            n_pos += 1
    avg_ce = total_ce / max(n_pos, 1)
    bpb = avg_ce / math.log(2)
    print(f"  Mean CE:  {avg_ce:.4f}")
    print(f"  BPB:      {bpb:.4f}")

    # 3. 延迟分布
    print("\n--- 3. 延迟分布 ---")
    lat = benchmark_latency(model, seq, 100)
    print(f"  mean={lat['mean_ms']:.3f}ms  median={lat['median_ms']:.3f}ms")
    print(f"  p90={lat['p90_ms']:.3f}ms  p99={lat['p99_ms']:.3f}ms")
    print(f"  min={lat['min_ms']:.3f}ms  max={lat['max_ms']:.3f}ms")

    # 4. 内存 (仅 --all)
    if args.all:
        print("\n--- 4. 内存占用 (不同规模) ---")
        for m in benchmark_memory([64, 128, 256]):
            extra = f"  GPU={m.get('gpu_mb', 0):.1f}MB" if "gpu_mb" in m else ""
            print(f"  h_size={m['hidden_size']}:  CPU={m['cpu_mb']:.1f}MB{extra}")

    print("\nDone.")


if __name__ == "__main__":
    main()
