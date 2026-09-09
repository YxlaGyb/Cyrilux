"""一次性调试: spawn 子进程的 CUDA 工作为何在父进程 NVML 上不可见."""

import contextlib
import multiprocessing as mp
import subprocess
import sys
import threading
import time

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOG = "out/tmp_spawn_debug.txt"


def child(stop, log_path):
    t0 = time.perf_counter()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"child start t={t0:.3f}\n")
    a = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
    b = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
    c = torch.empty(1024, 1024, dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"child cuda ready t={time.perf_counter() - t0:.3f}s\n")
    n = 0
    while not stop.is_set():
        t1 = time.perf_counter()
        for _ in range(32):
            torch.mm(a, b, out=c)
        torch.cuda.synchronize()
        n += 32
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"child burst {n} at t={t1 - t0:.2f}s sync={time.perf_counter() - t1:.4f}s\n")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"child exit n={n}\n")


def smi_sampler(stop, out):
    while not stop.wait(0.5):
        with contextlib.suppress(Exception):
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=3)
            out.append((round(time.perf_counter(), 2), r.stdout.strip()))


if __name__ == "__main__":
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("parent start\n")
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    p = ctx.Process(target=child, args=(stop, LOG), daemon=True)
    t0 = time.perf_counter()
    p.start()
    out = []
    th = threading.Thread(target=smi_sampler, args=(stop, out), daemon=True)
    th.start()
    time.sleep(10.0)
    stop.set()
    th.join(timeout=3)
    p.join(timeout=10)
    print(f"parent: child alive={p.is_alive()} exitcode={p.exitcode}")
    print("smi samples:", out)
    with open(LOG, encoding="utf-8") as f:
        print(f.read())
