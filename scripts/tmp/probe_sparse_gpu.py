"""稀疏管线 GPU 诊断: train_step 每步墙钟 + GPU 占用 (同 GTX 1650 Ti).

只读; 不修改任何稀疏/密集代码. 输出 out/probe_sparse_gpu.json.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

# 归档包 training/__init__ 是 P0 已知冻结断链 (dataset 已移出) — 绕过包 __init__ 直装模块
import importlib.util
import types

_pkg = types.ModuleType("model._archived_sparse.training")
_pkg.__path__ = []
sys.modules["model._archived_sparse.training"] = _pkg

_spec = importlib.util.spec_from_file_location(
    "model._archived_sparse.training.config",
    os.path.join(_ROOT, "model/_archived_sparse/training/config.py"),
)
_config_mod = importlib.util.module_from_spec(_spec)
sys.modules["model._archived_sparse.training.config"] = _config_mod
_spec.loader.exec_module(_config_mod)

_spec2 = importlib.util.spec_from_file_location(
    "model._archived_sparse.training.loop",
    os.path.join(_ROOT, "model/_archived_sparse/training/loop.py"),
)
_loop_mod = importlib.util.module_from_spec(_spec2)
sys.modules["model._archived_sparse.training.loop"] = _loop_mod
_spec2.loader.exec_module(_loop_mod)

TrainingConfig = _config_mod.TrainingConfig
TrainingLoop = _loop_mod.TrainingLoop


def sampler(secs, out):
    start = time.time()
    while time.time() - start < secs:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            out.append(float(r.stdout.strip().splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--out", default="out/probe_sparse_gpu.json")
    args = ap.parse_args()

    cfg = TrainingConfig(hidden_size=256, max_seq_len=args.seq_len, seed=0)
    loop = TrainingLoop(cfg)
    loop.runner = loop._build_model()
    loop.warmup()

    seq = torch.randint(0, 256, (1, args.seq_len), dtype=torch.long,
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    labels = seq.clone()
    label = torch.zeros(1, args.seq_len, dtype=torch.long, device=seq.device)

    samples = []
    thr = threading.Thread(target=sampler, args=(args.steps * 0.35, samples), daemon=True)
    thr.start()

    for _ in range(args.warmup):
        loop.train_step(seq, labels)
    walls = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        loop.train_step(seq, labels)
        t1 = time.perf_counter()
        walls.append((t1 - t0) * 1e3)
    thr.join(timeout=5)

    walls.sort()
    vals = sorted(samples)
    rep = {
        "n": len(walls),
        "step_wall_ms_median": round(walls[len(walls) // 2], 2),
        "step_wall_ms_p10": round(walls[len(walls) // 10], 2),
        "step_wall_ms_p90": round(walls[9 * len(walls) // 10], 2),
        "util_median": vals[len(vals) // 2] if vals else None,
        "note": "稀疏 encode+训练步墙钟 (逐样本); 同 GPU 与密集基线: 密集回声 1350ms/感知 139ms",
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=1))
    print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
