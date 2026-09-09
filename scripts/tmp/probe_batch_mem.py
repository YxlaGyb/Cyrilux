"""batch 显存标定探针: N=1/2/4/8 的模型驻留与单步峰值显存 (真配置, CUDA).

口径: 每个 N 独立进程 → 峰值不含上一个 N 的分配器残留 (allocator 会复用释放块).
峰值 = torch 分配器口径; nvml_used = 进程总占用 (含 CUDA 上下文/库, 与 N 无关的常数项).
"""

import argparse
import json

import torch

from model import CyreneModel, DensePCNet

S_MAX = 256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--seq", type=int, default=S_MAX)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用")
    dev = torch.device("cuda")
    torch.manual_seed(0)

    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.synchronize()
    used0 = total - torch.cuda.mem_get_info()[0]

    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=S_MAX)
    net = DensePCNet(cfg).to(dev)
    torch.cuda.synchronize()
    init_alloc = torch.cuda.memory_allocated()
    used_init = total - torch.cuda.mem_get_info()[0]
    torch.cuda.reset_peak_memory_stats()

    x = torch.randint(0, 256, (a.n, a.seq), dtype=torch.long, device=dev)
    err = ""
    try:
        net.learn(x)
        torch.cuda.synchronize()
        ok = True
    except Exception as e:  # 形状/语义不支持 → 记录原文, 不吞
        ok = False
        err = f"{type(e).__name__}: {str(e)[:300]}"

    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    used_end = total - torch.cuda.mem_get_info()[0]

    rec = {
        "n": a.n,
        "seq": a.seq,
        "ok": ok,
        "err": err,
        "total_MiB": round(total / 2**20, 1),
        "ctx_MiB": round(used0 / 2**20, 1),          # 纯上下文 (模型未建)
        "init_alloc_MiB": round(init_alloc / 2**20, 1),   # 权重+迹+缓冲 驻留
        "init_used_MiB": round(used_init / 2**20, 1),     # 含上下文
        "step_delta_MiB": round((peak - init_alloc) / 2**20, 1),  # 单步激活峰值
        "peak_alloc_MiB": round(peak / 2**20, 1),
        "reserved_MiB": round(reserved / 2**20, 1),
        "end_used_MiB": round(used_end / 2**20, 1),
    }
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
