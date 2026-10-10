"""迁移旧检查点至 CoreBrain 架构 (接口对齐, 不改主线).

CoreBrain 落地后 `DensePCNet.load` (crates/model/model_cyrene.py) 对缺 `_core_brain.*`
键的检查点 fail-fast。本脚本对旧检查点注入 **CoreBrain 设计出生态** (Q=0.5I、
P=soft_norm(randn/√d_in)、buffers 全零), 使新架构可加载。

安全性依据 (E9e-3b + 仪器自检): 出生 CoreBrain 的窗级 Δ 经读出链后被压到 fp16
分辨率以内 — Δ≡0 与大 Δ 对生成 logits 的改动均 ≤1 ULP (见 out/*/e9e_generation_path_diagnosis.md)。
故本迁移对旧检查点行为零可观测影响, 由 `--assert` 的逐字节断言验收。

约定:
- 输入旧检查点; 输出 **新版本目录** `out/v{N}-.../migrated/<name>` (不覆盖原文件)。
- 幂等: 已含 `_core_brain.*` 键的检查点 no-op, 输出与输入逐字节一致。
- 不改 `crates/` 任何代码; 不训练; 只读原文件 + 写新文件。

用法:
    python scripts/migrate_ckpt.py out/v201-20260923-093527/merge.safetensors
    python scripts/migrate_ckpt.py <ckpt> --out-dir out/v240-20261009-191826/migrated
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from safetensors.torch import load_file, save_file  # noqa: E402

from crates.corebrain import CoreBrain, CoreBrainConfig  # noqa: E402
from crates.model.model_cyrene import DensePCNet, read_checkpoint_metadata  # noqa: E402
from crates.utils.outver import ensure_run_dir  # noqa: E402

torch.set_grad_enabled(False)

OUT = Path(ROOT) / "out"
GATE_PROMPT = "春"
GATE_N = 64
GATE_SEED = 117


def birth_corebrain_state(d_in: int) -> dict[str, torch.Tensor]:
    cb = CoreBrain(CoreBrainConfig(d_in=d_in))
    return {k: v.detach().to(torch.float16) for k, v in cb.state_dict().items()}


def inject_birth_keys(sd: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], bool]:
    """对缺 _core_brain.* 的 state dict 注入出生键; 返回 (sd, 是否注入)."""
    if any(k.startswith("_core_brain.") for k in sd):
        return sd, False
    d_in = int(sd["W_04"].shape[0])
    for k, v in birth_corebrain_state(d_in).items():
        sd[f"_core_brain.{k}"] = v.clone()
    return sd, True


def load_original(path: str) -> DensePCNet:
    """加载旧检查点 (内存内注入出生键, 不写盘) — 仅用于断言对照组."""
    import safetensors.torch as st

    real_load = st.load_file

    def patched(p, device="cpu"):
        sd, _ = inject_birth_keys(real_load(p, device=device))
        return sd

    st.load_file = patched
    try:
        return DensePCNet.load(path)
    finally:
        st.load_file = real_load


def gen_sha(net: DensePCNet, dev: torch.device) -> str:
    torch.manual_seed(GATE_SEED)
    out = net.to(dev).generate(GATE_PROMPT, n_tokens=GATE_N, temperature=0.0, dev=dev)
    return hashlib.sha256(bytes(out)).hexdigest()


def migrate_one(src: str, out_dir: Path, assert_zero_change: bool) -> dict:
    meta = read_checkpoint_metadata(src)
    if "state_ver" not in meta:
        raise ValueError(f"{src} 不是完整检查点 (缺 state_ver), 拒绝迁移")
    sd = load_file(src, device="cpu")
    before = len(sd)
    sd, injected = inject_birth_keys(sd)
    out_path = out_dir / os.path.basename(src)
    # metadata 键序规范化: read_checkpoint_metadata 走 safetensors Rust HashMap, 迭代序不确定,
    # 不排序会导致同一输入两次迁移产出不同文件字节 (内容相同但哈希漂移), 破坏可复现性.
    canon_meta = {k: meta[k] for k in sorted(meta)}
    save_file({k: v.contiguous() for k, v in sd.items()}, str(out_path), metadata=canon_meta)
    rec = {
        "src": src,
        "dst": str(out_path),
        "injected": injected,
        "n_before": before,
        "n_after": len(sd),
    }
    if injected and assert_zero_change:
        dev = torch.device("cpu")
        sha_src = gen_sha(load_original(src), dev)
        sha_dst = gen_sha(DensePCNet.load(str(out_path)), dev)
        rec["sha_src"] = sha_src
        rec["sha_dst"] = sha_dst
        rec["zero_change"] = sha_src == sha_dst
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--no-assert", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    out_dir = Path(args.out_dir) if args.out_dir else Path(ensure_run_dir(OUT)) / "migrated"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[migrate_ckpt] 输出目录: {out_dir}", flush=True)

    ok = True
    for src in args.ckpts:
        rec = migrate_one(src, out_dir, assert_zero_change=not args.no_assert)
        tag = "注入出生键" if rec["injected"] else "已迁移 (no-op)"
        print(f"  {os.path.basename(src)}: {tag}  {rec['n_before']}→{rec['n_after']} 张量 → {rec['dst']}")
        if "zero_change" in rec:
            state = "PASS" if rec["zero_change"] else "FAIL"
            ok = ok and rec["zero_change"]
            print(f"    零变更断言: {state}  sha {rec['sha_src'][:12]} vs {rec['sha_dst'][:12]}")
    print(f"[migrate_ckpt] {'全部通过' if ok else '存在失败'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
