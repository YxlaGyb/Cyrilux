"""迁移旧检查点至 CoreBrain 架构 (接口对齐, 不改主线).

CoreBrain 落地后 `HybridPCNet.load` (crates/model/model_cyrene.py) 对缺 `_core_brain.*`
键的检查点 fail-fast。本脚本对旧检查点注入 **CoreBrain 设计出生态** (Q=0.5I、
P=soft_norm(randn/√d_in)、buffers 全零), 使新架构可加载。

安全性依据 (E9e-3b + 仪器自检): 出生 CoreBrain 的窗级 Δ 经读出链后被压到 fp16
分辨率以内 — Δ≡0 与大 Δ 对生成 logits 的改动均 ≤1 ULP (见 out/*/e9e_generation_path_diagnosis.md)。
故本迁移对旧检查点行为零可观测影响, 由 `--assert` 的逐字节断言验收。

约定:
- 输入旧检查点; 输出 **新版本目录** `out/v{N}-.../migrated/<name>` (不覆盖原文件)。
- 出生键注入: 已含 `_core_brain.*` 键的检查点不再注入。
- config 快照修复: 先按 `LEGACY_CFG_RENAMED` 重命名已改名 cfg 字段 (lr_hebbian → lr_rtr),
  再删 `LEGACY_CFG_FALSE` 白名单内且值为 False 的已退役 cfg 字段 (cfg 字段化石, 死代码删除的
  连带迁移); 白名单外未知字段或非 False 值一律 raise (维持 C9 fail-fast)。
- 退役 state 键剔除: 按 `LEGACY_STATE_RETIRED` 移除已退役 buffer (如 _metab_rpe_ema),
  否则 `HybridPCNet.load` 的 C9 fail-fast 会拒绝。
- 不改 `crates/` 任何代码; 不训练; 只读原文件 + 写新文件。

用法:
    python scripts/migrate_ckpt.py out/v201-20260923-093527/merge.safetensors
    python scripts/migrate_ckpt.py <ckpt> --out-dir out/v240-20261009-191826/migrated
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from safetensors.torch import load_file, save_file  # noqa: E402

from crates.corebrain import CoreBrain, CoreBrainConfig  # noqa: E402
from crates.model.model_cyrene import (  # noqa: E402
    _CFG_FIELDS,
    HybridPCNet,
    read_checkpoint_metadata,
)
from crates.utils.outver import ensure_run_dir  # noqa: E402

torch.set_grad_enabled(False)

OUT = Path(ROOT) / "out"
GATE_PROMPT = "春"
GATE_N = 64
GATE_SEED = 117

# 已退役且历史恒为 False 的 cfg 字段白名单 (死代码删除的连带迁移); 值非 False → 该检查点
# 曾在此开关下训练, 迁移会改变行为, 拒绝并交人工裁决.
LEGACY_CFG_FALSE = ("lm_presyn_whiten",)

# 已改名 cfg 字段 (旧名 → 新名); 重命名后值直接沿用, 不参与下方白名单裁决.
LEGACY_CFG_RENAMED = {"lr_hebbian": "lr_rtr"}

# 已退役 state 键 (删代码的连带迁移); 从 state dict 剔除后方可过 C9 fail-fast.
LEGACY_STATE_RETIRED = ("_metab_rpe_ema", "_rtr_v", "_rtr_proj")


def repair_config_snap(src: str, meta: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """config 快照退役字段修复: 先改名, 再删白名单内且值为 False 的字段; 其余未知字段维持 C9 fail-fast."""
    raw = json.loads(meta["config"])
    for old, new in LEGACY_CFG_RENAMED.items():
        if old in raw:
            raw[new] = raw.pop(old)
    dropped = [k for k in raw if k not in _CFG_FIELDS]
    bad = [k for k in dropped if k not in LEGACY_CFG_FALSE or raw[k] is not False]
    if bad:
        raise ValueError(
            f"{src} config 快照含非白名单/非 False 的退役字段 {bad} — 拒绝迁移, 交人工裁决"
        )
    for k in dropped:
        raw.pop(k)
    meta = dict(meta)
    meta["config"] = json.dumps(raw)
    return meta, dropped


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


def load_original(path: str) -> HybridPCNet:
    """加载旧检查点 (内存内注入出生键 + config 修复, 不写盘) — 仅用于断言对照组."""
    import safetensors.torch as st

    import crates.model.model_cyrene as _mc

    real_load = st.load_file
    real_meta = _mc.read_checkpoint_metadata

    def patched_load(p, device="cpu"):
        sd = real_load(p, device=device)
        for k in LEGACY_STATE_RETIRED:
            sd.pop(k, None)
        sd, _ = inject_birth_keys(sd)
        return sd

    def patched_meta(p):
        meta, _ = repair_config_snap(p, real_meta(p))
        return meta

    st.load_file = patched_load
    _mc.read_checkpoint_metadata = patched_meta
    try:
        return HybridPCNet.load(path)
    finally:
        st.load_file = real_load
        _mc.read_checkpoint_metadata = real_meta


def gen_sha(net: HybridPCNet, dev: torch.device) -> str:
    torch.manual_seed(GATE_SEED)
    out = net.to(dev).generate(GATE_PROMPT, n_tokens=GATE_N, temperature=0.0, dev=dev)
    return hashlib.sha256(bytes(out)).hexdigest()


def migrate_one(src: str, out_dir: Path, assert_zero_change: bool) -> dict:
    meta = read_checkpoint_metadata(src)
    if "state_ver" not in meta:
        raise ValueError(f"{src} 不是完整检查点 (缺 state_ver), 拒绝迁移")
    if "config" not in meta:
        raise ValueError(f"{src} 缺 config 快照 (有 state_ver 就必须有), 拒绝迁移")
    meta, config_dropped = repair_config_snap(src, meta)
    sd = load_file(src, device="cpu")
    retired = [k for k in LEGACY_STATE_RETIRED if k in sd]
    for k in retired:
        sd.pop(k)
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
        "config_dropped": config_dropped,
        "retired": retired,
        "n_before": before,
        "n_after": len(sd),
    }
    if injected and assert_zero_change:
        dev = torch.device("cpu")
        sha_src = gen_sha(load_original(src), dev)
        sha_dst = gen_sha(HybridPCNet.load(str(out_path)), dev)
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
        tags = []
        if rec["injected"]:
            tags.append("注入出生键")
        if rec["config_dropped"]:
            tags.append(f"config 修复 (删 {rec['config_dropped']})")
        if rec["retired"]:
            tags.append(f"退役键剔除 (删 {rec['retired']})")
        tag = " + ".join(tags) if tags else "已是当前口径 (no-op)"
        print(f"  {os.path.basename(src)}: {tag}  {rec['n_before']}→{rec['n_after']} 张量 → {rec['dst']}")
        if "zero_change" in rec:
            state = "PASS" if rec["zero_change"] else "FAIL"
            ok = ok and rec["zero_change"]
            print(f"    零变更断言: {state}  sha {rec['sha_src'][:12]} vs {rec['sha_dst'][:12]}")
    print(f"[migrate_ckpt] {'全部通过' if ok else '存在失败'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
