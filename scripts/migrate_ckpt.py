"""检查点迁移: weights-only 化石 → 完整 state_ver 检查点 (P4 前置, 用户裁决主链 7).

3da511c 起 load() 带 state_ver 门, 全部存量存档 (weights-only 快照) 被拒收. 本脚本按
W_04/W_42/W_23/W_56/W1 形状反推 dims/mem_k0 建模 → 形状过滤载入存档张量 (缺省运行时
张量按 P2/P3 重臂先例取构造默认) → save() 写完整检查点 → load() 验证门 + 往返逐位
对比 + 前向动力学对比. 原化石只读不动.

产物: out/migrated/<name>.safetensors + migration_manifest.json
用法: uv run python scripts/migrate_ckpt.py
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import load_file

from dataset import ByteDataset
from model import CyreneModel, DensePCNet
from pkg.outver import ensure_run_dir

torch.set_grad_enabled(False)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NAMES = [
    "chat107_pool_fixed",
    "exp114_say",
    "exp115_say_preP1",
    "exp115_say",
    "exp115_p2_ev",
    "exp115_p3c_ev",
    "exp115_p3b_ev",
]
OUT_DIR = Path(ensure_run_dir(Path("out"))) / "migrated"  # 版本目录规范 out/v{N}-{时间戳} (pkg/outver)
DATA = "dataset/pretrain_t2t_mini.jsonl"
S_MAX = 256
PARITY_WINDOWS = 2


def resolve(name):
    cands = [OUT_DIR.parent / f"{name}.safetensors", Path("out") / f"{name}.safetensors"]
    cands += sorted(Path("out").glob(f"*/{name}.safetensors"))
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f"{name}: 无存档可寻")


def build_from_fossil(path):
    sd = load_file(str(path), device="cpu")
    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=S_MAX, lm_freeze_w1=True)
    d_l4 = sd["W_04"].shape[0]
    mem_k0 = (sd["W1"].shape[0] - 32) // d_l4 - 1
    cfg = replace(
        cfg,
        d_l4=d_l4,
        d_l2=sd["W_42"].shape[0],
        d_l3=sd["W_23"].shape[0],
        d_l5=sd["W_56"].shape[1],
        d_l6=sd["W_56"].shape[0],
        mem_k0=mem_k0,
        input_history=sd["W_04"].shape[1] != 256,
    )
    net = DensePCNet(cfg)
    ref = net.state_dict()
    fit = {k: v for k, v in sd.items() if k in ref and ref[k].shape == v.shape}
    missing = sorted(k for k in ref if k not in sd)  # 模型需要而文件缺 → 重臂 (构造默认)
    retired = sorted(k for k in sd if k not in ref)  # 文件有而已退役 → 留在化石里
    net.load_state_dict(fit, strict=False)
    return net, missing, retired


def dynamics_key(net):
    return torch.cat([net._z4.flatten(), net._bind_vec.flatten(), net._mem_out.flatten()])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = ByteDataset(DATA, max_length=S_MAX, max_samples=1270000, lazy=True)
    wins = [ds[i] for i in range(PARITY_WINDOWS)]
    manifest = {}
    for name in NAMES:
        src = resolve(name)
        net, missing, retired = build_from_fossil(src)
        dst = OUT_DIR / f"{name}.safetensors"
        net.save(str(dst))
        net2 = DensePCNet.load(str(dst))  # state_ver 门本身即验证
        # 往返逐位: save/load 不得动任何张量 (别名/存储指针错误的守卫)
        rt = {k: (net.state_dict()[k], net2.state_dict()[k]) for k in net2.state_dict()}
        rt_diff = max(float((a.float() - b.float()).abs().max()) for a, b in rt.values())
        # 前向动力学逐位: 同权重同序 → 必须逐位一致
        dmax = 0.0
        for w in wins:
            x = w.unsqueeze(0)
            net.forward(x)
            v1 = dynamics_key(net)
            net2.forward(x)
            v2 = dynamics_key(net2)
            dmax = max(dmax, float((v1.float() - v2.float()).abs().max()))
        ok = rt_diff == 0.0 and dmax == 0.0
        manifest[name] = {
            "src": str(src),
            "dst": str(dst),
            "file_keys": len(load_file(str(src), device="cpu")),
            "rearmed_keys": missing,
            "retired_keys": retired,
            "roundtrip_max_diff": rt_diff,
            "forward_max_diff": dmax,
            "ok": ok,
        }
        print(
            f"{name:22s} {src} → {dst}  文件键={manifest[name]['file_keys']} "
            f"重臂={len(missing)} 退役={len(retired)}  往返diff={rt_diff} 前向diff={dmax} {'✓' if ok else '✗ FAIL'}"
        )
        if not ok:
            sys.exit(1)
    (OUT_DIR / "migration_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"全部 {len(NAMES)} 个迁移完成 → {OUT_DIR}")


if __name__ == "__main__":
    main()
