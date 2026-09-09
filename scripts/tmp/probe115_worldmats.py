"""115 阶段一探针 (CPU): NaN 快照验尸 + 世界模型矩阵链式补测.

A. exp113_say_nan3537.pt 全 state_dict 扫描: 哪些张量非有限 (参数+buffer),
   各自非有限条目数 / inf / NaN 计数 — 判决"第一个溢出的是哪个量".
   对照: exp113_say.pt (健康终态) 全有限.
B. checkpoint 链 entrywise rel_frob (复用 probe114_deepstack 口径):
   补 114/deepstack 都未追踪的 W_pred_54 / W_pred_43 / b_diff,
   并延长链到 113→114. 世界模型侧 = W_diff/b_diff/W_state_pred/W_pred_54/
   W_pred_43; W_42/W_t4/W_04 作对照参照.
C. NaN 快照 vs exp111_resume (崩溃Run的起点): 世界模型矩阵在崩溃前动了多少.

用法: .venv/Scripts/python.exe scripts/probe115_worldmats.py
输出: out/probe115_worldmats.json
"""
import json
import sys

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHAIN = [
    ("chat107→110c", "out/chat107_pool_fixed.pt", "out/exp110c_world.pt"),
    ("110c→111s1500", "out/exp110c_world.pt", "out/exp111_say_step1500.pt"),
    ("111s1500→111resume", "out/exp111_say_step1500.pt", "out/exp111_resume.pt"),
    ("111resume→113", "out/exp111_resume.pt", "out/exp113_say.pt"),
    ("113→114", "out/exp113_say.pt", "out/exp114_say.pt"),
]
WM_KEYS = ["W_diff", "b_diff", "W_state_pred", "W_pred_54", "W_pred_43"]
REF_KEYS = ["W_42", "W_t4", "W_04", "W_lm"]


def scan_finite(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    bad = {}
    for k, v in sd.items():
        if not torch.is_floating_point(v):
            continue
        finite = torch.isfinite(v)
        if finite.all():
            continue
        nf = (~finite).sum().item()
        inf_cnt = torch.isinf(v).sum().item()
        nan_cnt = torch.isnan(v).sum().item()
        vabs = v[finite].abs() if finite.any() else torch.zeros(1)
        bad[k] = {
            "numel": v.numel(),
            "nonfinite": nf,
            "nonfinite_frac": nf / v.numel(),
            "inf": inf_cnt,
            "nan": nan_cnt,
            "max_abs_finite": vabs.max().item() if vabs.numel() else None,
            "shape": tuple(v.shape),
        }
    return bad


def cmp_pair(tag, pa, pb, keys):
    sa = torch.load(pa, map_location="cpu", weights_only=True)
    sb = torch.load(pb, map_location="cpu", weights_only=True)
    out = {}
    for k in keys:
        if k not in sa or k not in sb:
            out[k] = "missing"
            continue
        wa, wb = sa[k].float(), sb[k].float()
        if wa.shape != wb.shape:
            out[k] = f"shape {tuple(wa.shape)}→{tuple(wb.shape)}"
            continue
        d = (wb - wa).norm().item()
        na = wa.norm().item()
        changed = (wa != wb).float().mean().item()
        out[k] = {
            "rel_frob": d / (na + 1e-12),
            "frac_entries_changed": changed,
            "max_abs_delta": (wb - wa).abs().max().item(),
            "frob_init": na,
        }
    return out


def main():
    res = {}

    # A. NaN 快照扫描
    res["nan_snapshot_scan"] = scan_finite("out/exp113_say_nan3537.pt")
    res["healthy_113_scan"] = scan_finite("out/exp113_say.pt")
    print(f"[A] NaN 快照非有限张量: {list(res['nan_snapshot_scan'].keys())}")
    for k, v in res["nan_snapshot_scan"].items():
        print(f"    {k}: nonfinite={v['nonfinite']}/{v['numel']} "
              f"(inf={v['inf']}, nan={v['nan']}) "
              f"max_abs_finite={v['max_abs_finite']}")
    print(f"[A] 健康终态 (113) 非有限张量: {list(res['healthy_113_scan'].keys()) or '无'}")

    # B. 链式 rel_frob (世界模型侧 + 参照)
    allk = WM_KEYS + REF_KEYS
    res["chain_rel_frob"] = {}
    for tag, pa, pb in CHAIN:
        r = cmp_pair(tag, pa, pb, allk)
        res["chain_rel_frob"][tag] = r
        print(f"[B] {tag}", flush=True)
        for k in allk:
            v = r[k]
            if isinstance(v, dict):
                print(f"    {k:14s} rel_frob={v['rel_frob']:.3e} "
                      f"changed={v['frac_entries_changed']*100:6.2f}% "
                      f"max|Δ|={v['max_abs_delta']:.2e}")
            else:
                print(f"    {k:14s} {v}")

    # C. NaN 快照 vs 111resume (崩溃 Run 起点)
    r = cmp_pair("nan3537_vs_111resume", "out/exp111_resume.pt",
                 "out/exp113_say_nan3537.pt", allk + ["W1", "W_act", "W_bind"])
    res["nan_vs_111resume"] = r
    print("[C] NaN快照 vs 111resume (崩溃Run内移动量)", flush=True)
    for k, v in r.items():
        if isinstance(v, dict):
            print(f"    {k:14s} rel_frob={v['rel_frob']:.3e} "
                  f"changed={v['frac_entries_changed']*100:6.2f}%")
        else:
            print(f"    {k:14s} {v}")

    with open("out/probe115_worldmats.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("saved -> out/probe115_worldmats.json")


if __name__ == "__main__":
    main()
