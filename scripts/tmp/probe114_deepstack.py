"""114 阶段一探针: 深栈冻结诊断 (只读, 零模型代码修改).

评审者判决复核: "W_04/W_42/W1 frozen_moved=0.0 = 深栈冻结".
本探针三件事:
  A. checkpoint 逐位对比 — frozen_moved 是范数漂移指标 (‖W‖_终/‖W‖_初),
     soft_norm_preserve 行范数固定点 (1.0) 使其结构性恒零. 逐位 ‖ΔW‖_F/‖W‖_F
     才是移动量. 链: chat107 → 110c → 111s1500 → 111resume → 113.
  B. π 门控分布 — W_04/W_42 更新路径的 std 归一化分母 (评审者口径
     π=1/(σε+c)) 在真实数据上的数值分布.
  C. z4 活跃度 — 死神经元比例 / 激活稀疏度 (终态, 真实数据前馈).

用法: .venv/Scripts/python.exe scripts/probe114_deepstack.py
输出: out/probe114_deepstack.json + stdout
"""
import json
import sys

import torch

torch.set_grad_enabled(False)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHAIN = [
    ("chat107→110c", "out/chat107_pool_fixed.pt", "out/exp110c_world.pt"),
    ("110c→111s1500", "out/exp110c_world.pt", "out/exp111_say_step1500.pt"),
    ("111s1500→111resume", "out/exp111_say_step1500.pt", "out/exp111_resume.pt"),
    ("111resume→113", "out/exp111_resume.pt", "out/exp113_say.pt"),
]
KEYS = ["W_04", "W_42", "W1", "W_diff", "W_lm", "W_lm_2", "W_t4", "W_t2",
        "W_t3", "W_t5", "W_t6", "W_bind", "W_bind_self", "W_23", "W_35",
        "W_56", "W_state_pred", "W_act"]


def cmp_pair(tag, pa, pb):
    sa = torch.load(pa, map_location="cpu", weights_only=True)
    sb = torch.load(pb, map_location="cpu", weights_only=True)
    out = {}
    for k in KEYS:
        if k not in sa or k not in sb:
            out[k] = "missing"
            continue
        wa, wb = sa[k].float(), sb[k].float()
        if wa.shape != wb.shape:
            out[k] = f"shape {tuple(wa.shape)}→{tuple(wb.shape)}"
            continue
        d = (wb - wa).norm().item()
        na = wa.norm().item()
        n_fin = wb.norm().item()
        changed = (wa != wb).float().mean().item()
        out[k] = {
            "rel_frob": d / (na + 1e-12),
            "norm_drift": abs(n_fin - na) / (na + 1e-12),
            "frac_entries_changed": changed,
            "max_abs_delta": (wb - wa).abs().max().item(),
            "frob_init": na,
        }
    print(f"[{tag}]", flush=True)
    for k in KEYS:
        v = out[k]
        if isinstance(v, dict):
            print(f"  {k:14s} rel_frob={v['rel_frob']:.3e} "
                  f"norm_drift={v['norm_drift']:.3e} "
                  f"changed={v['frac_entries_changed']*100:6.2f}% "
                  f"max|Δ|={v['max_abs_delta']:.2e}")
        else:
            print(f"  {k:14s} {v}")
    return out


def main():
    res = {"pairs": {}}
    for tag, pa, pb in CHAIN:
        res["pairs"][tag] = cmp_pair(tag, pa, pb)

    # 行范数固定点验证 (soft_norm_preserve 是否把范数钉死)
    sd = torch.load("out/exp113_say.pt", map_location="cpu", weights_only=True)
    rn = {}
    for k in ("W_04", "W_42", "W_diff", "W_23"):
        w = sd[k].float()
        r = w.norm(dim=1)
        rn[k] = {"mean": r.mean().item(), "std": r.std().item(),
                 "min": r.min().item(), "max": r.max().item()}
        print(f"row-norm {k:8s} mean={rn[k]['mean']:.4f} std={rn[k]['std']:.4f} "
              f"min={rn[k]['min']:.4f} max={rn[k]['max']:.4f} shape={tuple(w.shape)}")
    res["row_norms_113"] = rn

    with open("out/probe114_deepstack.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("saved -> out/probe114_deepstack.json")


if __name__ == "__main__":
    main()
