"""115 阶段二探针 (GPU): 深栈 dW 符号相关判决 — 真学习 vs 极限环 vs 噪声搅拌.

A. 链式 cos(W_T, W_0) 扫描 (CPU): W_t 家族/W_42/W_pred_* rel_frob 全挤在
   1.39-1.42 ≈ √2 的疑云 — √2 正是"方向完全去相关但范数保持"的解析签名
   (rel_frob² = 2-2cos, cos=0 → √2). 直接测 cos 与范数比, 判决趋同机制.
B. 符号相关探针 (GPU, 本轮最重要数字): 从 114 终态续跑 exp114 交替循环
   (感知/回声 + 世界物理 R) 110 步, 每步记录 W_42/W_t4/W_t2 落盘 dW;
   monkey-patch soft_norm_preserve/_decorr_W/_spectral_radius_guard 按
   data_ptr 分解 decorr/soft/guard 贡献 → hebb = disk - 其余 (缩放前口径).
   相邻更新事件 dW 逐条目符号一致率 + Pearson (活跃阈值 p75, 仅双步均超阈
   的条目), lag 1..10; 正相关→真学习, 负相关→极限环, 零→噪声.
C. 双口径遥测: 每步 disk/decorr/soft/hebb 范数 (对照 probe114_dial 的
   W_42 单步 2.93 = 落盘后口径), 位移比 ‖Σd‖/Σ‖d‖ (定向运动占比).

只读诊断: 内存副本, 不回写 checkpoint; patch 只在本进程, 不耗 RNG,
动力学与无 patch 运行逐位一致 (clone 不动 RNG).
用法: .venv/Scripts/python.exe scripts/probe115_signcorr.py
输出: out/probe115_signcorr.json
"""
import json
import math
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from collections import deque

import torch

torch.set_grad_enabled(False)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, "scripts")
import exp114_say as E
from model import CyreneModel, DensePCNet
from model.dense.learning import feedforward as ff_mod
from model.dense.learning import temporal as t_mod
from dataset import DualChannelDataset
from world_lang import WorldLangPhysics

S_MAX = 256
SEED_N = 16
DEV = torch.device("cuda")
N_STEPS = 110
RING = 34
DATA = "dataset/pretrain_t2t_mini.jsonl"

CHAIN = [
    ("chat107→110c", "out/chat107_pool_fixed.pt", "out/exp110c_world.pt"),
    ("110c→111s1500", "out/exp110c_world.pt", "out/exp111_say_step1500.pt"),
    ("111s1500→111resume", "out/exp111_say_step1500.pt", "out/exp111_resume.pt"),
    ("111resume→113", "out/exp111_resume.pt", "out/exp113_say.pt"),
    ("113→114", "out/exp113_say.pt", "out/exp114_say.pt"),
    ("chat107→114(全链)", "out/chat107_pool_fixed.pt", "out/exp114_say.pt"),
]
COS_KEYS = ["W_t2", "W_t3", "W_t4", "W_t5", "W_t6", "W_42", "W_04",
            "W_pred_54", "W_pred_43", "W_state_pred", "W_diff", "W_lm", "W_bind"]


def chain_cos():
    out = {}
    for tag, pa, pb in CHAIN:
        sa = torch.load(pa, map_location="cpu", weights_only=True)
        sb = torch.load(pb, map_location="cpu", weights_only=True)
        seg = {}
        for k in COS_KEYS:
            if k not in sa or k not in sb:
                seg[k] = "missing"
                continue
            wa, wb = sa[k].float(), sb[k].float()
            ip = (wa * wb).sum()
            na, nb = wa.norm(), wb.norm()
            seg[k] = {"cos": (ip / (na * nb + 1e-12)).item(),
                      "norm_ratio": (nb / (na + 1e-12)).item(),
                      "rel_frob_check": ((wa - wb).norm() / (na + 1e-12)).item()}
        out[tag] = seg
        print(f"[cos] {tag}: " + " ".join(
            f"{k}={v['cos']:+.3f}/nr={v['norm_ratio']:.3f}"
            if isinstance(v, dict) else f"{k}={v}" for k, v in seg.items()),
            flush=True)
        del sa, sb
    return out


def pair_stats(d_a, d_b, qs=(0.5, 0.75, 0.9)):
    """相邻 dW 对统计: 各分位活跃阈下的符号一致率 + Pearson."""
    aa, ab = d_a.abs().flatten(), d_b.abs().flatten()
    both = torch.cat([aa, ab])
    res = {}
    for q in qs:
        thr = both.quantile(q).clamp(min=1e-12)
        mask = (aa > thr) & (ab > thr)
        n = int(mask.sum())
        if n < 10:
            res[f"p{int(q*100)}"] = None
            continue
        xa, xb = d_a.flatten()[mask].float(), d_b.flatten()[mask].float()
        agree = (torch.sign(xa) == torch.sign(xb)).float().mean().item()
        res[f"p{int(q*100)}"] = {"n_active": n, "thr": thr.item(),
                                 "sign_agree": agree}
    fa, fb = d_a.flatten().float(), d_b.flatten().float()
    ca, cb = fa - fa.mean(), fb - fb.mean()
    den = ca.norm() * cb.norm()
    res["pearson_all"] = ((ca * cb).sum() / (den + 1e-12)).item() if den > 0 else None
    return res


def main():
    t0 = time.time()
    cos_out = chain_cos()

    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=S_MAX,
                        lm_freeze_w1=True)
    net = DensePCNet.load("out/exp114_say.pt", cfg).to(DEV)
    world = WorldLangPhysics(DATA)
    with open("out/exp114_world_state.json", encoding="utf-8") as f:
        st = json.load(f)
    step0, gt0 = world.load_state(st)
    net._gen_temp = torch.tensor(gt0, dtype=torch.float16, device=DEV)
    E._inject_world(net, world, DEV)
    net._echo_entropy = False
    print(f"[B] init 114 终态 sidecar step={step0} τ={gt0:.3f} E={world.E:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    torch.manual_seed(0)
    random.seed(0)
    ds = DualChannelDataset(DATA, max_length=S_MAX, max_samples=1270000, lazy=True)
    idxs = list(range(len(ds)))
    random.shuffle(idxs)

    targets = {"W_42": net.W_42, "W_t4": net.W_t4, "W_t2": net.W_t2}
    REG = {}
    for n, p in targets.items():
        REG[p.data.data_ptr()] = n
    acc = {}

    # 累加器: (name, kind) -> float32 GPU 张量
    def accum(name_kind, delta):
        if name_kind in acc:
            acc[name_kind] = acc[name_kind] + delta
        else:
            acc[name_kind] = delta

    orig_soft_ff, orig_decorr_ff = ff_mod.soft_norm_preserve, ff_mod._decorr_W
    orig_soft_t, orig_decorr_t = t_mod.soft_norm_preserve, t_mod._decorr_W
    orig_guard_t = t_mod._spectral_radius_guard

    def make_soft(orig, kind):
        def f(W):
            ptr = W.data_ptr()
            if ptr in REG:
                pre = W.detach().clone()
                orig(W)
                accum((REG[ptr], kind), W.detach().float() - pre.float())
            else:
                orig(W)
        return f

    def make_decorr(orig, kind):
        def f(W, *a, **kw):
            ptr = W.data_ptr()
            if ptr in REG:
                pre = W.detach().clone()
                r = orig(W, *a, **kw)
                accum((REG[ptr], kind), W.detach().float() - pre.float())
                return r
            return orig(W, *a, **kw)
        return f

    def guard_wrap(W, *a, **kw):
        ptr = W.data_ptr()
        if ptr in REG:
            pre = W.detach().clone()
            r = orig_guard_t(W, *a, **kw)
            accum((REG[ptr], "guard"), W.detach().float() - pre.float())
            return r
        return orig_guard_t(W, *a, **kw)

    ff_mod.soft_norm_preserve = make_soft(orig_soft_ff, "soft")
    ff_mod._decorr_W = make_decorr(orig_decorr_ff, "decorr")
    t_mod.soft_norm_preserve = make_soft(orig_soft_t, "soft")
    t_mod._decorr_W = make_decorr(orig_decorr_t, "decorr")
    t_mod._spectral_radius_guard = guard_wrap

    rings = {n: {k: deque(maxlen=RING) for k in ("disk", "hebb")}
             for n in targets}
    norms_log = {n: {k: [] for k in ("disk", "hebb", "decorr", "soft", "guard",
                                     "cos_dW_W", "frac_changed")} for n in targets}
    w_start = {n: p.data.float().clone() for n, p in targets.items()}
    w1_start = net.W1.data.float().clone()
    tau_hist, r_hist, e_hist, q_hist = [], [], [], []
    last_text_tail = None
    nan_at = None
    step = step0

    for i in range(1, N_STEPS + 1):
        step = step0 + i
        acc.clear()
        pre = {n: p.data.detach().clone() for n, p in targets.items()}
        if step % 2 == 1:
            b, _ = ds[idxs[(step // 2) % len(idxs)]]
            x = b.unsqueeze(0).to(DEV)
            net.learn(x)
            last_text_tail = x[0, -SEED_N:]
        else:
            net._echo_seed = (last_text_tail.unsqueeze(0)
                              if last_text_tail is not None
                              else torch.zeros(1, 1, dtype=torch.long, device=DEV))
            net.learn(None, free_run=False)
            trace_norm = float(net.W_act_elig.norm().item())
            gen_bytes = bytes(int(v) for v in net._gen_bytes[0].tolist())
            sc = world.score(gen_bytes)
            world.record(gen_bytes)
            r_world = world.step_E(sc["q"], trace_norm)
            net._world_R = torch.tensor(r_world, dtype=torch.float16, device=DEV)
            net._world_E = torch.tensor(world.E, dtype=torch.float16, device=DEV)
            eps_now = float(getattr(net, "_lm_eps", torch.tensor(0.0)).item())
            world.certify(sc["q"], eps_now)
            tau_hist.append(float(net._gen_temp.item()))
            r_hist.append(r_world)
            e_hist.append(world.E)
            q_hist.append(sc["q"])

        bad = [n for n, p in net.named_parameters() if not torch.isfinite(p).all()]
        if bad:
            nan_at = (step, bad[:3])
            print(f"[B] step {step}: NaN={bad[:3]} -> 中止", flush=True)
            break

        for n, p in targets.items():
            disk = p.data.float() - pre[n].float()
            hebb = disk.clone()
            for kind in ("decorr", "soft", "guard"):
                if (n, kind) in acc:
                    hebb = hebb - acc[(n, kind)]
            resid = (disk - hebb).norm().item()
            if resid > 1e-4:
                print(f"[B] 警告: {n} 分解残差 {resid:.2e} (未捕获写入路径?)",
                      flush=True)
            rings[n]["disk"].append(disk.cpu())
            rings[n]["hebb"].append(hebb.cpu())
            wref = p.data.float()
            for k in ("disk", "hebb", "decorr", "soft", "guard"):
                v = disk if k == "disk" else (hebb if k == "hebb" else
                                              acc.get((n, k),
                                                      torch.zeros(1, device=DEV)))
                norms_log[n][k].append(float(v.norm().item()))
            cd = torch.nn.functional.cosine_similarity(
                disk.flatten(), wref.flatten(), dim=0).item()
            norms_log[n]["cos_dW_W"].append(cd)
            norms_log[n]["frac_changed"].append(
                (p.data != pre[n]).float().mean().item())
        w1_delta = (net.W1.data.float() - w1_start).norm().item()
        if step % 10 == 0:
            msg = (f"[B] step {step}: τ={tau_hist[-1] if tau_hist else float('nan'):.2f} "
                   f"R={r_hist[-1] if r_hist else float('nan'):+.3f} "
                   f"E={e_hist[-1] if e_hist else float('nan'):.3f} "
                   f"q={q_hist[-1] if q_hist else float('nan'):.3f} | "
                   + " ".join(f"{n}:disk={norms_log[n]['disk'][-1]:.3f}"
                              f"/hebb={norms_log[n]['hebb'][-1]:.3f}"
                              for n in targets)
                   + f" | W1Δ累计={w1_delta:.2e}")
            print(msg, flush=True)

    ff_mod.soft_norm_preserve = orig_soft_ff
    ff_mod._decorr_W = orig_decorr_ff
    t_mod.soft_norm_preserve = orig_soft_t
    t_mod._decorr_W = orig_decorr_t
    t_mod._spectral_radius_guard = orig_guard_t

    # ── 分析: 更新事件序列 = disk 范数非零的步 ──
    analysis = {}
    for n in targets:
        ev_disk, ev_hebb = [], []
        for j, d in enumerate(rings[n]["disk"]):
            if d.norm().item() > 1e-9:
                ev_disk.append(d)
                ev_hebb.append(rings[n]["hebb"][j])
        lags = {}
        for lag in range(1, 11):
            ds_, hs_ = [], []
            for a in range(len(ev_disk) - lag):
                ds_.append(pair_stats(ev_disk[a], ev_disk[a + lag]))
                hs_.append(pair_stats(ev_hebb[a], ev_hebb[a + lag]))
            lags[lag] = {}
            for caliber, lst in (("disk", ds_), ("hebb", hs_)):
                agg = {}
                for qk in ("p50", "p75", "p90"):
                    vals = [p[qk]["sign_agree"] for p in lst
                            if p.get(qk) is not None]
                    if vals:
                        agg[qk] = {"mean_sign_agree": sum(vals) / len(vals),
                                   "n_pairs": len(vals)}
                peas = [p["pearson_all"] for p in lst
                        if p["pearson_all"] is not None]
                if peas:
                    agg["pearson_all"] = {"mean": sum(peas) / len(peas),
                                          "frac_positive":
                                              sum(1 for v in peas if v > 0) / len(peas),
                                          "min": min(peas), "max": max(peas)}
                lags[lag][caliber] = agg
        w_fin = targets[n].data.float()
        disp = (w_fin - w_start[n]).norm().item()
        path = sum(norms_log[n]["disk"])
        path_ev = sum(d.norm().item() for d in ev_disk)
        analysis[n] = {
            "n_events": len(ev_disk),
            "n_steps_total": len(norms_log[n]["disk"]),
            "per_step_norm_mean": {k: (sum(v) / len(v) if v else None)
                                   for k, v in norms_log[n].items()
                                   if k not in ("cos_dW_W", "frac_changed")},
            "cos_dW_W_mean": (sum(norms_log[n]["cos_dW_W"])
                              / len(norms_log[n]["cos_dW_W"])),
            "frac_changed_mean": (sum(norms_log[n]["frac_changed"])
                                  / len(norms_log[n]["frac_changed"])),
            "displacement_norm": disp,
            "path_length_events": path_ev,
            "directed_ratio": disp / (path_ev + 1e-12),
            "W_norm_start": w_start[n].norm().item(),
            "W_norm_end": w_fin.norm().item(),
            "lags": lags,
        }
        print(f"[C] {n}: events={len(ev_disk)}/{len(norms_log[n]['disk'])} "
              f"disk̄={analysis[n]['per_step_norm_mean']['disk']:.3f} "
              f"hebb̄={analysis[n]['per_step_norm_mean']['hebb']:.3f} "
              f"decorr̄={analysis[n]['per_step_norm_mean']['decorr']:.3f} "
              f"soft̄={analysis[n]['per_step_norm_mean']['soft']:.3f} "
              f"位移比={analysis[n]['directed_ratio']:.4f} "
              f"cos(dW,W)={analysis[n]['cos_dW_W_mean']:+.4f}", flush=True)
        for lag in (1, 2, 3):
            for cal in ("disk", "hebb"):
                a = lags[lag][cal]
                sa = a.get("p75", {}).get("mean_sign_agree")
                pe = a.get("pearson_all", {}).get("mean")
                print(f"    lag{lag} {cal}: sign_agree(p75)={sa:.4f} "
                      f"pearson={pe:+.4f}"
                      if sa is not None and pe is not None else
                      f"    lag{lag} {cal}: insufficient", flush=True)

    w1_fin = net.W1.data.float()
    out = {
        "chain_cos": cos_out,
        "loop": {"n_steps": N_STEPS, "step0": step0, "nan_at": nan_at,
                 "tau_mean": sum(tau_hist) / max(1, len(tau_hist)),
                 "r_mean": sum(r_hist) / max(1, len(r_hist)),
                 "e_mean": sum(e_hist) / max(1, len(e_hist)),
                 "q_mean": sum(q_hist) / max(1, len(q_hist))},
        "W1_frozen_check": {"delta_norm": (w1_fin - w1_start).norm().item()},
        "analysis": analysis,
    }
    with open("out/probe115_signcorr.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"saved -> out/probe115_signcorr.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
