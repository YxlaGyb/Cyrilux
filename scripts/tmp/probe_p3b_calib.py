"""P3-b 标定探针: 行为门 κ 网格 (CPU 闭环仿真, 零 GPU 网格扫描).

只读检查点 (不落盘, 绝不调 maybe_prune — 探针不得修改模型).
流程: ① 从检查点跑正常/诱导饥荒两段交替步 (感知/回声), 记录每步
      (行为, f_perc, f_eco, trace_norm, stress, A, nov);
      ② CPU 闭环仿真 P3-b 账本+门: per-behavior F 分账, E/E_ref/应激/MAD/R,
         行为账本 gain EMA, p_say = σ(κ_g·(g_eco−g_perc) − κ_s·stress
         + κ_A·(A−0.5) − κ_n·(nov−0.5)), 采样路由, 消费对应行为 F 流;
      ③ 网格: κ_g×κ_s×κ_A×κ_n×α_g = 750 组合 + div 子网格 (R 契约复测).
判据 (正常段): echo 占比 ∈[0.3,0.7] / 均连续回声 ≤30 / 均连续感知 ≤40 /
      max 连续回声 <120 / 应激均值 <0.3 / median(leg) ∈[0.3,1];
      饥荒段: p_say 均值 <0.25 (应激压制"说").
输出: out/probe_p3b_calib.json — 网格全表 + 推荐 κ + f/A/nov 序列留档.
"""

import argparse
import json
import math
import os
import random
import sys

import torch

torch.set_grad_enabled(False)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from model import CyreneModel, DensePCNet  # noqa: E402
from model.dense.learning.metabolism import gate_psay  # noqa: E402
from dataset import DualChannelDataset  # noqa: E402

S_MAX = 256
SEED_N = 16


def _collect(net, ds, dev, steps, mode, rng):
    """交替步 (感知奇/回声偶), 返回每步 (行为, f 按轨, trace_norm, stress, A, nov)."""
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    last_tail = None
    recs = []
    for i in range(1, steps + 1):
        if i % 2 == 1:
            if mode == "normal":
                b, _ = ds[idxs[(i // 2) % len(idxs)]]
                x = b.unsqueeze(0).to(dev)
            else:  # famine_rand: 随机字节 = 绝对货币下乱码暴涨
                x = torch.randint(0, 256, (1, S_MAX), dtype=torch.long, device=dev)
            net.learn(x)
            f = float(net._metab_F_prev_perc.item())
            nov = float(net._metab_nov.item())
            last_tail = x[0, -SEED_N:]
        else:
            net._echo_seed = (
                last_tail.unsqueeze(0)
                if last_tail is not None
                else torch.zeros(1, 1, dtype=torch.long, device=dev)
            )
            net.learn(None, free_run=False)
            f = float(net._metab_F_prev_eco.item())
            nov = None
        a = float(net._intr_sin.index_select(0, net._intr_cnt.long().squeeze(0)).item())
        net.sync_gate()  # 渲染 CPU 镜像 (本探针只读, 不用于路由 — 交替由调用方锁定)
        recs.append({
            "t": i % 2,  # 1=感知, 0=回声
            "f": f,
            "tr": float(net.W_act_elig.norm().item()) if i % 2 == 0 else None,
            "stress": float(getattr(net, "_metab_stress", torch.tensor(0.0)).item()),
            "A": a,
            "nov": nov,
        })
    return recs


def _sim(recs, k_g, k_s, k_a, k_n, a_g, div, cfg, w1_rows, seed):
    """CPU 闭环仿真 (per-behavior F 分账 + 门采样路由), 返回指标 dict."""
    rng = random.Random(seed)
    # 收集流 → 按行为拆流 (仿真按自己的行为序消费)
    perc_fs = [r["f"] for r in recs if r["t"] == 1]
    eco_fs = [r["f"] for r in recs if r["t"] == 0]
    trace_fs = [r["tr"] for r in recs if r["t"] == 0 and r["tr"] is not None]
    nov_fs = [r["nov"] for r in recs if r["nov"] is not None]
    # 状态 (与 metabolism.py 公式一一对应)
    e = eref = mad = 0.0
    fp_p = fp_e = None
    g_p = g_e = 0.0
    nov = 0.5  # 首个回声步前 (感知从未运行 → 中性)
    perc_i = eco_i = nov_i = trace_i = 0
    settled = False
    es, stresses, psays, legs, behaviors = [], [], [], [], []
    for step, rec in enumerate(recs):
        stress = stresses[-1] if stresses else 0.0
        p = gate_psay(g_e, g_p, stress, rec["A"], nov, k_g, k_s, k_a, k_n)
        hit = rng.random() < p
        behaviors.append(1 if hit else 0)
        psays.append(p)
        if hit:  # 回声
            f = eco_fs[eco_i] if eco_i < len(eco_fs) else eco_fs[-1]
            eco_i += 1
            if fp_e is None:
                fp_e = f
                df = 0.0
            else:
                df = fp_e - f
                fp_e = f
            cost = cfg.metab_cost_learn * math.sqrt(f) + cfg.metab_cost_mem * w1_rows + cfg.metab_cost_say
            if not settled:
                settled = True
                continue
            e = e * (1.0 - cfg.metab_d) + cfg.metab_c * df - cost
            r = math.tanh(df / (div * mad)) if mad > 0 else 0.0
            g_e = g_e * (1.0 - a_g) + a_g * abs(r)
            legs.append((trace_fs[trace_i] if trace_i < len(trace_fs) else trace_fs[-1]) * abs(r))
            trace_i += 1
        else:  # 感知
            f = perc_fs[perc_i] if perc_i < len(perc_fs) else perc_fs[-1]
            perc_i += 1
            if fp_p is None:
                fp_p = f
                df = 0.0
            else:
                df = fp_p - f
                fp_p = f
            nov = nov_fs[nov_i] if nov_i < len(nov_fs) else 0.5
            nov_i += 1
            cost = cfg.metab_cost_perc * f + cfg.metab_cost_learn * math.sqrt(f) + cfg.metab_cost_mem * w1_rows
            if not settled:
                settled = True
                continue
            e = e * (1.0 - cfg.metab_d) + cfg.metab_c * df - cost
            r = math.tanh(df / (div * mad)) if mad > 0 else 0.0
            g_p = g_p * (1.0 - a_g) + a_g * abs(r)
        eref = eref * (1.0 - cfg.metab_eref_rate) + cfg.metab_eref_rate * e
        stress = math.tanh(max(0.0, eref - e) * cfg.metab_famine_scale)
        mad = mad * cfg.metab_mad_alpha + (1.0 - cfg.metab_mad_alpha) * abs(df)
        if mad <= 0:
            mad = max(abs(df), 1e-5)
        es.append(e)
        stresses.append(stress)
    # 指标
    n = len(behaviors)
    share = sum(behaviors) / n
    runs_eco, runs_perc, cur, cur_b = [], [], 0, behaviors[0]
    # 连续段统计
    runs = [(b, len(list(g))) for b, g in __import__("itertools").groupby(behaviors)]
    e_runs = [l for b, l in runs if b == 1]
    p_runs = [l for b, l in runs if b == 0]
    med_leg = sorted(legs)[len(legs) // 2] if legs else 0.0
    return {
        "echo_share": round(share, 4),
        "mean_echo_run": round(sum(e_runs) / len(e_runs), 1) if e_runs else 0.0,
        "mean_perc_run": round(sum(p_runs) / len(p_runs), 1) if p_runs else 0.0,
        "max_echo_run": max(e_runs) if e_runs else 0,
        "stress_mean": round(sum(stresses) / len(stresses), 4) if stresses else 0.0,
        "median_leg": round(med_leg, 4),
        "psay_mean": round(sum(psays) / len(psays), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/exp115_p3c_ev.pt")
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--normal-steps", type=int, default=600)
    ap.add_argument("--famine-steps", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/probe_p3b_calib.json")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    net = DensePCNet.load(args.ckpt).to(dev)
    ds = DualChannelDataset(args.data, max_length=S_MAX, max_samples=1270000, lazy=True)
    rng = random.Random(args.seed)
    cfg = net.cfg
    w1_rows = net.W1.shape[0]

    print("collecting normal ...", flush=True)
    normal = _collect(net, ds, dev, args.normal_steps, "normal", rng)
    print("collecting famine_rand ...", flush=True)
    famine = _collect(net, ds, dev, args.famine_steps, "famine_rand", rng)

    grid = {}
    for k_g in (1.0, 2.0, 3.0, 4.0, 5.0):
        for k_s in (1.0, 2.0, 3.0, 4.0):
            grid[f"kg{k_g}_ks{k_s}"] = {}
            for k_a in (0.5, 1.0, 2.0):
                for k_n in (0.5, 1.0, 2.0):
                    for a_g in (0.02, 0.05, 0.1, 0.2):
                        m = _sim(normal, k_g, k_s, k_a, k_n, a_g, cfg.metab_tanh_div,
                                 cfg, w1_rows, args.seed)
                        fm = _sim(famine, k_g, k_s, k_a, k_n, a_g, cfg.metab_tanh_div,
                                  cfg, w1_rows, args.seed + 1)
                        key = f"ka{k_a}_kn{k_n}_ag{a_g}"
                        grid[f"kg{k_g}_ks{k_s}"][key] = {
                            **m, "famine_psay_mean": fm["psay_mean"],
                        }
    # 选优: 正常段全部判据 + 饥荒 p_say 压制; 评分 = 偏离 0.5 占比 + 饥荒 p_say + max 回声段
    cands = []
    for kg, ks_group in grid.items():
        for key, m in ks_group.items():
            if not (0.3 <= m["echo_share"] <= 0.7):
                continue
            if m["mean_echo_run"] > 30 or m["mean_perc_run"] > 40:
                continue
            if m["max_echo_run"] >= 120:
                continue
            if m["stress_mean"] >= 0.3:
                continue
            if not (0.3 <= m["median_leg"] <= 1.0):
                continue
            if m["famine_psay_mean"] >= 0.25:
                continue
            score = abs(m["echo_share"] - 0.5) + 0.5 * m["famine_psay_mean"] + 0.001 * m["max_echo_run"]
            cands.append((score, kg, key, m))
    cands.sort(key=lambda x: x[0])
    report = {
        "ckpt": args.ckpt,
        "normal_steps": args.normal_steps,
        "famine_steps": args.famine_steps,
        "w1_rows": w1_rows,
        "tab_div": cfg.metab_tanh_div,
        "grid": grid,
        "candidates": [(kg, key, round(s, 5), m) for s, kg, key, m in cands[:40]],
        "recommended": ({"kappa_g": float(kg.split("_")[0][2:]),
                         "kappa_s": float(kg.split("_")[1][2:]),
                         "kappa_a": float(key.split("_")[0][2:]),
                         "kappa_n": float(key.split("_")[1][2:]),
                         "alpha_g": float(key.split("_")[2][2:])}
                        if cands else None),
        # 序列留档 (换 κ 免重跑): [t, f, trace|None, stress, A, nov|None]
        "series": {"normal": [[r["t"], r["f"], r["tr"], r["stress"], r["A"], r["nov"]] for r in normal],
                   "famine": [[r["t"], r["f"], r["tr"], r["stress"], r["A"], r["nov"]] for r in famine]},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"candidates={len(cands)}")
    if cands:
        print(f"recommended = {cands[0][1]} {cands[0][2]} score={cands[0][0]:.5f} {cands[0][3]}")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
