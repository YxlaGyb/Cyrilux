"""P3-c 术前标定探针: 成本 κ 网格 (CPU 重放, 零 GPU 网格扫描).

只读检查点 (不落盘, 绝不调 maybe_prune — 探针不得修改模型).
流程: ① 从检查点跑正常/诱导饥荒两段交替步 (感知/回声), 记录每步 (f_now, 步型, trace_norm);
      ② CPU 重放 P3-c 新账本: E ← E(1−d) + c·ΔF − Σcost(κ), E_ref = E 慢 EMA,应激,
         R = tanh(ΔF/(div·MAD)); ③ 扫 κ 总量标度 a × 说/看比例 r_s × div 网格.
判据: 健康段末 500 步 ⟨E⟩≈⟨E_ref⟩ (应激均值 <0.3) / 支出结构按步型可辨识 /
      median(leg) ∈ [0.3,1] / R 均值无常态负.
输出: out/probe_p3c_calib.json — 网格全表 + 推荐 κ 组合.
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
from dataset import DualChannelDataset  # noqa: E402

S_MAX = 256
SEED_N = 16


def _collect(net, ds, dev, steps, mode, rng):
    """交替步 (感知奇/回声偶), 返回每步 (f, 步型, trace_norm|echo 步)."""
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
            last_tail = x[0, -SEED_N:]
        else:
            net._echo_seed = (
                last_tail.unsqueeze(0)
                if last_tail is not None
                else torch.zeros(1, 1, dtype=torch.long, device=dev)
            )
            net.learn(None, free_run=False)
        # P3-b 分账: 取刚运行行为的 F 轨 (odd=感知, even=回声) — 与旧单轨序列同语义
        if i % 2 == 1:
            f = float(net._metab_F_prev_perc.item())
        else:
            f = float(net._metab_F_prev_eco.item())
        tr = float(net.W_act_elig.norm().item()) if i % 2 == 0 else None
        recs.append({"t": i % 2, "f": f, "tr": tr})
    return recs


def _replay(recs, kappa, d, c, alpha, div, eref_rate, famine_scale, w1_rows):
    """CPU 重放 P3-c 账本, 返回 (E 序列, E_ref 序列, stress 序列, R 序列, leg 序列)."""
    kp, kl, km, ks = kappa
    e = 0.0
    eref = 0.0
    f_prev = None
    mad = 0.0
    es, refs, stresses, rs, legs = [], [], [], [], []
    for k, rec in enumerate(recs):
        f = rec["f"]
        if k == 0:  # 首步哨兵: 只登记 F_prev
            f_prev = f
            es.append(e)
            refs.append(eref)
            stresses.append(0.0)
            rs.append(0.0)
            legs.append(0.0)
            continue
        df = f_prev - f
        f_prev = f
        cost = kl * math.sqrt(f) + km * w1_rows
        if rec["t"] == 1:
            cost += kp * f
        else:
            cost += ks
        e = e * (1.0 - d) + c * df - cost
        eref = eref * (1.0 - eref_rate) + eref_rate * e
        stress = math.tanh(max(0.0, eref - e) * famine_scale)
        mad = mad * alpha + (1.0 - alpha) * abs(df)
        if mad <= 0:
            mad = max(abs(df), 1e-5)
        r = math.tanh(df / (div * mad)) if mad > 0 else 0.0
        es.append(e)
        refs.append(eref)
        stresses.append(stress)
        rs.append(r)
        legs.append((rec["tr"] or 0.0) * abs(r))
    return es, refs, stresses, rs, legs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/exp115_p2_ev.pt")
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--normal-steps", type=int, default=600)
    ap.add_argument("--famine-steps", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/probe_p3c_calib.json")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # config=None → load 从检查点形状推导 (P2 后已修剪: l4=925/K=1, 显式全尺寸 cfg 会错配)
    net = DensePCNet.load(args.ckpt).to(dev)
    ds = DualChannelDataset(args.data, max_length=S_MAX, max_samples=1270000, lazy=True)
    rng = random.Random(args.seed)

    print("collecting normal ...", flush=True)
    normal = _collect(net, ds, dev, args.normal_steps, "normal", rng)
    print("collecting famine_rand ...", flush=True)
    famine = _collect(net, ds, dev, args.famine_steps, "famine_rand", rng)

    w1_rows = net.W1.shape[0]
    d, c = net.cfg.metab_d, net.cfg.metab_c
    alpha, div0 = net.cfg.metab_mad_alpha, net.cfg.metab_tanh_div
    eref_rate, fscale = net.cfg.metab_eref_rate, net.cfg.metab_famine_scale
    print(f"w1_rows={w1_rows} d={d} c={c} alpha={alpha} div0={div0}")

    grid = {}
    # κ 参数化: κ_p = a·1e-3, κ_l = a·1e-3, κ_s = r_s·a·1e-3, κ_m = a·(3e-7)
    for a in (0.0, 0.3, 0.5, 1.0, 2.0, 3.0):
        for rs in (0.5, 1.0, 2.0):
            for div in (1.0, 1.5, 2.0, 3.0):
                kappa = (a * 1e-3, a * 1e-3, a * 3e-7, rs * a * 1e-3)
                kp, kl, km, ks = kappa
                es, refs, stresses, rs_, legs = _replay(
                    normal, kappa, d, c, alpha, div, eref_rate, fscale, w1_rows
                )
                tail = es[-500:]
                tail_ref = refs[-500:]
                e_dev = sum(abs(e - r) for e, r in zip(tail, tail_ref)) / 500
                stress_mean = sum(stresses[-500:]) / 500
                # 支出/收入按步型分账 (ΔE 口径)
                de_perc, de_echo = [], []
                for k in range(1, len(normal)):
                    de = es[k] - es[k - 1]
                    (de_perc if normal[k]["t"] == 1 else de_echo).append(de)
                r_echo = [rs_[k] for k in range(1, len(normal)) if normal[k]["t"] == 0]
                # leg 契约口径 (R1): leg = |raw 代谢 R| — 报告字段 R=生存信号=R/‖迹‖,
                # leg = 迹·|R_报告| = |R_raw| (trace 与除迹量对消). median(|R|) ∈ [0.3,1].
                med_leg = sorted(abs(x) for x in r_echo)[len(r_echo) // 2] if r_echo else 0.0
                r_mean = sum(r_echo) / len(r_echo) if r_echo else 0.0
                # 成本占比: mean(cost)/mean(|ΔE|) — 成本必须是账本的可辨识部分 (非装饰)
                cost_mean, de_abs_mean = 0.0, 0.0
                for k in range(1, len(normal)):
                    f = normal[k]["f"]
                    cost = kl * math.sqrt(f) + km * w1_rows
                    if normal[k]["t"] == 1:
                        cost += kp * f
                    else:
                        cost += ks
                    cost_mean += cost
                    de_abs_mean += abs(es[k] - es[k - 1])
                cost_mean /= len(normal) - 1
                de_abs_mean /= len(normal) - 1
                cost_share = cost_mean / de_abs_mean if de_abs_mean > 0 else 0.0
                grid[f"a{a}_rs{rs}_div{div}"] = {
                    "kappa": [round(x, 8) for x in kappa],
                    "E_dev_tail": round(e_dev, 6),
                    "stress_mean_tail": round(stress_mean, 6),
                    "de_mean_perc": round(sum(de_perc) / len(de_perc), 6),
                    "de_mean_echo": round(sum(de_echo) / len(de_echo), 6),
                    "de_sep": round(abs(sum(de_perc) / len(de_perc)
                                        - sum(de_echo) / len(de_echo)), 6),
                    "median_leg": round(med_leg, 4),
                    "r_mean_echo": round(r_mean, 5),
                    "cost_share": round(cost_share, 4),
                }
    # 饥荒段应激验证 (推荐组合下应激应显著 > 健康段)
    fes, frefs, fstresses, frs, flegs = _replay(
        famine, (1.0e-3, 1.0e-3, 3e-7, 1.0e-3), d, c, alpha, div0, eref_rate, fscale, w1_rows
    )

    # 选优: 应激<0.3 / median_leg∈[0.3,1] / r_mean 无常态负 / 成本占比≥0.3 (非装饰) /
    # 支出分离度最大 (成本结构可辨识), 平手取 |E−E_ref| 小
    cands = [v for v in grid.values() if v["stress_mean_tail"] < 0.3
             and 0.3 <= v["median_leg"] <= 1.0
             and v["r_mean_echo"] > -0.10
             and v["cost_share"] >= 0.3]
    best = None
    if cands:
        best = max(cands, key=lambda v: (v["de_sep"], -abs(v["E_dev_tail"])))
    report = {
        "ckpt": args.ckpt,
        "normal_steps": args.normal_steps,
        "famine_steps": args.famine_steps,
        "w1_rows": w1_rows,
        "grid": grid,
        "famine_stress_mean": round(sum(fstresses[-200:]) / 200, 6),
        "recommended": best,
        # f 序列留档 (换 κ/div 免重跑, P2 calib 同款): [t, f, trace_norm|None]
        "f_series": {"normal": [[r["t"], r["f"], r["tr"]] for r in normal],
                     "famine": [[r["t"], r["f"], r["tr"]] for r in famine]},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"famine stress_mean(tail) = {report['famine_stress_mean']:.4f}")
    print(f"recommended = {best}")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
