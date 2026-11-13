"""exp132 回执: 四臂 ρ 扫描对比 (B1 引用 exp129 v460) + 预写预测裁决。

只读已有产物, 零训练, 零 GPU。

B1 (ρ=0.25) 的数据来源: exp129 v460 本体给五段量与逐步量; U/T 分解与 from-start
coherence 取自 exp130 v463 (逐位一致的同一条轨迹, 三重保真门已证)。
B1 的 C4 from-start 取 exp129 遥测 cos_cum_g_cum_run 最后一个非空值 (step 495)。
B1 无温度扫描 (权重已删) ⇒ s* / s=1 列为 n/a, 这是 exp131 已认的不可恢复损失。
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARMS = {
    "B1": ("out/v463-20261022-185708/exp130.json",
           "out/v460-20261021-220406/exp129_telemetry.jsonl"),
    "R2": ("out/v472-20261022-224522/exp132_R2.json",
           "out/v472-20261022-224522/exp132_R2_telemetry.jsonl"),
    "R3": ("out/v474-20261024-165104/exp132_R3.json",
           "out/v474-20261024-165104/exp132_R3_telemetry.jsonl"),
    "R4": ("out/v475-20261024-170947/exp132_R4.json",
           "out/v475-20261024-170947/exp132_R4_telemetry.jsonl"),
}
RHO = {"B1": 0.25, "R2": 0.05, "R3": 0.02, "R4": 0.01}
ORDER = ["B1", "R2", "R3", "R4"]


def main():
    D = {}
    for a, (jp, tp) in ARMS.items():
        j = json.load(open(os.path.join(ROOT, jp), encoding="utf-8"))
        rows = [json.loads(l) for l in open(os.path.join(ROOT, tp), encoding="utf-8") if l.strip()]
        rr = [r for r in rows if r.get("cos_rule") is not None]
        ca = j.get("coherence_all", {})
        segs = j["segment_verdicts"]
        sw = j.get("sweep") or {}
        crit = j.get("criteria", {})
        ub = j.get("update_budget", {})
        ts = [r["trust_scale"] for r in rr if r.get("trust_scale") is not None]
        cs = [r["wlm_cap_scale"] for r in rr if r.get("wlm_cap_scale") is not None]
        w = [r["W_lm_norm"] for r in rr if r.get("W_lm_norm") is not None]
        c4 = ca.get("cos_sumraw_sumg_eff")
        if c4 is None:
            vals = [r["cos_cum_g_cum_run"] for r in rr if r.get("cos_cum_g_cum_run") is not None]
            c4 = vals[-1] if vals else None
        n = sum(s["n_updates"] for s in segs)
        D[a] = {
            "rho": RHO[a], "N": len(rr), "n_seg": n,
            "s1_bpb": (j.get("fidelity") or {}).get("post_bpb_s1"),
            "sstar": (sw.get("best") or {}).get("scale"),
            "sstar_bpb": (sw.get("best") or {}).get("bpb"),
            "sstar_top1": (sw.get("best") or {}).get("top1"),
            "L1": (crit.get("L1_sstar_bpb_lt_7") or {}).get("pass"),
            "L2": (crit.get("L2_sstar_bpb_lt_5") or {}).get("pass"),
            "wh": (crit.get("C3_word_hits_ge_1") or {}).get("value"),
            "C4": c4,
            "cos_osc_w": sum(s["cos_osc"] * s["n_updates"] for s in segs) / n,
            "cos_osc_seg": [s["cos_osc"] for s in segs],
            "cos_rule_w": sum(s["cos_rule"] * s["n_updates"] for s in segs) / n,
            "trust_bind": sum(1 for t in ts if t < 1.0 - 1e-9) / len(ts),
            "trust_mean": sum(ts) / len(ts),
            "cap_bind": sum(1 for c in cs if c < 1.0 - 1e-9) / len(cs),
            "cap_mean": sum(cs) / len(cs),
            "gamma_mean": sum(1.0 - c for c in cs) / len(cs),
            "W_first": w[0], "W_end": w[-1], "W_max": max(w),
            "coh": {k: ca.get("coherence_" + k) for k in ("g", "raw", "eff", "betar", "rad")},
            "RA": ca.get("RA", {}),
            "U": ca.get("sum_betar_norm"), "T": ca.get("sum_rad_norm"),
            "E": ca.get("sum_dw_eff_norm"),
            "cosUT": ca.get("cos_betar_rad"),
            "T_over_E": (ca.get("sum_rad_norm") / ca.get("sum_dw_eff_norm")
                         if ca.get("sum_rad_norm") and ca.get("sum_dw_eff_norm") else None),
            "beh1": sum(1 for r in rows if r.get("behavior") == 1),
            "budget_flag": ub.get("flag"),
            "abort": j.get("abort"),
            "pre_bpb": j.get("pre_bpb"),
            "secs": j.get("secs"),
            "seg_cgc": [s["cos_cum_g_cum"] for s in segs],
        }

    L = []
    A = L.append
    A("=" * 112)
    A("exp132 回执 — 四臂 ρ 扫描 (B1 引用 exp129 v460; 强制温度扫描)")
    A("=" * 112)
    for a in ORDER:
        d = D[a]
        A(f"{a} (ρ={d['rho']}): s=1 bpb={d['s1_bpb']} | s*={d['sstar']} s* bpb={d['sstar_bpb']} "
          f"| L1<7.0 {'PASS' if d['L1'] else 'FAIL'} L2<5.0 {'PASS' if d['L2'] else 'FAIL'} | "
          f"top1@s*={d['sstar_top1']} word_hits={d['wh']} | abort={d['abort']} "
          f"pre_bpb={d['pre_bpb']} N={d['N']} secs={d['secs']}")
    A("  (B1 无温度扫描 — 权重已删, exp131 已认的不可恢复损失; s*/s* bpb 两列对 B1 为 n/a)")
    A("")

    def _bf(v, fmt=">5"):
        return "n/a" if v is None else format(v, fmt)

    A("表 1 · 主表 (四臂同口径)")
    A(f"{'臂':>4} {'ρ':>6} {'s*':>5} {'s* bpb':>8} {'L1':>5} {'s=1 bpb':>9} {'Δs1 vs pre':>11} "
      f"{'top1@s*':>9} {'wh':>3} {'C4':>7} {'cos_osc':>9} {'trust绑':>8} {'cap绑':>7} "
      f"{'mean|γ|':>8} {'‖T‖/‖Σe‖':>9} {'N':>4} {'预算':>5}")
    for a in ORDER:
        d = D[a]
        ds1 = (d["s1_bpb"] - 7.664317404722619) if d["s1_bpb"] else None
        A(f"{a:>4} {d['rho']:>6} {str(d['sstar']):>5} "
          f"{('%.4f' % d['sstar_bpb']) if d['sstar_bpb'] else 'n/a':>8} "
          f"{('PASS' if d['L1'] else 'FAIL'):>5} {d['s1_bpb']:>9.4f} {ds1:>+11.4f} "
          f"{('%.4f' % d['sstar_top1']) if d['sstar_top1'] else 'n/a':>9} {str(d['wh']):>3} "
          f"{d['C4']:>7.4f} {d['cos_osc_w']:>9.4f} {d['trust_bind']:>8.3f} "
          f"{d['cap_bind']:>7.3f} {d['gamma_mean']:>8.5f} {d['T_over_E']:>9.3f} "
          f"{d['N']:>4} {_bf(d['budget_flag'])}")
    A("")

    A("表 2 · 预写可证伪预测的裁决 (预登记 §一, 不许事后改)")
    A("  P1: ∃ ρ*≤0.05 使 cos_osc 段均 ≥ −0.10 且无中止、s* bpb 不劣于基线")
    A("  P2: 所有 ρ≤0.05 要么振荡仍在 (cos_osc < −0.10) 要么 bpb 明显劣化")
    A("  P3: ‖T‖/‖ΣdW_eff‖ 随 ρ 减小单调下降")
    A(f"{'臂':>4} {'ρ':>6} {'cos_osc 加权':>13} {'≥−0.10?':>9} {'‖T‖/‖Σe‖':>10} "
      f"{'s* bpb':>8} {'P1':>6} {'P2':>6} {'P3':>6}")
    for a in ORDER:
        d = D[a]
        p1 = "—" if a == "B1" else ("成立" if d["cos_osc_w"] >= -0.10 else "否")
        p2 = "—" if a == "B1" else ("支持" if d["cos_osc_w"] < -0.10 else "否")
        p3 = ("成立" if d["T_over_E"] < D["B1"]["T_over_E"] else "否") if a != "B1" else "基线"
        A(f"{a:>4} {d['rho']:>6} {d['cos_osc_w']:>13.4f} "
          f"{('是' if d['cos_osc_w'] >= -0.10 else '否'):>9} {d['T_over_E']:>10.3f} "
          f"{('%.4f' % d['sstar_bpb']) if d['sstar_bpb'] else 'n/a':>8} {p1:>6} {p2:>6} {p3:>6}")
    A("")
    A("  P3 单调性检查 (‖T‖/‖Σe‖ 随 ρ): " +
      " → ".join(f"{a}={D[a]['T_over_E']:.3f}" for a in ORDER))
    A("  ⇒ " + ("单调下降, P3 成立" if all(D[ORDER[i]]["T_over_E"] > D[ORDER[i+1]]["T_over_E"]
                                          for i in range(3)) else "非单调, P3 不成立"))
    A("")

    def _f(v, fmt=">8.2f"):
        return "n/a" if v is None else format(v, fmt)

    A("表 3 · U/T 分解 (全程 from-start; B1 值引自 exp130 v463, 逐位一致轨迹)")
    A(f"{'臂':>4} {'ρ':>6} {'coh_g':>7} {'coh_raw':>8} {'raw/g':>7} {'coh_eff':>8} "
      f"{'‖U‖':>8} {'‖T‖':>8} {'‖Σe‖':>8} {'cos(U,T)':>9} {'‖T‖/‖Σe‖':>9} {'coh_T':>7}")
    for a in ORDER:
        d = D[a]
        A(f"{a:>4} {d['rho']:>6} {d['coh']['g']:>7.4f} {d['coh']['raw']:>8.4f} "
          f"{(d['coh']['raw']/d['coh']['g']):>7.3f} {d['coh']['eff']:>8.4f} "
          f"{_f(d['U'])} {_f(d['T'])} {_f(d['E'])} {_f(d['cosUT'], '>9.4f')} "
          f"{_f(d['T_over_E'], '>9.3f')} {d['coh']['rad']:>7.4f}")
    A("")

    A("表 4 · 区制与 ‖W‖")
    A(f"{'臂':>4} {'ρ':>6} {'trust绑':>8} {'mean t':>8} {'cap绑':>7} {'mean cap':>9} "
      f"{'mean|γ|':>9} {'‖W‖首':>8} {'‖W‖末':>8} {'‖W‖max':>8} {'末/首':>7} {'beh==1':>7}")
    for a in ORDER:
        d = D[a]
        A(f"{a:>4} {d['rho']:>6} {d['trust_bind']:>8.3f} {d['trust_mean']:>8.4f} "
          f"{d['cap_bind']:>7.3f} {d['cap_mean']:>9.5f} {d['gamma_mean']:>9.5f} "
          f"{d['W_first']:>8.2f} {d['W_end']:>8.2f} {d['W_max']:>8.2f} "
          f"{(d['W_end']/d['W_first']):>7.3f} {d['beh1']:>7}")
    A("")

    A("表 5 · cos_osc 五段 (振荡是否随 ρ 减小而消失)")
    A(f"{'臂':>4} {'ρ':>6} " + " ".join(f"{'seg'+str(i+1):>8}" for i in range(5)) + "  加权均值")
    for a in ORDER:
        d = D[a]
        A(f"{a:>4} {d['rho']:>6} " + " ".join(f"{v:>8.4f}" for v in d["cos_osc_seg"])
          + f"  {d['cos_osc_w']:+.4f}")
    A("")
    A("表 6 · C4 双口径 (from-start 全程 vs 五段原值; 禁止混用)")
    A(f"{'臂':>4} {'ρ':>6} {'C4 from-start':>15} {'五段 cos_cum_g_cum':>46}")
    for a in ORDER:
        d = D[a]
        A(f"{a:>4} {d['rho']:>6} {d['C4']:>15.4f} " +
          " ".join(f"{v:+.4f}" for v in d["seg_cgc"]))
    A("")

    outdir = os.path.join(ROOT, "out", "exp132_report")
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "exp132_report.txt")
    open(p, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n写出 {p}")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
