"""exp131 回执: 四臂对比 (B1 引用 exp129 v460) + 交互效应 + 机制落支。

只读已有产物, 零训练, 零 GPU。
关键一致性处理: B1 (exp129 v460) 没有落盘 from-start 的 cos(ΣdW_eff, Σg),
但其遥测有逐步 running 值 `cos_cum_g_cum_run` —— 最后一个非空更新步的值即全程值。
故四臂的 C4 用**同一口径** (全程 from-start) 与**段均值** 双报。
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARMS = {
    "B1": ("out/v460-20261021-220406/exp129.json",
           "out/v460-20261021-220406/exp129_telemetry.jsonl"),
    "B2": ("out/v467-20261022-210620/exp131_B2.json",
           "out/v467-20261022-210620/exp131_B2_telemetry.jsonl"),
    "B3": ("out/v468-20261022-212442/exp131_B3.json",
           "out/v468-20261022-212442/exp131_B3_telemetry.jsonl"),
    "B4": ("out/v469-20261022-214411/exp131_B4.json",
           "out/v469-20261022-214411/exp131_B4_telemetry.jsonl"),
}
ORDER = ["B1", "B2", "B3", "B4"]


def load(jp, tp):
    j = json.load(open(os.path.join(ROOT, jp), encoding="utf-8"))
    rows = [json.loads(l) for l in open(os.path.join(ROOT, tp), encoding="utf-8") if l.strip()]
    return j, rows


def main():
    D = {}
    for a, (jp, tp) in ARMS.items():
        j, rows = load(jp, tp)
        rr = [r for r in rows if r.get("cos_rule") is not None]
        ca = j.get("coherence_all", {})
        # 全程 cos(ΣdW_eff, Σg): B1 取遥测 running 值最后一个非空; 其余用 json 键
        c4 = ca.get("cos_sumraw_sumg_eff")
        if c4 is None:
            vals = [r["cos_cum_g_cum_run"] for r in rr if r.get("cos_cum_g_cum_run") is not None]
            c4 = vals[-1] if vals else None
        segs = j["segment_verdicts"]
        ts = [r["trust_scale"] for r in rr if r.get("trust_scale") is not None]
        cs = [r["wlm_cap_scale"] for r in rr if r.get("wlm_cap_scale") is not None]
        D[a] = {
            "json": j, "rows": rr, "ca": ca, "C4": c4, "segs": segs,
            "N": len(rr),
            "post_bpb": j["fidelity"]["post_bpb_s1"],
            "top1": (j.get("criteria", {}).get("C2_top1_gt_025", {}) or {}).get("value"),
            "word_hits": (j.get("criteria", {}).get("C3_word_hits_ge_1", {}) or {}).get("value"),
            "pre_bpb": j.get("pre_bpb"),
            "abort": j.get("abort"),
            "secs": j.get("secs"),
            "trust_mean": (sum(ts) / len(ts)) if ts else None,
            "trust_bound": sum(1 for t in ts if t < 1.0 - 1e-9),
            "cap_mean": (sum(cs) / len(cs)) if cs else None,
            "cap_min": min(cs) if cs else None,
            "W_end": rr[-1]["W_lm_norm"] if rr else None,
            "W_max": max(r["W_lm_norm"] for r in rr if r.get("W_lm_norm")) if rr else None,
            "seg_cgc_mean": (sum(s["cos_cum_g_cum"] for s in segs) / len(segs)) if segs else None,
            "coh": {k: ca.get("coherence_" + k) for k in ("g", "raw", "eff", "betar", "rad")},
            "RA": ca.get("RA", {}),
            "U": ca.get("sum_betar_norm"), "T": ca.get("sum_rad_norm"),
            "E": ca.get("sum_dw_eff_norm"),
            "cosUT": ca.get("cos_betar_rad"),
            "ratio_raw_g": ca.get("ratio_raw_over_g"),
        }
        if D[a]["U"] and D[a]["E"]:
            D[a]["T_over_E"] = D[a]["T"] / D[a]["E"]

    L = []
    A = L.append
    A("=" * 104)
    A("exp131 回执 — 2×2 批处理 × cap 解耦 (B1 引用 exp129 v460)")
    A("=" * 104)
    for a in ORDER:
        d = D[a]
        A(f"{a}: post_bpb={d['post_bpb']} top1={d['top1']} word_hits={d['word_hits']} "
          f"pre_bpb={d['pre_bpb']} (协议门 {'PASS' if d['pre_bpb'] and abs(d['pre_bpb']-7.664317404722619)<1e-6 else 'FAIL'}) "
          f"abort={d['abort']} N_upd={d['N']} secs={d['secs']}")
    A("")

    A("表 1 · 判据矩阵 (预登记 §四)")
    A(f"{'臂':>4} {'C1 bpb<5':>10} {'C2 top1>.25':>12} {'C3 wh>=1':>10} {'C4 cumcos>.5':>14}")
    for a in ORDER:
        d = D[a]
        A(f"{a:>4} {str(d['post_bpb'] < 5.0):>10} {str((d['top1'] or 0) > 0.25):>12} "
          f"{str((d['word_hits'] or 0) >= 1):>10} {str(d['C4'] is not None and d['C4'] > 0.5):>14}")
    A("")

    A("表 2 · 主效应与交互 (预写公式)")
    b1, b2, b3, b4 = (D[a]["post_bpb"] for a in ORDER)
    A(f"  post bpb:  B1={b1:.6f} B2={b2:.6f} B3={b3:.6f} B4={b4:.6f}")
    A(f"    cap 解耦主效应 (K=1):  B2−B1 = {b2-b1:+.6f}")
    A(f"    批处理主效应   (K=1):  B3−B1 = {b3-b1:+.6f}")
    A(f"    交互 B4−B2−B3+B1   = {b4-b2-b3+b1:+.6f}   ← 正=合起来比各自更差")
    c1, c2, c3, c4v = (D[a]["C4"] for a in ORDER)
    A(f"  C4 全程累积 cos:  B1={c1:.4f} B2={c2:.4f} B3={c3:.4f} B4={c4v:.4f}")
    A(f"    cap 解耦主效应: {c2-c1:+.4f} | 批处理主效应: {c3-c1:+.4f} | "
      f"交互: {c4v-c2-c3+c1:+.4f}")
    s1, s2, s3, s4 = (D[a]["seg_cgc_mean"] for a in ORDER)
    A(f"  C4 段均值 (四臂同口径): B1={s1:.4f} B2={s2:.4f} B3={s3:.4f} B4={s4:.4f}")
    A(f"    cap 解耦主效应: {s2-s1:+.4f} | 批处理主效应: {s3-s1:+.4f} | "
      f"交互: {s4-s2-s3+s1:+.4f}")
    A("")

    A("表 3 · 四臂可比口径 (全部来自各臂 segment_verdicts / 遥测; B1 = exp129 v460,")
    A("        其未落盘 ‖ΣdW_raw‖ 与 U/T 分解 — exp130 识别的缺口, 故这两列对 B1 为 n/a)")
    A(f"{'臂':>4} {'N':>4} {'cos_rule':>9} {'cos_osc':>8} {'cum_raw_g':>10} {'cum_eff_g':>10} "
      f"{'coh_g':>7} {'coh_eff':>8} {'coh_raw':>8}")
    for a in ORDER:
        d = D[a]
        segs, rows = d["segs"], d["rows"]
        n = sum(s["n_updates"] for s in segs)
        cr = sum(s["cos_rule"] * s["n_updates"] for s in segs) / n
        co = sum(s["cos_osc"] * s["n_updates"] for s in segs) / n
        crg = sum(s["cos_cum_raw_g_cum"] * s["n_updates"] for s in segs) / n
        ceg = sum(s["cos_cum_g_cum"] * s["n_updates"] for s in segs) / n
        # coherence_g / coherence_eff: 有则用段值, 无则用 sum_*_norm ÷ Σ‖·‖ 现算
        cg = ce = None
        try:
            cg = sum(s["coherence_g"] * s["n_updates"] for s in segs) / n
            ce = sum(s["coherence_eff"] * s["n_updates"] for s in segs) / n
        except KeyError:
            lo = hi = 0
            cg = ce = None
        craw = None
        try:
            craw = sum(s["coherence_raw"] * s["n_updates"] for s in segs) / n
        except KeyError:
            craw = None
        if cg is None:
            sg = se = 0.0
            for s in segs:
                a0, b0 = (s["segment"] - 1) * 100, s["segment"] * 100
                rr = [r for r in rows if a0 < r["step"] <= b0 and r.get("cos_rule") is not None]
                sg += s["sum_neg_g_norm"]
                se += s["sum_dw_eff_norm"]
            tot_g = sum(r["neg_g_norm"] for r in rows if r.get("neg_g_norm") is not None)
            tot_e = sum(r["dW_eff_norm"] for r in rows if r.get("dW_eff_norm") is not None)
            cg = sg / tot_g
            ce = se / tot_e
        d["seg"] = {"n": n, "cos_rule": cr, "cos_osc": co, "cum_raw_g": crg,
                    "cum_eff_g": ceg, "coh_g": cg, "coh_eff": ce, "coh_raw": craw}
        A(f"{a:>4} {n:>4} {cr:>9.4f} {co:>8.4f} {crg:>10.4f} {ceg:>10.4f} "
          f"{cg:>7.4f} {ce:>8.4f} {('n/a' if craw is None else f'{craw:8.4f}'):>8}")
    A("")

    A("表 3b · U/T 分解 (仅 B2/B3/B4; B1 未落盘向量和, 不可补)")
    A(f"{'臂':>4} {'N':>4} {'coh_raw':>8} {'raw/g':>7} {'‖U‖':>10} {'‖T‖':>10} {'‖Σe‖':>9} "
      f"{'cos(U,T)':>9} {'‖T‖/‖Σe‖':>9} {'coh_T':>7}")
    for a in ORDER[1:]:
        d = D[a]
        A(f"{a:>4} {d['N']:>4} {d['coh']['raw']:>8.4f} {d['ratio_raw_g']:>7.3f} "
          f"{d['U']:>10.2f} {d['T']:>10.2f} {d['E']:>9.2f} {d['cosUT']:>9.4f} "
          f"{d['T_over_E']:>9.2f} {d['coh']['rad']:>7.4f}")
    A("")

    A("表 4 · trust/cap 区制 (批处理是否把系统推离 trust 饱和?)")
    A(f"{'臂':>4} {'trust均值':>9} {'绑定步/N':>10} {'cap均值':>9} {'cap_min':>8} "
      f"{'‖W‖末':>8} {'‖W‖max':>8}")
    for a in ORDER:
        d = D[a]
        A(f"{a:>4} {d['trust_mean']:>9.4f} {str(d['trust_bound'])+'/'+str(d['N']):>10} "
          f"{d['cap_mean']:>9.5f} {d['cap_min']:>8.4f} {d['W_end']:>8.2f} {d['W_max']:>8.2f}")
    A("")

    A("表 5 · 五段序列")
    A(f"{'臂':>4} {'cos_osc':>34} {'cos_rule(生效)':>34} {'cos_rule_rawwin':>34}")
    for a in ORDER:
        d = D[a]
        A(f"{a:>4} " + " ".join(f"{s['cos_osc']:+.3f}" for s in d["segs"]))
        A(f"{'':>4} " + " ".join(f"{s['cos_rule']:+.3f}" for s in d["segs"]))
        A(f"{'':>4} " + " ".join(
            f"{(s.get('cos_rule_rawwin') if s.get('cos_rule_rawwin') is not None else float('nan')):+.3f}"
            for s in d["segs"]))
    A("")

    outdir = os.path.join(ROOT, "out", "exp131_report")
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "exp131_report.txt")
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
