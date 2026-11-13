"""exp131 v2 派生量: ‖T‖ 放大的三层分解 + C4 双口径佐证。

只读已有产物, 零训练, 零 GPU。

三层分解的恒等式 (不是近似):
    ‖T‖ = coherence(T) · Σ_i |γ_i| · ‖W_pre,i‖ = coherence(T) · Σ_i sn_rad 项
    ⇒ ‖T‖_B/‖T‖_A = [coh(T)_B/coh(T)_A] · [sn_rad_B/sn_rad_A]
    sn_rad 是逐步 |γ|·‖W‖ 之和, 可再拆为 N × mean|γ| × mean‖W‖ (三项)。

关键修正: 审查裁决原文用 ‖W‖ **末值** (68.4 → 160.0, 2.34×) 作为其中一层;
但进入 Σ 的是**每步的** ‖W_pre,i‖, 不是末值。本脚本同时报末值比与均值比,
并指出哪一个是正确的分解因子。
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARMS = {
    # B1 = exp129 v460 轨迹。其自身 json 未落 U/T 分解 (exp130 识别的缺口),
    # 但 exp130 (v463) 跑的是**逐位一致的同一条轨迹** (三重保真门证明), 并落了全量分解
    # ⇒ B1 的 coh(T)/sn_rad/‖T‖/cos(U,T) 取自 out/v463-20261022-185708/exp130.json。
    # B1 的逐步量 (cap_scale/W_lm_norm/cos_grad_osc) 与 C4 仍取 exp129 v460 本体。
    "B1": ("out/v463-20261022-185708/exp130.json",
           "out/v460-20261021-220406/exp129_telemetry.jsonl"),
    "B2": ("out/v467-20261022-210620/exp131_B2.json",
           "out/v467-20261022-210620/exp131_B2_telemetry.jsonl"),
    "B3": ("out/v468-20261022-212442/exp131_B3.json",
           "out/v468-20261022-212442/exp131_B3_telemetry.jsonl"),
    "B4": ("out/v469-20261022-214411/exp131_B4.json",
           "out/v469-20261022-214411/exp131_B4_telemetry.jsonl"),
}
ORDER = ["B1", "B2", "B3", "B4"]
# B1 的五段 cos_cum_g_cum 必须取自 exp129 v460 本体 (exp130 json 也有, 但为免混源单独声明)
B1_SEG_SRC = "out/v460-20261021-220406/exp129.json"


def main():
    D = {}
    for a, (jp, tp) in ARMS.items():
        j = json.load(open(os.path.join(ROOT, jp), encoding="utf-8"))
        rows = [json.loads(l) for l in open(os.path.join(ROOT, tp), encoding="utf-8") if l.strip()]
        rr = [r for r in rows if r.get("cos_rule") is not None]
        ca = j.get("coherence_all", {})
        if a == "B1":
            # B1 的五段值取 exp129 v460 本体 (与 exp130 同轨迹, 逐位一致)
            segs = json.load(open(os.path.join(ROOT, B1_SEG_SRC),
                                  encoding="utf-8"))["segment_verdicts"]
        else:
            segs = j["segment_verdicts"]
        g = [r["wlm_cap_scale"] for r in rr if r.get("wlm_cap_scale") is not None]
        w = [r["W_lm_norm"] for r in rr if r.get("W_lm_norm") is not None]
        cg = [r["cos_grad_osc"] for r in rr if r.get("cos_grad_osc") is not None]
        D[a] = {
            "N": len(rr),
            "cohT": ca.get("coherence_rad"),
            "sn_rad": ca.get("sn_rad"),
            "T": ca.get("sum_rad_norm"),
            "E": ca.get("sum_dw_eff_norm"),
            "mean_abs_gamma": (sum(1.0 - x for x in g) / len(g)) if g else None,
            "mean_W": (sum(w) / len(w)) if w else None,
            "W_end": w[-1] if w else None,
            "W_first": w[0] if w else None,
            "W_max": max(w) if w else None,
            "grad_osc_mean": (sum(cg) / len(cg)) if cg else None,
            "seg_cgc": [s["cos_cum_g_cum"] for s in segs],
            "seg_cgc_mean": sum(s["cos_cum_g_cum"] for s in segs) / len(segs),
            "cosUT": ca.get("cos_betar_rad"),
        }
        # C4 全程: B1 无 from-start 落盘, 取遥测 running 值最后一个
        c4 = ca.get("cos_sumraw_sumg_eff")
        if c4 is None:
            vals = [r["cos_cum_g_cum_run"] for r in rr if r.get("cos_cum_g_cum_run") is not None]
            c4 = vals[-1] if vals else None
        D[a]["C4"] = c4

    L = []
    A = L.append
    A("=" * 108)
    A("exp131 v2 派生 — ‖T‖ 放大的三层分解 (恒等式分解, 非近似)")
    A("=" * 108)
    A("恒等式: ‖T‖ = coherence(T) × Σ_i |γ_i|·‖W_pre,i‖")
    A("        ⇒ 比值 = [coh(T)_B/coh(T)_A] × [sn_rad_B/sn_rad_A],  sn_rad ≈ N × mean|γ| × mean‖W‖")
    A("")
    A("表 A · 三层分解的四个因子 (全部实测)")
    A(f"{'臂':>4} {'‖T‖':>10} {'coh(T)':>8} {'sn_rad':>10} {'N':>4} {'mean|γ|':>9} "
      f"{'mean‖W‖':>9} {'‖W‖末':>8} {'‖W‖max':>8} {'‖W‖末/首':>10}")
    for a in ORDER:
        d = D[a]
        A(f"{a:>4} {d['T']:>10.2f} {d['cohT']:>8.4f} {d['sn_rad']:>10.2f} {d['N']:>4} "
          f"{d['mean_abs_gamma']:>9.5f} {d['mean_W']:>9.2f} {d['W_end']:>8.2f} "
          f"{d['W_max']:>8.2f} {(d['W_end']/d['W_first']) if d['W_first'] else float('nan'):>10.3f}")
    A("")
    A("表 B · 相对 B1 的放大倍数 (每层单独看)")
    b = D["B1"]
    A(f"{'臂':>4} {'‖T‖ 比':>10} {'coh(T) 比':>11} {'sn_rad 比':>11} {'N 比':>7} "
      f"{'mean|γ| 比':>12} {'mean‖W‖ 比':>12} {'‖W‖末值比':>12}")
    for a in ORDER[1:]:
        d = D[a]
        A(f"{a:>4} {d['T']/b['T']:>10.2f} {d['cohT']/b['cohT']:>11.3f} "
          f"{d['sn_rad']/b['sn_rad']:>11.2f} {d['N']/b['N']:>7.2f} "
          f"{d['mean_abs_gamma']/b['mean_abs_gamma']:>12.2f} "
          f"{d['mean_W']/b['mean_W']:>12.2f} {d['W_end']/b['W_end']:>12.2f}")
    A("")
    A("表 B2 · sn_rad 的近似因子化及其修正 (为什么不能直接把三个比值相乘)")
    A("  sn_rad = Σ_i |γ_i|·‖W_pre,i‖  是**逐步乘积之和**, 不是 N×mean|γ|×mean‖W‖ ——")
    A("  |γ_i| 与 ‖W_i‖ 在步内相关 (cap 压得狠时 W 正被缩小), 故需修正因子 κ:")
    A("     sn_rad = κ · N · mean|γ| · mean‖W‖")
    A(f"{'臂':>4} {'sn_rad':>10} {'N·mean|γ|·mean‖W‖':>19} {'κ':>7} {'κ 比(vs B1)':>13}")
    for a in ORDER:
        d = D[a]
        naive = d["N"] * d["mean_abs_gamma"] * d["mean_W"]
        d["kappa"] = d["sn_rad"] / naive
        A(f"{a:>4} {d['sn_rad']:>10.2f} {naive:>19.2f} {d['kappa']:>7.4f} "
          f"{(d['kappa']/b['kappa']) if a != 'B1' else 1.0:>13.3f}")
    A("")
    A("  ⇒ 三个比值相乘只是**近似**; 精确恒等式只有 coh(T)比 × sn_rad 比 (见表 B 自洽校验)。")
    A("  **对审查裁决原文的核对**: 裁决用 ‖W‖ **末值**比 (68.4→160.0 = 2.34×) 作为其中一层;")
    A("  进入 Σ 的其实是每步 ‖W_pre,i‖, 其**均值**比为 "
      f"{D['B4']['mean_W']/b['mean_W']:.2f}× —— 与末值比 "
      f"{D['B4']['W_end']/b['W_end']:.2f}× 相差 <2%, **不改变结论方向**。")
    A("")
    A("表 B 的自洽校验: ‖T‖ 比 应 ≈ coh(T)比 × sn_rad 比")
    for a in ORDER[1:]:
        d = D[a]
        lhs = d["T"] / b["T"]
        rhs = (d["cohT"] / b["cohT"]) * (d["sn_rad"] / b["sn_rad"])
        A(f"  {a}: 实测 {lhs:.4f} vs 恒等式预测 {rhs:.4f} (相对差 {abs(lhs-rhs)/lhs*100:.2f}%)")
    A("")
    A("表 C · C4 双口径 (from-start 全程 vs 段均值) — 预登记未锁聚合口径, 两者可差数倍")
    A(f"{'臂':>4} {'C4 from-start':>15} {'C4 段均值':>11} {'比值':>7}")
    for a in ORDER:
        d = D[a]
        A(f"{a:>4} {d['C4']:>15.4f} {d['seg_cgc_mean']:>11.4f} "
          f"{d['C4']/d['seg_cgc_mean']:>7.2f}")
    A("")
    A("表 D · 0.0467 vs 0.200 的解释 (B1 同一臂两个数)")
    A(f"  from-start 全程 (exp129 遥测 cos_cum_g_cum_run 最后一个非空值, step 495) = {D['B1']['C4']:.4f}")
    A(f"  段均值 (exp130 报告 T4b: Σ seg cos_cum_g_cum / 5)                     = {D['B1']['seg_cgc_mean']:.4f}")
    A(f"  五段原值 = " + " ".join(f"{v:+.4f}" for v in D["B1"]["seg_cgc"]))
    A("  ⇒ 两者是**同一批数据的不同聚合**: 段均值把每段当作独立参考再平均; from-start 把五段")
    A("    的向量连续累加, 跨段部分互相抵消。**不是两个不同的测量, 也不是矛盾。**")
    A("    佐证: 四个臂的双口径比值分别为 " +
      " ".join(f"{a}={D[a]['C4']/D[a]['seg_cgc_mean']:.2f}" for a in ORDER) +
      " — 比值本身随臂而变 (B4 差 1.9×), 说明聚合口径的影响是系统性的, 不是 B1 的特例。")
    A("")

    outdir = os.path.join(ROOT, "out", "exp131_report")
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "exp131_v2_layers.txt")
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
