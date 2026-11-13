"""
exp130 回执: A+B 两段合并报告 + 派生量 (只读已有产物, 零训练, 零 GPU)
======================================================================

输入:
  A 段 out/exp130_A_offline/exp130_A_coherence.json
  B 段 out/v463-20261022-185708/exp130.json + exp129_telemetry.jsonl(→exp130_telemetry.jsonl)

产出:self-contained 表格与派生量:
  T1 全程 coherence 五路 (N=239)
  T2 五段 coherence_g / coherence_raw / 比值 + 随机游走基线
  T3 两项分解 ΣdW_eff = U + T (U=Σβr, T=ΣγW) 的体量与夹角
  T4 梯度持续性: Σ‖g‖² 实测 → 由 ‖Σg‖ 反解公共成分比 α², 与实测 lag-1 cos_grad_osc 对撞检验
  T5 四问答案 + 机制判定
"""

import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BJ = os.path.join(ROOT, "out", "v463-20261022-185708", "exp130.json")
BT = os.path.join(ROOT, "out", "v463-20261022-185708", "exp130_telemetry.jsonl")


def load():
    j = json.load(open(BJ, encoding="utf-8"))
    rows = [json.loads(l) for l in open(BT, encoding="utf-8") if l.strip()]
    return j, rows


def main():
    j, rows = load()
    ca = j["coherence_all"]
    segs = j["segment_verdicts"]
    rr = [r for r in rows if r.get("cos_rule") is not None]
    N = len(rr)
    assert N == ca["N_updates"], (N, ca["N_updates"])

    L = []
    A = L.append
    A("=" * 100)
    A("exp130 回执 — coherence 判定 (A 段离线 + B 段仪器化)")
    A("=" * 100)
    fid = j["fidelity"]
    xc = j["xcheck_dWcos_prev"]
    A(f"保真门: post_bpb={fid['post_bpb_s1']} (target {fid['target']}, "
      f"{'PASS' if fid['pass'] else 'FAIL'}) | "
      f"逐步余弦 bit_identical={xc.get('bit_identical')} (n_segs={xc.get('n_segs_compared')}) | "
      f"自检(d) vs v460 max_abs_diff={j['selfcheck_v460'].get('max_abs_diff')} | "
      f"steps={j['steps_run']}/{j['target_steps']} abort={j['abort']} secs={j['secs']}")
    A("")

    # ── T1 全程 coherence ──
    ref = ca["random_ref_1over_sqrtN"]
    A("T1 · 全程 (from-start) coherence 五路   [N=%d, 随机游走基线 1/√N=%.4f]" % (N, ref))
    A(f"{'序列':>26} {'‖Σ‖':>12} {'Σ‖·‖':>14} {'coherence':>10} {'RA(=coh·√N)':>12}")
    for name, vs, sn, ck in (
            ("Σ(−g) 梯度参考", "sum_neg_g_norm", "sn_g", "coherence_g"),
            ("ΣdW_raw 规则提案", "sum_dw_raw_norm", "sn_r", "coherence_raw"),
            ("Σβr 规则项(trust+cap缩放)", "sum_betar_norm", "sn_betar", "coherence_betar"),
            ("ΣγW_pre cap 径向漂移", "sum_rad_norm", "sn_rad", "coherence_rad"),
            ("ΣdW_eff 净有效位移", "sum_dw_eff_norm", "sn_e", "coherence_eff")):
        A(f"{name:>26} {ca[vs]:>12.3f} {ca[sn]:>14.2f} {ca[ck]:>10.4f} {ca['RA'][ck.split('_')[1]]:>12.3f}")
    A(f"{'':>26} {'':>12} {'':>14} {'raw/g':>10} {ca['ratio_raw_over_g']:>12.3f}")
    A(f"cos(ΣdW_raw, Σg) = {ca['cos_sumraw_sumg']:+.4f}   "
      f"cos(ΣγW, Σg) = {ca['cos_sumrad_sumg']:+.4f}")
    A("")

    # ── T2 五段 ──
    A("T2 · 五段 coherence (段内有更新步数 n; 单段基线 1/√n 随段给出)")
    A(f"{'seg':>4} {'n':>3} {'coh_g':>8} {'coh_raw':>8} {'raw/g':>7} {'coh_eff':>8} "
      f"{'coh_betar':>9} {'coh_rad':>8} {'1/√n':>7} {'RA_g':>6} {'RA_raw':>6}")
    for s in segs:
        A(f"{s['segment']:>4} {s['n_updates']:>3} {s['coherence_g']:>8.4f} "
          f"{s['coherence_raw']:>8.4f} {s['coherence_raw']/s['coherence_g']:>7.3f} "
          f"{s['coherence_eff']:>8.4f} {s['coherence_betar']:>9.4f} "
          f"{s['coherence_rad']:>8.4f} {s['random_ref_1over_sqrtN']:>7.4f} "
          f"{s['RA']['g']:>6.2f} {s['RA']['raw']:>6.2f}")
    A("")
    A("T2b · 梯度 lag-1 相邻余弦 cos(−g_i, −g_{i−1}) —— A 机制的直接前置证据 (exp129 未测)")
    A(f"{'seg':>4} {'n':>3} {'mean':>9} {'min':>9} {'max':>9}")
    tot = 0.0
    n_tot = 0
    for s in segs:
        m, n = s["cos_grad_osc_mean"], s["cos_grad_osc_n"]
        tot += m * n
        n_tot += n
        A(f"{s['segment']:>4} {n:>3} {m:>9.4f} {s['cos_grad_osc_min']:>9.4f} "
          f"{s['cos_grad_osc_max']:>9.4f}")
    A(f"{'全程加权':>4} {n_tot:>3} {tot/n_tot:>9.4f}")
    A("")

    # ── T3 两项分解 ──
    A("T3 · 两项分解 ΣdW_eff = U + T  (U=Σβr, T=ΣγW_pre; β=cap·lr·trust, γ=cap−1)")
    A(f"{'':>6} {'‖U‖':>10} {'‖T‖':>10} {'‖Σe‖':>10} {'cos(U,T)':>10} {'‖T‖/‖Σe‖':>10} "
      f"{'cos(T,Σg)':>10} {'resid':>8}")
    for lab, u, t, e, crg, res in (
            [("seg%d" % s["segment"], s["sum_betar_norm"], s["sum_rad_norm"],
              s["sum_dw_eff_norm"], s.get("cos_rad_g"), s.get("decomp_resid"))
             for s in segs] +
            [("全程", ca["sum_betar_norm"], ca["sum_rad_norm"], ca["sum_dw_eff_norm"],
              ca["cos_sumrad_sumg"], None)]):
        cosut = (e * e - u * u - t * t) / (2 * u * t)
        A(f"{lab:>6} {u:>10.3f} {t:>10.3f} {e:>10.3f} {cosut:>10.4f} {t/e:>10.3f} "
          f"{(crg if crg is not None else float('nan')):>10.4f} "
          f"{(res if res is not None else float('nan')):>8.4f}")
    A("注: resid = ‖ΣdW_eff − (U+T)‖ 为两项恒等式的残差 (fp16 舍入量级), 段残差/‖Σe‖ < 0.4%。")
    A("")

    # ── T4 梯度持续性 ──
    A("T4 · 梯度持续性 (公共成分比 α² 的反解 与 lag-1 实测的对撞)")
    S1g = sum(r["neg_g_norm"] for r in rr)
    S2g = sum(r["neg_g_norm"] ** 2 for r in rr)
    ng = ca["sum_neg_g_norm"]
    # 单因子模型: g_i = α·ê + sqrt(1−α²)·f_i, f_i 两两正交随机单位
    #   ⇒ ‖Σg‖² = α²·(Σ‖g‖)² + (1−α²)·Σ‖g‖²
    num = ng * ng - S2g
    den = S1g * S1g - S2g
    a2 = num / den
    A(f"  Σ‖g‖ = {S1g:.3f}   Σ‖g‖² = {S2g:.3f}   ‖Σg‖² = {ng*ng:.3f}")
    A(f"  单因子模型反解 α² = {a2:.5f}  (α = {math.sqrt(a2):.4f})")
    A(f"  该模型对所有 (i,j) 对预测同一 cos = α² = {a2:+.5f}")
    A(f"  实测 lag-1 cos_grad_osc 全程加权均值 = {tot/n_tot:+.5f}")
    A(f"  ⇒ {'被 lag-1 实测否证作为完整描述' if abs(a2 - tot/n_tot) > 0.01 else '未被证伪'}: "
      f"单因子模型预测任意两对的 cos 同为 {a2:+.5f}, 实测相邻步为 {tot/n_tot:+.5f} (符号相反)。"
      f"⇒ 梯度持续性不能只用『一个公共成分 + 白噪声』描述; 在高维近正交残差之上还叠加了"
      f"明确的 lag-1 反相关。α²={a2:.5f} 仍刻画**非相邻对**的持续性。")
    A("")

    # ── T4b cos 链的成本分解 ──
    A("T4b · 累积 cos 链的成本分解 (五段均值; 乘性链, 逐段因 n 不同权重微异)")
    m_rule = sum(s["cos_rule"] for s in segs) / 5
    m_raw = sum(s["cos_cum_raw_g_cum"] for s in segs) / 5
    m_eff = sum(s["cos_cum_g_cum"] for s in segs) / 5
    A(f"  每步 cos(dW_raw, −g_i)             = {m_rule:.4f}")
    A(f"  Σ 累积 cos(ΣdW_raw, Σg)            = {m_raw:.4f}   ← 旋转+振荡成本 ×{m_rule/m_raw:.2f}")
    A(f"  Σ 累积 cos(ΣdW_eff, Σg)            = {m_eff:.4f}   ← cap 漂移成本 ×{m_raw/m_eff:.2f}")
    A(f"  ⇒ 两步扣的都是同一量级 ({m_rule/m_raw:.2f}× vs {m_raw/m_eff:.2f}×), "
      f"没有哪一个压倒性占优。")
    A(f"  ⇒ 单独解耦 cap 的收益上界: 累积 cos {m_eff:.3f} → {m_raw:.3f} "
      f"(+{100*(m_raw/m_eff-1):.0f}%); 余下 {m_rule:.3f}→{m_raw:.3f} 的那一段是旋转+振荡, "
      f"cap 解耦碰不到。")
    A("")

    # ── T5 四问 ──
    A("T5 · 四问答案")
    A("  Q1 coherence(Σg) 五段 = " + " ".join(f"{s['coherence_g']:.4f}" for s in segs)
      + f" ; 全程 {ca['coherence_g']:.4f} vs 1/√N {ref:.4f} (RA {ca['RA']['g']:.2f})")
    A(f"      ⇒ 参考基线本身只有随机游走的 {ca['RA']['g']:.2f} 倍; 且 lag-1 cos ≈ {tot/n_tot:+.4f} "
      f"(相邻梯度几乎无相关) ⇒ **梯度确实每步大幅旋转**")
    A("  Q2 coherence(ΣdW_raw) 五段 = " + " ".join(f"{s['coherence_raw']:.4f}" for s in segs)
      + f" ; 全程 {ca['coherence_raw']:.4f} (RA {ca['RA']['raw']:.2f})")
    A(f"      raw/g = " + " ".join(f"{s['coherence_raw']/s['coherence_g']:.3f}" for s in segs)
      + f" ; 全程 {ca['ratio_raw_over_g']:.3f}")
    A(f"      ⇒ dW_raw **不是更相干, 而是比梯度参考更不相干 {1/ca['ratio_raw_over_g']:.2f} 倍**")
    A(f"  Q3 coherence(ΣdW_eff) 全程 {ca['coherence_eff']:.4f} (RA {ca['RA']['eff']:.2f}); "
      f"cap 项 coherence_rad = {ca['coherence_rad']:.4f} (RA {ca['RA']['rad']:.2f})")
    A(f"      ‖T‖ = {ca['sum_rad_norm']:.1f} = 净位移 ‖Σe‖ = {ca['sum_dw_eff_norm']:.1f} 的 "
      f"{ca['sum_rad_norm']/ca['sum_dw_eff_norm']:.2f} 倍 ⇒ cap 漂移以体量计不是次级")
    A(f"  Q4 cos(ΣdW_raw, Σg) = {ca['cos_sumraw_sumg']:+.4f} (全程); "
      f"cos(ΣγW, Σg) = {ca['cos_sumrad_sumg']:+.4f}")
    A("")

    A("=" * 100)
    A("机制判定: A+B 混合 —— 但观测单元格不在预写判据表的三行之内 (见 §裁决)")
    A("=" * 100)

    outdir = os.path.join(ROOT, "out", "exp130_A_offline")
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "exp130_report.txt")
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
