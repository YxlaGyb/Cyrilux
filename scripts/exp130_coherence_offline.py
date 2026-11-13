"""
exp130 A 段 (离线): coherence 判定 — 四问尽可能从 exp129 v460 既有遥测精确算出
================================================================================

立项 (exp129 审查裁决, 2026-10-22):
  exp129 把 "G 低" 归到四条实测成因链, 但其中的 "振荡" 只是**描述** (相邻更新反相关),
  不是**机制**。审查裁决的数学事实: 若相邻更新只是随机反相关 (每步方向的意义独立抖动),
  累积余弦应**升高**而非降低 — cos(ΣdW, Σu) = a·sqrt(N)/sqrt(a²N+b²) → 1。实测从每步
  0.78 掉到累积 0.14, 不可能由随机抖动解释, 必须有**跨步相关的系统分量**。两种候选:

    A. 梯度旋转: h 每窗不同 → u_i = −g_i 方向大幅变 → Σu 自身很小 → 参考本身塌。
    B. 系统漂移: dW 有跨步同向的非梯度分量 (cap 收缩 / 常数偏置 / rms_norm 方向偏差)。

  判定量 coherence(x) = ‖Σx‖ / Σ‖x‖  ( = ‖Σx‖ / (N·mean‖x‖) ):
    随机游走 ≈ 1/sqrt(N) ; 系统漂移 ≫ 1/sqrt(N)。

可用性判定 (读 scripts/exp129_direction_stability.py 后得到, 见 §判别):
  落盘的向量和范数只有两个: sum_neg_g_norm = ‖Σ_段(−g)‖ ; sum_dw_eff_norm = ‖Σ_段dW_eff‖。
  **‖Σ_段 dW_raw‖ 没有落盘** — exp129 只在内存累加 sr 并算 _cos(sr, sg), 从未写出其范数。
  cos 抹掉模长, 且无 running 范数量可与之联立 ⇒ ‖ΣdW_raw‖ **不可离线恢复**。
  ⇒ Q1/Q4 精确可算; Q2/Q3 需要 ‖ΣdW_raw‖ ⇒ 需 B 段 (仪器化重跑)。

  本件的补偿策略 (全部标注性质, 不把派生量冒充测量):
   (a) coherence(ΣdW_eff) 精确可算, 作为 coherence(ΣdW_raw) 的替代观测 (含 cap 贡献)。
   (b) 用逐步解析恒等式给 cap 贡献定界, 得到 non-cap 部分向量和 ‖U‖ 的**严格区间**:
         dW_eff_i = β_i·dW_raw_i + γ_i·W_pre_i,  β_i = c_i·lr·t_i, γ_i = c_i − 1
         (trust 是标量缩放, 不旋转; cap10 是作用于整个 W_lm 的全局标量 c_i)
       ⇒ ΣdW_eff = U + T,  U = Σβ_i r_i,  T = Σγ_i W_pre_i
       ⇒ ‖T‖ ≤ Σ|γ_i|·‖W_pre_i‖ = Tmax ⇒ ‖U‖ ∈ [max(0,‖Σe‖−Tmax), ‖Σe‖+Tmax]
   (c) ‖W_pre_i‖ 可逐步反解 (不必依赖上一行):
         e − β·r = γ·W_pre ⇒ ‖W_pre‖ = ‖e−βr‖ / |γ|,
         ‖e−βr‖² = ‖e‖² − 2β<e,r> + β²‖r‖²,  <e,r> = cap_rotation·‖e‖·‖r‖。
       并用遥测上一行的 W_lm_norm 交叉验证 —— 两条独立路径应一致; 若不符则两项模型
       被证伪, 本件所有派生量作废。

铁律: 只读已有目录; 零训练; 零 GPU; crates/ 零改动; 权重不落盘; 不 kill 任何进程。
"""

import hashlib
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "out", "v460-20261021-220406")
JSON_PATH = os.path.join(SRC_DIR, "exp129.json")
TEL_PATH = os.path.join(SRC_DIR, "exp129_telemetry.jsonl")

LR = 0.05
N_SEG = 5


def sha256(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load():
    with open(JSON_PATH, encoding="utf-8") as f:
        j = json.load(f)
    rows = []
    with open(TEL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return j, rows


def main():
    j, rows = load()
    segs = {s["segment"]: s for s in j["segment_verdicts"]}

    # ── 逐步派生: 两term模型 e = β·r + γ·W_pre 的量 ──
    # W_prev_norm[row index]: 上一行的 W_lm_norm (早退行保留上一次真更新的后验值,
    # 恰等于本更新步的 W_pre 范数; 见 analysis note)
    perstep = []
    for k, r in enumerate(rows):
        d = {"step": r["step"]}
        instr = r.get("cos_rule") is not None
        d["instr"] = instr
        if not instr:
            perstep.append(d)
            continue
        nr = r.get("dW_raw_norm")
        ne = r.get("dW_eff_norm")
        t = r.get("trust_scale")
        c = r.get("wlm_cap_scale")
        cr = r.get("cap_rotation")
        d["dW_raw_norm"] = nr
        d["dW_eff_norm"] = ne
        d["trust_scale"] = t
        d["cap_scale"] = c
        if None in (nr, ne, t, c, cr):
            d["model_ok"] = False
            perstep.append(d)
            continue
        beta = c * LR * t
        gamma = c - 1.0
        d["beta"] = beta
        d["gamma"] = gamma
        d["dtheo"] = LR * t * nr                      # = ‖lr·t·r‖ = β/c·‖r‖
        d["beta_nr"] = beta * nr                       # = c · dtheo
        if abs(gamma) > 1e-9:
            # ‖e − βr‖² = ‖e‖² − 2β<e,r> + β²‖r‖²,  <e,r> = cr·‖e‖·‖r‖
            er = cr * ne * nr
            res2 = ne * ne - 2.0 * beta * er + (beta * nr) ** 2
            d["norm_e_minus_betar"] = math.sqrt(res2) if res2 > 0 else None
            d["Wpre_norm_derived"] = (math.sqrt(res2) / abs(gamma)) if res2 > 0 else None
            # <W,r>/‖W‖‖r‖ —— 由 <e,r> = β‖r‖² + γ<W,r> 解出
            wr = (er - beta * nr * nr) / gamma
            denom = d["Wpre_norm_derived"] * nr if d["Wpre_norm_derived"] else 0.0
            d["cos_W_r"] = (wr / denom) if denom > 0 else None
            d["radial_mag"] = abs(gamma) * (d["Wpre_norm_derived"] or 0.0)
        else:
            d["norm_e_minus_betar"] = None
            d["Wpre_norm_derived"] = None
            d["cos_W_r"] = None
            d["radial_mag"] = 0.0
        # 上一行 W_lm_norm 作为 W_pre 范数的独立交叉验证
        d["Wpre_norm_prevrow"] = rows[k - 1]["W_lm_norm"] if k > 0 else None
        d["model_ok"] = True
        perstep.append(d)

    # ── 逐段 coherence ──
    out_segs = []
    for k in range(1, N_SEG + 1):
        s = segs[k]
        lo, hi = (k - 1) * 100, k * 100
        ps = [p for p, r in zip(perstep, rows) if lo < r["step"] <= hi and p["instr"]]
        N = len(ps)
        assert N == s["n_updates"], f"seg{k} N mismatch {N} vs {s['n_updates']}"

        # ── 逐步范数求和 (从原始遥测行取) ──
        rr = [r for r in rows if lo < r["step"] <= hi and r.get("cos_rule") is not None]
        S_g = sum(r["neg_g_norm"] for r in rr)
        S_r = sum(r["dW_raw_norm"] for r in rr)
        S_e = sum(r["dW_eff_norm"] for r in rr)
        nSg = s["sum_neg_g_norm"]      # ‖Σ(−g)‖
        nSe = s["sum_dw_eff_norm"]     # ‖ΣdW_eff‖

        coh_g = nSg / S_g
        coh_e = nSe / S_e
        ref = 1.0 / math.sqrt(N)

        # cap 径向预算 (对派生 step 字典求和, 跳过模型失效步)
        def _sum(key):
            return sum(v for v in (p.get(key) for p in ps) if v is not None)

        Tmax = _sum("radial_mag")
        S_beta_r = sum(( d["beta_nr"] if d.get("beta_nr") is not None
                         else (d.get("dtheo") or 0.0) * (d.get("cap_scale") or 1.0))
                       for d in ps)
        U_lo = max(0.0, nSe - Tmax)
        U_hi = nSe + Tmax
        coh_U_lo = U_lo / S_beta_r if S_beta_r > 0 else None
        coh_U_hi = U_hi / S_beta_r if S_beta_r > 0 else None

        # W_pre 交叉验证
        dev = [abs(d["Wpre_norm_derived"] - d["Wpre_norm_prevrow"]) / d["Wpre_norm_prevrow"]
               for d in ps
               if d.get("Wpre_norm_derived") and d.get("Wpre_norm_prevrow")]
        # cos_W_r 合法性 (两term模型的可证伪检验)
        cw = [d["cos_W_r"] for d in ps if d.get("cos_W_r") is not None]
        bad = [v for v in cw if v < -1.0005 or v > 1.0005]

        n_cap_fired = sum(1 for d in ps if d.get("cap_scale") is not None
                          and d["cap_scale"] < 1.0 - 1e-9)
        oneminus = [1.0 - d["cap_scale"] for d in ps
                    if d.get("cap_scale") is not None]

        # ── 段级协变量 (用于 "趋势" 列: 找与 cos_osc 衰减共变的量) ──
        def _m(key):
            vals = [r[key] for r in rr if r.get(key) is not None]
            return (sum(vals) / len(vals)) if vals else None

        n_trust_bound = sum(1 for r in rr if r.get("trust_scale") is not None
                            and r["trust_scale"] < 1.0 - 1e-9)
        w_end = rr[-1]["W_lm_norm"] if rr else None
        # 有效步长相对规模: mean(‖ΔW‖ / ‖W‖)
        steps_rel = [r["dW_eff_norm"] / rows[rows.index(r) - 1]["W_lm_norm"]
                     for r in rr if rows.index(r) > 0 and r.get("dW_eff_norm")]

        out_segs.append({
            "segment": k, "N": N,
            "cos_osc_mean": s["cos_osc"], "cos_osc_max": s["cos_osc_max"],
            "mean_trust_scale": _m("trust_scale"),
            "n_trust_bound": n_trust_bound,
            "mean_cap_scale": _m("wlm_cap_scale"),
            "mean_dW_raw_norm": _m("dW_raw_norm"),
            "mean_dW_eff_norm": _m("dW_eff_norm"),
            "mean_neg_g_norm": _m("neg_g_norm"),
            "mean_mask": _m("mask_mean"),
            "mean_step_rel": (sum(steps_rel) / len(steps_rel)) if steps_rel else None,
            "W_lm_norm_end": w_end,
            "sum_norm_g": S_g, "norm_sum_g": nSg,
            "sum_norm_eff": S_e, "norm_sum_eff": nSe,
            "sum_norm_raw": S_r,
            "coherence_g": coh_g, "coherence_eff": coh_e,
            "random_ref_1over_sqrtN": ref,
            "RA_g": coh_g / ref, "RA_eff": coh_e / ref,     # 相对随机游走的放大倍率
            "ratio_eff_over_g": coh_e / coh_g,
            "cos_cum_g_cum": s["cos_cum_g_cum"],            # cos(ΣdW_eff, Σg)
            "cos_cum_raw_g_cum": s["cos_cum_raw_g_cum"],    # cos(ΣdW_raw, Σg)  ← Q4 精确值
            "cos_rule_mean": s["cos_rule"], "cos_eff_mean": s["cos_eff"],
            # cap 径向预算 (严格上界 Tmax) 与 non-cap 部分 ‖U‖ 的严格区间
            "Tmax_cap_radial": Tmax,
            "sum_beta_r": S_beta_r,
            "norm_U_lo": U_lo, "norm_U_hi": U_hi,
            "coherence_U_lo": coh_U_lo, "coherence_U_hi": coh_U_hi,
            "n_cap_fired": n_cap_fired,
            "mean_1minus_cap": (sum(oneminus) / len(oneminus)) if oneminus else None,
            "max_1minus_cap": max(oneminus) if oneminus else None,
            # 交叉验证
            "Wpre_xcheck_n": len(dev),
            "Wpre_xcheck_maxrel": max(dev) if dev else None,
            "Wpre_xcheck_meanrel": (sum(dev) / len(dev)) if dev else None,
            "cos_W_r_n": len(cw),
            "cos_W_r_outofrange": len(bad),
            "cos_W_r_mean": (sum(cw) / len(cw)) if cw else None,
        })

    res = {
        "exp": "exp130", "stage": "A_offline",
        "source_json": JSON_PATH, "source_json_sha256": sha256(JSON_PATH),
        "source_tel": TEL_PATH, "source_tel_sha256": sha256(TEL_PATH),
        "note": ("所有 coherence 为分段精确的向量和范数 / 逐步范数之和。"
                 "‖ΣdW_raw‖ 未落盘 ⇒ coherence_raw 不可离线恢复;"
                 "coherence_U_* 为 β·r 部分的严格区间 (非点估计)。"),
        "segments": out_segs,
    }

    outdir = os.path.join(ROOT, "out", "exp130_A_offline")
    os.makedirs(outdir, exist_ok=True)
    p_out = os.path.join(outdir, "exp130_A_coherence.json")
    with open(p_out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    # ── 首句 ──
    L = []
    L.append("=" * 96)
    L.append("exp130 A 段 (离线) — coherence 判定")
    L.append("=" * 96)
    L.append(f"源: {JSON_PATH}")
    L.append(f"    sha256 {res['source_json_sha256']}")
    L.append(f"源: {TEL_PATH}")
    L.append(f"    sha256 {res['source_tel_sha256']}")
    L.append("")
    L.append("【可用性裁定】‖ΣdW_raw‖ 未落盘 ⇒ coherence(ΣdW_raw) 不可离线恢复 (详见文件头 §判别)。")
    L.append("")
    L.append("表 1 · coherence 五段 (精确; RA = 相对随机游走 1/√N 的放大倍率)")
    L.append(f"{'seg':>4} {'N':>3} {'‖Σg‖':>8} {'Σ‖g‖':>9} {'coh_g':>7} {'1/√N':>6} {'RA_g':>6} "
             f"{'‖Σe‖':>8} {'Σ‖e‖':>9} {'coh_eff':>8} {'RA_eff':>6} {'eff/g':>6}")
    for s in out_segs:
        L.append(f"{s['segment']:>4} {s['N']:>3} {s['norm_sum_g']:>8.3f} {s['sum_norm_g']:>9.2f} "
                 f"{s['coherence_g']:>7.4f} {s['random_ref_1over_sqrtN']:>6.4f} {s['RA_g']:>6.2f} "
                 f"{s['norm_sum_eff']:>8.3f} {s['sum_norm_eff']:>9.2f} {s['coherence_eff']:>8.4f} "
                 f"{s['RA_eff']:>6.2f} {s['ratio_eff_over_g']:>6.2f}")
    L.append("")
    L.append("表 2 · cap 径向预算 与 non-cap 部分 ‖U‖ 严格区间 (derived, 非点估计)")
    L.append(f"{'seg':>4} {'n_cap':>5} {'mean(1-c)':>9} {'max(1-c)':>9} {'Σ|γ|‖W‖':>9} "
             f"{'‖Σe‖':>8} {'‖U‖lo':>8} {'‖U‖hi':>8} {'Σβ‖r‖':>9} {'coh_U_lo':>9} {'coh_U_hi':>9}")
    for s in out_segs:
        L.append(f"{s['segment']:>4} {s['n_cap_fired']:>5} {s['mean_1minus_cap']:>9.5f} "
                 f"{s['max_1minus_cap']:>9.5f} {s['Tmax_cap_radial']:>9.2f} "
                 f"{s['norm_sum_eff']:>8.3f} {s['norm_U_lo']:>8.3f} {s['norm_U_hi']:>8.3f} "
                 f"{s['sum_beta_r']:>9.2f} {s['coherence_U_lo']:>9.4f} {s['coherence_U_hi']:>9.4f}")
    L.append("")
    L.append("表 3 · 四问答案状态")
    L.append(f"{'seg':>4} {'Q1 coh_g':>10} {'Q1 判(≈1/√N?)':>15} {'Q4 cos(Σr,Σg)':>14} "
             f"{'cos(Σe,Σg)':>12} {'Q2/Q3':>10}")
    for s in out_segs:
        q1 = "≈1/√N ⇒旋转" if s["RA_g"] < 1.6 else ("≫1/√N ⇒稳定" if s["RA_g"] > 3.0 else "中间待定")
        L.append(f"{s['segment']:>4} {s['coherence_g']:>10.4f} {q1:>15} "
                 f"{s['cos_cum_raw_g_cum']:>14.4f} {s['cos_cum_g_cum']:>12.4f} "
                 f"{'需B段':>10}")
    L.append("")
    L.append("表 4 · 两term模型 (e = βr + γW) 交叉验证")
    L.append(f"{'seg':>4} {'n':>3} {'Wpre 相对偏差 max':>18} {'mean':>10} "
             f"{'cos(W,r) mean':>14} {'越界个数':>9}")
    for s in out_segs:
        L.append(f"{s['segment']:>4} {s['Wpre_xcheck_n']:>3} {s['Wpre_xcheck_maxrel']:>18.5f} "
                 f"{s['Wpre_xcheck_meanrel']:>10.5f} {s['cos_W_r_mean']:>14.4f} "
                 f"{s['cos_W_r_outofrange']:>9}")
    L.append("")
    L.append("表 5 · 段级协变量趋势 (与 cos_osc 衰减共变量的候选; εξ全部均值 over 段内更新步)")
    L.append(f"{'seg':>4} {'cos_osc':>8} {'Δ vs 上段':>10} {'cos_osc_max':>12} {'trust':>7} "
             f"{'bound n':>8} {'cap':>7} {'‖r̄‖':>9} {'‖ē‖':>7} {'‖ḡ‖':>7} "
             f"{'‖ΔW‖/‖W‖':>10} {'‖W‖段末':>8}")
    prev = None
    for s in out_segs:
        d = "" if prev is None else f"{s['cos_osc_mean'] - prev:+.4f}"
        prev = s["cos_osc_mean"]
        L.append(f"{s['segment']:>4} {s['cos_osc_mean']:>8.4f} {d:>10} {s['cos_osc_max']:>12.4f} "
                 f"{s['mean_trust_scale']:>7.4f} {s['n_trust_bound']:>8} {s['mean_cap_scale']:>7.5f} "
                 f"{s['mean_dW_raw_norm']:>9.2f} {s['mean_dW_eff_norm']:>7.3f} "
                 f"{s['mean_neg_g_norm']:>7.3f} {s['mean_step_rel']:>10.5f} "
                 f"{s['W_lm_norm_end']:>8.3f}")
    L.append("")
    L.append(f"写出 {p_out}")
    txt = "\n".join(L)
    p_txt = os.path.join(outdir, "exp130_A_firstline.txt")
    with open(p_txt, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    main()
