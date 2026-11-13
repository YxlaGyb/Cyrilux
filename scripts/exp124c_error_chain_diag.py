"""
exp124c: 误差链落点诊断 (gain/dz4_sig 压平假说定量) — 500 步遥测实验
====================================================================

立项依据 (2026-10-18 审查裁决, 聊天记录): exp124 执行层通过, 结论层推翻 —
不是"eps_b 幅度假说被证伪", 是**剂量/落点失败**:
  (1) amp = 2r/(1+r) 上限 2 不可达 → 融合 eps_b + 0.5·amp·eps_lm_b 中 lm 项
      权重 0.5·amp < 1 恒成立 → 混合按构造永不可翻转为 lm 主导 (机制所需
      regime 不可达; 实测个体最大 amp 1.7197 → lm 权重 0.860 仍未过线);
  (2) readout.py:223-235 在 amp 之前已有 gain (227-229 软饱和) + dz4_sig
      (223-224 max 归一) 两层 + 242 行 rms_norm — eps_proj_raw 分布平坦是
      空结果的第一性原因假设 (exp124 报告 §四已观测未接论)。
裁决: 下一步不是 exp124b (暂缓), 是 exp124c 落点诊断; 最便宜的诊断先做 —
测 eps_proj_raw 的分布, 确认是否真的被 gain/dz4_sig 压平。

─── 实验设计 (对照 = exp124 去掉 amp) ───

- 训练规则: E3 replica 打 ReadoutMixin._update_lm_head (bias 冻结等, 与
  exp124 同一函数, 见依赖披露); FeedforwardMixin._update_W35 **保持部署版
  不动** — exp124 selfcheck 已证 amp≡1 复刻与部署版逐位相同, 故"E3 + 部署
  W35"即 amp 关断的 exp124 基线, 本跑是该基线的纯遥测版。
- 唯一新增: ReadoutMixin._build_lm_signal 包层遥测 (零训练改动, 只读
  LmSignal 字段 + 复刻 223-229 两个中间量)。
- 同检查点 (md5 硬门) / 同数据 / 同种子 (0) / 同协议 (首步 force_perception,
  P3-b 行为门路由, sync_gate + maybe_prune, i%100 dump, S0 基线 dump,
  pre/post bpb 锚点) — 与 exp124 (v419) 唯一差别是 amp 不乘入。

─── 遥测对象 (六级 + 两个派生) ───

  e0      = ‖eps_total‖ 逐位范数        [readout.py:227 gain 的输入]
  gain    = e0/(0.5·err_ref+0.5·e0)      [复刻 227-229, ∈(0,2)]
  dz4_sig = ‖z4[1:]−z4[:-1]‖/(max+1e-3)  [复刻 223-224, ∈(0,1)]
  p       = m/(dz4_sig·gain)             [投影 RMS, 除法反解, 有效位]
  m       = ‖eps_proj_raw‖ 逐位 RMS      [amp 闸门输入, exp124 同定义]
  r       = m/(m.mean()+1e-3)            [误差倍数 = amp 的自变量]
  amp     = m/(0.5·ref+0.5·m)            [反推闸门输出, 与 v419 互为旁证]
  pg      = p·gain                       [归因用中间级]
  每级每步: n / mean / p50 / p90 / p95 / p99 / min / max / p95/mean / max/mean。
  **两套总体** (v2 修订, 冒烟发现掩码位污染): 归因级联 (e0/p/gain/dz4_sig/
  pg/m) 用**有效位总体** (dz4_sig·gain>1e-6, 即误差非零的信号位 — 掩码位
  m=0 会压低均值、虚高 max/mean, 污染归因); m/r/amp 另记**全位统计**
  (与 v419 逐步 p95 同口径, 含掩码零) 用于对拍。
  另记: frac_valid (m>0), frac_amp_gt_1, frac_amp_gt_1_5 (有效位, 回答
  "高剂量位置到底多稀少" — v419 遥测答不了的一项)。

─── 判读规则 (预写, 跑前固化) ───

记 S(x) = x 的 max/mean 跨感知步均值 (p95/mean 为辅助口径)。
  1. overall = S(m)/S(e0)。overall < 0.5 → **确认上游链压平**误差能量;
     overall ≥ 0.5 → 未压平 (或压平不显著), amp 公式为唯一瓶颈。
  2. 归因 (边际): gain_marginal = S(p) − S(pg); dz4_marginal = S(pg) − S(m)。
     正差大者 = 主导压平门; 两者皆负 → 压平不来自乘法门, 而在软饱和映射
     (gain/dz4_sig 对其自变量的凹/截断) 或投影本身。
  3. 下一步分叉 (对应审查裁决):
     - 确认压平 → 修 gain/dz4_sig 本身 (更大变更面, 单独立项, 零改动纪律
       下待批), 不再调 amp;
     - 未压平 → 换 amp 公式 (固定参考或更陡函数, 让大误差真正产生
       amp ≫ 1), 记为 exp124d。
  4. 旁证: amp 反推的逐步 p95 均值 / 个体 max 应与 v419 (1.2381 / 1.7197)
     在早期步接近 (同起点; 轨迹随 amp 有无会分叉, 全跑均值允许漂移)。

─── 偏离披露 ───

- 依赖披露: E3 replica (_update_lm_head_e3_biasfrozen) / eval 仪器
  (eval_bpb_fresh) / word_hits_of / PNAMES / param_norms 从
  scripts/exp124_epsb_amp.py import (运行时记录其 sha256); exp124c 自身不
  复刻任何训练规则 — 若该文件被改, 本跑基线随之变, 故哈希入账。
- 后端 ref (crates/graphs set_backend 公开开关; capture 路径 step 19
  device assert 未归因, 且回放帧不执行 python 侧遥测 — exp124 实证)。
- _migrate_expand=False (topology.compact 漏同步 migrate 通道, crates 既有
  缺口, 开 migrate 的训练 death 回合后崩于 readout.py:84 — 已单独立项)。
- 陈旧读修正: 回声步显式清空遥测 (exp124 的回声步读到上一感知步陈旧 amp,
  全量口径混入 168 个陈旧样本; 本跑只统计感知步, n 应为 332)。遥测 stash 用
  模块级 global 不挂 net — net.save 的 _extra_tensors 会把 dict 属性的张量
  序列化进快照, dump 时 load 新实例即崩 (冒烟 step 100 实证, 详见代码注释)。
- 统计口径: 所有分位数/极值在 .float() 上计算 (仅统计路径, 不触训练张量,
  exp121 dt_lut 同款先例); fp16 训练全程不变。
- stdout 通道故障 (宿主 PowerShell 有文件副作用无回显): 长跑落盘
  stdout/stderr + 退出码, 非交互前台; 运行不 kill (纪律 13)。

铁律: crates/ 零改动 (仅进程内类方法 monkeypatch); 不动 v244/迁移件;
全 fp16 (无 clamp 数值); 权重不落盘; run_file 自增新目录。
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

import torch

from exp115_say import N_EVAL_WIN, S_MAX, _sample_dump, build_gauge
from exp124_epsb_amp import (CKPT, CKPT_MD5, PNAMES, TARGET_STEPS, _getp,
                             _update_lm_head_e3_biasfrozen, eval_bpb_fresh,
                             param_norms, word_hits_of)
from exp116_z4_probe import _md5

from crates.cli.utils import resolve_path, run_file
from crates.graphs import set_backend
set_backend("ref")
from crates.hybrid.learning.readout import ReadoutMixin
from crates.model import CyreneModel, HybridPCNet
from crates.utils.codec import ByteDataset

torch.set_grad_enabled(False)

LEVELS = ["e0", "p", "gain", "dz4_sig", "pg", "m", "r", "amp"]
CASCADE = ["e0", "p", "pg", "m"]

# 遥测 stash 用模块级 global, 不挂 net: model_cyrene.py:785-791 _extra_tensors 会
# 递归把 net.__dict__ 里 dict 型属性的张量序列化进 net.save 快照 (键 "_diag_t.e0"),
# load 到新实例时 getattr(net, "_diag_t") 不存在即崩 (exp124c 冒烟 step 100 实证);
# exp124 的 _tel_* 为裸张量 (无点号键, load 走 setattr) 故未崩 — 同一坑的两面。
_DIAG = {"t": None}
_call_counts = {"perception": 0, "echo": 0, "free_run": 0}

_orig_build_lm_signal = ReadoutMixin._build_lm_signal


def _diag_build_lm_signal(self, ctx):
    """包层遥测: 调原 _build_lm_signal, 感知步挂误差链张量, 回声/free_run 显式清空。"""
    lm = _orig_build_lm_signal(self, ctx)
    if lm is None or ctx.free_run:
        _call_counts["free_run"] += 1
        _DIAG["t"] = None
        return lm
    if ctx.echo_world_frozen:
        _call_counts["echo"] += 1
        _DIAG["t"] = None
        return lm
    _call_counts["perception"] += 1
    net = self.net
    e0 = lm.eps_total.norm(dim=-1)
    err_ref = e0.mean() + 1e-3
    gain = e0 / (0.5 * err_ref + 0.5 * e0)
    z4 = net._z4
    dz4 = (z4[:, 1:] - z4[:, :-1]).norm(dim=-1)
    dz4_sig = dz4 / (dz4.max() + 1e-3)
    m = lm.eps_proj_raw.square().mean(dim=-1).sqrt()
    prod = dz4_sig * gain
    valid = prod > 1e-6
    p = torch.where(valid, m / torch.where(valid, prod, torch.ones_like(prod)),
                    torch.zeros_like(m))
    amp = m / (0.5 * (m.mean() + 1e-3) + 0.5 * m)
    r = m / (m.mean() + 1e-3)
    _DIAG["t"] = {
        "e0": e0.detach(), "p": p.detach(), "gain": gain.detach(),
        "dz4_sig": dz4_sig.detach(), "pg": (p * gain).detach(), "m": m.detach(),
        "r": r.detach(), "amp": amp.detach(), "valid": valid.detach(),
    }
    return lm


def level_stats(t, mask=None):
    af = t.flatten().float()
    if mask is not None:
        mf = mask.flatten()
        af = af[mf]
    n = af.numel()
    if n == 0:
        return {"n": 0, "mean": None, "max": None, "p95_over_mean": None, "max_over_mean": None}
    mean = float(af.mean())
    qs = torch.quantile(af, torch.tensor([0.5, 0.9, 0.95, 0.99], device=af.device))
    p50, p90, p95, p99 = (float(q) for q in qs)
    mx, mn = float(af.max()), float(af.min())
    return {"n": n, "mean": mean, "p50": p50, "p90": p90, "p95": p95, "p99": p99,
            "min": mn, "max": mx,
            "p95_over_mean": (p95 / mean) if mean else None,
            "max_over_mean": (mx / mean) if mean else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--steps", type=int, default=TARGET_STEPS)
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0, help="冒烟步数 (0=正式)")
    args = ap.parse_args()

    ck = resolve_path(args.ckpt)
    md5_now = _md5(ck)
    if md5_now != CKPT_MD5:
        print(f"exp124c: FATAL md5 硬门不一致: 实测 {md5_now} ≠ 预期 {CKPT_MD5}; 拒跑", flush=True)
        sys.exit(2)
    dep = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp124_epsb_amp.py")
    dep_sha = hashlib.sha256(open(dep, "rb").read()).hexdigest()
    print(f"exp124c: ckpt={ck}\nexp124c: md5={md5_now} (硬门 PASS)\n"
          f"exp124c: 依赖 exp124_epsb_amp.py sha256={dep_sha}", flush=True)

    ReadoutMixin._build_lm_signal = _diag_build_lm_signal
    ReadoutMixin._update_lm_head = _update_lm_head_e3_biasfrozen
    print("exp124c: monkeypatch 已打 (E3 replica [import] + _build_lm_signal 遥测包层; "
          "W35 保持部署版 = amp 关断基线; load 前类方法级)", flush=True)

    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=S_MAX, lm_freeze_w1=False)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    net = HybridPCNet.load(ck, cfg).to(args.device)
    net.eval()
    net._migrate_expand = False
    dev = args.device
    print(f"exp124c: net loaded; mem_k={net._mem_m.shape[0]}; _migrate_expand=False", flush=True)

    gauge, _, _ = build_gauge(args.data, dev)
    eval_wins = gauge["tail"][:N_EVAL_WIN]

    run_json = run_file("exp124c_error_chain.json")
    logf = open(run_file("exp124c_error_chain.log"), "w", encoding="utf-8", buffering=1)
    jsonl = open(run_file("exp124c_error_chain_telemetry.jsonl"), "w", encoding="utf-8", buffering=1)

    init_norms = {}
    for n in PNAMES:
        try:
            init_norms[n] = float(_getp(net, n).norm())
        except Exception:
            init_norms[n] = None
    meta = {"pnames": PNAMES, "init_norms": init_norms,
            "ckpt_md5": md5_now, "ckpt_md5_expected": CKPT_MD5,
            "rule": "E3 replica (import exp124_epsb_amp) + 部署 _update_W35 (amp 关断) "
                    "+ _build_lm_signal 遥测包层 (零训练改动)",
            "dep_exp124_sha256": dep_sha,
            "levels": LEVELS,
            "verdict_rule": "S(x)=max/mean 跨感知步均值; overall=S(m)/S(e0)<0.5 → 上游压平确认; "
                            "归因 gain_marginal=S(p)−S(pg), dz4_marginal=S(pg)−S(m); "
                            "确认压平 → 修 gain/dz4_sig (立项); 未压平 → exp124d 换 amp 公式",
            "design": "对照 = exp124 去掉 amp; 同检查点/数据/种子/协议, 唯一差别 amp 不乘入"}
    jsonl.write(json.dumps({"meta": meta}, ensure_ascii=True) + "\n")

    t0 = time.time()
    pre = eval_bpb_fresh(net, dev, eval_wins)
    print(f"exp124c: pre bpb@s1={pre['bpb_s1']:.6f} top1={pre['top1_s1']:.6f} "
          f"(exp123b pre=7.666549; 偏差即重建口径差)", flush=True)
    base_lines = _sample_dump(net, "exp124c", cfg, dev, 0, 3, 240, 0.8, logf)
    base_hits = word_hits_of(base_lines)

    ds = ByteDataset(args.data, max_length=S_MAX, max_samples=1270000)
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    print(f"exp124c: data n={len(ds)} (P3-b 行为门路由, 首步 force_perception)", flush=True)

    steps = args.smoke if args.smoke > 0 else args.steps
    abort = None
    f_perc_first = f_perc_last = None
    E_last = R_last = None
    lvl_rows = []
    all_rows = []
    frac_rows = []
    for i in range(1, steps + 1):
        x = ds[idxs[(i - 1) % len(idxs)]].unsqueeze(0).to(dev)
        net.learn(x, force_perception=(i == 1))
        net.sync_gate()
        net.maybe_prune(net._step_counter)
        exec_b = net._behavior_py
        f_perc_last = float(net._metab_F_prev_perc)
        E_last = float(net._metab_E)
        R_last = float(net._metab_R)
        if f_perc_first is None:
            f_perc_first = f_perc_last
        norms = param_norms(net)
        nonfinite = [n for n, v in zip(PNAMES, norms) if v is None or not math.isfinite(v)]

        diag = _DIAG["t"]
        row = {"step": i, "behavior": int(exec_b), "f_now_perc": f_perc_last,
               "metab_E": E_last, "metab_R": R_last, "norms": norms}
        if diag is not None:
            vmask = diag["valid"]
            st = {lv: level_stats(diag[lv], vmask) for lv in LEVELS}
            st_all = {lv: level_stats(diag[lv]) for lv in ("m", "r", "amp")}
            valid = diag["valid"].flatten().float()
            amp_v = diag["amp"].flatten().float()[valid > 0]
            row["levels"] = st
            row["levels_all_positions"] = st_all
            row["frac_valid"] = float(valid.mean())
            row["frac_amp_gt_1"] = float((amp_v > 1.0).float().mean()) if amp_v.numel() else None
            row["frac_amp_gt_1_5"] = float((amp_v > 1.5).float().mean()) if amp_v.numel() else None
            frac_rows.append(row)
            bad = [lv for grp in (st, st_all) for lv in grp
                   if grp[lv]["mean"] is None or not math.isfinite(grp[lv]["mean"])
                   or (grp[lv]["max"] is not None and not math.isfinite(grp[lv]["max"]))]
            if bad:
                abort = f"R3-lite: 非有限 level={bad} step={i}"
            lvl_rows.append(st)
            all_rows.append(st_all)
        else:
            row["levels"] = None
        jsonl.write(json.dumps(row, ensure_ascii=True) + "\n")
        if nonfinite:
            abort = f"R3-lite: norms 非有限 {nonfinite} step={i}"
        if abort:
            break
        if i % 100 == 0 and i < steps:
            _sample_dump(net, "exp124c", cfg, dev, i, 3, 240, 0.8, logf)

    post = eval_bpb_fresh(net, dev, eval_wins)
    print(f"exp124c: post bpb@s1={post['bpb_s1']:.6f} top1={post['top1_s1']:.6f} "
          f"(exp123b post=19.877569)", flush=True)

    def agg(key, sub="max_over_mean"):
        vals = [st[key][sub] for st in lvl_rows
                if st.get(key) and st[key].get(sub) is not None]
        return (sum(vals) / len(vals)) if vals else None

    def agg_first_last(key, sub):
        vals = [st[key][sub] for st in lvl_rows
                if st.get(key) and st[key].get(sub) is not None]
        return {"first": vals[0], "last": vals[-1]} if vals else None

    S = {lv: agg(lv) for lv in LEVELS}
    S95 = {lv: agg(lv, "p95_over_mean") for lv in LEVELS}
    S_cascade = {lv: agg(lv) for lv in CASCADE}
    S95_cascade = {lv: agg(lv, "p95_over_mean") for lv in CASCADE}
    overall = (S_cascade["m"] / S_cascade["e0"]) if S_cascade["m"] and S_cascade["e0"] else None
    overall95 = (S95_cascade["m"] / S95_cascade["e0"]) if S95_cascade["m"] and S95_cascade["e0"] else None
    gain_marg = (S_cascade["p"] - S_cascade["pg"]) if None not in (S_cascade["p"], S_cascade["pg"]) else None
    dz4_marg = (S_cascade["pg"] - S_cascade["m"]) if None not in (S_cascade["pg"], S_cascade["m"]) else None
    if gain_marg is not None and dz4_marg is not None:
        if gain_marg <= 0 and dz4_marg <= 0:
            dominant = "none (两乘法门均未降低 spread; 压平来自软饱和映射或投影)"
        else:
            dominant = "gain" if gain_marg >= dz4_marg else "dz4_sig"
    else:
        dominant = None

    def frac_agg(key):
        vals = [r[key] for r in frac_rows if r.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    hi_dose = {"frac_amp_gt_1_mean": frac_agg("frac_amp_gt_1"),
               "frac_amp_gt_1_5_mean": frac_agg("frac_amp_gt_1_5"),
               "frac_amp_gt_1_first": frac_rows[0]["frac_amp_gt_1"] if frac_rows else None,
               "frac_amp_gt_1_5_first": frac_rows[0]["frac_amp_gt_1_5"] if frac_rows else None,
               "frac_amp_gt_1_last": frac_rows[-1]["frac_amp_gt_1"] if frac_rows else None,
               "frac_amp_gt_1_5_last": frac_rows[-1]["frac_amp_gt_1_5"] if frac_rows else None,
               "frac_valid_mean": frac_agg("frac_valid"),
               "note": "有效位 (m>0) 口径; v419 遥测无此项 — 高剂量频率是其盲区"}
    amp_indiv_max = max((st["amp"]["max"] for st in all_rows
                         if st.get("amp") and st["amp"].get("max") is not None), default=None)
    amp_all_p95 = [st["amp"]["p95_over_mean"] for st in all_rows
                   if st.get("amp") and st["amp"].get("p95_over_mean") is not None]
    amp_all_p95_mean = (sum(amp_all_p95) / len(amp_all_p95)) if amp_all_p95 else None
    out = {
        "ckpt": args.ckpt, "ckpt_md5": md5_now, "ckpt_md5_expected": CKPT_MD5,
        "ckpt_md5_gate": "PASS (启动时复算比对一致)",
        "dep_exp124_sha256": dep_sha,
        "design": meta["design"], "verdict_rule": meta["verdict_rule"],
        "steps_run": i, "target_steps": args.steps, "abort": abort,
        "perception_rows": len(lvl_rows), "wrapper_call_counts": _call_counts,
        "bpb_anchor": {"pre": pre["bpb_s1"], "post": post["bpb_s1"],
                       "exp123b_pre": 7.666549, "exp123b_post": 19.877569,
                       "exp124_pre": 7.664317, "exp124_post": 38.028540},
        "fnow_trajectory": {"f_now_perc_first": f_perc_first, "f_now_perc_last": f_perc_last,
                            "E_last": E_last, "R_last": R_last},
        "level_aggregates": {
            lv: {"max_over_mean_mean": S[lv], "p95_over_mean_mean": S95[lv],
                 "max_first_last": agg_first_last(lv, "max")}
            for lv in LEVELS},
        "high_dose_frequency": hi_dose,
        "amp_individual_max": amp_indiv_max,
        "amp_all_positions_p95_over_mean_mean": amp_all_p95_mean,
        "attribution": {
            "S_max_over_mean": S_cascade,
            "S_p95_over_mean": S95_cascade,
            "overall_compression_ratio": overall,
            "overall_compression_ratio_p95": overall95,
            "flattened_confirmed": (overall < 0.5) if overall is not None else None,
            "gain_marginal": gain_marg, "dz4_marginal": dz4_marg,
            "dominant_gate": dominant},
        "crosscheck_v419": {"v419_amp_p95_mean": 1.2381, "v419_amp_individual_max": 1.7197,
                            "note": "同起点跑; amp 关断轨迹与 v419 (amp 开) 会分叉, 全跑均值允许漂移"},
        "drift_top5": sorted(
            [{"name": n, "init": init_norms.get(n),
              "final": float(_getp(net, n).norm()),
              "ratio": (float(_getp(net, n).norm()) / init_norms[n]) if init_norms.get(n) else None}
             for n in PNAMES if init_norms.get(n)],
            key=lambda d: -(d["ratio"] or 0))[:5],
        "word_hits_baseline": base_hits,
        "secs": round(time.time() - t0, 1),
        "iron_rules": "crates/ 零改动 (进程内 monkeypatch); 不动 v244/迁移件; 全 fp16; "
                      "权重不落盘; run_file 自增新目录; 运行不 kill; 回声步显式清空遥测",
    }
    with open(run_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"exp124c: 写出 {run_json}", flush=True)
    print(f"exp124c: 判定摘要 abort={abort} bpb {pre['bpb_s1']:.4f}→{post['bpb_s1']:.4f} "
          f"overall_compression={overall} dominant={dominant} perception_rows={len(lvl_rows)}", flush=True)


if __name__ == "__main__":
    main()
