"""
exp125c: C 型两层拆解 — 矛盾闭合 + 尺度修法 + 方向判据 (500 步 x 3 臂)
=====================================================================

立项依据 (2026-10-19 审查裁决, 聊天记录): exp125b 主因确认 (Q2 尺度
miscalibration), 但报告有两处必须解开的矛盾 + 一个必须放大的实话:

  矛盾 1: cos(dW, W_lm) = -0.314 (更新把 W_lm 往缩小推) 与 W_lm 涨 1.18x
          不能同时成立 — 须逐行范数分解 + cap 触发前后分解;
  矛盾 2: "h 侧贡献 ~3x" 与 rms_norm 应钉住 h 的 RMS 不能同时成立 —
          rms_norm 钉的是 RMS 不是 std, 须分开直测 + saturate/cubic 前后分布;
  实话 3: 尺度校准后 bpb 只到 7.42, top1 0.132 = 87% 位 argmax 错 —
          尺度修好是必要不充分, 须预写两层判据 (尺度层 + 方向层)。

执行口径: agent 直接跑 exp125c。首句给: h 的 RMS vs std (矛盾 2 解开) +
cos(dW, W_lm) 与 W_lm 范数的逐行分解 (矛盾 1 解开) + 尺度修法后 post bpb +
top1 + word_hits + 是否出现第二层 (方向)。

预登记: docs/plan/exp125c_preregistration.md (判读格预写, 跑前锁哈希;
本文件跑前写入, sha256 入账)。

─── 先行证据 (exp125b 遥测重放预分析, out/exp125c_preanalysis.txt) ───

矛盾 1 已被先验数据定量闭合: 重放 W <- (W + dW*lr)*min(1, 10/max_row)
复现实测 W_lm 轨迹, 最大相对误差 0.0004, 逐步符号一致 275/275; 275/275
步更新平方范数变化全为正 — 余弦为负但 lr*||dW|| ~ 266 ~ 2.8x*||W|| ~ 96,
二次项每步压过方向项, cap10 每步绑定 (绑定率 100%) 把 ~2.6x 的膨胀缩回。
即: "余弦 vs 范数" 的张力是"无权重余弦 vs 幅度主导的范数动力学"的口径差,
不是数据矛盾。exp125c D 臂用逐行仪表对此做独立复核 (判据: 重放最大相对
误差 < 5%)。

─── 代码级发现 (crates/hybrid/learning/readout.py:90-99, 一级证据) ───

训练期 logits 与评测期 logits 是**两条温度路径**:
  训练: logits_lm = (h @ W_lm + bias) * inv_h 经 lc 能量调制
        ((x-mean)/std / max|x| * 60) — 对 raw logits 整体缩放**近似不变**;
        eps_total = target - softmax(lc logits) 即训练误差在 lc 温度上。
  评测: eval_bpb_fresh -> _HeadEval 用 **raw logits** (h @ W_lm + bias)*inv_h,
        后置尺度扫描即扫这条路径的温度。
推论: raw logits 尺度漂移 (x3.5) 对训练误差温度近似不可见 (lc 归一化吃掉),
只伤害评测校准 — 但 contrast_w (E3 更新内, 用 raw logits_d) 与 e_h /
eps_proj_raw (表示层反馈, 随 ||W_lm|| 缩放) 是尺度敏感通道, 漂移经它们
回流训练。kernel 精确形式 (crates/kernels): saturate(x)=x/(1+|x|) 有界
(-1,1); cubic(x)=x*(1-0.5x^2) 奇软饱和 (峰值 0.544 @ x=0.816);
rms_norm 钉 RMS 至 1/sqrt(1.01) ~ 0.995 (带 1.01 因子), 不钉 std。

─── 设计 (与预登记逐字对应) ───

三臂 (全部 lr 0.05 = exp125 C 条件, 同检查点/数据/种子/协议, 500 步):
  D  诊断臂: E3 原样 + 全仪表 (逐行范数分解 / cap 前后 / 逐行余弦 /
     h 四级 RMS/std/mean/尾度 / 重放闭合)。复刻保真硬门: post bpb 须与
     exp125b ctrl 逐位相同 (10.797670696653336)。
  S  尺度修法臂 (审查候选 1, 输出尺度结构化): E3 + raw logits std 钉住 —
     每感知步更新+cap10 之后, 若 logits_d std > 目标则全局缩放 W_lm 回目标
     (只缩不涨, 与 cap10 同型); 目标 = 本臂 step-1 pre-update logits std
     (检查点初值校准, ~0.574)。cap10 沿用。
  U  步长修法臂 (审查候选 3, 数据修正形式): E3 + 相对步长约束 — 若
     ||dW*lr|| > rho*||W_lm_pre|| 则缩放 dW (只缩不涨); rho = 0.25 预写。
     形式修正披露: 审查候选 3 原形式"行范数正则 (去径向分量)"经预分析证伪
     方向 — 增长非径向 (径向分量为负), 是幅度主导 (二次项); 故改为相对步长
     约束, 意图不变 (阻止更新本身推高尺度)。

h 形状修法 (审查候选 2) **不在 exp125c 范围**: 其实现须复刻
_build_lm_signal (crates/hybrid/learning/readout.py:35-261, 含
_lang_eps_ema EMA 与 adaptive_traction 等副作用分支), 复刻风险高; 且审查
原文将其设为条件候选 ("如果 h 变尖是主因")。D 臂的矛盾 2 仪表即为该条件的
判据; 若确认, h 形状修法成为 exp125d 首臂 (自带复刻保真门)。此 Scope
决定已披露于预登记。

判据 (预写, 两层):
  主判据: post bpb (s=1) < 5.0
  第一层 (尺度修好): 后置扫描最优 bpb < 8.0 (S 臂另报 s=1 bpb, 应 ~ 最优)
  第二层 (方向改善): top1_post > 0.25
  副判据: word_hits >= 1
  分流: 两层都过 -> 双修 (bpb<5 则主判据过, exp125d 固化; 否则记"双修未达
        流畅"复议); 只过第一层 -> 诚实报告"尺度修好、方向有上限", 转方向
        诊断立项, **不得继续调尺度**; 两层都不过 -> C 型非尺度能修, 转更深
        立项 (更新方向/表示层)。

偏离披露:
  - 宿主 PowerShell stdout 通道故障: 启动即重定向 stdout/stderr 到 run 目录;
    运行不 kill; Python 须用 .venv (系统 PATH python.exe 为 WindowsApps 存根)。
  - 墙钟预期 ~15-20 分钟/臂 (exp125b 实测 14.3-15.6 分钟); 三臂串行独立进程。
  - lr 固定 0.05, 不作变量。

铁律: crates/ 零改动 (仅进程内类方法 monkeypatch); 不动 v244/迁移件 (只读
归档检查点, md5 硬门); 全 fp16 (无 clamp 数值; 统计全 .float() 只读路径);
权重不落盘; run_file 自增新目录。
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter

import torch
import torch.nn.functional as F

import exp124_epsb_amp as e124
from exp115_say import N_EVAL_WIN, S_MAX, _sample_dump, build_gauge
from exp116_z4_probe import _md5, infer, score_logits

from crates.cli.utils import resolve_path, run_file
from crates.graphs import set_backend
set_backend("ref")
from crates.kernels import cubic, rms_norm, saturate
from crates.kernels.normalize import soft_norm_preserve
from crates.modal.text.geometry import ALPHABET
from crates.hybrid.learning.readout import ReadoutMixin
from crates.model import CyreneModel, HybridPCNet
from crates.utils.codec import ByteDataset

torch.set_grad_enabled(False)

LR = 0.05
TARGET_STEPS = 500
ENT_WIN = 8192
ENT_MIN = 256
ENT_STREAK = 100
TRUST_RHO = 0.25
ARMS = {"D": {"pin": False, "trust": False},
        "S": {"pin": True, "trust": False},
        "U": {"pin": False, "trust": True}}
SWEEP_SCALES = [1.6, 1.3, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]

_ARM = dict(ARMS["D"])
_TEL = {"fresh": False}
_PIN = {"target": None}


def _stage_stats(t):
    tf = t.detach().float()
    flat = tf.flatten()
    rms = float(flat.square().mean().sqrt())
    std = float(flat.std())
    mean = float(flat.mean())
    q99 = float(torch.quantile(flat.abs(), 0.99))
    return {"rms": rms, "std": std, "mean": mean, "p99abs": q99,
            "p99_over_rms": (q99 / rms) if rms > 0 else None}


def _row_stats(rn):
    rf = rn.detach().float()
    qs = torch.quantile(rf, torch.tensor([0.1, 0.5, 0.9, 0.99], device=rf.device))
    return {"p10": float(qs[0]), "p50": float(qs[1]), "p90": float(qs[2]),
            "p99": float(qs[3]), "max": float(rf.max()), "mean": float(rf.mean()),
            "n_gt10": float((rf > 10.0).float().sum())}


def _grown_share(w_pre_rows, w_post_rows):
    delta = (w_post_rows - w_pre_rows).clamp_min(0.0)
    total = float(delta.sum())
    if total <= 0:
        return None
    top10 = float(torch.topk(delta.flatten(), 10).values.sum())
    return top10 / total


# ─────────── E3 逐字复刻 + 只读仪表 + 双臂修法 (D 双臂关 = exp125b ctrl) ───────────

def _update_lm_head_e3c_tel(self, ctx, lm):
    net = self.net
    if ctx.free_run or ctx.echo_world_frozen:
        _TEL["fresh"] = False
        return net._one_s[: ctx.N, : ctx.S - 1]
    d_h = net.d_h
    inv_h = 1.0 / math.sqrt(d_h)
    byte_ids = ctx.byte_ids

    err_norm = lm.eps_total.norm(dim=-1, keepdim=True) * 1.01
    alive = (err_norm > 1e-8).to(lm.eps_total.dtype)
    denom = torch.where(alive > 0, err_norm, torch.ones_like(err_norm))
    err_scaled = lm.eps_total * alive / denom

    target_oh = F.one_hot(byte_ids[:, 1:], num_classes=ALPHABET).to(torch.float16)
    logits_d = (lm.h[:, :-1] @ net.W_lm) * inv_h
    tgt_logits = (logits_d * target_oh).sum(dim=-1, keepdim=True)
    contrast_w = 1.0 / (1.0 + (logits_d - tgt_logits).abs() * 0.1)
    contrast_w = contrast_w * (1.0 - target_oh) + target_oh
    err_contrast = err_scaled * contrast_w

    if hasattr(net, "_novelty"):
        nov = net._novelty
        tau = net._theta_novelty
        tau.mul_(0.99).add_(0.01 * nov.mean())
        d_t = torch.sigmoid((nov - tau) * 500.0).unsqueeze(-1)[:, :-1]
    else:
        d_t = net._one_s[: ctx.N, : ctx.S - 1]
    lm_update_mask = ctx.learn_mask.to(torch.float16).unsqueeze(-1)
    dW_lm = (
        rms_norm(lm.h[:, :-1]).transpose(-2, -1) @ (err_contrast * lm_update_mask * d_t)
    ).mean(dim=0) * math.sqrt(d_h)

    net._tel_dWlm_t = dW_lm.abs().mean().detach()
    net._tel_err_t = err_scaled.abs().mean().detach()

    # ── 只读仪表快照 (clone/detach; 不动训练张量, 不耗 RNG) ──
    wlm_pre = net.W_lm.data.clone()
    dW_prev = _TEL.get("dW_prev")
    _TEL["dW_prev"] = dW_lm.detach().clone()
    nov_mean = float(nov.detach().float().mean()) if hasattr(net, "_novelty") else None
    tau_mean = float(tau.detach().float().mean()) if hasattr(net, "_novelty") else None
    row_pre = _row_stats(wlm_pre.norm(dim=1))
    dW_norm_pre = float(dW_lm.detach().float().norm())
    step_norm_pre = dW_norm_pre * LR
    wpre_rows = wlm_pre.detach().float().norm(dim=1)
    wpre_norm_f = float(wlm_pre.detach().float().norm())

    # U 臂: 相对步长约束 (只缩不涨; rho 预写 0.25)
    trust_scale = 1.0
    if _ARM["trust"] and step_norm_pre > TRUST_RHO * wpre_norm_f:
        trust_scale = (TRUST_RHO * wpre_norm_f) / (step_norm_pre + 1e-12)
        dW_lm = dW_lm * trust_scale
        net._tel_dWlm_t = dW_lm.abs().mean().detach()

    net.W_lm.data += dW_lm * LR
    rn_post = net.W_lm.data.norm(dim=1)
    mx_lm = rn_post.max()
    cap_scale = float(torch.minimum(torch.ones_like(mx_lm),
                                    10.0 / (mx_lm + 1e-6)).to(torch.float16))
    net.W_lm.data.mul_(
        torch.minimum(torch.ones_like(mx_lm), 10.0 / (mx_lm + 1e-6)).to(torch.float16)
    )
    row_post_precap = _row_stats(rn_post)
    rn_cap = net.W_lm.data.norm(dim=1)
    row_post_cap = _row_stats(rn_cap)

    # S 臂: raw logits std 钉住 (目标 = step-1 pre-update logits std; 只缩不涨)
    pin_scale = 1.0
    std_now = None
    if _ARM["pin"]:
        lg_now = (lm.h[:, :-1] @ net.W_lm) * inv_h
        std_now = float(lg_now.detach().float().std())
        if _PIN["target"] is None:
            _PIN["target"] = float(logits_d.detach().float().std())
        if std_now > _PIN["target"]:
            pin_scale = _PIN["target"] / (std_now + 1e-12)
            net.W_lm.data.mul_(pin_scale)
            rn_cap = net.W_lm.data.norm(dim=1)
            row_post_cap = _row_stats(rn_cap)

    e_h = (err_scaled @ net.W_lm.T) * lm.h_deriv[:, :-1]
    lm.e_h = e_h
    dW1 = (rms_norm(lm.zh[:, :-1]).transpose(-2, -1) @ rms_norm(e_h)).mean(dim=0)
    net._dW1_absmax_raw = dW1.abs().max().detach()
    mx1 = dW1.abs().max() + 1e-4
    dW1 = dW1 / mx1 / 64.0
    net._tel_dW1_t = dW1.abs().mean().detach()
    if not net.cfg.lm_freeze_w1:
        net.W1.data += dW1 * LR
        soft_norm_preserve(net.W1.data)

    net._tel_Wlm_norm_t = net.W_lm.data.norm().detach()
    net._tel_W1_norm_t = net.W1.data.norm().detach()
    net._tel_bias_norm_t = net.bias_lm.data.norm().detach()

    h_m = lm.h[:, :-1]
    h_mean = h_m.mean()
    net._tel_h_mean_t = h_mean.detach()
    net._tel_h_var_t = (h_m - h_mean).square().mean().detach()

    # ── h 四级分解 (矛盾 2 仪表; 与 readout.py:72-77 同式复算) ──
    zh_c = lm.zh[:, :-1]
    y = zh_c @ net.W1
    y_s = saturate(y)
    y_n = rms_norm(y_s)
    h_rec = cubic(y_n)
    st_y, st_ys, st_yn, st_h = (_stage_stats(y), _stage_stats(y_s),
                                _stage_stats(y_n), _stage_stats(h_rec))
    h_true = lm.h[:, :-1]
    h_fid_std_ratio = (st_h["std"] / float(h_true.detach().float().std())) if st_h["std"] else None
    h_fid_maxdiff = float((h_rec.detach().float() - h_true.detach().float()).abs().max())

    # ── 逐行分解 (矛盾 1 仪表) ──
    dW_r = dW_lm.detach().float()
    w_r = wlm_pre.detach().float()
    dot_r = (dW_r * w_r).sum(dim=1)
    nw_r = w_r.norm(dim=1) * dW_r.norm(dim=1) + 1e-12
    row_cos_mean = float((dot_r / nw_r).mean())
    grow_n = int((rn_cap.detach().float() - wpre_rows > 0).sum())
    grow_share = _grown_share(wpre_rows, rn_cap.detach().float())
    top5_rows = [float(v) for v in torch.topk(rn_cap.detach().float().flatten(), 5).values]

    dW_f = dW_lm.detach().float().flatten()
    wpre_f = w_r.flatten()
    nrm = lambda v: float(v.norm()) if v.numel() else 0.0
    cos = lambda a, b: float((a @ b) / (a.norm() * b.norm() + 1e-12))
    lg_f = logits_d.detach().float()
    p_sm = torch.softmax(lg_f, dim=-1)
    nt = (1.0 - target_oh.detach().float()) > 0.5
    cw_f = contrast_w.detach().float()
    dt_f = d_t.detach().float().flatten()
    _TEL.update({
        "fresh": True,
        "logits_std": float(lg_f.std()),
        "sm_entropy": float(-(p_sm * p_sm.clamp_min(1e-9).log2()).sum(dim=-1).mean()),
        "sm_maxp": float(p_sm.max(dim=-1).values.mean()),
        "cw_nt_mean": float((cw_f * nt.float()).sum() / nt.float().sum().clamp_min(1.0)),
        "cw_nt_frac_lt_05": float((cw_f[nt] < 0.5).float().mean()) if bool(nt.any()) else None,
        "dt_mean": float(dt_f.mean()),
        "dt_p95": float(torch.quantile(dt_f, 0.95)),
        "dt_frac_gt_05": float((dt_f > 0.5).float().mean()),
        "dt_frac_lt_01": float((dt_f < 0.1).float().mean()),
        "nov_mean": nov_mean,
        "tau_mean": tau_mean,
        "wlm_pre_norm": wpre_norm_f,
        "dW_norm_pre": dW_norm_pre,
        "dW_norm": nrm(dW_f),
        "dW_absmean": float(dW_f.abs().mean()),
        "dW_cos_wlm": cos(dW_f, wpre_f),
        "dW_cos_row_mean": row_cos_mean,
        "dW_cos_prev": cos(dW_f, dW_prev.detach().float().flatten()) if dW_prev is not None else None,
        "dW_delta_theo_norm": nrm(dW_lm.detach().float().flatten() * LR),
        "dW_delta_act_norm": float((net.W_lm.data.float().flatten() - wpre_f).norm()),
        "trust_scale": trust_scale,
        "step_norm_pre": step_norm_pre,
        "wlm_row_pre": row_pre,
        "wlm_row_post_precap": row_post_precap,
        "wlm_row_post_cap": row_post_cap,
        "wlm_cap_scale": cap_scale,
        "wlm_grow_n": grow_n,
        "wlm_grow_share": grow_share,
        "wlm_top5_rows": top5_rows,
        "pin_scale": pin_scale,
        "logits_std_now": std_now,
        "h_stage_y": st_y, "h_stage_y_saturate": st_ys,
        "h_stage_y_norm": st_yn, "h_stage_h": st_h,
        "h_fid_std_ratio": h_fid_std_ratio, "h_fid_maxdiff": h_fid_maxdiff,
        "e0_mean": float(lm.eps_total.detach().float().norm(dim=-1).mean()),
        "alive_frac": float(alive.detach().float().mean()),
    })

    if hasattr(net, "_novelty"):
        del net._novelty
    return d_t


class _HeadEvalNoBias:
    def __init__(self, net):
        self.net = net

    def eval(self):
        return self

    def __call__(self, x):
        net = self.net
        return (x @ net.W_lm) * (1.0 / math.sqrt(net.d_h))


def post_sweeps(net, dev, eval_wins, tag):
    h, y = e124.extract_h(net, dev, eval_wins, tag)
    lg = infer(e124._HeadEval(net), h, dev, 8192)
    rows = []
    for s in SWEEP_SCALES:
        sc = score_logits(lg, y, scale=s)
        rows.append({"scale": s, "temp": round(1.0 / s, 3),
                     "bpb": sc["bpb"], "top1": sc["top1"]})
    best = min(rows, key=lambda r: r["bpb"])
    lg_nb = infer(_HeadEvalNoBias(net), h, dev, 8192)
    nb1 = score_logits(lg_nb, y, scale=1.0)
    nbb = score_logits(lg_nb, y, scale=best["scale"])
    return {"sweep": rows, "best": best,
            "bpb_s1": next(r["bpb"] for r in rows if r["scale"] == 1.0),
            "gain_ratio": best["bpb"] / next(r["bpb"] for r in rows if r["scale"] == 1.0),
            "bias_off_s1": nb1, "bias_off_best": nbb}


def replay_closure(tel_rows, lr=LR):
    """矛盾 1 闭合检验: W <- ((W + dW*lr)*cap)*pin 重放实测轨迹."""
    errs = []
    agree = 0
    for r in tel_rows:
        wpre = r.get("wlm_pre_norm")
        dWn = r.get("dW_norm")
        cosv = r.get("dW_cos_wlm")
        cap = r.get("wlm_cap_scale")
        pin = r.get("pin_scale") or 1.0
        meas = r.get("W_lm_norm")
        if None in (wpre, dWn, cosv, cap, meas):
            continue
        theo2 = wpre * wpre + 2.0 * cosv * wpre * dWn * lr + (dWn * lr) ** 2
        pred = math.sqrt(max(theo2, 0.0)) * cap * pin
        errs.append(abs(pred - meas) / meas)
        if ((meas - wpre) >= 0) == ((pred - wpre) >= 0):
            agree += 1
    if not errs:
        return None
    return {"n": len(errs), "max_rel_err": max(errs), "mean_rel_err": sum(errs) / len(errs),
            "sign_agreement": agree}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--steps", type=int, default=TARGET_STEPS)
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0,
                    help="冒烟步数 (0=正式; 冒烟跳过 eval/dump/audit, 保留后置扫描验证)")
    args = ap.parse_args()
    arm = args.arm
    _ARM.update(ARMS[arm])
    _PIN["target"] = None
    tag = f"exp125c_{arm}"
    smoke = args.smoke > 0
    steps = args.smoke if smoke else args.steps

    stdout_path = run_file(tag + "_stdout.log")
    logf_console = open(stdout_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = logf_console
    sys.stderr = logf_console

    ck = resolve_path(e124.CKPT)
    md5_now = _md5(ck)
    if md5_now != e124.CKPT_MD5:
        print(f"exp125c: FATAL md5 硬门不一致: 实测 {md5_now} != 预期 {e124.CKPT_MD5}; 拒跑", flush=True)
        sys.exit(2)
    dep = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp124_epsb_amp.py")
    dep_sha = hashlib.sha256(open(dep, "rb").read()).hexdigest()
    self_sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    print(f"exp125c: arm={arm} lr={LR} switches={ARMS[arm]} smoke={smoke} steps={steps}\n"
          f"exp125c: ckpt={ck}\nexp125c: md5={md5_now} (硬门 PASS)\n"
          f"exp125c: 依赖 exp124_epsb_amp.py sha256={dep_sha}\n"
          f"exp125c: 执行件 sha256={self_sha}", flush=True)

    ReadoutMixin._update_lm_head = _update_lm_head_e3c_tel
    print(f"exp125c: monkeypatch: E3 复刻+仪表 (pin={'ON' if _ARM['pin'] else 'off'}, "
          f"trust={'ON rho=' + str(TRUST_RHO) if _ARM['trust'] else 'off'}) + 部署 _update_W35 (amp 关断)",
          flush=True)

    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=S_MAX, lm_freeze_w1=False)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = torch.device(args.device)
    net = HybridPCNet.load(ck, cfg).to(dev)
    net.eval()
    net._migrate_expand = False
    print(f"exp125c: net loaded; d_h={net.d_h} mem_k={net._mem_m.shape[0]}; _migrate_expand=False", flush=True)

    gauge, _, _ = build_gauge(args.data, dev)
    eval_wins = gauge["tail"][:N_EVAL_WIN]
    print(f"exp125c: eval_wins={len(eval_wins)}", flush=True)

    run_json = run_file(tag + ".json")
    logf = open(run_file(tag + ".log"), "w", encoding="utf-8", buffering=1)
    jsonl = open(run_file(tag + "_telemetry.jsonl"), "w", encoding="utf-8", buffering=1)

    init_norms = {}
    for n in e124.PNAMES:
        try:
            init_norms[n] = float(e124._getp(net, n).norm())
        except Exception:
            init_norms[n] = None
    meta = {"pnames": e124.PNAMES, "init_norms": init_norms,
            "exp": "exp125c", "arm": arm, "lr": LR, "switches": ARMS[arm], "smoke": smoke,
            "ckpt": e124.CKPT, "ckpt_md5": md5_now, "ckpt_md5_expected": e124.CKPT_MD5,
            "dep_exp124_sha256": dep_sha, "self_sha256": self_sha,
            "unique_variable": "臂修法 (S: raw logits std 钉住 / U: 相对步长约束 rho=0.25); lr 固定 0.05",
            "rule": "E3 逐字复刻+只读仪表 (D=原样) + 部署 _update_W35 (amp 关断)",
            "fidelity_gate": {"D_post_bpb": 10.797670696653336,
                              "note": "D 臂须与 exp125b ctrl / exp125 C 逐位相同"},
            "verdict_rule": "主判据 post bpb<5.0; 第一层 扫描最优 bpb<8.0; 第二层 top1>0.25; "
                            "副判据 word_hits>=1; c_type:=post>=8.0 且 top1>=0.05 且 f_now max<100"}
    jsonl.write(json.dumps({"meta": meta}, ensure_ascii=True) + "\n")

    t0 = time.time()
    if smoke:
        pre = post = None
        base_lines = []
        base_hits = None
        print("exp125c: 冒烟模式 — 跳过 pre/post eval / dump / audit (保留后置扫描)", flush=True)
    else:
        pre = e124.eval_bpb_fresh(net, dev, eval_wins)
        print(f"exp125c[{arm}]: pre bpb@s1={pre['bpb_s1']:.6f} top1={pre['top1_s1']:.6f} "
              f"(exp125 C pre=7.664317404722619; 须逐位相同=协议保真)", flush=True)
        base_lines = _sample_dump(net, tag, cfg, dev, 0, 3, 240, 0.8, logf)
        base_hits = e124.word_hits_of(base_lines)

    ds = ByteDataset(args.data, max_length=S_MAX, max_samples=1270000)
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    print(f"exp125c[{arm}]: data n={len(ds)}", flush=True)

    ent_buf = bytearray()
    ent_first = ent_last = None
    ent_min = ent_max = None
    ent_updates = 0
    decline_streak = rise_streak = 0
    abort = None
    f_perc_first = f_perc_last = None
    f_eco_last = None
    E_last = R_last = None
    f_perc_max = None
    death_cause = None
    dump_ents = {}
    tel_rows = []

    def entropy_now():
        if len(ent_buf) < ENT_MIN:
            return None
        cnt = [0] * 256
        for b in ent_buf:
            cnt[b] += 1
        n = len(ent_buf)
        h = 0.0
        for c in cnt:
            if c:
                p = c / n
                h -= p * math.log2(p)
        return h

    def push_entropy(bs):
        nonlocal ent_first, ent_last, ent_min, ent_max, ent_updates
        nonlocal decline_streak, rise_streak
        ent_buf.extend(bs)
        if len(ent_buf) > ENT_WIN:
            del ent_buf[: len(ent_buf) - ENT_WIN]
        h = entropy_now()
        if h is None:
            return None
        if ent_first is None:
            ent_first = h
            ent_min = ent_max = h
            decline_streak = rise_streak = 0
        else:
            if h < ent_last:
                decline_streak += 1
                rise_streak = 0
            elif h > ent_last:
                rise_streak += 1
                decline_streak = 0
            ent_min = min(ent_min, h)
            ent_max = max(ent_max, h)
        ent_last = h
        ent_updates += 1
        return h

    i = 0
    for i in range(1, steps + 1):
        x = ds[idxs[(i - 1) % len(idxs)]].unsqueeze(0).to(dev)
        net.learn(x, force_perception=(i == 1))
        net.sync_gate()
        death_cause = net.maybe_prune(net._step_counter)
        exec_b = net._behavior_py
        if exec_b and hasattr(net, "_gen_bytes"):
            push_entropy(bytes(int(v) for v in net._gen_bytes[0].tolist()))

        f_perc_last = float(net._metab_F_prev_perc)
        f_eco_last = float(net._metab_F_prev_eco)
        E_last = float(net._metab_E)
        R_last = float(net._metab_R)
        if f_perc_first is None:
            f_perc_first = f_perc_last
        f_perc_max = f_perc_last if f_perc_max is None else max(f_perc_max, f_perc_last)
        norms = e124.param_norms(net)
        nonfinite = [n for n, v in zip(e124.PNAMES, norms) if v is None or not math.isfinite(v)]
        d1r = float(net._dW1_absmax_raw) if hasattr(net, "_dW1_absmax_raw") else None

        def _gt(name):
            t = getattr(net, name, None)
            return float(t) if t is not None else None

        tel = dict(_TEL) if _TEL.get("fresh") else {}
        _TEL["fresh"] = False
        row = {"step": i, "lr": LR, "arm": arm, "behavior": int(exec_b),
               "W_lm_norm": _gt("_tel_Wlm_norm_t"), "W1_norm": _gt("_tel_W1_norm_t"),
               "bias_lm_norm": _gt("_tel_bias_norm_t"),
               "dWlm_absmean": _gt("_tel_dWlm_t"), "dW1_absmean": _gt("_tel_dW1_t"),
               "dW1_absmax_raw": d1r,
               "f_now_perc": f_perc_last, "f_now_eco": f_eco_last,
               "metab_E": E_last, "metab_R": R_last,
               "char_entropy": entropy_now(), "ent_decline_streak": decline_streak,
               "ent_rise_streak": rise_streak, "ent_updates": ent_updates,
               "norms": norms, "mem_k": net._mem_m.shape[0]}
        for k in ("logits_std", "sm_entropy", "sm_maxp", "cw_nt_mean", "cw_nt_frac_lt_05",
                  "dt_mean", "dt_p95", "dt_frac_gt_05", "dt_frac_lt_01", "nov_mean", "tau_mean",
                  "wlm_pre_norm", "dW_norm_pre", "dW_norm", "dW_absmean", "dW_cos_wlm",
                  "dW_cos_row_mean", "dW_cos_prev", "dW_delta_theo_norm", "dW_delta_act_norm",
                  "trust_scale", "step_norm_pre", "wlm_row_pre", "wlm_row_post_precap",
                  "wlm_row_post_cap", "wlm_cap_scale", "wlm_grow_n", "wlm_grow_share",
                  "wlm_top5_rows", "pin_scale", "logits_std_now", "h_stage_y",
                  "h_stage_y_saturate", "h_stage_y_norm", "h_stage_h", "h_fid_std_ratio",
                  "h_fid_maxdiff", "e0_mean", "alive_frac"):
            row[k] = tel.get(k)
        jsonl.write(json.dumps(row, ensure_ascii=True) + "\n")
        tel_rows.append(row)

        if nonfinite or (d1r is not None and not math.isfinite(d1r)):
            abort = f"R3: 数值异常 nonfinite={nonfinite} dW1_absmax_raw={d1r}"
            break
        for n, v in zip(e124.PNAMES, norms):
            iv = init_norms.get(n)
            if iv is None or v is None or iv <= 0:
                continue
            if v > 3.0 * iv:
                abort = f"R1: {n} 范数 {v} > 3x init {3.0 * iv}"
                break
        if abort:
            break
        if decline_streak >= ENT_STREAK or rise_streak >= ENT_STREAK:
            abort = f"R2: 熵 streak decline={decline_streak} rise={rise_streak}"
            break
        if not smoke and i % 100 == 0 and i < steps:
            lines = _sample_dump(net, tag, cfg, dev, i, 3, 240, 0.8, logf)
            dump_ents[i] = lines
            for t in lines:
                push_entropy(t.encode("utf-8", errors="replace"))
            if len(set(lines)) < len(lines):
                abort = f"R2b: step {i} dump 内 3 个不同种子样本逐字相同"
                break
        if i % 100 == 0:
            print(f"exp125c[{arm}]: step {i}/{steps} f_perc={f_perc_last:.4f} "
                  f"logits_std={tel.get('logits_std')} cap={tel.get('wlm_cap_scale')} "
                  f"trust={tel.get('trust_scale')} pin={tel.get('pin_scale')} "
                  f"W_lm={norms[e124.PNAMES.index('W_lm')]:.4f}", flush=True)
        if death_cause:
            print(f"exp125c[{arm}]: 机体自维护 death_cause={death_cause} (照常运行, 不中止)", flush=True)
            death_cause = None

    sweep = None
    if smoke:
        final_lines = []
        wh = None
        print(f"exp125c[{arm}]: 冒烟注入自检 — step1 dWlm_absmean={_gt('_tel_dWlm_t')} "
              f"W_lm_norm={_gt('_tel_Wlm_norm_t')} "
              f"logits_std={tel_rows[0].get('logits_std') if tel_rows else None} "
              f"h_fid_ratio={tel_rows[0].get('h_fid_std_ratio') if tel_rows else None}", flush=True)
        sweep = post_sweeps(net, dev, eval_wins, tag + "_sweep")
        print(f"exp125c[{arm}]: 冒烟后置扫描验证 — best scale={sweep['best']['scale']} "
              f"bpb={sweep['best']['bpb']:.4f} vs s1={sweep['bpb_s1']:.4f} (冒烟值不作判读)", flush=True)
    else:
        if abort is None and i == steps:
            final_lines = _sample_dump(net, tag, cfg, dev, steps, 3, 240, 0.8, logf)
            dump_ents[steps] = final_lines
            for t in final_lines:
                push_entropy(t.encode("utf-8", errors="replace"))
            if len(set(final_lines)) < len(final_lines):
                abort = f"R2b: step {steps} dump 内 3 个不同种子样本逐字相同"
        else:
            final_lines = []
        post = e124.eval_bpb_fresh(net, dev, eval_wins)
        print(f"exp125c[{arm}]: post bpb@s1={post['bpb_s1']:.6f} top1={post['top1_s1']:.6f} "
              f"(D 复现目标 10.797670696653336)", flush=True)
        wh = e124.word_hits_of(final_lines)
        sweep = post_sweeps(net, dev, eval_wins, tag + "_sweep")
        print(f"exp125c[{arm}]: 后置扫描 best scale={sweep['best']['scale']} "
              f"(T={sweep['best']['temp']}) bpb={sweep['best']['bpb']:.4f} vs s1={sweep['bpb_s1']:.4f} "
              f"gain={sweep['gain_ratio']:.3f} bias_off_s1={sweep['bias_off_s1']['bpb']:.4f}", flush=True)

        rows = [("S0 tau=0.8 (基线)", "".join(base_lines))]
        for s in sorted(dump_ents):
            rows.append((f"S{s} tau=0.8", "".join(dump_ents[s])))
        corpus_texts = []
        with open(args.data, encoding="utf-8") as f:
            for j, line in enumerate(f):
                if j >= 24:
                    break
                corpus_texts.append(json.loads(line)["text"])
        corpus_text = "".join(corpus_texts)
        cnt = Counter()
        with open(args.data, encoding="utf-8") as f:
            for j, line in enumerate(f):
                if j >= 20000:
                    break
                cnt.update(json.loads(line)["text"])
        top_sets = {"top100": {c for c, _ in cnt.most_common(100)},
                    "top1000": {c for c, _ in cnt.most_common(1000)},
                    "top3000": {c for c, _ in cnt.most_common(3000)}}
        with open(run_file(tag + "_audit_char.txt"), "w", encoding="utf-8", buffering=1) as f:
            head = (f"{'条件':<28} {'字符数':>6} {'CJK':>6} {'top100':>8} {'top1000':>8} "
                    f"{'top3000':>8} {'ASCII%':>7} {'词命中':>5}")
            f.write(head + "\n")

            def wrow(name, text):
                cjk = [c for c in text if ord(c) > 127]
                asc = (len(text) - len(cjk)) / max(1, len(text))
                t100 = sum(1 for c in cjk if c in top_sets["top100"]) / max(1, len(cjk))
                t1000 = sum(1 for c in cjk if c in top_sets["top1000"]) / max(1, len(cjk))
                t3000 = sum(1 for c in cjk if c in top_sets["top3000"]) / max(1, len(cjk))
                f.write(f"{name:<28} {len(text):>6} {len(cjk):>6} {t100:>8.3f} {t1000:>8.3f} "
                        f"{t3000:>8.3f} {asc:>7.3f} {e124.word_hits_of([text]):>5}\n")

            for name, text in rows:
                wrow(name, text)
            wrow("语料基线", corpus_text)
            wrow("纯物理(零学习)", e124.physics_null())

    norms_final = e124.param_norms(net)
    drift = []
    for n, v in zip(e124.PNAMES, norms_final):
        iv = init_norms.get(n)
        if iv is not None and iv > 0 and v is not None:
            drift.append({"name": n, "init": iv, "final": v, "ratio": v / iv})
    drift.sort(key=lambda r: -r["ratio"])

    def agg(key):
        vals = [r.get(key) for r in tel_rows]
        vals = [v for v in vals if v is not None]
        return (sum(vals) / len(vals)) if vals else None

    def dt_win(lo, hi):
        vals = [r["dt_mean"] for r in tel_rows
                if r.get("dt_mean") is not None and lo < r["step"] <= hi]
        return (sum(vals) / len(vals)) if vals else None

    n_perc = len([r for r in tel_rows if r.get("wlm_cap_scale") is not None])
    cap_bind = (sum(1 for r in tel_rows if (r.get("wlm_cap_scale") or 1.0) < 1.0) / n_perc) if n_perc else None
    perc = [r for r in tel_rows if r.get("logits_std") is not None]
    first_p, last_p = (perc[0] if perc else {}), (perc[-1] if perc else {})
    closure = replay_closure([r for r in tel_rows if r.get("dW_cos_wlm") is not None])

    if smoke:
        post_bpb = None
        learned = c_type = crash_type = stable = None
        layer1 = layer2 = None
    else:
        post_bpb = post["bpb_s1"]
        top1_post = post["top1_s1"]
        learned = post_bpb < 5.0
        stable = post_bpb < 8.0
        c_type = (post_bpb >= 8.0) and (top1_post >= 0.05) and ((f_perc_max or 0.0) < 100.0)
        crash_type = (post_bpb >= 8.0) and not c_type
        layer1 = sweep["best"]["bpb"] < 8.0
        layer2 = top1_post > 0.25

    out = {
        "exp": "exp125c", "arm": arm, "lr": LR, "switches": ARMS[arm], "smoke": smoke,
        "rule": meta["rule"], "fidelity_gate": meta["fidelity_gate"],
        "ckpt": e124.CKPT, "ckpt_md5": md5_now, "ckpt_md5_expected": e124.CKPT_MD5,
        "ckpt_md5_gate": "PASS (启动时复算比对一致)",
        "dep_exp124_sha256": dep_sha, "self_sha256": self_sha,
        "steps_run": i, "target_steps": steps, "abort": abort,
        "bpb": {"pre": (pre["bpb_s1"] if pre else None), "post": post_bpb,
                "top1_pre": (pre["top1_s1"] if pre else None),
                "top1_post": (post["top1_s1"] if post else None)},
        "verdict": {
            "main_bpb_lt_5": {"post": post_bpb, "pass": (post_bpb is not None and post_bpb < 5.0)},
            "layer1_scale_sweep_lt_8": {"sweep_best_bpb": (sweep["best"]["bpb"] if sweep else None),
                                        "sweep_best_scale": (sweep["best"]["scale"] if sweep else None),
                                        "bpb_s1": (sweep["bpb_s1"] if sweep else None),
                                        "pass": layer1},
            "layer2_direction_top1_gt_025": {"top1_post": (post["top1_s1"] if post else None),
                                             "pass": layer2},
            "word_hits_ge_1": {"word_hits_T08": wh, "pass": (wh is not None and wh >= 1)}},
        "failure_mode": {"learned": learned, "stable": stable,
                         "c_type_calibration": c_type, "crash_type": crash_type},
        "contradiction1_row_cap": {
            "replay_closure": closure,
            "dW_cos_wlm_flat_mean": agg("dW_cos_wlm"),
            "dW_cos_row_mean_mean": agg("dW_cos_row_mean"),
            "dW_norm_pre_mean": agg("dW_norm_pre"),
            "step_norm_pre_mean": agg("step_norm_pre"),
            "wlm_pre_norm_first": first_p.get("wlm_pre_norm"),
            "cap10_binding_frac": cap_bind,
            "row_max_pre_first": (first_p.get("wlm_row_pre") or {}).get("max"),
            "row_max_pre_last": (last_p.get("wlm_row_pre") or {}).get("max"),
            "row_max_postcap_last": (last_p.get("wlm_row_post_cap") or {}).get("max"),
            "row_p50_first": (first_p.get("wlm_row_pre") or {}).get("p50"),
            "row_p50_last": (last_p.get("wlm_row_pre") or {}).get("p50"),
            "row_p99_first": (first_p.get("wlm_row_pre") or {}).get("p99"),
            "row_p99_last": (last_p.get("wlm_row_pre") or {}).get("p99"),
            "row_n_gt10_last": (last_p.get("wlm_row_post_cap") or {}).get("n_gt10"),
            "grow_n_mean": agg("wlm_grow_n"),
            "grow_top10_share_mean": agg("wlm_grow_share"),
            "note": "闭合判据: replay max_rel_err < 5% (先验 0.0004); grow_share = top10 增长行占正增量份额"},
        "contradiction2_h_stage": {
            "y_rms_first_last": [(first_p.get("h_stage_y") or {}).get("rms"),
                                 (last_p.get("h_stage_y") or {}).get("rms")],
            "y_std_first_last": [(first_p.get("h_stage_y") or {}).get("std"),
                                 (last_p.get("h_stage_y") or {}).get("std")],
            "y_saturate_std_first_last": [(first_p.get("h_stage_y_saturate") or {}).get("std"),
                                          (last_p.get("h_stage_y_saturate") or {}).get("std")],
            "y_norm_rms_first_last": [(first_p.get("h_stage_y_norm") or {}).get("rms"),
                                      (last_p.get("h_stage_y_norm") or {}).get("rms")],
            "y_norm_std_first_last": [(first_p.get("h_stage_y_norm") or {}).get("std"),
                                      (last_p.get("h_stage_y_norm") or {}).get("std")],
            "h_rms_first_last": [(first_p.get("h_stage_h") or {}).get("rms"),
                                 (last_p.get("h_stage_h") or {}).get("rms")],
            "h_std_first_last": [(first_p.get("h_stage_h") or {}).get("std"),
                                 (last_p.get("h_stage_h") or {}).get("std")],
            "h_p99_over_rms_first_last": [(first_p.get("h_stage_h") or {}).get("p99_over_rms"),
                                          (last_p.get("h_stage_h") or {}).get("p99_over_rms")],
            "h_fid_std_ratio_mean": agg("h_fid_std_ratio"),
            "h_fid_maxdiff_max": max([r["h_fid_maxdiff"] for r in tel_rows
                                      if r.get("h_fid_maxdiff") is not None], default=None),
            "logits_std_first_last": [first_p.get("logits_std"), last_p.get("logits_std")],
            "note": "rms_norm 钉 RMS (~0.995) 不钉 std; std(h) 漂移 = 分布形状 (尖峰化) 证据"},
        "growth_diag": {
            "W_lm_init": init_norms.get("W_lm"),
            "W_lm_final": (norms_final[e124.PNAMES.index("W_lm")] if not smoke else None),
            "logits_std_first": first_p.get("logits_std"),
            "logits_std_last": last_p.get("logits_std"),
            "sm_maxp_first": first_p.get("sm_maxp"), "sm_maxp_last": last_p.get("sm_maxp"),
            "sm_entropy_first": first_p.get("sm_entropy"), "sm_entropy_last": last_p.get("sm_entropy"),
            "e0_mean_first": first_p.get("e0_mean"), "e0_mean_last": last_p.get("e0_mean"),
            "trust_scale_mean": agg("trust_scale"), "pin_scale_mean": agg("pin_scale"),
            "pin_target": _PIN["target"],
            "dW_cos_prev_mean": agg("dW_cos_prev"),
            "dt_mean_first100": dt_win(0, 100), "dt_mean_last100": dt_win(len(tel_rows) - 100, len(tel_rows) + 1)},
        "sweep": sweep,
        "entropy_bidirectional": {"first_H": ent_first, "last_H": ent_last,
                                  "min_H": ent_min, "max_H": ent_max,
                                  "updates": ent_updates,
                                  "decline_streak_end": decline_streak,
                                  "rise_streak_end": rise_streak,
                                  "pass_bidirectional": (decline_streak < ENT_STREAK
                                                         and rise_streak < ENT_STREAK)},
        "fnow_trajectory": {"f_now_perc_first": f_perc_first, "f_now_perc_last": f_perc_last,
                            "f_now_perc_max": f_perc_max, "f_now_eco_last": f_eco_last,
                            "E_last": E_last, "R_last": R_last},
        "drift_top10": drift[:10],
        "bias_isolation_check": {"bias_norm_init": init_norms.get("bias_lm"),
                                 "bias_norm_final": float(net.bias_lm.data.norm())},
        "secs": round(time.time() - t0, 1),
        "iron_rules": "crates/ 零改动 (进程内 monkeypatch); 不动 v244/迁移件 (只读归档检查点); "
                      "全 fp16 (无 clamp 数值; 统计全 .float() 只读路径); 权重不落盘; "
                      "run_file 自增新目录; stdout/stderr 落盘 run 目录; 运行不 kill; 三臂独立进程串行",
        "products": {"verdict": tag + ".json",
                     "telemetry": tag + "_telemetry.jsonl (首行 meta; 每行 norms 全数组 + "
                                   "f_now/E/R + 逐行范数/cap分解/h四级 + dW 余弦 + pin/trust)",
                     "samples_log": tag + ".log",
                     "stdout": tag + "_stdout.log",
                     "audit": (tag + "_audit_char.txt" if not smoke else None),
                     "sweep": ("内嵌 verdict.sweep" if not smoke else None)},
    }
    with open(run_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"exp125c[{arm}]: 写出 {run_json}", flush=True)
    print(f"exp125c[{arm}]: 判定摘要 abort={abort} bpb {pre['bpb_s1'] if pre else None}->{post_bpb} "
          f"top1={post['top1_s1'] if post else None} word_hits={wh} "
          f"layer1={layer1} layer2={layer2} learned={learned} c_type={c_type} crash={crash_type}",
          flush=True)
    if closure:
        print(f"exp125c[{arm}]: 矛盾1 重放闭合 n={closure['n']} max_rel_err={closure['max_rel_err']:.5f} "
              f"sign={closure['sign_agreement']}/{closure['n']}", flush=True)
    print(f"exp125c[{arm}]: DONE", flush=True)
    logf.close()
    jsonl.close()
    logf_console.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        a = "unknown"
        for v in sys.argv:
            if v in ARMS:
                a = v
        crash = resolve_path(f"out/exp125c_crash_{a}.txt")
        with open(crash, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        sys.stderr.write(traceback.format_exc())
        sys.exit(1)
