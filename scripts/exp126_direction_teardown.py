"""
exp126: 方向诊断 — 特征探针 (P) + 梯度余弦 (G) + 训练信号温度 (T)
=====================================================================

立项依据 (2026-10-20 审查裁决, 聊天记录): exp125c v2 后第一层 (尺度) 已过,
第二层 (方向) 不过 (top1 0.132/0.116/0.055); 批准 exp126 方向诊断 (两跑),
执行口径首句给 "P top1 (step 0/500) + G cos 均值 + T post bpb/top1/word_hits
+ 分流落支"。

预登记: docs/plan/exp126_preregistration.md (判读格预写, 跑前锁哈希;
本文件跑前写入, sha256 入账)。

─── 先行框架 (exp125c v2 升格) ───

训练/评测两条温度路径分离 (readout.py:90-99 lc vs _HeadEval raw): 训练误差
对 raw 尺度近似不变且饱和 — C 型第一性解释。T 臂直接检验"训练信号饱和是否
为方向天花板的原因"。

─── 设计 (与预登记逐字对应) ───

两跑 (全部 lr 0.05 + U 信任域 rho=0.25 — 审查定点: U 是唯一 e0 下降且
bpb (校准维度) 低于 pre 的合格起点; S 把 W_lm 钉到 0.30x 容量钉没):
  R1  U + 仪器: P 探针 (step 0/500, linear+MLP, 反传仅作仪器零写回) +
      G 梯度余弦 (每 100 步, g=∇W_lm CE(raw,无掩码,T=1) 于快照实例)。
  R2  U + T 臂: 训练误差改 raw logits (T=1) CE 替代 lcx60 (包装
      _build_lm_signal, 仅覆写 eps_total/eps_lm/eps_proj_raw/eps_proj_ms/
      eps_lm_proj; 保留打印掩码=结构约束; flag 默认关可回退; 单变量)。

仪器隔离 (保真构造): 所有仪器在快照加载的独立实例上运行 (net.save -> load
-> 测量 -> 删), 训练实例零额外前向 — R1 轨迹与 exp125c U 逐位相同的构造性
保证; 保真目标 post bpb = 6.988054104305917。

判据 (预写): P: top1_probe(500) >=0.5 高 / <=0.2 低 / 中间不单独触发;
G: cos 均值 >0.3 高 / |均值|<0.1 低; T: top1>0.25 或 bpb<5.0 过。
分流 (预写): T 过 -> 温度修复固化候选; T 不过+P 低 -> 表示层; T 不过+P 高
+G 低 -> 改更新规则; T 不过+P 高+G 高 -> 步长/振荡机制; P 中以 G/T 仲裁。

偏离披露:
  - g 在 fp32 下计算 (仪器精度; h/W 为 fp16 张量的精确提升; 零写回) —
    与 exp116 探针全 fp16 的差异: 探针保持 fp16 (先例), 梯度仪器用 fp32
    累积 (余弦对噪声敏感, 精度优先; 属测量路径不触训练张量)。
  - G 的 ΔW 用训练窗落库变化, g 用 200 固定评测窗 — 数据面不同但同为模型
    当前行为在语料分布上的采样; 固定窗保证跨区间可比。
  - T 臂的 eps 重算覆盖全部已核消费者 (readout E3 / feedforward W_35 /
    metabolism F / predict / bind / pool); 不覆写 eps_t2_total (W_lm_2
    冻结路径) 与 _lang_eps_ema (只被 action.py 恒温器读, 本路径不消费)。
  - 宿主 stdout 故障: 启动即重定向到 run 目录; 运行不 kill; .venv。

铁律: crates/ 零改动 (仅进程内 monkeypatch; exp127 是审查正式豁免的独立
项); 不动 v244/迁移件 (只读归档检查点, md5 硬门); 权重不落盘 (瞬态快照即
删); run_file 自增新目录。
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
import torch.nn as nn
import torch.nn.functional as F

import exp124_epsb_amp as e124
from exp115_say import N_EVAL_WIN, S_MAX, _sample_dump, build_gauge
from exp116_z4_probe import _md5, calib_s, infer, run_probe, score_logits

from crates.cli.utils import resolve_path, run_file
from crates.graphs import set_backend
set_backend("ref")
from crates.kernels import cubic, rms_norm, saturate
from crates.kernels.normalize import soft_norm_preserve
from crates.modal.text.geometry import ALPHABET
from crates.hybrid.learning.readout import ReadoutMixin
from crates.model.constants import F_UNIT_SCALE
from crates.model import CyreneModel, HybridPCNet
from crates.utils.codec import ByteDataset

torch.set_grad_enabled(False)

LR = 0.05
TARGET_STEPS = 500
ENT_WIN = 8192
ENT_MIN = 256
ENT_STREAK = 100
TRUST_RHO = 0.25
SWEEP_SCALES = [1.6, 1.3, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]
FIDELITY_TARGET = 6.988054104305917

_TEL = {"fresh": False}
_TEL_RAW = {}
_TARM = {"on": False}
_orig_build_lm_signal = ReadoutMixin._build_lm_signal


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


# ─────────── T 臂: _build_lm_signal 包装 (raw 温度训练误差; flag 默认关) ───────────

def _raw_temp_build_lm_signal(self, ctx):
    lm = _orig_build_lm_signal(self, ctx)
    if not _TARM["on"] or lm is None or ctx.free_run:
        return lm
    net = self.net
    d_h = net.d_h
    inv_h = 1.0 / math.sqrt(d_h)
    byte_ids = ctx.byte_ids
    mask = (1.0 - net._mask_print) * -1e4

    raw1 = (lm.h @ net.W_lm + net.bias_lm) * inv_h + mask
    raw2 = (lm.h2 @ net.W_lm_2 + net.bias_lm) * inv_h + mask
    probs1 = torch.softmax(raw1, dim=-1)
    probs2 = torch.softmax(raw2, dim=-1)

    target_lm = F.one_hot(byte_ids[:, 1:], num_classes=ALPHABET).to(torch.float16)
    target_lm2 = F.one_hot(byte_ids[:, 2:], num_classes=ALPHABET).to(torch.float16)
    lm_mask = ctx.learn_mask.to(torch.float16).unsqueeze(-1)
    eps_lm = (target_lm - probs1[:, :-1]).detach() * lm_mask
    diff2 = (target_lm2 - target_lm[:, :-1]) - (probs2 - probs1[:, :-2])
    diff2 = diff2 * (ctx.learn_mask[:, :-1] & ctx.learn_mask[:, 1:]).to(torch.float16).unsqueeze(-1)
    diff2 = torch.cat([diff2, net._padmax[:, :, :256]], dim=1)
    eps_total = (eps_lm + 0.2 * diff2).detach()

    z4 = net._z4
    dz4 = (z4[:, 1:] - z4[:, :-1]).norm(dim=-1, keepdim=True)
    dz4_sig = dz4 / (dz4.max() + 1e-3)
    err_mag = eps_total.norm(dim=-1, keepdim=True)
    err_ref = err_mag.mean() + 1e-3
    gain = err_mag / (0.5 * err_ref + 0.5 * err_mag)
    eps_in = eps_total / 64.0
    eps_proj_raw = (eps_in @ net.W_lm.T @ net.W1.T)[:, :, :ctx.dim_4] * dz4_sig * gain
    eps_proj_ms = eps_proj_raw.square().mean() * F_UNIT_SCALE
    eps_lm_proj = rms_norm(eps_proj_raw)

    _TEL_RAW["raw_logits_std"] = float(raw1.detach().float().std())
    _TEL_RAW["raw_sm_maxp"] = float(probs1.max(dim=-1).values.detach().float().mean())
    _TEL_RAW["raw_eps_norm"] = float(eps_total.detach().float().norm(dim=-1).mean())
    _TEL_RAW["lc_eps_norm"] = float(lm.eps_total.detach().float().norm(dim=-1).mean())

    lm.eps_total = eps_total
    lm.eps_lm = eps_lm
    lm.eps_proj_raw = eps_proj_raw
    lm.eps_proj_ms = eps_proj_ms
    lm.eps_lm_proj = eps_lm_proj
    return lm


# ─────────── E3 复刻 + 只读仪表 + U 信任域 (两跑共用) ───────────

def _update_lm_head_e3u_tel(self, ctx, lm):
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

    trust_scale = 1.0
    if step_norm_pre > TRUST_RHO * wpre_norm_f:
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


# ─────────── P 探针 (仪器: 反传仅测量, 零写回; exp116/exp118 先例) ───────────

def probe_point(snap_path, cfg, dev, gauge, tag, epochs=30, quick=False):
    fresh = HybridPCNet.load(snap_path, cfg).to(dev)
    fresh.eval()
    fresh._migrate_expand = False
    n_tr = 16 if quick else 1000
    n_ca = 16 if quick else 200
    tr_wins = gauge["tail"][N_EVAL_WIN:N_EVAL_WIN + n_tr]
    te_wins = gauge["tail"][:N_EVAL_WIN]
    ca_wins = gauge["tail"][N_EVAL_WIN + n_tr:N_EVAL_WIN + n_tr + n_ca]
    Xtr, ytr = e124.extract_h(fresh, dev, tr_wins, tag + "_tr")
    Xte, yte = e124.extract_h(fresh, dev, te_wins, tag + "_te")
    Xca, yca = e124.extract_h(fresh, dev, ca_wins, tag + "_ca")
    res = {"n_train_windows": len(tr_wins), "n_test_windows": len(te_wins)}
    ep = 2 if quick else epochs
    Xtr_g = Xtr.to(dev, torch.float16)
    ytr_g = ytr.to(dev)
    for kind in ("linear", "mlp"):
        model = (nn.Linear(256, 256) if kind == "linear" else
                 nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256))).half()
        run_probe(model, Xtr_g, ytr_g, dev, ep, 2e-3, 8192, 0)
        lte = infer(model, Xte, dev, 8192)
        lca = infer(model, Xca, dev, 8192)
        s_star, _ = calib_s(lca, yca)
        te_s1 = score_logits(lte, yte, scale=1.0)
        te_ss = score_logits(lte, yte, scale=s_star)
        res[kind] = {"top1_s1": te_s1["top1"], "bpb_s1": te_s1["bpb"],
                     "top15_s1": te_s1["top15"], "s_star": s_star,
                     "bpb_s_star": te_ss["bpb"], "top1_s_star": te_ss["top1"]}
        del model
        torch.cuda.empty_cache()
    del fresh
    torch.cuda.empty_cache()
    return res


# ─────────── G 梯度余弦 (仪器: 叶克隆 fp32 反传, 零写回) ───────────

def grad_measure(snap_path, cfg, dev, eval_wins):
    fresh = HybridPCNet.load(snap_path, cfg).to(dev)
    fresh.eval()
    fresh._migrate_expand = False
    h, y = e124.extract_h(fresh, dev, eval_wins, "grad")
    h = h.to(dev, torch.float16).float()
    y = y.to(dev)
    W_leaf = fresh.W_lm.detach().float().clone().requires_grad_(True)
    with torch.enable_grad():
        logits = (h @ W_leaf + fresh.bias_lm.detach().float()) * (1.0 / math.sqrt(fresh.d_h))
        loss = F.cross_entropy(logits, y)
        loss.backward()
    g = W_leaf.grad.detach()
    del fresh, W_leaf
    torch.cuda.empty_cache()
    return g, float(loss), float(g.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=["R1", "R2"])
    ap.add_argument("--t", default="auto", choices=["on", "off", "auto"],
                    help="T 臂开关 (auto: R2=on, R1=off)")
    ap.add_argument("--steps", type=int, default=TARGET_STEPS)
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0,
                    help="冒烟步数 (0=正式; 冒烟跳过 eval/dump/audit, 保留迷你仪器验证)")
    args = ap.parse_args()
    run = args.run
    t_on = (args.t == "on") or (args.t == "auto" and run == "R2")
    _TARM["on"] = t_on
    tag = f"exp126_{run}"
    smoke = args.smoke > 0
    steps = args.smoke if smoke else args.steps

    stdout_path = run_file(tag + "_stdout.log")
    logf_console = open(stdout_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = logf_console
    sys.stderr = logf_console

    ck = resolve_path(e124.CKPT)
    md5_now = _md5(ck)
    if md5_now != e124.CKPT_MD5:
        print(f"exp126: FATAL md5 硬门不一致: 实测 {md5_now} != 预期 {e124.CKPT_MD5}; 拒跑", flush=True)
        sys.exit(2)
    dep = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp124_epsb_amp.py")
    dep_sha = hashlib.sha256(open(dep, "rb").read()).hexdigest()
    self_sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    print(f"exp126: run={run} lr={LR} trust_rho={TRUST_RHO} T_arm={t_on} smoke={smoke} steps={steps}\n"
          f"exp126: ckpt={ck}\nexp126: md5={md5_now} (硬门 PASS)\n"
          f"exp126: 依赖 exp124_epsb_amp.py sha256={dep_sha}\n"
          f"exp126: 执行件 sha256={self_sha}", flush=True)

    if t_on:
        ReadoutMixin._build_lm_signal = _raw_temp_build_lm_signal
        print("exp126: monkeypatch: T 臂 ON (训练误差 raw T=1 CE, 覆写 eps_total/eps_lm/"
              "eps_proj_raw/eps_proj_ms/eps_lm_proj; 掩码保留; flag 可回退)", flush=True)
    else:
        print("exp126: monkeypatch: T 臂 OFF (lc 训练路径原样)", flush=True)
    ReadoutMixin._update_lm_head = _update_lm_head_e3u_tel
    print(f"exp126: monkeypatch: E3 复刻+仪表 + U 信任域 rho={TRUST_RHO}", flush=True)

    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=S_MAX, lm_freeze_w1=False)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = torch.device(args.device)
    net = HybridPCNet.load(ck, cfg).to(dev)
    net.eval()
    net._migrate_expand = False
    print(f"exp126: net loaded; d_h={net.d_h} mem_k={net._mem_m.shape[0]}; _migrate_expand=False", flush=True)

    snap0 = run_file(tag + "_snapW0.safetensors")
    net.save(snap0)
    print(f"exp126: step-0 快照 (pristine) {snap0}", flush=True)

    gauge, _, _ = build_gauge(args.data, dev)
    eval_wins = gauge["tail"][:N_EVAL_WIN]
    print(f"exp126: eval_wins={len(eval_wins)}", flush=True)

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
            "exp": "exp126", "run": run, "lr": LR, "trust_rho": TRUST_RHO,
            "T_arm": t_on, "smoke": smoke,
            "ckpt": e124.CKPT, "ckpt_md5": md5_now, "ckpt_md5_expected": e124.CKPT_MD5,
            "dep_exp124_sha256": dep_sha, "self_sha256": self_sha,
            "unique_variable": ("R2: 训练误差温度 (lcx60 -> raw T=1); R1: 无 (U 基线+仪器)"
                                if run == "R2" else "无 (U 基线+仪器; 与 exp125c U 同规则)"),
            "rule": "E3 复刻+只读仪表 + U 信任域 rho=0.25" + (" + T 臂 raw 温度" if t_on else ""),
            "fidelity_gate": {"R1_post_bpb": FIDELITY_TARGET,
                              "note": "R1 须与 exp125c U 逐位相同 (仪器隔离构造保证)"},
            "verdict_rule": "P: top1_probe(500)>=0.5 高 /<=0.2 低; G: cos 均值>0.3 高 /|<0.1| 低; "
                            "T: top1>0.25 或 bpb<5.0 过"}
    jsonl.write(json.dumps({"meta": meta}, ensure_ascii=True) + "\n")

    t0 = time.time()
    if smoke:
        pre = post = None
        base_lines = []
        base_hits = None
        print("exp126: 冒烟模式 — 跳过 pre/post eval / dump / audit (保留迷你仪器验证)", flush=True)
    else:
        pre = e124.eval_bpb_fresh(net, dev, eval_wins)
        print(f"exp126[{run}]: pre bpb@s1={pre['bpb_s1']:.6f} top1={pre['top1_s1']:.6f} "
              f"(exp125 C pre=7.664317404722619; 须逐位相同=协议保真)", flush=True)
        base_lines = _sample_dump(net, tag, cfg, dev, 0, 3, 240, 0.8, logf)
        base_hits = e124.word_hits_of(base_lines)

    ds = ByteDataset(args.data, max_length=S_MAX, max_samples=1270000)
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    print(f"exp126[{run}]: data n={len(ds)}", flush=True)

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
    W_series = {}
    snaps = {}

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
        if tel:
            for k in ("raw_logits_std", "raw_sm_maxp", "raw_eps_norm", "lc_eps_norm"):
                tel[k] = _TEL_RAW.get(k)
            _TEL_RAW.clear()
        row = {"step": i, "lr": LR, "run": run, "behavior": int(exec_b),
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
                  "wlm_top5_rows", "h_stage_y", "h_stage_y_saturate", "h_stage_y_norm",
                  "h_stage_h", "h_fid_std_ratio", "h_fid_maxdiff", "e0_mean", "alive_frac",
                  "raw_logits_std", "raw_sm_maxp", "raw_eps_norm", "lc_eps_norm"):
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
            W_series[i] = net.W_lm.data.clone()
            if not smoke:
                sp = run_file(tag + f"_snapW{i}.safetensors")
                net.save(sp)
                snaps[i] = sp
            print(f"exp126[{run}]: step {i}/{steps} f_perc={f_perc_last:.4f} "
                  f"logits_std={tel.get('logits_std')} raw_std={tel.get('raw_logits_std')} "
                  f"raw_maxp={tel.get('raw_sm_maxp')} trust={tel.get('trust_scale')} "
                  f"W_lm={norms[e124.PNAMES.index('W_lm')]:.4f}", flush=True)
        if death_cause:
            print(f"exp126[{run}]: 机体自维护 death_cause={death_cause} (照常运行, 不中止)", flush=True)
            death_cause = None

    sweep = None
    probe_res = None
    grad_res = None
    if smoke:
        final_lines = []
        wh = None
        print(f"exp126[{run}]: 冒烟注入自检 — step1 dWlm_absmean={_gt('_tel_dWlm_t')} "
              f"W_lm_norm={_gt('_tel_Wlm_norm_t')} logits_std={tel_rows[0].get('logits_std')} "
              f"raw_std={tel_rows[0].get('raw_logits_std')}", flush=True)
        sweep = post_sweeps(net, dev, eval_wins, tag + "_sweep")
        probe_res = probe_point(snap0, cfg, dev, gauge, tag + "_probe0", quick=True)
        g, ce, gn = grad_measure(snap0, cfg, dev, eval_wins)
        grad_res = {"smoke": True, "ce": ce, "g_norm": gn}
        print(f"exp126[{run}]: 冒烟仪器验证 — probe_linear_top1={probe_res['linear']['top1_s1']:.4f} "
              f"grad_ce={ce:.4f} g_norm={gn:.2f} (冒烟值不作判读)", flush=True)
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
        print(f"exp126[{run}]: post bpb@s1={post['bpb_s1']:.6f} top1={post['top1_s1']:.6f} "
              f"(R1 保真目标 {FIDELITY_TARGET})", flush=True)
        wh = e124.word_hits_of(final_lines)
        sweep = post_sweeps(net, dev, eval_wins, tag + "_sweep")
        print(f"exp126[{run}]: 后置扫描 best scale={sweep['best']['scale']} "
              f"(T={sweep['best']['temp']}) bpb={sweep['best']['bpb']:.4f} vs s1={sweep['bpb_s1']:.4f} "
              f"gain={sweep['gain_ratio']:.3f} bias_off_s1={sweep['bias_off_s1']['bpb']:.4f}", flush=True)

        # ── P 探针 (step 0 / step 500 快照) ──
        try:
            p0 = probe_point(snap0, cfg, dev, gauge, tag + "_probe0")
            p5 = probe_point(snaps.get(steps, snap0), cfg, dev, gauge, tag + "_probe500")
            probe_res = {"step0": p0, "step500": p5}
            print(f"exp126[{run}]: P 探针 step0 linear_top1={p0['linear']['top1_s1']:.4f} "
                  f"mlp_top1={p0['mlp']['top1_s1']:.4f} | step500 linear_top1={p5['linear']['top1_s1']:.4f} "
                  f"mlp_top1={p5['mlp']['top1_s1']:.4f}", flush=True)
        except Exception as e:  # noqa: BLE001
            probe_res = {"instrument_fail": repr(e)}
            print(f"exp126[{run}]: WARN P 探针失败 (不中止): {e!r}", flush=True)

        # ── G 梯度余弦 (每 100 步区间) ──
        try:
            gcache = {}
            intervals = []
            pts = sorted(k for k in W_series if k > 0)
            for k in pts:
                g, ce, gn = grad_measure(snaps[k], cfg, dev, eval_wins)
                gcache[k] = g
            for idx in range(1, len(pts)):
                k_hi, k_lo = pts[idx], pts[idx - 1]
                dW = (W_series[k_hi].float() - W_series[k_lo].float()).flatten()
                g = gcache[k_hi]
                cosv = float((dW @ (-g).flatten()) / (dW.norm() * g.norm() + 1e-12))
                intervals.append({"interval": [k_lo, k_hi], "cos_dW_negG": cosv,
                                  "dW_norm": float(dW.norm()), "g_norm": float(g.norm())})
                del gcache[k_hi]
                torch.cuda.empty_cache()
            cos_vals = [iv["cos_dW_negG"] for iv in intervals]
            grad_res = {"intervals": intervals,
                        "cos_mean": (sum(cos_vals) / len(cos_vals)) if cos_vals else None}
            print(f"exp126[{run}]: G 梯度余弦 mean={grad_res['cos_mean']:.4f} "
                  f"intervals={[(iv['interval'], round(iv['cos_dW_negG'], 3)) for iv in intervals]}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            grad_res = {"instrument_fail": repr(e)}
            print(f"exp126[{run}]: WARN G 梯度测量失败 (不中止): {e!r}", flush=True)

        for k, sp in snaps.items():
            try:
                os.remove(sp)
            except OSError:
                pass
        try:
            os.remove(snap0)
        except OSError:
            pass

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

    perc = [r for r in tel_rows if r.get("logits_std") is not None]
    first_p, last_p = (perc[0] if perc else {}), (perc[-1] if perc else {})

    if smoke:
        post_bpb = None
        p_high = p_low = g_high = g_low = t_pass = None
        learned = c_type = crash_type = stable = None
        layer1 = layer2 = None
        t1l = t1p = t1 = cm = None
    else:
        post_bpb = post["bpb_s1"]
        top1_post = post["top1_s1"]
        learned = post_bpb < 5.0
        stable = post_bpb < 8.0
        c_type = (post_bpb >= 8.0) and (top1_post >= 0.05) and ((f_perc_max or 0.0) < 100.0)
        crash_type = (post_bpb >= 8.0) and not c_type
        layer1 = sweep["best"]["bpb"] < 8.0
        layer2 = top1_post > 0.25
        t_pass = layer2 or learned
        if isinstance(probe_res, dict) and "step500" in probe_res:
            t1p = probe_res["step500"]["mlp"]["top1_s1"]
            t1l = probe_res["step500"]["linear"]["top1_s1"]
            t1 = max(t1p, t1l)
            p_high = t1 >= 0.5
            p_low = t1 <= 0.2
        else:
            t1p = t1l = t1 = None
        if isinstance(grad_res, dict) and grad_res.get("cos_mean") is not None:
            cm = grad_res["cos_mean"]
            g_high = cm > 0.3
            g_low = abs(cm) < 0.1
        else:
            cm = None

    out = {
        "exp": "exp126", "run": run, "lr": LR, "trust_rho": TRUST_RHO,
        "T_arm": t_on, "smoke": smoke,
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
        "P_feature_probe": probe_res,
        "G_gradient_cos": grad_res,
        "T_criteria": {"pass": t_pass, "rule": "top1>0.25 或 bpb<5.0",
                       "top1_post": (post["top1_s1"] if post else None),
                       "post_bpb": post_bpb},
        "triage_flags": {"P_high": p_high, "P_low": p_low, "G_high": g_high, "G_low": g_low,
                         "probe_top1_step500_linear": t1l, "probe_top1_step500_mlp": t1p,
                         "probe_top1_step500_max": t1, "grad_cos_mean": cm,
                         "note": "分流按预登记 §四 (跨跑合读)"},
        "failure_mode": {"learned": learned, "stable": stable,
                         "c_type_calibration": c_type, "crash_type": crash_type},
        "growth_diag": {
            "W_lm_init": init_norms.get("W_lm"),
            "W_lm_final": (norms_final[e124.PNAMES.index("W_lm")] if not smoke else None),
            "logits_std_first": first_p.get("logits_std"),
            "logits_std_last": last_p.get("logits_std"),
            "raw_logits_std_first": first_p.get("raw_logits_std"),
            "raw_logits_std_last": last_p.get("raw_logits_std"),
            "raw_sm_maxp_first": first_p.get("raw_sm_maxp"),
            "raw_sm_maxp_last": last_p.get("raw_sm_maxp"),
            "raw_eps_norm_first": first_p.get("raw_eps_norm"),
            "raw_eps_norm_last": last_p.get("raw_eps_norm"),
            "lc_eps_norm_first": first_p.get("lc_eps_norm"),
            "lc_eps_norm_last": last_p.get("lc_eps_norm"),
            "sm_maxp_first": first_p.get("sm_maxp"), "sm_maxp_last": last_p.get("sm_maxp"),
            "e0_mean_first": first_p.get("e0_mean"), "e0_mean_last": last_p.get("e0_mean"),
            "trust_scale_mean": agg("trust_scale"),
            "dW_cos_prev_mean": agg("dW_cos_prev"),
            "cap10_binding_frac": (sum(1 for r in tel_rows if (r.get("wlm_cap_scale") or 1.0) < 1.0)
                                   / max(1, len([r for r in tel_rows if r.get("wlm_cap_scale") is not None])))},
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
                      "全 fp16 (无 clamp 数值; 统计 .float() 只读路径; 探针 fp16 先例, 梯度仪器 "
                      "fp32 零写回); 权重不落盘 (瞬态快照即删); run_file 自增新目录; "
                      "stdout/stderr 落盘 run 目录; 运行不 kill; R1/R2 独立进程串行",
        "products": {"verdict": tag + ".json",
                     "telemetry": tag + "_telemetry.jsonl",
                     "samples_log": tag + ".log",
                     "stdout": tag + "_stdout.log",
                     "audit": (tag + "_audit_char.txt" if not smoke else None),
                     "sweep": ("内嵌 verdict.sweep" if not smoke else None)},
    }
    with open(run_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"exp126[{run}]: 写出 {run_json}", flush=True)
    print(f"exp126[{run}]: 判定摘要 abort={abort} bpb {pre['bpb_s1'] if pre else None}->{post_bpb} "
          f"top1={post['top1_s1'] if post else None} word_hits={wh} "
          f"layer1={layer1} layer2={layer2} T_pass={t_pass} "
          f"P(max t1)={t1 if not smoke else None} G(cos)={cm if not smoke else None}", flush=True)
    print(f"exp126[{run}]: DONE", flush=True)
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
            if v in ("R1", "R2"):
                a = v
        crash = resolve_path(f"out/exp126_crash_{a}.txt")
        with open(crash, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        sys.stderr.write(traceback.format_exc())
        sys.exit(1)
