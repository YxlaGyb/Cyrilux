"""
exp130 B 段: coherence 判定 — 仪器化重跑, 补 exp129 未落盘的 ‖ΣdW_raw‖ / ‖Σβr‖ / ‖ΣγW‖
========================================================================================

本件 = scripts/exp129_direction_stability.py **逐字派生件** (见 §派生说明), 只增 exp130
专属仪器与输出, 训练路径 / 更新规则 / dump 日程 / 保真门 **一个字节不改**。

立项 (exp129 审查裁决, 2026-10-22):
  exp129 把 "G 低" 归到四条实测成因链, 但 "振荡" 只是**描述** (相邻更新反相关), 不是
  **机制**。裁决的数学事实: 相邻更新若只是随机反相关, 累积余弦应**升高**
  (cos(ΣdW,Σu)=a·sqrt(N)/sqrt(a²N+b²) → 1); 实测每步 0.78 → 累积 0.14, 必然存在
  **跨步相关的系统分量**。两种候选:
    A. 梯度旋转: h 每窗不同 → u_i = −g_i 大幅变 → Σu 自身很小 → 参考本身塌。→ 批 K 窗。
    B. 系统漂移: dW 有跨步同向的非梯度分量 (cap 收缩 / 偏置 / rms_norm)。→ 识别并扣除。
  判定量 coherence(x) = ‖Σx‖ / Σ‖x‖ (随机游走 ≈ 1/sqrt(N); 系统漂移 ≫ 1/sqrt(N))。

为什么必须点火 B 段 (A 段可用性裁定, 见 scripts/exp130_coherence_offline.py):
  exp129 落盘的向量和范数只有 ‖Σ(−g)‖ 与 ‖ΣdW_eff‖ 两个; **‖ΣdW_raw‖ 未落盘**,
  且 jsonl 只有 cos (抹掉模长) 而无 running 范数量可联立 ⇒ 不可离线恢复。
  A 段已精确得到: coherence_g = 0.200/0.211/0.256/0.201/0.227 (RA_g 1.38–1.72 ≈ 1/√N 量级
  ⇒ 梯度确实大幅旋转); coherence_eff = 0.079/0.057/0.052/0.051/0.052 (RA_eff 0.35–0.54
  **低于随机游走基线**); eff/g 比 0.20–0.39。**缺的正是 Q2/Q3 要的 ‖ΣdW_raw‖。**

B 段新增仪器 (全部 fp32 只读零写回, 不触碰任何训练状态):
  1. 每段/全程 累加三个向量和 ΣdW_raw / ΣdW_eff / Σ(−g) **并写出其范数** ← 核心缺口补齐
  2. 逐步 cos(−g_i, −g_{i-1}) = **cos_grad_osc**: 梯度自身的相邻旋转 —— A/B 最干净的判据,
     exp129 完全没有测。
  3. 逐步解析分解 dW_eff = β·dW_raw + γ·W_pre (β=cap·lr·trust, γ=cap−1; trust 标量不旋转,
     cap10 是作用于整个 W_lm 的全局标量), 分别累加 U=Σβr 与 T=ΣγW_pre 并写出 ‖U‖·‖T‖
     ⇒ 把 "次级 cap 收缩" 从估计升级为**实测**, 并给出 Q3 的点估计 (A 段只能给严格区间)。
  4. 每段/全程 Σ‖x‖ 精确累加 ⇒ coherence 分母不再是近似。
  5. cos(ΣγW_pre, Σ(−g)): 漂移方向与梯度方向的失配角 —— Q4 的定向版本。

自检 (四重, 全部跑后自动核对):
  (a) md5 硬门; (b) post bpb@s1 逐位 = 6.988054104305917;
  (c) 逐步 dW_cos_prev 与 v451 遥测五段逐位一致 (bit_identical) —— 证明新增仪器零扰动;
  (d) 本件复算的 ΣdW_eff / Σ(−g) 相关段量必须与 exp129 v460 同段一致 (仪器自洽交叉核对)。

铁律: crates/ 零改动 (进程内 monkeypatch); 不动 v244/迁移件 (md5 硬门); 全 fp16 训练
(每步 −g 仪器 fp32 叶克隆零写回); 权重不落盘 (瞬态快照即删); run_file 自增新目录;
运行不 kill; device 入账 (v457 教训: 历代 GPU)。

派生说明: 由 exp129 执行件复制后, 仅在下述位置改动 — ① 本文件头; ② _fresh_seg/EXP129
累加器; ③ 仪器 A 块 (cos_grad_osc + ‖·‖ 累加); ④ 仪器 C 块 (β/γ 分解累加);
⑤ segment_verdict 新增键; ⑥ jsonl row 新增键; ⑦ tag / 末段输出。训练路径零改动。
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
from exp116_z4_probe import _md5
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
TRUST_RHO = 0.25
ENT_WIN = 8192
ENT_MIN = 256
ENT_STREAK = 100
FIDELITY_TARGET = 6.988054104305917  # exp126 R1 / exp125c U post bpb@s1
V451_TEL = "out/v451-20261020-194546/exp126_R1_telemetry.jsonl"  # 逐步余弦交叉核对源

_TARM = {"on": False}   # R1 = T 臂 OFF (部署 lc 训练路径)
_orig_build_lm_signal = ReadoutMixin._build_lm_signal
_TEL = {"fresh": False}
_TEL_RAW = {}

# ─────────── exp129 仪器状态 (每步累加, 段末由主循环读取后重置) ───────────

def _fresh_seg():
    """exp129 原字段 + exp130 新增 accumulator。

    sum_betar / sum_rad : 解析分解 dW_eff = β·dW_raw + γ·W_pre 的两部分向量和。
    sn_*                : 逐步范数之和, 作为 coherence 的精确分母。
    """
    return {"sum_dw_eff": None, "sum_neg_g": None, "sum_dw_raw": None,
            "last_neg_g": None, "rows": [],
            "sum_betar": None, "sum_rad": None,
            "sn_r": 0.0, "sn_e": 0.0, "sn_g": 0.0, "sn_betar": 0.0, "sn_rad": 0.0}


EXP129 = {"seg": _fresh_seg(),
          "all": {"sum_dw_eff": None, "sum_neg_g": None, "sum_dw_raw": None,
                  "sum_betar": None, "sum_rad": None,
                  "sn_r": 0.0, "sn_e": 0.0, "sn_g": 0.0, "sn_betar": 0.0, "sn_rad": 0.0},
          "prev_dw_eff": None, "prev_dw_at": None, "last_row": None,
          "prev_neg_g": None}   # exp130: 用于逐步 cos(−g_i, −g_{i−1})


def _cos(a, b):
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def _neg_g_current(net, lm, byte_ids, inv_h):
    """同窗同 (更新前) 权重的真梯度 −g: autograd CE(调制 logits_lm, target) w.r.t.
    W_lm 的负梯度; fp32 叶克隆, 只读零写回; 与 exp128 _train_window_ctl 同式。"""
    h_tr = lm.h[:, :-1].detach().to(torch.float16).float()      # [N,S-1,d_h]
    y_tr = byte_ids[:, 1:].reshape(-1).long()
    bias_f = net.bias_lm.detach().float()
    mask_f = (1.0 - net._mask_print.detach().float()) * -1e4
    W_leaf = net.W_lm.detach().float().clone().requires_grad_(True)
    with torch.enable_grad():
        raw = (h_tr @ W_leaf + bias_f) * inv_h
        lc = (raw - raw.mean(-1, keepdim=True)) / (raw.std(-1, keepdim=True) + 1e-4)
        ll = lc / lc.abs().max(-1, keepdim=True).values * 60.0 + mask_f
        loss = F.cross_entropy(ll.reshape(-1, 256), y_tr)
        loss.backward()
    g = W_leaf.grad.detach()
    del W_leaf
    return (-g).float()


# ─────────── E3 复刻 + U 信任域 + exp126 只读仪表 (逐字复用) + exp129 仪器 ───────────

def _update_lm_head_e3u_tel(self, ctx, lm):
    net = self.net
    EXP129["last_row"] = None   # 早退步不得残留上一步仪器行 (jsonl 防张冠李戴)
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

    # ── exp129 仪器 A: dW_raw (trust 前提案) + −g_current (同窗同更新前权重) ──
    e129 = {}
    try:
        dW_raw = dW_lm.detach().clone()
        neg_g = _neg_g_current(net, lm, byte_ids, inv_h)
        e129["cos_rule"] = _cos(dW_raw.flatten(), neg_g.flatten())
        e129["dW_raw_norm"] = float(dW_raw.float().norm())
        e129["neg_g_norm"] = float(neg_g.norm())
        e129["mask_mean"] = float(lm_update_mask.detach().float().mean())
        # ── exp130 新增 (1): 梯度自身相邻旋转 —— A/B 最干净的判据, exp129 未测 ──
        png = EXP129["prev_neg_g"]
        if png is not None:
            e129["cos_grad_osc"] = _cos(neg_g.flatten(), png.flatten())
        EXP129["prev_neg_g"] = neg_g.detach().clone()
        # ── exp130 新增 (2): Σ‖·‖ 精确累加 (coherence 的分母) ──
        seg = EXP129["seg"]
        allc = EXP129["all"]
        seg["sn_g"] += float(neg_g.norm())
        allc["sn_g"] += float(neg_g.norm())
        seg["sn_r"] += float(dW_raw.float().norm())
        allc["sn_r"] += float(dW_raw.float().norm())
        if seg["sum_dw_raw"] is None:
            seg["sum_dw_raw"] = dW_raw.float().clone()
            seg["sum_neg_g"] = neg_g.clone()
        else:
            seg["sum_dw_raw"] += dW_raw.float()
            seg["sum_neg_g"] += neg_g
        if allc["sum_dw_raw"] is None:
            allc["sum_dw_raw"] = dW_raw.float().clone()
            allc["sum_neg_g"] = neg_g.clone()
        else:
            allc["sum_dw_raw"] += dW_raw.float()
            allc["sum_neg_g"] += neg_g
        EXP129["_cur"] = {"dW_raw": dW_raw, "neg_g": neg_g}
    except Exception as ex:  # noqa: BLE001
        import traceback as _tb
        e129["errs"] = repr(ex)
        e129["tb"] = _tb.format_exc()
        EXP129["_cur"] = None

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

    # ── exp129 仪器 B: dW_after_trust (trust 后提案) ──
    dW_after_trust = dW_lm.detach().clone()

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

    # ── exp129 仪器 C: dW_eff (落库有效变化) + 逐步/累积 cos ──
    cur = EXP129.get("_cur")
    if cur is not None:
        try:
            dW_eff = (net.W_lm.data.float() - wlm_pre.float())
            neg_g = cur["neg_g"]
            dW_raw = cur["dW_raw"]
            e129["cos_eff"] = _cos(dW_eff.flatten(), neg_g.flatten())
            e129["cap_rotation"] = _cos(dW_raw.flatten(), dW_eff.flatten())
            e129["trust_rotation"] = _cos(dW_raw.flatten(), dW_after_trust.flatten())
            if EXP129["prev_dw_eff"] is not None:
                e129["cos_osc"] = _cos(dW_eff.flatten(), EXP129["prev_dw_eff"].flatten())
            if EXP129["prev_dw_at"] is not None:
                e129["cos_osc_raw"] = _cos(dW_after_trust.flatten(),
                                           EXP129["prev_dw_at"].flatten())
            e129["dW_eff_norm"] = float(dW_eff.norm())
            # ── exp130 新增 (3): 解析分解 dW_eff = β·dW_raw + γ·W_pre ──
            #   trust 是标量缩放 (不旋转) ; cap10 是作用于整个 W_lm 的全局标量 c。
            #   W_after = c·(W_pre + lr·t·dW_raw) ⇒ dW_eff = β·dW_raw + γ·W_pre,
            #   β = c·lr·t, γ = c − 1。把两部分分别累加 ⇒ Q3 从估计升级为实测。
            beta = cap_scale * LR * trust_scale
            gamma = cap_scale - 1.0
            e129["beta"] = beta
            e129["gamma"] = gamma
            e129["Wpre_norm"] = float(wlm_pre.float().norm())
            betar_v = dW_raw.float() * beta
            rad_v = wlm_pre.float() * gamma
            seg = EXP129["seg"]
            allc = EXP129["all"]
            for acc in (seg, allc):
                acc["sn_e"] += float(dW_eff.norm())
                acc["sn_betar"] += beta * float(dW_raw.float().norm())
                acc["sn_rad"] += abs(gamma) * float(wlm_pre.float().norm())
                if acc["sum_betar"] is None:
                    acc["sum_betar"] = betar_v.clone()
                    acc["sum_rad"] = rad_v.clone()
                else:
                    acc["sum_betar"] += betar_v
                    acc["sum_rad"] += rad_v
            if seg["sum_dw_eff"] is None:
                seg["sum_dw_eff"] = dW_eff.clone()
            else:
                seg["sum_dw_eff"] += dW_eff
            if allc["sum_dw_eff"] is None:
                allc["sum_dw_eff"] = dW_eff.clone()
            else:
                allc["sum_dw_eff"] += dW_eff
            seg["last_neg_g"] = neg_g.clone()
            EXP129["prev_dw_eff"] = dW_eff.clone()
            EXP129["prev_dw_at"] = dW_after_trust.clone()
            # 累积 running 值 (from-start)
            if allc["sum_neg_g"] is not None:
                e129["cos_cum_g_cum_run"] = _cos(allc["sum_dw_eff"].flatten(),
                                                 allc["sum_neg_g"].flatten())
                e129["cos_cum_g_end_run"] = _cos(allc["sum_dw_eff"].flatten(),
                                                 neg_g.flatten())
                e129["cos_cum_raw_g_cum_run"] = _cos(allc["sum_dw_raw"].flatten(),
                                                     allc["sum_neg_g"].flatten())
        except Exception as ex:  # noqa: BLE001
            import traceback as _tb
            e129["errs_eff"] = repr(ex)
            e129["tb_eff"] = _tb.format_exc()

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

    # ── exp126 只读仪表 (逐字复用; dW_cos_prev 为保真交叉核对量) ──
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
        "e0_mean": float(lm.eps_total.detach().float().norm(dim=-1).mean()),
        "alive_frac": float(alive.detach().float().mean()),
    })

    EXP129["seg"]["rows"].append(e129)
    EXP129["last_row"] = e129
    EXP129["_cur"] = None

    if hasattr(net, "_novelty"):
        del net._novelty
    return d_t


# ─────────── exp126 仪表辅助 (逐字复用) ───────────

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


# ─────────── 段末评测窗 −g (瞬态快照独立实例; exp128 同式) ───────────

def eval_batch_neg_g(snap_path, cfg, dev, eval_wins):
    fresh = HybridPCNet.load(snap_path, cfg).to(dev)
    fresh.eval()
    fresh._migrate_expand = False
    h, y = e124.extract_h(fresh, dev, eval_wins, "grad")
    h = h.to(dev, torch.float16).float()
    y = y.to(dev)
    W_leaf = fresh.W_lm.detach().float().clone().requires_grad_(True)
    bias = fresh.bias_lm.detach().float()
    inv_h = 1.0 / math.sqrt(fresh.d_h)
    with torch.enable_grad():
        raw = (h @ W_leaf + bias) * inv_h
        logits_c = (raw - raw.mean(-1, keepdim=True)) / (raw.std(-1, keepdim=True) + 1e-4)
        logits_lm = logits_c / logits_c.abs().max(-1, keepdim=True).values * 60.0
        logits_lm = logits_lm + (1.0 - fresh._mask_print.detach().float()) * -1e4
        loss = F.cross_entropy(logits_lm, y)
        loss.backward()
    g = W_leaf.grad.detach()
    del fresh, W_leaf
    torch.cuda.empty_cache()
    return (-g).float()


# ─────────── 分段判词 (预写判据, 机械应用) ───────────

def segment_verdict(seg_state, seg_idx, geval_neg_g=None, w_hi=None, w_lo=None):
    rows = seg_state["rows"]
    sd = seg_state["sum_dw_eff"]
    sg = seg_state["sum_neg_g"]
    sr = seg_state["sum_dw_raw"]
    last_g = seg_state["last_neg_g"]

    def _m(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    v = {"segment": seg_idx, "n_updates": len(rows)}
    for k in ("cos_rule", "cos_eff", "cos_osc", "cap_rotation",
              "trust_rotation", "cos_osc_raw"):
        v[k] = _m(k)
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            v[k + "_min"] = min(vals)
            v[k + "_max"] = max(vals)
    if sd is not None and sg is not None:
        v["cos_cum_g_cum"] = _cos(sd.flatten(), sg.flatten())
        v["cos_cum_g_end"] = _cos(sd.flatten(), last_g.flatten())
        v["cos_cum_raw_g_cum"] = _cos(sr.flatten(), sg.flatten())
        v["sum_dw_eff_norm"] = float(sd.norm())
        v["sum_neg_g_norm"] = float(sg.norm())

    # ── exp130 新增: coherence 段值 (向量和范数 / 逐步范数之和) ──
    # coherence(x) = ‖Σx‖ / Σ‖x‖ ; 随机游走 ≈ 1/√N ; 系统漂移 ≫ 1/√N
    sb = seg_state.get("sum_betar")
    srad = seg_state.get("sum_rad")
    sns = {k: seg_state.get("sn_" + k) for k in ("r", "e", "g", "betar", "rad")}
    v["sum_dw_raw_norm"] = float(sr.norm()) if sr is not None else None
    v["sum_betar_norm"] = float(sb.norm()) if sb is not None else None
    v["sum_rad_norm"] = float(srad.norm()) if srad is not None else None
    v["sum_norm"] = dict(sns)
    N = v["n_updates"]
    v["random_ref_1over_sqrtN"] = (N ** -0.5) if N else None

    def _coh(norm_sum, sum_norm):
        if norm_sum is None or not sum_norm:
            return None
        return norm_sum / sum_norm

    v["coherence_g"] = _coh(v.get("sum_neg_g_norm"), sns["g"])
    v["coherence_eff"] = _coh(v.get("sum_dw_eff_norm"), sns["e"])
    v["coherence_raw"] = _coh(v.get("sum_dw_raw_norm"), sns["r"])
    v["coherence_betar"] = _coh(v.get("sum_betar_norm"), sns["betar"])
    v["coherence_rad"] = _coh(v.get("sum_rad_norm"), sns["rad"])
    rr = v["random_ref_1over_sqrtN"]
    v["RA"] = {k: (v["coherence_" + k] / rr if (v.get("coherence_" + k) is not None and rr)
                   else None)
               for k in ("g", "eff", "raw", "betar", "rad")}
    # 漂移方向的定向版本 (Q4'): 径向 cap 分量 vs 累积梯度
    if srad is not None and sg is not None:
        v["cos_rad_g"] = _cos(srad.flatten(), sg.flatten())
    if sb is not None and srad is not None:
        v["cos_betar_rad"] = _cos(sb.flatten(), srad.flatten())
    # 两term分解自洽: ΣdW_eff 必须 = Σβr + ΣγW_pre
    if sd is not None and sb is not None and srad is not None:
        v["decomp_resid"] = float((sd.float() - (sb + srad)).norm())
    # 逐步 cos(Σ段−g_i, −g_{i−1}) 段均值
    _go = [r["cos_grad_osc"] for r in rows if r.get("cos_grad_osc") is not None]
    if _go:
        v["cos_grad_osc_mean"] = sum(_go) / len(_go)
        v["cos_grad_osc_min"] = min(_go)
        v["cos_grad_osc_max"] = max(_go)
        v["cos_grad_osc_n"] = len(_go)
    if geval_neg_g is not None and sd is not None:
        v["cos_cum_geval_end"] = _cos(sd.flatten(), geval_neg_g.flatten())
    if w_hi is not None and w_lo is not None and sd is not None:
        v["consistency_dW_norm"] = float((w_hi.float() - w_lo.float()).norm())

    br = []
    cr, ce, co, cap = v.get("cos_rule"), v.get("cos_eff"), v.get("cos_osc"), v.get("cap_rotation")
    cgc, cge = v.get("cos_cum_g_cum"), v.get("cos_cum_g_end")
    if cr is not None and ce is not None and cap is not None:
        if cr > 0.3 and ce < 0.1 and abs(cap) < 0.9:
            br.append("cap/trust 旋转源")
    if cr is not None and ce is not None and co is not None:
        if cr > 0.3 and ce > 0.3 and co < -0.2:
            br.append("振荡")
    if ce is not None and cgc is not None and cge is not None:
        if ce > 0.3 and cgc < 0.1 and cge > 0.2:
            br.append("moving target")
    v["branches"] = br if br else ["其他组合 (记录, 不预设)"]
    return v


def suggestion_from_branches(all_br):
    """修法建议模板: 按落支机械给出方向 (不预设结论, 待批)。"""
    cnt = Counter(br for seg in all_br for br in seg["branches"] if "其他" not in br)
    if not cnt:
        return ("未落预写支 — 记录原始七量, 建议先补口径 (如 from-start 累积/分段重划) 再判, "
                "不预设修法。")
    dom = cnt.most_common()
    parts = []
    for br, n in dom:
        if br == "振荡":
            parts.append(f"振荡 ({n}/5 段): 反振荡机制候选 — 更新平均/动量抵消/相邻窗去相关 "
                         f"(如梯度 EMA、跨步累积缓冲); 具体修法待批")
        elif br == "moving target":
            parts.append(f"moving target ({n}/5 段): 目标稳定化候选 — 每步追瞬时 −g 改为追 "
                         f"慢变目标 (如 −g 的 EMA、小批量平均); 具体修法待批")
        elif br == "cap/trust 旋转源":
            parts.append(f"cap/trust 旋转 ({n}/5 段): 后处理口径候选 — cap 只缩超限行/软帽/"
                         f"解耦 trust 与 lr; 具体修法待批")
    return "; ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=TARGET_STEPS)
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0,
                    help="冒烟步数 (0=正式; 冒烟跳过 pre/post eval/dump/段末评测窗仪器)")
    args = ap.parse_args()
    dev = torch.device(args.device)
    smoke = args.smoke > 0
    steps = args.smoke if smoke else args.steps

    tag = "exp130"
    stdout_path = run_file(tag + "_stdout.log")
    logf_console = open(stdout_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = logf_console
    sys.stderr = logf_console

    ck = resolve_path(e124.CKPT)
    md5_now = _md5(ck)
    if md5_now != e124.CKPT_MD5:
        print(f"exp129: FATAL md5 硬门不一致: 实测 {md5_now} != 预期 {e124.CKPT_MD5}; 拒跑", flush=True)
        sys.exit(2)
    dep = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp124_epsb_amp.py")
    dep_sha = hashlib.sha256(open(dep, "rb").read()).hexdigest()
    dep126 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "exp126_direction_teardown.py")
    dep126_sha = hashlib.sha256(open(dep126, "rb").read()).hexdigest()
    self_sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    print(f"exp129: R1 轨迹 (U rho={TRUST_RHO}, T_arm=OFF) lr={LR} steps={steps} "
          f"smoke={smoke} device={args.device}\n"
          f"exp129: ckpt={ck}\nexp129: md5={md5_now} (硬门 PASS)\n"
          f"exp129: 依赖 exp124_epsb_amp.py sha256={dep_sha}\n"
          f"exp129: 依赖 exp126_direction_teardown.py sha256={dep126_sha} (规则/仪表逐字复用源)\n"
          f"exp129: 执行件 sha256={self_sha}", flush=True)

    if _TARM["on"]:
        ReadoutMixin._build_lm_signal = _raw_temp_build_lm_signal
        print("exp129: monkeypatch: T 臂 ON", flush=True)
    else:
        print("exp129: monkeypatch: T 臂 OFF (lc 训练路径原样; R1)", flush=True)
    ReadoutMixin._update_lm_head = _update_lm_head_e3u_tel
    print(f"exp129: monkeypatch: E3 复刻 + U 信任域 rho={TRUST_RHO} + exp126 仪表 + exp129 七量仪器",
          flush=True)

    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=S_MAX, lm_freeze_w1=False)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    net = HybridPCNet.load(ck, cfg).to(dev)
    net.eval()
    net._migrate_expand = False
    print(f"exp129: net loaded; d_h={net.d_h} mem_k={net._mem_m.shape[0]}", flush=True)

    gauge, _, _ = build_gauge(args.data, dev)
    eval_wins = gauge["tail"][:N_EVAL_WIN]
    print(f"exp129: eval_wins={len(eval_wins)}", flush=True)

    run_json = run_file(tag + ".json")
    logf = open(run_file(tag + ".log"), "w", encoding="utf-8", buffering=1)
    jsonl = open(run_file(tag + "_telemetry.jsonl"), "w", encoding="utf-8", buffering=1)

    init_norms = {}
    for n in e124.PNAMES:
        try:
            init_norms[n] = float(e124._getp(net, n).norm())
        except Exception:
            init_norms[n] = None

    t0 = time.time()
    if smoke:
        pre = post = None
        base_lines = []
        print("exp129: 冒烟模式 — 跳过 pre/post eval / dump / 段末评测窗仪器 "
              "(保留每步七量仪器验证)", flush=True)
    else:
        pre = e124.eval_bpb_fresh(net, dev, eval_wins)
        print(f"exp129: pre bpb@s1={pre['bpb_s1']:.6f} top1={pre['top1_s1']:.6f} "
              f"(exp126 R1 pre 须逐位相同 = 协议保真)", flush=True)
        base_lines = _sample_dump(net, tag, cfg, dev, 0, 3, 240, 0.8, logf)

    ds = ByteDataset(args.data, max_length=S_MAX, max_samples=1270000)
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    print(f"exp129: data n={len(ds)}", flush=True)

    ent_buf = bytearray()
    ent_first = ent_last = None
    ent_min = ent_max = None
    ent_updates = 0
    decline_streak = rise_streak = 0
    abort = None
    death_cause = None
    dump_ents = {}
    W_series = {0: net.W_lm.data.clone()}
    snaps = {}
    seg_verdicts = []
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

        norms = e124.param_norms(net)
        nonfinite = [n for n, v in zip(e124.PNAMES, norms) if v is None or not math.isfinite(v)]
        d1r = float(net._dW1_absmax_raw) if hasattr(net, "_dW1_absmax_raw") else None

        def _gt(name):
            t = getattr(net, name, None)
            return float(t) if t is not None else None

        tel = dict(_TEL) if _TEL.get("fresh") else {}
        _TEL["fresh"] = False
        e129 = EXP129["last_row"] or {}
        row = {"step": i, "lr": LR, "run": "R1",
               "W_lm_norm": _gt("_tel_Wlm_norm_t"), "W1_norm": _gt("_tel_W1_norm_t"),
               "dW_cos_prev": tel.get("dW_cos_prev"), "trust_scale": tel.get("trust_scale"),
               "wlm_cap_scale": tel.get("wlm_cap_scale"),
               "dW_delta_theo_norm": tel.get("dW_delta_theo_norm"),
               "dW_delta_act_norm": tel.get("dW_delta_act_norm"),
               "dW_norm_pre": tel.get("dW_norm_pre"),
               "cos_rule": e129.get("cos_rule"), "cos_eff": e129.get("cos_eff"),
               "cos_osc": e129.get("cos_osc"), "cap_rotation": e129.get("cap_rotation"),
               "trust_rotation": e129.get("trust_rotation"),
               "cos_osc_raw": e129.get("cos_osc_raw"),
               "dW_raw_norm": e129.get("dW_raw_norm"), "dW_eff_norm": e129.get("dW_eff_norm"),
               "neg_g_norm": e129.get("neg_g_norm"), "mask_mean": e129.get("mask_mean"),
               "cos_cum_g_cum_run": e129.get("cos_cum_g_cum_run"),
               "cos_cum_g_end_run": e129.get("cos_cum_g_end_run"),
               "cos_cum_raw_g_cum_run": e129.get("cos_cum_raw_g_cum_run"),
               # ── exp130 新增 per-step 键 ──
               "cos_grad_osc": e129.get("cos_grad_osc"),
               "beta": e129.get("beta"), "gamma": e129.get("gamma"),
               "Wpre_norm": e129.get("Wpre_norm"),
               "errs": e129.get("errs") or e129.get("errs_eff"),
               "behavior": int(exec_b), "mem_k": net._mem_m.shape[0]}
        if row["errs"]:
            row["tb"] = e129.get("tb") or e129.get("tb_eff")
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
            # ── 分段判词 (七量段值 + 预写判据) ──
            geval = None
            sp = None
            if not smoke:
                sp = run_file(tag + f"_snapW{i}.safetensors")
                net.save(sp)
                snaps[i] = sp
                try:
                    geval = eval_batch_neg_g(sp, cfg, dev, eval_wins)
                except Exception as ex:  # noqa: BLE001
                    print(f"exp129: WARN 段末评测窗 −g 失败 step {i}: {ex!r}", flush=True)
            seg_idx = i // 100
            sv = segment_verdict(EXP129["seg"], seg_idx, geval_neg_g=geval,
                                 w_hi=W_series[i], w_lo=W_series[i - 100])
            seg_verdicts.append(sv)
            r = sv
            print(f"exp129: seg{seg_idx} (steps {i-99}..{i}) n_upd={r['n_updates']} | "
                  f"cos_rule={r.get('cos_rule')} cos_eff={r.get('cos_eff')} "
                  f"cos_osc={r.get('cos_osc')} | cos_cum_g_cum={r.get('cos_cum_g_cum')} "
                  f"cos_cum_g_end={r.get('cos_cum_g_end')} | cap_rot={r.get('cap_rotation')} "
                  f"trust_rot={r.get('trust_rotation')} | 落支={r['branches']}", flush=True)
            if sp is not None:
                try:
                    os.remove(sp)
                except OSError:
                    pass
            EXP129["seg"] = _fresh_seg()
        if death_cause:
            print(f"exp129: 机体自维护 death_cause={death_cause} (照常运行, 不中止)", flush=True)
            death_cause = None

    final_lines = []
    if not smoke:
        if abort is None and i == steps:
            final_lines = _sample_dump(net, tag, cfg, dev, steps, 3, 240, 0.8, logf)
            dump_ents[steps] = final_lines
            for t in final_lines:
                push_entropy(t.encode("utf-8", errors="replace"))
            if len(set(final_lines)) < len(final_lines):
                abort = f"R2b: step {steps} dump 内 3 个不同种子样本逐字相同"
        post = e124.eval_bpb_fresh(net, dev, eval_wins)
        print(f"exp129: post bpb@s1={post['bpb_s1']:.6f} top1={post['top1_s1']:.6f} "
              f"(R1 保真目标 {FIDELITY_TARGET})", flush=True)
    else:
        post = None

    # ── 保真交叉核对: 逐步余弦 (post-trust dW_cos_prev) vs v451 遥测 ──
    xcheck = {}
    try:
        ref = [json.loads(l) for l in open(resolve_path(V451_TEL), encoding="utf-8")
               if '"meta"' not in l]
        for seg in range(1, 6):
            rr = [r["dW_cos_prev"] for r in ref if (seg - 1) * 100 < r["step"] <= seg * 100
                  and r.get("dW_cos_prev") is not None]
            mm = [r["dW_cos_prev"] for r in tel_rows
                  if (seg - 1) * 100 < r["step"] <= seg * 100 and r.get("dW_cos_prev") is not None]
            if rr and mm:
                xcheck[f"seg{seg}"] = {
                    "v451_mean": round(sum(rr) / len(rr), 6), "exp129_mean": round(sum(mm) / len(mm), 6),
                    "n_v451": len(rr), "n_exp129": len(mm),
                    "max_abs_diff": round(max(abs(a - b) for a, b in zip(rr, mm)), 8) if len(rr) == len(mm) else None}
        # exp130 收紧: 原 exp129 写法 `all(v[...] for v in ... if ... is not None)`
        # 在五段计数不等时退化为空集真值 ⇒ 假 PASS。这里改为要求 5 段全部到位且全 0.0。
        diffs = {k: v["max_abs_diff"] for k, v in xcheck.items()
                 if isinstance(v, dict) and v.get("max_abs_diff") is not None}
        xcheck["n_segs_compared"] = len(diffs)
        ok = (len(diffs) == 5) and all(d == 0.0 for d in diffs.values())
        xcheck["bit_identical"] = bool(ok) and bool(xcheck)
        print(f"exp129: 保真交叉核对 (逐步余弦 vs v451): {xcheck}", flush=True)
    except Exception as ex:  # noqa: BLE001
        xcheck = {"instrument_fail": repr(ex)}
        print(f"exp129: WARN 交叉核对失败 (不中止): {ex!r}", flush=True)

    sugg = suggestion_from_branches(seg_verdicts)

    # ── exp130 新增: 全程 (from-start) coherence — N≈239, 统计上远强于单段 N≈48 ──
    allc = EXP129["all"]
    n_all = sum(1 for r in tel_rows if r.get("cos_rule") is not None)

    def _nrm(t):
        return float(t.norm()) if t is not None else None

    def _coh_all(vsum, snorm):
        a = _nrm(vsum)
        return (a / snorm) if (a is not None and snorm) else None

    ref_all = (n_all ** -0.5) if n_all else None
    coherence_all = {
        "N_updates": n_all, "random_ref_1over_sqrtN": ref_all,
        "sum_dw_raw_norm": _nrm(allc["sum_dw_raw"]), "sn_r": allc["sn_r"],
        "sum_dw_eff_norm": _nrm(allc["sum_dw_eff"]), "sn_e": allc["sn_e"],
        "sum_neg_g_norm": _nrm(allc["sum_neg_g"]), "sn_g": allc["sn_g"],
        "sum_betar_norm": _nrm(allc["sum_betar"]), "sn_betar": allc["sn_betar"],
        "sum_rad_norm": _nrm(allc["sum_rad"]), "sn_rad": allc["sn_rad"],
        "coherence_raw": _coh_all(allc["sum_dw_raw"], allc["sn_r"]),
        "coherence_eff": _coh_all(allc["sum_dw_eff"], allc["sn_e"]),
        "coherence_g": _coh_all(allc["sum_neg_g"], allc["sn_g"]),
        "coherence_betar": _coh_all(allc["sum_betar"], allc["sn_betar"]),
        "coherence_rad": _coh_all(allc["sum_rad"], allc["sn_rad"]),
    }
    coherence_all["RA"] = {k: (coherence_all["coherence_" + k] / ref_all
                               if coherence_all.get("coherence_" + k) is not None and ref_all
                               else None)
                           for k in ("raw", "eff", "g", "betar", "rad")}
    coherence_all["ratio_raw_over_g"] = (
        coherence_all["coherence_raw"] / coherence_all["coherence_g"]
        if coherence_all["coherence_raw"] and coherence_all["coherence_g"] else None)
    coherence_all["cos_sumraw_sumg"] = (
        _cos(allc["sum_dw_raw"].flatten(), allc["sum_neg_g"].flatten())
        if allc["sum_dw_raw"] is not None and allc["sum_neg_g"] is not None else None)
    coherence_all["cos_sumrad_sumg"] = (
        _cos(allc["sum_rad"].flatten(), allc["sum_neg_g"].flatten())
        if allc["sum_rad"] is not None and allc["sum_neg_g"] is not None else None)

    # ── exp130 自检 (d): 本件复算的段量必须与 exp129 v460 同段逐位/逐值一致 ──
    v460_cross = {}
    try:
        p460 = resolve_path(os.path.join("out", "v460-20261021-220406", "exp129.json"))
        j460 = json.load(open(p460, encoding="utf-8"))
        s460 = {s["segment"]: s for s in j460["segment_verdicts"]}
        for sv in seg_verdicts:
            r = s460.get(sv["segment"], {})
            v460_cross[f"seg{sv['segment']}"] = {
                "sum_dw_eff_norm": [sv.get("sum_dw_eff_norm"), r.get("sum_dw_eff_norm")],
                "sum_neg_g_norm": [sv.get("sum_neg_g_norm"), r.get("sum_neg_g_norm")],
                "cos_cum_g_cum": [sv.get("cos_cum_g_cum"), r.get("cos_cum_g_cum")],
                "cos_cum_raw_g_cum": [sv.get("cos_cum_raw_g_cum"), r.get("cos_cum_raw_g_cum")],
            }
        pairs = [ab for v in v460_cross.values() for ab in v.values()
                 if ab[0] is not None and ab[1] is not None]
        v460_cross["max_abs_diff"] = max((abs(a - b) for a, b in pairs), default=None)
        print(f"exp130: 自检(d) vs exp129 v460 段量 max_abs_diff={v460_cross['max_abs_diff']}",
              flush=True)
    except Exception as ex:  # noqa: BLE001
        v460_cross = {"instrument_fail": repr(ex)}
        print(f"exp130: WARN 自检(d) 失败 (不中止): {ex!r}", flush=True)

    out = {
        "exp": "exp130", "stage": "B_instrumented", "run": "R1", "lr": LR, "trust_rho": TRUST_RHO,
        "coherence_all": coherence_all, "selfcheck_v460": v460_cross,
        "T_arm": _TARM["on"], "device": args.device, "smoke": smoke,
        "device_note": "历代 GPU (v457 教训: device 入账); R1 轨迹与 exp126 R1 同构造",
        "ckpt_md5": md5_now, "ckpt_md5_gate": "PASS",
        "dep_exp124_sha256": dep_sha, "dep_exp126_sha256": dep126_sha,
        "self_sha256": self_sha,
        "steps_run": i, "target_steps": args.steps, "abort": abort,
        "fidelity": {"post_bpb_s1": (post or {}).get("bpb_s1") if post else None,
                     "target": FIDELITY_TARGET,
                     "pass": (post is not None and abs(post["bpb_s1"] - FIDELITY_TARGET) < 1e-9)},
        "xcheck_dWcos_prev": xcheck,
        "pre_bpb": (pre or {}).get("bpb_s1") if pre else None,
        "segment_verdicts": seg_verdicts,
        "suggestion": sugg,
        "secs": round(time.time() - t0, 1),
        "iron_rules": "crates/ 零改动 (进程内 monkeypatch); md5 硬门; 全 fp16 训练 "
                      "(仪器 fp32 只读零写回); 权重不落盘 (瞬态快照即删); run_file 自增; "
                      "运行不 kill",
    }
    with open(run_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"exp129: 写出 {run_json}", flush=True)

    # ── 首句 (执行口径: 七量五段序列 + 分段判词 + 修法建议 + 保真门) ──
    def _fmt(r, k):
        v = r.get(k)
        return "n/a" if v is None else f"{v:+.3f}"

    seq_txt = " | ".join(
        f"seg{r['segment']}: rule={_fmt(r,'cos_rule')} eff={_fmt(r,'cos_eff')} "
        f"osc={_fmt(r,'cos_osc')} cum_g_cum={_fmt(r,'cos_cum_g_cum')} "
        f"cum_g_end={_fmt(r,'cos_cum_g_end')} cap={_fmt(r,'cap_rotation')} "
        f"trust={_fmt(r,'trust_rotation')} [{','.join(r['branches'])}]"
        for r in seg_verdicts)
    fid = out["fidelity"]
    ca = coherence_all
    def fmt_seq(key):
        return " ".join(("n/a" if r.get(key) is None else f"{r[key]:+.4f}")
                        for r in seg_verdicts)

    print("exp130 首句: " + " || ".join([
        f"全程 coherence (N={ca['N_updates']}, 1/√N={ca['random_ref_1over_sqrtN']:.4f}): "
        f"raw={ca['coherence_raw']:.4f} (RA {ca['RA']['raw']:.2f}) "
        f"eff={ca['coherence_eff']:.4f} (RA {ca['RA']['eff']:.2f}) "
        f"g={ca['coherence_g']:.4f} (RA {ca['RA']['g']:.2f}) "
        f"betar={ca['coherence_betar']:.4f} rad={ca['coherence_rad']:.4f} "
        f"| raw/g={ca['ratio_raw_over_g']:.3f}",
        f"五段 coherence_g: {fmt_seq('coherence_g')} | 五段 coherence_raw: {fmt_seq('coherence_raw')}",
        f"五段 coherence_eff: {fmt_seq('coherence_eff')} | 五段 cos_grad_osc: {fmt_seq('cos_grad_osc_mean')}",
        f"自检(d) vs v460 max_abs_diff={v460_cross.get('max_abs_diff')}",
        f"保真门: post_bpb={fid['post_bpb_s1']} ({'PASS' if fid['pass'] else 'FAIL'}) | "
        f"bit_identical={xcheck.get('bit_identical')}",
    ]), flush=True)
    print("exp130: DONE", flush=True)
    logf.close()
    jsonl.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        crash = resolve_path("out/exp129_crash.txt")
        with open(crash, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        sys.stderr.write(traceback.format_exc())
        sys.exit(1)
