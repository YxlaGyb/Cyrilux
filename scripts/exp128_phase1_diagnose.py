"""
exp128 Phase 1: delta 规则重建 — 诊断（隔离 G 低机制）
=====================================================

预登记: docs/plan/exp128_phase1_preregistration.md (跑前锁死; 本件执行口径见该文档 §首句).

设计回顾 (2026-10-20 审查裁决):
  G 低是测量事实 (effective ΔW 含 cap10 + U 信任域 ρ=0.25 后处理, 与 −∇CE 去相关),
  但机制未拆: (a) Hebbian 规则本身方向错 / (b) 规则方向对但 cap/trust 后处理旋错.
  本脚本在 R2 raw-CE 轨迹上, 每 100 步用六变体逐站替换, 测 cos 从 <0.1 跳到 >0.3 的
  "方向破坏层".

  V0 = W_lm(step) − W_lm(step−100)            (含 cap/trust 累积 effective ΔW; = exp126 G 低)
  V1 = rms_norm(h)ᵀ @ (err_scaled·contrast_w·d_t·mask)·√d_h   (去 cap/trust)
  V2 = hᵀ @ (err_scaled·contrast_w·d_t·mask)                   (去 rms_norm 抹幅度)
  V3 = hᵀ @ (err_scaled·d_t·mask)                             (去 contrast 重加权)
  V4 = hᵀ @ (err_scaled·mask)                                 (去 d_t 门控; 本轨迹 d_t=1 ⇒ V3≡V4)
  V5 = hᵀ @ (target − softmax(logits_lm))·mask ≡ −g          (判别器, 见下)
  V5r = hᵀ @ eps_total·mask                                   (隔离 err 单位能归一变换)

  −g 取读出端真目标梯度 = autograd(CE(logits_lm, target); w.r.t. W_lm), logits_lm 为
  _build_lm_signal 实际调制后输出 (中心/归一/max_abs→[-60,60] + 打印掩码). V1..V4 用
  训练步 intermediates; V5/V5r/−g 用评测窗 (固定 200 窗) — 坐标差异按预登记 §err 坐标
  一致性说明显式标注, 不隐藏.

─── 审查裁决附加项 (2026-10-20, exp128 Phase 1 审查聊天记录; v457 重跑批准条件) ───

  V0..V4 接受为有效 (全程 |cos|<0.06); V5/V5r 形状 bug (h_eval 二维被 .mean(0) 压成
  [256], 与 [65536] 的 −g 对不上) 修复方向获接受; 但重跑暂缓 — 存在比 V5/V5r 更致命
  的二义: **统计上限**。单窗口梯度与全批量 −g 的余弦期望 ≈ 1/√B (B=窗口位置数),
  255 位置 ⇒ ≈0.063, 与实测 |cos|<0.06 同量级, 无法排除"G 低是统计学不是机制"。
  重跑前强制两项附加项 (本件落实):

    附加项 1 — 随机 baseline (必备): 同一 −g 框架下, 100 个随机单窗 (seed-0 shuffled
      流 idxs[1000:1100], 与训练步 1..500 所用 idxs[0:500] 不相交) 的解析 delta 梯度
      (与 V5 同公式, 无 autograd) 与全批量 −g 的 cos 分布 (均值/std/95 分位)。
      没有 baseline 的相关性数字是修辞, 不是证据。

    附加项 2 — 训练窗 −g 对照 (必备): −g 也在训练窗上算, 与 ΔW/V1..V4 **同一窗口、
      同一 (更新前) 权重**:
        (a) autograd 真梯度 −g_train (经完整调制 Jacobian; fp32 叶克隆, 只读零写回);
        (b) 解析 delta V5_train (V5 同公式, 训练窗, 带 mask)。
      cos_train     = 变体 vs 训练窗 autograd −g   (区分"规则方向错" vs "窗口不匹配")
      cos_vs_v5train = 变体 vs 训练窗解析 delta     (剥离调制 Jacobian 混淆)
      注意: 评测窗径 −g 用更新后快照权重 (既有口径), 训练窗径 −g 用更新前权重
      (与 V1..V4 intermediates 严格同权重) — 差异 ≤1 步 lr, 显式标注不隐藏。

  V5 角色升级为**判别器** (非参考上界): V5 评测窗 cos 验证测量框架是否有效 —
  解析梯度能否在与 −g 同窗口同权重下测出高 cos。V5 cos ≈ baseline ⇒ 测量框架问题,
  Phase 1 需重设计 (如累积 100 步 ΔW vs 100 步累积 −g); V5 cos ≫ baseline ⇒ 框架
  成立, V0..V4 的 <0.06 是真信号。

  预检模式 (--additions-only): 不跑 500 步训练; 在基准权重 (V5 修复自检 + 随机
  baseline) 与前 N 步真实轨迹步 (训练窗对照) 上取数, 约 1–2 分钟; 产物目录锚定
  out/exp128_additions_precheck/ (pin_run_dir, 不占用 v 版本号; v457 保留给正式
  重跑)。预检通过 (V5 自检 cos 高 + 两项附加项有数) 后直接启动 v457 全量重跑。

铁律: crates/ 零改动 (进程内 monkeypatch); 不动 v244/迁移件 (md5 硬门); 全 fp16 训练
(统计/仪器 fp32 只读零写回); 权重不落盘瞬态快照即删; run_file 自增新目录.
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
import torch.nn as nn
import torch.nn.functional as F

import exp124_epsb_amp as e124
from exp115_say import N_EVAL_WIN, S_MAX, build_gauge
from exp116_z4_probe import _md5
from crates.cli.utils import resolve_path, run_file, pin_run_dir
from crates.graphs import set_backend
set_backend("ref")
from crates.kernels import rms_norm, saturate, cubic
from crates.modal.text.geometry import ALPHABET
from crates.hybrid.learning.readout import ReadoutMixin
from crates.model.constants import F_UNIT_SCALE
from crates.model import CyreneModel, HybridPCNet
from crates.utils.codec import ByteDataset

torch.set_grad_enabled(False)

LR = 0.05
TARGET_STEPS = 500
TRUST_RHO = 0.25
CKPT = e124.CKPT
CKPT_MD5 = e124.CKPT_MD5

_TARM = {"on": True}
_orig_build_lm_signal = ReadoutMixin._build_lm_signal
_TEL = {}
_TEL_RAW = {}

# 单步 intermediates（每步覆盖；仅 checkpoint 步取用）
STEP_INTERMEDIATES = {}
CUR_STEP = {"i": 0}

# 审查裁决附加项 2: 训练窗 −g 对照暂存 (step -> dict)
TRAIN_CTL = {}

# 预检模式开关: on 时 patch 每步都采集训练窗对照 (否则仅 %100 checkpoint 步)
_PRECHECK = {"on": False}


# ─────────── T 臂 monkeypatch (raw 温度训练误差; 与 exp126 同; flag 默认关) ───────────

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


# ─────────── E3 复刻 + U 信任域 (与 exp126 同) + 单步 intermediates 暂存 ───────────

def _update_lm_head_e3u_tel(self, ctx, lm):
    net = self.net
    if ctx.free_run or ctx.echo_world_frozen:
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

    # ── 暂存单步 intermediates（供 exp128 Phase 1 变体计算）──
    STEP_INTERMEDIATES[CUR_STEP["i"]] = {
        "h": lm.h[:, :-1].detach().clone(),
        "err_scaled": err_scaled.detach().clone(),
        "eps_raw": lm.eps_total.detach().clone(),
        "contrast_w": contrast_w.detach().clone(),
        "d_t": d_t.detach().clone(),
        "mask": lm_update_mask.detach().clone(),
    }

    # ── 审查裁决附加项 2: 训练窗 −g 对照 (权重更新前采集, 与 V1..V4 同窗同权重) ──
    if _PRECHECK["on"] or (CUR_STEP["i"] % 100 == 0):
        try:
            TRAIN_CTL[CUR_STEP["i"]] = _train_window_ctl(net, lm, byte_ids,
                                                         lm_update_mask, inv_h)
        except Exception as e:
            import traceback as _tb
            TRAIN_CTL[CUR_STEP["i"]] = {"errs": repr(e), "tb": _tb.format_exc()}

    wlm_pre = net.W_lm.data.clone()
    trust_scale = 1.0
    step_norm_pre = float(dW_lm.detach().float().norm()) * LR
    wpre_norm_f = float(wlm_pre.detach().float().norm())
    if step_norm_pre > TRUST_RHO * wpre_norm_f:
        trust_scale = (TRUST_RHO * wpre_norm_f) / (step_norm_pre + 1e-12)
        dW_lm = dW_lm * trust_scale
    net.W_lm.data += dW_lm * LR
    rn_post = net.W_lm.data.norm(dim=1)
    mx_lm = rn_post.max()
    cap_scale = float(torch.minimum(torch.ones_like(mx_lm),
                                    10.0 / (mx_lm + 1e-6)).to(torch.float16))
    net.W_lm.data.mul_(
        torch.minimum(torch.ones_like(mx_lm), 10.0 / (mx_lm + 1e-6)).to(torch.float16)
    )
    e_h = (err_scaled @ net.W_lm.T) * lm.h_deriv[:, :-1]
    lm.e_h = e_h
    dW1 = (rms_norm(lm.zh[:, :-1]).transpose(-2, -1) @ rms_norm(e_h)).mean(dim=0)
    mx1 = dW1.abs().max() + 1e-4
    dW1 = dW1 / mx1 / 64.0
    if not net.cfg.lm_freeze_w1:
        net.W1.data += dW1 * LR
        from crates.kernels.normalize import soft_norm_preserve
        soft_norm_preserve(net.W1.data)
    if hasattr(net, "_novelty"):
        del net._novelty
    return d_t


# ─────────── 审查裁决附加项 2: 训练窗 −g 对照 (patch 内, 更新前权重) ───────────

def _train_window_ctl(net, lm, byte_ids, lm_update_mask, inv_h):
    """训练窗对照: 用本步 intermediates 同源张量 (lm.h / 本步更新前权重) 计算:

      (a) neg_g_train — autograd 真梯度 −g (经完整调制 Jacobian: 中心化/归一/
          max_abs→60 + 打印掩码; fp32 叶克隆, 只读零写回);
      (b) V5_train   — 解析 delta 梯度 (V5 同公式, 训练窗, 带 learn_mask).

    与 V1..V4 严格同窗口、同 (更新前) 权重; fp16 张量转 fp32 只读。
    返回 dict 含 shapes/mask_mean 自报。
    """
    h_tr = lm.h[:, :-1].detach().to(torch.float16).float()      # [N,S-1,d_h]
    y_tr = byte_ids[:, 1:].reshape(-1).long()                   # [N*(S-1)]
    tgt_tr = F.one_hot(byte_ids[:, 1:], num_classes=ALPHABET).to(torch.float32)
    m_tr = lm_update_mask.detach().float()                      # [N,S-1,1]
    bias_f = net.bias_lm.detach().float()
    mask_f = (1.0 - net._mask_print.detach().float()) * -1e4

    def _mod(w_leaf):
        raw = (h_tr @ w_leaf + bias_f) * inv_h
        lc = (raw - raw.mean(-1, keepdim=True)) / (raw.std(-1, keepdim=True) + 1e-4)
        ll = lc / lc.abs().max(-1, keepdim=True).values * 60.0
        return ll + mask_f

    # (a) autograd 真梯度 (fp32 叶, 零写回)
    W_leaf = net.W_lm.detach().float().clone().requires_grad_(True)
    with torch.enable_grad():
        loss = F.cross_entropy(_mod(W_leaf).reshape(-1, 256), y_tr)
        loss.backward()
    neg_g_train = (-W_leaf.grad.detach()).float().flatten()
    del W_leaf

    # (b) 解析 delta (V5 同公式, 训练窗)
    with torch.no_grad():
        probs_tr = torch.softmax(_mod(net.W_lm.detach().float()), -1)
        err_tr = (tgt_tr - probs_tr) * m_tr
        n_tr = int(h_tr.shape[0] * h_tr.shape[1])
        V5_train = (h_tr.transpose(-2, -1) @ err_tr).mean(0) / n_tr

    return {
        "neg_g_train": neg_g_train.detach(),
        "V5_train": V5_train.detach().float(),
        "shapes": {"h_train": list(h_tr.shape), "neg_g_train": list(neg_g_train.shape),
                   "V5_train": list(V5_train.shape)},
        "mask_mean": float(m_tr.mean()),
        "n_rows": n_tr,
    }


# ─────────── −g (调制 logits_lm 真目标梯度) + V5/V5r (评测窗) ───────────

def grad_measure_modulated(snap_path, cfg, dev, eval_wins):
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
        mask_term = (1.0 - fresh._mask_print) * -1e4
        logits_lm = logits_lm + mask_term
        loss = F.cross_entropy(logits_lm, y)
        loss.backward()
    g = W_leaf.grad.detach()
    probs = torch.softmax(logits_lm.detach(), -1)
    target_oh = F.one_hot(y, num_classes=ALPHABET).to(probs.dtype)
    eps_total_eval = (target_oh - probs)  # 评测窗 error（raw, R2 坐标）
    del fresh, W_leaf
    return g, h.detach(), probs.detach(), target_oh.detach(), eps_total_eval.detach()


# ─────────── 审查裁决附加项 1: 随机 baseline (100 随机单窗 vs 全批量 −g) ───────────

def random_window_baseline(snap_path, cfg, dev, rand_wins, ng):
    """随机 baseline: 同一快照权重上, len(rand_wins) 个随机单窗各自的解析 delta
    梯度 (与 V5 同公式, 无 autograd — 解析式 ≡ 该窗 −g 至调制 Jacobian 差, 与 V5
    同口径) 与全批量 −g 的 cos 分布。返回均值/std/分位 + 全量 values + 形状自报。
    """
    fresh = HybridPCNet.load(snap_path, cfg).to(dev)
    fresh.eval()
    fresh._migrate_expand = False
    h, y = e124.extract_h(fresh, dev, rand_wins, "rand")
    h = h.to(dev, torch.float16).float()
    y = y.to(dev)
    bias = fresh.bias_lm.detach().float()
    inv_h = 1.0 / math.sqrt(fresh.d_h)
    mask_term = (1.0 - fresh._mask_print.detach().float()) * -1e4
    with torch.no_grad():
        raw = (h @ fresh.W_lm.detach().float() + bias) * inv_h
        logits_c = (raw - raw.mean(-1, keepdim=True)) / (raw.std(-1, keepdim=True) + 1e-4)
        logits_lm = logits_c / logits_c.abs().max(-1, keepdim=True).values * 60.0 + mask_term
        probs = torch.softmax(logits_lm, -1)
        target_oh = F.one_hot(y, num_classes=ALPHABET).to(probs.dtype)
    del fresh
    n = len(rand_wins)
    rows = h.shape[0] // n
    h3 = h.reshape(n, rows, -1)
    e3 = (target_oh - probs).reshape(n, rows, -1)
    deltas = (h3.transpose(1, 2) @ e3) / rows              # [n, d_h, 256]
    cos_vals = [_cos(deltas[i].flatten(), ng) for i in range(n)]
    abs_vals = [abs(v) for v in cos_vals]
    mean = sum(cos_vals) / n
    var = sum((v - mean) ** 2 for v in cos_vals) / max(1, n - 1)
    return {
        "n": n, "rows_per_window": rows,
        "cos_mean": mean, "cos_std": math.sqrt(var),
        "cos_min": min(cos_vals), "cos_max": max(cos_vals),
        "cos_p5": _pct(cos_vals, 0.05), "cos_p95": _pct(cos_vals, 0.95),
        "abs_mean": sum(abs_vals) / n,
        "abs_p95": _pct(abs_vals, 0.95), "abs_max": max(abs_vals),
        "values": cos_vals,
        "theory_anchor": {"inv_sqrt_255": 1.0 / math.sqrt(255),
                          "sqrt_rows_over_evalrows": math.sqrt(rows / (N_EVAL_WIN * 255))},
        "windows_def": "seed-0 shuffled 流 idxs[1000:1100] (与训练步 1..500 的 idxs[0:500] 不相交)",
        "shapes": {"h_rand": list(h.shape), "deltas": list(deltas.shape),
                   "ng": list(ng.shape)},
    }


def _pct(xs, q):
    s = sorted(xs)
    if not s:
        return None
    k = q * (len(s) - 1)
    f = int(math.floor(k))
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _cos(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


# ─────────── 变体张量 (V1..V4, V5r; 逐变体容错 + 形状自报) ───────────

def rule_variant_tensors(it, dev):
    """由单步 intermediates 计算 V1..V4/V5r 张量。逐变体 try/except, 任何单点
    失败只记 errs+shapes, 不拖垮整轮 (v456 自报习惯的保持)。"""
    tensors, errs, shapes = {}, {}, {}
    try:
        h = it["h"].to(dev, torch.float16).float()
        err_scaled = it["err_scaled"].to(dev, torch.float16).float()
        eps_raw = it["eps_raw"].to(dev, torch.float16).float()
        contrast_w = it["contrast_w"].to(dev, torch.float16).float()
        d_t = it["d_t"].to(dev, torch.float16).float()
        mask = it["mask"].to(dev, torch.float16).float()
        shapes["h"] = list(h.shape)
        shapes["err_scaled"] = list(err_scaled.shape)
        shapes["eps_raw"] = list(eps_raw.shape)
        dh = h.shape[-1]
        sqrt_dh = math.sqrt(dh)
        rn_h = rms_norm(h)
        cands = {
            "V1": (rn_h.transpose(-2, -1) @ (err_scaled * contrast_w * d_t * mask)).mean(0) * sqrt_dh,
            "V2": (h.transpose(-2, -1) @ (err_scaled * contrast_w * d_t * mask)).mean(0),
            "V3": (h.transpose(-2, -1) @ (err_scaled * d_t * mask)).mean(0),
            "V4": (h.transpose(-2, -1) @ (err_scaled * mask)).mean(0),
            "V5r": (h.transpose(-2, -1) @ (eps_raw * mask)).mean(0),
        }
    except Exception as e:
        import traceback as _tb
        errs["variants"] = repr(e)
        errs["tb_var"] = _tb.format_exc()
        return tensors, errs, shapes
    for name, V in cands.items():
        try:
            tensors[name] = V.detach()
            shapes[name] = list(V.flatten().shape)
        except Exception as e:
            errs[name] = repr(e)
    return tensors, errs, shapes


def compute_variants(k, snaps, W_series, eval_wins, cfg, dev, rand_wins=None):
    """逐变体计算 cos(-g) + 审查裁决两项附加项。任何单步/单变体失败都被捕获并
    记录形状, 不让整轮崩空。"""
    res = {"step": k, "cos": {}, "cos_train": {}, "cos_vs_v5train": {},
           "errs": {}, "shapes": {}, "additions": {}}
    try:
        g, h_eval, probs, target_oh, eps_total_eval = grad_measure_modulated(
            snaps[k], cfg, dev, eval_wins)
    except Exception as e:
        import traceback as _tb
        res["errs"]["grad"] = repr(e)
        res["tb_grad"] = _tb.format_exc()
        return res
    ng = (-g).flatten()  # 下降方向参考
    res["shapes"]["ng"] = list(ng.shape)
    res["shapes"]["h_eval"] = list(h_eval.shape)
    res["g_norm"] = float(g.norm())

    # V0: effective ΔW (含 cap/trust 后处理)
    V0_t = None
    try:
        if (k - 100) in W_series:
            V0_t = (W_series[k].float() - W_series[k - 100].float()).flatten()
            res["shapes"]["V0"] = list(V0_t.shape)
            res["cos"]["V0"] = _cos(V0_t, ng)
        else:
            res["errs"]["V0"] = f"W_series 缺基线 {k - 100}"
    except Exception as e:
        res["errs"]["V0"] = repr(e)
        res["shapes"]["V0"] = "n/a"

    # 训练窗 −g 对照参考 (附加项 2; 与 intermediates 同窗同更新前权重)
    tc = TRAIN_CTL.get(k)
    ngt = v5t = None
    if tc is not None:
        if "neg_g_train" in tc:
            ngt = tc["neg_g_train"]
            res["shapes"]["neg_g_train"] = tc["shapes"]["neg_g_train"]
            res["additions"]["train_ctl"] = {"mask_mean": tc.get("mask_mean"),
                                             "n_rows": tc.get("n_rows"),
                                             "weight_timing": "更新前 (与 V1..V4 同权重)"}
        if "V5_train" in tc:
            v5t = tc["V5_train"].float().flatten()
            res["shapes"]["V5_train"] = list(v5t.shape)
        if "errs" in tc:
            res["errs"]["train_ctl"] = tc["errs"]
            res["tb_train_ctl"] = tc.get("tb")

    # 训练步 intermediates (V1..V4/V5r); 缺失 ⇒ 该步 update 早退 (echo/free_run)
    it = STEP_INTERMEDIATES.get(k)
    if it is None:
        res["errs"]["interm"] = "缺失 (该步 _update_lm_head 早退, 无 ΔW 可析)"
    else:
        tensors, verrs, vshapes = rule_variant_tensors(it, dev)
        res["errs"].update(verrs)
        res["shapes"].update(vshapes)
        for name in ("V1", "V2", "V3", "V4", "V5r"):
            V = tensors.get(name)
            if V is None:
                continue
            Vf = V.flatten()
            try:
                res["cos"][name] = _cos(Vf, ng)
            except Exception as e:
                res["errs"][name] = repr(e)
                res["shapes"][name] = list(Vf.shape)
            if ngt is not None:
                try:
                    res["cos_train"][name] = _cos(Vf, ngt)
                except Exception as e:
                    res["errs"][name + "_train"] = repr(e)
            if v5t is not None:
                try:
                    res["cos_vs_v5train"][name] = _cos(Vf, v5t)
                except Exception as e:
                    res["errs"][name + "_v5train"] = repr(e)

    # V5 (评测窗, 已修复形状 bug): 纯 delta ≡ −g (判别器; 不依赖 intermediates)
    try:
        n_eval = h_eval.shape[0]
        V5 = (h_eval.transpose(-2, -1) @ (target_oh - probs)) / n_eval
        res["shapes"]["V5"] = list(V5.flatten().shape)
        res["cos"]["V5"] = _cos(V5.flatten(), ng)
    except Exception as e:
        res["errs"]["V5"] = repr(e)
        res["shapes"]["V5"] = "n/a"

    # 训练窗径汇总: V5_train vs 训练窗 autograd −g; V0 时间跨度不匹配, 仅参考
    if ngt is not None and v5t is not None:
        try:
            res["cos_train"]["V5"] = _cos(v5t, ngt)
        except Exception as e:
            res["errs"]["V5_train"] = repr(e)
    if V0_t is not None:
        if ngt is not None:
            try:
                res["cos_train"]["V0"] = _cos(V0_t, ngt)
            except Exception as e:
                res["errs"]["V0_train"] = repr(e)
        if v5t is not None:
            try:
                res["cos_vs_v5train"]["V0"] = _cos(V0_t, v5t)
            except Exception as e:
                res["errs"]["V0_v5train"] = repr(e)

    # ── 附加项 1: 随机 baseline (同一快照权重, 与 ng 同源) ──
    if rand_wins:
        try:
            res["additions"]["random_baseline"] = random_window_baseline(
                snaps[k], cfg, dev, rand_wins, ng)
        except Exception as e:
            import traceback as _tb
            res["errs"]["random_baseline"] = repr(e)
            res["tb_rand"] = _tb.format_exc()
    return res


# ─────────── 分层判词 (审查裁决执行口径首句四要素) ───────────

def layered_verdict(results):
    """L1 测量框架 (V5 判别器) / L2 统计上限 (随机 baseline + V0..V4 落位) /
    L3 机制 (训练窗径)。判词规则按审查裁决预写, 数值实测填入。"""
    out = {"L1_framework": {}, "L2_baseline": {}, "L3_mechanism": {}}
    steps = [r["step"] for r in results]

    # L1: V5 判别器 — 评测窗径 + 训练窗径
    v5e = {r["step"]: r["cos"].get("V5") for r in results if r["cos"].get("V5") is not None}
    v5t = {r["step"]: r["cos_train"].get("V5") for r in results
           if r["cos_train"].get("V5") is not None}
    out["L1_framework"]["V5_eval_cos"] = v5e
    out["L1_framework"]["V5_train_cos"] = v5t
    rbs = {r["step"]: r["additions"]["random_baseline"] for r in results
           if "random_baseline" in r.get("additions", {})}
    if rbs:
        ex = next(iter(rbs.values()))
        out["L2_baseline"]["random"] = {
            s: {"mean": round(rb["cos_mean"], 4), "std": round(rb["cos_std"], 4),
                "abs_mean": round(rb["abs_mean"], 4), "abs_p95": round(rb["abs_p95"], 4),
                "p5": round(rb["cos_p5"], 4), "p95": round(rb["cos_p95"], 4)}
            for s, rb in rbs.items()}
        out["L2_baseline"]["theory_anchor"] = ex["theory_anchor"]
    # V0..V4 落位: |cos| 对 baseline |cos| 分布 (内 = |cos| ≤ abs_p95)
    inside = {}
    for r in results:
        rb = r.get("additions", {}).get("random_baseline")
        if rb is None:
            continue
        for name in ("V0", "V1", "V2", "V3", "V4"):
            cv = r["cos"].get(name)
            if cv is None:
                continue
            inside.setdefault(name, {})[r["step"]] = {
                "cos": round(cv, 4), "abs_cos": round(abs(cv), 4),
                "inside_abs_p95": bool(abs(cv) <= rb["abs_p95"]),
                "inside_signed_p5_p95": bool(rb["cos_p5"] <= cv <= rb["cos_p95"])}
    out["L2_baseline"]["V0_V4_vs_baseline"] = inside
    # L3: 训练窗径变体 cos (同窗同权重 vs autograd −g)
    tr = {}
    for r in results:
        for name, cv in r.get("cos_train", {}).items():
            if cv is not None:
                tr.setdefault(name, {})[r["step"]] = round(cv, 4)
    out["L3_mechanism"]["cos_train"] = tr
    v5tr = {}
    for r in results:
        for name, cv in r.get("cos_vs_v5train", {}).items():
            if cv is not None:
                v5tr.setdefault(name, {})[r["step"]] = round(cv, 4)
    out["L3_mechanism"]["cos_vs_v5train"] = v5tr

    # ── 判词 (规则预写, 数值代入) ──
    v5e_vals = [v for v in v5e.values() if v is not None]
    v5t_vals = [v for v in v5t.values() if v is not None]
    rb_means = [rb["cos_mean"] for rb in rbs.values()]
    judge = []
    if v5e_vals and rb_means:
        bm = sum(rb_means) / len(rb_means)
        v5m = sum(v5e_vals) / len(v5e_vals)
        # 注意: V5 与 −g 同窗口 (200 评测窗), 随机 baseline 是"随机单窗 vs 全批量" —
        # 两者的参照不同, V5 的正确标尺是 baseline 均值 (典型单窗水平) + 审查预写的
        # 0.2 阈值, 不是 baseline |cos| 的重尾 p95 (v457 首版判词误设, 已修正)。
        if v5m > max(0.2, 3 * bm):
            judge.append(f"L1 测量框架成立: V5 评测窗 cos 均值 {v5m:.3f} ≫ 随机 baseline "
                         f"均值 {bm:.3f} (且 ≫0.2 预写阈值) — 解析梯度可测出高 cos, 管道有效")
        else:
            judge.append(f"L1 测量框架存疑: V5 评测窗 cos 均值 {v5m:.3f} 未显著超过随机 "
                         f"baseline 均值 {bm:.3f} — 单窗口 vs 全批量统计上限主导, "
                         f"Phase 1 测量需重设计 (如 100 步累积 ΔW vs 100 步累积 −g)")
    if v5t_vals:
        judge.append(f"V5 训练窗 cos (vs 同窗 autograd −g): 均值 "
                     f"{sum(v5t_vals) / len(v5t_vals):.3f}")
    n_in = sum(1 for d in inside.values() for v in d.values() if v["inside_abs_p95"])
    n_tot = sum(len(d) for d in inside.values())
    if n_tot:
        judge.append(f"L2 统计落位: V0..V4 共 {n_tot} 个 (变体×步) 测量中 {n_in} 个落在随机 "
                     f"baseline |cos|p95 内 — 落在内 ⇒ 该低 cos 无规则信息量 (统计现象); "
                     f"落在外 ⇒ 有信号")
    tr_rule = [v for n, d in tr.items() if n != "V0" for v in d.values() if v is not None]
    tr_v0 = [v for v in tr.get("V0", {}).values() if v is not None]
    if tr_rule:
        hi = [v for v in tr_rule if v > 0.2]
        judge.append(f"L3 机制 (训练窗径, 同窗同权重 vs 真 −g): 规则变体 V1..V5/V5r cos 范围 "
                     f"[{min(tr_rule):.3f}, {max(tr_rule):.3f}], >0.2 的测量 {len(hi)}/{len(tr_rule)}"
                     f" — 高 ⇒ eval 径低 cos 系窗口不匹配 (统计), 非规则破坏方向"
                     + (f"; V0 (100 步累积 vs 单窗 −g, 时间跨度不匹配仅参考) cos="
                        f"{[round(v, 3) for v in tr_v0]}" if tr_v0 else ""))
    out["judge"] = judge
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=TARGET_STEPS)
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--additions-only", action="store_true",
                    help="审查裁决附加项预检: 不跑 500 步; 只算 V5 修复自检 + 随机 "
                         "baseline + 训练窗 −g 对照 (约 1-2 分钟); 产物锚定 "
                         "out/exp128_additions_precheck/, 不占用 v 版本号")
    ap.add_argument("--precheck-steps", type=int, default=10,
                    help="预检模式真实轨迹步数 (capture 训练窗对照用)")
    args = ap.parse_args()
    steps = args.steps
    dev = torch.device(args.device)

    tag = "exp128_phase1"
    if args.additions_only:
        _PRECHECK["on"] = True
        pre_dir = resolve_path("out/exp128_additions_precheck")
        os.makedirs(pre_dir, exist_ok=True)
        pin_run_dir(pre_dir)
        print(f"exp128: 预检模式 (--additions-only); 产物目录锚定 {pre_dir} "
              f"(pin_run_dir, 不占用 v 版本号; v457 保留给正式重跑)", flush=True)
    stdout_path = run_file(tag + "_stdout.log")
    logf_console = open(stdout_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = logf_console
    sys.stderr = logf_console

    ck = resolve_path(CKPT)
    md5_now = _md5(ck)
    if md5_now != CKPT_MD5:
        print(f"exp128: FATAL md5 硬门不一致: 实测 {md5_now} != 预期 {CKPT_MD5}; 拒跑", flush=True)
        sys.exit(2)
    dep = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp124_epsb_amp.py")
    dep_sha = hashlib.sha256(open(dep, "rb").read()).hexdigest()
    self_sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    print(f"exp128: T_arm=on lr={LR} trust_rho={TRUST_RHO} steps={steps} "
          f"additions_only={args.additions_only} device={args.device}\n"
          f"exp128: ckpt={ck}\nexp128: md5={md5_now} (硬门 PASS)\n"
          f"exp128: 依赖 exp124_epsb_amp.py sha256={dep_sha}\n"
          f"exp128: 执行件 sha256={self_sha}", flush=True)

    ReadoutMixin._build_lm_signal = _raw_temp_build_lm_signal
    ReadoutMixin._update_lm_head = _update_lm_head_e3u_tel
    print("exp128: monkeypatch: T 臂 ON (raw T=1 CE, R2 轨迹) + E3 复刻+U 信任域 rho=0.25",
          flush=True)

    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=S_MAX, lm_freeze_w1=False)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    net = HybridPCNet.load(ck, cfg).to(dev)
    net.eval()
    net._migrate_expand = False
    print(f"exp128: net loaded; d_h={net.d_h}", flush=True)

    gauge, _, _ = build_gauge(args.data, dev)
    eval_wins = gauge["tail"][:N_EVAL_WIN]
    print(f"exp128: eval_wins={len(eval_wins)}", flush=True)

    out_json = run_file(tag + ("" if not args.additions_only else "_additions_precheck") + ".json")
    snaps = {}
    W_series = {}
    W_series[0] = net.W_lm.data.clone()  # 基线 (step0 权重); V0@100 需 W_series[0]

    t0 = time.time()
    ds = ByteDataset(args.data, max_length=S_MAX, max_samples=1270000)
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    print(f"exp128: data n={len(ds)}", flush=True)

    # 审查裁决附加项 1: 随机 baseline 窗口 (固定 100 个, 与训练步窗口不相交)
    rand_wins = [ds[idxs[j]] for j in range(1000, 1100)]

    if args.additions_only:
        # ── 预检: V5 修复自检 + 随机 baseline (基准权重) + 训练窗对照 (前 K 步) ──
        print("exp128 预检 [1/3]: V5 修复自检 + 随机 baseline (基准权重, ckpt 快照)...",
              flush=True)
        g0, h0, p0, t0oh, _ = grad_measure_modulated(ck, cfg, dev, eval_wins)
        ng0 = (-g0).flatten()
        n_eval0 = h0.shape[0]
        V5_0 = (h0.transpose(-2, -1) @ (t0oh - p0)) / n_eval0
        v5_eval_cos = _cos(V5_0.flatten(), ng0)
        rb0 = random_window_baseline(ck, cfg, dev, rand_wins, ng0)
        print(f"exp128 预检 [1/3]: V5(评测窗,基准权重) cos={v5_eval_cos:.6f} "
              f"(判别器自检; 期望 ≈1) | shapes h_eval={list(h0.shape)} ng={list(ng0.shape)}",
              flush=True)
        print(f"exp128 预检 [1/3]: 随机 baseline n={rb0['n']} mean={rb0['cos_mean']:.4f} "
              f"std={rb0['cos_std']:.4f} |cos|mean={rb0['abs_mean']:.4f} "
              f"|cos|p95={rb0['abs_p95']:.4f} (理论锚 1/√255≈{rb0['theory_anchor']['inv_sqrt_255']:.3f}, "
              f"√(255/51000)≈{rb0['theory_anchor']['sqrt_rows_over_evalrows']:.3f})",
              flush=True)

        print(f"exp128 预检 [2/3]: 训练窗 −g 对照 (前 {args.precheck_steps} 步真实轨迹步, "
              f"patch 每步采集)...", flush=True)
        for i in range(1, args.precheck_steps + 1):
            CUR_STEP["i"] = i
            x = ds[idxs[(i - 1) % len(idxs)]].unsqueeze(0).to(dev)
            net.learn(x, force_perception=(i == 1))
            net.sync_gate()

        print("exp128 预检 [3/3]: 汇总训练窗径 cos...", flush=True)
        pre_rows = []
        for i in sorted(TRAIN_CTL):
            tc = TRAIN_CTL[i]
            it = STEP_INTERMEDIATES.get(i)
            if it is None or "neg_g_train" not in tc:
                pre_rows.append({"step": i, "errs": tc.get("errs", "interm 缺失")})
                continue
            tensors, verrs, vshapes = rule_variant_tensors(it, dev)
            ngt = tc["neg_g_train"]
            v5t = tc["V5_train"].float().flatten()
            row = {"step": i, "mask_mean": tc["mask_mean"], "n_rows": tc["n_rows"],
                   "shapes": {**tc["shapes"], **vshapes},
                   "cos_train": {"V5": _cos(v5t, ngt)},
                   "cos_vs_v5train": {}, "errs": verrs}
            for name in ("V1", "V2", "V3", "V4", "V5r"):
                V = tensors.get(name)
                if V is None:
                    continue
                Vf = V.flatten()
                row["cos_train"][name] = _cos(Vf, ngt)
                row["cos_vs_v5train"][name] = _cos(Vf, v5t)
            pre_rows.append(row)
            ct = row["cos_train"]

            def _f(v):
                return f"{v:.4f}" if isinstance(v, float) else "-"

            print(f"exp128 预检 step {i}: mask_mean={row['mask_mean']:.4f} | "
                  f"cos_train V1={_f(ct.get('V1'))} V2={_f(ct.get('V2'))} "
                  f"V3={_f(ct.get('V3'))} V4={_f(ct.get('V4'))} "
                  f"V5r={_f(ct.get('V5r'))} V5={_f(ct.get('V5'))}",
                  flush=True)

        pre = {
            "exp": "exp128_phase1_additions_precheck",
            "mode": "additions-only (审查裁决附加项预检; 不跑 500 步; v457 保留给正式重跑)",
            "device": args.device,
            "device_note": "v453-v456 历代运行均为 GPU (--device cuda, exp126 同款默认); "
                           "本字段为运行参数自报 (v456 线无此字段, 本次补齐)",
            "lr": LR, "trust_rho": TRUST_RHO, "T_arm": True,
            "ckpt_md5": md5_now, "ckpt_md5_gate": "PASS",
            "dep_exp124_sha256": dep_sha, "self_sha256": self_sha,
            "precheck_steps": args.precheck_steps,
            "V5_fix_selfcheck_eval": {"cos": v5_eval_cos,
                                      "note": "V5 修复版 (hᵀ(target−probs)/N_rows ≡ −g) 在基准 "
                                              "权重评测窗的自检; 判别器预览"},
            "random_baseline_base": rb0,
            "train_window_ctl": pre_rows,
            "secs": round(time.time() - t0, 1),
            "iron_rules": "crates/ 零改动 (进程内 monkeypatch); md5 硬门; 全 fp16 训练 "
                          "(统计 fp32 只读零写回); 权重不落盘; pin_run_dir 锚定预检目录",
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(pre, f, ensure_ascii=False, indent=1)
        v5t_vals = [r["cos_train"].get("V5") for r in pre_rows
                    if isinstance(r.get("cos_train"), dict) and r["cos_train"].get("V5") is not None]
        v5t_mean = (sum(v5t_vals) / len(v5t_vals)) if v5t_vals else None
        print(f"exp128 预检首句: V5修复自检(评测窗,基准权重)cos={v5_eval_cos:.4f} | "
              f"随机baseline n={rb0['n']} mean={rb0['cos_mean']:.4f} "
              f"|cos|p95={rb0['abs_p95']:.4f} | V5训练窗cos均值="
              f"{v5t_mean if v5t_mean is not None else float('nan'):.4f} "
              f"(vs 同窗 autograd −g) | 预检结论见 json {out_json}", flush=True)
        print("exp128: DONE (预检)", flush=True)
        logf_console.close()
        return

    results = []
    for i in range(1, steps + 1):
        CUR_STEP["i"] = i
        x = ds[idxs[(i - 1) % len(idxs)]].unsqueeze(0).to(dev)
        net.learn(x, force_perception=(i == 1))
        net.sync_gate()
        if i % 100 == 0:
            W_series[i] = net.W_lm.data.clone()
            sp = run_file(tag + f"_snapW{i}.safetensors")
            net.save(sp)
            snaps[i] = sp
            # 计算 6 变体 cos + 两项附加项 (compute_variants 内部已容错, 不会整轮崩)
            res = compute_variants(i, snaps, W_series, eval_wins, cfg, dev,
                                   rand_wins=rand_wins)
            results.append(res)
            c = res.get("cos", {})
            ct = res.get("cos_train", {})
            rb = res.get("additions", {}).get("random_baseline")
            if c:
                print(f"exp128: step {i}/{steps} cos V0..V5 = "
                      f"{c.get('V0','-')}/{c.get('V1','-')}/{c.get('V2','-')}/"
                      f"{c.get('V3','-')}/{c.get('V4','-')}/{c.get('V5','-')} "
                      f"(V5r={c.get('V5r','-')})", flush=True)
                print(f"exp128: step {i}/{steps} 训练窗径 cos V0..V5 = "
                      f"{ct.get('V0','-')}/{ct.get('V1','-')}/{ct.get('V2','-')}/"
                      f"{ct.get('V3','-')}/{ct.get('V4','-')}/{ct.get('V5','-')} "
                      f"(V5r={ct.get('V5r','-')})"
                      + (f" | 随机baseline mean={rb['cos_mean']:.4f} "
                         f"|cos|p95={rb['abs_p95']:.4f}" if rb else ""), flush=True)
            else:
                print(f"exp128: step {i}/{steps} 无 cos (errs={res.get('errs')})", flush=True)
            # 清理瞬态快照
            try:
                os.remove(sp)
            except OSError:
                pass

    # ── 指认方向破坏层 ──
    def identify(seq):
        # seq: list of (variant_name, cos); 找第一个从 <0.1 跳到 >0.3 的变体
        for idx in range(1, len(seq)):
            if seq[idx - 1][1] < 0.1 and seq[idx][1] > 0.3:
                return seq[idx][0], idx
        return None, None

    verdict = []
    for res in results:
        c = res.get("cos", {})
        seq = [(n, c[n]) for n in ("V0", "V1", "V2", "V3", "V4", "V5") if n in c]
        layer, at = identify(seq)
        verdict.append({"step": res["step"],
                        "series": {n: round(cv, 4) for n, cv in seq},
                        "jump_variant": layer, "jump_at_index": at,
                        "cos_train": {n: round(cv, 4) for n, cv in res.get("cos_train", {}).items()},
                        "errs": res.get("errs", {}), "shapes": res.get("shapes", {})})

    va = layered_verdict(results)
    out = {
        "exp": "exp128_phase1",
        "device": args.device,
        "device_note": "v453-v456 历代运行均为 GPU (--device cuda, exp126 同款默认); "
                       "本字段为运行参数自报 (v456 线无此字段, 本次补齐)",
        "lr": LR, "trust_rho": TRUST_RHO, "T_arm": True,
        "ckpt_md5": md5_now, "ckpt_md5_gate": "PASS",
        "dep_exp124_sha256": dep_sha, "self_sha256": self_sha,
        "variants": ["V0", "V1", "V2", "V3", "V4", "V5", "V5r"],
        "variant_defs": {
            "V0": "W_lm(step)-W_lm(step-100) 含 cap/trust 累积 effective ΔW",
            "V1": "rms_norm(h)ᵀ@(err_scaled·contrast_w·d_t·mask)·√d_h (去 cap/trust)",
            "V2": "hᵀ@(err_scaled·contrast_w·d_t·mask) (去 rms_norm 抹幅度)",
            "V3": "hᵀ@(err_scaled·d_t·mask) (去 contrast)",
            "V4": "hᵀ@(err_scaled·mask) (去 d_t; 本轨迹 d_t=1 ⇒ V3≡V4)",
            "V5": "hᵀ@(target−softmax(logits_lm))·mask ≡ −g 判别器 (测量框架有效性验证)",
            "V5r": "hᵀ@eps_total·mask (隔离 err 单位能归一)",
        },
        "neg_g_def": "autograd CE(logits_lm 调制后, target); w.r.t. W_lm (fp32 叶克隆零写回)",
        "coordinate_note": "V1..V4 用训练步 intermediates; V5/V5r/−g 用评测窗 (固定 200 窗). "
                           "err 坐标: R2 T 臂 eps=raw-CE; −g/V5 用调制 logits_lm 真 err — 差异显式标注.",
        "additions_note": "审查裁决 (2026-10-20) 附加项: (1) 随机 baseline — 100 个随机单窗 "
                          "(seed-0 shuffled 流 idxs[1000:1100], 与训练步 1..500 窗口不相交) 的解析 "
                          "delta 梯度 (V5 同公式) vs 全批量 −g 的 cos 分布; (2) 训练窗 −g 对照 — "
                          "与 ΔW/V1..V4 同一窗口、同一更新前权重下的 autograd 真 −g (经完整调制 "
                          "Jacobian) + 解析 delta (V5 同公式)。cos_train=变体 vs 训练窗 autograd "
                          "−g; cos_vs_v5train=变体 vs 训练窗解析 delta。权重时点差异: 评测窗径 −g "
                          "用更新后快照 (既有口径), 训练窗径 −g 用更新前权重 (与 V1..V4 同权重), "
                          "≤1 步 lr, 显式标注。V5 升级为判别器: 其评测窗 cos 验证测量框架是否有效。",
        "results": results,
        "verdict": verdict,
        "verdict_additions": va,
        "secs": round(time.time() - t0, 1),
        "iron_rules": "crates/ 零改动 (进程内 monkeypatch); 不动 v244/迁移件 (md5 硬门); "
                      "全 fp16 训练 (统计/仪器 fp32 只读零写回); 权重不落盘 (瞬态快照即删); "
                      "run_file 自增新目录",
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"exp128: 写出 {out_json}", flush=True)
    for j in va.get("judge", []):
        print(f"exp128 判词: {j}", flush=True)
    # 首句 (审查裁决执行口径: V5 评测窗/训练窗 cos + 随机 baseline + V0..V4 落位 + 分层判词)
    v5e = va["L1_framework"]["V5_eval_cos"]
    v5tr = va["L1_framework"]["V5_train_cos"]
    rbs = va["L2_baseline"].get("random", {})
    ex_step = next(iter(rbs)) if rbs else None
    rb_txt = (f"随机baseline(step{ex_step}) mean={rbs[ex_step]['mean']:.4f} "
              f"|cos|p95={rbs[ex_step]['abs_p95']:.4f}") if ex_step else "随机baseline=n/a"
    n_in = sum(1 for d in va["L2_baseline"].get("V0_V4_vs_baseline", {}).values()
               for v in d.values() if v["inside_abs_p95"])
    n_tot = sum(len(d) for d in va["L2_baseline"].get("V0_V4_vs_baseline", {}).values())
    print("exp128 首句: " + " | ".join([
        f"V5判别器 评测窗cos={ {k: round(v, 4) for k, v in v5e.items()} }",
        f"V5判别器 训练窗cos={ {k: round(v, 4) for k, v in v5tr.items()} }",
        rb_txt,
        f"V0..V4落baseline内 {n_in}/{n_tot}",
        "判词: " + " ;; ".join(va.get("judge", [])),
    ]), flush=True)
    print("exp128: DONE", flush=True)
    logf_console.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        crash = resolve_path("out/exp128_phase1_crash.txt")
        with open(crash, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        sys.stderr.write(traceback.format_exc())
        sys.exit(1)
