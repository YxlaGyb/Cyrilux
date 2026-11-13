"""
exp133: K=16 x 扩展温度扫描 x 双基线 (rho=0.02)
========================================================================================

预登记 (跑前锁死): docs/plan/exp133_preregistration.md, sha256
`410161C880FAA9902A4437EDEABF36A4E251547071FF9CBB2074944590305D0B`。

立项 (exp132 审查裁决, 2026-10-24):
  exp132 PASS: P3 成立 (||T||/||Sigma_e|| 2.241->1.078->0.739->0.515 严格单调, rho 是真旋钮),
  P1 证伪 (cos_osc 25x rho 变化纹丝不动), trust 四臂 100% 绑定 (lr 空转再确认)。
  裁决四处修正: (1) 跨臂 s* 比较作废, 改用臂内自比; (2) R3 与 exp125c U 的 6.9035 逐位
  相同必须核实 (离线三种假说均无法判定, 移交 exp133 §五 附带检验); (3) 振荡归因改写为
  "窗口序列 x 更新规则放大" (窗口 -0.031 -> 更新 -0.24, 8x 放大在更新侧); (4) Phase 2
  "cos_osc >= -0.10" 判据是按 rho=0.25 写的, 在 rho=0.02 下 cap 不承重, 不适用。

  本轮: 在 rho=0.02 (cap 绑定率 82.9%, mean|gamma|=0.00164) 这个 cap 几乎不承重的
  regime 上点火 K=16, 回答"压振荡 + 抬累积 cos"; 同时用扩展扫描解掉 exp132 三臂
  s* 全顶在 1.6 上界的系统性截断 (exp132 局限 1 授权的口径变更, 另立预登记)。

两臂 (均 rho=0.02, lr=0.05, 500 步, seed=0, 全局 cap):
  A (基线) K=1  |  B (干预) K=16   -- 只动 K 一个变量; A = exp132 R3 同配置重跑

扩展网格: [0.2, 0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0] (exp125c 原 9 点 + 2.0/2.5/3.0),
两臂同一网格; 口径逐字复用 exp125c post_sweeps (进程内替换模块 SWEEP_SCALES, 函数体不改)。

判据 (预登记 §三.2): L1 = B 的 s* bpb 严格 < A (臂内自比, 不用 exp125c U 旧数);
L2 = B s* bpb < 5.0; C4' = B cos_osc >= -0.10 (预期落支, 非通过门);
C5 = B cos(Sigma_from-start dW_eff, Sigma_from-start (-g)) > 0.5 (≡ exp132 的 C4 定义,
编号更名避免与 L 层混淆); C6 = B 未触发 R4; C7 = B word_hits >= 1。

预写可证伪预测 (预登记 §四): Q1 B 的 s* > 1.6; Q2 B 的 cos_osc >= +0.5;
Q3 B 的 ||T||/||Sigma_e|| >= 1.2 且不触发 R4; Q4 B 的 C5 比 A 提升 >= 50%。

附带检验 (预登记 §五, 不是判据): A 臂 sweep 全表 vs out/v474-20261024-165104/exp132_R3.json
逐点比对 — A = R3 同配置重跑, 逐位相同 ⇒ 训练+评测链路确定性坐实 (并分割 exp132 v2
修正 2 的假说空间: 碰撞重现 ⇒ 系统性评测现象; 不重现 ⇒ 偶发)。

派生说明: 由 scripts/exp132_rho_sweep.py 复制后改动 - 文档头; 臂参数改为 A/B (均
rho=0.02, K=1/16); 温度扫描改扩展网格 (进程内替换 e125c.SWEEP_SCALES); 判据表与
首句按 exp133 预登记重写; 段判词补 ||W|| 段首末; 新增 --peer-json (B 臂判 L1/Q4 用)
与 R3 确定性附带检验。**训练路径 / 更新规则 / dump 日程 / md5 硬门 / 保真目标 /
中止规则 / 全部七量仪器逐字节未改。**

铁律: crates/ 零改动 (进程内 monkeypatch); 不动 v244/迁移件 (md5 硬门); 全 fp16 训练
(仪器 fp32 只读零写回); 权重不落盘 (瞬态快照即删; 扫描在内存中做); run_file 自增;
运行不 kill; device 入账。
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
# exp133 预登记 §三.1: 扩展温度网格 (exp125c 原 9 点 + 2.0/2.5/3.0; 解 exp132 局限 1
# 的 s*=1.6 上界截断; 两臂同一网格, 进程内替换 e125c.SWEEP_SCALES, post_sweeps 函数体不改)
EXT_SCALES = [0.2, 0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0]
V451_TEL = "out/v451-20261020-194546/exp126_R1_telemetry.jsonl"  # 逐步余弦交叉核对源

_TARM = {"on": False}   # R1 = T 臂 OFF (部署 lc 训练路径)
_orig_build_lm_signal = ReadoutMixin._build_lm_signal
_TEL = {"fresh": False}
_TEL_RAW = {}

# ─────────── exp132 臂参数 (预登记 §二, 跑前锁死) ───────────
# exp133 臂参数 (预登记 §二, 跑前锁死): 两臂均 rho=0.02, 只动 K 一个变量
# (A: K=1 不批处理; B: K=16 滑动平均), cap=全局标量 (保留作安全网)。
ARM = {"name": "A?", "rho": 0.02, "batch_k": 1, "cap_decouple": False}

PRE_BPB_TARGET = 7.664317404722619   # 协议一致性 (exp129/exp130/exp131 三次实测同一值)

# exp132 强制条款 5: 新增中止 R4 — ‖W‖ > 0.95×cap 天花板 连续 10 步 ⇒ 离开 trust 区制
# cap10 上限 = 10 × sqrt(256) = 160.0; 0.95× = 152.0
CAP_CEILING = 160.0
R4_THRESHOLD = 0.95 * CAP_CEILING
R4_STREAK = 10

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
          "prev_neg_g": None,      # exp130: 用于逐步 cos(−g_i, −g_{i−1})
          # exp131: K 窗滑动平均提案的缓冲区 (fp32 克隆, 只读零写回)
          "batch_buf": []}


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
        # exp131: 本窗自身提案的对齐 (批处理后 cos_rule 指"被应用的平均提案",
        # 两者之差即批处理的每步对齐代价 — 预登记 §七 的预期代价, 不是失败)
        e129["cos_rule_rawwin"] = e129["cos_rule"]
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

    # ═══════════ exp131 干预 (1): K 窗滑动平均提案 (预登记 §三.1) ═══════════
    # m_i = mean(最近 min(K,i) 个已应用规则提案)。K=1 时 m_i ≡ dW_raw (fp16→fp32→fp16 无损),
    # 故 B1 语义不变。注入点在 trust 之前: 之后 trust → cap → 更新 整条流水线一字不改。
    K = ARM["batch_k"]
    batch_n = None
    if K > 1:
        buf = EXP129["batch_buf"]
        buf.append(dW_lm.detach().float().clone())
        while len(buf) > K:
            buf.pop(0)
        batch_n = len(buf)
        m_i = torch.stack(buf, dim=0).mean(dim=0)
        dW_lm = m_i.to(dW_lm.dtype)          # 回落到训练路径 dtype (fp16), 保持零写回
        del buf, m_i
    e129["batch_n"] = batch_n
    e129["batch_k"] = K
    if batch_n is not None and EXP129.get("_cur") is not None:
        ng = EXP129["_cur"]["neg_g"]
        e129["cos_rule"] = _cos(dW_lm.detach().float().flatten(), ng.flatten())

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
    if step_norm_pre > ARM["rho"] * wpre_norm_f:
        trust_scale = (ARM["rho"] * wpre_norm_f) / (step_norm_pre + 1e-12)
        dW_lm = dW_lm * trust_scale
        net._tel_dWlm_t = dW_lm.abs().mean().detach()

    # ── exp129 仪器 B: dW_after_trust (trust 后提案) ──
    dW_after_trust = dW_lm.detach().clone()

    net.W_lm.data += dW_lm * LR
    rn_post = net.W_lm.data.norm(dim=1)
    mx_lm = rn_post.max()
    # cap_scale = 原全局标量值; coupled 臂施加它, decoupled 臂**只记录不施加** (预登记 §三.2)
    cap_scale = float(torch.minimum(torch.ones_like(mx_lm),
                                    10.0 / (mx_lm + 1e-6)).to(torch.float16))
    row_cap = None
    row_cap_n_scaled = None
    row_cap_mean = None
    if ARM["cap_decouple"]:
        # 逐行 row_cap_r = min(1, 10/‖W[r]‖): 只缩超限行, 未超限行一个字节不动
        row_cap = torch.clamp(10.0 / (rn_post.float() + 1e-6),
                              max=1.0).to(net.W_lm.data.dtype)
        net.W_lm.data.mul_(row_cap.unsqueeze(-1))
        row_cap_n_scaled = int((row_cap < 1.0 - 1e-6).sum())
        row_cap_mean = float(row_cap.float().mean())
    else:
        net.W_lm.data.mul_(
            torch.minimum(torch.ones_like(mx_lm), 10.0 / (mx_lm + 1e-6)).to(torch.float16)
        )
    e129["cap_mode"] = "decouple" if ARM["cap_decouple"] else "global"
    e129["row_cap_n_scaled"] = row_cap_n_scaled
    e129["row_cap_mean"] = row_cap_mean
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
            # ── exp130/131 分解: dW_eff = D·(lr·t·dW_raw) + (D−I)·W_pre ──
            #   coupled (B1/B3): D = c·I (全局标量 c=cap_scale) ⇒ 退化为 exp130 的
            #     dW_eff = β·dW_raw + γ·W_pre, β=c·lr·t, γ=c−1, 与 exp130 逐位同口径。
            #   decoupled (B2/B4): D = diag(row_cap) ⇒ 行权重形式, 仍是精确恒等式。
            beta = cap_scale * LR * trust_scale
            gamma = cap_scale - 1.0
            e129["beta"] = beta
            e129["gamma"] = gamma
            e129["Wpre_norm"] = float(wlm_pre.float().norm())
            if row_cap is not None:
                rc = row_cap.float().unsqueeze(-1)
                betar_v = rc * (dW_raw.float() * (LR * trust_scale))
                rad_v = (rc - 1.0) * wlm_pre.float()
            else:
                betar_v = dW_raw.float() * beta
                rad_v = wlm_pre.float() * gamma
            seg = EXP129["seg"]
            allc = EXP129["all"]
            for acc in (seg, allc):
                acc["sn_e"] += float(dW_eff.norm())
                acc["sn_betar"] += float(betar_v.norm())
                acc["sn_rad"] += float(rad_v.norm())
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
              "trust_rotation", "cos_osc_raw", "cos_rule_rawwin"):
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
    ap.add_argument("--arm", default="A", choices=["A", "B"],
                    help="exp133 臂 (预登记 §二, 均 rho=0.02): A 基线 K=1 (= exp132 R3 同配置) / "
                         "B 干预 K=16")
    ap.add_argument("--peer-json", default=None,
                    help="对臂 json 路径 (B 臂判 L1/Q4 时传 A 臂 exp133 json; A 臂不传)")
    ap.add_argument("--rho", type=float, default=None,
                    help="信任域 rho (默认由 --arm 推出)")
    ap.add_argument("--batch-k", type=int, default=None,
                    help="[exp132 固定为 1] 仅为兼容保留; 非 1 需显式声明并记偏离")
    ap.add_argument("--cap-decouple", default=None, choices=["0", "1"],
                    help="[exp132 固定为全局] 仅为兼容保留; 置 1 需显式声明并记偏离")
    args = ap.parse_args()
    dev = torch.device(args.device)
    smoke = args.smoke > 0
    steps = args.smoke if smoke else args.steps

    # ── 臂参数 (预登记 §二, 锁死): 两臂均 rho=0.02, 只动 K 一个变量 ──
    _ARM_DEF = {"A": 0.02, "B": 0.02}
    _rho = _ARM_DEF[args.arm]
    if args.rho is not None:
        _rho = args.rho
    ARM["name"] = args.arm
    ARM["rho"] = _rho
    ARM["batch_k"] = (16 if args.arm == "B" else 1) if args.batch_k is None else args.batch_k
    ARM["cap_decouple"] = False if args.cap_decouple is None else (args.cap_decouple == "1")
    if ARM["batch_k"] not in (1, 16) or ARM["cap_decouple"]:
        print(f"exp133: WARN 偏离预登记 — batch_k={ARM['batch_k']} "
              f"cap_decouple={ARM['cap_decouple']} (预登记 §二: A=K=1 / B=K=16, 均全局 cap)",
              flush=True)

    peer = None
    if args.peer_json:
        try:
            pj = json.load(open(resolve_path(args.peer_json), encoding="utf-8"))
            peer = {"arm": pj.get("arm", {}).get("name"),
                    "sstar_bpb": (pj.get("sweep") or {}).get("best", {}).get("bpb"),
                    "sstar_scale": (pj.get("sweep") or {}).get("best", {}).get("scale"),
                    "C5": (pj.get("coherence_all") or {}).get("cos_sumraw_sumg_eff"),
                    "cos_osc_weighted_mean": (pj.get("predictions") or {}).get("cos_osc_weighted_mean")}
            print(f"exp133: peer 对臂 {peer['arm']} (s* bpb={peer['sstar_bpb']}, "
                  f"C5={peer['C5']}) 从 {args.peer_json} 读入", flush=True)
        except Exception as ex:  # noqa: BLE001
            peer = None
            print(f"exp133: WARN peer json 读取失败 (L1/Q4 将记待批): {ex!r}", flush=True)

    tag = "exp133_" + args.arm
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
    print(f"exp131: 臂 {ARM['name']} (batch_k={ARM['batch_k']}, "
          f"cap={'解耦-只缩超限行' if ARM['cap_decouple'] else '全局标量'}) "
          f"lr={LR} steps={steps} smoke={smoke} device={args.device}\n"
          f"exp131: ckpt={ck}\nexp131: md5={md5_now} (硬门 PASS)\n"
          f"exp131: 依赖 exp124_epsb_amp.py sha256={dep_sha}\n"
          f"exp131: 依赖 exp126_direction_teardown.py sha256={dep126_sha} (规则/仪表逐字复用源)\n"
          f"exp131: 执行件 sha256={self_sha}", flush=True)

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
        # exp131 协议一致性门 (预登记 §六): 新臂无逐步余弦 bit_identical 门, 但 pre bpb 必须复现
        _pb_ok = (pre is not None
                  and abs(pre["bpb_s1"] - PRE_BPB_TARGET) < 1e-6)
        print(f"exp131: pre bpb@s1={pre['bpb_s1']:.6f} top1={pre['top1_s1']:.6f} "
              f"(协议一致性目标 {PRE_BPB_TARGET}, 容差 1e-6, "
              f"{'PASS' if _pb_ok else 'FAIL'})", flush=True)
        if not _pb_ok:
            print("exp131: FATAL pre bpb 协议一致性 FAIL; 拒跑", flush=True)
            sys.exit(3)
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
    r4_streak = 0
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
               # ── exp131 新增 per-step 键 ──
               "batch_k": e129.get("batch_k"), "batch_n": e129.get("batch_n"),
               "cos_rule_rawwin": e129.get("cos_rule_rawwin"),
               "cap_mode": e129.get("cap_mode"),
               "row_cap_n_scaled": e129.get("row_cap_n_scaled"),
               "row_cap_mean": e129.get("row_cap_mean"),
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
        # ── exp132 新增中止 R4 (预登记 §三.5): ‖W‖ > 0.95×cap 天花板 连续 10 步 ──
        _wn = _gt("_tel_Wlm_norm_t")
        if _wn is not None and _wn > R4_THRESHOLD:
            r4_streak += 1
        else:
            r4_streak = 0
        if r4_streak >= R4_STREAK:
            abort = (f"R4: ‖W‖={_wn:.3f} > 0.95×cap 天花板 {R4_THRESHOLD:.1f} "
                     f"连续 {r4_streak} 步 — 离开 trust 区制")
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
            _seg_rows = [rw for rw in tel_rows
                         if (seg_idx - 1) * 100 < rw["step"] <= seg_idx * 100]
            sv["W_lm_norm_first"] = _seg_rows[0].get("W_lm_norm") if _seg_rows else None
            sv["W_lm_norm_last"] = _seg_rows[-1].get("W_lm_norm") if _seg_rows else None
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
        print(f"exp133: post bpb@s1={post['bpb_s1']:.6f} top1={post['top1_s1']:.6f} "
              f"(新臂; v460 参考值 {FIDELITY_TARGET}, 不复刻)", flush=True)
        # exp131 C3: word_hits (15 常见双字词, T=0.8, 3 样本) — 与 exp124/exp126 同口径
        word_hits = e124.word_hits_of(final_lines) if final_lines else None
        print(f"exp133: word_hits(T=0.8, 3 样本) = {word_hits}", flush=True)
        # ── exp133 强制条款 1 (预登记 §三.1): 扩展温度扫描, 口径逐字复用 exp125c ──
        # 在内存中做 (权重不落盘); 只读权重前向, 不写回, 训练循环已结束故不影响轨迹。
        try:
            import exp125c_scale_direction_teardown as e125c
            e125c.SWEEP_SCALES = list(EXT_SCALES)  # exp133 §三.1: 扩展网格 (进程内替换)
            sweep = e125c.post_sweeps(net, dev, eval_wins, tag + "_sweep")
            sweep["scales"] = list(EXT_SCALES)
            print(f"exp133: 温度扫描 best s*={sweep['best']['scale']} "
                  f"(T={sweep['best']['temp']}) bpb={sweep['best']['bpb']:.4f} "
                  f"vs s=1 bpb={sweep['bpb_s1']:.4f} gain={sweep['gain_ratio']:.3f} "
                  f"| bias_off s1={sweep['bias_off_s1']['bpb']:.4f}", flush=True)
            print("exp133: 扫描全表 " + " ".join(
                f"s={r['scale']}:{r['bpb']:.4f}" for r in sweep["sweep"]), flush=True)
        except Exception as ex:  # noqa: BLE001
            import traceback as _tb
            sweep = {"instrument_fail": repr(ex), "tb": _tb.format_exc()}
            print(f"exp133: WARN 温度扫描失败 (不中止): {ex!r}", flush=True)
    else:
        post = None
        word_hits = None
        sweep = None

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

    # ── exp133 §五 附带检验: A 臂 (= R3 同配置) sweep 全表 vs exp132 R3 逐点比对 ──
    r3_check = {}
    try:
        p_r3 = resolve_path(os.path.join("out", "v474-20261024-165104", "exp132_R3.json"))
        j3 = json.load(open(p_r3, encoding="utf-8"))
        r3_sweep = {r_["scale"]: r_["bpb"] for r_ in j3["sweep"]["sweep"]}
        mine = {r_["scale"]: r_["bpb"] for r_ in (sweep or {}).get("sweep", [])}
        common = sorted(set(r3_sweep) & set(mine))
        r3_check = {
            "n_common_scales": len(common),
            "per_scale": {str(s_): {"exp133_arm": mine[s_], "exp132_R3": r3_sweep[s_],
                                    "bit_identical": bool(mine[s_] == r3_sweep[s_])}
                          for s_ in common},
            "all_bit_identical": bool(common) and all(mine[s_] == r3_sweep[s_] for s_ in common),
            "note": ("exp133 预登记 §五: A 臂 = exp132 R3 同配置 (rho=0.02, K=1, 同 ckpt/数据/"
                     "种子/协议) 重跑; 交集 scale 逐位相同 => 训练+评测链路确定性坐实; "
                     "同时分割 exp132 v2 修正 2 假说空间 (与 U 的碰撞重现 => 系统性评测现象)")}
        print(f"exp133: 附带检验 A vs R3 — 共同 scale {len(common)} 个, "
              f"all_bit_identical={r3_check['all_bit_identical']}", flush=True)
    except Exception as ex:  # noqa: BLE001
        r3_check = {"instrument_fail": repr(ex)}
        print(f"exp133: WARN R3 附带检验失败 (不中止): {ex!r}", flush=True)

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
    coherence_all["T_over_E"] = (
        coherence_all["sum_rad_norm"] / coherence_all["sum_dw_eff_norm"]
        if coherence_all.get("sum_rad_norm") and coherence_all.get("sum_dw_eff_norm") else None)
    # exp132 强制条款 3: C4 只用 from-start 向量和; 全程与分段分两张表, 禁止混用
    coherence_all["c4_definition"] = ("cos(Σ_from-start dW_eff, Σ_from-start (−g)); "
                                      "分段值见 segment_verdicts[*].cos_cum_g_cum, "
                                      "两者聚合口径不同, 禁止混用 (exp131 v2)")
    coherence_all["cos_sumraw_sumg"] = (
        _cos(allc["sum_dw_raw"].flatten(), allc["sum_neg_g"].flatten())
        if allc["sum_dw_raw"] is not None and allc["sum_neg_g"] is not None else None)
    coherence_all["cos_sumrad_sumg"] = (
        _cos(allc["sum_rad"].flatten(), allc["sum_neg_g"].flatten())
        if allc["sum_rad"] is not None and allc["sum_neg_g"] is not None else None)
    # exp131 C4: 累积 cos(ΣdW_eff, Σg) 全程值 — "修好 vs 改观"的区分判据 (预登记 §四)
    coherence_all["cos_sumraw_sumg_eff"] = (
        _cos(allc["sum_dw_eff"].flatten(), allc["sum_neg_g"].flatten())
        if allc["sum_dw_eff"] is not None and allc["sum_neg_g"] is not None else None)
    # exp131: U/T 相互夹角 (decoupled 臂为行权重形式, 仍是精确恒等式)
    if allc["sum_betar"] is not None and allc["sum_rad"] is not None:
        coherence_all["cos_betar_rad"] = _cos(allc["sum_betar"].flatten(),
                                              allc["sum_rad"].flatten())
    coherence_all["batch_k"] = ARM["batch_k"]
    coherence_all["cap_mode"] = "decouple" if ARM["cap_decouple"] else "global"
    ca = coherence_all   # 供下方 out 判据与首句引用

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

    _co_wmean = (sum(s["cos_osc"] * s["n_updates"] for s in seg_verdicts)
                 / sum(s["n_updates"] for s in seg_verdicts)) if seg_verdicts else None
    _peer_sstar = peer.get("sstar_bpb") if peer else None
    _peer_c5 = peer.get("C5") if peer else None
    _sstar_bpb = (sweep or {}).get("best", {}).get("bpb") if sweep else None
    _sstar_scale = (sweep or {}).get("best", {}).get("scale") if sweep else None
    _r4_ok = (abort is None or "R4" not in str(abort))

    out = {
        "exp": "exp133", "stage": "k16_extsweep", "run": ARM["name"], "lr": LR,
        "trust_rho": ARM["rho"], "rho": ARM["rho"],
        "coherence_all": coherence_all, "selfcheck_v460": v460_cross,
        "sweep": sweep,
        "peer_arm": peer,
        "r3_determinism_check": r3_check,
        "T_arm": _TARM["on"], "device": args.device, "smoke": smoke,
        "device_note": "历代 GPU (v457 教训: device 入账); 同 exp126 R1 构造, 只改 rho",
        "ckpt_md5": md5_now, "ckpt_md5_gate": "PASS",
        "dep_exp124_sha256": dep_sha, "dep_exp126_sha256": dep126_sha,
        "self_sha256": self_sha,
        "steps_run": i, "target_steps": args.steps, "abort": abort,
        "arm": dict(ARM),
        "fidelity": {"post_bpb_s1": (post or {}).get("bpb_s1") if post else None,
                     "target_bit_identical": FIDELITY_TARGET,
                     "bit_identical": (post is not None
                                       and abs(post["bpb_s1"] - FIDELITY_TARGET) < 1e-9),
                     "note": "exp133 为新臂, 预期与 v460 轨迹分歧; 此处仅记录不复刻"},
        # exp133 判据 (预登记 §三.2): 臂内自比 L1 + 绝对 L2 + 振荡预期 C4' + 累积 C5 + C6 + C7
        "criteria": {
            "L1_b_lt_a_strict": {
                "value": _sstar_bpb, "baseline_A_sstar_bpb": _peer_sstar,
                "pass": bool(peer and _sstar_bpb is not None and _peer_sstar is not None
                             and _sstar_bpb < _peer_sstar),
                "note": ("臂内自比 (B 严格 < A); 不用 exp125c U 的 6.9035 — exp132 审查修正 1: "
                         "跨臂 s* 比较作废; L1<7.0 阈值 PASS 无区分力 (基线与干预都过)")},
            "L2_sstar_bpb_lt_5": {"value": _sstar_bpb,
                                  "pass": bool(sweep and _sstar_bpb is not None
                                               and _sstar_bpb < 5.0)},
            "s1_bpb": {"value": (sweep or {}).get("bpb_s1") if sweep else None,
                       "note": "仅记录; 受 ‖W‖ 尺度污染, 不单独作为判据 (陷阱 6)"},
            "sstar_scale": {"value": _sstar_scale},
            "top1_at_sstar": {"value": (sweep or {}).get("best", {}).get("top1")
                              if sweep else None},
            "C4p_cos_osc_ge_m010": {
                "value": _co_wmean,
                "pass": (_co_wmean is not None and _co_wmean >= -0.10),
                "note": "预期落支, 不是通过门 (预登记 §三.2: K=16 应把 -0.24 压到正值)"},
            "C5_cum_cos_gt_05": {
                "value": ca.get("cos_sumraw_sumg_eff"),
                "pass": (ca.get("cos_sumraw_sumg_eff") is not None
                         and ca["cos_sumraw_sumg_eff"] > 0.5),
                "note": "≡ exp132 的 C4 定义 (cos(Σ_from-start dW_eff, Σ_from-start (−g)), "
                        "全程与分段分两张表禁止混用); 编号更名避免与 L 层混淆"},
            "C6_r4_not_triggered": {"abort": abort, "pass": bool(_r4_ok),
                                    "note": "承重墙安全网 (‖W‖ 未触 152 天花板连续 10 步)"},
            "C7_word_hits_ge_1": {"value": (word_hits if post else None),
                                  "pass": (post is not None and word_hits is not None
                                           and word_hits >= 1)},
        },
        # exp132 强制条款 4 (预登记 §三.4): 更新次数协变量 + 预算匹配标注
        "update_budget": {
            "n_updates": ca.get("N_updates"),
            "baseline_n": 239,
            "ratio_vs_baseline": (ca.get("N_updates") / 239.0) if ca.get("N_updates") else None,
            "matched": bool(ca.get("N_updates") and ca["N_updates"] <= 1.2 * 239),
            "flag": ("更新预算未匹配" if ca.get("N_updates") and ca["N_updates"] > 1.2 * 239
                     else "ok"),
        },
        # exp133 预写可证伪预测 (预登记 §四; Q1/Q2/Q3 单臂可判, Q4 需 peer)
        "predictions": {
            "Q1_sstar_gt_16": {"value": _sstar_scale,
                               "verdict": (bool(sweep and _sstar_scale is not None
                                                and _sstar_scale > 1.6))},
            "Q2_cos_osc_ge_050": {"value": _co_wmean,
                                  "verdict": (_co_wmean is not None and _co_wmean >= 0.5)},
            "Q3_T_over_E_ge_12_no_R4": {"value": ca.get("T_over_E"), "r4_triggered": (not _r4_ok),
                                        "verdict": (ca.get("T_over_E") is not None
                                                    and ca["T_over_E"] >= 1.2 and _r4_ok)},
            "Q4_c5_gain_ge_50pct": {"value": ca.get("cos_sumraw_sumg_eff"),
                                    "baseline_A_C5": _peer_c5,
                                    "verdict": (peer is not None and _peer_c5 is not None
                                                and ca.get("cos_sumraw_sumg_eff") is not None
                                                and ca["cos_sumraw_sumg_eff"] >= 1.5 * _peer_c5)},
            "cos_osc_weighted_mean": _co_wmean,
            "P3_T_over_E": ca.get("T_over_E"),
        },
        "pre_bpb_protocol": {"value": (pre or {}).get("bpb_s1") if pre else None,
                             "target": PRE_BPB_TARGET,
                             "pass": (pre is not None
                                      and abs(pre["bpb_s1"] - PRE_BPB_TARGET) < 1e-6)},
        "xcheck_dWcos_prev": xcheck,
        "xcheck_note": "exp131 为新臂, 逐步余弦与 v451 预期分歧 — 此栏不是失败项",
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
    def fmt_seq(key):
        return " ".join(("n/a" if r.get(key) is None else f"{r[key]:+.4f}")
                        for r in seg_verdicts)

    # ── exp133 首句: 裁决执行口径 — A/B 臂 s* 与 s* bpb / L1-L2 / cos_osc(含 raw) /
    #    C5 / R4 / Q1-Q4 落支 / ‖W‖ 逐段 / cap 绑定 / 附带检验 ──
    _rows = [r for r in tel_rows if r.get("cos_rule") is not None]
    _trust_bind = (sum(1 for r in _rows if (r.get("trust_scale") or 1.0) < 1.0 - 1e-9)
                   / len(_rows)) if _rows else None
    _cap_bind = (sum(1 for r in _rows if (r.get("wlm_cap_scale") or 1.0) < 1.0 - 1e-9)
                 / len(_rows)) if _rows else None
    _cap_mean = (sum(r["wlm_cap_scale"] for r in _rows if r.get("wlm_cap_scale") is not None)
                 / max(1, sum(1 for r in _rows if r.get("wlm_cap_scale") is not None))) \
        if _rows else None
    _trust_mean = (sum(r["trust_scale"] for r in _rows if r.get("trust_scale") is not None)
                   / max(1, sum(1 for r in _rows if r.get("trust_scale") is not None))) \
        if _rows else None
    _W0 = _rows[0]["W_lm_norm"] if _rows else None
    _Wend = _rows[-1]["W_lm_norm"] if _rows else None
    _beh1 = sum(1 for r in tel_rows if r.get("behavior") == 1)
    def _s(v, fmt="+.4f"):
        return "n/a" if v is None else format(v, fmt)

    print("exp133 首句: " + " || ".join([
        f"臂 {ARM['name']} ρ={ARM['rho']} (K={ARM['batch_k']}, cap=全局)",
        f"温度扫描(扩展 {len(EXT_SCALES)} 档): s*={out['criteria']['sstar_scale']['value']} "
        f"s* bpb={out['criteria']['L2_sstar_bpb_lt_5']['value']} "
        f"(L1 B<A 严格: {'PASS' if out['criteria']['L1_b_lt_a_strict']['pass'] else 'FAIL'}"
        + (f" (基线 A={peer['sstar_bpb']:.4f})"
           if peer and peer.get('sstar_bpb') is not None
           else (" (无 peer, 待回执)" if not peer else " (peer 无 s*, 待回执)"))
        + f", L2<5.0: {'PASS' if out['criteria']['L2_sstar_bpb_lt_5']['pass'] else 'FAIL'}) | "
        f"s=1 bpb={fid['post_bpb_s1']} | top1@s*={out['criteria']['top1_at_sstar']['value']} | "
        f"word_hits={out['criteria']['C7_word_hits_ge_1']['value']}",
        f"区制: trust 绑定率={_trust_bind:.3f} (mean t={_trust_mean:.4f}) | "
        f"cap 绑定率={_cap_bind:.3f} (mean cap={_cap_mean:.4f}, mean |γ|={1-(_cap_mean or 1):.4f})",
        f"‖W‖: 首={_W0} 末={_Wend} (末/首={(_Wend/_W0) if _W0 else None:.3f}) 五段首末="
        + " ".join(f"seg{r['segment']}:{r.get('W_lm_norm_first')}→{r.get('W_lm_norm_last')}"
                   for r in seg_verdicts)
        + f" | 更新步={ca['N_updates']} (基线 239, 比值 "
        f"{out['update_budget']['ratio_vs_baseline']:.3f}, {out['update_budget']['flag']}) | "
        f"behavior==1 步数={_beh1}",
        f"振荡: cos_osc 五段={fmt_seq('cos_osc')} | 加权均值="
        f"{_s(out['predictions']['cos_osc_weighted_mean'])} "
        f"(C4' 预期 ≥−0.10, 非通过门: "
        f"{'成立' if out['criteria']['C4p_cos_osc_ge_m010']['pass'] else '不成立'}) | "
        f"cos_osc_raw 五段={fmt_seq('cos_osc_raw')}",
        f"U/T: ‖U‖={ca['sum_betar_norm']:.2f} ‖T‖={ca['sum_rad_norm']:.2f} "
        f"‖Σe‖={ca['sum_dw_eff_norm']:.2f} cos(U,T)={ca.get('cos_betar_rad')} | "
        f"‖T‖/‖Σe‖={ca.get('T_over_E')} (Q3 '≥1.2 且不触发 R4': "
        f"{'成立' if out['predictions']['Q3_T_over_E_ge_12_no_R4']['verdict'] else '不成立'})",
        f"全程 coherence (N={ca['N_updates']}, 1/√N={ca['random_ref_1over_sqrtN']:.4f}): "
        f"raw={ca['coherence_raw']:.4f} (RA {ca['RA']['raw']:.2f}) "
        f"eff={ca['coherence_eff']:.4f} (RA {ca['RA']['eff']:.2f}) "
        f"g={ca['coherence_g']:.4f} (RA {ca['RA']['g']:.2f}) "
        f"rad={ca['coherence_rad']:.4f} | raw/g={ca['ratio_raw_over_g']:.3f}",
        f"C5 (from-start, 全程; ≡ exp132 C4 口径) = {ca.get('cos_sumraw_sumg_eff')} "
        f"(>0.5: {'PASS' if out['criteria']['C5_cum_cos_gt_05']['pass'] else 'FAIL'}) | "
        f"五段 cos_cum_g_cum = {fmt_seq('cos_cum_g_cum')} (两张表分列, 禁止混用) | "
        f"cos_rule_rawwin 五段={fmt_seq('cos_rule_rawwin')} (K=16 每步对齐代价)",
        f"Q1 (s*>1.6): {'成立' if out['predictions']['Q1_sstar_gt_16']['verdict'] else '不成立'} | "
        f"Q2 (cos_osc≥+0.5): {'成立' if out['predictions']['Q2_cos_osc_ge_050']['verdict'] else '不成立'} | "
        f"Q4 (C5 比 A +≥50%): {'成立' if out['predictions']['Q4_c5_gain_ge_50pct']['verdict'] else '不成立'}"
        + (f" (A 臂 C5={peer['C5']})" if peer and peer.get('C5') is not None else " (无 peer)"),
        f"协议一致性: pre_bpb={out['pre_bpb_protocol']['value']} "
        f"({'PASS' if out['pre_bpb_protocol']['pass'] else 'FAIL'}) | md5 硬门 PASS | "
        f"abort={abort}",
        f"附带检验 (预登记 §五): A vs exp132 R3 共同 scale "
        f"{r3_check.get('n_common_scales')} 个, all_bit_identical="
        f"{r3_check.get('all_bit_identical')}",
    ]), flush=True)
    print("exp133 首句(旧字段兼容): " + " || ".join([
        f"臂 {ARM['name']} (K={ARM['batch_k']}, cap={'解耦' if ARM['cap_decouple'] else '全局'})",
        f"全程 coherence (N={ca['N_updates']}, 1/√N={ca['random_ref_1over_sqrtN']:.4f}): "
        f"raw={ca['coherence_raw']:.4f} (RA {ca['RA']['raw']:.2f}) "
        f"eff={ca['coherence_eff']:.4f} (RA {ca['RA']['eff']:.2f}) "
        f"g={ca['coherence_g']:.4f} (RA {ca['RA']['g']:.2f}) "
        f"betar={ca['coherence_betar']:.4f} rad={ca['coherence_rad']:.4f} "
        f"| raw/g={ca['ratio_raw_over_g']:.3f}",
        f"五段 coherence_g: {fmt_seq('coherence_g')} | 五段 coherence_raw: {fmt_seq('coherence_raw')}",
        f"五段 coherence_eff: {fmt_seq('coherence_eff')} | 五段 cos_grad_osc: {fmt_seq('cos_grad_osc_mean')}",
        f"五段 cos_osc: {fmt_seq('cos_osc')} | 五段 cos_rule: {fmt_seq('cos_rule')} | "
        f"五段 cos_rule_rawwin: {fmt_seq('cos_rule_rawwin')}",
        f"U/T: ‖U‖={ca['sum_betar_norm']:.2f} ‖T‖={ca['sum_rad_norm']:.2f} "
        f"‖Σe‖={ca['sum_dw_eff_norm']:.2f} cos(U,T)={ca.get('cos_betar_rad')} "
        f"‖T‖/‖Σe‖={ca.get('T_over_E')}",
        f"判据: L1 B<A 严格 {'PASS' if out['criteria']['L1_b_lt_a_strict']['pass'] else 'FAIL'} | "
        f"L2 s* bpb<5.0 {'PASS' if out['criteria']['L2_sstar_bpb_lt_5']['pass'] else 'FAIL'} | "
        f"s=1 bpb={fid['post_bpb_s1']} (仅记录, 尺度污染, 陷阱 6) | "
        f"word_hits={out['criteria']['C7_word_hits_ge_1']['value']} "
        f"({'PASS' if out['criteria']['C7_word_hits_ge_1']['pass'] else 'FAIL'}) | "
        f"C5 from-start={ca.get('cos_sumraw_sumg_eff')} "
        f"({'PASS' if out['criteria']['C5_cum_cos_gt_05']['pass'] else 'FAIL'})",
        f"协议一致性: pre_bpb={out['pre_bpb_protocol']['value']} "
        f"({'PASS' if out['pre_bpb_protocol']['pass'] else 'FAIL'}) | "
        f"md5 硬门 PASS | abort={abort}",
        f"自检(d) vs v460 max_abs_diff={v460_cross.get('max_abs_diff')} (新臂, 仅供对照)",
    ]), flush=True)
    print("exp133: DONE", flush=True)
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
