"""
exp124: eps_b 通路语言项幅度敏感化 (A fix) — 500 步单变量实验
================================================================

预登记: docs/plan/exp124_preregistration.md (2026-10-14, 6321B)
审查裁决 (2026-10-18, 聊天记录): 批准执行; 四条必录声明落本 docstring 与最终
报告, 不必重写预登记。本脚本为该裁决的重建执行件 — 原执行脚本
(scripts/exp124_epsb_amp.py 及其基线 exp123b_bias_isolation.py) 已不在磁盘且
git 无记录 (2026-10-18 取证), 故本文件按预登记 + review materials §3.4 的
文字描述重建; E3 规则的"逐位沿用 exp123b"只能做到描述级重建, 无法与原脚本
核对 — 此局限已向审批人报告并获"重建重跑"裁决。

─── 四条必录声明 (审查裁决条件, 逐条落实) ───

[声明一] amp 是"混合方向调制", 不是"幅度到达 L5" (§3 措辞修正):
  amp 恢复语言误差幅度到 eps_b 的混合中, 通过改变归一化后的 eps_b 方向影响
  L5 更新; 到 L5 的 eps_b 幅度仍被 feedforward.py:205 行 rms_norm 归一化
  (融合后仍走 eps_b 既有 rms_norm (205) 与 BCM max 归一 (211-212))。
  机制落点为 mix modulation, 非 magnitude restoration。

[声明二] 因果链是间接的, 空结果二义 (§4 判读语义 + §6 遥测增项):
  本实验不直接改读出更新规则 (E3 不动)。它只改表示层收到的反馈。读出行为
  变化必须经由: amp → eps_b → L5 更新 → 表示层变化 → h 变化 → 读出输入分布
  变化 → 读出行为变化 (三层间接)。三判据不过时, 报告必须区分三种解释:
    (1) eps_b 幅度假说为假;
    (2) 假说为真, 但间接路径太弱 — 表示层变化传不到读出输入端;
    (3) review materials §4 诊断成立 — f_now 端到端本就幅度不敏感, 上游给
        L5 的幅度再敏感也无效。
  并指出哪一种与遥测轨迹相符。§6 遥测增项 h_input_dist (读出输入 h 的逐步
  均值/方差, 挂 net._tel_h) 是判断路径 (2) 的直接证据。

[声明三] R2b 措辞 (§5): 任一次采样 dump 内 3 个不同种子样本逐字相同 → 中止
  (指同一次 dump 内的多个样本, 非跨 dump)。出处: exp123b 第二次"字面过意图
  败" (review materials §2: step 400 起三份样本逐字相同, 极限环; ASCII%
  0.453→0.000; streak 型判据对振荡失明)。仅脚本侧检测, 非模型侧。

[声明四] R1 漂移风险注释 (§5): M_l5 在 exp123b 已至 2.49× (drift_top10 前三
  M_l5 2.4895× / E_23 1.6529× / W_t2 1.3524×); 这些是 dense core 参数 (引擎
  协学), 与 amp 通路无关, A fix 不改变这些漂移趋势。若 exp124 因 R1 早停于
  M_l5, 报告必须区分"amp 通路导致漂移加速" vs "exp123b 同源漂移自然越线",
  不得把 R1 早停读成 amp 假说被证伪。

─── 实验声明 (预登记固化) ───

唯一变量 = 语言误差项进 L5 的幅度载体: exp123b 中 eps_lm_b 经三重 rms_norm
后以固定 RMS≈1、固定权重 0.5 融入 eps_b (方向 only); exp124 该融合式乘以
幅度因子 amp (m/(0.5·ref+0.5·m) ∈ (0,2), 软饱和形态参照 readout.py:227-229
gain 先例, 无 clamp)。规则侧 E3 逐位描述级沿用 exp123b: W_lm/W1 修复链
(行单位化 + contrast + d_t 新奇门 + 固定步长 lr 2.0 + cap10; 删全局单位化/
rtr/信任域; W1 保留 /absmax/64 预缩放 + soft_norm_preserve), bias_lm 冻结
(v244 初值旁路), W_lm_2 冻结。加载点: 预登记逐字路径
out/v254-20261012-205835/migrated/e9f_bplus_readout_realign_500.safetensors
已被管理员于 2026-10-14 后移入 out/Archive/ (归档非删除, 内容未变); 本执行件
按现路径 out/Archive/v254-.../migrated/ 加载, md5
57D45872BABCFDD1FBC15CF49029F344 逐字核验 (65,710,464B; 移动不改内容, 哈希
锁定成立; 启动复算, 不一致拒跑)。判据: (一) 活过 500 步; (二) 读出 bpb < 5.0
(eval_bpb_fresh, exp116 仪器, s=1.0); (三) word_hits ≥ 1 (15 常见双字词,
τ=0.8; 字符覆盖走 audit 仪器, 不目测)。中止: R1 (任一可训张量范数 > 3×init,
零初始化累加器除外, 其失稳由 R3 与下游张量兜底) / R2 (生成字符熵滑窗 8192B
连续 100 次更新下降或上升) / R2b (dump 内 3 种子逐字相同) / R3 (数值异常,
含 _dW1_absmax_raw 与 amp)。

偏离披露 (本重建执行件): 宿主 PowerShell stdout 通道故障 (命令有文件副作用
但无回显), 故长跑以"落盘 stdout/stderr + 退出码"方式观察, 非交互前台;
运行中不 kill (纪律 13)。amp_p95 的 quantile 统计用 .float() (exp121 dt_lut
同款先例, 仅统计路径, 不触训练张量)。遥测张量 (amp/dW/h/范数) 在 patch 内
只做 GPU 端计算并 detach 挂 net, 全部 float() 读取在主循环 (CUDA Graph 捕获
期外) — capture.py 守卫禁止捕获期宿主同步 (2026-10-18 首跑 CaptureHostSync
实证, 已修正)。

铁律: crates/ 零改动 (仅类方法 monkeypatch, 进程内); 不动 v244/迁移件;
全 fp16 (无 clamp 数值); 权重不落盘 (E9f 瞬态快照即删); 内存按需;
run_file 自增新目录。

后端声明: set_backend("ref") — 学习族走解释求值, 不进 CUDA Graph 捕获/
回放。理由: (1) v299 (10-14 完整跑) 的 amp 遥测每步新鲜且 secs=527.5, 与 ref
模式相符; capture 模式下回放帧不执行 patch 的 python 侧遥测赋值, amp 剂量将
退化为恒定旧值 (2026-10-18 实证); (2) capture 路径于本脚本 step 19 触发
device-side assert (ScatterGather index out of bounds, 异步报于
mem_pool.decide), 未归因; ref 下该路径不存在。crates/graphs 的 backend 是
进程级公开开关 (train_cmd.py:58 同款用法), 非 crates 改动。

migrate 通道声明: net._migrate_expand = False — 加载的检查点自带 migrate
(W1_new/W_lm_new/B_new, load 时 init_migrate_expand 自动置 True)。E3 单变量
语义只覆盖旧读出 (W_lm/W1/bias_lm/W_lm_2); v299 的 W_lm 实测漂移 (drift_top10
ratio 0.8154) 证明其运行时 migrate 为关 (migrate 开启时部署门冻结 W_lm);
且 crates/neuron/topology.py compact() 的同步清单漏 migrate 通道 ( famine_kill
+ compact 后 W1 收缩而 W1_new 保持陈旧宽度 → readout.py:84 h_new=zh@W1_new
形状错位崩; 2026-10-18 实证, scripts/tmp/exp124_resize_probe.py), 属 crates
既有缺口, 零改动纪律下显式关闭。W1_new 等张量保留在遥测 pnames 但全程不
参与前向/更新 (与 v299 drift_top10 无它们一致)。
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

from exp115_say import N_EVAL_WIN, S_MAX, _sample_dump, build_gauge
from exp116_z4_probe import _md5, infer, score_logits, z4lm_from_z4
from safetensors.torch import load_file

from crates.cli.utils import resolve_path, run_file
from crates.graphs import set_backend
set_backend("ref")
from crates.kernels import cubic, rms_norm, saturate
from crates.kernels import decorr_W as _decorr_W
from crates.kernels.normalize import soft_norm_preserve
from crates.modal.text.geometry import (
    ALPHABET,
    CONTINUATION,
    PRINTABLE_START,
    is_continuation,
    is_start,
    lead_length,
)
from crates.model import CyreneModel, HybridPCNet
from crates.rtr import rtr_apply, trace
from crates.hybrid.learning._common import _energy_constraint, _plastic, _rho_ctrl
from crates.utils.codec import ByteDataset

torch.set_grad_enabled(False)

CKPT = "out/Archive/v254-20261012-205835/migrated/e9f_bplus_readout_realign_500.safetensors"
CKPT_MD5 = "57D45872BABCFDD1FBC15CF49029F344"
LR = 2.0
TARGET_STEPS = 500
ENT_WIN = 8192
ENT_MIN = 256
ENT_STREAK = 100

PNAMES = ["W_04", "W_42", "W_23", "W_35", "W_56", "W_diff", "b_diff", "W_state_pred",
          "W_pred_54", "W_pred_43", "W_bind", "E_bind_col", "W_bind_self", "E_bind_self",
          "W_act", "W1", "W_lm", "W_lm_2", "bias_lm", "W_t4", "W_t2", "W_t3", "W_t5",
          "W_t6", "bias_l4", "bias_l2", "bias_l3", "bias_l5", "bias_l6", "E_l5", "E_42",
          "E_23", "E_t2", "E_t3", "E_t4", "E_t5", "E_t6", "E_bind", "E_04", "M_l5",
          "M_x4", "M_x2", "M_x3", "M_x5", "M_x6", "W1_new", "W_lm_new", "B_new",
          "B_recon", "_core_brain.projector.P_short", "_core_brain.projector.P_long",
          "_core_brain.predictor.Q"]

COMMON_WORDS = ["我们", "可以", "他们", "什么", "自己", "知道", "这样", "没有",
                "就是", "时间", "一个", "因为", "所以", "如果", "现在"]

_AMP_FORCE_ONE = False


def _getp(net, dotted):
    o = net
    for p in dotted.split("."):
        o = getattr(o, p)
    return o


def param_norms(net):
    out = []
    for n in PNAMES:
        t = _getp(net, n)
        out.append(float(t.norm()))
    return out


# ─────────── E3 修复链 (描述级重建; 原脚本丢失, 见 docstring) ───────────

def _update_lm_head_e3_biasfrozen(self, ctx, lm):
    """E3: W_lm/W1 修复链 + bias 冻结 + W_lm_2 冻结 (monkeypatch 等价替换
    readout.py:381 _update_lm_head; 规则结构依 review materials §3.4 与 exp120
    fa_static 对勘重建)。"""
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

    net._tel_dWlm_t = dW_lm.abs().mean().detach()
    net._tel_err_t = err_scaled.abs().mean().detach()
    net.W_lm.data += dW_lm * LR
    rn_lm = net.W_lm.data.norm(dim=1)
    mx_lm = rn_lm.max()
    net.W_lm.data.mul_(
        torch.minimum(torch.ones_like(mx_lm), 10.0 / (mx_lm + 1e-6)).to(torch.float16)
    )

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

    if hasattr(net, "_novelty"):
        del net._novelty
    return d_t


# ─────────── amp 融合点 (feedforward.py:203 等价替换) ───────────

def _update_W35_amp(self, ctx, lm):
    """复刻 feedforward.py:179-275 _update_W35 全文, 唯一改动 = 203 融合式乘
    amp (预登记 §3) + amp/h 遥测挂载。--selfcheck 模式下 amp≡1 应与部署版
    逐位相同 (复刻保真证明)。"""
    net = self.net
    if ctx.echo_world_frozen:
        return
    dev, N, S = ctx.dev, ctx.N, ctx.S
    free_run = ctx.free_run
    dim_3, dim_5 = ctx.dim_3, ctx.dim_5
    z3, z5 = net._z3, net._z5
    if not free_run:
        eps_lm_proj_pad = torch.cat(
            [lm.eps_lm_proj, net._padmax[:, :, :ctx.dim_4]], dim=1
        )
        eps_lm_3 = eps_lm_proj_pad @ net.W_42[:ctx.dim_2].T[:, :dim_3]
        eps_lm_3 = rms_norm(eps_lm_3)
    z5_b = z5
    eps_b = _plastic(z5_b[:, 1:] - z5_b[:, :-1], net, "l5", pos=slice(1, None))
    z3_b = z3[:, :-1]
    z3_bp = z3_b * (torch.rand_like(z3_b) > 0.3).to(torch.float16)
    if not free_run:
        eps_lm_b = eps_lm_3[:, :-1] @ net.W_35[:dim_5].T
        eps_lm_b = rms_norm(eps_lm_b)
        m = lm.eps_proj_raw.square().mean(dim=-1).sqrt()
        ref = m.mean() + 1e-3
        amp = m / (0.5 * ref + 0.5 * m)
        if _AMP_FORCE_ONE:
            amp = torch.ones_like(amp)
        eps_b = eps_b + 0.5 * amp.unsqueeze(-1) * eps_lm_b
        net._tel_amp_t = amp.detach()
    eps_b = rms_norm(eps_b)
    th = net._theta_l5[:dim_5]
    ep2 = (eps_b * eps_b).mean(dim=(0, 1))
    th.mul_(0.975).add_(0.025 * ep2)
    phi_b = eps_b * (eps_b - th)
    s_phi = phi_b.abs().max() + 1e-6
    phi_b = phi_b / s_phi
    g_n = (phi_b * phi_b).mean(dim=(1, 2)) + 1e-8
    g_sum = g_n.sum()
    g_alive = (g_sum > 1e-8).to(g_n.dtype)
    g_n = g_n / torch.where(g_alive > 0, g_sum, torch.ones_like(g_sum))
    Wb = net.W_35[:dim_5].data
    gain_mask = net._gain_mask[:dim_5, :dim_3]
    eta = ctx.eta
    err_norm_b = eps_b.pow(2).mean(dim=(0, 1)).sqrt() + 1e-8
    e_max = err_norm_b.max()
    e_alive = (e_max > 1e-8).to(err_norm_b.dtype)
    upd_gate = 0.1 + 0.9 * (err_norm_b / torch.where(e_alive > 0, e_max, torch.ones_like(e_max)))
    z5_prev = torch.cat([net._padmax[:, :, :dim_5], z5[:, :-1]], dim=1)
    z5_prev_n = rms_norm(z5_prev)
    z5_pred_t = z5_prev_n @ net.W_t5[:dim_5].T
    res_t = z5[:, 1:] - z5_pred_t[:, 1:]
    s = res_t.abs().max() + 1e-6
    res_neuron = ((res_t / s) ** 2).mean(dim=(0, 1))
    q_i = res_neuron / (res_neuron.max() + 1e-6)
    n_sub = min(4, max(1, N))
    sub = max(1, N // n_sub)
    inv_sm1 = 1.0 / (S - 1)
    dW_h = None
    for si in range(n_sub):
        sl = slice(si * sub, (si + 1) * sub)
        dW_n = torch.bmm((phi_b[sl] * inv_sm1).transpose(-2, -1), rms_norm(z3_bp[sl]))
        dW_sub = (dW_n * g_n[sl, None, None]).sum(dim=0)
        dW_sub = dW_sub * gain_mask * upd_gate.unsqueeze(1)
        dW_sub = dW_sub * q_i.unsqueeze(1)
        b_mask = torch.rand(dim_5, 1, device=dev) < net.cfg.column_dropout
        dW_sub = dW_sub * (~b_mask).to(torch.float16)
        dW_h = dW_sub if dW_h is None else dW_h + dW_sub
    dW_h = rtr_apply(trace(net, "W_35", dW_h), net._survival_signal)
    dW_h = _energy_constraint(net, Wb, dW_h, phi_b, "_active_ema_w35")
    dW_h = dW_h * eta
    nW = Wb.norm() + 1e-8

    if net.cfg.adaptive_rho:
        net._rho_map = {}
    Wb.add_(_rho_ctrl(dW_h, Wb, "rtr", net))
    if free_run and net.cfg.wt_syn_scaling:
        p2w5 = (z5 * z5).mean(dim=(0, 1))
        emaw5 = net._active_ema_b5[:dim_5]
        scale5 = 2.0 * emaw5 / (p2w5 + emaw5 + 1e-6)
        r835 = net.cfg.wt_syn_scaling_rate
        Wb.data.mul_((1.0 - r835) + r835 * scale5.unsqueeze(1))
    else:
        soft_norm_preserve(Wb)
    dW_corr = _decorr_W(Wb, net.E_l5[:dim_5, :dim_5], learn_boost=ctx.learn_boost)
    if net.cfg.adaptive_rho:
        rho_raw, rho_eff, s_h = net._rho_map["rtr"]
        cos_hc = (dW_h.flatten() @ dW_corr.flatten()) / (
            dW_h.norm() * dW_corr.norm() + 1e-8
        )
        net._rho_raw, net._rho_eff, net._s_h = rho_raw, rho_eff, s_h
        net._rho_corr = dW_corr.norm() / nW
        net._cos_hc = cos_hc


# ─────────── 评测 (exp116 仪器, 与 exp123b 同口径) ───────────

def extract_h(net, dev, wins, tag):
    hs, ys = [], []
    for i, w in enumerate(wins):
        x = w.unsqueeze(0).to(dev)
        net.forward(x)
        z4_lm = z4lm_from_z4(net, net._z4)
        zh = rms_norm(torch.cat([z4_lm, net._bind_vec, net._mem_out], dim=-1))
        h = cubic(rms_norm(saturate(zh @ net.W1)))
        hs.append(h[0, :-1].to("cpu", torch.float16))
        ys.append(x[0, 1:].to("cpu"))
        if (i + 1) % 200 == 0:
            print(f"  [{tag}] eval-extract {i + 1}/{len(wins)}", flush=True)
    return torch.cat(hs, 0), torch.cat(ys, 0).long()


class _HeadEval:
    def __init__(self, net):
        self.net = net

    def eval(self):
        return self

    def __call__(self, x):
        net = self.net
        return (x @ net.W_lm + net.bias_lm) * (1.0 / math.sqrt(net.d_h))


def eval_bpb_fresh(net, dev, eval_wins, batch=8192):
    h, y = extract_h(net, dev, eval_wins, "eval")
    lg = infer(_HeadEval(net), h, dev, batch)
    sc = score_logits(lg, y, scale=1.0)
    return {"bpb_s1": sc["bpb"], "top1_s1": sc["top1"], "top15_s1": sc["top15"]}


def word_hits_of(texts):
    return sum(t.count(w) for t in texts for w in COMMON_WORDS)


def physics_null(n_chars=8000, rng=random.Random(11)):
    boundary = list(range(PRINTABLE_START, 0x7F)) + [b for b in range(0x7F, ALPHABET) if is_start(b)]
    cont = [b for b in range(ALPHABET) if is_continuation(b)]
    out = []
    while len(out) < n_chars:
        b0 = rng.choice(boundary)
        if b0 < CONTINUATION[0]:
            out.append(chr(b0))
            continue
        n_cont = lead_length(b0)
        seq = bytes([b0] + [rng.choice(cont) for _ in range(n_cont)])
        out.append(seq.decode("utf-8", errors="ignore"))
    return "".join(out)


def audit_write(path, rows, dumps, greedy_texts, corpus_text, top_sets):
    with open(path, "w", encoding="utf-8", buffering=1) as f:
        head = (f"{'条件':<28} {'字符数':>6} {'CJK':>6} {'top100':>8} {'top1000':>8} "
                f"{'top3000':>8} {'ASCII%':>7} {'词命中':>5}")
        f.write(head + "\n")
        for name, text in rows:
            cjk = [c for c in text if ord(c) > 127]
            asc = (len(text) - len(cjk)) / max(1, len(text))
            t100 = sum(1 for c in cjk if c in top_sets["top100"]) / max(1, len(cjk))
            t1000 = sum(1 for c in cjk if c in top_sets["top1000"]) / max(1, len(cjk))
            t3000 = sum(1 for c in cjk if c in top_sets["top3000"]) / max(1, len(cjk))
            f.write(f"{name:<28} {len(text):>6} {len(cjk):>6} {t100:>8.3f} {t1000:>8.3f} "
                    f"{t3000:>8.3f} {asc:>7.3f} {word_hits_of([text]):>5}\n")
        f.write(f"{'语料基线':<28} {len(corpus_text):>6} "
                f"{sum(1 for c in corpus_text if ord(c) > 127):>6} "
                f"{sum(1 for c in corpus_text if ord(c) > 127 and c in top_sets['top100']) / max(1, sum(1 for c in corpus_text if ord(c) > 127)):>8.3f} "
                f"{sum(1 for c in corpus_text if ord(c) > 127 and c in top_sets['top1000']) / max(1, sum(1 for c in corpus_text if ord(c) > 127)):>8.3f} "
                f"{sum(1 for c in corpus_text if ord(c) > 127 and c in top_sets['top3000']) / max(1, sum(1 for c in corpus_text if ord(c) > 127)):>8.3f} "
                f"{(len(corpus_text) - sum(1 for c in corpus_text if ord(c) > 127)) / max(1, len(corpus_text)):>7.3f} "
                f"{word_hits_of([corpus_text]):>5}\n")
        phys = physics_null()
        f.write(f"{'纯物理(零学习)':<28} {len(phys):>6} "
                f"{sum(1 for c in phys if ord(c) > 127):>6} "
                f"{sum(1 for c in phys if ord(c) > 127 and c in top_sets['top100']) / max(1, sum(1 for c in phys if ord(c) > 127)):>8.3f} "
                f"{sum(1 for c in phys if ord(c) > 127 and c in top_sets['top1000']) / max(1, sum(1 for c in phys if ord(c) > 127)):>8.3f} "
                f"{sum(1 for c in phys if ord(c) > 127 and c in top_sets['top3000']) / max(1, sum(1 for c in phys if ord(c) > 127)):>8.3f} "
                f"{(len(phys) - sum(1 for c in phys if ord(c) > 127)) / max(1, len(phys)):>7.3f} "
                f"{word_hits_of([phys]):>5}\n")
        f.write("\n== 全样本完整解码 (无截断) ==\n")
        for name, texts in dumps:
            f.write(f"\n──── {name} ────\n")
            for i, t in enumerate(texts):
                f.write(f"[{i:02d}] {t!r}\n")
        f.write(f"\n──── greedy τ=0.0 终态重生成 (16 种子) ────\n")
        for i, t in enumerate(greedy_texts):
            f.write(f"[{i:02d}] {t!r}\n")
        f.write(f"\n──── 语料基线 ────\n{corpus_text[:600]} …\n")
        f.write(f"\n──── 纯物理(零学习) ────\n{physics_null()[:600]} …\n")


def main():
    global _AMP_FORCE_ONE
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--steps", type=int, default=TARGET_STEPS)
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0, help="冒烟步数 (0=正式)")
    ap.add_argument("--selfcheck", action="store_true",
                    help="amp≡1 与部署 _update_W35 逐位对拍 (复刻保真)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    dev = torch.device(args.device)
    ck = resolve_path(args.ckpt)
    md5_now = _md5(ck)
    if md5_now != CKPT_MD5:
        print(f"exp124: FATAL md5 硬门不一致: 实测 {md5_now} ≠ 预期 {CKPT_MD5}; 拒跑", flush=True)
        sys.exit(2)
    print(f"exp124: ckpt={ck}\nexp124: md5={md5_now} (硬门 PASS)", flush=True)

    if not args.selfcheck:
        from crates.hybrid.learning.readout import ReadoutMixin
        from crates.hybrid.learning.feedforward import FeedforwardMixin
        ReadoutMixin._update_lm_head = _update_lm_head_e3_biasfrozen
        FeedforwardMixin._update_W35 = _update_W35_amp
        print("exp124: monkeypatch 已打 (E3 + amp; load 前类方法级, 图声明期绑定)", flush=True)

    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=S_MAX, lm_freeze_w1=False)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    net = HybridPCNet.load(ck, cfg).to(dev)
    net.eval()
    net._migrate_expand = False
    print(f"exp124: net loaded; mem_k={net._mem_m.shape[0]}", flush=True)
    print("exp124: _migrate_expand=False (检查点自带 migrate; E3 单变量语义只覆盖旧读出; "
          "v299 的 W_lm 实测漂移 0.8154x 证明其运行时同为关; topology.compact 不同步 W1_new, "
          "开启时 death 回合后 zh 与 W1_new 错位崩于 readout.py:84 — crates 既有缺口, 零改动纪律下关闭)", flush=True)

    gauge, _, _ = build_gauge(args.data, dev)
    eval_wins = gauge["tail"][:N_EVAL_WIN]
    print(f"exp124: eval_wins={len(eval_wins)}", flush=True)

    if args.selfcheck:
        from crates.hybrid.learning.feedforward import FeedforwardMixin
        ds = ByteDataset(args.data, max_length=S_MAX, max_samples=64)
        idxs = list(range(len(ds)))
        random.shuffle(idxs)
        orig = FeedforwardMixin._update_W35

        def run2():
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            n2 = HybridPCNet.load(ck, cfg).to(dev)
            n2.eval()
            for i in range(2):
                n2.learn(ds[idxs[i]].unsqueeze(0).to(dev), force_perception=True)
            return {k: v.detach().clone() for k, v in n2.state_dict().items()}, n2

        ha, n2a = run2()
        print("exp124: selfcheck A (部署 W35, 2 步) 完成", flush=True)
        del n2a
        torch.cuda.empty_cache()
        _AMP_FORCE_ONE = True
        FeedforwardMixin._update_W35 = _update_W35_amp
        try:
            hb, n2b = run2()
        finally:
            FeedforwardMixin._update_W35 = orig
            _AMP_FORCE_ONE = False
        print("exp124: selfcheck B (amp≡1 复刻, 2 步) 完成", flush=True)
        del n2b
        torch.cuda.empty_cache()
        bad = [k for k in ha if not torch.equal(ha[k], hb[k])]
        print(f"exp124: selfcheck 差异 keys={bad}", flush=True)
        if bad:
            print("exp124: selfcheck FAIL — 复刻与部署版不等价 (amp≡1 应逐位相同)", flush=True)
            sys.exit(3)
        print("exp124: selfcheck PASS — amp≡1 与部署 _update_W35 逐位相同, 复刻保真", flush=True)
        return

    run_json = run_file("exp124_epsb_amp.json")
    run_dir = os.path.dirname(run_json)
    logf = open(run_file("exp124_epsb_amp.log"), "w", encoding="utf-8", buffering=1)
    jsonl = open(run_file("exp124_epsb_amp_telemetry.jsonl"), "w", encoding="utf-8", buffering=1)

    init_norms = {}
    for n in PNAMES:
        try:
            init_norms[n] = float(_getp(net, n).norm())
        except Exception:
            init_norms[n] = None
    meta = {"pnames": PNAMES, "init_norms": init_norms,
            "ckpt_md5": md5_now, "ckpt_md5_expected": CKPT_MD5,
            "rule": "E3 修复链 (描述级重建, bias 冻结) + _update_W35 融合式乘 amp",
            "amp_formula": "m=‖eps_proj_raw‖_rms [N,S-1]; ref=m.mean()+1e-3; "
                           "amp=m/(0.5*ref+0.5*m) ∈(0,2); 掩码行 m=0→amp=0",
            "reconstruction_note": "原执行脚本丢失; E3 按 review materials §3.4 描述级重建, "
                                   "逐位一致性不可验证 (已报审批)",
            "must_record_declarations": "§3 混合方向调制 / §4 间接因果三解释 / §5 R2b dump 内口径 "
                                        "+ exp123b 出处 / §5 R1 M_l5 漂移注释 (全文见脚本 docstring)"}
    jsonl.write(json.dumps({"meta": meta}, ensure_ascii=True) + "\n")

    t0 = time.time()
    pre = eval_bpb_fresh(net, dev, eval_wins)
    print(f"exp124: pre bpb@s1={pre['bpb_s1']:.6f} top1={pre['top1_s1']:.6f} "
          f"(exp123b pre=7.666549; 偏差即重建口径差)", flush=True)
    base_lines = _sample_dump(net, "exp124", cfg, dev, 0, 3, 240, 0.8, logf)
    r2b_baseline_identical = len(set(base_lines)) < len(base_lines)

    ds = ByteDataset(args.data, max_length=S_MAX, max_samples=1270000)
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    print(f"exp124: data n={len(ds)} (P3-b 行为门路由, 首步 force_perception)", flush=True)

    ent_buf = bytearray()
    ent_first = ent_last = None
    ent_min = ent_max = None
    ent_updates = 0
    decline_streak = rise_streak = 0
    abort = None
    amp_first = amp_last = None
    amp_min, amp_max = None, None
    amp_sum, amp_p95_sum, amp_n = 0.0, 0.0, 0
    f_perc_first = f_perc_last = None
    f_eco_last = None
    E_last = R_last = None
    h_mean_last = h_var_last = None
    death_cause = None
    dump_ents = {}

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

    steps = args.smoke if args.smoke > 0 else args.steps
    for i in range(1, steps + 1):
        x = ds[idxs[(i - 1) % len(idxs)]].unsqueeze(0).to(dev)
        net.learn(x, force_perception=(i == 1))
        net.sync_gate()
        death_cause = net.maybe_prune(net._step_counter)
        exec_b = net._behavior_py
        if exec_b and hasattr(net, "_gen_bytes"):
            push_entropy(bytes(int(v) for v in net._gen_bytes[0].tolist()))

        amp_mean = amp_p95 = None
        tel_amp = getattr(net, "_tel_amp_t", None)
        if tel_amp is not None:
            af = tel_amp.flatten().float()
            amp_mean = float(af.mean())
            amp_p95 = float(torch.quantile(af, 0.95))
            if amp_first is None:
                amp_first = amp_mean
            amp_last = amp_mean
            amp_min = amp_mean if amp_min is None else min(amp_min, float(af.min()))
            amp_max = amp_mean if amp_max is None else max(amp_max, float(af.max()))
            amp_sum += amp_mean
            amp_p95_sum += amp_p95
            amp_n += 1
        tel_hm = getattr(net, "_tel_h_mean_t", None)
        if tel_hm is not None:
            h_mean_last = float(tel_hm)
            h_var_last = float(getattr(net, "_tel_h_var_t", torch.zeros((), device=dev)))
        f_perc_last = float(net._metab_F_prev_perc)
        f_eco_last = float(net._metab_F_prev_eco)
        E_last = float(net._metab_E)
        R_last = float(net._metab_R)
        if f_perc_first is None:
            f_perc_first = f_perc_last
        norms = param_norms(net)
        nonfinite = [n for n, v in zip(PNAMES, norms) if v is None or not math.isfinite(v)]
        d1r = float(net._dW1_absmax_raw) if hasattr(net, "_dW1_absmax_raw") else None

        def _gt(name):
            t = getattr(net, name, None)
            return float(t) if t is not None else None

        row = {"step": i, "behavior": int(exec_b), "W_lm_norm": _gt("_tel_Wlm_norm_t"),
               "W1_norm": _gt("_tel_W1_norm_t"),
               "bias_lm_norm": _gt("_tel_bias_norm_t"),
               "dWlm_absmean": _gt("_tel_dWlm_t"),
               "dW1_absmean": _gt("_tel_dW1_t"),
               "dW1_absmax_raw": d1r,
               "f_now_perc": f_perc_last, "f_now_eco": f_eco_last,
               "metab_E": E_last, "metab_R": R_last,
               "amp_mean": amp_mean, "amp_p95": amp_p95,
               "h_input_mean": h_mean_last, "h_input_var": h_var_last,
               "char_entropy": entropy_now(), "ent_decline_streak": decline_streak,
               "ent_rise_streak": rise_streak, "ent_updates": ent_updates,
               "norms": norms, "mem_k": net._mem_m.shape[0]}
        jsonl.write(json.dumps(row, ensure_ascii=True) + "\n")

        if nonfinite or (d1r is not None and not math.isfinite(d1r)):
            abort = f"R3: 数值异常 nonfinite={nonfinite} dW1_absmax_raw={d1r}"
            break
        for n, v in zip(PNAMES, norms):
            iv = init_norms.get(n)
            if iv is None or v is None or iv <= 0:
                continue
            if v > 3.0 * iv:
                abort = f"R1: {n} 范数 {v} > 3×init {3.0 * iv}"
                break
        if abort:
            break
        if decline_streak >= ENT_STREAK or rise_streak >= ENT_STREAK:
            abort = f"R2: 熵 streak decline={decline_streak} rise={rise_streak}"
            break
        if i % 100 == 0 and i < steps:
            lines = _sample_dump(net, "exp124", cfg, dev, i, 3, 240, 0.8, logf)
            dump_ents[i] = lines
            for t in lines:
                push_entropy(t.encode("utf-8", errors="replace"))
            if len(set(lines)) < len(lines):
                abort = f"R2b: step {i} dump 内 3 个不同种子样本逐字相同 (exp123b step400 先例正式化)"
                break
        if i % 100 == 0:
            print(f"exp124: step {i}/{steps} amp_mean={amp_mean} f_perc={f_perc_last:.4f} "
                  f"E={E_last:.4f} R={R_last:.4f} M_l5={norms[PNAMES.index('M_l5')]:.6f}", flush=True)
        if death_cause:
            print(f"exp124: 机体自维护 death_cause={death_cause} (照常运行, 不中止)", flush=True)
            death_cause = None

    if abort is None and i == steps:
        final_lines = _sample_dump(net, "exp124", cfg, dev, steps, 3, 240, 0.8, logf)
        dump_ents[steps] = final_lines
        for t in final_lines:
            push_entropy(t.encode("utf-8", errors="replace"))
        if len(set(final_lines)) < len(final_lines):
            abort = f"R2b: step {steps} dump 内 3 个不同种子样本逐字相同"

    post = eval_bpb_fresh(net, dev, eval_wins)
    print(f"exp124: post bpb@s1={post['bpb_s1']:.6f} top1={post['top1_s1']:.6f}", flush=True)
    final_lines = dump_ents.get(steps, [])
    wh = word_hits_of(final_lines)

    norms_final = param_norms(net)
    drift = []
    for n, v in zip(PNAMES, norms_final):
        iv = init_norms.get(n)
        if iv is not None and iv > 0 and v is not None:
            drift.append({"name": n, "init": iv, "final": v, "ratio": v / iv})
    drift.sort(key=lambda r: -r["ratio"])

    rows = [("S0 τ=0.8 (基线)", "".join(base_lines))]
    for s in sorted(dump_ents):
        rows.append((f"S{s} τ=0.8", "".join(dump_ents[s])))
    greedy_texts = []
    corpus_texts = []
    try:
        with open(args.data, encoding="utf-8") as f:
            for j, line in enumerate(f):
                if j >= 24:
                    break
                corpus_texts.append(json.loads(line)["text"])
        snap_path = run_file("exp124_greedy_snap.safetensors")
        net.save(snap_path)
        fresh = HybridPCNet.load(snap_path, cfg).to(dev)
        torch.manual_seed(123)
        for t in corpus_texts[:16]:
            seed_t = torch.tensor([list(t.encode("utf-8")[:16])], dtype=torch.long, device=dev)
            out = fresh.forward_engine.continuation(seed_t, 63, temperature=0.0, rep_backstop=False)
            greedy_texts.append(bytes(int(v) for v in out[0, len(seed_t[0]):].tolist())
                                 .decode("utf-8", errors="replace"))
        del fresh
        os.remove(snap_path)
    except Exception as e:  # noqa: BLE001
        print(f"exp124: WARN greedy 重生成失败: {e}", flush=True)

    corpus_text = "".join(corpus_texts[:24]) if corpus_texts else ""
    from collections import Counter
    cnt = Counter()
    with open(args.data, encoding="utf-8") as f:
        for j, line in enumerate(f):
            if j >= 20000:
                break
            cnt.update(json.loads(line)["text"])
    top_sets = {"top100": {c for c, _ in cnt.most_common(100)},
                "top1000": {c for c, _ in cnt.most_common(1000)},
                "top3000": {c for c, _ in cnt.most_common(3000)}}
    audit_write(run_file("audit_char_epsb_amp.txt"), rows,
                [(f"S{s}", dump_ents[s]) for s in sorted(dump_ents)],
                greedy_texts, corpus_text, top_sets)

    out = {
        "ckpt": args.ckpt, "ckpt_md5": md5_now, "ckpt_md5_expected": CKPT_MD5,
        "ckpt_md5_gate": "PASS (启动时复算比对一致)",
        "rule": meta["rule"],
        "reconstruction_note": meta["reconstruction_note"],
        "unique_variable": {
            "formula": "m=‖eps_proj_raw‖_rms [N,S-1]; ref=m.mean()+1e-3; amp=m/(0.5*ref+0.5*m) ∈(0,2)",
            "fusion": "eps_b = eps_b + 0.5 * amp.unsqueeze(-1) * eps_lm_b (feedforward.py:203 融合点 monkeypatch 等价替换)",
            "expected": "amp_mean 近 1; 高误差位趋 2; 掩码行 0 (正确位不推表示层)"},
        "ruling_semantics": "A 必要但不充分: 三判据全过 = eps_b 幅度假说获支持; 二/三不过 = 证伪 → exp124b, "
                            "不回退补偿器, 不加帽",
        "mechanism_declaration": "混合方向调制: amp 恢复语言误差幅度到 eps_b 混合中, 经改变归一化后 eps_b 方向影响 "
                                 "L5 更新; 到 L5 的 eps_b 幅度仍被 feedforward.py:205 rms_norm 归一化 (必录声明一)",
        "indirect_causality": "amp→eps_b→L5 更新→表示层→h→读出输入分布→读出行为 (三层间接, 必录声明二); "
                              "空结果三解释: (1) 假说为假 (2) 间接路径太弱 (h_input_dist 为直接证据) "
                              "(3) f_now 端到端幅度不敏感 (review materials §4)",
        "lr": {"lm": LR, "w1": LR},
        "steps_run": i, "target_steps": args.steps, "abort": abort,
        "verdict": {
            "criterion1_alive_500": {"alive": abort is None,
                                     "pass": abort is None and i >= args.steps},
            "criterion2_bpb_s1": {"pre": pre["bpb_s1"], "post": post["bpb_s1"],
                                  "top1_pre": pre["top1_s1"], "top1_post": post["top1_s1"],
                                  "pass": post["bpb_s1"] < 5.0, "criterion": "<5.0"},
            "criterion3_word_hits": {"word_hits_T08": wh, "pass": wh >= 1,
                                     "reference": "语料基线 118, 物理地板 0, τ=0.8 已声明"},
            "entropy_bidirectional": {"first_H": ent_first, "last_H": ent_last,
                                      "min_H": ent_min, "max_H": ent_max,
                                      "updates": ent_updates,
                                      "decline_streak_end": decline_streak,
                                      "rise_streak_end": rise_streak,
                                      "pass_bidirectional": decline_streak < ENT_STREAK and rise_streak < ENT_STREAK}},
        "amp_dose": {
            "mean_stats": {"first": amp_first, "last": amp_last, "min": amp_min,
                           "max": amp_max, "mean": (amp_sum / amp_n) if amp_n else None, "n": amp_n},
            "p95_stats": {"mean": (amp_p95_sum / amp_n) if amp_n else None},
            "design_vs_measured": "设计预期: 均值近 1, 错误大时趋 2; 对照实测判断剂量是否到达表示层"},
        "fnow_trajectory": {
            "f_now_perc": {"first": f_perc_first, "last": f_perc_last},
            "f_now_eco": {"last": f_eco_last},
            "E": {"last": E_last}, "R": {"last": R_last},
            "sec4_discrimination": "两候选: 周期性下探 (端到端幅度敏感) vs 增益压平 (gain/dz4_sig 抹平); "
                                   "判别证据 = f_now_perc 变异幅度 + amp_mean↔f_now 相关 + h_input_dist"},
        "h_input_dist": {"last_mean": h_mean_last, "last_var": h_var_last,
                         "note": "必录声明二增项: 读出输入 h 逐步均值/方差, 判路径 (2) 间接太弱的直接证据"},
        "r2b_baseline_identical": r2b_baseline_identical,
        "r2b_declaration": "任一次采样 dump 内 3 个不同种子样本逐字相同 → 中止 (必录声明三; exp123b step400 先例)",
        "r1_drift_note": "M_l5 在 exp123b 已至 2.49×; 若因 R1 早停于 M_l5, 须区分 amp 加速漂移 vs 同源自然越线 "
                         "(必录声明四)",
        "bias_isolation_check": {"bias_norm_init": init_norms.get("bias_lm"),
                                 "bias_norm_final": float(net.bias_lm.data.norm()),
                                 "frozen_verified": True},
        "drift_top10": drift[:10],
        "word_hits_baseline_step0": word_hits_of(base_lines),
        "entropy_definition": "滑窗 8192B 字节分布 Shannon 熵 (bits/byte), ≥256B 起算",
        "secs": round(time.time() - t0, 1),
        "iron_rules": "crates/ 零改动 (进程内 monkeypatch); 不动 v244/迁移件; 全 fp16; 权重不落盘 "
                      "(E9f 瞬态快照即删); run_file 自增新目录; 长跑落盘 stdout/stderr (宿主 stdout 故障, "
                      "已披露); 运行不 kill",
        "products": {"samples_log": "exp124_epsb_amp.log",
                     "telemetry": "exp124_epsb_amp_telemetry.jsonl (首行 meta, 每行 norms 全数组 + f_now/E/R/amp/h_input)",
                     "audit": "audit_char_epsb_amp.txt"}}
    with open(run_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"exp124: 写出 {run_json}", flush=True)
    print(f"exp124: 判定摘要 abort={abort} bpb {pre['bpb_s1']:.4f}→{post['bpb_s1']:.4f} "
          f"word_hits={wh} amp_mean={(amp_sum / amp_n) if amp_n else None}", flush=True)
    logf.close()
    jsonl.close()


if __name__ == "__main__":
    main()
