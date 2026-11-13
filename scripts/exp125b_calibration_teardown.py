"""
exp125b: 读出校准错误拆解 (C 型失败模式归因) — 500 步 x 3 臂 + 后置尺度扫描
=====================================================================

立项依据 (2026-10-19 审查裁决, 聊天记录): exp125 执行 PASS, B 线确认, 但
分流读浅一层 — C 组 (lr 0.05) 的失败模式是**校准错误型** (top1 0.1316 约为
均匀 0.0039 的 34 倍, bpb 10.80 双峰: 约 13% 位自信对 / 87% 位自信错,
f_now 峰值 13.27 从未过 100, 代谢未崩), 与 A/B **崩溃型** (f_now 30.8/44.9,
top1 约 0) 不同。不批更细 lr 网格 (lr 轴在"能不能学好"意义上已饱和), 不批
延长步数 (要修结构不是耐心)。批准 exp125b, 设计必须回答四问:

  Q1 为什么 lr 0.05 时 W_lm 仍在长 (1.18x) 而误差没降? (d_t 门方向驱动 /
     contrast_w 列重加权偏置 / cap10 量化伪影)
  Q2 为什么校准错误而 top1 好? (inv_h 尺度问题 → logits 放大 → 过度自信 /
     rms_norm(h) 与 bias_lm 交互)
  Q3 d_t 门正反馈? (nov 高 → d_t 高 → W_lm 大步 → h 变 → nov 更高)
  Q4 moving target? (eps_total = target − softmax(logits), logits 每步变)

执行口径: agent 直接跑 exp125b。首句给: 四问中哪个/哪些是主因 + post bpb +
word_hits + 是否仍出现 C 型校准错误。

预登记: docs/plan/exp125b_preregistration.md (判读格预写, 跑前锁哈希;
本文件跑前写入, sha256 入账)。

─── 设计 (与预登记逐字对应) ───

三臂 (全部 lr 0.05 = exp125 C 条件, 同检查点/数据/种子/协议, 500 步):
  ctrl        E3 原样 (复跑 C) + 全遥测 + 后置扫描 — 复现保真对照
  nocontrast  contrast_w ≡ 1 (关列重加权; 其余逐位同 E3) — Q1/Q2 消融
  nodt        d_t ≡ 1 (关新奇门; 其余逐位同 E3) — Q1/Q3 消融

E3 副本说明: 本文件内 `_update_lm_head_e3b_tel` 是
exp124_epsb_amp._update_lm_head_e3_biasfrozen 的**逐字复刻 + 只读遥测 +
双臂开关**。复刻保真证明 (硬门): ctrl 臂与 exp125 C 正式跑 (v432) 同协议,
step-1 dWlm_absmean / W_lm_norm 须逐位相同 (1.5673828125 / 75.3125), 且
500 步 post bpb 须逐位相同 (10.797670696653336); 冒烟口径对照 exp125 冒烟
(v429) step-1 (1.6025390625 / 74.625)。不一致即复刻失败, 该臂数据作废,
记偏差。遥测全部只读 (clone/detach/.float() 统计路径, 不触训练张量, 不耗
RNG); 模块级 _TEL dict 不挂 net (net.save 会把 net.__dict__ 的 dict 属性
序列化进快照, exp124c 冒烟 step 100 实证崩; _tel_* 裸张量沿用原样)。

后置扫描 (每臂跑毕, 只读, 零训练): 尺度/温度扫描 scale ∈
{1.6,1.3,1.0,0.8,0.6,0.5,0.4,0.3,0.2} (score_logits 的 scale 参数,
logits×scale; scale<1 = 升温软化) + bias_lm 消融 (去掉 bias 的读出头,
在 s=1 与臂最优 scale 两点)。回答 Q2: 若 bpb(s*) ≤ 0.7×bpb(s=1) → 尺度
miscalibration 坐实 (logits 尺度过大); 否则方向性错误为主。辅助判据:
s* 与 (init logits std / final logits std) 相差 ≤25% → 尺度增长与 W_lm
增长同源。

判据 (预写, 见预登记): learned := post bpb < 5.0; c_type (校准错误型) :=
post bpb ≥ 8.0 ∧ top1 ≥ 0.05 ∧ f_now max < 100; crash_type := post ≥ 8.0
∧ ¬c_type; R1/R2/R2b/R3 中止沿用。Q1/Q2/Q3/Q4 判定规则全部预写于预登记,
由回执跨臂执行 (单臂 json 只出原料)。

偏离披露:
  - 宿主 PowerShell stdout 通道故障 (exp124/exp124c/exp125 同款): 启动即
    重定向 stdout/stderr 到 run 目录 exp125b_{arm}_stdout.log; 运行不 kill;
    Python 须用项目 .venv (系统 PATH python.exe 为 WindowsApps 存根)。
  - 墙钟预期 ~30-35 分钟/臂 (exp125 实测 28-32 分钟, 未归因); 三臂串行
    独立进程, 任一组崩溃不影响其余。
  - lr 固定 0.05 (审查定点 C 条件), 不作为变量。

铁律: crates/ 零改动 (仅进程内类方法 monkeypatch); 不动 v244/迁移件
(只读归档检查点, md5 硬门); 全 fp16 (无 clamp 数值; 统计全走 .float()
只读路径); 权重不落盘 (瞬态快照即删); run_file 自增新目录。
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
from crates.kernels import rms_norm
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
ARMS = {"ctrl": {"contrast": True, "dt": True},
        "nocontrast": {"contrast": False, "dt": True},
        "nodt": {"contrast": True, "dt": False}}
SWEEP_SCALES = [1.6, 1.3, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]

_ARM = dict(ARMS["ctrl"])
_TEL = {"fresh": False}


# ─────────── E3 逐字复刻 + 只读遥测 + 双臂开关 (ctrl 双臂开 = 原函数) ───────────

def _update_lm_head_e3b_tel(self, ctx, lm):
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
    if not _ARM["contrast"]:
        contrast_w = torch.ones_like(contrast_w)
    err_contrast = err_scaled * contrast_w

    if hasattr(net, "_novelty"):
        nov = net._novelty
        tau = net._theta_novelty
        tau.mul_(0.99).add_(0.01 * nov.mean())
        d_t = torch.sigmoid((nov - tau) * 500.0).unsqueeze(-1)[:, :-1]
    else:
        d_t = net._one_s[: ctx.N, : ctx.S - 1]
    if not _ARM["dt"]:
        d_t = torch.ones_like(d_t)
    lm_update_mask = ctx.learn_mask.to(torch.float16).unsqueeze(-1)
    dW_lm = (
        rms_norm(lm.h[:, :-1]).transpose(-2, -1) @ (err_contrast * lm_update_mask * d_t)
    ).mean(dim=0) * math.sqrt(d_h)

    net._tel_dWlm_t = dW_lm.abs().mean().detach()
    net._tel_err_t = err_scaled.abs().mean().detach()

    # ── 只读遥测快照 (clone/detach; 不动训练张量, 不耗 RNG) ──
    wlm_pre = net.W_lm.data.clone()
    dW_prev = _TEL.get("dW_prev")
    _TEL["dW_prev"] = dW_lm.detach().clone()
    nov_mean = float(nov.detach().float().mean()) if hasattr(net, "_novelty") else None
    tau_mean = float(tau.detach().float().mean()) if hasattr(net, "_novelty") else None

    net.W_lm.data += dW_lm * LR
    rn_lm = net.W_lm.data.norm(dim=1)
    mx_lm = rn_lm.max()
    cap_scale = float(torch.minimum(torch.ones_like(mx_lm),
                                    10.0 / (mx_lm + 1e-6)).to(torch.float16))
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

    # ── 遥测汇总 (原函数数学路径已全部完成; 以下纯读) ──
    lg_f = logits_d.detach().float()
    p_sm = torch.softmax(lg_f, dim=-1)
    nt = (1.0 - target_oh.detach().float()) > 0.5
    cw_f = contrast_w.detach().float()
    dt_f = d_t.detach().float().flatten()
    dW_f = dW_lm.detach().float().flatten()
    wpre_f = wlm_pre.detach().float().flatten()
    nrm = lambda v: float(v.norm()) if v.numel() else 0.0
    cos = lambda a, b: float((a @ b) / (a.norm() * b.norm() + 1e-12))
    _TEL.update({
        "fresh": True,
        "logits_std": float(lg_f.std()),
        "sm_entropy": float(-(p_sm * p_sm.clamp_min(1e-9).log2()).sum(dim=-1).mean()),
        "sm_maxp": float(p_sm.max(dim=-1).values.mean()),
        "cw_nt_mean": float((cw_f * nt.float()).sum() / nt.float().sum().clamp_min(1.0)),
        "cw_nt_min": float(cw_f[nt].min()) if bool(nt.any()) else None,
        "cw_nt_frac_lt_05": float((cw_f[nt] < 0.5).float().mean()) if bool(nt.any()) else None,
        "dt_mean": float(dt_f.mean()),
        "dt_p95": float(torch.quantile(dt_f, 0.95)),
        "dt_frac_gt_05": float((dt_f > 0.5).float().mean()),
        "dt_frac_lt_01": float((dt_f < 0.1).float().mean()),
        "nov_mean": nov_mean,
        "tau_mean": tau_mean,
        "wlm_pre_norm": nrm(wpre_f),
        "wlm_row_max_precap": float(rn_lm.detach().float().max()),
        "wlm_row_mean_precap": float(rn_lm.detach().float().mean()),
        "wlm_cap_scale": cap_scale,
        "wlm_row_max_postcap": float(rn_lm.detach().float().max()) * cap_scale,
        "dW_absmean": float(dW_f.abs().mean()),
        "dW_norm": nrm(dW_f),
        "dW_cos_wlm": cos(dW_f, wpre_f),
        "dW_cos_prev": cos(dW_f, dW_prev.detach().float().flatten()) if dW_prev is not None else None,
        "dW_delta_theo_norm": nrm(dW_lm.detach().float().flatten() * LR),
        "dW_delta_act_norm": float((net.W_lm.data.float().flatten() - wpre_f).norm()),
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
    tag = f"exp125b_{arm}"
    smoke = args.smoke > 0
    steps = args.smoke if smoke else args.steps

    stdout_path = run_file(tag + "_stdout.log")
    logf_console = open(stdout_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = logf_console
    sys.stderr = logf_console

    ck = resolve_path(e124.CKPT)
    md5_now = _md5(ck)
    if md5_now != e124.CKPT_MD5:
        print(f"exp125b: FATAL md5 硬门不一致: 实测 {md5_now} != 预期 {e124.CKPT_MD5}; 拒跑", flush=True)
        sys.exit(2)
    dep = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp124_epsb_amp.py")
    dep_sha = hashlib.sha256(open(dep, "rb").read()).hexdigest()
    self_sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    print(f"exp125b: arm={arm} lr={LR} switches={ARMS[arm]} smoke={smoke} steps={steps}\n"
          f"exp125b: ckpt={ck}\nexp125b: md5={md5_now} (硬门 PASS)\n"
          f"exp125b: 依赖 exp124_epsb_amp.py sha256={dep_sha}\n"
          f"exp125b: 执行件 sha256={self_sha}", flush=True)

    ReadoutMixin._update_lm_head = _update_lm_head_e3b_tel
    print(f"exp125b: monkeypatch: E3 复刻+遥测 (contrast={'on' if _ARM['contrast'] else 'OFF'}, "
          f"d_t={'on' if _ARM['dt'] else 'OFF'}) + 部署 _update_W35 (amp 关断)", flush=True)

    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=S_MAX, lm_freeze_w1=False)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = torch.device(args.device)
    net = HybridPCNet.load(ck, cfg).to(dev)
    net.eval()
    net._migrate_expand = False
    print(f"exp125b: net loaded; d_h={net.d_h} mem_k={net._mem_m.shape[0]}; "
          f"_migrate_expand=False (topology.compact 漏同步 migrate 通道, crates 既有缺口)", flush=True)

    gauge, _, _ = build_gauge(args.data, dev)
    eval_wins = gauge["tail"][:N_EVAL_WIN]
    print(f"exp125b: eval_wins={len(eval_wins)}", flush=True)

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
            "exp": "exp125b", "arm": arm, "lr": LR, "switches": ARMS[arm], "smoke": smoke,
            "ckpt": e124.CKPT, "ckpt_md5": md5_now, "ckpt_md5_expected": e124.CKPT_MD5,
            "dep_exp124_sha256": dep_sha, "self_sha256": self_sha,
            "unique_variable": f"臂开关 (contrast_w≡1 / d_t≡1); lr 固定 {LR} (C 条件)",
            "rule": "E3 逐字复刻+只读遥测 (ctrl=E3 原样) + 部署 _update_W35 (amp 关断)",
            "fidelity_gate": {"ctrl_step1_dWlm_absmean": 1.5673828125,
                              "ctrl_step1_W_lm_norm": 75.3125,
                              "ctrl_post_bpb": 10.797670696653336,
                              "smoke_step1_dWlm_absmean": 1.6025390625,
                              "smoke_step1_W_lm_norm": 74.625,
                              "note": "与 exp125 C 正式跑 v432 / 冒烟 v429 对拍; 不一致=复刻失败"},
            "verdict_rule": "learned:=post bpb<5.0; c_type:=post>=8.0 且 top1>=0.05 且 f_now max<100; "
                            "crash_type:=post>=8.0 且非 c_type; Q2: bpb(s*)<=0.7*bpb(s=1) 即尺度主因; "
                            "Q3: dt 末段>首段x1.2 且 corr(dt,delta比)>0.3; Q4: dW 余弦均值>0.3 即方向稳定"}
    jsonl.write(json.dumps({"meta": meta}, ensure_ascii=True) + "\n")

    t0 = time.time()
    if smoke:
        pre = post = None
        base_lines = []
        base_hits = None
        print("exp125b: 冒烟模式 — 跳过 pre/post eval / dump / audit (保留后置扫描)", flush=True)
    else:
        pre = e124.eval_bpb_fresh(net, dev, eval_wins)
        print(f"exp125b[{arm}]: pre bpb@s1={pre['bpb_s1']:.6f} top1={pre['top1_s1']:.6f} "
              f"(exp125 C pre=7.664317404722619; 须逐位相同=协议保真)", flush=True)
        base_lines = _sample_dump(net, tag, cfg, dev, 0, 3, 240, 0.8, logf)
        base_hits = e124.word_hits_of(base_lines)

    ds = ByteDataset(args.data, max_length=S_MAX, max_samples=1270000)
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    print(f"exp125b[{arm}]: data n={len(ds)}", flush=True)

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
        for k in ("logits_std", "sm_entropy", "sm_maxp", "cw_nt_mean", "cw_nt_min",
                  "cw_nt_frac_lt_05", "dt_mean", "dt_p95", "dt_frac_gt_05", "dt_frac_lt_01",
                  "nov_mean", "tau_mean", "wlm_pre_norm", "wlm_row_max_precap",
                  "wlm_row_mean_precap", "wlm_cap_scale", "wlm_row_max_postcap",
                  "dW_absmean", "dW_norm", "dW_cos_wlm", "dW_cos_prev",
                  "dW_delta_theo_norm", "dW_delta_act_norm", "e0_mean", "alive_frac"):
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
            print(f"exp125b[{arm}]: step {i}/{steps} f_perc={f_perc_last:.4f} "
                  f"logits_std={tel.get('logits_std')} dt_mean={tel.get('dt_mean')} "
                  f"cap_scale={tel.get('wlm_cap_scale')} W_lm={norms[e124.PNAMES.index('W_lm')]:.4f}",
                  flush=True)
        if death_cause:
            print(f"exp125b[{arm}]: 机体自维护 death_cause={death_cause} (照常运行, 不中止)", flush=True)
            death_cause = None

    sweep = None
    if smoke:
        final_lines = []
        wh = None
        print(f"exp125b[{arm}]: 冒烟注入自检 — step1 dWlm_absmean={_gt('_tel_dWlm_t')} "
              f"W_lm_norm={_gt('_tel_Wlm_norm_t')} logits_std={tel_rows[0].get('logits_std') if tel_rows else None}",
              flush=True)
        sweep = post_sweeps(net, dev, eval_wins, tag + "_sweep")
        print(f"exp125b[{arm}]: 冒烟后置扫描验证 — best scale={sweep['best']['scale']} "
              f"bpb={sweep['best']['bpb']:.4f} vs s1={sweep['bpb_s1']:.4f} "
              f"gain={sweep['gain_ratio']:.3f} (冒烟扫描值不作判读)", flush=True)
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
        print(f"exp125b[{arm}]: post bpb@s1={post['bpb_s1']:.6f} top1={post['top1_s1']:.6f} "
              f"(ctrl 复现目标 10.797670696653336)", flush=True)
        wh = e124.word_hits_of(final_lines)
        sweep = post_sweeps(net, dev, eval_wins, tag + "_sweep")
        print(f"exp125b[{arm}]: 后置扫描 best scale={sweep['best']['scale']} "
              f"(T={sweep['best']['temp']}) bpb={sweep['best']['bpb']:.4f} "
              f"vs s1={sweep['bpb_s1']:.4f} gain={sweep['gain_ratio']:.3f} "
              f"bias_off_s1={sweep['bias_off_s1']['bpb']:.4f}", flush=True)

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

    def agg(key, subset=None):
        vals = [r[key] for r in tel_rows
                if r.get(key) is not None and (subset is None or subset(r))]
        return (sum(vals) / len(vals)) if vals else None

    perc = [r for r in tel_rows if r.get("logits_std") is not None]
    lg_first = perc[0]["logits_std"] if perc else None
    lg_last = perc[-1]["logits_std"] if perc else None
    dt_first100 = agg("dt_mean", lambda r: r["step"] <= 100)
    dt_last100 = agg("dt_mean", lambda r: r["step"] > len(tel_rows) - 100)
    cap_vals = [r["wlm_cap_scale"] for r in tel_rows if r.get("wlm_cap_scale") is not None]
    cap_bind_frac = (sum(1 for v in cap_vals if v < 1.0) / len(cap_vals)) if cap_vals else None
    delta_ratio = [r["dW_delta_act_norm"] / r["dW_delta_theo_norm"]
                   for r in tel_rows
                   if r.get("dW_delta_act_norm") and r.get("dW_delta_theo_norm")]
    corr_dt_delta = None
    xs = [r["dt_mean"] for r in tel_rows if r.get("dt_mean") is not None]
    ys = [r["dW_delta_act_norm"] / r["dW_delta_theo_norm"] for r in tel_rows
          if r.get("dt_mean") is not None and r.get("dW_delta_act_norm") and r.get("dW_delta_theo_norm")]
    if len(xs) == len(ys) and len(xs) > 10:
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sx = math.sqrt(sum((v - mx) ** 2 for v in xs))
        sy = math.sqrt(sum((v - my) ** 2 for v in ys))
        if sx > 0 and sy > 0:
            corr_dt_delta = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)

    if smoke:
        post_bpb = None
        learned = c_type = crash_type = stable = None
    else:
        post_bpb = post["bpb_s1"]
        top1_post = post["top1_s1"]
        learned = post_bpb < 5.0
        stable = post_bpb < 8.0
        c_type = (post_bpb >= 8.0) and (top1_post >= 0.05) and ((f_perc_max or 0.0) < 100.0)
        crash_type = (post_bpb >= 8.0) and not c_type

    out = {
        "exp": "exp125b", "arm": arm, "lr": LR, "switches": ARMS[arm], "smoke": smoke,
        "rule": meta["rule"], "fidelity_gate": meta["fidelity_gate"],
        "ckpt": e124.CKPT, "ckpt_md5": md5_now, "ckpt_md5_expected": e124.CKPT_MD5,
        "ckpt_md5_gate": "PASS (启动时复算比对一致)",
        "dep_exp124_sha256": dep_sha, "self_sha256": self_sha,
        "steps_run": i, "target_steps": steps, "abort": abort,
        "bpb": {"pre": (pre["bpb_s1"] if pre else None), "post": post_bpb,
                "top1_pre": (pre["top1_s1"] if pre else None),
                "top1_post": (post["top1_s1"] if post else None)},
        "verdict": {
            "criterion1_alive": {"alive": abort is None,
                                 "pass": (abort is None and i >= steps and not smoke)},
            "criterion2_bpb_s1_lt_5": {"post": post_bpb,
                                       "pass": (post_bpb is not None and post_bpb < 5.0),
                                       "criterion": "<5.0"},
            "criterion3_word_hits_ge_1": {"word_hits_T08": wh,
                                          "pass": (wh is not None and wh >= 1)}},
        "failure_mode": {"learned": learned, "stable": stable,
                         "c_type_calibration": c_type, "crash_type": crash_type,
                         "definitions": "learned:=post<5.0; c_type:=post>=8.0 且 top1>=0.05 且 "
                                        "f_now max<100; crash_type:=post>=8.0 且非 c_type"},
        "growth_diag": {
            "W_lm_init": init_norms.get("W_lm"),
            "W_lm_final": (norms_final[e124.PNAMES.index("W_lm")] if not smoke else None),
            "logits_std_first": lg_first, "logits_std_last": lg_last,
            "logits_std_ratio": (lg_last / lg_first) if (lg_first and lg_last) else None,
            "sm_entropy_first": (perc[0]["sm_entropy"] if perc else None),
            "sm_entropy_last": (perc[-1]["sm_entropy"] if perc else None),
            "sm_maxp_first": (perc[0]["sm_maxp"] if perc else None),
            "sm_maxp_last": (perc[-1]["sm_maxp"] if perc else None),
            "dW_cos_wlm_mean": agg("dW_cos_wlm"),
            "dW_cos_prev_mean": agg("dW_cos_prev"),
            "dW_delta_ratio_mean": (sum(delta_ratio) / len(delta_ratio)) if delta_ratio else None,
            "cap10_binding_frac": cap_bind_frac,
            "wlm_row_max_precap_last": (perc[-1]["wlm_row_max_precap"] if perc else None),
            "e0_mean_first": (perc[0]["e0_mean"] if perc else None),
            "e0_mean_last": (perc[-1]["e0_mean"] if perc else None),
            "dt_mean_first100": dt_first100, "dt_mean_last100": dt_last100,
            "dt_frac_gt_05_mean": agg("dt_frac_gt_05"),
            "corr_dt_delta_ratio": corr_dt_delta,
            "cw_nt_mean_first": (perc[0]["cw_nt_mean"] if perc else None),
            "cw_nt_mean_last": (perc[-1]["cw_nt_mean"] if perc else None),
            "cw_nt_frac_lt_05_mean": agg("cw_nt_frac_lt_05")},
        "sweep": sweep,
        "q_flags_per_arm": {
            "Q2_scale_vs_directional": (
                None if sweep is None else
                ("scale" if sweep["gain_ratio"] <= 0.7 else "directional")),
            "Q3_dt_feedback": (
                None if smoke else
                ("suspected" if (dt_first100 is not None and dt_last100 is not None
                                 and dt_last100 > 1.2 * dt_first100
                                 and (corr_dt_delta if corr_dt_delta is not None else 0.0) > 0.3)
                 else "no_evidence")),
            "Q4_moving_target": (
                None if smoke else
                ("stable_direction" if (agg("dW_cos_prev") or 0.0) > 0.3 else "moving_target")),
            "note": "Q1 跨臂判定在 exp125b_result.md (预登记规则); 本 json 只出原料"},
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
                                   "f_now/E/R + d_t/contrast/logits/softmax熵/cap10/dW 余弦)",
                     "samples_log": tag + ".log",
                     "stdout": tag + "_stdout.log",
                     "audit": (tag + "_audit_char.txt" if not smoke else None),
                     "sweep": ("内嵌 verdict.sweep" if not smoke else None)},
    }
    with open(run_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"exp125b[{arm}]: 写出 {run_json}", flush=True)
    print(f"exp125b[{arm}]: 判定摘要 abort={abort} bpb {pre['bpb_s1'] if pre else None}->{post_bpb} "
          f"word_hits={wh} learned={learned} c_type={c_type} crash_type={crash_type} "
          f"logits_std {lg_first}->{lg_last}", flush=True)
    print(f"exp125b[{arm}]: DONE", flush=True)
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
        crash = resolve_path(f"out/exp125b_crash_{a}.txt")
        with open(crash, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        sys.stderr.write(traceback.format_exc())
        sys.exit(1)
