"""
exp125: 端到端 lr 校准 (E3 x lr 小网格) — 500 步 x 3 组
=====================================================

立项依据 (2026-10-18 审查裁决, 聊天记录):
  "exp125 (端到端 lr 校准) 批准执行, A/B/C 三组 lr in {0.5, 0.2, 0.05}, 先跑"
  "执行口径: agent 直接跑 exp125。不再问'要不要开始'。
   跑完首句给三组 bpb@s1 + 哪一组稳定 + word_hits。"

核心观察 (裁决原文): E3 在 harness (batch 8192, lr 2.0) 收敛到 5.56; 同一规则
端到端 (batch 255, lr 2.0) 塌到 19.88。单窗更新的梯度噪声是 harness 的
sqrt(8192/255) ~ 5.7 倍 — lr 2.0 是为大批校准的, 端到端单窗下它太高了。

待检验假设 (非结论): 端到端塌缩的首要原因是 lr 未随 batch 缩小重新校准,
而非结构缺陷 (moving target / cap 交互 / 正反馈)。

预登记: docs/plan/exp125_preregistration.md (判读格与分流规则预写, 跑前锁哈希;
本文件跑前写入, 其 sha256 一并入账)。

─── 设计 (与预登记逐字对应) ───

唯一变量 = E3 修复链的固定步长 lr (W_lm 与 W1 共用同一 LR 常量, 与 replica
结构一致): A=0.5 / B=0.2 / C=0.05, 三组独立进程串行。

规则 = E3 replica (exp124_epsb_amp._update_lm_head_e3_biasfrozen 逐位沿用,
仅其模块级 LR 常量按组覆盖 — 函数体零改动, 不 fork 复刻) + 部署版
_update_W35 (amp 关断):
  - amp 是 exp124 新增, 不属 exp123b 规则; 裁决设计表写 "E3 规则: 同 exp123b";
  - exp124c (v426) 已证 amp 开 (v419) 与 amp 关 post bpb 逐位相同
    (38.02853959343252), amp 状态对本类实验为中性变量, 按 exp123b 规则取关断。

同参承诺 (与 exp123b/exp124/exp124c 同口径):
  同检查点 (md5 硬门 57D45872BABCFDD1FBC15CF49029F344, 启动复算不一致拒跑) /
  同数据 (dataset/pretrain_t2t_mini.jsonl, ByteDataset S_MAX=256,
  max_samples=1270000, python random seed 0 洗牌) / 同种子 (0) /
  同协议 (首步 force_perception; P3-b 行为门路由; sync_gate + maybe_prune;
  S0 基线 dump; i%100 dump + S500 final dump (3 种子, 240B, tau=0.8);
  pre/post bpb 锚点 eval_bpb_fresh s=1.0) / 同中止规则 R1/R2/R2b/R3 /
  同遥测口径 (52 参数范数全数组 + f_now/E/R + 熵滑窗)。

基线 (预写参照, 不重跑 lr 2.0 — 裁决只批 A/B/C):
  - exp123b 历史 (脚本丢失, 口径不同, 仅参照): 7.666549 -> 19.877569 (lr 2.0)
  - 重建锚点 v426 (同规则 amp 关断, lr 2.0, 同检查点/数据/种子/协议):
    7.664317 -> 38.028540
  - harness 参照 (batch 8192, lr 2.0): 5.56 (审查引用)

判据与分流 (预登记 §三/§四, 操作化定义):
  stable  := abort==null 且 post bpb@s1 < 8.0 (8.0 = 均匀分布参照上界;
             >=8.0 即 "自信地错反向预测" regime, exp124 materials 用语)
  learned := post bpb@s1 < 5.0
  collapsed := abort!=null 或 post bpb@s1 >= 8.0
  A 线 (lr 校准坐实): 存在组 stable 且 learned
  B 线 (结构性问题): 全部组 collapsed -> exp125b 拆结构
  C 线 (中间态): 其余 -> exp125 未通过; 最稳组 post<pre 记 "lr 部分有效"
             (建议延长步数复测), post>=pre 按 B 线处理

中止规则 (预写, 必须执行; 与 exp124/exp123b 逐位同口径):
  R1  任一可训张量范数 > 3x init (零初始化累加器 iv<=0 跳过; 其失稳由 R3 与
      下游张量兜底)
  R2  生成字符熵 (滑窗 8192B, >=256B 起算, bits/byte) 连续 100 次更新下降
      或上升
  R2b 任一次 dump 内 3 个不同种子样本逐字相同 (同一次 dump 内口径, 非跨
      dump; exp123b step400 先例正式化)
  R3  数值异常 (norms 非有限 / _dW1_absmax_raw 非有限)
  机体自维护 (maybe_prune 死亡回合) 照常运行, 不中止

偏离披露:
  - 宿主 PowerShell stdout 通道故障 (命令有文件副作用但无回显; exp124/exp124c
    同款): 本脚本启动即把 sys.stdout/sys.stderr 重定向到 run 目录内
    exp125_{group}_stdout.log (run_file 首次调用创建版本目录), 非交互前台;
    运行中不 kill (纪律 13)。Python 解析器须用项目 .venv (系统 PATH 上的
    python.exe 是 WindowsApps 占位存根, 2026-10-18 实证 exit 1 无副作用)。
  - E3 replica 的 LR 注入方式: 模块级常量 e124.LR = lr (函数经 __globals__
    读取), 非函数体改动 — exp124c 同款 import 依赖, 运行时 sha256 入账; 若
    exp124_epsb_amp.py 被改, 本跑基线随之变。
  - 遥测无 amp 列 (amp 关断; v426 同口径), 行Schema = exp124c 行 + lr 字段
    + W_lm/W1/bias 范数与 dW 幅度 (exp124 口径) - amp 列。
  - 冒烟模式 (--smoke N): 跳过 pre/post eval / S0 dump / 全部 dump / audit,
    仅验证注入与遥测管线; 冒烟不作为判读依据。

铁律: crates/ 零改动 (仅进程内类方法 monkeypatch); 不动 v244/迁移件 (只读
归档检查点); 全 fp16 (无 clamp 数值); 权重不落盘 (E9f 瞬态快照即删);
run_file 自增新目录; 三组独立进程串行 (任一组崩溃不影响其余)。
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

import exp124_epsb_amp as e124  # 依赖: E3 replica + eval 仪器 + PNAMES (sha256 运行时入账)
from exp115_say import N_EVAL_WIN, S_MAX, _sample_dump, build_gauge
from exp116_z4_probe import _md5

from crates.cli.utils import resolve_path, run_file
from crates.graphs import set_backend
set_backend("ref")
from crates.hybrid.learning.readout import ReadoutMixin
from crates.model import CyreneModel, HybridPCNet
from crates.utils.codec import ByteDataset

torch.set_grad_enabled(False)

TARGET_STEPS = 500
ENT_WIN = 8192
ENT_MIN = 256
ENT_STREAK = 100
GROUPS = {"A": 0.5, "B": 0.2, "C": 0.05}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=sorted(GROUPS))
    ap.add_argument("--steps", type=int, default=TARGET_STEPS)
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0,
                    help="冒烟步数 (0=正式; 冒烟跳过 eval/dump/audit, 不作判读依据)")
    args = ap.parse_args()
    lr = GROUPS[args.group]
    tag = f"exp125_{args.group}"
    smoke = args.smoke > 0
    steps = args.smoke if smoke else args.steps

    # stdout/stderr 落盘 run 目录 (宿主 stdout 通道故障; run_file 首调用建版本目录)
    stdout_path = run_file(tag + "_stdout.log")
    logf_console = open(stdout_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = logf_console
    sys.stderr = logf_console

    ck = resolve_path(e124.CKPT)
    md5_now = _md5(ck)
    if md5_now != e124.CKPT_MD5:
        print(f"exp125: FATAL md5 硬门不一致: 实测 {md5_now} != 预期 {e124.CKPT_MD5}; 拒跑", flush=True)
        sys.exit(2)
    dep = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp124_epsb_amp.py")
    dep_sha = hashlib.sha256(open(dep, "rb").read()).hexdigest()
    print(f"exp125: group={args.group} lr={lr} (唯一变量), smoke={smoke}, steps={steps}\n"
          f"exp125: ckpt={ck}\nexp125: md5={md5_now} (硬门 PASS)\n"
          f"exp125: 依赖 exp124_epsb_amp.py sha256={dep_sha}", flush=True)

    # E3 replica + lr 单变量 (模块级常量注入, 函数体零改动)
    e124.LR = lr
    assert e124.LR == lr, "LR 注入失败"
    ReadoutMixin._update_lm_head = e124._update_lm_head_e3_biasfrozen
    print(f"exp125: monkeypatch: E3 replica (LR={lr} 经模块级常量注入函数 __globals__) "
          f"+ 部署 _update_W35 (amp 关断 — exp124c 已证 amp 开/关 post bpb 逐位相同)", flush=True)

    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=S_MAX, lm_freeze_w1=False)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = torch.device(args.device)
    net = HybridPCNet.load(ck, cfg).to(dev)
    net.eval()
    net._migrate_expand = False
    print(f"exp125: net loaded; mem_k={net._mem_m.shape[0]}; _migrate_expand=False "
          f"(topology.compact 漏同步 migrate 通道, crates 既有缺口, 零改动纪律下关闭)", flush=True)

    gauge, _, _ = build_gauge(args.data, dev)
    eval_wins = gauge["tail"][:N_EVAL_WIN]
    print(f"exp125: eval_wins={len(eval_wins)}", flush=True)

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
            "exp": "exp125", "group": args.group, "lr": lr, "smoke": smoke,
            "ckpt": e124.CKPT, "ckpt_md5": md5_now, "ckpt_md5_expected": e124.CKPT_MD5,
            "dep_exp124_sha256": dep_sha,
            "unique_variable": f"E3 修复链固定步长 lr (W_lm/W1 共用同一 LR 常量): {lr}",
            "rule": "E3 replica (exp124_epsb_amp._update_lm_head_e3_biasfrozen, 逐位沿用, "
                    "仅模块级 LR 常量按组覆盖) + 部署 _update_W35 (amp 关断)",
            "baselines": {
                "exp123b_historical_lr2.0": {"pre": 7.666549, "post": 19.877569,
                                             "note": "脚本丢失, 口径不同, 仅参照"},
                "v426_reconstruction_lr2.0_amp_off": {"pre": 7.664317, "post": 38.028540,
                                                      "note": "同规则/同检查点/同数据/同种子/同协议"},
                "harness_batch8192_lr2.0": {"bpb": 5.56,
                                            "note": "审查引用; 端到端/harness 梯度噪声比 sqrt(8192/255)~5.7"}},
            "protocol": "首步 force_perception; P3-b 行为门路由; sync_gate + maybe_prune; "
                        "S0 dump; i%100 dump + S500 final dump (3 种子 240B tau=0.8); "
                        "pre/post eval_bpb_fresh s=1.0; R1/R2/R2b/R3",
            "verdict_rule": "stable:=abort==null 且 post bpb<8.0; learned:=post bpb<5.0; "
                            "collapsed:=abort!=null 或 post bpb>=8.0; A 线: 存在组 stable 且 learned; "
                            "B 线: 全部 collapsed; C 线: 其余 (最稳组 post<pre 记 lr 部分有效, 否则按 B 线)"}
    jsonl.write(json.dumps({"meta": meta}, ensure_ascii=True) + "\n")

    t0 = time.time()
    if smoke:
        pre = post = None
        base_lines = []
        base_hits = None
        print("exp125: 冒烟模式 — 跳过 pre/post eval / S0 dump / dump / audit", flush=True)
    else:
        pre = e124.eval_bpb_fresh(net, dev, eval_wins)
        print(f"exp125: pre bpb@s1={pre['bpb_s1']:.6f} top1={pre['top1_s1']:.6f} "
              f"(v426 重建锚点 pre=7.664317; exp123b 历史 pre=7.666549)", flush=True)
        base_lines = _sample_dump(net, tag, cfg, dev, 0, 3, 240, 0.8, logf)
        base_hits = e124.word_hits_of(base_lines)

    ds = ByteDataset(args.data, max_length=S_MAX, max_samples=1270000)
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    print(f"exp125: data n={len(ds)} (P3-b 行为门路由, 首步 force_perception)", flush=True)

    ent_buf = bytearray()
    ent_first = ent_last = None
    ent_min = ent_max = None
    ent_updates = 0
    decline_streak = rise_streak = 0
    abort = None
    f_perc_first = f_perc_last = None
    f_eco_last = None
    E_last = R_last = None
    death_cause = None
    dump_ents = {}
    wlm_init = float(net.W_lm.data.norm())

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
        norms = e124.param_norms(net)
        nonfinite = [n for n, v in zip(e124.PNAMES, norms) if v is None or not math.isfinite(v)]
        d1r = float(net._dW1_absmax_raw) if hasattr(net, "_dW1_absmax_raw") else None

        def _gt(name):
            t = getattr(net, name, None)
            return float(t) if t is not None else None

        row = {"step": i, "lr": lr, "behavior": int(exec_b),
               "W_lm_norm": _gt("_tel_Wlm_norm_t"), "W1_norm": _gt("_tel_W1_norm_t"),
               "bias_lm_norm": _gt("_tel_bias_norm_t"),
               "dWlm_absmean": _gt("_tel_dWlm_t"), "dW1_absmean": _gt("_tel_dW1_t"),
               "dW1_absmax_raw": d1r,
               "f_now_perc": f_perc_last, "f_now_eco": f_eco_last,
               "metab_E": E_last, "metab_R": R_last,
               "char_entropy": entropy_now(), "ent_decline_streak": decline_streak,
               "ent_rise_streak": rise_streak, "ent_updates": ent_updates,
               "norms": norms, "mem_k": net._mem_m.shape[0]}
        jsonl.write(json.dumps(row, ensure_ascii=True) + "\n")

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
                abort = f"R2b: step {i} dump 内 3 个不同种子样本逐字相同 (exp123b step400 先例正式化)"
                break
        if i % 100 == 0:
            print(f"exp125[{args.group} lr={lr}]: step {i}/{steps} f_perc={f_perc_last:.4f} "
                  f"E={E_last:.4f} R={R_last:.4f} W_lm={norms[e124.PNAMES.index('W_lm')]:.6f} "
                  f"M_l5={norms[e124.PNAMES.index('M_l5')]:.6f} mem_k={net._mem_m.shape[0]}", flush=True)
        if death_cause:
            print(f"exp125[{args.group}]: 机体自维护 death_cause={death_cause} (照常运行, 不中止)", flush=True)
            death_cause = None

    if smoke:
        final_lines = []
        wh = None
        print(f"exp125[{args.group}]: 冒烟注入自检 — W_lm_init_norm={wlm_init:.6f} "
              f"step1_dWlm_absmean={_gt('_tel_dWlm_t')} step1_W_lm_norm={_gt('_tel_Wlm_norm_t')} "
              f"(跨组 dWlm_absmean 应一致; W_lm 增量应正比于 lr)", flush=True)
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
        print(f"exp125[{args.group} lr={lr}]: post bpb@s1={post['bpb_s1']:.6f} "
              f"top1={post['top1_s1']:.6f} (v426 lr2.0 锚点 post=38.028540; "
              f"exp123b 历史 post=19.877569; harness 参照 5.56)", flush=True)
        wh = e124.word_hits_of(final_lines)

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
        audit_path = run_file(tag + "_audit_char.txt")
        with open(audit_path, "w", encoding="utf-8", buffering=1) as f:
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
            f.write("\n== 全样本完整解码 (无截断) ==\n")
            for name, texts in [(f"S{s}", dump_ents[s]) for s in sorted(dump_ents)]:
                f.write(f"\n---- {name} ----\n")
                for k, t in enumerate(texts):
                    f.write(f"[{k:02d}] {t!r}\n")
            f.write(f"\n---- 语料基线 ----\n{corpus_text[:600]} ...\n")

    norms_final = e124.param_norms(net)
    drift = []
    for n, v in zip(e124.PNAMES, norms_final):
        iv = init_norms.get(n)
        if iv is not None and iv > 0 and v is not None:
            drift.append({"name": n, "init": iv, "final": v, "ratio": v / iv})
    drift.sort(key=lambda r: -r["ratio"])

    if smoke:
        post_bpb = None
        stable = learned = collapsed = None
    else:
        post_bpb = post["bpb_s1"]
        stable = (abort is None) and (post_bpb < 8.0)
        learned = post_bpb < 5.0
        collapsed = not stable

    out = {
        "exp": "exp125", "group": args.group, "lr": lr, "smoke": smoke,
        "unique_variable": f"E3 修复链固定步长 lr (W_lm/W1 共用同一 LR 常量): {lr}",
        "rule": meta["rule"],
        "ckpt": e124.CKPT, "ckpt_md5": md5_now, "ckpt_md5_expected": e124.CKPT_MD5,
        "ckpt_md5_gate": "PASS (启动时复算比对一致)",
        "dep_exp124_sha256": dep_sha,
        "baselines": meta["baselines"],
        "protocol": meta["protocol"],
        "steps_run": i, "target_steps": steps, "abort": abort,
        "bpb": {"pre": (pre["bpb_s1"] if pre else None),
                "post": post_bpb,
                "top1_pre": (pre["top1_s1"] if pre else None),
                "top1_post": (post["top1_s1"] if post else None)},
        "verdict": {
            "criterion1_alive": {"alive": abort is None,
                                 "pass": (abort is None and i >= steps and not smoke)},
            "criterion2_bpb_s1_lt_5": {"post": post_bpb,
                                       "pass": (post_bpb is not None and post_bpb < 5.0),
                                       "criterion": "<5.0"},
            "criterion3_word_hits_ge_1": {"word_hits_T08": wh, "pass": (wh is not None and wh >= 1),
                                          "reference": "语料基线 118, 物理地板 0, tau=0.8 已声明"}},
        "triage_flags": {"stable": stable, "learned": learned, "collapsed": collapsed,
                         "definitions": "stable:=abort==null 且 post bpb<8.0; learned:=post bpb<5.0; "
                                        "collapsed:=abort!=null 或 post bpb>=8.0",
                         "note": "跨组 A/B/C 分流结论见 docs/plan/exp125_result.md (预登记 §四)"},
        "word_hits_trajectory": ({s: e124.word_hits_of(dump_ents[s]) for s in sorted(dump_ents)}
                                 if not smoke else None),
        "word_hits_baseline_step0": base_hits,
        "entropy_bidirectional": {"first_H": ent_first, "last_H": ent_last,
                                  "min_H": ent_min, "max_H": ent_max,
                                  "updates": ent_updates,
                                  "decline_streak_end": decline_streak,
                                  "rise_streak_end": rise_streak,
                                  "pass_bidirectional": (decline_streak < ENT_STREAK
                                                         and rise_streak < ENT_STREAK)},
        "fnow_trajectory": {"f_now_perc_first": f_perc_first, "f_now_perc_last": f_perc_last,
                            "f_now_eco_last": f_eco_last, "E_last": E_last, "R_last": R_last},
        "drift_top10": drift[:10],
        "bias_isolation_check": {"bias_norm_init": init_norms.get("bias_lm"),
                                 "bias_norm_final": float(net.bias_lm.data.norm())},
        "secs": round(time.time() - t0, 1),
        "iron_rules": "crates/ 零改动 (进程内 monkeypatch); 不动 v244/迁移件 (只读归档检查点); "
                      "全 fp16 (无 clamp 数值); 权重不落盘 (E9f 瞬态快照即删); run_file 自增新目录; "
                      "stdout/stderr 落盘 run 目录 (宿主通道故障); 运行不 kill; 独立进程串行",
        "products": {"verdict": tag + ".json",
                     "telemetry": tag + "_telemetry.jsonl (首行 meta, 每行 norms 全数组 + f_now/E/R + lr)",
                     "samples_log": tag + ".log",
                     "stdout": tag + "_stdout.log",
                     "audit": (tag + "_audit_char.txt" if not smoke else None)},
    }
    with open(run_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"exp125[{args.group} lr={lr}]: 写出 {run_json}", flush=True)
    print(f"exp125[{args.group} lr={lr}]: 判定摘要 abort={abort} "
          f"bpb {pre['bpb_s1'] if pre else None}->{post_bpb} word_hits={wh} "
          f"stable={stable} learned={learned} collapsed={collapsed}", flush=True)
    print(f"exp125[{args.group} lr={lr}]: DONE", flush=True)
    logf.close()
    jsonl.close()
    logf_console.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        g = "unknown"
        for a in sys.argv:
            if a in GROUPS:
                g = a
        crash = resolve_path(f"out/exp125_crash_{g}.txt")
        with open(crash, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        sys.stderr.write(traceback.format_exc())
        sys.exit(1)
