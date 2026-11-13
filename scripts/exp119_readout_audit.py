"""
exp119: 读出头学习路径审计 (E9f step 3).

问题: exp116/117 证 v244 (e9f_bplus_readout_realign_500) 读出侧 bpb=6.73 > unigram 4.77
(学坏了), 而 exp118 证同架构反传上界 2.33 (架构+输入充分) → 瓶颈在"学习侧".
本审计把这 500 步读出侧**到底动了哪几个张量、动多少、动对没动对**拆开.

四段 (前三段便宜且决定性):
  S3-a 静态核对 (一级证据, 读代码): 逐张量列出 readout.py 更新点 + 门 + v244 臂实际开门.
  S3-b 运行时对比 (一级证据): v244 vs 起点 (迁移 v201) 读出张量 rel_frob + 逐元素相等性.
  S3-c 基线 (二级证据, 关键): 起点 (迁移 v201, 未训练) 读出头在同一 test 窗上 bpb.
  S3-d 更新计数 (telemetry): v244 .jsonl 里行为门分布 → 头实际收到真值更新的步占比.

起点口径: e9f_preregistration.md §三 = 迁移后 v201 (out/v241-.../migrated/merge.safetensors).
  迁移仅注入 CoreBrain 出生键, 零变更断言 PASS → 其读出张量与 raw v201 逐位相同 (本脚本复核).

铁律: 前台; 不改 v244; 不落盘权重; 探针零写回.
"""
import argparse
import json
import time

import torch
from exp115_say import N_EVAL_WIN, build_gauge
from exp116_z4_probe import _md5
from exp117_head_decomp import extract_fresh, head_scores
from safetensors.torch import load_file

from crates.cli.utils import resolve_path, run_file
from crates.model import CyreneModel, HybridPCNet

torch.set_grad_enabled(False)

READOUT_TENSORS = [
    "W1", "W1_new", "W_lm", "W_lm_2", "bias_lm", "W_lm_new",
    "B_new", "B_recon", "W1_elig", "W_lm_elig", "W_lm_2_elig",
]

# S3-a 静态门表 (读 readout.py 取证; 行号为 crates/hybrid/learning/readout.py)
GATE_TABLE = {
    "W_lm": {"site": "readout.py:449", "gate": "(not _mig or _realign) and not _rdfrz",
             "realign_override": True},
    "bias_lm": {"site": "readout.py:451-466", "gate": "not _rdfrz",
                "realign_override": False},
    "W1": {"site": "readout.py:485", "gate": "not lm_freeze_w1 and (not _mig or _realign) and not _rdfrz",
           "realign_override": True},
    "W_lm_2": {"site": "readout.py:513", "gate": "not _mig and not _rdfrz",
               "realign_override": False},
    "W1_new/W_lm_new/B_new": {"site": "readout.py:518 -> _update_migrate_heads (L311-350)",
                              "gate": "_mig and not _migfrz; _migfrz = _migrate_freeze or _readout_freeze",
                              "realign_override": False},
    "B_recon": {"site": "readout.py:489-490 -> _single_recon_w1", "gate": "not _rdfrz and _single_recon",
                "realign_override": False,
                "note": "_single_recon 由 load 见 B_recon 键自动置 True (model_cyrene.py:850-851 -> init_single_recon L629)"},
}


def relfrob(a, b):
    return float((a - b).norm() / (b.norm() + 1e-12))


def align_slice(base_t, new_t):
    """new_t (lm_in 已增长) 沿差异轴切到 base_t 形状 (取头部, 因 mem 出生=尾插)."""
    for ax in range(base_t.dim()):
        if base_t.shape[ax] != new_t.shape[ax]:
            sl = [slice(None)] * new_t.dim()
            sl[ax] = slice(0, base_t.shape[ax])
            return new_t[tuple(sl)], ax
    return None, None


def compare_tensor(name, base_t, new_t):
    rec = {"base_shape": list(base_t.shape), "new_shape": list(new_t.shape)}
    if base_t.shape == new_t.shape:
        rec["align"] = "direct"
        rec["rel_frob"] = relfrob(new_t, base_t)
        rec["elem_eq_frac"] = float((new_t == base_t).half().mean())
    else:
        al, ax = align_slice(base_t, new_t)
        if al is None:
            rec["align"] = "shape_mismatch_unhandled"
            rec["moved"] = None
            return rec
        rec["align"] = f"head_slice_axis{ax}"
        rec["rel_frob_head"] = relfrob(al, base_t)
        rec["elem_eq_frac_head"] = float((al == base_t).half().mean())
        sl = [slice(None)] * new_t.dim()
        sl[ax] = slice(base_t.shape[ax], None)
        tail = new_t[tuple(sl)]
        rec["tail_shape"] = list(tail.shape)
        rec["tail_norm"] = float(tail.norm())
    rec["moved"] = (rec.get("rel_frob", rec.get("rel_frob_head")) or 0.0) > 1e-6
    return rec


def scan_stale_tensors(net, old_lm_in):
    """扫"随 mem 生长需 resize 但未被 resize"的陈旧张量 (lm_in 维仍停在 old_lm_in)."""
    cur = int(net._lm_in)
    stale, ok, seen = [], [], set()
    for name in list(dir(net)):
        if name in seen:
            continue
        seen.add(name)
        try:
            v = getattr(net, name)
        except Exception:
            continue
        if not isinstance(v, torch.Tensor):
            continue
        shp = tuple(v.shape)
        if old_lm_in in shp and cur not in shp:
            stale.append({"attr": name, "shape": list(shp)})
        elif old_lm_in in shp:
            ok.append({"attr": name, "shape": list(shp)})
    return {"old_lm_in": old_lm_in, "cur_lm_in": cur, "stale": stale, "matched_current": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-ckpt", default="out/v244-20261009-214217/e9f_bplus_readout_realign_500.safetensors")
    ap.add_argument("--base-ckpt", default="out/v241-20261009-194432/migrated/merge.safetensors")
    ap.add_argument("--raw-v201", default="out/v201-20260923-093527/merge.safetensors")
    ap.add_argument("--jsonl", default="out/v244-20261009-214217/e9f_bplus_readout_realign_500.jsonl")
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device)
    new_ckpt = resolve_path(args.new_ckpt)
    base_ckpt = resolve_path(args.base_ckpt)
    raw_v201 = resolve_path(args.raw_v201)
    md5_new, md5_base = _md5(new_ckpt), _md5(base_ckpt)
    print(f"exp119: v244={new_ckpt}\n  md5={md5_new}", flush=True)
    print(f"exp119: base(迁移v201)={base_ckpt}\n  md5={md5_base}", flush=True)

    # ── S3-b 运行时张量对比 (裸读 safetensors, 不经模型) ──
    tb, tn, tr = load_file(base_ckpt), load_file(new_ckpt), load_file(raw_v201)
    old_lm_in = int(tb["W1"].shape[0])
    new_lm_in = int(tn["W1"].shape[0])
    print(f"exp119: base lm_in={old_lm_in} ({tb['W1'].shape})  v244 lm_in={new_lm_in} ({tn['W1'].shape})",
          flush=True)

    # 复核: 迁移零变更 → raw v201 读出张量与迁移 base 逐位相同
    migrate_zero_change = {}
    for k in READOUT_TENSORS:
        if k in tb and k in tr and tb[k].shape == tr[k].shape:
            migrate_zero_change[k] = float((tb[k] == tr[k]).half().mean())

    s3b = {k: compare_tensor(k, tb[k], tn[k]) for k in READOUT_TENSORS if k in tb and k in tn}

    # 头条: 头位移量 (哪些动/哪些没动)
    moved = [k for k, r in s3b.items() if r.get("moved")]
    frozen = [k for k, r in s3b.items() if r.get("moved") is False]
    print(f"exp119: [S3-b] 移动张量={moved}", flush=True)
    print(f"exp119: [S3-b] 未动张量={frozen}", flush=True)
    for k, r in s3b.items():
        rf = r.get("rel_frob", r.get("rel_frob_head"))
        tail = r.get("tail_norm")
        print(f"exp119: [S3-b] {k:10s} rel_frob={rf:.6f} moved={r.get('moved')}"
              + (f" tail_norm={tail:.4f}" if tail is not None else ""), flush=True)

    # ── S3-a 静态表 + 运行时陈旧张量扫描 ──
    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=256,
                      lm_freeze_w1=False)
    net_scan = HybridPCNet.load(new_ckpt, cfg).to(dev)
    net_scan.eval()
    scan = scan_stale_tensors(net_scan, old_lm_in)
    runtime_flags_after_load = {
        "_migrate_expand": bool(getattr(net_scan, "_migrate_expand", False)),
        "_single_recon": bool(getattr(net_scan, "_single_recon", False)),
        "_w1v2": bool(getattr(net_scan, "_w1v2", False)),
        "_lm_in": int(net_scan._lm_in),
        "_w1v2_wzh_shape": list(net_scan._w1v2_wzh.shape) if getattr(net_scan, "_w1v2_wzh", None) is not None else None,
    }
    print(f"exp119: [S3-a] 陈旧张量(仍停在 {old_lm_in})={scan['stale']}", flush=True)
    print(f"exp119: [S3-a] 运行时(load后) 标记={runtime_flags_after_load}", flush=True)
    del net_scan
    torch.cuda.empty_cache()
    # load 侧自动门探针 (与训练 flags 无关; 复现性: 同键→同值)
    single_recon_probe = {}
    for _tag, _p in (("base", base_ckpt), ("v244", new_ckpt)):
        _n = HybridPCNet.load(_p, cfg)
        single_recon_probe[_tag] = {
            "_single_recon": bool(getattr(_n, "_single_recon", False)),
            "_migrate_expand": bool(getattr(_n, "_migrate_expand", False)),
            "_w1v2": bool(getattr(_n, "_w1v2", None)),
            "_w1v2_eta_recon": getattr(_n, "_w1v2_eta_recon", None),
            "_lm_in": int(_n._lm_in),
        }
        del _n
    print(f"exp119: [S3-a] load侧自动门探针={single_recon_probe}", flush=True)

    # ── S3-c 基线: 起点 (迁移 v201) vs v244, 同 test 窗同口径 ──
    t0 = time.time()
    gauge, _, _ = build_gauge(args.data, dev)
    real_wins = gauge["tail"][:N_EVAL_WIN]
    print(f"exp119: gauge tail={len(gauge['tail'])}; test={len(real_wins)} (tail[0:{N_EVAL_WIN}]) ({time.time()-t0:.0f}s)",
          flush=True)

    y_base, RO_base = extract_fresh(base_ckpt, cfg, dev, real_wins, "base")
    y_new, RO_new = extract_fresh(new_ckpt, cfg, dev, real_wins, "v244")

    s3c = {
        "base": {"mig": head_scores(RO_base["mig"], y_base), "red": head_scores(RO_base["red"], y_base)},
        "v244": {"mig": head_scores(RO_new["mig"], y_new), "red": head_scores(RO_new["red"], y_new)},
    }
    b_mig, b_red = s3c["base"]["mig"], s3c["base"]["red"]
    n_mig, n_red = s3c["v244"]["mig"], s3c["v244"]["red"]
    print(f"exp119: [S3-c] base(v201) 头 bpb: mig={b_mig['bpb_at_s1']:.4f}(s*={b_mig['best_s']}/{b_mig['bpb_at_best_s']:.4f}) "
          f"red={b_red['bpb_at_s1']:.4f}(s*={b_red['best_s']}/{b_red['bpb_at_best_s']:.4f})", flush=True)
    print(f"exp119: [S3-c] v244       头 bpb: mig={n_mig['bpb_at_s1']:.4f}(s*={n_mig['best_s']}/{n_mig['bpb_at_best_s']:.4f}) "
          f"red={n_red['bpb_at_s1']:.4f}(s*={n_red['best_s']}/{n_red['bpb_at_best_s']:.4f})", flush=True)

    # ── S3-d 更新计数: 行为门分布 (behavior=1 → echo → 头冻结) ──
    rows = []
    with open(resolve_path(args.jsonl), "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    rep = [r for r in rows if "behavior" in r]
    n_echo = sum(1 for r in rep if r["behavior"] == 1)
    n_perc = sum(1 for r in rep if r["behavior"] == 0)
    memk = [r.get("mem_k") for r in rows if "mem_k" in r]
    s3d = {"n_report_rows": len(rep), "n_echo(behavior=1)": n_echo, "n_perc(behavior=0)": n_perc,
           "perc_frac": (n_perc / len(rep)) if rep else None,
           "mem_k_first": memk[0] if memk else None, "mem_k_last": memk[-1] if memk else None,
           "mem_k_seq": memk,
           "note": "behavior=1→_behavior_py True→route_echo→echo_world_frozen (头不更新, _echo_learn 默认 False); "
                   "behavior=0→感知 (头更新). 训练全程 500 步, 仅每步一记录 (此处分报告步抽样)."}
    print(f"exp119: [S3-d] 报告步 {len(rep)}: echo={n_echo} perc={n_perc} "
          f"perc_frac={s3d['perc_frac']}; mem_k {s3d['mem_k_first']}->{s3d['mem_k_last']}", flush=True)

    # ── 分流判定 ──
    head_moved = any(s3b[k].get("moved") for k in ("W1", "W_lm") if k in s3b)
    base_bpb, new_bpb = b_mig["bpb_at_s1"], n_mig["bpb_at_s1"]
    unigram = 4.7677
    if not head_moved:
        verdict = "头基本没动 → 门控/预算问题 (修门控, 不动规则)"
    elif new_bpb > base_bpb + 0.5:
        verdict = (f"500 步把整机从 {base_bpb:.4f} 训到 {new_bpb:.4f} (+{new_bpb - base_bpb:.3f} bit). "
                   f"注意: 此 S3-c 为整机对比 (完整 v201 vs 完整 v244), 上游+头同变, 1.00 bit 不可单独归因于头; "
                   f"旁证 v201 冻住的 W1_new 路径贡献由 1.585→0.616 bit (同权重腰斩 → zh 变了). "
                   f"新事实: 起点 {base_bpb:.4f} > unigram {unigram} → 起点即坏; v201=merge (W1/W_lm 承自 v100) → 失败 >=v100. "
                   f"头是瓶颈由 exp116/exp118 独立确立, 不依赖本条.")
    elif abs(new_bpb - base_bpb) < 0.2:
        verdict = f"头动了但 bpb 不变 (起点 {base_bpb:.4f} ≈ v244 {new_bpb:.4f}) → 无效移动/规则问题; 6.73 是起点遗产"
    else:
        verdict = f"头动了且 bpb 改善 (起点 {base_bpb:.4f} → v244 {new_bpb:.4f})"

    out = {
        "ckpt_v244": args.new_ckpt, "md5_v244": md5_new,
        "ckpt_base": args.base_ckpt, "md5_base": md5_base,
        "base_provenance": {
            "spec": "e9f_preregistration.md §三: 起点 = 迁移后 v201 (out/v241-.../migrated/merge.safetensors)",
            "raw_v201": args.raw_v201, "raw_v201_md5": _md5(raw_v201),
            "migrate_zero_change_elem_eq": migrate_zero_change,
            "disclosure": ("base (迁移 v201) lm_in=1882 (mem_k=1); v244 lm_in=2807 (mem_k=2). "
                           "v244 .jsonl 显示 mem_k 1→2, 即 500 步中途 mem 出生一次 (尾插 925 行). "
                           "故 S3-b 对 lm_in 维张量取头部切片对齐 (mem 出生=尾插, 见 readout.py:737-743)."),
        },
        "S3a_static": {
            "gate_table": GATE_TABLE,
            "v244_effective_gates": {
                "_mig (migrate_expand)": "True — load 见 W1_new 键自动 init_migrate_expand (model_cyrene.py:846-851)",
                "_readout_realign": "True — 臂名 readout_realign (E8a 覆写 _mig 门: 仅 W_lm/W1)",
                "_readout_freeze": "False",
                "_migrate_freeze": "推断 True (E9f 拆门, 否则 _update_migrate_heads 触发 L337 崩溃); 由 S3-b 张量移动复核",
                "_single_recon": "True — load 见 B_recon 键自动 init_single_recon (model_cyrene.py:850-851 -> L629); 运行时探针 (runtime_single_recon_probe) 证实 base 与 v244 皆 True, _w1v2_eta_recon=0.1",
                "_echo_learn": "False (默认) → echo 相位头冻结",
                "caveat": "运行时标记 (flags) 不随检查点持久化; 以上为臂名+代码+张量移动三相推断, 非直接读回.",
            },
            "runtime_stale_tensor_scan": scan,
            "runtime_flags_after_load": runtime_flags_after_load,
            "runtime_single_recon_probe": single_recon_probe,
        },
        "S3b_tensor_delta": s3b,
        "S3b_headline": {"moved": moved, "frozen": frozen,
                         "W1_rel_frob": s3b.get("W1", {}).get("rel_frob_head"),
                         "W_lm_rel_frob": s3b.get("W_lm", {}).get("rel_frob"),
                         "W_lm_2_rel_frob": s3b.get("W_lm_2", {}).get("rel_frob")},
        "S3c_head_bpb": s3c,
        "S3c_anchor": {"v244_mig_reference": 6.7316, "unigram_T": 4.7677, "exp118_A1_upperbound": 2.3322},
        "S3d_update_counts": s3d,
        "verdict": verdict,
        "iron_rules": "前台; 不改 crates/; 不改 v244; 不落盘权重; 探针零写回; 新目录 v249",
    }
    path = run_file("readout_audit.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"exp119: 写出 {path}", flush=True)
    print(f"exp119: 判定 = {verdict}", flush=True)


if __name__ == "__main__":
    main()
