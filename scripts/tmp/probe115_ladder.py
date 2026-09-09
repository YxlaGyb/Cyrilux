"""115 阶段三: 同臂 n-gram 梯级判决 (零梯度零学习, 纯测量).

零假设升级: 114 附录"标定 bpb 优于单字表"是与错配基线比较的产物.
正确对照 = 同分布拟合的 unigram/bigram, 同一些字节位置上评分,
同款温度 s 扫描自由度.

臂划分 (沿用 tmp_rank_probe, 同一评分口径逐行复刻):
- en 臂 eval: held-out tail 窗 [0:200] (模型未见); 标定切片 tail 窗 [200:600]
  (模型未见, 与 eval 严格分离)
- zh 臂 eval: 语料头 200 行首 256B (训练分布); 标定切片 head 行 201-1200
  的满 256B 行 (与 eval 行号严格分离)
- 评分位置: 每窗 t=1..255 (模型/基线完全同位; bigram 上下文恒在窗内,
  无需首字节回退)

基线族: 标定切片 80% 拟合 + 20% 内部验证选 (α, s);
- unigram: p=(c+0.5)/(N+128), 温度 s: p_s ∝ p^(1/s)
- bigram: p(b|a)=(C(a,b)+α·p_uni(b))/(C(a)+α), α 网格 {0.1,1,4,16,64}
模型: s 由标定切片 bpb 网格决定 (阶段四硬约束: 标量, 无梯度, 无裁判,
无 R), eval 上报最优 s 的 bpb; 同时报 s=4 连续性口径与 top-1/top-15/
中位名次 (tmp_rank_probe 复现).

用法: .venv/Scripts/python.exe scripts/probe115_ladder.py [--ckpt ...] [--out ...]
输出: out/probe115_ladder.json
"""
import argparse
import json
import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

torch.set_grad_enabled(False)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, "scripts")
import exp114_say as E
from model import CyreneModel
from model.modulation import rms_norm
from dataset import DualChannelDataset

S_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0, 12.0)
ALPHA_GRID = (0.1, 1.0, 4.0, 16.0, 64.0)
LOG2 = math.log(2)


def model_score(net, wins, dev):
    """tmp_rank_probe 逐行同口径: 前向 + 读出, s 网格 nll + 名次统计."""
    a4 = net.active_size["l4"]
    inv_h = 1.0 / math.sqrt(net.d_h)
    nll = {s: 0.0 for s in S_GRID}
    n, top1, top15 = 0, 0, 0
    ranks = []
    for w in wins:
        x = w.unsqueeze(0).to(dev)
        net.forward(x)
        z4 = net._z4
        z4_n_ = z4 / (z4.norm(dim=-1, keepdim=True) + 1e-3)
        pred_delta = (z4_n_ @ net.W_diff[:a4, :a4].T
                      + net.b_diff[:a4].unsqueeze(0).unsqueeze(0))
        z4r = z4 + pred_delta
        z4_lm = z4r / (1.0 + z4r.abs())
        z4_lm = rms_norm(z4_lm)
        z4_lm = z4_lm * (1.0 - 0.5 * z4_lm.pow(2))
        z4_lm = z4_lm / (1.0 + z4_lm.abs())
        zh = torch.cat([z4_lm, net._bind_vec, net._mem_out], dim=-1)
        zh = rms_norm(zh)
        h = zh @ net.W1
        h = h / (1.0 + h.abs())
        h = rms_norm(h)
        h = h * (1.0 - 0.5 * h.pow(2))
        logits = (h @ net.W_lm + net.bias_lm) * inv_h
        lc = ((logits - logits.mean(dim=-1, keepdim=True))
              / (logits.std(dim=-1, keepdim=True) + 1e-4))
        lc = lc / lc.abs().max(dim=-1, keepdim=True).values
        tgt = x[0, 1:]
        pos = torch.arange(255, device=dev)
        for s in S_GRID:
            lsm = torch.log_softmax(lc[0] * s, dim=-1).float()
            nll[s] += (-lsm[pos, tgt] / LOG2).sum().item()
        order = (lc[0] * 60.0).argsort(dim=-1, descending=True)
        rank_of = order.argsort(dim=-1)
        r = rank_of[pos, tgt]
        ranks.append(r.float())
        top1 += (r == 0).sum().item()
        top15 += (r < 15).sum().item()
        n += 255
    ranks = torch.cat(ranks)
    return {"bpb_per_s": {str(s): nll[s] / n for s in S_GRID},
            "top1": top1 / n, "top15": top15 / n,
            "rank_median": ranks.median().item(),
            "rank_mean": ranks.mean().item(), "n": n}


def byte_composition(wins):
    """字节语言构成 (阶段四.4): CJK 3 字节序列 / ASCII 字母 / 其他."""
    cjk = letters = other = 0
    for w in wins:
        arr = w.tolist()
        i = 0
        while i < len(arr):
            b = arr[i]
            if 0xE4 <= b <= 0xE9 and i + 2 < len(arr) \
                    and 0x80 <= arr[i + 1] <= 0xBF and 0x80 <= arr[i + 2] <= 0xBF:
                cjk += 3
                i += 3
            elif (0x41 <= b <= 0x5A) or (0x61 <= b <= 0x7A):
                letters += 1
                i += 1
            else:
                other += 1
                i += 1
    tot = cjk + letters + other
    return {"n_bytes": tot, "cjk_frac": cjk / tot, "ascii_letter_frac": letters / tot,
            "other_frac": other / tot}


def counts(wins):
    """(unigram 计数, bigram 计数) — 评分同位: t=1..255, 上下文 t-1."""
    uni = torch.zeros(256, dtype=torch.long)
    bi = torch.zeros(256 * 256, dtype=torch.long)
    for w in wins:
        t = w if torch.is_tensor(w) else torch.tensor(w.tolist())
        arr = t.long()
        uni += torch.bincount(arr, minlength=256)
        pairs = arr[:-1] * 256 + arr[1:]
        bi += torch.bincount(pairs, minlength=256 * 256)
    return uni, bi.view(256, 256)


def fit_models(uni, bi):
    p_uni = (uni.float() + 0.5) / (uni.sum().item() + 128.0)
    log_uni = p_uni.clamp_min(1e-12).log()
    row = bi.sum(dim=1).float()  # 上下文计数 [256]
    return p_uni, log_uni, row


def bpb_unigram(wins, log_uni, s):
    n, nll = 0, 0.0
    lsm = torch.log_softmax(log_uni / s, dim=-1)
    for w in wins:
        arr = (w if torch.is_tensor(w) else torch.tensor(w.tolist())).long()
        nll += (-lsm[arr[1:]] / LOG2).sum().item()
        n += arr.numel() - 1
    return nll / n, n


def bpb_bigram(wins, bi, row, log_uni, alpha, s):
    log_bi = ((bi.float() + alpha * torch.exp(log_uni))
              / (row.unsqueeze(1) + alpha)).clamp_min(1e-12).log()
    lsm_row = torch.log_softmax(log_bi / s, dim=-1)
    n, nll = 0, 0.0
    for w in wins:
        arr = (w if torch.is_tensor(w) else torch.tensor(w.tolist())).long()
        ctx, tgt = arr[:-1], arr[1:]
        nll += (-lsm_row[ctx, tgt] / LOG2).sum().item()
        n += tgt.numel()
    return nll / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/exp114_say.pt")
    ap.add_argument("--out", default="out/probe115_ladder.json")
    args = ap.parse_args()
    dev = torch.device("cuda")
    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=256,
                        lm_freeze_w1=True)
    net = E.DensePCNet.load(args.ckpt, cfg).to(dev)
    print(f"checkpoint: {args.ckpt}", flush=True)

    gauge, _, _ = E.build_gauge("dataset/pretrain_t2t_mini.jsonl", dev)
    en_eval = gauge["tail"][:200]
    en_cal = gauge["tail"][200:1000]

    zh_wins = []
    with open("dataset/pretrain_t2t_mini.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 1200:
                break
            b = DualChannelDataset._extract_with_roles(json.loads(line))[0]
            b = b.encode("utf-8")[:256]
            if len(b) == 256:
                zh_wins.append(torch.tensor(list(b), dtype=torch.long))
    zh_eval = zh_wins[:200]
    zh_cal = zh_wins[200:1200]

    out = {"ckpt": args.ckpt,
           "arms": {
               "en": {"eval": "tail 窗 [0:200] (模型未见)",
                      "cal": "tail 窗 [200:1000] (模型未见, 与 eval 分离)",
                      "n_eval_win": len(en_eval), "n_cal_win": len(en_cal)},
               "zh": {"eval": "head 行 1-200 首 256B (训练分布)",
                      "cal": "head 行 201-1200 满 256B 行 (行号与 eval 分离)",
                      "n_eval_win": len(zh_eval), "n_cal_win": len(zh_cal)}},
           "composition": {
               "tail_all_1121win": byte_composition(gauge["tail"]),
               "trunc_2975win": byte_composition(gauge["trunc"][:1000]),
               "zh_head_1200lines": byte_composition(zh_wins),
               "en_eval_win": byte_composition(en_eval),
               "en_cal_win": byte_composition(en_cal),
               "zh_eval_win": byte_composition(zh_eval),
               "zh_cal_win": byte_composition(zh_cal)}}
    print(f"臂: en eval={len(en_eval)} cal={len(en_cal)} | "
          f"zh eval={len(zh_eval)} cal={len(zh_cal)}", flush=True)
    for k, v in out["composition"].items():
        print(f"  [comp] {k}: zh={v['cjk_frac']:.3f} en={v['ascii_letter_frac']:.3f} "
              f"other={v['other_frac']:.3f} (n={v['n_bytes']})", flush=True)

    # ── 模型: 先 eval (tmp_rank_probe 同序: en→zh, 新鲜记忆态, 数值可直接
    # 对照 114 基线), 后标定切片 (s 选择用) ──
    ev_scores = {}
    for arm, eval_w in (("en", en_eval), ("zh", zh_eval)):
        ev_scores[arm] = model_score(net, eval_w, dev)
    cal_scores = {}
    for arm, cal_w in (("en", en_cal), ("zh", zh_cal)):
        cal_scores[arm] = model_score(net, cal_w, dev)
    for arm in ("en", "zh"):
        cal, ev = cal_scores[arm], ev_scores[arm]
        best_s = min(S_GRID, key=lambda s: cal["bpb_per_s"][str(s)])
        out["model_" + arm] = {
            "cal_bpb_per_s": cal["bpb_per_s"],
            "eval_bpb_per_s": ev["bpb_per_s"],
            "best_s_on_cal": best_s,
            "eval_bpb_at_best_s": ev["bpb_per_s"][str(best_s)],
            "eval_bpb_at_s4": ev["bpb_per_s"]["4.0"],
            "top1": ev["top1"], "top15": ev["top15"],
            "rank_median": ev["rank_median"], "rank_mean": ev["rank_mean"],
            "n_scored": ev["n"]}
        print(f"[model {arm}] best_s={best_s} "
              f"eval_bpb={ev['bpb_per_s'][str(best_s)]:.3f} "
              f"(s=4: {ev['bpb_per_s']['4.0']:.3f}) top1={ev['top1']:.4f} "
              f"top15={ev['top15']:.4f} median={ev['rank_median']:.0f}",
              flush=True)

    # ── 基线: 标定切片 80/20 拆分, (α, s) 在内部验证选 ──
    for arm, eval_w, cal_w in (("en", en_eval, en_cal), ("zh", zh_eval, zh_cal)):
        n_fit = int(len(cal_w) * 0.8)
        fit_w, val_w = cal_w[:n_fit], cal_w[n_fit:]
        uni, bi = counts(fit_w)
        p_uni, log_uni, row = fit_models(uni, bi)

        best_u = None
        for s in S_GRID:
            b, n = bpb_unigram(val_w, log_uni, s)
            if best_u is None or b < best_u[1]:
                best_u = (s, b)
        u_eval, _ = bpb_unigram(eval_w, log_uni, best_u[0])

        best_b = None
        for a in ALPHA_GRID:
            for s in S_GRID:
                b, n = bpb_bigram(val_w, bi, row, log_uni, a, s)
                if best_b is None or b < best_b[2]:
                    best_b = (a, s, b)
        b_eval, _ = bpb_bigram(eval_w, bi, row, log_uni, best_b[0], best_b[1])

        top_p, top_i = p_uni.topk(15)
        struct = {
            "unigram_top1_byte": int(top_i[0].item()),
            "unigram_top1_byte_hex": f"0x{int(top_i[0].item()):02X}",
            "unigram_top1_p": float(top_p[0].item()),
            "unigram_top15_coverage": float(top_p.sum().item()),
            "unigram_n_fit_bytes": int(uni.sum().item()),
            "bigram_unseen_pair_frac_fit": float(
                (bi == 0).float().mean().item()),
        }
        out["baseline_" + arm] = {
            "unigram": {"best_s": best_u[0], "val_bpb": best_u[1],
                        "eval_bpb": u_eval},
            "bigram": {"alpha": best_b[0], "best_s": best_b[1],
                       "val_bpb": best_b[2], "eval_bpb": b_eval},
            "structural": struct,
            "n_fit_win": len(fit_w), "n_val_win": len(val_w),
            "fit_val_boundary": f"标定切片前 {n_fit}/{len(cal_w)} 窗拟合"}
        print(f"[base {arm}] unigram(s={best_u[0]}) eval_bpb={u_eval:.3f} | "
              f"bigram(α={best_b[0]},s={best_b[1]}) eval_bpb={b_eval:.3f} | "
              f"top1字节=0x{int(top_i[0].item()):02X} p={top_p[0].item():.4f} "
              f"top15覆盖={top_p.sum().item():.4f}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
