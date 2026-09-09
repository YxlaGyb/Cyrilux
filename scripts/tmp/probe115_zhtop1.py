"""115 跑后 zh 臂 top1 崩塌诊断 (0.118 -> 0.0057, top15 却 0.676):
对 zh eval 200 窗统计 s=60 argmax 字节分布 — 若 argmax 集中在少数错误
字节, top1 崩塌是"恒定错字第一"型; 若分散, 是分布弥散型.
对照跑前 (exp114_say.pt) 同口径.
用法: .venv/Scripts/python.exe scripts/probe115_zhtop1.py
输出: out/probe115_zhtop1.json
"""
import json
import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter

import torch

torch.set_grad_enabled(False)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, "scripts")
import exp115_say as E
from model import CyreneModel, DensePCNet
from dataset import DualChannelDataset

DEV = torch.device("cuda")


def zh_windows():
    wins = []
    with open("dataset/pretrain_t2t_mini.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 200:
                break
            b = DualChannelDataset._extract_with_roles(json.loads(line))[0]
            b = b.encode("utf-8")[:256]
            if len(b) == 256:
                wins.append(torch.tensor(list(b), dtype=torch.long))
    return wins


def en_windows():
    gauge, _, _ = E.build_gauge("dataset/pretrain_t2t_mini.jsonl", DEV)
    return gauge["tail"][:200]


def argmax_stats(ckpt, wins, arm):
    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=256,
                        lm_freeze_w1=True)
    net = DensePCNet.load(ckpt, cfg).to(DEV)
    pos = torch.arange(255, device=DEV)
    mask = torch.zeros(256, dtype=torch.float16, device=DEV)
    mask[32:] = 1.0  # readout.py L114-116 可打印掩码同款
    am = Counter()
    tgt_of_am = Counter()  # argmax 字节 == 真字节?
    tgt_cnt = Counter()
    top3_cnt = Counter()
    # 掩码口径 bpb (s 网格) + 名次 — 训练部署口径 vs 测量口径差
    # 控制字节目标 (zh 文本含 \n 0.24%) 在掩码下结构性零概率 → 两种口径
    # 都剔除这些位置, 得到纯净的 0x00 尖峰代价测量
    S_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0, 12.0)
    nll = {s: 0.0 for s in S_GRID}
    nll_um = {s: 0.0 for s in S_GRID}
    top1m = top15m = 0
    n_ctrl_tgt = 0
    ranks = []
    for w in wins:
        x = w.unsqueeze(0).to(DEV)
        lc = E._readout_lc(net, x, DEV)
        sc = (lc[0] * 60.0)[pos]  # [255, 256]
        tgt = x[0, 1:]
        for j in range(255):
            ab = int(sc[j].argmax().item())
            tb = int(tgt[j].item())
            am[ab] += 1
            tgt_cnt[tb] += 1
            if ab == tb:
                tgt_of_am[ab] += 1
            top3 = torch.topk(sc[j], 3).indices.tolist()
            for k in top3:
                top3_cnt[int(k)] += 1
        # 掩码口径: 与部署一致 (0-31 控制字节不可见); lc 尺度 (±1),
        # masked=-60 恒低于可打印区 — 名次与 s 网格与 ladder 同尺度
        lcm = lc[0] * mask + (1.0 - mask) * (-60.0)
        ok = tgt >= 32  # 控制字节目标位置剔除 (两口径同子集)
        n_ctrl_tgt += int((~ok).sum().item())
        for s in S_GRID:
            lsm = torch.log_softmax(lcm.float() * s, dim=-1)
            nll[s] += (-lsm[pos, tgt] / math.log(2))[ok].sum().item()
            lsm_u = torch.log_softmax(lc[0].float() * s, dim=-1)
            nll_um[s] += (-lsm_u[pos, tgt] / math.log(2))[ok].sum().item()
        order = lcm.argsort(dim=-1, descending=True)
        r = order.argsort(dim=-1)[pos, tgt]
        ranks.append(r.float())
        top1m += (r == 0).sum().item()
        top15m += (r < 15).sum().item()
    n = sum(am.values())
    nr = len(ranks) * 255
    n_ok = nr - n_ctrl_tgt
    ranks = torch.cat(ranks)
    best_s = min(S_GRID, key=lambda s: nll[s])
    best_s_um = min(S_GRID, key=lambda s: nll_um[s])
    out = {
        "ckpt": ckpt,
        "n_positions": n,
        "argmax_top5": [{"byte": f"0x{b:02X}", "count": c, "frac": c / n,
                         "hit_frac": tgt_of_am.get(b, 0) / c}
                        for b, c in am.most_common(5)],
        "argmax_n_distinct": len(am),
        "argmax_top5_frac": sum(c for _, c in am.most_common(5)) / n,
        "target_top5": [{"byte": f"0x{b:02X}", "count": c, "frac": c / n}
                        for b, c in tgt_cnt.most_common(5)],
        "model_top3_freq_top5": [{"byte": f"0x{b:02X}",
                                  "frac": c / (3 * n)}
                                 for b, c in top3_cnt.most_common(5)],
        "masked_caliber": {
            "note": "控制字节目标位置 (\\n 0.24%) 已从两口径剔除",
            "n_ctrl_tgt": n_ctrl_tgt, "n_scored": n_ok,
            "bpb_per_s": {str(s): nll[s] / n_ok for s in S_GRID},
            "best_s": best_s,
            "bpb_at_best_s": nll[best_s] / n_ok,
            "unmasked_bpb_per_s_same_subset": {str(s): nll_um[s] / n_ok
                                               for s in S_GRID},
            "unmasked_best_s": best_s_um,
            "unmasked_bpb_at_best_s_same_subset": nll_um[best_s_um] / n_ok,
            "top1": top1m / nr, "top15": top15m / nr,
            "rank_median": ranks.median().item()},
    }
    del net
    torch.cuda.empty_cache()
    return out


def main():
    arms = {"zh": zh_windows(), "en": en_windows()}
    res = {}
    for arm, wins in arms.items():
        res[arm] = {"after_115": argmax_stats("out/exp115_say.pt", wins, arm),
                    "before_114": argmax_stats("out/exp114_say.pt", wins, arm)}
        for k, v in res[arm].items():
            print(f"[{arm} {k}] argmax top5: " + " ".join(
                f"{d['byte']}:{d['frac']:.3f}(hit={d['hit_frac']:.2f})"
                for d in v["argmax_top5"]) +
                f" | distinct={v['argmax_n_distinct']}")
            print(f"     target top5: " + " ".join(
                f"{d['byte']}:{d['frac']:.3f}" for d in v["target_top5"]))
            m = v["masked_caliber"]
            print(f"     掩码口径 (ctrl 剔除 n={m['n_scored']}): "
                  f"masked@{m['best_s']}={m['bpb_at_best_s']:.3f} vs "
                  f"unmasked@{m['unmasked_best_s']}="
                  f"{m['unmasked_bpb_at_best_s_same_subset']:.3f} | "
                  f"top1={m['top1']:.4f} top15={m['top15']:.4f} "
                  f"median={m['rank_median']:.0f}")
    with open("out/probe115_zhtop1.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("saved -> out/probe115_zhtop1.json")


if __name__ == "__main__":
    main()
