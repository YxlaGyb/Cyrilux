"""tmp: bpb 对照测量 (终态 checkpoint) — 区分"从未学会"vs"漂移遗忘".
(1) 训练窗口 bpb: 训练行前 256B (感知相位可见) 的同口径读出;
(2) 单字频率基线: 训练语料字节频率表在 held-out 目标上的 bpb (静态参照).
用完即删."""
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
import exp114_say as E
from model import CyreneModel
from dataset import DualChannelDataset

S_MAX = 256
DATA = "dataset/pretrain_t2t_mini.jsonl"
dev = torch.device("cuda")
cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=256, lm_freeze_w1=True)

cnt = Counter()
train_wins = []
n_train = 256
with open(DATA, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 1270000:
            break
        text = DualChannelDataset._extract_with_roles(json.loads(line))[0]
        b = text.encode("utf-8")[:S_MAX]
        if len(b) < S_MAX:
            continue
        cnt.update(b)
        if len(train_wins) < n_train:
            train_wins.append(torch.tensor(list(b), dtype=torch.long))
total = sum(cnt.values())
p_uni = {k: v / total for k, v in cnt.items()}
ent = -sum(p * math.log2(p) for p in p_uni.values())
print(f"训练前 256B 字节频率表: {len(cnt)}/256 字节, 熵={ent:.3f} bits", flush=True)

gauge, _, _ = E.build_gauge(DATA, dev)


def unigram_bpb(wins):
    nll, n = 0.0, 0
    for w in wins:
        for t in w[1:].tolist():
            nll -= math.log2(p_uni.get(t, 1e-9))
            n += 1
    return nll / n


for s in ("tail", "trunc"):
    print(f"  单字基线 [{s}]: {unigram_bpb(gauge[s]):.4f} bpb", flush=True)

# 对照: 文件尾部附近行 (与 trunc 同源行) 的前 256B — 区分
# "文件头尾内容分布差异" vs "行首 256B 窗口结构"
tail_raw, before_raw = E._tail_lines_bin(DATA, 238, 4000, None)
late_wins = []
for l in before_raw[:300]:
    text = DualChannelDataset._extract_with_roles(json.loads(l))[0]
    b = text.encode("utf-8")[:S_MAX]
    if len(b) < S_MAX:
        continue
    late_wins.append(torch.tensor(list(b), dtype=torch.long))
print(f"尾部区段行首窗口 n={len(late_wins)}", flush=True)

r = E.eval_bpb("out/exp114_say.pt", cfg,
               {"train": train_wins, "late_head": late_wins}, dev)
print(f"训练窗口 bpb (文件头行首 256B): {r['train']:.4f}", flush=True)
print(f"尾部区段行首 bpb (同 trunc 源行, 行首 256B): {r['late_head']:.4f}", flush=True)
