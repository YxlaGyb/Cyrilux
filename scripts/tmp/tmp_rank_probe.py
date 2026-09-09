"""tmp: 真字节排名探针 — 区分读出"无信号" vs "反信号".
对 held-out tail 窗口测: top-1 命中率 / 真字节平均排名 / 频率基线排名.
用完即删."""
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

dev = torch.device("cuda")
cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=256, lm_freeze_w1=True)
CKPT = sys.argv[1] if len(sys.argv) > 1 else "out/exp114_say.pt"
net = E.DensePCNet.load(CKPT, cfg).to(dev)
print(f"checkpoint: {CKPT}", flush=True)

gauge, _, _ = E.build_gauge("dataset/pretrain_t2t_mini.jsonl", dev)

# 中文侧对照: 文件头行首 256B (训练分布主导语言)
import json
from dataset import DualChannelDataset
zh_wins = []
with open("dataset/pretrain_t2t_mini.jsonl", encoding="utf-8") as f:
    for line in f:
        if len(zh_wins) >= 200:
            break
        b = DualChannelDataset._extract_with_roles(json.loads(line))[0].encode("utf-8")[:256]
        if len(b) == 256:
            zh_wins.append(torch.tensor(list(b), dtype=torch.long))

for tag, wins in (("en_tail(英文留出)", gauge["tail"][:200]),
                  ("zh_head(中文行首)", zh_wins)):
    top1 = 0
    top15 = 0
    ranks = []
    n = 0
    SCALES = (4, 8, 16, 30, 60)
    nll_s = [0.0 for _ in SCALES]
    for w in wins:
        x = w.unsqueeze(0).to(dev)
        net.forward(x)
        z4 = net._z4
        a4 = net.active_size["l4"]
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
        logits = (h @ net.W_lm + net.bias_lm) * (1.0 / math.sqrt(net.d_h))
        lc = ((logits - logits.mean(dim=-1, keepdim=True))
              / (logits.std(dim=-1, keepdim=True) + 1e-4))
        lc = lc / lc.abs().max(dim=-1, keepdim=True).values
        # 标定扫描: 同一排序, 不同置信度缩放
        tgt = x[0, 1:]
        pos = torch.arange(255, device=dev)
        for j, s in enumerate(SCALES):
            lsm = torch.log_softmax(lc[0] * s, dim=-1).float()
            nll_s[j] += (-lsm[pos, tgt] / math.log(2)).sum().item()
        logits = lc * 60.0
        order = logits[0].argsort(dim=-1, descending=True)
        rank_of = order.argsort(dim=-1)
        r = rank_of[pos, tgt]
        ranks.append(r.float())
        top1 += (r == 0).sum().item()
        top15 += (r < 15).sum().item()
        n += 255

    ranks = torch.cat(ranks)
    print(f"[{tag}] n={n}")
    print(f"  top-1 命中率: {top1/n:.4f} | top-15 命中率: {top15/n:.4f} (均匀 0.0039/0.0586)")
    print(f"  真字节名次: mean={ranks.mean():.1f} median={ranks.median():.0f} "
          f"p25={ranks.quantile(0.25):.0f} p75={ranks.quantile(0.75):.0f} (均匀期望 127.5)")
    print(f"  标定扫描 bpb (同一排序): " +
          " ".join(f"s={s}:{nll_s[j]/n:.2f}" for j, s in enumerate(SCALES)))
