"""
同架构反传上界 (读出瓶颈分解 step 2).

冻结 v244 终态, 抽读出头的**真实输入** zh = rms_norm(cat([z4_lm, _bind_vec, _mem_out]))
(readout.py:78-79 同构), 用反传训三个外部对照头, 判: 架构/输入 是否充分.

判据 (A1 = 原样架构):
  ≤3.0     → 架构+输入充分 → 仅排除"架构/输入不行"; 瓶颈在**学习侧** (规则+更新次数+目标形式+门控), 非单一变量 → (a)
  ≈6.7     → 架构本身表达不了 → 架构问题, 重新立项
  3.0–6.0  → 部分表达 → 学习侧 + 架构各担一部分

探针是外部仪器: 冻结网络零写入 (只前馈), 探针全 fp16 (与模型同精度, 无高位浮点对照路).
不加生成掩码 (与 Hebbian 头 6.73 口径对齐); 评分复用 exp116 score_logits + cal 网格标定.
"""
import argparse
import json
import math
import time

import torch
import torch.nn as nn
from exp115_say import N_EVAL_WIN, GraphedPredict, build_gauge
from exp116_z4_probe import _md5, calib_s, infer, score_logits, z4lm_from_z4

from crates.cli.utils import resolve_path, run_file
from crates.kernels import cubic, rms_norm, saturate
from crates.model import CyreneModel, HybridPCNet

torch.set_grad_enabled(False)

ANCHORS = {
    "hebbian_head_mig_real_bpb": {"v": 6.7324, "src": "exp117 v247 head_decomp.json (W1+W1_new, s1)"},
    "hebbian_head_red_real_bpb": {"v": 7.3483, "src": "exp117 v247 head_decomp.json (仅 W1, 部署生成器, s1)"},
    "probe_z4_lm_mlp": {"v": 2.4144, "src": "exp116 v246 z4_lang_probe.json"},
    "probe_z4_lm_linear": {"v": 2.7044, "src": "exp116 v246 z4_lang_probe.json"},
    "unigram_T": {"v": 4.7677, "src": "exp117 v247"},
    "bigram_T": {"v": 3.7183, "src": "exp117 v247"},
}


def zh_from_net(net):
    """读出头的真实输入 (readout.py:78-79 同构): rms_norm(cat([z4_lm, _bind_vec, _mem_out]))."""
    z4_lm = z4lm_from_z4(net, net._z4)
    return rms_norm(torch.cat([z4_lm, net._bind_vec, net._mem_out], dim=-1))


def extract_zh(net, dev, wins, tag):
    xs, ys = [], []
    t0 = time.time()
    for i, w in enumerate(wins):
        x = w.unsqueeze(0).to(dev)
        net.forward(x)
        xs.append(zh_from_net(net)[0, :-1].to("cpu", torch.float16))
        ys.append(x[0, 1:].to("cpu"))
        if (i + 1) % 500 == 0:
            print(f"  [{tag}] extract {i + 1}/{len(wins)} ({time.time() - t0:.0f}s)", flush=True)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0).long()


class HeadA1(nn.Module):
    """原样架构: Linear(d_in,256) → saturate → rms_norm → cubic → Linear(256,256), ×1/√d_h."""

    def __init__(self, d_in, d_h=256):
        super().__init__()
        self.W1 = nn.Linear(d_in, d_h)
        self.W_lm = nn.Linear(d_h, 256)
        self.inv_h = 1.0 / math.sqrt(d_h)

    def forward(self, x):
        h = saturate(self.W1(x))
        h = rms_norm(h)
        h = cubic(h)
        return self.W_lm(h) * self.inv_h


def build_head(kind, d_in, hidden):
    if kind == "A1":
        return HeadA1(d_in, 256).half()
    if kind == "A2":
        return nn.Linear(d_in, 256).half()
    if kind == "A3":
        return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, 256)).half()
    raise ValueError(kind)


def train_head(model, Xtr, ytr, dev, epochs, lr, batch, seed):
    """反传训练 (features 常驻 host fp16, 逐 batch 上 GPU; 与 exp116 run_probe 同超参)."""
    torch.set_grad_enabled(True)
    torch.manual_seed(seed)
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4)  # eps: fp16 下默认 1e-8 舍入为 0 → 零梯度行 0/0=NaN
    lossf = nn.CrossEntropyLoss()
    n = Xtr.shape[0]
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = Xtr[idx].to(dev, torch.float16)
            yb = ytr[idx].to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * idx.numel()
        print(f"    epoch {ep + 1}/{epochs} ce={tot / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.set_grad_enabled(False)
    return model


def band(bpb):
    if bpb <= 3.0:
        return "a: 架构+输入充分 (仅排除架构/输入不行), 瓶颈在学习侧 (规则+更新次数+目标形式+门控), 非单一变量"
    if bpb >= 6.0:
        return "架构问题: 架构本身表达不了 (≈6.7)"
    return "部分表达: 学习侧 + 架构各担一部分 (3.0–6.0)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/v244-20261009-214217/e9f_bplus_readout_realign_500.safetensors")
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.85)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--max-train", type=int, default=0, help="train 窗数上限 (0=全量); 内存吃紧时降窗并标注")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dev = torch.device(args.device)
    ckpt = resolve_path(args.ckpt)
    md5 = _md5(ckpt)
    print(f"exp118: ckpt={ckpt}\n  md5={md5}", flush=True)

    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=256,
                      lm_freeze_w1=False)
    net = HybridPCNet.load(ckpt, cfg).to(dev)
    net.eval()
    dim_4 = net.active_size["l4"]

    gw = {"obj": None}

    def _wrap_graph():
        print("exp118: 前向罩 CUDA Graph", flush=True)
        gw["obj"] = GraphedPredict(net.forward_engine, net)
        net.forward_engine._predict = gw["obj"]

    def _unwrap_graph():
        if gw["obj"] is not None:
            net.forward_engine._predict = gw["obj"].raw
            gw["obj"].graphs.clear()
            gw["obj"] = None
            torch.cuda.empty_cache()

    if args.smoke:
        gauge, _, _ = build_gauge(args.data, dev)
        w = gauge["tail"][0].unsqueeze(0).to(dev)
        net.forward(w)
        z = zh_from_net(net)
        print(f"exp118 smoke: zh={tuple(z.shape)} d_in={z.shape[-1]}", flush=True)
        if not args.no_graph and dev.type == "cuda":
            _wrap_graph()
            net.forward(w)
            d = float((zh_from_net(net) - z).abs().max().item())
            print(f"exp118 smoke: graph vs eager max|dzh|={d:.3e}", flush=True)
        return

    if not args.no_graph and dev.type == "cuda":
        _wrap_graph()

    torch.manual_seed(args.seed)
    tg = time.time()
    gauge, nb_t, nb_r = build_gauge(args.data, dev)
    print(f"exp118: gauge tail={len(gauge['tail'])} trunc={len(gauge['trunc'])} ({time.time() - tg:.0f}s)",
          flush=True)

    test_wins = gauge["tail"][:N_EVAL_WIN]
    fit_pool = gauge["trunc"] + gauge["tail"][N_EVAL_WIN:]
    order = torch.randperm(len(fit_pool), generator=torch.Generator().manual_seed(args.seed)).tolist()
    fit_pool = [fit_pool[i] for i in order]
    n_tr = int(len(fit_pool) * args.train_frac)
    train_wins, cal_wins = fit_pool[:n_tr], fit_pool[n_tr:]
    capped = False
    if args.max_train and len(train_wins) > args.max_train:
        train_wins = train_wins[:args.max_train]
        capped = True
    print(f"exp118: split train={len(train_wins)} cal={len(cal_wins)} test={len(test_wins)}"
          f"{' [CAPPED, 口径与 exp116 split 不同]' if capped else ''}", flush=True)

    Xtr, ytr = extract_zh(net, dev, train_wins, "train")
    Xca, yca = extract_zh(net, dev, cal_wins, "cal")
    Xte, yte = extract_zh(net, dev, test_wins, "test")
    _unwrap_graph()
    d_in = Xtr.shape[-1]
    print(f"exp118: zh d_in={d_in} (dim_4={dim_4}) shapes tr={tuple(Xtr.shape)} "
          f"ca={tuple(Xca.shape)} te={tuple(Xte.shape)}", flush=True)

    results, verdicts = {}, {}
    for kind in ("A1", "A2", "A3"):
        model = build_head(kind, d_in, args.hidden)
        train_head(model, Xtr, ytr, dev, args.epochs, args.lr, args.batch, args.seed)
        l_te = infer(model, Xte, dev, args.batch)
        l_ca = infer(model, Xca, dev, args.batch)
        s_star, per_s = calib_s(l_ca, yca)
        te = score_logits(l_te, yte, scale=s_star)
        te_s1 = score_logits(l_te, yte, scale=1.0)
        results[kind] = {"arch": str(model), "s_star": s_star, "cal_bpb_per_s": per_s,
                         "test_at_s_star": te, "test_at_s1": te_s1}
        verdicts[kind] = band(te["bpb"])
        print(f"exp118: {kind} test bpb(s*)={te['bpb']:.4f}@s={s_star} bpb(s1)={te_s1['bpb']:.4f} "
              f"top1={te['top1']:.4f} top15={te['top15']:.4f} median={te['rank_median']:.0f}",
              flush=True)
        del model
        torch.cuda.empty_cache()

    a1 = results["A1"]["test_at_s_star"]
    print(f"exp118: 【主指标 A1 原样架构】test bpb={a1['bpb']:.4f} top1={a1['top1']:.4f} "
          f"→ {verdicts['A1']}", flush=True)

    out = {
        "ckpt": args.ckpt, "ckpt_md5": md5, "data": args.data, "seed": args.seed,
        "d_in": d_in, "dim_4": dim_4, "d_h": net.d_h, "mem_k": int(net._mem_m.shape[0]),
        "zh_basis": "readout.py:78-79 同构: rms_norm(cat([z4_lm, _bind_vec, _mem_out]))",
        "split": {"train_windows": len(train_wins), "cal_windows": len(cal_wins),
                  "test_windows": len(test_wins), "capped": capped,
                  "test_source": f"gauge.tail[0:{N_EVAL_WIN}] (exp115/116 eval 口径)",
                  "note": "train 被 --max-train 截断, 与 exp116 split 不逐窗一致" if capped else
                          "与 exp116 split 逐窗一致 (seed 0)"},
        "head_cfg": {"A1": "Linear(d_in,256)-saturate-rms_norm-cubic-Linear(256,256), ×1/√256 (原样架构)",
                     "A2": "Linear(d_in,256) 纯线性",
                     "A3": f"Linear(d_in,{args.hidden})-GELU-Linear({args.hidden},256) 非线性上界",
                     "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
                     "mask_print": False,
                     "precision": "全链路 fp16 (features fp16 常驻 host, 逐 batch 上 GPU; 头内部 fp16)",
                     "note_features_host_resident": "zh 维度 2807 → train 特征 ~4.7GB, 不上 GPU (4GB 显存)"},
        "anchors": ANCHORS,
        "anchor_note": ("A1 的逐位同构锚点是 red (7.3483, 单路径 W1->sat->rms->cubic->W_lm), 差 5.02 bit; "
                        "mig (6.7324) 是 A1 再加平行 W1_new 支, 差 4.40 bit → 主锚点=red. "
                        "另: A1 用 zh (2807, 含 bind+mem), 比 exp116 的 z4_lm (925) 输入更全, "
                        "故 2.33 vs 2.41 非'同输入下反传胜探针'."),
        "results": results,
        "verdict": verdicts,
    }
    path = run_file("head_upperbound.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"exp118: 写出 {path}", flush=True)


if __name__ == "__main__":
    main()
