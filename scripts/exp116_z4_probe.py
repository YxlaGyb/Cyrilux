"""
Z4 语言性探针.

冻结 v244 终态, 从 net._z4 抽特征, 训线性/MLP 探针预测下一字节, 与 unigram 基线对比.
判定: 探针 ≫ unigram → 读出瓶颈 (读法 A); 探针 ≈ unigram → 表示瓶颈 (读法 B).

探针是外部测量仪器: 冻结网络零写入 (只前馈), 探针分类器独立于模型学习链路;
探针内部 fp32 属报告口径 (同 exp115_say._score_windows 的 tensor-guard: report).
"""
import argparse
import hashlib
import json
import math
import time

import torch
import torch.nn as nn
from exp115_say import N_EVAL_WIN, build_gauge

from crates.cli.utils import resolve_path, run_file
from crates.model import CyreneModel, DensePCNet

torch.set_grad_enabled(False)

S_GRID = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 60.0)


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def extract_z4(net, dev, wins, tag):
    xs, ys = [], []
    t0 = time.time()
    for i, w in enumerate(wins):
        x = w.unsqueeze(0).to(dev)
        net.forward(x)
        xs.append(net._z4[0, :-1].to("cpu", torch.float16))
        ys.append(x[0, 1:].to("cpu"))
        if (i + 1) % 200 == 0:
            print(f"  [{tag}] extract {i + 1}/{len(wins)} ({time.time() - t0:.0f}s)", flush=True)
    X = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0).long()
    return X, y


def score_logits(logits, y, scale=1.0):
    lg = logits * scale
    lsm = torch.log_softmax(lg, dim=-1)
    bpb = (-lsm.gather(1, y.unsqueeze(1)).squeeze(1)).mean().item() / math.log(2)
    order = lg.argsort(dim=-1, descending=True)
    r = order.argsort(dim=-1).gather(1, y.unsqueeze(1)).squeeze(1)
    return {"bpb": bpb, "top1": (r == 0).float().mean().item(),
            "top15": (r < 15).float().mean().item(),
            "rank_median": r.median().item(), "rank_mean": r.float().mean().item(),
            "n": int(y.numel())}


def run_probe(model, Xtr, ytr, dev, epochs, lr, bs, seed):
    torch.set_grad_enabled(True)
    torch.manual_seed(seed)
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = Xtr.shape[0]
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr[idx].to(dev, torch.float32)
            yb = ytr[idx].to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * idx.numel()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    epoch {ep + 1}/{epochs} ce={tot / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.set_grad_enabled(False)
    return model


def infer(model, X, dev, bs):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            outs.append(model(X[i:i + bs].to(dev, torch.float32)).to("cpu"))
    return torch.cat(outs)


def calib_s(logits, y):
    best_s, best_bpb = 1.0, None
    per_s = {}
    for s in S_GRID:
        bpb = score_logits(logits, y, scale=s)["bpb"]
        per_s[str(s)] = bpb
        if best_bpb is None or bpb < best_bpb:
            best_bpb, best_s = bpb, s
    return best_s, per_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/v244-20261009-214217/e9f_bplus_readout_realign_500.safetensors")
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.85)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--graph", action="store_true",
                    help="单窗前向罩 CUDA Graph (exp115 GraphedPredict, 与 eval 口径同款)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dev = torch.device(args.device)
    ckpt = resolve_path(args.ckpt)
    md5 = _md5(ckpt)
    print(f"exp116: ckpt={ckpt}\n  md5={md5}", flush=True)

    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=256,
                      lm_freeze_w1=False, lm_presyn_whiten=False)
    net = DensePCNet.load(ckpt, cfg).to(dev)
    net.eval()
    dim_4 = net.active_size["l4"]
    print(f"exp116: dim_4={dim_4} d_h={net.d_h} mem_k={net._mem_m.shape[0]} "
          f"active={net.active_size}", flush=True)

    def _wrap_graph():
        from exp115_say import GraphedPredict
        print("exp116: 前向罩 CUDA Graph (单窗形状 (1,256))", flush=True)
        net.forward_engine._predict = GraphedPredict(net.forward_engine, net)

    if args.smoke:
        gauge, nb_t, nb_r = build_gauge(args.data, dev)
        wins = gauge["tail"][:8]
        net.forward(wins[0].unsqueeze(0).to(dev))
        z_eager = net._z4.clone()
        if args.graph and dev.type == "cuda":
            _wrap_graph()
            net.forward(wins[0].unsqueeze(0).to(dev))
            d = float((net._z4 - z_eager).abs().max().item())
            print(f"exp116 smoke: graph vs eager max|dZ4|={d:.3e}", flush=True)
        t0 = time.time()
        extract_z4(net, dev, wins, "smoke")
        dt = (time.time() - t0) / len(wins)
        print(f"exp116 smoke: {dt * 1000:.1f} ms/forward → "
              f"4096 窗约 {dt * 4096:.0f}s", flush=True)
        print(f"exp116 smoke: tail={len(gauge['tail'])} 窗 trunc={len(gauge['trunc'])} 窗 "
              f"({nb_t}B/{nb_r}B)", flush=True)
        return

    if args.graph and dev.type == "cuda":
        _wrap_graph()

    torch.manual_seed(args.seed)
    tg = time.time()
    gauge, nb_t, nb_r = build_gauge(args.data, dev)
    print(f"exp116: gauge tail={len(gauge['tail'])} 窗 ({nb_t}B) "
          f"trunc={len(gauge['trunc'])} 窗 ({nb_r}B) ({time.time() - tg:.0f}s)", flush=True)

    test_wins = gauge["tail"][:N_EVAL_WIN]
    fit_pool = gauge["trunc"] + gauge["tail"][N_EVAL_WIN:]
    order = torch.randperm(len(fit_pool), generator=torch.Generator().manual_seed(args.seed)).tolist()
    fit_pool = [fit_pool[i] for i in order]
    n_tr = int(len(fit_pool) * args.train_frac)
    train_wins, cal_wins = fit_pool[:n_tr], fit_pool[n_tr:]
    print(f"exp116: split train={len(train_wins)} cal={len(cal_wins)} "
          f"test={len(test_wins)} (test = exp115 eval 口径 tail[0:{N_EVAL_WIN}], 全程不参与拟合)",
          flush=True)

    Xtr, ytr = extract_z4(net, dev, train_wins, "train")
    Xca, yca = extract_z4(net, dev, cal_wins, "cal")
    Xte, yte = extract_z4(net, dev, test_wins, "test")
    print(f"exp116: features train={tuple(Xtr.shape)} cal={tuple(Xca.shape)} "
          f"test={tuple(Xte.shape)}", flush=True)

    cnt = torch.bincount(ytr, minlength=256).double()
    p = cnt / cnt.sum()
    p = p.clamp_min(1.0 / (2.0 * cnt.sum())).float()
    uni_logits = torch.log(p)
    uni_te = score_logits(uni_logits.unsqueeze(0).expand(yte.numel(), -1), yte)
    uni_ca = score_logits(uni_logits.unsqueeze(0).expand(yca.numel(), -1), yca)
    print(f"exp116: unigram  test bpb={uni_te['bpb']:.4f} top1={uni_te['top1']:.4f} "
          f"top15={uni_te['top15']:.4f} median={uni_te['rank_median']:.0f}", flush=True)

    lin = nn.Linear(dim_4, 256)
    run_probe(lin, Xtr, ytr, dev, args.epochs, args.lr, args.batch, args.seed)
    lin_logits_te = infer(lin, Xte, dev, args.batch)
    lin_logits_ca = infer(lin, Xca, dev, args.batch)
    lin_s, lin_per_s = calib_s(lin_logits_ca, yca)
    lin_te = score_logits(lin_logits_te, yte, scale=lin_s)
    lin_te_raw = score_logits(lin_logits_te, yte, scale=1.0)
    print(f"exp116: linear   test bpb(s*)={lin_te['bpb']:.4f}@s={lin_s} "
          f"bpb(s1)={lin_te_raw['bpb']:.4f} top1={lin_te['top1']:.4f} "
          f"top15={lin_te['top15']:.4f} median={lin_te['rank_median']:.0f}", flush=True)
    del lin
    torch.cuda.empty_cache()

    mlp = nn.Sequential(nn.Linear(dim_4, args.hidden), nn.GELU(), nn.Linear(args.hidden, 256))
    run_probe(mlp, Xtr, ytr, dev, args.epochs, args.lr, args.batch, args.seed)
    mlp_logits_te = infer(mlp, Xte, dev, args.batch)
    mlp_logits_ca = infer(mlp, Xca, dev, args.batch)
    mlp_s, mlp_per_s = calib_s(mlp_logits_ca, yca)
    mlp_te = score_logits(mlp_logits_te, yte, scale=mlp_s)
    mlp_te_raw = score_logits(mlp_logits_te, yte, scale=1.0)
    print(f"exp116: mlp      test bpb(s*)={mlp_te['bpb']:.4f}@s={mlp_s} "
          f"bpb(s1)={mlp_te_raw['bpb']:.4f} top1={mlp_te['top1']:.4f} "
          f"top15={mlp_te['top15']:.4f} median={mlp_te['rank_median']:.0f}", flush=True)

    out = {
        "ckpt": args.ckpt, "ckpt_md5": md5,
        "data": args.data, "seed": args.seed,
        "dim_4": dim_4, "d_h": net.d_h, "mem_k": int(net._mem_m.shape[0]),
        "active_size": {k: int(v) for k, v in net.active_size.items()},
        "split": {"train_windows": len(train_wins), "cal_windows": len(cal_wins),
                  "test_windows": len(test_wins),
                  "test_source": f"gauge.tail[0:{N_EVAL_WIN}] (exp115 eval 口径)"},
        "probe_cfg": {"linear": "Linear(925,256)", "mlp": f"Linear(925,{args.hidden})-GELU-Linear({args.hidden},256)",
                      "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
                      "precision": "features fp16 (net 原生), 探针内部 fp32 (报告口径)"},
        "unigram": {"test": uni_te, "cal": uni_ca},
        "probe_linear": {"test_at_s_star": lin_te, "test_at_s1": lin_te_raw,
                         "cal_bpb_per_s": lin_per_s, "s_star": lin_s},
        "probe_mlp": {"test_at_s_star": mlp_te, "test_at_s1": mlp_te_raw,
                      "cal_bpb_per_s": mlp_per_s, "s_star": mlp_s},
    }
    path = run_file("z4_lang_probe.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"exp116: 写出 {path}", flush=True)


if __name__ == "__main__":
    main()
