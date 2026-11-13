"""
Z4 语言性探针 + 读出瓶颈细分.

冻结 v244 终态, 抽 net._z4 与其读出输入变换 z4_lm, 训线性/MLP 探针预测下一字节;
与 unigram 基线、模型自身读出 (同口径 raw logits) 对比.

判定: 探针 ≫ unigram → 表示含语言 (读法 A: 读出瓶颈); 探针 ≈ unigram → 读法 B (表示瓶颈).
细分: z4_raw 探针 ≫ z4_lm 探针 → 读出输入变换毁信息; 二者相当 → 读出学习规则不会学.

探针是外部测量仪器: 冻结网络零写入 (只前馈), 探针分类器独立于模型学习链路;
探针全 fp16 (与模型同精度, 无高位浮点对照路).
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
from crates.kernels import cubic, rms_norm, saturate
from crates.model import CyreneModel, HybridPCNet

torch.set_grad_enabled(False)

S_GRID = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 60.0)


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def z4lm_from_z4(net, z4):
    """读出输入变换 (readout.py:69-76 同构): z4 + W_diff 预测差 → 三阶非线性整流."""
    dim_4 = net.active_size["l4"]
    z4_n = z4 / (z4.norm(dim=-1, keepdim=True) + 1e-3)
    pred = z4_n @ net.W_diff[:dim_4, :dim_4].T + net.b_diff[:dim_4]
    z4r = z4 + pred
    z4_lm = saturate(z4r)
    z4_lm = rms_norm(z4_lm)
    z4_lm = cubic(z4_lm)
    return saturate(z4_lm)


def readout_logits(net, migrate=True):
    """模型自身读出前向 (readout.py:79-108 同构): W1 混合头 + [migrate] W1_new 新通道; raw logits."""
    assert not getattr(net, "_w1v2", False), "w1v2 分支不在本探针口径内"
    z4_lm = z4lm_from_z4(net, net._z4)
    zh = rms_norm(torch.cat([z4_lm, net._bind_vec, net._mem_out], dim=-1))
    h = cubic(rms_norm(saturate(zh @ net.W1)))
    logits = (h @ net.W_lm + net.bias_lm) * (1.0 / math.sqrt(net.d_h))
    if migrate and getattr(net, "_migrate_expand", False):
        h_new = cubic(rms_norm(saturate(zh @ net.W1_new)))
        logits = logits + h_new @ net.W_lm_new
    return logits


def readout_lc(logits):
    """部署口径能量调制 (readout.py:110-113; exp115_say._readout_lc 同构): 中心化 → /std → max|·|.
    注意: 生成掩码 (readout.py:115, 0x00-0x1F→-1e4) 属生成约束, 不进 bpb 评估 (会罚换行 0x0A)."""
    lg = logits
    lc = (lg - lg.mean(dim=-1, keepdim=True)) / (lg.std(dim=-1, keepdim=True) + 1e-4)
    return lc / lc.abs().max(dim=-1, keepdim=True).values


def extract(net, dev, wins, tag, want_readout=False):
    xs, ys = [], []
    ro_mig, ro_red = [], []
    t0 = time.time()
    for i, w in enumerate(wins):
        x = w.unsqueeze(0).to(dev)
        net.forward(x)
        xs.append(net._z4[0, :-1].to("cpu", torch.float16))
        ys.append(x[0, 1:].to("cpu"))
        if want_readout:
            ro_mig.append(readout_logits(net, migrate=True)[0, :-1].to("cpu", torch.float16))
            ro_red.append(readout_logits(net, migrate=False)[0, :-1].to("cpu", torch.float16))
        if (i + 1) % 500 == 0:
            print(f"  [{tag}] extract {i + 1}/{len(wins)} ({time.time() - t0:.0f}s)", flush=True)
    X = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0).long()
    RO = ({"mig": torch.cat(ro_mig, dim=0), "red": torch.cat(ro_red, dim=0)}
          if want_readout else None)
    return X, y, RO


def score_logits(logits, y, scale=1.0):
    lg = logits * scale
    lsm = torch.log_softmax(lg, dim=-1)
    bpb = (-lsm.gather(1, y.unsqueeze(1)).squeeze(1)).mean().item() / math.log(2)
    order = lg.argsort(dim=-1, descending=True)
    r = order.argsort(dim=-1).gather(1, y.unsqueeze(1)).squeeze(1)
    return {"bpb": bpb, "top1": (r == 0).half().mean().item(),
            "top15": (r < 15).half().mean().item(),
            "rank_median": r.median().item(), "rank_mean": r.to(torch.float16).mean().item(),
            "n": int(y.numel())}


def run_probe(model, Xtr, ytr, dev, epochs, lr, bs, seed):
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
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr[idx].to(torch.float16)
            yb = ytr[idx]
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
            xb = X[i:i + bs].to(dev, torch.float16)
            outs.append(model(xb).to("cpu"))
    return torch.cat(outs)


def bpb_only(logits, y, scale=1.0):
    lg = logits * scale
    lsm = torch.log_softmax(lg, dim=-1)
    return (-lsm.gather(1, y.unsqueeze(1)).squeeze(1)).mean().item() / math.log(2)


def calib_s(logits, y):
    best_s, best_bpb = 1.0, None
    per_s = {}
    for s in S_GRID:
        bpb = bpb_only(logits, y, scale=s)
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
    ap.add_argument("--epochs", type=int, default=30)
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
                      lm_freeze_w1=False)
    net = HybridPCNet.load(ckpt, cfg).to(dev)
    net.eval()
    dim_4 = net.active_size["l4"]
    print(f"exp116: dim_4={dim_4} d_h={net.d_h} mem_k={net._mem_m.shape[0]} "
          f"active={net.active_size}", flush=True)

    gw = {"obj": None}

    def _wrap_graph():
        from exp115_say import GraphedPredict
        print("exp116: 前向罩 CUDA Graph (单窗形状 (1,256))", flush=True)
        gw["obj"] = GraphedPredict(net.forward_engine, net)
        net.forward_engine._predict = gw["obj"]

    def _unwrap_graph():
        if gw["obj"] is not None:
            net.forward_engine._predict = gw["obj"].raw
            gw["obj"].graphs.clear()
            gw["obj"] = None
            torch.cuda.empty_cache()

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
        extract(net, dev, wins, "smoke")
        dt = (time.time() - t0) / len(wins)
        print(f"exp116 smoke: {dt * 1000:.1f} ms/forward → "
              f"4096 窗约 {dt * 4096:.0f}s", flush=True)
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

    Xtr, ytr, _ = extract(net, dev, train_wins, "train")
    Xca, yca, RO_ca = extract(net, dev, cal_wins, "cal", want_readout=True)
    Xte, yte, RO_te = extract(net, dev, test_wins, "test", want_readout=True)
    _unwrap_graph()
    print(f"exp116: features train={tuple(Xtr.shape)} cal={tuple(Xca.shape)} "
          f"test={tuple(Xte.shape)}", flush=True)

    cnt = torch.bincount(ytr, minlength=256).to(torch.float16)
    p = cnt / int(cnt.sum())
    p = p.clamp_min(1.0 / int(2 * cnt.sum()))
    uni_logits = torch.log(p)
    uni_te = score_logits(uni_logits.unsqueeze(0).expand(yte.numel(), -1), yte)
    uni_ca = score_logits(uni_logits.unsqueeze(0).expand(yca.numel(), -1), yca)
    print(f"exp116: unigram   test bpb={uni_te['bpb']:.4f} top1={uni_te['top1']:.4f} "
          f"top15={uni_te['top15']:.4f} median={uni_te['rank_median']:.0f}", flush=True)

    ro_s, ro_per_s = calib_s(RO_ca["mig"], yca)
    ro_te = score_logits(RO_te["mig"], yte, scale=ro_s)
    ro_te_s1 = score_logits(RO_te["mig"], yte, scale=1.0)
    print(f"exp116: readout[raw 头]  test bpb(s*)={ro_te['bpb']:.4f}@s={ro_s} "
          f"bpb(s1)={ro_te_s1['bpb']:.4f} top1={ro_te['top1']:.4f} "
          f"top15={ro_te['top15']:.4f} median={ro_te['rank_median']:.0f}", flush=True)

    lc_mig_ca, lc_mig_te = readout_lc(RO_ca["mig"]), readout_lc(RO_te["mig"])
    ev_s, ev_per_s = calib_s(lc_mig_ca, yca)
    ev_te = score_logits(lc_mig_te, yte, scale=60.0)
    ev_te_cal = score_logits(lc_mig_te, yte, scale=ev_s)
    print(f"exp116: readout[评估lc] test bpb(s60)={ev_te['bpb']:.4f} "
          f"bpb(s*)={ev_te_cal['bpb']:.4f}@s={ev_s} top1={ev_te['top1']:.4f} "
          f"top15={ev_te['top15']:.4f} median={ev_te['rank_median']:.0f}", flush=True)

    lc_red_ca, lc_red_te = readout_lc(RO_ca["red"]), readout_lc(RO_te["red"])
    red_s, red_per_s = calib_s(lc_red_ca, yca)
    red_te = score_logits(lc_red_te, yte, scale=60.0)
    red_te_cal = score_logits(lc_red_te, yte, scale=red_s)
    print(f"exp116: readout[exp115减] test bpb(s60)={red_te['bpb']:.4f} "
          f"bpb(s*)={red_te_cal['bpb']:.4f}@s={red_s} top1={red_te['top1']:.4f} "
          f"top15={red_te['top15']:.4f} median={red_te['rank_median']:.0f}", flush=True)

    mk_te = score_logits(lc_mig_te * 60.0 + (1.0 - net._mask_print.cpu()) * -1e4, yte, scale=1.0)
    print(f"exp116: readout[生成掩码 仅诊断] test bpb={mk_te['bpb']:.2f} "
          f"(0x00-0x1F→-1e4 罚换行, 不可用于 bpb)", flush=True)

    Xtr_gpu = Xtr.to(dev, torch.float16)
    ytr_gpu = ytr.to(dev)
    probes = {}
    for tf_name in ("z4_raw", "z4_lm"):
        if tf_name == "z4_lm":
            Xtr_t = z4lm_from_z4(net, Xtr_gpu)
            del Xtr_gpu
            torch.cuda.empty_cache()
        else:
            Xtr_t = Xtr_gpu
        conv = (lambda X: z4lm_from_z4(net, X.to(dev, torch.float16))) if tf_name == "z4_lm" \
            else (lambda X: X.to(dev, torch.float16))
        Xte_g, Xca_g = conv(Xte), conv(Xca)
        probes[tf_name] = {}
        for kind in ("linear", "mlp"):
            model = (nn.Linear(dim_4, 256) if kind == "linear" else
                     nn.Sequential(nn.Linear(dim_4, args.hidden), nn.GELU(),
                                   nn.Linear(args.hidden, 256))).half()
            run_probe(model, Xtr_t, ytr_gpu, dev, args.epochs, args.lr, args.batch, args.seed)
            l_te = infer(model, Xte_g, dev, args.batch)
            l_ca = infer(model, Xca_g, dev, args.batch)
            s_star, per_s = calib_s(l_ca, yca)
            te = score_logits(l_te, yte, scale=s_star)
            te_s1 = score_logits(l_te, yte, scale=1.0)
            probes[tf_name][kind] = {"s_star": s_star, "cal_bpb_per_s": per_s,
                                     "test_at_s_star": te, "test_at_s1": te_s1}
            print(f"exp116: {tf_name:6s} {kind:6s} test bpb(s*)={te['bpb']:.4f}@s={s_star} "
                  f"bpb(s1)={te_s1['bpb']:.4f} top1={te['top1']:.4f} "
                  f"top15={te['top15']:.4f} median={te['rank_median']:.0f}", flush=True)
            del model
            torch.cuda.empty_cache()
        del Xtr_t, Xte_g, Xca_g
        torch.cuda.empty_cache()

    out = {
        "ckpt": args.ckpt, "ckpt_md5": md5,
        "data": args.data, "seed": args.seed,
        "dim_4": dim_4, "d_h": net.d_h, "mem_k": int(net._mem_m.shape[0]),
        "active_size": {k: int(v) for k, v in net.active_size.items()},
        "split": {"train_windows": len(train_wins), "cal_windows": len(cal_wins),
                  "test_windows": len(test_wins),
                  "test_source": f"gauge.tail[0:{N_EVAL_WIN}] (exp115 eval 口径)"},
        "probe_cfg": {"linear": f"Linear({dim_4},256)",
                      "mlp": f"Linear({dim_4},{args.hidden})-GELU-Linear({args.hidden},256)",
                      "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
                      "transforms": {"z4_raw": "net._z4 原样",
                                     "z4_lm": "readout.py:70-80 同构 (W_diff 预测差+三阶整流)"},
                      "precision": "全链路 fp16 (features 与探针同精度)"},
        "unigram": {"test": uni_te, "cal": uni_ca},
        "model_readout_raw": {"basis": "readout.py:79-108 (W1 头 + W1_new 新通道); 调制前 raw logits; "
                                       "与探针同 score_logits 口径",
                              "s_star": ro_s, "cal_bpb_per_s": ro_per_s,
                              "test_at_s_star": ro_te, "test_at_s1": ro_te_s1},
        "model_readout_eval_lc": {"basis": "readout.py:79-113 能量调制 lc∈[-1,1] (无生成掩码); 部署 bpb 用 lc×60",
                                  "s_star": ev_s, "cal_bpb_per_s": ev_per_s,
                                  "test_at_s60": ev_te, "test_at_s_star": ev_te_cal},
        "model_readout_exp115": {"basis": "exp115_say._readout_lc 口径 (仅 W1 头, 能量调制, 无掩码); "
                                          "锚点 bpb(s60) 应 ≈ 17.207",
                                 "s_star": red_s, "cal_bpb_per_s": red_per_s,
                                 "test_at_s60": red_te, "test_at_s_star": red_te_cal},
        "mask_generation_only": {"basis": "readout.py:115 生成掩码 0x00-0x1F→-1e4 (生成约束, 罚换行 0x0A, "
                                          "不可用于 bpb 评估)",
                                 "test_at_s1": mk_te},
        "probe_z4_raw": probes["z4_raw"],
        "probe_z4_lm": probes["z4_lm"],
    }
    path = run_file("z4_lang_probe.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"exp116: 写出 {path}", flush=True)


if __name__ == "__main__":
    main()
