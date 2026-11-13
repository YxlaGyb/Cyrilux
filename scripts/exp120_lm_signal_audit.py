"""
exp120: 学习信号审计 (S3-f).

问题: exp119/S3-c 证 v201 起点读出头即坏 (整机 5.73 > unigram 4.77), v244 更差 (6.73);
exp118 证同架构反传上界 2.33 (架构充分) → 瓶颈在"学习侧". 本审计拆学习信号:

F-a 静态 (一级证据, 读代码): _build_lm_signal (readout.py:56-260) 的 err/d_t/更新式, 逐项对照 delta 规则.
F-b 对照 (二级证据, 关键): v244 的 zh 上, Hebbian 部署态初始化 A1 头, 反传训 {只 W_lm / 只 W1 / 都训}, 找破坏源.
F-c 起点 (二级证据): v201 raw 上跑 exp116 的 z4_lm MLP 探针 → 起点上游是否也有信息.

F-b 初始化 = Hebbian 部署态 (用户裁决): W1←v244.W1, W_lm←v244.W_lm, bias←v244.bias_lm (red 单路径).
  冻结的那层停在部署值 → "固定一层(保持其部署值), 反传能否救回另一层".
铁律: 前台; 不改 crates/; 不改 v244/v201; 不落盘权重; 探针零写回; 新目录 v250.
"""
import argparse
import json
import time

import torch
import torch.nn as nn
from exp115_say import N_EVAL_WIN, GraphedPredict, build_gauge
from exp116_z4_probe import (
    _md5,
    calib_s,
    extract,
    infer,
    readout_logits,
    run_probe,
    score_logits,
    z4lm_from_z4,
)
from exp118_head_upperbound import HeadA1, extract_zh, train_head, zh_from_net
from safetensors.torch import load_file

from crates.cli.utils import resolve_path, run_file
from crates.model import CyreneModel, HybridPCNet

torch.set_grad_enabled(False)

VARIANTS = {"both": (), "only_W_lm": ("W1",), "only_W1": ("W_lm",)}


def init_hebbian(head, sd):
    """A1 头 ← v244 实装 red 单路径: W1(无 bias)/W_lm/bias_lm; ×1/√d_h 同 readout. 全 fp16."""
    head = head.half()
    with torch.no_grad():
        head.W1.weight.copy_(sd["W1"].T.contiguous())
        head.W1.bias.zero_()
        head.W_lm.weight.copy_(sd["W_lm"].T.contiguous())
        head.W_lm.bias.copy_(sd["bias_lm"])
    return head


def set_freeze(head, frozen):
    for name in ("W1", "W_lm"):
        req = name not in frozen
        getattr(head, name).weight.requires_grad_(req)
        getattr(head, name).bias.requires_grad_(req)


def fa_static():
    return {
        "scope": "readout.py:56-260 (_build_lm_signal) + readout.py:375-520 (_update_lm_head)",
        "err_caliber": {
            "logits": "(h @ W_lm + bias_lm) * (1/sqrt(d_h)) [+ h_new @ W_lm_new if migrate] (L106-108)",
            "energy_mod": "逐行中心化 → /(std+1e-4) → ×60/max|·| → logits in [-60,60] (L110-113)",
            "probs": "softmax(logits) fp16 (L132)",
            "eps_lm": "target_lm - probs_lm[:, :-1], detach, ×lm_mask (L129,134,147)",
            "diff2": "(t_{t+2}-t_{t+1}) - (p_{t+2}-p_{t+1}), ×0.2, 掩码+pad 到 S-1 (L136-144)",
            "eps_total": "eps_lm + 0.2*diff2  <- 驱动 W_lm 与 W1 (L178)",
            "eps_t2_total": "eps_t2 + 0.2*diff2[:, :-1] <- 驱动 W_lm_2 (L179)",
        },
        "transform_in_update": {
            "err_scaled": "eps_total 逐行单位化: eps*alive/(|eps|*1.01) (L396-399) <- 幅度信息被剥离",
            "contrast_w": "1/(1+0.1*|logits_d - logit_target|), 目标位=1 (L401-411) <- 非梯度列重加权",
            "d_t": "sigmoid((novelty - tau)*500) in (0,1), tau EMA(0.99); 无 _novelty → 恒 1 (L414-421)",
        },
        "updates": {
            "W_lm": "dW_lm=(rms_norm(h)^T @ (err_contrast*mask*d_t)).mean(0)*sqrt(d_h) (L423-425); 全局单位化 (L443-445); +elig(W_lm)*_survival_signal (L446-448); 门 (not _mig or _realign) and not _rdfrz; W_lm += dW*eta_lm*|W_lm|*TR*_bhw (L449-450); 等比帽 10 (L461-466)",
            "W1": "e_h=(err_scaled @ W_lm^T)*h_deriv, h_deriv=1-1.5*h_in^2 (L474); dW1=(rms_norm(zh)^T @ rms_norm(e_h)).mean(0) (L476); /max/64 (L481); 全局单位化 (L482-484); +elig(W1)*_survival_signal +_single_recon_w1 (L486-490); 门 not lm_freeze_w1 and (not _mig or _realign) and not _rdfrz; W1 += dW*eta_lm*|W1|*TR*_bhw (L491); soft_norm_preserve(W1) 行范数保持 0.8-1.2 (L492)",
            "bias_lm": "bias_d=mean(eps_lm, dim=(0,1)); bias_lm += (bias_d-mean(bias_d))*(eta_lm/20); 范数帽 10 (L451-460)",
        },
        "constants": {"LM_TRUST_REGION": 0.025, "ELIG_GAMMA": 0.95, "inv_h": "1/sqrt(256)"},
        "delta_rule_check": [
            "方向: dW_lm ∝ h^T(y-p) → target 列 +h, 其余 -h = 交叉熵下降方向; 符号正确, W_lm 单独看是合法 softmax 分类方向.",
            "偏差1: err 逐行单位化 (L396-399) 抹掉 |y-p| 置信度加权 → 自信且正确 (y-p≈0) 的样本仍产生满幅单位更新 → 噪声注入.",
            "偏差2: contrast_w 乘性列重加权 (L406-411) 非梯度, 扭曲各列步长.",
            "偏差3: d_t 新奇度门 (L418) 缩放更新幅度, 与 CE 无关.",
            "偏差4: eps_total 混入 0.2*diff2 (t+2/t+1 差分目标) → W_lm/W1 同被喂两个预测视界.",
            "偏差5: dW 全局单位化 + 步长 eta*|W|*TR (相对信任域 0.025) → 步长取决于层范数而非梯度范数.",
            "偏差6: W1 非局部规则: e_h=err_scaled@W_lm^T (反传穿过 W_lm) 再乘 h_deriv → W1/W_lm 共享同一误差, 构成耦合的 1 层反传, 非两个独立 Hebbian 规则.",
            "偏差7: 更新后 soft_norm_preserve(W1) 重写行范数 (0.8-1.2) → 可部分抵消更新方向.",
            "偏差8: elig 迹默认惯性 (_survival_signal 默认 0, _common.py:70-76) → 迹只记录不注入; W1_elig/W_lm_elig 位移仅是缓冲, 非权重来源.",
            "结论: '方向合法、幅度/权重被破坏' 的 CE 近似 + 层间耦合反传 → 与观察 '头动了但变坏' 一致.",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-v244", default="out/v244-20261009-214217/e9f_bplus_readout_realign_500.safetensors")
    ap.add_argument("--raw-v201", default="out/v201-20260923-093527/merge.safetensors")
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.85)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dev = torch.device(args.device)
    ck24 = resolve_path(args.ckpt_v244)
    ck01 = resolve_path(args.raw_v201)
    print(f"exp120: v244={ck24} md5={_md5(ck24)}", flush=True)
    print(f"exp120: raw v201={ck01} md5={_md5(ck01)}", flush=True)

    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=256,
                      lm_freeze_w1=False)

    torch.manual_seed(args.seed)
    gauge, _, _ = build_gauge(args.data, dev)
    test_wins = gauge["tail"][:N_EVAL_WIN]
    fit_pool = gauge["trunc"] + gauge["tail"][N_EVAL_WIN:]
    order = torch.randperm(len(fit_pool), generator=torch.Generator().manual_seed(args.seed)).tolist()
    fit_pool = [fit_pool[i] for i in order]
    n_tr = int(len(fit_pool) * args.train_frac)
    train_wins, cal_wins = fit_pool[:n_tr], fit_pool[n_tr:]
    print(f"exp120: split train={len(train_wins)} cal={len(cal_wins)} test={len(test_wins)}", flush=True)

    out = {
        "ckpt_v244": args.ckpt_v244, "md5_v244": _md5(ck24),
        "raw_v201": args.raw_v201, "md5_raw_v201": _md5(ck01),
        "seed": args.seed,
        "split": {"train_windows": len(train_wins), "cal_windows": len(cal_wins),
                  "test_windows": len(test_wins),
                  "test_source": f"gauge.tail[0:{N_EVAL_WIN}] (exp115/116 eval 口径)"},
        "config": {"epochs": args.epochs, "lr": args.lr, "batch": args.batch,
                   "head": "HeadA1 Linear(d_in,256)-sat-rms-cubic-Linear(256,256), ×1/sqrt(256)",
                   "Fb_init": "Hebbian 部署态: W1<-v244.W1, W_lm<-v244.W_lm, bias<-v244.bias_lm (red 单路径)",
                   "Fb_freeze": "冻结层停在部署值",
                   "precision": "全链路 fp16 (探针=模型同精度; Adam eps=1e-4 fp16 适配)"},
        "Fa_static": fa_static(),
        "anchors": {"v244_mig_real": 6.7324, "v244_red_real": 7.3483, "v201_mig_real": 5.7285,
                    "unigram_T": 4.7677, "exp118_A1_upperbound": 2.3322,
                    "exp116_v244_z4lm_mlp": 2.4144,
                    "caliber": "锚点为旧 fp32 仪器历史化石; 本实验全链路 fp16 口径"},
    }

    # ── F-b: v244 zh 上的三个反传变体 (Hebbian 部署态初始化) ──
    if args.smoke:
        net = HybridPCNet.load(ck24, cfg).to(dev)
        net.eval()
        w = gauge["tail"][0].unsqueeze(0).to(dev)
        net.forward(w)
        z = zh_from_net(net)
        head = init_hebbian(HeadA1(z.shape[-1], 256), load_file(ck24)).to(dev)
        d = float((head(z[0, :-1]) - readout_logits(net, migrate=False)[0, :-1]).abs().max().item())
        print(f"exp120 smoke: zh={tuple(z.shape)} init-vs-readout max|d|={d:.3e}", flush=True)
        return

    net24 = HybridPCNet.load(ck24, cfg).to(dev)
    net24.eval()
    gw = {"obj": None}

    def wrap(net):
        gw["obj"] = GraphedPredict(net.forward_engine, net)
        net.forward_engine._predict = gw["obj"]

    if not args.no_graph and dev.type == "cuda":
        wrap(net24)

    Xtr, ytr = extract_zh(net24, dev, train_wins, "train")
    Xca, yca = extract_zh(net24, dev, cal_wins, "cal")
    Xte, yte = extract_zh(net24, dev, test_wins, "test")
    d_in = Xtr.shape[-1]

    sd = load_file(ck24)
    h0 = init_hebbian(HeadA1(d_in, 256), sd).to(dev)
    wref = gauge["tail"][0].unsqueeze(0).to(dev)
    net24.forward(wref)
    ref = readout_logits(net24, migrate=False)[0, :-1]
    chk = float((h0(zh_from_net(net24)[0, :-1]) - ref).abs().max().item())
    out["Fb_init_selfcheck"] = {"init_vs_readout_red_max_abs": chk,
                               "note": "A1(Hebbian init) 前向应 == exp116 readout_logits(red); 验证转置初始化正确"}
    print(f"exp120: [F-b] init 自检 max|head - readout_red| = {chk:.3e}", flush=True)
    del h0
    torch.cuda.empty_cache()

    print(f"exp120: [F-b] zh d_in={d_in} tr={tuple(Xtr.shape)} ca={tuple(Xca.shape)} te={tuple(Xte.shape)}", flush=True)
    fb = {}
    for vname, frozen in VARIANTS.items():
        head = init_hebbian(HeadA1(d_in, 256), sd)
        set_freeze(head, frozen)
        t0 = time.time()
        train_head(head, Xtr, ytr, dev, args.epochs, args.lr, args.batch, args.seed)
        l_te = infer(head, Xte, dev, args.batch)
        l_ca = infer(head, Xca, dev, args.batch)
        s_star, per_s = calib_s(l_ca, yca)
        te = score_logits(l_te, yte, scale=s_star)
        te_s1 = score_logits(l_te, yte, scale=1.0)
        fb[vname] = {"frozen": list(frozen), "s_star": s_star, "cal_bpb_per_s": per_s,
                     "test_at_s_star": te, "test_at_s1": te_s1}
        print(f"exp120: [F-b] {vname:10s} frozen={list(frozen)} bpb(s1)={te_s1['bpb']:.4f} "
              f"bpb(s*)={te['bpb']:.4f}@s={s_star} top1={te['top1']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        del head
        torch.cuda.empty_cache()

    out["Fb_variants"] = fb
    ranking = sorted(fb.items(), key=lambda kv: kv[1]["test_at_s1"]["bpb"])
    out["Fb_ranking"] = [{"variant": k, "bpb_s1": v["test_at_s1"]["bpb"],
                          "bpb_s_star": v["test_at_s_star"]["bpb"], "s_star": v["s_star"]}
                         for k, v in ranking]
    print("exp120: [F-b] 排序(好→坏)=" +
          ", ".join(f"{k}:{v['test_at_s1']['bpb']:.3f}" for k, v in ranking), flush=True)

    def ok(b):
        return b <= 3.0

    b_wlm = fb["only_W_lm"]["test_at_s1"]["bpb"]
    b_w1 = fb["only_W1"]["test_at_s1"]["bpb"]
    if ok(b_wlm) and not ok(b_w1):
        verdict = ("只训 W_lm 可救回 (其余固定) -> Hebbian W1 的 h 可解码, 解码层 W_lm 是主要破坏源 "
                   "(改 W_lm 更新规则/尺度)")
    elif ok(b_w1) and not ok(b_wlm):
        verdict = ("只训 W1 可救回 -> Hebbian W_lm 可读, 混合层 W1 是主要破坏源 (改 W1 信号或冻结它)")
    elif ok(b_wlm) and ok(b_w1):
        verdict = ("两层单独重训都可救回 -> 非单层容量问题; 两信号耦合/共同漂移 (拆信号, 各给各的)")
    else:
        verdict = ("两层单独重训都救不回, 只'都训'救回 -> 强耦合: 必须联合重训 (现有逐步 Hebbian 无法协调两层)")
    out["Fb_verdict"] = verdict
    print(f"exp120: [F-b] 判定 = {verdict}", flush=True)

    del net24, Xtr, Xca, Xte
    torch.cuda.empty_cache()

    # ── F-c: v201 raw 上跑 exp116 的 z4_lm MLP 探针 ──
    net01 = HybridPCNet.load(ck01, cfg).to(dev)
    net01.eval()
    if not args.no_graph and dev.type == "cuda":
        gw["obj"] = GraphedPredict(net01.forward_engine, net01)
        net01.forward_engine._predict = gw["obj"]
    dim_4 = net01.active_size["l4"]
    Xr, yr, _ = extract(net01, dev, train_wins, "v201-train")
    Xc, yc, _ = extract(net01, dev, cal_wins, "v201-cal")
    Xe, ye, _ = extract(net01, dev, test_wins, "v201-test")
    if gw["obj"] is not None:
        net01.forward_engine._predict = gw["obj"].raw
        gw["obj"].graphs.clear()
        gw["obj"] = None
        torch.cuda.empty_cache()

    Xr = z4lm_from_z4(net01, Xr.to(dev, torch.float16))
    Xc_g = z4lm_from_z4(net01, Xc.to(dev, torch.float16))
    Xe_g = z4lm_from_z4(net01, Xe.to(dev, torch.float16))
    model = nn.Sequential(nn.Linear(dim_4, args.hidden), nn.GELU(), nn.Linear(args.hidden, 256)).half()
    run_probe(model, Xr, yr.to(dev), dev, args.epochs, args.lr, args.batch, args.seed)
    l_te = infer(model, Xe_g, dev, args.batch)
    l_ca = infer(model, Xc_g, dev, args.batch)
    s_star, per_s = calib_s(l_ca, yc)
    te = score_logits(l_te, ye, scale=s_star)
    te_s1 = score_logits(l_te, ye, scale=1.0)
    out["Fc_v201_z4lm_mlp"] = {"dim_4": dim_4, "s_star": s_star, "cal_bpb_per_s": per_s,
                               "test_at_s_star": te, "test_at_s1": te_s1}
    print(f"exp120: [F-c] v201 z4_lm MLP bpb(s1)={te_s1['bpb']:.4f} bpb(s*)={te['bpb']:.4f}@s={s_star} "
          f"top1={te['top1']:.4f} top15={te['top15']:.4f}", flush=True)
    fc = te_s1["bpb"]
    if fc <= 3.0:
        fc_v = "v201 上游有信息 (z4_lm MLP ≈2.x) -> 头是纯粹的失败, 问题在头, 不在上游"
    elif fc >= 5.0:
        fc_v = "v201 上游也坏 (z4_lm MLP ≈5.x) -> 问题再往前推一格 (表示层)"
    else:
        fc_v = f"v201 z4_lm MLP={fc:.3f} -> 上游部分可读 (中间态)"
    out["Fc_verdict"] = fc_v
    print(f"exp120: [F-c] 判定 = {fc_v}", flush=True)

    out["iron_rules"] = "前台; 不改 crates/; 不改 v244/v201; 不落盘权重; 探针零写回; 新目录 v250"
    path = run_file("lm_signal_audit.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"exp120: 写出 {path}", flush=True)


if __name__ == "__main__":
    main()
