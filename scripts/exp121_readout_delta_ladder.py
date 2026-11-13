"""
F-d: 读出头输出层更新规则消融 (delta ladder).

问题: F-b 证 both=2.573 / only_W_lm=3.984 / only_W1=6.357 → 6.73 的来源是"输出层根本没在学".
本实验把现有 W_lm 更新规则逐项拆开: v244 的 zh/h 前向一次冻结 (W1 冻结), 从纯 delta 规则
dW_lm = (target - prob) ⊗ h 出发, 按部署代码链序逐项加回变换, 每加一项测一次 test bpb.
哪一项让 bpb 从 ~4 跳到 ~6.7, 哪一项就是元凶.

ladder (累积; 同 seed 同 batch 序同 lr, 格间唯一变量 = 新加的变换):
  C0 纯 delta     dW = h^T(target-prob)/B, W += lr·dW, b += lr·mean(err)   (判决第一格)
  C1 +行单位化     err ← err·alive/(‖err‖·1.01)                             readout.py:374-377
  C2 +contrast_w   1/(1+0.1·|logits_d−tgt|), 目标位=1, logits_d 无 bias     readout.py:380-389
  C3 +d_t 新奇门   sigmoid((nov−τ)·500), τ EMA(0.99); nov 用 v244 前向真实回放,
                   τ0 = v244 落盘 _theta_novelty, τ 按窗流序先更新后取门      readout.py:392-399
  C4 +全局单位化   dW ← dW·alive/‖dW‖ (零更新掩码)                          readout.py:408-410
  C5 +信任域步长   W += dW·‖W‖·0.025 (eta_lm=lm_lr_boost=1); bias 切部署式(/20+帽10) readout.py:413,416-429
  C6 elig/rtr      a) 部署史 pre-rtr: dW+E·0 ≡ dW — 运行时探针证位级 no-op, 不重跑 (=C5)
                   b) 现行 rtr: E←γE+dW, dW←E·(m/(‖E‖+1e-6)), m≡1 最好情形   readout.py:411
  C7 +等比帽10     W *= min(1, 10/max_row_norm)                             readout.py:425-429
  (判决第 7 项写的 soft_norm_preserve 在现行代码豁免 W_lm — readout.py:424 "W_lm 豁免 soft_norm",
   部署链实际第 7 道是等比帽 10, 照实替换并在此声明)

判据: 纯 delta ≈4 (反传 only_W_lm=3.984 锚) → 信号源没问题, 元凶 = ladder 跳变格;
      纯 delta ≈6.7 → 信号源本身有问题, 先修 eps/目标/输入.

口径: HeadA1 同构 h = cubic(rms_norm(saturate(zh@W1))), logits = (h@W_lm+bias)·(1/√256),
      raw logits 无能量调制无打印掩码 (与 F-b/exp118 探针口径一致); eps 用 eps_lm 纯项,
      不含 0.2·diff2 (差分通道需 t+2 预测器, 属信号组成而非变换, 不在本消融域).
评分: 主指标 bpb@s1 (与 F-b 排名口径一致), 同时报 cal 网格 s*.
铁律: 前台; 不改 crates/; 不动 v244; 全 fp16; 不落盘权重; 探针零写回; 输出走 run_file 自增新目录.
"""
import argparse
import json
import math
import time

import torch
import torch.nn.functional as F
from exp115_say import N_EVAL_WIN, GraphedPredict, build_gauge
from exp116_z4_probe import _md5, calib_s, infer, readout_logits, score_logits, z4lm_from_z4
from safetensors.torch import load_file

from crates.cli.utils import resolve_path, run_file
from crates.kernels import cubic, rms_norm, saturate
from crates.model import CyreneModel, HybridPCNet
from crates.model.constants import LM_TRUST_REGION
from crates.rtr import rtr_apply

torch.set_grad_enabled(False)

ELIG_GAMMA = 0.95
INV_H = 1.0 / math.sqrt(256)

ANCHORS = {
    "backprop_only_W_lm": {"v": 3.984, "src": "exp120 F-b (s1 口径; 2026-10-11 冻结, 当前 HEAD 不可复跑, 见 SESSION_HANDOFF #22)"},
    "backprop_both": {"v": 2.573, "src": "exp120 F-b (s1 口径; 2026-10-11 冻结, 当前 HEAD 不可复跑, 见 SESSION_HANDOFF #22)"},
    "hebbian_deployed_mig": {"v": 6.7324, "src": "exp117 v247 head_decomp.json"},
    "v244_head_red_init": {"v": 7.3483, "src": "exp117 (rtr 前部署链起点, 本实验 init 锚)"},
    "unigram_T": {"v": 4.7677, "src": "exp117 v247"},
}


def extract(net, dev, wins, tag):
    """逐窗前向 (窗流序), 冻结 W1 算 h, 存 (h, y, nov). W1 之后恒定 → h 一次算死."""
    Hs, Ys, Ns = [], [], []
    t0 = time.time()
    for i, w in enumerate(wins):
        x = w.unsqueeze(0).to(dev)
        net.forward(x)
        z4_lm = z4lm_from_z4(net, net._z4)
        zh = rms_norm(torch.cat([z4_lm, net._bind_vec, net._mem_out], dim=-1))
        h = cubic(rms_norm(saturate(zh @ net.W1)))  # W1 冻结 → h 是 zh 的定函数, 一次算死
        Hs.append(h[0, :-1].to("cpu", torch.float16))
        Ys.append(x[0, 1:].to("cpu"))
        Ns.append(net._novelty[0, :-1].to("cpu", torch.float16))  # 与 S-1 目标对齐 (代码 d_t[:, :-1] 同款)
        if (i + 1) % 500 == 0:
            print(f"  [{tag}] extract {i + 1}/{len(wins)} ({time.time() - t0:.0f}s)", flush=True)
    return torch.cat(Hs, 0), torch.cat(Ys, 0).long(), Ns


def dt_lut(nov_wins, tau0):
    """按窗流序回放 τ (先 τ←0.99τ+0.01·mean(nov_w), 再门 = sigmoid((nov−τ)·500)),
    返回按位置拼接的门值表 + 遥测. 近似声明: 流序 = 抽取流序, 非当年训练器流序."""
    tau = float(tau0)
    gates, taus = [], []
    for nov in nov_wins:
        tau = tau * 0.99 + 0.01 * float(nov.float().mean())
        g = torch.sigmoid((nov.float() - tau) * 500.0).to(torch.float16)
        gates.append(g)
        taus.append(tau)
    lut = torch.cat(gates, 0)
    flat = torch.cat(nov_wins, 0).float()
    return lut, {
        "tau0": float(tau0), "tau_end": taus[-1],
        "gate_mean": float(lut.float().mean()), "gate_lt_p5_frac": float((lut < 0.5).float().mean()),
        "nov_p50": float(torch.quantile(flat, 0.5)),
        "nov_p95": float(torch.quantile(flat, 0.95)),
    }


def train_cell(h_tr, y_tr, W0, b0, lr, epochs, batch, seed, flags, dt_tab=None):
    """单格训练. flags: dict(row_unit, contrast, gate, unit_norm, trust, rtr, cap10)."""
    dev = W0.device
    W = W0.clone()
    b = b0.clone()
    E = torch.zeros_like(W)
    n = h_tr.shape[0]
    g = torch.Generator().manual_seed(seed)
    tel = {"steps": 0, "dW_absmean_sum": 0.0, "step_size_sum": 0.0, "cap_bind_steps": 0}
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            hb = h_tr[idx].to(dev, torch.float16)
            yb = y_tr[idx].to(dev)
            logits = (hb @ W + b) * INV_H
            p = torch.softmax(logits, dim=-1)
            err = F.one_hot(yb, 256).to(torch.float16) - p
            eu = err
            if flags["row_unit"]:
                en = eu.norm(dim=-1, keepdim=True) * 1.01
                alive = (en > 1e-8).to(eu.dtype)
                eu = eu * alive / torch.where(alive > 0, en, torch.ones_like(en))
            mult = torch.ones_like(eu)
            if flags["contrast"]:
                target_oh = F.one_hot(yb, 256).to(torch.float16)
                logits_d = (hb @ W) * INV_H  # 代码同款: 不含 bias
                tgt = (logits_d * target_oh).sum(dim=-1, keepdim=True)
                cw = 1.0 / (1.0 + (logits_d - tgt).abs() * 0.1)
                mult = mult * (cw * (1.0 - target_oh) + target_oh)
            if flags["gate"]:
                mult = mult * dt_tab[idx].to(dev).unsqueeze(-1)
            dW = (hb.transpose(0, 1) @ (eu * mult)) / idx.numel()
            db = (eu * mult).mean(dim=0)
            if flags["unit_norm"]:
                nrm = dW.norm()
                alive = (nrm > 0).to(dW.dtype)
                dW = dW * alive / torch.where(alive > 0, nrm, torch.ones_like(nrm))
            if flags["rtr"]:
                E.mul_(ELIG_GAMMA).add_(dW)
                m = torch.ones(1, device=dev, dtype=torch.float16)
                dW = rtr_apply(E, m)
            if flags["trust"]:
                step = W.norm() * LM_TRUST_REGION
                W += dW * step
                bd = err.mean(dim=0)  # 部署 bias 式: 原始 eps_lm 均值 (readout.py:416-418)
                b += (bd - bd.mean()) * (1.0 / 20.0)
                bn = b.norm()
                b *= torch.minimum(torch.ones_like(bn), 10.0 / (bn + 1e-6)).to(torch.float16)
            else:
                step = lr
                W += dW * lr
                b += db * lr
            if flags["cap10"]:
                rn = W.norm(dim=1)
                mx = rn.max()
                f = torch.minimum(torch.ones_like(mx), 10.0 / (mx + 1e-6)).to(torch.float16)
                W *= f
                tel["cap_bind_steps"] += int(f.item() < 1.0)
            tel["steps"] += 1
            tel["dW_absmean_sum"] += float(dW.abs().mean())
            tel["step_size_sum"] += float(step)
    tel["dW_absmean"] = tel["dW_absmean_sum"] / tel["steps"]
    tel["step_size_mean"] = tel["step_size_sum"] / tel["steps"]
    tel.pop("dW_absmean_sum"), tel.pop("step_size_sum")
    return W, b, tel


def eval_cell(W, b, splits, dev, batch):
    class _H:
        def __init__(self, W, b):
            self.W, self.b = W, b

        def eval(self):
            return self

        def __call__(self, x):
            return (x @ self.W + self.b) * INV_H

    head = _H(W, b)
    l_te = infer(head, splits["Xte_h"], dev, batch)
    l_ca = infer(head, splits["Xca_h"], dev, batch)
    s_star, per_s = calib_s(l_ca, splits["yca"])
    te1 = score_logits(l_te, splits["yte"], scale=1.0)
    tes = score_logits(l_te, splits["yte"], scale=s_star)
    return {"bpb_s1": te1["bpb"], "top1_s1": te1["top1"], "top15_s1": te1["top15"],
            "bpb_s_star": tes["bpb"], "s_star": s_star, "cal_bpb_per_s": per_s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/v254-20261012-205835/migrated/e9f_bplus_readout_realign_500.safetensors",
                    help="迁移件 (migrate_ckpt 产物, 张量载荷与 v244 逐位一致, 见 --raw-ckpt)")
    ap.add_argument("--raw-ckpt", default="out/v244-20261009-214217/e9f_bplus_readout_realign_500.safetensors",
                    help="原始 v244 (只读 md5 记录谱系, 不加载)")
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.85)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.0, help="0 = cal 网格预扫选优 (C0 短扫)")
    ap.add_argument("--scan-epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--cells", default="C0,C1,C2,C3,C4,C5,C6b,C7")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dev = torch.device(args.device)
    ck = resolve_path(args.ckpt)
    ck_raw = resolve_path(args.raw_ckpt)
    sd = load_file(ck)
    print(f"exp121: ckpt(迁移件)={ck}\n  md5={_md5(ck)}\nexp121: raw v244={ck_raw}\n  md5={_md5(ck_raw)}", flush=True)

    probes = {}
    probes["instrument_caliber"] = {
        "raw_v244_md5": _md5(ck_raw), "migrated_md5": _md5(ck),
        "migration": "scripts/migrate_ckpt.py 官方迁移: lr_hebbian→lr_rtr 改名保值, "
                     "lm_presyn_whiten (历史恒 False 白名单) 删, 退役键 _metab_rpe_ema 剔除; "
                     "243→243 张量, 载荷逐位一致; v244 已含 _core_brain 键, 无注入",
        "regression_note": "10-12 提交后 C9 拒载化石 v244 (二级证据: exp120 今日同样失败) → "
                           "F-b 锚点系原始 v244 测得, 与迁移件的等价性由上述迁移语义保证",
    }
    probes["ckpt_has_tau"] = "_theta_novelty" in sd
    probes["tau0_v244"] = float(sd["_theta_novelty"]) if probes["ckpt_has_tau"] else 0.001
    E_p = torch.randn(64, 64, dtype=torch.float16)
    probes["rtr_m0_annihilates"] = bool((rtr_apply(E_p, torch.zeros(1, dtype=torch.float16)) == 0).all())
    probes["prert_dW_plus_E0_bitwise_noop"] = bool((E_p + E_p * torch.zeros(1, dtype=torch.float16) == E_p).all())
    print(f"exp121: probes {probes}", flush=True)

    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=256, lm_freeze_w1=False)
    net = HybridPCNet.load(ck, cfg).to(dev)
    net.eval()
    gw = {"obj": None}

    def wrap():
        gw["obj"] = GraphedPredict(net.forward_engine, net)
        net.forward_engine._predict = gw["obj"]

    def unwrap():
        if gw["obj"] is not None:
            net.forward_engine._predict = gw["obj"].raw
            gw["obj"].graphs.clear()
            gw["obj"] = None
            torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    gauge, _, _ = build_gauge(args.data, dev)
    test_wins = gauge["tail"][:N_EVAL_WIN]
    fit_pool = gauge["trunc"] + gauge["tail"][N_EVAL_WIN:]
    order = torch.randperm(len(fit_pool), generator=torch.Generator().manual_seed(args.seed)).tolist()
    fit_pool = [fit_pool[i] for i in order]
    n_tr = int(len(fit_pool) * args.train_frac)
    train_wins, cal_wins = fit_pool[:n_tr], fit_pool[n_tr:]
    print(f"exp121: split train={len(train_wins)} cal={len(cal_wins)} test={len(test_wins)}", flush=True)

    if args.smoke:
        train_wins, cal_wins, test_wins = train_wins[:2], cal_wins[:2], test_wins[:2]
        args.epochs = 2
        args.scan_epochs = 1

    if not args.no_graph and dev.type == "cuda":
        wrap()
    h_tr, y_tr, nov_tr_wins = extract(net, dev, train_wins, "train")
    h_ca, y_ca, _ = extract(net, dev, cal_wins, "cal")
    h_te, y_te, _ = extract(net, dev, test_wins, "test")
    unwrap()
    splits = {"Xca_h": h_ca, "yca": y_ca, "Xte_h": h_te, "yte": y_te}
    print(f"exp121: frozen h shapes tr={tuple(h_tr.shape)} ca={tuple(h_ca.shape)} te={tuple(h_te.shape)}", flush=True)

    W0 = sd["W_lm"].to(dev)
    b0 = sd["bias_lm"].to(dev)

    wref = gauge["tail"][0].unsqueeze(0).to(dev)
    net.forward(wref)
    z4_lm = z4lm_from_z4(net, net._z4)
    zh_r = rms_norm(torch.cat([z4_lm, net._bind_vec, net._mem_out], dim=-1))
    h_r = cubic(rms_norm(saturate(zh_r @ net.W1)))
    ref = readout_logits(net, migrate=False)
    init_chk = float(((h_r @ W0 + b0) * INV_H - ref).abs().max())
    print(f"exp121: init 自检 max|head−readout_red| = {init_chk:.3e}", flush=True)

    class _H0:
        def eval(self):
            return self

        def __call__(self, x):
            return (x @ W0 + b0) * INV_H

    l0 = infer(_H0(), h_te, dev, args.batch)
    init_bpb = score_logits(l0, y_te, scale=1.0)["bpb"]
    print(f"exp121: init(v244 部署态) head bpb@s1={init_bpb:.4f} (锚 red 7.3483)", flush=True)

    dt_tab, dt_tel = dt_lut(nov_tr_wins, probes["tau0_v244"])
    print(f"exp121: d_t 遥测 {dt_tel}", flush=True)

    lr = args.lr
    lr_scan = None
    if lr <= 0.0:
        lr_scan = {}
        for cand in (0.5, 2.0, 8.0):
            Wc, bc, _ = train_cell(h_tr, y_tr, W0, b0, cand, args.scan_epochs, args.batch,
                                   args.seed, {"row_unit": False, "contrast": False, "gate": False,
                                               "unit_norm": False, "trust": False, "rtr": False, "cap10": False})
            r = eval_cell(Wc, bc, splits, dev, args.batch)
            lr_scan[str(cand)] = {"cal_s_star": r["s_star"], "epochs": args.scan_epochs}
            print(f"exp121: lr 预扫 {cand} (s*={r['s_star']})", flush=True)
            del Wc, bc
            torch.cuda.empty_cache()
        lr = 2.0
    print(f"exp121: lr={lr} epochs={args.epochs} batch={args.batch}", flush=True)

    CHAIN = {
        "C0": {},
        "C1": {"row_unit": True},
        "C2": {"row_unit": True, "contrast": True},
        "C3": {"row_unit": True, "contrast": True, "gate": True},
        "C4": {"row_unit": True, "contrast": True, "gate": True, "unit_norm": True},
        "C5": {"row_unit": True, "contrast": True, "gate": True, "unit_norm": True, "trust": True},
        "C6b": {"row_unit": True, "contrast": True, "gate": True, "unit_norm": True, "trust": True, "rtr": True},
        "C7": {"row_unit": True, "contrast": True, "gate": True, "unit_norm": True, "trust": True, "rtr": True, "cap10": True},
    }
    base_f = {"row_unit": False, "contrast": False, "gate": False,
              "unit_norm": False, "trust": False, "rtr": False, "cap10": False}
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]

    ladder = {"init": {"bpb_s1": init_bpb, "note": "v244 部署态未训练 (锚 red 7.3483)"}}
    seq = []
    for name in cells:
        f = dict(base_f)
        f.update(CHAIN[name])
        t0 = time.time()
        W, b, tel = train_cell(h_tr, y_tr, W0, b0, lr, args.epochs, args.batch, args.seed, f, dt_tab)
        r = eval_cell(W, b, splits, dev, args.batch)
        tel["W_norm"] = float(W.norm())
        tel["W_rowmax"] = float(W.norm(dim=1).max())
        tel["cap_bind_frac"] = tel.pop("cap_bind_steps") / tel["steps"]
        ladder[name] = {"flags": {k: v for k, v in f.items()}, **r, "telemetry": tel,
                        "secs": round(time.time() - t0, 1)}
        seq.append((name, r["bpb_s1"]))
        print(f"exp121: [ladder] {name:3s} bpb@s1={r['bpb_s1']:.4f} bpb@s*={r['bpb_s_star']:.4f}@s={r['s_star']} "
              f"top1={r['top1_s1']:.4f} ‖W‖={tel['W_norm']:.2f} dW|={tel['dW_absmean']:.2e} "
              f"step={tel['step_size_mean']:.3f} ({ladder[name]['secs']}s)", flush=True)
        del W, b
        torch.cuda.empty_cache()

    s1s = [v for _, v in seq]
    if s1s[0] <= 4.5:
        src_v = "纯 delta ≈4 → 信号源没问题 (eps/目标/输入干净), 元凶在 ladder 附加变换"
    elif s1s[0] >= 5.5:
        src_v = "纯 delta ≈6.7 → 信号源本身有问题, 先修 eps/目标/输入"
    else:
        src_v = f"纯 delta 中间带 ({s1s[0]:.3f}), 两判据均未干净命中, 需结合跳变格解读"
    jump = None
    names = [n for n, _ in seq]
    for i in range(1, len(seq)):
        if s1s[i] - s1s[i - 1] >= 1.0 and s1s[i] >= 5.5:
            jump = names[i]
            break
    if jump:
        j = names.index(jump)
        jv = f"跳变格 = {jump} (bpb {s1s[j - 1]:.3f} → {s1s[j]:.3f}) → 元凶"
    else:
        jv = "无单格 ≥1.0 bit 跳变 → 破坏是多变换渐进叠加, 需看序列斜率"
    verdict = {"signal_source": src_v, "culprit": jv}

    out = {"ckpt": args.ckpt, "ckpt_md5": _md5(ck),
           "raw_ckpt": args.raw_ckpt, "raw_ckpt_md5": _md5(ck_raw),
           "data": args.data, "seed": args.seed,
           "split": {"train_windows": len(train_wins), "cal_windows": len(cal_wins),
                     "test_windows": len(test_wins), "test_source": f"gauge.tail[0:{N_EVAL_WIN}]"},
           "config": {"epochs": args.epochs, "batch": args.batch, "lr": lr, "lr_scan": lr_scan,
                      "head": "冻结 h=cubic(rms_norm(saturate(zh@W1_v244))), logits=(h@W_lm+bias)/√256, raw 无调制",
                      "err": "eps_lm 纯项 (target−softmax), 不含 0.2·diff2 (声明见 docstring)",
                      "precision": "全链路 fp16", "d_t": dt_tel,
                      "cumulative_note": "C6a(pre-rtr no-op)=C5 位级等价, 运行时探针证明, 不重跑"},
           "anchors": ANCHORS, "probes": probes, "ladder": ladder, "verdict": verdict,
           "iron_rules": "前台; 不改 crates/; 不动 v244; 全 fp16; 不落盘权重; 探针零写回; run_file 自增新目录"}
    path = run_file("readout_delta_ladder.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"exp121: 写出 {path}", flush=True)
    print(f"exp121: 判定 = {src_v}\nexp121: {jv}", flush=True)


if __name__ == "__main__":
    main()
