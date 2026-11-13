"""
F-e: 读出头规则修复 harness 验证 (E1/E1b/E2dep/E2/E3). 内存按需制: zh 磁盘 memmap 后备,
训练每步只取一个 batch (~46MB) 进内存/GPU, 常驻内存 <2GB; 缓存落 run 目录, 跑完即删.

F-d 判决: 元凶 = dW 全局单位化 (readout.py L408-410), 信任域 ∝‖W‖ 是次生爆炸源,
cap10 是承重墙; d_t 门/行单位化/contrast_w 各 <0.2 bit 无罪. W1 静态审计 (本轮):
L443-444 /max/64 预缩放 + L445-447 全局单位化 + L449 rtr (输出范数≡m, 第二台单位化器)
+ L452 信任域 ∝‖W‖ (被 L453 soft_norm_preserve 行范数 ~1 钉住, 爆炸不可达, 改为恒定 ~1.3 步长)
+ W_lm_2 同病 (L468-474, 本轮不修, 披露).

cells (全在迁移件 v244 上, 冻结 dense 引擎):
  E1    W_lm 修复链: 行单位化+contrast+d_t+固定步长(lr=2.0, 同 C0)+cap10; 删单位化/rtr/信任域
        判据: bpb@s1 ∈ [4.2, 4.8]
  E1b   E1 + rtr(m≡1)  — 检验 "rtr 是第二台单位化器" 后门假设 (观察格)
  E2dep W1 部署链照实复刻 (/max/64+单位化+rtr+信任域+soft_norm), W_lm+bias 冻结 (对照格)
  E2    W1 修复链: e_h 转置传播+h_deriv+/max/64(铁律保留)+固定步长(cal 预扫)+soft_norm;
        删单位化/rtr/信任域; W_lm+bias 冻结. 判据: bpb@s1 < 5.0 (反超 only_W1 锚 6.357)
  E3    W_lm(修复链)+W1(修复链) 同步联合更新 (部署序: 同批 err, e_h 用旧 W_lm).
        判据: bpb@s1 < 3.5 (逼近 both 锚 2.573)

口径声明:
  - zh 冻结近似: dense 引擎冻结, W1 更新不回灌引擎 (与 F-d 冻结 h 同类近似, 逐窗前向一次)
  - E1/E1b 不再预存 h: W1 冻结 → h 每批由 zh 现算 (与 F-d 预存 h 数值等价, init 自检佐证)
  - E2/E3 的 W1 误差用部署口径 err_scaled (行单位化, 不含 contrast/d_t — readout.py L437 照实)
  - E3 的 W_lm 分支含 contrast/d_t (L402 照实); bias 用 C0 式 db·lr (与 E1 一致)
  - h_in = rms_norm(saturate(zh@W1)), h = cubic(h_in), h_deriv = 1-1.5·h_in² (L63-69 照实)
  - raw logits 无能量调制/打印掩码 (F-b/exp118/exp121 同口径); eps = eps_lm 纯项
  - dW 在 batch 内 mean over 窗数、不除位置数 (部署 .mean(dim=0) 照实, 与 exp121 /idx.numel() 同)
铁律: 前台; 不改 crates/; 不动 v244 (迁移件只读); 全 fp16; 不落盘权重 (zh 激活缓存非权重, 即用即删);
      run_file 自增新目录.
"""
import argparse
import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from exp115_say import N_EVAL_WIN, GraphedPredict, build_gauge
from exp116_z4_probe import _md5, calib_s, infer, readout_logits, score_logits, z4lm_from_z4
from safetensors.torch import load_file

from crates.cli.utils import resolve_path, run_file, run_dir
from crates.kernels import cubic, rms_norm, saturate
from crates.kernels.normalize import soft_norm_preserve
from crates.model import CyreneModel, HybridPCNet
from crates.model.constants import LM_TRUST_REGION
from crates.rtr import rtr_apply

torch.set_grad_enabled(False)

INV_H = 1.0 / math.sqrt(256)
ELIG_GAMMA = 0.95
CRITERIA = {
    "E1nc": None,
    "E1": ("band", 4.2, 4.8),
    "E1b": None,
    "E2dep": None,
    "E2": ("lt", 5.0),
    "E3": ("lt", 3.5),
}
ANCHORS = {
    "backprop_only_W_lm": {"v": 3.984, "src": "exp120 F-b (s1 口径; 2026-10-11 冻结, 当前 HEAD 不可复跑, 见 SESSION_HANDOFF #22)"},
    "backprop_both": {"v": 2.573, "src": "exp120 F-b (s1 口径; 2026-10-11 冻结, 当前 HEAD 不可复跑, 见 SESSION_HANDOFF #22)"},
    "hebbian_deployed_mig": {"v": 6.7324, "src": "exp117 v247 head_decomp.json"},
    "v244_head_red_init": {"v": 7.3483, "src": "exp117 (rtr 前部署链起点)"},
    "unigram_T": {"v": 4.7677, "src": "exp117 v247"},
}


def head_pre(zhb, W1):
    a = saturate(zhb @ W1)
    h_in = rms_norm(a)
    return cubic(h_in), 1.0 - 1.5 * h_in.pow(2)


def row_unit(e):
    en = e.norm(dim=-1, keepdim=True) * 1.01
    alive = (en > 1e-8).to(e.dtype)
    return e * alive / torch.where(alive > 0, en, torch.ones_like(en))


def dt_lut(nov_flat, tau0):
    rows = nov_flat.view(-1, 255)
    tau = float(tau0)
    gates, taus = [], []
    for i in range(rows.shape[0]):
        nov = rows[i]
        tau = tau * 0.99 + 0.01 * float(nov.float().mean())
        g = torch.sigmoid((nov.float() - tau) * 500.0).to(torch.float16)
        gates.append(g)
        taus.append(tau)
    lut = torch.cat(gates, 0)
    flat = nov_flat.float()
    return lut, {
        "tau0": float(tau0), "tau_end": taus[-1],
        "gate_mean": float(lut.float().mean()), "gate_lt_p5_frac": float((lut < 0.5).float().mean()),
        "nov_p50": float(torch.quantile(flat, 0.5)),
        "nov_p95": float(torch.quantile(flat, 0.95)),
    }


class ZhStore:
    def __init__(self, run_d, name, n, s1, d):
        self.path = os.path.join(run_d, f"zh_{name}.f16.cache")
        self.n, self.s1, self.d = n, s1, d
        self.t = None
        self.mm = np.memmap(self.path, dtype=np.float16, mode="w+", shape=(n, s1, d))

    def fill(self, i, zh_cpu):
        self.mm[i] = zh_cpu.numpy()

    def flush(self):
        self.mm.flush()
        self.mm._mmap.close()

    def view(self):
        if self.t is None:
            self.mm = np.memmap(self.path, dtype=np.float16, mode="r+", shape=(self.n, self.s1, self.d))
            self.t = torch.from_numpy(self.mm).view(self.n * self.s1, self.d)
        return self.t

    def drop(self):
        self.t = None
        del self.mm
        self.mm = None
        try:
            os.remove(self.path)
        except OSError:
            gc.collect()
            try:
                os.remove(self.path)
            except OSError as e:
                print(f"exp122: WARN 缓存文件未删 {self.path}: {e}", flush=True)


def extract_zh_stream(net, dev, wins, tag, store):
    ys, novs = [], []
    t0 = time.time()
    s1 = None
    for i, w in enumerate(wins):
        x = w.unsqueeze(0).to(dev)
        net.forward(x)
        z4_lm = z4lm_from_z4(net, net._z4)
        zh = rms_norm(torch.cat([z4_lm, net._bind_vec, net._mem_out], dim=-1))
        if s1 is None:
            s1 = zh.shape[1] - 1
            assert store.s1 == s1, f"zh 位置数 {s1} ≠ 预分配 {store.s1}"
        store.fill(i, zh[0, :-1].cpu())
        ys.append(x[0, 1:].to("cpu"))
        novs.append(net._novelty[0, :-1].to("cpu", torch.float16))
        if (i + 1) % 1000 == 0:
            print(f"  [{tag}] extract {i + 1}/{len(wins)} ({time.time() - t0:.0f}s)", flush=True)
    store.flush()
    return torch.cat(ys, 0).long(), novs


def train_cell(zh_tr, y_tr, W1_0, Wlm0, b0, lr_w1, lr_lm, epochs, batch, seed,
               w1_flags, lm_flags, dt_tab=None):
    """统一训练格. w1_flags=None → W1 冻结 (E1/E1b); lm_flags=None → W_lm/bias 冻结 (E2/E2dep)."""
    dev = Wlm0.device
    W1 = W1_0.clone() if w1_flags is not None else W1_0
    Wlm = Wlm0.clone()
    b = b0.clone()
    E1 = torch.zeros_like(W1) if w1_flags is not None else None
    E2 = torch.zeros_like(Wlm)
    n = zh_tr.shape[0]
    g = torch.Generator().manual_seed(seed)
    steps = 0
    dW1_sum = torch.zeros(1, device=dev, dtype=torch.float32)
    dWlm_sum = torch.zeros(1, device=dev, dtype=torch.float32)
    cap_sum = torch.zeros(1, device=dev, dtype=torch.float32)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            zhb = zh_tr[idx].to(dev, torch.float16)
            yb = y_tr[idx].to(dev)
            h, hder = head_pre(zhb, W1)
            logits = (h @ Wlm + b) * INV_H
            err = F.one_hot(yb, 256).to(torch.float16) - torch.softmax(logits, dim=-1)
            eu = row_unit(err)
            if w1_flags is not None:
                e_h = (eu @ Wlm.T) * hder
                dW1 = (rms_norm(zhb).transpose(-2, -1) @ rms_norm(e_h)).mean(dim=0)
                dW1 = dW1 / (dW1.abs().max() + 1e-4) / 64.0
                if w1_flags["unit_norm"]:
                    nrm = dW1.norm()
                    alive = (nrm > 0).to(dW1.dtype)
                    dW1 = dW1 * alive / torch.where(alive > 0, nrm, torch.ones_like(nrm))
                if w1_flags["rtr"]:
                    E1.mul_(ELIG_GAMMA).add_(dW1)
                    dW1 = rtr_apply(E1, torch.ones(1, device=dev, dtype=torch.float16))
                if w1_flags["trust"]:
                    W1 += dW1 * (W1.norm() * LM_TRUST_REGION)
                else:
                    W1 += dW1 * lr_w1
                soft_norm_preserve(W1)
                dW1_sum += dW1.abs().mean().to(torch.float32)
            if lm_flags is not None:
                mult = torch.ones_like(eu)
                if lm_flags["contrast"]:
                    target_oh = F.one_hot(yb, 256).to(torch.float16)
                    logits_d = (h @ Wlm) * INV_H
                    tgt = (logits_d * target_oh).sum(dim=-1, keepdim=True)
                    cw = 1.0 / (1.0 + (logits_d - tgt).abs() * 0.1)
                    mult = mult * (cw * (1.0 - target_oh) + target_oh)
                if lm_flags["gate"]:
                    mult = mult * dt_tab[idx].to(dev).unsqueeze(-1)
                dWlm = (h.transpose(0, 1) @ (eu * mult)) / idx.numel()
                if lm_flags["rtr"]:
                    E2.mul_(ELIG_GAMMA).add_(dWlm)
                    dWlm = rtr_apply(E2, torch.ones(1, device=dev, dtype=torch.float16))
                Wlm += dWlm * lr_lm
                b += (eu * mult).mean(dim=0) * lr_lm
                if lm_flags["cap10"]:
                    rn = Wlm.norm(dim=1)
                    mx = rn.max()
                    f = torch.minimum(torch.ones_like(mx), 10.0 / (mx + 1e-6)).to(torch.float16)
                    Wlm *= f
                    cap_sum += (f < 1.0).to(torch.float32)
                dWlm_sum += dWlm.abs().mean().to(torch.float32)
            steps += 1
    tel = {"steps": steps}
    if w1_flags is not None:
        tel["dW1_absmean"] = float(dW1_sum.item()) / steps
    if lm_flags is not None:
        tel["dWlm_absmean"] = float(dWlm_sum.item()) / steps
        tel["step_size_mean"] = lr_lm
        tel["cap_bind_frac"] = float(cap_sum.item()) / steps
        tel["W_norm"] = float(Wlm.norm())
    return W1, Wlm, b, tel


def eval_w1_cell(W1, Wlm, b, zh_ca, y_ca, zh_te, y_te, dev, batch):
    class _H:
        def eval(self):
            return self

        def __call__(self, x):
            return (head_pre(x, W1)[0] @ Wlm + b) * INV_H

    l_te = infer(_H(), zh_te, dev, batch)
    l_ca = infer(_H(), zh_ca, dev, batch)
    s_star, per_s = calib_s(l_ca, y_ca)
    te1 = score_logits(l_te, y_te, scale=1.0)
    tes = score_logits(l_te, y_te, scale=s_star)
    return {"bpb_s1": te1["bpb"], "top1_s1": te1["top1"], "top15_s1": te1["top15"],
            "bpb_s_star": tes["bpb"], "s_star": s_star, "cal_bpb_per_s": per_s}


def w1_stats(W):
    rf = W.norm(dim=1).float().cpu()
    return {"W1_norm": float(W.norm()), "row_p50": float(torch.quantile(rf, 0.5)),
            "row_p95": float(torch.quantile(rf, 0.95))}


def verdict_for(name, bpb):
    c = CRITERIA[name]
    if c is None:
        return "观察格 (无判据)"
    if math.isnan(bpb):
        return "FAIL (NaN)"
    if c[0] == "band":
        ok = c[1] <= bpb <= c[2]
        return f"{'PASS' if ok else 'FAIL'} (判据 {c[1]}–{c[2]})"
    ok = bpb < c[1]
    return f"{'PASS' if ok else 'FAIL'} (判据 <{c[1]})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/v254-20261012-205835/migrated/e9f_bplus_readout_realign_500.safetensors")
    ap.add_argument("--raw-ckpt", default="out/v244-20261009-214217/e9f_bplus_readout_realign_500.safetensors")
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.85)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr-lm", type=float, default=2.0)
    ap.add_argument("--lr-w1", type=float, default=0.0, help="0 = cal 预扫 {0.125,0.5,2,8} 选优 (仅 W1 格)")
    ap.add_argument("--scan-epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--cells", default="E1,E1b,E2dep,E2,E3")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    dev = torch.device(args.device)
    ck = resolve_path(args.ckpt)
    sd = load_file(ck)
    run_d = run_dir()
    print(f"exp122: ckpt={ck} md5={_md5(ck)}\nexp122: run_dir={run_d}", flush=True)
    print(f"exp122: 谱系源 v244 (safetensors, 只读) md5={_md5(resolve_path(args.raw_ckpt))}", flush=True)

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
    print(f"exp122: split train={len(train_wins)} cal={len(cal_wins)} test={len(test_wins)}", flush=True)

    if args.smoke:
        train_wins, cal_wins, test_wins = train_wins[:8], cal_wins[:4], test_wins[:4]
        args.epochs = 2
        args.scan_epochs = 1

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    lm_dim = int(sd["W1"].shape[0])
    cache_bytes = 0
    stores = {}
    try:
        if not args.no_graph and dev.type == "cuda":
            wrap()
        for name, wins in (("train", train_wins), ("cal", cal_wins), ("test", test_wins)):
            stores[name] = ZhStore(run_d, name, len(wins), 255, lm_dim)
        y_tr, nov_wins = extract_zh_stream(net, dev, train_wins, "train", stores["train"])
        y_ca, _ = extract_zh_stream(net, dev, cal_wins, "cal", stores["cal"])
        y_te, _ = extract_zh_stream(net, dev, test_wins, "test", stores["test"])
        unwrap()
        nov_flat = torch.cat(nov_wins, 0)
        for st in stores.values():
            cache_bytes += os.path.getsize(st.path)
        print(f"exp122: zh 缓存 (磁盘 memmap, 按需分页) {cache_bytes / 1e9:.2f} GB @ {run_d}", flush=True)

        W1_0 = sd["W1"].to(dev)
        W0 = sd["W_lm"].to(dev)
        b0 = sd["bias_lm"].to(dev)

        wref = gauge["tail"][0].unsqueeze(0).to(dev)
        net.forward(wref)
        z4_lm = z4lm_from_z4(net, net._z4)
        zh_r = rms_norm(torch.cat([z4_lm, net._bind_vec, net._mem_out], dim=-1))
        h_r, _ = head_pre(zh_r, W1_0)
        ref = readout_logits(net, migrate=False)
        init_chk = float(((h_r @ W0 + b0) * INV_H - ref).abs().max())
        print(f"exp122: init 自检 max|head−readout_red| = {init_chk:.3e}", flush=True)

        zh_te, zh_ca = stores["test"].view(), stores["cal"].view()
        l0 = score_logits(infer(_RefHead0(W1_0, W0, b0), zh_te, dev, args.batch), y_te, scale=1.0)
        init_bpb = l0["bpb"]
        print(f"exp122: init(v244 部署态) bpb@s1={init_bpb:.4f}", flush=True)

        dt_tab, dt_tel = dt_lut(nov_flat, float(sd["_theta_novelty"]))
        print(f"exp122: d_t 遥测 {dt_tel}", flush=True)

        scan_only = len(cells) == 0
        need_w1 = scan_only or any(c in ("E2", "E3") for c in cells)
        lr_w1 = args.lr_w1
        lr_scan = None
        if lr_w1 <= 0.0 and not need_w1:
            lr_w1 = 2.0
        elif lr_w1 <= 0.0:
            lr_scan = {}
            zh_tr_v = stores["train"].view()
            for cand in (0.125, 0.5, 2.0, 8.0):
                W1c, _, _, _ = train_cell(zh_tr_v, y_tr, W1_0, W0, b0, cand, args.lr_lm,
                                          args.scan_epochs, args.batch, args.seed,
                                          {"unit_norm": False, "rtr": False, "trust": False}, None)
                r = eval_w1_cell(W1c, W0, b0, zh_ca, y_ca, zh_te, y_te, dev, args.batch)
                lr_scan[str(cand)] = {"cal_bpb_s1": r["bpb_s1"]}
                print(f"exp122: lr_w1 预扫 {cand} → cal bpb@s1={r['bpb_s1']:.4f}", flush=True)
                del W1c
                torch.cuda.empty_cache()
            lr_w1 = float(min(lr_scan, key=lambda k: lr_scan[k]["cal_bpb_s1"]))
        print(f"exp122: lr_w1={lr_w1} lr_lm={args.lr_lm} epochs={args.epochs} batch={args.batch}", flush=True)

        W1_CHAIN = {
            "E1nc": None,
            "E1": None,
            "E1b": None,
            "E2dep": {"unit_norm": True, "rtr": True, "trust": True},
            "E2": {"unit_norm": False, "rtr": False, "trust": False},
            "E3": {"unit_norm": False, "rtr": False, "trust": False},
        }
        LM_CHAIN = {
            "E1nc": {"contrast": True, "gate": True, "rtr": False, "cap10": False},
            "E1": {"contrast": True, "gate": True, "rtr": False, "cap10": True},
            "E1b": {"contrast": True, "gate": True, "rtr": True, "cap10": True},
            "E3": {"contrast": True, "gate": True, "rtr": False, "cap10": True},
        }
        zh_tr_v = stores["train"].view()
        results = {"init": {"bpb_s1": init_bpb, "W1": w1_stats(W1_0)}}
        for name in cells:
            t0 = time.time()
            wf, lf = W1_CHAIN[name], LM_CHAIN.get(name)
            W1c, Wlmc, bc, tel = train_cell(zh_tr_v, y_tr, W1_0, W0, b0, lr_w1, args.lr_lm,
                                            args.epochs, args.batch, args.seed, wf, lf, dt_tab)
            r = eval_w1_cell(W1c, Wlmc, bc, zh_ca, y_ca, zh_te, y_te, dev, args.batch)
            res = {"flags_w1": wf, "flags_lm": lf, **r, "telemetry": tel}
            if wf is not None:
                res["W1_end"] = w1_stats(W1c)
            res["secs"] = round(time.time() - t0, 1)
            res["verdict"] = verdict_for(name, r["bpb_s1"])
            results[name] = res
            print(f"exp122: [{name}] bpb@s1={r['bpb_s1']:.4f} bpb@s*={r['bpb_s_star']:.4f}@s={r['s_star']} "
                  f"top1={r['top1_s1']:.4f} → {res['verdict']} ({res['secs']}s)", flush=True)
            if wf is not None:
                print(f"exp122: [{name}] {res['W1_end']} dW1|={tel['dW1_absmean']:.2e}", flush=True)
            del W1c, Wlmc, bc
            torch.cuda.empty_cache()

        out = {"ckpt": args.ckpt, "ckpt_md5": _md5(ck),
               "raw_ckpt": args.raw_ckpt, "raw_ckpt_md5": _md5(resolve_path(args.raw_ckpt)),
               "data": args.data, "seed": args.seed,
               "split": {"train_windows": len(train_wins), "cal_windows": len(cal_wins),
                         "test_windows": len(test_wins), "test_source": f"gauge.tail[0:{N_EVAL_WIN}]"},
               "config": {"epochs": args.epochs, "batch": args.batch, "lr_lm": args.lr_lm,
                          "lr_w1": lr_w1, "lr_w1_scan": lr_scan,
                          "head": "h=cubic(rms_norm(saturate(zh@W1))), logits=(h@W_lm+bias)/√256, raw 无调制; zh 冻结",
                          "err_w1": "部署口径 err_scaled (行单位化, 不含 contrast/d_t, L437 照实)",
                          "memory": f"zh 磁盘 memmap {cache_bytes / 1e9:.2f} GB @ 本次 run 目录, 按需分页, "
                                    f"跑完即删, 常驻 <2GB, 无跨进程状态",
                          "precision": "全链路 fp16", "d_t": dt_tel,
                          "init_check": f"max|head−readout_red|={init_chk:.3e}"},
               "anchors": ANCHORS, "criteria": {k: v for k, v in CRITERIA.items()},
               "results": results,
               "iron_rules": "前台; 不改 crates/; 不动 v244; 全 fp16; 不落盘权重; 内存按需制 (zh 磁盘缓存即用即删); run_file 自增新目录"}
        path = run_file("fe_repair_harness.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"exp122: 写出 {path}", flush=True)
    finally:
        zh_tr_v = zh_ca = zh_te = None
        gc.collect()
        for st in stores.values():
            if st.mm is not None:
                st.drop()
        print("exp122: zh 缓存清理完成", flush=True)


class _RefHead0:
    def __init__(self, W1, Wlm, b):
        self.W1, self.Wlm, self.b = W1, Wlm, b

    def eval(self):
        return self

    def __call__(self, x):
        return (head_pre(x, self.W1)[0] @ self.Wlm + self.b) * INV_H


if __name__ == "__main__":
    main()
