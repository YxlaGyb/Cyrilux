"""115 阶段一探针 (GPU): W_diff 死锁验尸 + ε 统计.

A. W_diff 验尸 (复刻 114 对 W1 的流程): 8 个真实感知 learn 步,
   monkey-patch _apply_diff 记录计算出的 dW_avg (缩放前口径),
   对照落盘 ΔW_diff / b_diff 逐位变化, 并与 fp16 ULP 比较 —
   判决: 更新路径在算但被 fp16 量化吸收 (静默归零点), 还是量级够.
B. ε 统计 (114 终态前向 2000 窗, 零学习): future_err (W_diff 预测误差),
   eps4/eps2/eps3 (层误差), eps_lm (读出误差); 各自 mean/std,
   窗间 lag-1 自相关, 窗内逐位置 e_t lag-1 余弦 (基底漂移假说);
   另报 W_diff 对 dz4 的解释度 (‖pred_d‖/‖dz4‖, cos).

只读诊断: 内存副本跑, 不回写 checkpoint. monkey-patch 只在探针进程内.
用法: .venv/Scripts/python.exe scripts/probe115_wdiff_eps.py
输出: out/probe115_wdiff_eps.json
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
from model.dense.learning import temporal as temporal_mod
from dataset import DualChannelDataset

S_MAX = 256
dev = torch.device("cuda")
cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=S_MAX, lm_freeze_w1=False)


def ulp_of(t_f16):
    """fp16 张量逐条目 ULP: 2^(floor(log2|x|)-10)."""
    a = t_f16.float().abs()
    e = torch.floor(torch.log2(a + 1e-30))
    return torch.exp2(e - 10.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/exp115_say.pt")
    args = ap.parse_args()
    net = E.DensePCNet.load(args.ckpt, cfg).to(dev)
    a4 = net.active_size["l4"]
    print(f"checkpoint: {args.ckpt} a4={a4} "
          f"‖W_diff‖={net.W_diff.float().norm().item():.2f} "
          f"‖W_diff_elig‖={net.W_diff_elig.float().norm().item():.2e} "
          f"‖b_diff‖={net.b_diff.float().norm().item():.3e}", flush=True)

    # ── A. W_diff 验尸 ──
    recs = []
    orig_apply = temporal_mod.TemporalMixin._apply_diff

    def wrapped_apply(self, ctx, sh):
        if sh.diff is not None:
            d = sh.diff.dW_avg.float()
            recs.append({
                "dW_avg_norm": d.norm().item(),
                "dW_abs_median": d.abs().median().item(),
                "dW_abs_p95": d.abs().quantile(0.95).item(),
                "dW_abs_max": d.abs().max().item(),
                "eta": float(ctx.eta) if not torch.is_tensor(ctx.eta) else float(ctx.eta.item()),
            })
        return orig_apply(self, ctx, sh)

    temporal_mod.TemporalMixin._apply_diff = wrapped_apply
    ds = DualChannelDataset("dataset/pretrain_t2t_mini.jsonl", max_length=S_MAX,
                            max_samples=2000, lazy=False)
    w_pre = net.W_diff.data.clone()
    b_pre = net.b_diff.data.clone()
    elig_norm_pre = net.W_diff_elig.float().norm().item()
    for i in range(8):
        b, _ = ds[500 + i]
        x = b.unsqueeze(0).to(dev)
        net.learn(x)
    w_post = net.W_diff.data
    b_post = net.b_diff.data
    dw_disk = (w_post.float() - w_pre.float())
    db_disk = (b_post.float() - b_pre.float())
    # 逐位变化 (8 步累计)
    w_changed = (w_post != w_pre).float().mean().item()
    b_changed = (b_post != b_pre).float().mean().item()
    # ULP 对照 (步前权重)
    ulp = ulp_of(w_pre)
    upd_med = [r["dW_abs_median"] * r["eta"] for r in recs]
    upd_p95 = [r["dW_abs_p95"] * r["eta"] for r in recs]
    # 量化判决: 更新量 vs 半 ULP
    frac_above_halfulp = (ulp < 2.0 * max(upd_p95)).float().mean().item()
    aut = {
        "dW_avg_per_step": recs,
        "update_abs_median_x_eta": upd_med,
        "update_abs_p95_x_eta": upd_p95,
        "W_ulp_median": ulp.median().item(),
        "W_ulp_p05": ulp.quantile(0.05).item(),
        "W_abs_median": w_pre.float().abs().median().item(),
        "frac_entries_ulp_below_2x_update_p95": frac_above_halfulp,
        "W_diff_disk_delta_norm_8steps": dw_disk.norm().item(),
        "W_diff_disk_delta_rel": dw_disk.norm().item() / (w_pre.float().norm().item() + 1e-12),
        "W_diff_frac_entries_changed_8steps": w_changed,
        "W_diff_disk_abs_delta_max": dw_disk.abs().max().item(),
        "b_diff_disk_delta_norm_8steps": db_disk.norm().item(),
        "b_diff_frac_entries_changed_8steps": b_changed,
        "b_diff_disk_abs_delta_max": db_disk.abs().max().item(),
        "b_diff_abs_median": b_pre.float().abs().median().item(),
        "W_diff_elig_norm_pre": elig_norm_pre,
        "W_diff_elig_norm_post": net.W_diff_elig.float().norm().item(),
    }
    print(f"[A] dW_avg‖·‖ per step: {[round(r['dW_avg_norm'], 4) for r in recs]}", flush=True)
    print(f"[A] |dW|·eta: median={['%.2e' % v for v in upd_med]} "
          f"p95={['%.2e' % v for v in upd_p95]}", flush=True)
    print(f"[A] W ULP: median={aut['W_ulp_median']:.2e} |W| median={aut['W_abs_median']:.2e} "
          f"frac(ULP < 2·update_p95)={frac_above_halfulp:.4f}", flush=True)
    print(f"[A] 落盘 8 步: ΔW rel={aut['W_diff_disk_delta_rel']:.2e} "
          f"changed={w_changed*100:.2f}% max|ΔW|={aut['W_diff_disk_abs_delta_max']:.2e}", flush=True)
    print(f"[A] b_diff 8 步: Δ norm={aut['b_diff_disk_delta_norm_8steps']:.2e} "
          f"changed={b_changed*100:.2f}% |b| median={aut['b_diff_abs_median']:.2e}", flush=True)
    temporal_mod.TemporalMixin._apply_diff = orig_apply

    # ── B. ε 统计 (2000 窗, 零学习) ──
    gauge, _, _ = E.build_gauge("dataset/pretrain_t2t_mini.jsonl", dev)
    wins = list(gauge["tail"][:1000]) + list(gauge["trunc"][:1000])
    tags = ["tail"] * min(1000, len(gauge["tail"])) + ["trunc"] * (len(wins) - min(1000, len(gauge["tail"])))
    eps_stats = {k: {"future": [], "eps4": [], "eps2": [], "eps3": [], "eps_lm": []}
                 for k in ("tail", "trunc")}
    e_t_lag1_cos, pred_d_over_dz4, cos_pd_dz = [], [], []
    W_d = net.W_diff[:a4, :a4]
    inv_h = 1.0 / math.sqrt(net.d_h)
    for w, tg in zip(wins, tags):
        x = w.unsqueeze(0).to(dev)
        net.forward(x)
        z4, z2, z3, z0 = net._z4, net._z2, net._z3, net._z0
        # diff 窗误差 (temporal.py 同式, 静态前向)
        dz4 = z4[:, 1:] - z4[:, :-1]
        dz4_n = dz4 / (dz4.norm(dim=-1, keepdim=True) + 1e-3)
        z4r = z4
        S_full = z4r.shape[1]
        preds_k = {}
        for k in (2, 4, 8):
            k_eff = min(k, S_full - 1)
            z_shift = torch.cat([torch.zeros(1, k_eff, a4, dtype=z4r.dtype, device=dev), z4r[:, :-k_eff]], dim=1)
            z_shift_n = z_shift / (z_shift.norm(dim=-1, keepdim=True) + 1e-3)
            preds_k[k] = z_shift_n @ W_d.T + net.b_diff[:a4]
        w2, w4, w8 = (float(v) for v in net._w_soft[:3])
        pred_d = (w2 * preds_k[2][:, :-1] + w4 * preds_k[4][:, :-1] + w8 * preds_k[8][:, :-1])
        valid = torch.arange(S_full - 1, device=dev) >= 7
        e_t_all = (dz4 - pred_d).float()
        fut = (e_t_all[:, valid]).square().mean().item()
        # 层误差
        eps4 = (z4 - (z0 @ net.W_04[:a4].T + net.bias_l4[:a4])).float().square().mean().item()
        eps2 = (z2 - (z4 @ net.W_42[:net.active_size['l2']].T + net.bias_l2[:net.active_size['l2']])).float().square().mean().item()
        eps3 = (z3 - (z2 @ net.W_23[:net.active_size['l3']].T + net.bias_l3[:net.active_size['l3']])).float().square().mean().item()
        # eps_lm (读出误差, tmp_rank_probe 同式)
        z4_n_ = z4 / (z4.norm(dim=-1, keepdim=True) + 1e-3)
        pred_delta = z4_n_ @ W_d.T + net.b_diff[:a4].unsqueeze(0).unsqueeze(0)
        z4r2 = z4 + pred_delta
        z4_lm = z4r2 / (1.0 + z4r2.abs())
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
        mask_print = torch.zeros(256, dtype=torch.float16, device=dev)
        mask_print[32:] = 1.0
        logits_m = lc + (1.0 - mask_print) * -1e4
        probs = torch.softmax(logits_m.float(), dim=-1)
        tgt = x[0, 1:]
        pos = torch.arange(255, device=dev)
        p_t = probs[0, pos, tgt]
        eps_lm = float((1.0 - p_t).mean().item())
        # 窗内 e_t lag-1 余弦 (基底漂移: 相邻位置误差方向是否同向)
        ev = e_t_all[0, 1:]  # [S-1-1, a4] float
        ev_prev = e_t_all[0, :-1]
        cos = torch.nn.functional.cosine_similarity(ev, ev_prev, dim=-1)
        e_t_lag1_cos.append(cos[valid[1:]].mean().item())
        # W_diff 解释度
        pd = pred_d.float()
        dzn = dz4.float()
        pred_d_over_dz4.append((pd.norm() / (dzn.norm() + 1e-9)).item())
        cos_pd = torch.nn.functional.cosine_similarity(
            pd.reshape(-1, a4), dzn.reshape(-1, a4), dim=-1)
        cos_pd_dz.append(cos_pd[valid].mean().item())
        for key, v in (("future", fut), ("eps4", eps4), ("eps2", eps2),
                       ("eps3", eps3), ("eps_lm", eps_lm)):
            eps_stats[tg][key].append(v)
        for k_ in ("_z0", "_z4", "_z2", "_z3", "_z5", "_z6"):
            if hasattr(net, k_):
                delattr(net, k_)

    def acorr1(xs):
        xs = xs[1:] if xs and xs[0] == 0 else xs
        if len(xs) < 3:
            return None
        t = torch.tensor(xs)
        t = t - t.mean()
        den = (t * t).sum()
        if den == 0:
            return None
        return ((t[:-1] * t[1:]).sum() / den).item()

    res_eps = {}
    for tg, d in eps_stats.items():
        res_eps[tg] = {}
        for k, xs in d.items():
            t = torch.tensor(xs)
            res_eps[tg][k] = {"mean": t.mean().item(), "std": t.std().item(),
                              "lag1_autocorr": acorr1(xs), "n": len(xs)}
        print(f"[B] {tg}: " + " ".join(
            f"{k}={v['mean']:.4f}±{v['std']:.4f}(ac1={v['lag1_autocorr']:.3f})"
            for k, v in res_eps[tg].items()), flush=True)
    res_eps["e_t_lag1_cos_mean"] = sum(e_t_lag1_cos) / len(e_t_lag1_cos)
    res_eps["pred_d_norm_over_dz4_mean"] = sum(pred_d_over_dz4) / len(pred_d_over_dz4)
    res_eps["cos_pred_d_dz4_mean"] = sum(cos_pd_dz) / len(cos_pd_dz)
    print(f"[B] 窗内 e_t lag-1 cos={res_eps['e_t_lag1_cos_mean']:.4f} | "
          f"‖pred_d‖/‖dz4‖={res_eps['pred_d_norm_over_dz4_mean']:.4f} "
          f"cos(pred_d,dz4)={res_eps['cos_pred_d_dz4_mean']:.4f}", flush=True)

    out = {"wdiff_autopsy": aut, "eps_stats": res_eps}
    with open("out/probe115_wdiff_eps.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved -> out/probe115_wdiff_eps.json")


if __name__ == "__main__":
    main()
