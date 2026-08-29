"""probe111: 世界语言物理维度分离度测量 (只读, 零改权重, 先测量门).

111 门 (预注册于 .trae/documents/111_世界语言物理_恢复选择压.md):
  G1 汤区 (τ∈[8,12]) 的 q̄ vs 语料基线 q 分离 ≥ 3 倍
  G2 存在 τ 使 q̄(τ) > q̄(τ=10)·1.5 (更优温度区存在, q(τ) 非钉死)
过门 → exp111 世界物理接入; 不过 → 停, 报告, 重新设计评分维度.

同时测 ε(τ) (probe110 p_target_series 同口径, 跳过种子段) — E 认证
带锚的预期锚值范围 (q 高区的 ε 是多少, 检验认证带与感知带分离性).

口径: 生成与 audit_char_coverage B 组一致 (exp110c_world.pt, 种子=语料
尾 16B, continuation rep_backstop=False, use_w_act 不设 = 无意图注入);
评分用 world_lang.WorldLangPhysics (X 维度无时序历史语义, 报 q0=(L+S)/2).

用法: uv run python scripts/probe111_world_dims.py [--smoke]
"""
import argparse
import importlib.util
import json
import sys
import time

import torch

torch.set_grad_enabled(False)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from model.dense import DensePCConfig, DensePCNet
from world_lang import WorldLangPhysics

_spec = importlib.util.spec_from_file_location("probe110", "scripts/probe110_langnoise.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["probe110"] = _mod
_spec.loader.exec_module(_mod)
p_target_series = _mod.p_target_series
snapshot_state = _mod.snapshot_state
restore_state = _mod.restore_state
read_lines = _mod.read_lines
DEV = _mod.DEV
DATA = _mod.DATA

CKPT = "out/exp110c_world.pt"
TEMPS = [0.0, 1.0, 2.0, 4.0, 8.0, 10.0, 12.0, 16.0]
N_SEQ = 8
N_GEN = 63


def gen_stream(net, seed_bytes, temp):
    seed = torch.tensor([list(seed_bytes)], dtype=torch.long, device=DEV)
    out = net.forward_engine.continuation(seed, N_GEN, temperature=temp, rep_backstop=False)
    return bytes(int(v) for v in out[0, len(seed_bytes):].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="out/probe111_world_dims.json")
    ap.add_argument("--n-trigram", type=int, default=None,
                    help="3-gram 位图构建行数 (默认全量; smoke 用 5 万)")
    args = ap.parse_args()

    torch.manual_seed(123)
    n_seq = 2 if args.smoke else N_SEQ

    t0 = time.time()
    print("probe111: 构建世界语言物理 (常用字 2 万行 + 3-gram 位图)...", flush=True)
    world = WorldLangPhysics(DATA, n_trigram_lines=args.n_trigram or (50000 if args.smoke else None))
    print(f"  常用字={len(world.common_set)} 3-gram 数={world.n_trigrams} "
          f"q_base(heldout)={world.q_base:.4f} q_ref={world.q_ref:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    texts = read_lines(DATA, 5000, 100, 160)
    seeds = [t.encode("utf-8")[:16] for t in texts[:n_seq]]

    cfg = DensePCConfig(d_input=256, d_act=256, max_seq_len=256)
    net = DensePCNet.load(CKPT, cfg).to(DEV)
    net._entropy_sample = False
    snap = snapshot_state(net)

    # 语料基线 (同 audit C 口径的行, 世界评分维度)
    base_scores = [world.score(t.encode("utf-8")[:160]) for t in texts[:n_seq]]
    base = {k: sum(s[k] for s in base_scores) / len(base_scores) for k in ("L", "S", "q")}

    report = {"ckpt": CKPT, "temps": {}, "corpus_baseline": base,
              "q_base_heldout": world.q_base, "q_ref": world.q_ref,
              "n_trigrams": world.n_trigrams}
    print(f"\n语料基线: L={base['L']:.3f} S={base['S']:.3f} q={base['q']:.3f}", flush=True)
    print(f"\n{'τ':>6} {'L̄':>7} {'S̄':>7} {'X̄':>7} {'q̄':>7} {'ε̄(gen)':>9} {'std':>7}", flush=True)

    for T in TEMPS:
        world._hist.clear()  # 每温度冷启动新颖度历史 (时序语义: 连续发声的稳态)
        Ls, Ss, Xs, qs, eps = [], [], [], [], []
        for i in range(n_seq):
            restore_state(net, snap)
            stream = gen_stream(net, seeds[i], T)
            restore_state(net, snap)
            ids = torch.tensor([list(seeds[i]) + list(stream)], dtype=torch.long, device=DEV)
            p = p_target_series(net, ids)
            restore_state(net, snap)
            pl = [float(v) for v in p.tolist()]
            eps.append(1.0 - sum(pl[15:]) / max(1, len(pl[15:])))
            sc = world.score(stream)  # 含 X: 与本温度前序发声的重叠
            world.record(stream)
            Ls.append(sc["L"])
            Ss.append(sc["S"])
            Xs.append(sc["X"])
            qs.append(sc["q"])
        warm = slice(max(0, n_seq // 2), n_seq)  # 预热后样本 (历史已积累)
        m = {k: sum(v[warm]) / max(1, len(v[warm]))
             for k, v in (("L", Ls), ("S", Ss), ("X", Xs), ("q", qs))}
        eps_all = sum(eps) / len(eps)
        sd = (sum((e - eps_all) ** 2 for e in eps) / max(1, len(eps) - 1)) ** 0.5
        report["temps"][str(T)] = {"L": m["L"], "S": m["S"], "X": m["X"], "q": m["q"],
                                   "eps": eps_all, "eps_std": sd}
        print(f"{T:>6.1f} {m['L']:>7.3f} {m['S']:>7.3f} {m['X']:>7.3f} {m['q']:>7.3f} "
              f"{eps_all:>9.4f} {sd:>7.4f}", flush=True)

    # ── 门结算 (预注册) ──
    if not args.smoke:
        q_soup = report["temps"]["10.0"]["q"]
        q_best_tau = max(report["temps"], key=lambda t: report["temps"][t]["q"])
        q_best = report["temps"][q_best_tau]["q"]
        g1 = q_soup > 0 and base["q"] / q_soup >= 3.0
        g2 = q_best > 1.5 * q_soup
        eps_best = report["temps"][q_best_tau]["eps"]
        report["gate"] = {
            "q_soup_tau10": q_soup, "q_corpus": base["q"],
            "sep_ratio": base["q"] / q_soup if q_soup > 0 else float("inf"),
            "G1_sep_3x": g1,
            "best_tau": q_best_tau, "q_best": q_best,
            "G2_better_zone": g2, "eps_at_best_tau": eps_best,
            "pass": g1 and g2,
        }
        print(f"\n[probe111] G1 分离 {base['q']:.3f}/{q_soup:.3f}="
              f"{base['q']/q_soup:.1f}x (≥3x: {'过' if g1 else '不过'}) | "
              f"G2 最优 τ={q_best_tau} q={q_best:.3f} vs 汤 {q_soup:.3f} "
              f"({q_best/q_soup:.2f}x, ≥1.5x: {'过' if g2 else '不过'})", flush=True)
        print(f"  认证带锚预期: ε(q 最优区)={eps_best:.4f} (感知带 0.92; "
              f"分离则认证带可引导降温)", flush=True)
        print(f"  门总判定: {'过 — 进入 exp111' if (g1 and g2) else '不过 — 停, 重新设计'}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\n[probe111] 完成 {time.time()-t0:.0f}s → {args.out}", flush=True)


if __name__ == "__main__":
    main()
