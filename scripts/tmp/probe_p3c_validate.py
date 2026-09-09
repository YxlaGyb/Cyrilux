"""P3-c 术后验证: 正常段 (零死亡, E* 落位) → 诱导饥荒段 (死亡回合, cost_mem 随 W1 行数降)
→ 恢复段 (零新回合, 零 NaN). 复用 P2 验证结构, maybe_prune 全程激活.
输出: out/exp115_p3c_ev.pt + out/exp115_p3c_ev.jsonl (每步一行, 含步型/成本/应激).
"""

import argparse
import json
import math
import random
import sys
import os

import torch

torch.set_grad_enabled(False)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from model import CyreneModel, DensePCNet  # noqa: E402
from dataset import DualChannelDataset  # noqa: E402

S_MAX = 256
SEED_N = 16
POLL = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/exp115_say.pt")
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--normal-steps", type=int, default=600)
    ap.add_argument("--famine-steps", type=int, default=500)
    ap.add_argument("--recover-steps", type=int, default=200)
    ap.add_argument("--famine", default="rand", choices=("rand", "shift", "drift"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="out/exp115_p3c_ev.pt")
    ap.add_argument("--out", default="out/exp115_p3c_ev.jsonl")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # config=None → 维度从检查点形状推导 (P2 后已修剪: l4=925/K=1);
    # 随后豁免发育期 (离线 harness 必须见到死亡回合)
    net = DensePCNet.load(args.ckpt).to(dev)
    net.cfg.prune_warmup = 0
    print(f"P3c 验证: ckpt={args.ckpt} 迁移后 F_prev_perc={float(net._metab_F_prev_perc.item()):.3f} "
          f"(应为 -1 哨兵) ver={int(net._metab_ver.item())}", flush=True)
    ds = DualChannelDataset(args.data, max_length=S_MAX, max_samples=1270000, lazy=True)
    idxs = list(range(len(ds)))
    random.Random(args.seed).shuffle(idxs)

    jsonf = open(args.out, "w", encoding="utf-8", buffering=1)
    step_w = 0
    last_tail = None
    hist = {"leg": [], "E": [], "stress_norm": [], "stress_fam": [], "cost_mem": []}

    last_death_rounds = 0

    def step(mode: str):
        nonlocal step_w, last_death_rounds, last_tail
        step_w += 1
        is_echo = step_w % 2 == 0
        if not is_echo:
            if mode == "normal":
                b, _ = ds[idxs[(step_w // 2) % len(idxs)]]
                x = b.unsqueeze(0).to(dev)
            elif mode == "rand":
                x = torch.randint(0, 256, (1, S_MAX), dtype=torch.long, device=dev)
            else:
                b, _ = ds[idxs[(step_w // 2) % len(idxs)]]
                x = ((b + 1) % 256).unsqueeze(0).to(dev)
            net.learn(x)
            last_tail = x[0, -SEED_N:]
        else:
            net._echo_seed = last_tail.unsqueeze(0)
            net.learn(None, free_run=False)
        e = float(net._metab_E.item())
        st = float(getattr(net, "_metab_stress", torch.tensor(0.0)).item())
        cm = float(net._metab_cost_mem.item())
        sc = float(net._metab_starve_cnt.item())
        dr = int(net._metab_death_round_cnt.item())
        rec = {"phase": mode, "step": step_w, "E": e, "stress": st,
               "cost_perc": float(net._metab_cost_perc.item()),
               "cost_learn": float(net._metab_cost_learn.item()),
               "cost_mem": cm, "cost_say": float(net._metab_cost_say.item()),
               "cost_tot": float(net._metab_cost_tot.item()),
               "starve_cnt": sc, "death_rounds": dr,
               "mem_k": net._mem_m.shape[0],
               "active_size_l4": net.active_size["l4"],
               "active_size_l5": net.active_size["l5"],
               "w1_rows": net.W1.shape[0]}
        if is_echo:
            r_c = float(getattr(net, "_survival_signal", torch.tensor(0.0)).item())
            tn = float(net.W_act_elig.norm().item())
            hist["leg"].append(tn * abs(r_c))
            rec["R"] = r_c
            rec["leg"] = tn * abs(r_c)
            rec["gen"] = bytes(int(v) for v in net._gen_bytes[0].tolist()).decode(
                "utf-8", "replace")[:40]
        jsonf.write(json.dumps(rec) + "\n")
        net.maybe_prune(net._step_counter)
        dr = int(net._metab_death_round_cnt.item())
        if net._step_counter % POLL == 0 and dr > last_death_rounds:
            last_death_rounds = dr
            jsonf.write(json.dumps({"event": "death", "step": step_w,
                                    "w1_rows_before": None}) + "\n")

    def run(n, mode):
        for _ in range(n):
            step(mode)

    run(args.normal_steps, "normal")
    n_death_norm = int(net._metab_death_round_cnt.item())
    print(f"正常段完成: 死亡回合={n_death_norm} active_size l4={net.active_size['l4']} "
          f"mem_k={net._mem_m.shape[0]}", flush=True)
    run(args.famine_steps, args.famine)
    n_death_fam = int(net._metab_death_round_cnt.item())
    sz_fam = (net.active_size["l4"], net.active_size["l5"], net._mem_m.shape[0])
    print(f"饥荒段({args.famine})完成: 死亡回合={n_death_fam} (正常段 {n_death_norm}) "
          f"尺寸(l4,l5,K)={sz_fam}", flush=True)
    run(args.recover_steps, "normal")
    n_death_rec = int(net._metab_death_round_cnt.item())
    print(f"恢复段完成: 死亡回合={n_death_rec} (应等于 {n_death_fam})", flush=True)

    # 零 NaN + finite 检查
    for p in net.parameters():
        assert torch.isfinite(p).all(), f"参数 NaN: {p.shape}"
    assert torch.isfinite(net._metab_E)

    # R 契约 + 应激分腿
    legs = sorted(hist["leg"])
    med = legs[len(legs) // 2] if legs else float("nan")
    print(f"leg median={med:.4f} n={len(legs)}", flush=True)
    jsonf.flush()
    net.save(args.save)
    print(f"saved {args.save}", flush=True)


if __name__ == "__main__":
    main()
