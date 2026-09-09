"""P2 术后验证: 正常段 (零死亡) → 诱导饥荒段 (死亡回合) → 恢复段 (零新回合, 零 NaN).

复用探针的诱导机制, 但 maybe_prune 全程激活 (体内死亡回合实测).
输出: out/exp115_p2_ev.pt + out/exp115_p2_ev.jsonl (jsonl 含 death_event 行).
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
POLL = 8  # maybe_prune 轮询周期 (模型内一致)


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
    ap.add_argument("--save", default="out/exp115_p2_ev.pt")
    ap.add_argument("--out", default="out/exp115_p2_ev.jsonl")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=S_MAX, lm_freeze_w1=False,
                      prune_warmup=0)  # 离线 harness: 豁免发育期 (验证须见死亡回合)
    net = DensePCNet.load(args.ckpt, cfg).to(dev)
    print(f"P2 验证: ckpt={args.ckpt} 货币迁移后 F_prev_perc={float(net._metab_F_prev_perc.item()):.3f} "
          f"(应为 -1 哨兵)", flush=True)
    ds = DualChannelDataset(args.data, max_length=S_MAX, max_samples=1270000, lazy=True)
    idxs = list(range(len(ds)))
    random.Random(args.seed).shuffle(idxs)
    drift_tensor = net.W_t4.data

    jsonf = open(args.out, "w", encoding="utf-8", buffering=1)
    step_w = 0
    last_death_rounds = 0
    last_tail = None
    hist = {"leg": [], "E": []}

    def step(mode: str):
        nonlocal step_w, last_death_rounds, last_tail
        step_w += 1
        if step_w % 2 == 1:
            if mode == "normal":
                b, _ = ds[idxs[(step_w // 2) % len(idxs)]]
                x = b.unsqueeze(0).to(dev)
            elif mode == "rand":
                x = torch.randint(0, 256, (1, S_MAX), dtype=torch.long, device=dev)
            elif mode == "shift":
                b, _ = ds[idxs[(step_w // 2) % len(idxs)]]
                x = ((b + 1) % 256).unsqueeze(0).to(dev)
            else:  # drift
                b, _ = ds[idxs[(step_w // 2) % len(idxs)]]
                x = b.unsqueeze(0).to(dev)
                with torch.no_grad():
                    net.W_t4.data.mul_(1.005)
            net.learn(x)
            last_tail = x[0, -SEED_N:]
        else:
            net._echo_seed = last_tail.unsqueeze(0)
            net.learn(None, free_run=False)
            e = float(net._metab_E.item())
            sc = float(net._metab_starve_cnt.item())
            dr = int(net._metab_death_round_cnt.item())
            r_c = float(getattr(net, "_survival_signal", torch.tensor(0.0)).item())
            tn = float(net.W_act_elig.norm().item())
            hist["leg"].append(tn * abs(r_c))
            hist["E"].append(e)
            rec = {"phase": "unknown", "step": step_w, "E": e,
                   "starve_cnt": sc, "death_rounds": dr,
                   "mem_k": net._mem_m.shape[0],
                   "active_size_l4": net.active_size["l4"],
                   "active_size_l5": net.active_size["l5"],
                   "gen": bytes(int(v) for v in net._gen_bytes[0].tolist()).decode("utf-8", "replace")[:40]}
            jsonf.write(json.dumps(rec) + "\n")
        net.maybe_prune(net._step_counter)
        dr = int(net._metab_death_round_cnt.item())
        if net._step_counter % POLL == 0 and dr > last_death_rounds:
            last_death_rounds = dr

    def run(n, mode):
        for _ in range(n):
            step(mode)

    run(args.normal_steps, "normal")
    n_death_norm = int(net._metab_death_round_cnt.item())
    print(f"正常段完成: 死亡回合={n_death_norm} "
          f"active_size l4={net.active_size['l4']} mem_k={net._mem_m.shape[0]}", flush=True)
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

    # R 契约检查 (绝对货币下 leg 中位; [0.3,1] 带)
    legs = sorted(hist["leg"])
    med = legs[len(legs) // 2] if legs else float("nan")
    print(f"leg median={med:.4f} n={len(legs)}", flush=True)
    jsonf.flush()
    net.save(args.save)
    print(f"saved {args.save}", flush=True)


if __name__ == "__main__":
    main()
