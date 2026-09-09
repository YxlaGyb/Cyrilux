"""P3-b 术后验证: 门全开三段 — 正常 (零死亡, 节律涌现) → 诱导饥荒 (死亡回合)
→ 恢复 (强制感知, 零新回合, 零 NaN). 每步只喂世界 (learn(x)), 行为 (看/说) 由
模型内部门控决定 (P3-b 节律内生); 首步/恢复段 force_perception 旁路.
输出: out/exp115_p3b_ev.pt + out/exp115_p3b_ev.jsonl (每步一行, 含行为/门/账本).
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
POLL = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/exp115_p3c_ev.pt")
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--normal-steps", type=int, default=600)
    ap.add_argument("--famine-steps", type=int, default=500)
    ap.add_argument("--recover-steps", type=int, default=200)
    ap.add_argument("--famine", default="rand", choices=("rand", "shift"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="out/exp115_p3b_ev.pt")
    ap.add_argument("--out", default="out/exp115_p3b_ev.jsonl")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    net = DensePCNet.load(args.ckpt).to(dev)
    net.cfg.prune_warmup = 0  # 离线 harness 豁免发育期 (验证须见死亡回合)
    print(f"P3b 验证: ckpt={args.ckpt} ver={int(net._metab_ver.item())} "
          f"F_prev_perc={float(net._metab_F_prev_perc.item()):.3f} "
          f"κ=({net.cfg.metab_gate_kappa_g},{net.cfg.metab_gate_kappa_s},"
          f"{net.cfg.metab_gate_kappa_a},{net.cfg.metab_gate_kappa_n})", flush=True)
    ds = DualChannelDataset(args.data, max_length=S_MAX, max_samples=1270000, lazy=True)
    idxs = list(range(len(ds)))
    random.Random(args.seed).shuffle(idxs)

    jsonf = open(args.out, "w", encoding="utf-8", buffering=1)
    step_w = 0
    hist = {"leg": [], "behaviors": [], "psay": [], "stress": [], "f_now": []}
    last_death_rounds = 0
    sample_i = 0

    def step(mode: str, force_perception: bool = False):
        nonlocal step_w, last_death_rounds, sample_i
        step_w += 1
        if mode == "normal":
            b, _ = ds[idxs[sample_i % len(idxs)]]
            sample_i += 1
            x = b.unsqueeze(0).to(dev)
        else:  # rand: 随机字节 = 绝对货币下乱码暴涨 (饥荒)
            x = torch.randint(0, 256, (1, S_MAX), dtype=torch.long, device=dev)
        cur_b = net._behavior_py  # 本步执行的决策 (上一步末门算)
        net.learn(x, force_perception=force_perception)
        net.sync_gate()  # 路由镜像 (下步决策落地)
        e = float(net._metab_E.item())
        st = float(getattr(net, "_metab_stress", torch.tensor(0.0)).item())
        sc = float(net._metab_starve_cnt.item())
        dr = int(net._metab_death_round_cnt.item())
        rec = {"phase": mode, "step": step_w, "behavior": 1 if cur_b else 0,
               "psay": float(net._metab_psay.item()),
               "gain_perc": float(net._metab_gain_perc.item()),
               "gain_eco": float(net._metab_gain_eco.item()),
               "nov": float(net._metab_nov.item()),
               "E": e, "stress": st,
               "cost_perc": float(net._metab_cost_perc.item()),
               "cost_learn": float(net._metab_cost_learn.item()),
               "cost_mem": float(net._metab_cost_mem.item()),
               "cost_say": float(net._metab_cost_say.item()),
               "cost_tot": float(net._metab_cost_tot.item()),
               "starve_cnt": sc, "death_rounds": dr,
               "mem_k": net._mem_m.shape[0],
               "active_size_l4": net.active_size["l4"],
               "active_size_l5": net.active_size["l5"],
               "w1_rows": net.W1.shape[0]}
        # leg 口径: 原型 _metab_R 每步新鲜; leg = ‖W_act_elig‖·|R_raw| 仅回声步有效
        if cur_b:
            r = float(net._metab_R.item())
            tn = float(net.W_act_elig.norm().item())
            leg = tn * abs(r)
            hist["leg"].append(leg)
            rec["R"] = r
            rec["leg"] = leg
            rec["gen"] = bytes(int(v) for v in net._gen_bytes[0].tolist()).decode(
                "utf-8", "replace")[:40]
        else:
            rec["leg"] = None
        hist["behaviors"].append(rec["behavior"])
        hist["psay"].append(rec["psay"])
        hist["stress"].append(st)
        jsonf.write(json.dumps(rec) + "\n")
        net.maybe_prune(net._step_counter)
        if net._step_counter % POLL == 0 and dr > last_death_rounds:
            last_death_rounds = dr
            jsonf.write(json.dumps({"event": "death", "step": step_w,
                                    "phase": mode,
                                    "w1_rows_before": None}) + "\n")

    def run(n, mode, force=False):
        for _ in range(n):
            step(mode, force_perception=force)

    run(args.normal_steps, "normal", force=True)  # 首步 force_perception (开机先看)
    n_death_norm = int(net._metab_death_round_cnt.item())
    print(f"正常段完成: 死亡回合={n_death_norm} l4={net.active_size['l4']} "
          f"mem_k={net._mem_m.shape[0]}", flush=True)
    run(args.famine_steps, args.famine)
    n_death_fam = int(net._metab_death_round_cnt.item())
    sz_fam = (net.active_size["l4"], net.active_size["l5"], net._mem_m.shape[0])
    print(f"饥荒段({args.famine})完成: 死亡回合={n_death_fam} (正常段 {n_death_norm}) "
          f"尺寸={sz_fam}", flush=True)
    run(args.recover_steps, "normal", force=True)  # 恢复段强制感知 (治"不看世界"锁定)
    n_death_rec = int(net._metab_death_round_cnt.item())
    print(f"恢复段完成: 死亡回合={n_death_rec} (应等于 {n_death_fam})", flush=True)

    # ---- 判据 ----
    for p in net.parameters():
        assert torch.isfinite(p).all(), f"参数 NaN: {p.shape}"
    assert torch.isfinite(net._metab_E), "代谢账本 NaN"

    beh = hist["behaviors"]
    n_norm_beh = args.normal_steps
    share = sum(beh[:n_norm_beh]) / n_norm_beh
    runs = [(b, len(list(g))) for b, g in __import__("itertools").groupby(beh[:n_norm_beh])]
    e_runs = [l for b, l in runs if b == 1]
    max_echo = max(e_runs) if e_runs else 0
    # 行为序列熵 (bit): p·log2 p 对称
    p1 = share
    entropy = -(p1 * math.log2(p1) + (1 - p1) * math.log2(1 - p1)) if 0 < p1 < 1 else 0.0
    legs = sorted(hist["leg"])
    med_leg = legs[len(legs) // 2] if legs else float("nan")
    print(f"判据: echo_share={share:.3f} (∈[0.25,0.75]) max_echo_run={max_echo} (<120) "
          f"熵={entropy:.3f} bit (>0.6) leg_median={med_leg:.4f} (∈[0.3,1]) "
          f"正常段死亡={n_death_norm} 饥荒段死亡={n_death_fam - n_death_norm} "
          f"恢复段新增={n_death_rec - n_death_fam}", flush=True)
    assert 0.25 <= share <= 0.75, "正常段 echo 占比越界"
    assert max_echo < 120, "最大连续回声过长 (死亡钳裕度不足)"
    assert entropy > 0.6, "行为序列近确定性 (节律未涌现)"
    assert 0.3 <= med_leg <= 1.0, "R 契约越界 (leg 中位数)"
    assert n_death_fam > n_death_norm, "饥荒段未触发死亡回合"
    assert n_death_rec == n_death_fam, "恢复段出现新回合 (死亡不可逆语义破坏)"
    jsonf.flush()
    net.save(args.save)
    print(f"全部判据通过, saved {args.save}", flush=True)


if __name__ == "__main__":
    main()
