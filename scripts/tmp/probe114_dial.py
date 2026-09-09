"""114 阶段一探针 (GPU): π 门控分布 + z4 活跃度 + 单步 |dW| + 输出熵.

只读诊断 (不改模型代码; net 为内存副本, 不回写 checkpoint):
  1. π 分布 — W_04 路径 π_pred/π_recon (逐位置 std 归一化倒数) 与
     W_42 路径 π_2 = 1/(std(eps2)+1e-3) (层标量, _precise 口径).
  2. z4 活跃度 — 死神经元比例 / 稀疏度 / 幅度分布.
  3. 单步 |dW| — 终态 checkpoint 上跑真实感知 learn 步, 量 W_04/W_42
     实际收到的更新范数 (含全部门控, 软范数前).
  4. 输出分布熵 — 读出头条件分布熵 (teacher-forced) + 采样分布熵
     (top-15 截断后) @ τ=1, 供恒温器方案裁决.

用法: .venv/Scripts/python.exe scripts/probe114_dial.py
"""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

torch.set_grad_enabled(False)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from model import CyreneModel, DensePCNet
from model.modulation import rms_norm
from dataset import DualChannelDataset

S_MAX = 256
DEV = torch.device("cuda")


def stats_dict(t):
    t = t.float().flatten()
    return {"mean": t.mean().item(), "median": t.median().item(),
            "min": t.min().item(), "max": t.max().item(),
            "p05": t.quantile(0.05).item(), "p95": t.quantile(0.95).item()}


def main():
    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=S_MAX,
                        lm_freeze_w1=True)
    net = DensePCNet.load("out/exp113_say.pt", cfg).to(DEV)
    ds = DualChannelDataset("dataset/pretrain_t2t_mini.jsonl",
                            max_length=S_MAX, max_samples=2000, lazy=False)
    res = {"n_data": len(ds)}

    # ── 1+2: π 分布 + z4 活跃度 (纯前馈, 不学习) ──
    pis_pred, pis_recon, pis_2, ent_cond, ent_samp = [], [], [], [], []
    dead_cnt, sparsity, z4_mag = 0, [], []
    a4 = net.active_size["l4"]
    a2 = net.active_size["l2"]
    N_EVAL = 128
    for i in range(N_EVAL):
        b, _ = ds[i]
        x = b.unsqueeze(0).to(DEV)
        net.forward(x)  # _predict store_state=True
        z0, z4, z2 = net._z0, net._z4, net._z2
        # z4 活跃度
        zn = z4[0].abs()  # [S, a4]
        sparsity.append((zn < 0.01).float().mean().item())
        z4_mag.append(zn.mean().item())
        dead_cnt += (zn.max(dim=0).values < 1e-3).sum().item()
        # 逐层误差 (engine.py 同式)
        eps4 = z4 - (z0 @ net.W_04[:a4].T + net.bias_l4[:a4])
        eps2 = z2 - (z4 @ net.W_42[:a2].T + net.bias_l2[:a2])
        # π_pred / π_recon (feedforward.py L91-96 口径)
        sp = eps4.std(dim=-1, keepdim=True) * 1.01 + eps4.std() * 1e-3
        pis_recon.append((1.0 / sp).flatten())
        # π_2: _precise 层标量口径
        pis_2.append(torch.full((1,), 1.0 / (eps2.std().item() + 1e-3),
                                device=DEV))
        # 读出头条件分布 (readout.py 前向数学, teacher-forced)
        z4_n_ = z4 / (z4.norm(dim=-1, keepdim=True) + 1e-3)
        pred_delta = z4_n_ @ net.W_diff[:a4, :a4].T + net.b_diff[:a4].unsqueeze(0).unsqueeze(0)
        z4r = z4 + pred_delta
        z4_lm = z4r / (1.0 + z4r.abs())
        z4_lm = rms_norm(z4_lm)
        z4_lm = z4_lm * (1.0 - 0.5 * z4_lm.pow(2))
        z4_lm = z4_lm / (1.0 + z4_lm.abs())
        zh = torch.cat([z4_lm, net._bind_vec, net._mem_out], dim=-1)
        zh = rms_norm(zh)
        import math
        h = zh @ net.W1
        h = h / (1.0 + h.abs())
        h = rms_norm(h)
        h = h * (1.0 - 0.5 * h.pow(2))
        logits = (h @ net.W_lm + net.bias_lm) * (1.0 / math.sqrt(net.d_h))
        lc = (logits - logits.mean(dim=-1, keepdim=True)) / (logits.std(dim=-1, keepdim=True) + 1e-4)
        logits = lc / lc.abs().max(dim=-1, keepdim=True).values * 60.0
        mask_print = torch.zeros(256, dtype=torch.float16, device=DEV)
        mask_print[32:] = 1.0
        logits = logits + (1.0 - mask_print) * -1e4
        # 条件分布熵 (fp32 测量口径)
        lf = logits.float()
        ent = -(torch.softmax(lf, -1) * torch.log_softmax(lf, -1)).sum(-1)
        ent_cond.append(ent.flatten())
        # 采样分布熵 @ τ=1 + top-15 截断 (forward.py 口径)
        l15 = logits.clone()
        topv, _ = torch.topk(l15, min(15, 256), dim=-1)
        l15[l15 < topv[:, :, -1:]] = -float("inf")
        l15f = l15.float()
        p15 = torch.softmax(l15f, -1)
        lp15 = torch.log_softmax(l15f, -1)
        ent15 = -torch.where(p15 > 0, p15 * lp15, torch.zeros_like(p15)).sum(-1)
        ent_samp.append(ent15.flatten())
        del net._z0, net._z4, net._z2, net._z3, net._z5, net._z6

    res["z4"] = {
        "a4": a4,
        "dead_neurons": dead_cnt / N_EVAL,
        "dead_ratio": dead_cnt / (N_EVAL * a4),
        "sparsity_mean": sum(sparsity) / len(sparsity),
        "abs_mean": sum(z4_mag) / len(z4_mag),
    }
    res["pi_recon"] = stats_dict(torch.cat(pis_recon))
    res["pi_2_layer"] = stats_dict(torch.cat(pis_2))
    res["ent_cond_bits"] = stats_dict(torch.cat(ent_cond))
    res["ent_samp_top15_bits"] = stats_dict(torch.cat(ent_samp))
    print(f"z4: a4={a4} dead_ratio={res['z4']['dead_ratio']:.4f} "
          f"sparsity={res['z4']['sparsity_mean']:.3f} |z4|={res['z4']['abs_mean']:.4f}")
    print(f"π_recon (W_04 路径, 逐位置): {res['pi_recon']}")
    print(f"π_2 (W_42 路径, 层标量): {res['pi_2_layer']}")
    print(f"条件分布熵: {res['ent_cond_bits']}")
    print(f"采样熵 (top-15, τ=1): {res['ent_samp_top15_bits']}")

    # ── 3: 单步 |dW| (真实感知步, 含全部门控) ──
    dws = {"W_04": [], "W_42": []}
    for i in range(8):
        b, _ = ds[500 + i]
        x = b.unsqueeze(0).to(DEV)
        w04_pre = net.W_04.data.clone()
        w42_pre = net.W_42.data.clone()
        net.learn(x)
        dws["W_04"].append((net.W_04.data - w04_pre).float().norm().item())
        dws["W_42"].append((net.W_42.data - w42_pre).float().norm().item())
    res["dW_per_step"] = {k: {"mean": sum(v) / len(v), "max": max(v),
                              "vals": [round(x, 5) for x in v]}
                          for k, v in dws.items()}
    print(f"单步|dW| W_04: mean={res['dW_per_step']['W_04']['mean']:.5f} "
          f"(‖W_04‖={net.W_04.data.float().norm().item():.1f})")
    print(f"单步|dW| W_42: mean={res['dW_per_step']['W_42']['mean']:.5f} "
          f"(‖W_42‖={net.W_42.data.float().norm().item():.1f})")

    with open("out/probe114_dial.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("saved -> out/probe114_dial.json")


if __name__ == "__main__":
    main()
