"""第 111 轮核心实验: 世界语言物理 — 恢复选择压 (D1+D2+D5+D6 联跑).

前置 (本轮已裁决):
- probe110 (docs/es/110_probe.md): ε_lm 极性倒挂 — 循环 0.55 < 语言 0.89 <
  乱码 0.996 → 108 差分 R 方向结构性错误.
- 110c 舒适高原 (docs/es/110_say.md 附录): band_in 99.1%, R 净 +981 但
  输出是生僻字汤 (top1000 覆盖 0.13). 病根 = ε 代理失效: ε 维度上高温汤
  (0.913) 与真实语言 (0.920) 不可分, 恒温器升温展平落带, 选择压消失.
- probe111 (out/probe111_world_dims.json): 覆盖维度语料/汤 q 分离 4.8x
  (L 维度 10.6x, S 维度 1.6x), 贪心最优区 q=0.70 (τ=0), 认证带 ε≈0.54
  远离感知带 0.92 — 门过, 世界维度有判别力.

本实验 (感知/echo 交替, 世界宽度 lazy 全量):
- 感知步: learn(真实中文) — 世界模型继续变准 (第二条腿)
- echo 步: learn(None) — W_lm 声道续写 + 世界评分后注入 _world_R (fp16)
  资格迹选择压替换 ε band R (action.py 第 111 轮已改: _world_R 驱动
  _survival_signal, ε band 判别删除)
- 世界物理 (scripts/world_lang.py, 脚本层): L 覆盖率 × S 3gram 结构 ×
  (1-X 新颖税) → q → E 代谢 → R_world; 认证门 (q ≥ q_ref) 更新恒温器
  认证带锚 (net._world_eps_ema) — 汤不被认证 → 锚指向高 q 区 ε → 降温.
- 恒温器 (action.py): 认证锚方向控制 + 饥饿应激降温 (E < E_ref → 压温度).

判据 (预注册于 .trae/documents/111_世界语言物理_恢复选择压.md):
1. 零 NaN; 冻结核心 (W_04/W_42/W_diff/W_lm_2) 逐位不变
2. 温度离开廉价解区: 末段 τ̄ < 8 或持续下行轨迹
3. E 轨迹上升: 末段 E > 初段 E, 且不归零
4. 覆盖率审计 (audit_char_coverage 硬判据): top1000 ≥ 0.4 进步 / ≥ 0.6 成功
5. 意图电位占比 ≥ 3% (probe110c 口径, 长跑后单独测)
6. 无循环塌缩: rep_frac 不上升

证伪线 (触发则停, 如实报告, 112 重设计温度传导):
温度钉死 [9,11] + E 恒低 + 覆盖率 ≤0.15 → 世界 R 无法传导到行为/温度.

用法: uv run python scripts/exp111_say.py --steps 2000
      (冒烟: uv run python scripts/exp111_say.py --steps 100 --trigram-lines 50000)
"""
import argparse
import json
import random
import sys
import time

import torch

torch.set_grad_enabled(False)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from model.dense import DensePCConfig, DensePCNet
from model.training.dataset import DualChannelDataset
from world_lang import WorldLangPhysics

S_MAX = 256
SEED_N = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="out/exp110c_world.pt")
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--start-step", type=int, default=1,
                    help="从指定步数续跑 (用于断点续跑)")
    ap.add_argument("--tag", default="out/exp111_say")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=1270000,
                    help="感知相位语料量; 大值 (D3 世界宽度) 走 lazy 流式")
    ap.add_argument("--trigram-lines", type=int, default=None,
                    help="3-gram 位图构建行数 (默认全量; 冒烟用 5 万)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = torch.device(args.device)
    cfg = DensePCConfig(d_input=256, d_act=256, max_seq_len=S_MAX)
    net = DensePCNet.load(args.init, cfg).to(dev)
    net._echo_entropy = False
    print(f"exp111_say: init={args.init} steps={args.steps} dev={dev}", flush=True)

    t0 = time.time()
    print("exp111: 构建世界语言物理 (常用字 2 万行 + 3-gram 位图)...", flush=True)
    world = WorldLangPhysics(args.data, n_trigram_lines=args.trigram_lines)
    print(f"  常用字={len(world.common_set)} 3-gram 数={world.n_trigrams} "
          f"q_base={world.q_base:.4f} q_ref={world.q_ref:.4f} "
          f"E_ref={world.c * world.q_ref / world.d:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # 世界状态注入网络 (恒温器饥饿应激读取)
    net._world_E = torch.tensor(world.E, dtype=torch.float16, device=dev)
    net._world_E_ref = torch.tensor(world.c * world.q_ref / world.d,
                                    dtype=torch.float16, device=dev)

    lazy = args.max_samples > 100000
    ds = DualChannelDataset(args.data, max_length=S_MAX,
                            max_samples=args.max_samples or None, lazy=lazy)
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    print(f"exp111_say: data n={len(ds)} lazy={lazy} (感知/echo 交替)", flush=True)

    frozen = ("W_lm", "W_lm_2", "W_04", "W_42", "W_diff", "W_bind",
              "W_bind_self", "W_t4", "W_t2", "W_t3", "W_t5", "W_t6")
    fn_base = {n: float(getattr(net, n).detach().norm().item()) for n in frozen}

    logf = open(args.tag + ".log", "w", encoding="utf-8", buffering=1)
    jsonf = open(args.tag + ".jsonl", "w", encoding="utf-8", buffering=1)
    echo_eps_hist = []
    r_hist = []
    e_hist = []
    temp_hist = []
    last_text_tail = None
    last_q = 0.0
    last_gen = b""

    for step in range(1, args.steps + 1):
        if step % 2 == 1:
            b, _ = ds[idxs[(step // 2) % len(idxs)]]
            x = b.unsqueeze(0).to(dev)
            net.learn(x)
            last_text_tail = x[0, -SEED_N:]
        else:
            # echo 相位: W_lm 声道续写 (action.py 读 _world_R 驱动资格迹)
            net._echo_seed = (
                last_text_tail.unsqueeze(0)
                if last_text_tail is not None
                else torch.zeros(1, 1, dtype=torch.long, device=dev)
            )
            net.learn(None, free_run=False)

            # 世界评分 → 代谢 → 认证 (发声是进食行为)
            gen_bytes = bytes(int(v) for v in net._gen_bytes[0].tolist())
            last_gen = gen_bytes
            sc = world.score(gen_bytes)
            last_q = sc["q"]
            world.record(gen_bytes)
            r_world = world.step_E(sc["q"])
            net._world_R = torch.tensor(r_world, dtype=torch.float16, device=dev)
            net._world_E = torch.tensor(world.E, dtype=torch.float16, device=dev)

            eps_now = float(getattr(net, "_lm_eps", torch.tensor(0.0)).item())
            certified = world.certify(sc["q"], eps_now)
            if certified and world.world_eps_ema is not None:
                net._world_eps_ema = torch.tensor(world.world_eps_ema,
                                                  dtype=torch.float16, device=dev)
                net._world_eps_mad = torch.tensor(world.world_eps_mad,
                                                  dtype=torch.float16, device=dev)

            gt = getattr(net, "_gen_temp", None)
            gt_v = float(gt.item()) if gt is not None else 4.0
            ema = getattr(net, "_world_eps_ema", None) or getattr(net, "_lang_eps_ema", None)
            ema_v = float(ema.item()) if ema is not None else float("nan")
            rep = getattr(net, "_rep_frac", None)
            rep_v = float(rep.item()) if rep is not None else 0.0

            echo_eps_hist.append(eps_now)
            r_hist.append(r_world)
            e_hist.append(world.E)
            temp_hist.append(gt_v)
            rec = {"step": step, "eps_lm": eps_now, "R": r_world, "E": world.E,
                   "L": sc["L"], "S": sc["S"], "X": sc["X"], "q": sc["q"],
                   "gen_temp": gt_v, "anchor": ema_v, "certified": int(certified),
                   "rep_frac": rep_v}
            jsonf.write(json.dumps(rec) + "\n")

        bad = [n for n, p in net.named_parameters() if not torch.isfinite(p).all()]
        if bad:
            msg = f"step {step}: NaN={bad[:3]} -> 中止"
            print(msg, flush=True)
            logf.write(msg + "\n")
            break

        if step % 50 == 0:
            msg = (f"step {step}: τ={temp_hist[-1] if temp_hist else float('nan'):.2f} "
                   f"R={r_hist[-1] if r_hist else float('nan'):+.4f} "
                   f"E={e_hist[-1] if e_hist else float('nan'):.3f} "
                   f"q={last_q:.3f}")
            if step % 2 == 0:
                # 双窗口: 解码可读 (人类判读) + 字节数 (审计口径)
                msg += f" gen={last_gen[:24].decode('utf-8', errors='replace')!r} ({len(last_gen)}B)"
            print(msg, flush=True)
            logf.write(msg + "\n")
        if step % 250 == 0:
            net.save(args.tag + f"_step{step}.pt")

    net.save(args.tag + ".pt")

    fin = {n: float(getattr(net, n).detach().norm().item()) for n in frozen}
    moved = {n: abs(fin[n] - fn_base[n]) / (fn_base[n] + 1e-9) for n in frozen}
    n_echo = len(echo_eps_hist)
    half = n_echo // 2
    summary = {
        "steps_done": step,
        "echo_n": n_echo,
        "temp_first_half_mean": sum(temp_hist[:half]) / max(1, half),
        "temp_last_half_mean": sum(temp_hist[half:]) / max(1, n_echo - half),
        "temp_mean_last100": sum(temp_hist[-100:]) / max(1, min(100, n_echo)),
        "E_first_half_mean": sum(e_hist[:half]) / max(1, half),
        "E_last_half_mean": sum(e_hist[half:]) / max(1, n_echo - half),
        "E_min": min(e_hist),
        "r_mean_last100": sum(r_hist[-100:]) / max(1, min(100, n_echo)),
        "frozen_moved": moved,
    }
    print("frozen-check:", {n: f"{v:.2e}" for n, v in moved.items()}, flush=True)
    print(f"[exp111_say] τ({summary['temp_first_half_mean']:.2f}→"
          f"{summary['temp_last_half_mean']:.2f}) "
          f"E({summary['E_first_half_mean']:.3f}→{summary['E_last_half_mean']:.3f}) "
          f"R̄(last100)={summary['r_mean_last100']:+.4f} 用时 {time.time()-t0:.0f}s", flush=True)
    with open(args.tag + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    logf.close()
    jsonf.close()


if __name__ == "__main__":
    main()
