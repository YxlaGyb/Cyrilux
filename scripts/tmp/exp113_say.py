"""
第 113 轮主实验
R1 学习信号增益 (双侧自校准) → 学习腿启动.

前置 (本轮已裁决, 全部实测存档):
- out/probe113_ab.json (B2 轮): D1/D2 过 — 幅度绑定因果确认 (ratio
  0.154→8.92, ‖ΔW_act‖ 3.1×), 但 std(R)=0.46 过冲 28× → 汤崩溃 +
  E 濒死 0.036 (D3 败) → 增益重设计.
- out/probe113_ab_r1.json (R1 轮): D1'/D2'/D3' 全过 — R = tanh(ΔE/(2·MAD))
  /‖迹‖ 校准入区 (ratio_med 0.39 ∈ [0.3,1]), 零危机, 与对照臂同分布.
- world_lang.py step_E 已换 R1 (架构变更, 用户 2026-08-30 文字确认).
- W1 冻结 (用户裁决"修复就位 + 冻结重跑"): 首跑 step 3537 W1 NaN 中止,
  根因 = dW1 全局范数 fp16 溢出 (条目 O(10^4), 平方和 > 65504 → inf →
  更新恒零; 偶发条目溢出 → NaN 落迹). readout.py 已修复 (结构化预缩放),
  但本轮保持 W1 冻结 — 111/112 的全部实测本就是在 W1 死锁 (恒零更新)
  下取得, 冻结 = 保持实验条件连续, R1 仍是本轮唯一变量; W1 修复效果留
  待后续轮次单独验证 (一次只做一件事).

与 exp112_say 的差异 (其余逐行保持 — 感知/echo 交替、世界 lazy 全量、
NaN 守卫、冻结核心校验、双格式日志、侧车读写、--resume 全部不动):
1. tag/SIDECAR → out/exp113_say / out/exp113_world_state.json;
   BASE_STEP = 2500 (112 主跑未执行, 111 终态仍是最新态).
2. R 注入: echo 后读 ‖W_act_elig‖ 传入 step_E (R1 消费侧); R 新量级
   (std ≈ 0.04, |R| ≤ 1/‖迹‖).
3. jsonl 每条加 de_mad + leg (‖迹‖×|R_consumed|, 探针口径) 字段.
4. 50 步日志行加 R/MAD/leg; summary 加 r_std/leg_ratio 末 100 步统计
   (预注册判据 3 结算口径).

判据 (预注册于 .trae/documents/113_学习信号增益_MAD自校准.md, 累计 22500 步):
1. 零 NaN; 冻结核心 (W_04/W_42/W_diff/W_lm_2) 逐位不变
2. 学习腿启动 (主判据): 末段 2000 echo q̄ − 初段 ≥ +0.05
3. leg_ratio 末段中位 ≥ 0.3 (探针口径在线复验)
4. E 上移: 末段 Ē ≥ 0.72
5. 覆盖率审计: top1000 ≥ 0.45 (贪心 ≥ 0.50); word_hits ≥ 2 不倒退
6. 认证频率 ≥ 5%
7. 意图电位 ≥ 3%; rep_frac 末段 ≤ 初段 + 0.1

证伪线: (b) 17500 步后 q̄ 末段−初段 < +0.02 → 增益不足以启动学习腿;
(c) τ 失控 > 8 / rep_frac > 0.3 / NaN / 冻结核心漂移 → 中止;
(d) W_act 列范数守恒带 (0.8-1.2) 破坏或消费模块范数漂移 > 2× → 中止回滚.

用法: .venv/Scripts/python.exe scripts/exp113_say.py --steps 5000
      (续跑: ... --resume; 冒烟: ... --steps 100 --trigram-lines 50000)
"""
import argparse
import json
import os
import random
import sys
import time

import torch

torch.set_grad_enabled(False)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from model import CyreneModel, DensePCNet
from dataset import DualChannelDataset
from world_lang import WorldLangPhysics

S_MAX = 256
SEED_N = 16
SIDECAR = "out/exp113_world_state.json"
BASE_STEP = 2500  # 111 累计步 (exp111_say 2000 + resume 500; 112 未跑主跑)


def _terminal_111():
    """111 终态: E/锚/τ 取 resume jsonl 末记录, 认证数数两段 jsonl."""
    last = None
    with open("out/exp111_resume.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
    if last is None:
        raise FileNotFoundError("out/exp111_resume.jsonl 空 — 无法取 111 终态")
    n_cert = 0
    for p in ("out/exp111_say.jsonl", "out/exp111_resume.jsonl"):
        with open(p, encoding="utf-8") as f:
            n_cert += sum(1 for line in f if '"certified": 1' in line)
    return {"E": float(last["E"]), "anchor": float(last["anchor"]),
            "gen_temp": float(last["gen_temp"]), "n_certified": n_cert}


def _write_sidecar(world, step, net, path=SIDECAR):
    gt = getattr(net, "_gen_temp", None)
    st = world.save_state(step, float(gt.item()) if gt is not None else 4.0)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f)
    os.replace(tmp, path)


def _inject_world(net, world, dev):
    """世界动态状态 → 网络 (恒温器饥饿腿 + 锚)."""
    net._world_E = torch.tensor(world.E, dtype=torch.float16, device=dev)
    net._world_E_ref = torch.tensor(world.c * world.q_ref / world.d,
                                     dtype=torch.float16, device=dev)
    if world.world_eps_ema is not None:
        net._world_eps_ema = torch.tensor(world.world_eps_ema,
                                          dtype=torch.float16, device=dev)
        net._world_eps_mad = torch.tensor(world.world_eps_mad,
                                          dtype=torch.float16, device=dev)


class GraphedPredict:
    """echo 生成路径 CUDA graph 加速 (2026-08-30, 用户令"解决 GPU 跑不满").

    只图 (store_state=True, is_inference=True) 的逐字节 _predict — 该路径
    零随机数, 感知/学习相位 (ACh 噪声 RNG) 原样过, RNG 消费模式与 eager 一致.
    每字节 ~2400 个微内核的 CPU 启动税 (echo 步 87% 墙钟) → 一次图回放.

    等效性 (tmp_fastpath_ab 实测, 3.91x): W_act 范数轨迹逐位一致, 生成流与
    eager 差 1 个 fp16 ULP (cuBLAS 在图私有内存池上的 GEMM 算法选择差异,
    无法消除) — 自回归混沌放大后字节流不同, 对分布性判据 (leg_ratio/q̄/
    覆盖率) 无偏. 已记入 113 方案文档.

    状态处理:
    - _mem_m (注册 buffer, 跨调用 RMW): 图内烘焙读指针 = 规范缓冲, 每次
      回放后把图静态输出无条件回灌 (回放不重执行 Python 赋值);
      raw 路径 (感知/学习) 推进后同样回灌 (形失配时跳过).
    - _theta_bind (原地 mul_/add_): 回放重执行内核, 天然推进; 捕获预热
      ×3 会多推 — 捕获前保存/恢复.
    - 其余输出属性 (_z4/_z2/_bind_vec/...): 捕获末快照, 每次回放后重绑
      (raw 路径会重新指向 eager 张量).
    自愈 (两层):
    - 结构守卫 (第一层, 2026-08-30 step 16998 崩溃修复): 记忆单元死亡/
      出生经 _mem_resize 替换 _mem_*/W1/W1_elig (新 Parameter, 旧存储释放)
      — 输入形状不变但图内指针全部悬挂. 每次图调用比对结构签名
      (W1 行, _mem_m 形), 不符 → 全部弃图重捕获. 合成测试实证
      (tmp_struct_guard_test: 真实 _mem_resize K 2→1 后感知/echo/重放通过).
    - 捕获/回放异常 (第二层): RuntimeError → 清图回退 raw."""

    def __init__(self, engine, net):
        self.raw = engine._predict
        self.net = net
        self.graphs = {}
        self._struct = None  # 结构签名 (W1 行, _mem_m 形) — 记忆单元出生/死亡时变

    def _capture(self, byte_ids):
        net = self.net
        canon_mem = net._mem_m
        save_mem = canon_mem.clone()
        save_th = net._theta_bind.clone()
        static_in = byte_ids.clone()
        pre = {k: v for k, v in net.__dict__.items() if torch.is_tensor(v)}
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.raw(static_in, store_state=True, is_inference=True)
        torch.cuda.current_stream().wait_stream(s)
        canon_mem.copy_(save_mem)
        net._mem_m = canon_mem
        net._theta_bind.copy_(save_th)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self.raw(static_in, store_state=True, is_inference=True)
        snap = {}
        for k, v in net.__dict__.items():
            if torch.is_tensor(v) and (k not in pre or pre[k] is not v):
                snap[k] = v
        mem_static = net._mem_m
        if mem_static is not canon_mem:
            canon_mem.copy_(mem_static)
            net._mem_m = canon_mem
        return {"in": static_in, "g": g, "out": out, "snap": snap,
                "canon_mem": canon_mem, "mem_static": mem_static}

    def __call__(self, *a, **k):
        inf = k.get("is_inference", False)
        if not (inf and k.get("store_state", True)):
            out = self.raw(*a, **k)
            if self.graphs:
                try:
                    canon = next(iter(self.graphs.values()))["canon_mem"]
                    if (self.net._mem_m is not canon
                            and self.net._mem_m.shape == canon.shape):
                        canon.copy_(self.net._mem_m)
                        self.net._mem_m = canon
                except RuntimeError:
                    self.graphs.clear()
            return out
        # 结构守卫 (2026-08-30 step 16998 崩溃修复): 记忆单元死亡/出生改变
        # _mem_* 形与 W1 行, 但输入形状不变 — 旧图烘焙指针全部失效 (W1 被新
        # Parameter 替换, 旧存储已释放, 重放=读悬挂内存). 签名不符 → 全部弃图.
        cur = (self.net.W1.shape[0], tuple(self.net._mem_m.shape))
        if self._struct != cur:
            self.graphs.clear()
            self._struct = cur
        ent = self.graphs.get(tuple(a[0].shape))
        if ent is None:
            try:
                ent = self._capture(a[0])
                self.graphs[tuple(a[0].shape)] = ent
                return ent["out"]
            except (RuntimeError, torch.cuda.CUDAGraphError):
                return self.raw(*a, **k)
        ent["in"].copy_(a[0])
        try:
            ent["g"].replay()
            for name, obj in ent["snap"].items():
                setattr(self.net, name, obj)
            ent["canon_mem"].copy_(ent["mem_static"])
            self.net._mem_m = ent["canon_mem"]
        except RuntimeError:
            self.graphs.clear()
            return self.raw(*a, **k)
        return ent["out"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="out/exp111_resume.pt")
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--resume", action="store_true",
                    help="从终态检查点 + 侧车续跑 (分段=连续)")
    ap.add_argument("--tag", default="out/exp113_say")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=1270000)
    ap.add_argument("--trigram-lines", type=int, default=None)
    ap.add_argument("--eager", action="store_true",
                    help="禁用 CUDA graph 快路径 (逐字节 eager 基线)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = torch.device(args.device)
    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=S_MAX,
                        lm_freeze_w1=True)

    t0 = time.time()
    print("exp113: 构建世界语言物理 (常用字 2 万行 + 3-gram 位图)...", flush=True)
    world = WorldLangPhysics(args.data, n_trigram_lines=args.trigram_lines)
    print(f"  常用字={len(world.common_set)} 3-gram 数={world.n_trigrams} "
          f"q_base={world.q_base:.4f} q_ref={world.q_ref:.4f} "
          f"E_ref={world.c * world.q_ref / world.d:.4f} ({time.time()-t0:.0f}s)",
          flush=True)

    # 权重来源: 续跑 = 侧车步数对应的 step 档 (段中途崩溃恢复), 无则段末
    # 终态; 冷启动 = 111 终态 (111/112 侧车无 de_mad → R1 冷启动首笔,
    # |tanh(0.5)|/‖迹‖ ≈ 0.04 平滑入契约区 — 探针实测).
    if args.resume:
        with open(SIDECAR, encoding="utf-8") as f:
            st = json.load(f)
        step0, gt0 = world.load_state(st)
        load_path = f"{args.tag}_step{step0}.pt"
        if not os.path.exists(load_path):
            load_path = args.tag + ".pt"
        net = DensePCNet.load(load_path, cfg).to(dev)
        net._gen_temp = torch.tensor(gt0, dtype=torch.float16, device=dev)
        _inject_world(net, world, dev)
        print(f"exp113_say: resume 权重={load_path} 侧车 step={step0} "
              f"E={world.E:.4f} de_mad={world.de_mad} τ={gt0:.3f} "
              f"锚={world.world_eps_ema} hist={len(world._hist)}/{world.k_history} "
              f"cert={world.n_certified}", flush=True)
    else:
        term = _terminal_111()
        world.E = term["E"]
        world.world_eps_ema = term["anchor"]
        world.n_certified = term["n_certified"]
        net = DensePCNet.load(args.init, cfg).to(dev)
        net._gen_temp = torch.tensor(term["gen_temp"], dtype=torch.float16,
                                     device=dev)
        _inject_world(net, world, dev)
        step0 = BASE_STEP
        print(f"exp113_say: init={args.init} 111 终态侧车初值 "
              f"E={world.E:.4f} τ={term['gen_temp']:.2f} 锚={world.world_eps_ema:.4f} "
              f"cert={world.n_certified} (累计步 {BASE_STEP} 起, de_mad 冷启动)",
              flush=True)
    net._echo_entropy = False
    if not args.eager and dev.type == "cuda":
        net.forward_engine._predict = GraphedPredict(net.forward_engine, net)
        print("exp113_say: CUDA graph 快路径 ON (echo 生成路径, 3.9x; "
              "--eager 可回退)", flush=True)
    print(f"exp113_say: 本段 steps={args.steps} dev={dev} resume={args.resume}",
          flush=True)

    lazy = args.max_samples > 100000
    ds = DualChannelDataset(args.data, max_length=S_MAX,
                            max_samples=args.max_samples or None, lazy=lazy)
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    print(f"exp113_say: data n={len(ds)} lazy={lazy} (感知/echo 交替)", flush=True)

    frozen = ("W_lm", "W_lm_2", "W_04", "W_42", "W_diff", "W_bind",
              "W_bind_self", "W_t4", "W_t2", "W_t3", "W_t5", "W_t6", "W1")
    fn_base = {n: float(getattr(net, n).detach().norm().item()) for n in frozen}

    logf = open(args.tag + ".log", "a" if args.resume else "w",
                encoding="utf-8", buffering=1)
    jsonf = open(args.tag + ".jsonl", "a" if args.resume else "w",
                 encoding="utf-8", buffering=1)
    temp_hist, r_hist, e_hist, leg_hist, q_hist = [], [], [], [], []
    last_text_tail = None
    last_q = 0.0
    last_gen = b""
    step = step0

    for i in range(1, args.steps + 1):
        step = step0 + i
        if step % 2 == 1:
            b, _ = ds[idxs[(step // 2) % len(idxs)]]
            x = b.unsqueeze(0).to(dev)
            net.learn(x)
            last_text_tail = x[0, -SEED_N:]
        else:
            net._echo_seed = (
                last_text_tail.unsqueeze(0)
                if last_text_tail is not None
                else torch.zeros(1, 1, dtype=torch.long, device=dev)
            )
            net.learn(None, free_run=False)

            # R1: echo 后读迹范数 (消费侧除数) + 本次实际乘进迹通道的 R
            trace_norm = float(net.W_act_elig.norm().item())
            r_consumed = float(getattr(net, "_survival_signal",
                                       torch.tensor(0.0)).item())
            leg = trace_norm * abs(r_consumed)  # 探针口径 (基线项恒单位范数)

            gen_bytes = bytes(int(v) for v in net._gen_bytes[0].tolist())
            last_gen = gen_bytes
            sc = world.score(gen_bytes)
            last_q = sc["q"]
            world.record(gen_bytes)
            r_world = world.step_E(sc["q"], trace_norm)
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

            temp_hist.append(gt_v)
            r_hist.append(r_world)
            e_hist.append(world.E)
            leg_hist.append(leg)
            q_hist.append(sc["q"])
            rec = {"step": step, "eps_lm": eps_now, "R": r_world,
                   "E": world.E, "de_mad": world.de_mad, "leg": leg,
                   "trace_norm": trace_norm,
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
                   f"R={r_hist[-1] if r_hist else float('nan'):+.3f} "
                   f"MAD={world.de_mad if world.de_mad is not None else float('nan'):.4f} "
                   f"leg={leg_hist[-1] if leg_hist else float('nan'):.3f} "
                   f"E={e_hist[-1] if e_hist else float('nan'):.3f} "
                   f"q={last_q:.3f}")
            if step % 2 == 0:
                # 双窗口: 解码可读 (人类判读) + 字节数 (审计口径)
                msg += f" gen={last_gen[:24].decode('utf-8', errors='replace')!r} ({len(last_gen)}B)"
            print(msg, flush=True)
            logf.write(msg + "\n")
        if step % 500 == 0:
            net.save(args.tag + f"_step{step}.pt")
            _write_sidecar(world, step, net)

    net.save(args.tag + ".pt")
    _write_sidecar(world, step, net)
    # 段末清理: 中间 step 档删除, 只留终态 + 侧车 (线性链)
    for f in os.listdir("out"):
        if f.startswith(os.path.basename(args.tag) + "_step") and f.endswith(".pt"):
            os.remove(os.path.join("out", f))

    fin = {n: float(getattr(net, n).detach().norm().item()) for n in frozen}
    moved = {n: abs(fin[n] - fn_base[n]) / (fn_base[n] + 1e-9) for n in frozen}
    n_echo = len(e_hist)
    half = n_echo // 2
    last100 = slice(max(0, n_echo - 100), n_echo)
    summary = {
        "steps_done": step,
        "echo_n": n_echo,
        "temp_first_half_mean": sum(temp_hist[:half]) / max(1, half),
        "temp_last_half_mean": sum(temp_hist[half:]) / max(1, n_echo - half),
        "temp_mean_last100": sum(temp_hist[-100:]) / max(1, min(100, n_echo)),
        "E_first_half_mean": sum(e_hist[:half]) / max(1, half),
        "E_last_half_mean": sum(e_hist[half:]) / max(1, n_echo - half),
        "E_min": min(e_hist) if e_hist else None,
        "q_first_half_mean": sum(q_hist[:half]) / max(1, half),
        "q_last_half_mean": sum(q_hist[half:]) / max(1, n_echo - half),
        "q_mean_last100": sum(q_hist[last100]) / max(1, len(q_hist[last100])),
        "r_mean_last100": sum(r_hist[last100]) / max(1, len(r_hist[last100])),
        "r_std_last100": (sum((v - sum(r_hist[last100]) / max(1, len(r_hist[last100]))) ** 2
                              for v in r_hist[last100]) / max(1, len(r_hist[last100]))) ** 0.5,
        "leg_median_last100": (sorted(leg_hist[last100])[len(leg_hist[last100]) // 2]
                               if leg_hist[last100] else None),
        "frozen_moved": moved,
    }
    print("frozen-check:", {n: f"{v:.2e}" for n, v in moved.items()}, flush=True)
    print(f"[exp113_say] 段末 step={step} τ({summary['temp_first_half_mean']:.2f}→"
          f"{summary['temp_last_half_mean']:.2f}) "
          f"E({summary['E_first_half_mean']:.3f}→{summary['E_last_half_mean']:.3f}) "
          f"q̄({summary['q_first_half_mean']:.3f}→{summary['q_last_half_mean']:.3f}) "
          f"R̄(last100)={summary['r_mean_last100']:+.4f} "
          f"leg_med(last100)={summary['leg_median_last100']:.3f} "
          f"用时 {time.time()-t0:.0f}s", flush=True)
    with open(args.tag + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    logf.close()
    jsonf.close()


if __name__ == "__main__":
    main()
