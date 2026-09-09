"""第 114 轮短跑: 修好的恒温器 + held-out bpb 仪表 (评审诊断任务阶段三).

阶段二两处改动 (其余与 exp113 逐行保持):
1. 恒温器 τ 硬上限 2.0 (model/dense/learning/action.py, 方案 a; 方案 b 熵
   目标制被证伪: 语料字节熵 6.065 bits > top-15 采样熵上限 3.907 bits,
   目标结构性不可达 → 控制器饱和推向展平 = 重造 110c 病理).
2. held-out bpb 仪表 (本脚本, 纯测量零学习):
   - 留出集 1MB = 尾部 238 行 (训练行窗口外, 全未见) + 训练行 256 字节
     截断尾 (行内不可见字节, 拼至 1MB), 提取口径与 DualChannelDataset
     同源 (_extract_with_roles), 两子集分开报告;
   - 每 500 步 checkpoint 挂钩: 从刚写的 checkpoint 装载独立评估实例
     (零状态污染 — 评估记忆态不回流训练), 逐窗 teacher-forced 前馈 +
     读出数学 (与 _build_lm_signal 前向同构), 打印掩码前分布 (世界模型
     预测分布), fp32 log-softmax 测量口径 (fp16 模型不动);
   - 输出 out/exp114_bpb.jsonl.
3. frozen_moved 复测: 逐位 rel_frob (真移动量, 113 的范数漂移指标被
   soft_norm_preserve 行范数固定点结构性掩盖 — probe114_deepstack 实证
   W_04 rel_frob=0.754 / W_42=1.415 而范数漂移 8.7e-06).

判据 (预注册):
1. 零 NaN; W1 设计冻结不变 (lm_freeze_w1=True)
2. τ 全程 ≤ 2.0 (恒温器修复验证 — 结构保证 + 实测轨迹)
3. bpb 曲线全程记录 (升降均报, 曲线形状本身是诊断)
4. 结束时 τ=1 输出审计: top1000 覆盖率 / 词命中 (对照: τ=0 41.3%/2,
   语料 92.7%/118)
5. frozen_moved 复测: W_04/W_42/W1 逐位 rel_frob 报告

用法: .venv/Scripts/python.exe scripts/exp114_say.py --steps 5000
      (续跑: ... --resume; 冒烟: ... --steps 100)
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch

torch.set_grad_enabled(False)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from model import CyreneModel, DensePCNet
from model.modulation import rms_norm
from dataset import DualChannelDataset
from world_lang import WorldLangPhysics

S_MAX = 256
SEED_N = 16
SIDECAR = "out/exp114_world_state.json"
BASE_STEP = 22500  # 113 终态 (exp113_say.pt + 113 侧车 step=22500)
INIT_CKPT = "out/exp113_say.pt"
INIT_SIDECAR = "out/exp113_world_state.json"
TRAIN_LINES = 1270000
GAUGE_TOTAL = 1048576  # 1MB
BPB_LOG = "out/exp114_bpb.jsonl"


def _write_sidecar(world, step, net, path=SIDECAR):
    gt = getattr(net, "_gen_temp", None)
    st = world.save_state(step, float(gt.item()) if gt is not None else 4.0)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f)
    os.replace(tmp, path)


def _inject_world(net, world, dev):
    net._world_E = torch.tensor(world.E, dtype=torch.float16, device=dev)
    net._world_E_ref = torch.tensor(world.c * world.q_ref / world.d,
                                     dtype=torch.float16, device=dev)
    if world.world_eps_ema is not None:
        net._world_eps_ema = torch.tensor(world.world_eps_ema,
                                          dtype=torch.float16, device=dev)
        net._world_eps_mad = torch.tensor(world.world_eps_mad,
                                          dtype=torch.float16, device=dev)


def _tail_lines_bin(data_path, n_tail, n_before, total_lines):
    """二进制倒读文件尾: 取最后 n_tail+n_before+64 行 (多读防边界截断).
    返回 (tail_lines, before_lines) — json 字符串列表 (倒序区段, 原序返回)."""
    need = n_tail + n_before + 64
    with open(data_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        chunk = b""
        back = min(size, 64 * 1024 * 1024)
        while back <= size:
            f.seek(size - back)
            chunk = f.read(back)
            n_ln = chunk.count(b"\n")
            if n_ln >= need + 2:
                break
            back = min(size, back * 4)
            if back >= size:
                break
        f.seek(0)
        chunk = f.read(size) if back > size else chunk
    lines = chunk.split(b"\n")
    if lines and lines[-1] == b"":
        lines = lines[:-1]  # 文件尾换行
    lines = lines[1:]  # 首行可能被截断, 丢弃
    # 校验: 尾部行数应与全量扫描一致
    assert len(lines) >= n_tail, f"尾部读取不足: {len(lines)} < {n_tail}"
    tail = lines[-n_tail:]
    before = lines[-(n_tail + n_before):-n_tail]
    return [l.decode("utf-8") for l in tail], [l.decode("utf-8") for l in before]


def build_gauge(data_path, dev):
    """held-out 仪表集: 尾部全未见表 + 训练行 256B 截断尾 (行内不可见).
    前提: 文件总行数 > TRAIN_LINES (max_samples), 尾部 238 行在训练窗口外
    — main() 启动时打印 data n=... 供核对."""
    tail_raw, before_raw = _tail_lines_bin(data_path, 238, 4000, None)

    def extract(line_str):
        sample = json.loads(line_str)
        return DualChannelDataset._extract_with_roles(sample)[0]

    tail_bytes = bytearray()
    for l in tail_raw:
        tail_bytes += extract(l).encode("utf-8")
    # 截断尾: 从训练区末端倒序取 bytes[256:]
    trunc_bytes = bytearray()
    for l in before_raw:
        if len(trunc_bytes) >= GAUGE_TOTAL - len(tail_bytes):
            break
        b = extract(l).encode("utf-8")
        if len(b) > 256:
            trunc_bytes += b[256:]
    gauge = {}
    for name, buf in (("tail", tail_bytes), ("trunc", trunc_bytes)):
        n_win = len(buf) // S_MAX
        wins = []
        arr = torch.frombuffer(bytes(buf[: n_win * S_MAX]), dtype=torch.uint8)
        arr = arr.reshape(n_win, S_MAX).long()
        gauge[name] = [arr[i] for i in range(n_win)]
    return gauge, len(tail_bytes), len(trunc_bytes)


def _bpb_on_net(net, wins, dev):
    """单子集 bpb: 逐窗 teacher-forced 前馈 + 读出 (readout.py:40-97 同构),
    打印掩码前分布, fp32 log-softmax 测量口径."""
    a4 = net.active_size["l4"]
    inv_h = 1.0 / math.sqrt(net.d_h)
    nll_bits, n = 0.0, 0
    for w in wins:
        x = w.unsqueeze(0).to(dev)
        net.forward(x)  # 确定性 _predict (ACh 关)
        z4 = net._z4
        z4_n_ = z4 / (z4.norm(dim=-1, keepdim=True) + 1e-3)
        pred_delta = (z4_n_ @ net.W_diff[:a4, :a4].T
                      + net.b_diff[:a4].unsqueeze(0).unsqueeze(0))
        z4r = z4 + pred_delta
        z4_lm = z4r / (1.0 + z4r.abs())
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
        logits = lc / lc.abs().max(dim=-1, keepdim=True).values * 60.0
        lsm = torch.log_softmax(logits.float(), dim=-1)[0]  # [S,256] fp32
        tgt = x[0, 1:]  # byte_{t+1}, t=0..S-2
        nll_bits += (-lsm[torch.arange(S_MAX - 1, device=dev), tgt]
                     / math.log(2)).sum().item()
        n += S_MAX - 1
    return nll_bits / n


def eval_bpb(ckpt_path, cfg, gauge, dev):
    """held-out bpb (纯前向, 零学习零生成): 装载独立评估实例 (零状态污染)
    + 同款 CUDA graph 快路径 (测量端加速; 等效性 tmp_ab_eval A/B 实证)."""
    net = DensePCNet.load(ckpt_path, cfg).to(dev)
    if dev.type == "cuda":
        net.forward_engine._predict = GraphedPredict(net.forward_engine, net)
    out = {subset: _bpb_on_net(net, wins, dev) for subset, wins in gauge.items()}
    del net
    torch.cuda.empty_cache()
    return out


class GraphedPredict:
    """echo 生成路径 CUDA graph 加速 (同 exp113_say.py, 逐行拷贝 + 首调修复).

    第 114 轮修复 (exp113 版缺陷): CUDA graph 捕获期内核不执行 (实测: 图内
    add_ 捕获后值不变, 回放后才生效) → 旧版捕获后直接回灌未填充的 mem_static
    并返回未填充输出, 每次 (重)捕获后的首次调用产出垃圾 (tmp_probe_win 实证
    bpb 首窗 Δ+51.95 bits + 记忆暂态 1 窗衰减). 修复: 捕获后显式回放一次,
    落实本调用的输出与状态 (theta_bind 由回放内核原地推进一次, 语义 = 本该
    执行的那次调用). 训练 echo 路径与评估路径共用此修复 — 图路径契约本就是
    eager 等效, 113 版首调垃圾属契约违背非设计机制."""

    def __init__(self, engine, net):
        self.raw = engine._predict
        self.net = net
        self.graphs = {}
        self._struct = None

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
        g.replay()  # 捕获不执行内核 → 回放一次落实首调输出与状态推进
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
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", default="out/exp114_say")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=1270000)
    ap.add_argument("--trigram-lines", type=int, default=None)
    ap.add_argument("--eager", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = torch.device(args.device)
    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=S_MAX,
                        lm_freeze_w1=True)

    t0 = time.time()
    print("exp114: 构建世界语言物理 (常用字 2 万行 + 3-gram 位图)...", flush=True)
    world = WorldLangPhysics(args.data, n_trigram_lines=args.trigram_lines)
    print(f"  常用字={len(world.common_set)} 3-gram 数={world.n_trigrams} "
          f"q_base={world.q_base:.4f} q_ref={world.q_ref:.4f} "
          f"E_ref={world.c * world.q_ref / world.d:.4f} ({time.time()-t0:.0f}s)",
          flush=True)

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
        print(f"exp114_say: resume 权重={load_path} 侧车 step={step0} "
              f"E={world.E:.4f} τ={gt0:.3f} cert={world.n_certified}", flush=True)
    else:
        with open(INIT_SIDECAR, encoding="utf-8") as f:
            st = json.load(f)
        _, gt0 = world.load_state(st)
        net = DensePCNet.load(INIT_CKPT, cfg).to(dev)
        net._gen_temp = torch.tensor(gt0, dtype=torch.float16, device=dev)
        _inject_world(net, world, dev)
        step0 = BASE_STEP
        print(f"exp114_say: init={INIT_CKPT} 113 终态侧车 step={st['step']} "
              f"E={world.E:.4f} τ={gt0:.3f} 锚={world.world_eps_ema} "
              f"cert={world.n_certified} (累计步 {BASE_STEP} 起)", flush=True)
    net._echo_entropy = False
    if not args.eager and dev.type == "cuda":
        net.forward_engine._predict = GraphedPredict(net.forward_engine, net)
        print("exp114_say: CUDA graph 快路径 ON (echo 生成路径)", flush=True)

    # held-out 仪表集 (1MB, 确定性构建)
    tg = time.time()
    gauge, n_tail_b, n_trunc_b = build_gauge(args.data, dev)
    print(f"exp114: held-out 仪表集: tail={n_tail_b}B ({len(gauge['tail'])} 窗) "
          f"trunc={n_trunc_b}B ({len(gauge['trunc'])} 窗) "
          f"合计={(n_tail_b + n_trunc_b)/1048576:.3f}MB ({time.time()-tg:.0f}s)",
          flush=True)
    bpb_log = open(BPB_LOG, "a" if args.resume else "w",
                   encoding="utf-8", buffering=1)

    lazy = args.max_samples > 100000
    ds = DualChannelDataset(args.data, max_length=S_MAX,
                            max_samples=args.max_samples or None, lazy=lazy)
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    print(f"exp114_say: data n={len(ds)} lazy={lazy} (感知/echo 交替)", flush=True)

    tracked = ("W_lm", "W_lm_2", "W_04", "W_42", "W_diff", "W_bind",
               "W_t4", "W_t2", "W_t3", "W_t5", "W_t6", "W1")
    init_w = {n: getattr(net, n).detach().float().cpu().clone() for n in tracked}

    logf = open(args.tag + ".log", "a" if args.resume else "w",
                encoding="utf-8", buffering=1)
    jsonf = open(args.tag + ".jsonl", "a" if args.resume else "w",
                 encoding="utf-8", buffering=1)
    temp_hist, r_hist, e_hist, leg_hist, q_hist = [], [], [], [], []
    last_text_tail = None
    last_q = 0.0
    last_gen = b""
    step = step0
    nan_at = None

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

            trace_norm = float(net.W_act_elig.norm().item())
            r_consumed = float(getattr(net, "_survival_signal",
                                       torch.tensor(0.0)).item())
            leg = trace_norm * abs(r_consumed)

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
            nan_at = step
            hi = getattr(net, "_h_in_max", None)
            msg = (f"step {step}: NaN={bad[:3]} -> 中止 | "
                   f"‖W1_elig‖={float(net.W1_elig.norm().item()):.3e} "
                   f"‖W_act_elig‖={float(net.W_act_elig.norm().item()):.3e} "
                   f"h_in_max={float(hi.item()) if hi is not None else 'NA'} "
                   f"E={world.E:.3f} τ={temp_hist[-1] if temp_hist else 'NA'}")
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
                msg += f" gen={last_gen[:24].decode('utf-8', errors='replace')!r} ({len(last_gen)}B)"
            print(msg, flush=True)
            logf.write(msg + "\n")
        if step % 500 == 0:
            net.save(args.tag + f"_step{step}.pt")
            _write_sidecar(world, step, net)
            # ── bpb 仪表钩子: 独立评估实例 (零状态污染) ──
            te = time.time()
            bpb = eval_bpb(args.tag + f"_step{step}.pt", cfg, gauge, dev)
            rec = {"step": step,
                   "bpb_tail": bpb["tail"], "bpb_trunc": bpb["trunc"],
                   "bpb_all": (bpb["tail"] * len(gauge["tail"]) * 255
                               + bpb["trunc"] * len(gauge["trunc"]) * 255)
                              / ((len(gauge["tail"]) + len(gauge["trunc"])) * 255),
                   "n_win_tail": len(gauge["tail"]),
                   "n_win_trunc": len(gauge["trunc"]),
                   "eval_s": round(time.time() - te, 1)}
            bpb_log.write(json.dumps(rec) + "\n")
            print(f"  [bpb] step={step} tail={bpb['tail']:.4f} "
                  f"trunc={bpb['trunc']:.4f} ({rec['eval_s']}s)", flush=True)

    net.save(args.tag + ".pt")
    _write_sidecar(world, step, net)
    if nan_at is None or step % 500 != 0:
        te = time.time()
        bpb = eval_bpb(args.tag + ".pt", cfg, gauge, dev)
        rec = {"step": step,
               "bpb_tail": bpb["tail"], "bpb_trunc": bpb["trunc"],
               "bpb_all": (bpb["tail"] * len(gauge["tail"]) * 255
                           + bpb["trunc"] * len(gauge["trunc"]) * 255)
                          / ((len(gauge["tail"]) + len(gauge["trunc"])) * 255),
               "n_win_tail": len(gauge["tail"]),
               "n_win_trunc": len(gauge["trunc"]),
               "eval_s": round(time.time() - te, 1)}
        bpb_log.write(json.dumps(rec) + "\n")
        print(f"  [bpb] 终态 step={step} tail={bpb['tail']:.4f} "
              f"trunc={bpb['trunc']:.4f} ({rec['eval_s']}s)", flush=True)
    for f in os.listdir("out"):
        if f.startswith(os.path.basename(args.tag) + "_step") and f.endswith(".pt"):
            os.remove(os.path.join("out", f))

    # frozen_moved 复测: 逐位 rel_frob (真移动) + 范数漂移 (113 旧口径)
    fin_w = {n: getattr(net, n).detach().float().cpu() for n in tracked}
    moved, norm_drift = {}, {}
    for n in tracked:
        w0, w1 = init_w[n], fin_w[n]
        if w0.shape != w1.shape:
            moved[n] = f"shape {tuple(w0.shape)}→{tuple(w1.shape)}"
            norm_drift[n] = None
            continue
        moved[n] = ((w1 - w0).norm() / (w0.norm() + 1e-12)).item()
        norm_drift[n] = (abs(w1.norm() - w0.norm()) / (w0.norm() + 1e-12)).item()
    n_echo = len(e_hist)
    half = n_echo // 2
    last100 = slice(max(0, n_echo - 100), n_echo)
    summary = {
        "steps_done": step,
        "nan_at": nan_at,
        "echo_n": n_echo,
        "tau_max": max(temp_hist) if temp_hist else None,
        "tau_mean": (sum(temp_hist) / len(temp_hist)) if temp_hist else None,
        "temp_first_half_mean": sum(temp_hist[:half]) / max(1, half),
        "temp_last_half_mean": sum(temp_hist[half:]) / max(1, n_echo - half),
        "E_first_half_mean": sum(e_hist[:half]) / max(1, half),
        "E_last_half_mean": sum(e_hist[half:]) / max(1, n_echo - half),
        "q_first_half_mean": sum(q_hist[:half]) / max(1, half),
        "q_last_half_mean": sum(q_hist[half:]) / max(1, n_echo - half),
        "r_mean_last100": sum(r_hist[last100]) / max(1, len(r_hist[last100])),
        "r_std_last100": (sum((v - sum(r_hist[last100]) / max(1, len(r_hist[last100]))) ** 2
                              for v in r_hist[last100]) / max(1, len(r_hist[last100]))) ** 0.5,
        "leg_median_last100": (sorted(leg_hist[last100])[len(leg_hist[last100]) // 2]
                               if leg_hist[last100] else None),
        "frozen_moved_entrywise": moved,
        "frozen_norm_drift_old_metric": norm_drift,
    }
    print("frozen-check (entrywise rel_frob | 旧范数口径):", flush=True)
    for n in tracked:
        if isinstance(moved[n], str):
            print(f"  {n:12s} {moved[n]}", flush=True)
        else:
            print(f"  {n:12s} {moved[n]:.3e} | {norm_drift[n]:.3e}", flush=True)
    print(f"[exp114_say] 段末 step={step} τ(max)={summary['tau_max']} "
          f"E({summary['E_first_half_mean']:.3f}→{summary['E_last_half_mean']:.3f}) "
          f"q̄({summary['q_first_half_mean']:.3f}→{summary['q_last_half_mean']:.3f}) "
          f"用时 {time.time()-t0:.0f}s", flush=True)
    with open(args.tag + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    logf.close()
    jsonf.close()
    bpb_log.close()


if __name__ == "__main__":
    main()
