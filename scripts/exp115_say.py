"""
W1 解冻验证 + 仪表升级
"""
import argparse
import codecs
import json
import math
import os
import queue
import random
import sys
import threading
import time

import torch
from world_lang import WorldLangPhysics

from dataset import ByteDataset
from model import CyreneModel, DensePCNet
from model.modulation import rms_norm
from pkg.cli.utils import pin_run_dir, resolve_path, run_dir, run_file
from pkg.outver import latest_run_dir_with

torch.set_grad_enabled(False)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

S_MAX = 256
# SEED_N 已退役: 回声种子长度入模型 cfg.echo_seed_n (P3-b 内部路由)
RPT = 25  # 遥测报告周期 (步): .item() 读数是同步排空主嫌 — 报告降频到 1/RPT,
#          统计口径改为采样统计; 事件行 (death/NaN) 不受此限
SIDECAR_NAME = "exp115_world_state.json"  # 侧车文件名 (位于本次训练的版本目录内)
LEGACY_SIDECAR = "out/exp115_world_state.json"  # 旧固定路径 (时间戳方案时代, 只读兼容)
LEGACY_TAG = "out/exp115_say"  # 旧权重固定路径前缀 (只读兼容)


def _resolve_resume(tag_base: str = "exp115_say") -> tuple[str, str]:
    """续跑定位: 含本训练线侧车的最大 N 版本目录 → (权重 tag 路径, 侧车路径).

    版本目录 (v{N}-{YYYYMMDD}-{HHMMSS}) 优先; 无则回退旧固定路径 (只读兼容).
    续跑沿用原版本目录 (pin_run_dir) — 训练线与版本目录一一对应.
    """
    d = latest_run_dir_with(resolve_path("out"), SIDECAR_NAME)
    if d is not None:
        pin_run_dir(d)
        return os.path.join(d, tag_base), os.path.join(d, SIDECAR_NAME)
    return LEGACY_TAG, LEGACY_SIDECAR


TRAIN_LINES = 1270000
GAUGE_TOTAL = 1048576  # 1MB
S_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0, 12.0)
N_EVAL_WIN = 200   # en eval = tail[0:200] (tmp_rank_probe / ladder 同臂)
N_CAL_WIN = 800    # 标定切片 = tail[200:1000] (与 eval 严格分离)


def _write_sidecar(world, step, net, path=None):
    if path is None:
        path = run_file(SIDECAR_NAME)
    gt = getattr(net, "_gen_temp", None)
    st = world.save_state(step, float(gt.item()) if gt is not None else 4.0)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f)
    os.replace(tmp, path)


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
    assert len(lines) >= n_tail, f"尾部读取不足: {len(lines)} < {n_tail}"
    tail = lines[-n_tail:]
    before = lines[-(n_tail + n_before):-n_tail]
    return [l.decode("utf-8") for l in tail], [l.decode("utf-8") for l in before]


def build_gauge(data_path, dev):
    """held-out 仪表集: 尾部全未见表 + 训练行 256B 截断尾 (行内不可见)."""
    tail_raw, before_raw = _tail_lines_bin(data_path, 238, 4000, None)

    def extract(line_str):
        sample = json.loads(line_str)
        return ByteDataset._extract_with_roles(sample)[0]

    tail_bytes = bytearray()
    for l in tail_raw:
        tail_bytes += extract(l).encode("utf-8")
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
        arr = torch.frombuffer(bytes(buf[: n_win * S_MAX]), dtype=torch.uint8)
        arr = arr.reshape(n_win, S_MAX).long()
        gauge[name] = [arr[i] for i in range(n_win)]
    return gauge, len(tail_bytes), len(trunc_bytes)


def _readout_lc(net, x, dev):
    """teacher-forced 前向 + 读出 (readout.py 前向同构), 返回归一化 lc."""
    a4 = net.active_size["l4"]
    inv_h = 1.0 / math.sqrt(net.d_h)
    net.forward(x)
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
    return lc / lc.abs().max(dim=-1, keepdim=True).values


def _score_windows(net, wins, dev, want_ranks=False, want_sgrid=False):
    """一次前向多口径: s=60 旧口径 nll + (可选) s 网格 nll + 名次统计."""
    n = 0
    leg_nll = 0.0
    grid_nll = dict.fromkeys(S_GRID, 0.0) if want_sgrid else None
    top1 = top15 = 0
    ranks: list[torch.Tensor] = []
    pos = torch.arange(S_MAX - 1, device=dev)
    for w in wins:
        x = w.unsqueeze(0).to(dev)
        lc = _readout_lc(net, x, dev)
        tgt = x[0, 1:]
        lsm60 = torch.log_softmax(lc[0] * 60.0, dim=-1).float()  # tensor-guard: report (bpb 评估口径)
        leg_nll += (-lsm60[pos, tgt] / math.log(2)).sum().item()
        if want_sgrid:
            for s in S_GRID:
                lsm = torch.log_softmax(lc[0] * s, dim=-1).float()  # tensor-guard: report (bpb 评估口径)
                grid_nll[s] += (-lsm[pos, tgt] / math.log(2)).sum().item()
        if want_ranks:
            order = (lc[0] * 60.0).argsort(dim=-1, descending=True)
            r = order.argsort(dim=-1)[pos, tgt]
            ranks.append(r.float())  # tensor-guard: report (rank 统计口径)
            top1 += (r == 0).sum().item()
            top15 += (r < 15).sum().item()
        n += S_MAX - 1
    out = {"bpb_s60": leg_nll / n, "n": n}
    if want_sgrid:
        out["bpb_per_s"] = {str(s): grid_nll[s] / n for s in S_GRID}
    if want_ranks:
        rk = torch.cat(ranks)
        out.update({"top1": top1 / n, "top15": top15 / n,
                    "rank_median": rk.median().item(),
                    "rank_mean": rk.mean().item()})
    return out


def eval_gauge(ckpt_path, cfg, gauge, dev):
    """held-out 仪表 (纯前向零学习零生成, 独立实例零状态污染):
    协议 = 114 同序 (tail 全量 → trunc 全量, fresh net):
      tail[0:200]   → en eval 口径 (名次 + s 网格 + s60)
      tail[200:1000]→ 标定切片 (s 网格 → s*)
      tail[1000:]   → 旧口径补段
      trunc 全量    → 旧口径
    s* 由标定切片 bpb 网格 argmin 决定 (阶段四硬约束)."""
    net = DensePCNet.load(ckpt_path, cfg).to(dev)
    if dev.type == "cuda":
        net.forward_engine._predict = GraphedPredict(net.forward_engine, net)
    tail, trunc = gauge["tail"], gauge["trunc"]
    ev = _score_windows(net, tail[:N_EVAL_WIN], dev,
                        want_ranks=True, want_sgrid=True)
    cal = _score_windows(net, tail[N_EVAL_WIN:N_EVAL_WIN + N_CAL_WIN], dev,
                         want_sgrid=True)
    rest = _score_windows(net, tail[N_EVAL_WIN + N_CAL_WIN:], dev)
    tr = _score_windows(net, trunc, dev)
    n_t = ev["n"] + cal["n"] + rest["n"]
    bpb_tail = (ev["bpb_s60"] * ev["n"] + cal["bpb_s60"] * cal["n"]
                + rest["bpb_s60"] * rest["n"]) / n_t
    best_s = min(S_GRID, key=lambda s: cal["bpb_per_s"][str(s)])
    out = {"bpb_tail": bpb_tail, "bpb_trunc": tr["bpb_s60"],
           "best_s_on_cal": best_s,
           "cal_bpb_at_best_s": cal["bpb_per_s"][str(best_s)],
           "eval": {"bpb_s60": ev["bpb_s60"],
                    "bpb_at_best_s": ev["bpb_per_s"][str(best_s)],
                    "bpb_per_s": ev["bpb_per_s"],
                    "top1": ev["top1"], "top15": ev["top15"],
                    "rank_median": ev["rank_median"],
                    "rank_mean": ev["rank_mean"], "n": ev["n"]},
           "cal": {"bpb_per_s": cal["bpb_per_s"], "n": cal["n"]}}
    del net
    torch.cuda.empty_cache()
    return out


class GraphedPredict:
    """单窗前向 (_predict, is_inference) CUDA graph 加速 — 评测路径用."""

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
            except RuntimeError:  # AcceleratorError ⊂ RuntimeError; 捕获失败回退 eager
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


class GraphedContinuation:
    """整段 echo continuation (变窗 _predict × 63 + 采样) 罩进一张 CUDA Graph.

    裸 eager 3.20s → 单图回放 0.27s. 温度走 net._gen_temp 固定 buffer (就地覆写,
    地址不漂移 → 回放读当前值). 键 = (输入形状, n_gen, rep_backstop, entropy_sample);
    结构变化 (权重/缓冲换张量或改形状) 或回放异常 → 清缓存重捕; 非张量温度/捕获失败 → eager.
    """

    def __init__(self, engine, net):
        self.raw = engine.continuation
        self.net = net
        self.graphs = {}
        self._struct = None
        self.disabled = False
        self.capture_wall_s = 0.0

    def _struct_key(self):
        # 结构守卫: 剪枝会换掉权重/缓冲 (新张量 = 旧图地址失效) → 全量形状 + 参数身份
        net = self.net
        p = tuple(sorted((k, tuple(v.shape), id(v)) for k, v in net.named_parameters()))
        b = tuple(sorted((k, tuple(v.shape)) for k, v in net.named_buffers()))
        return (p, b)

    def _capture(self, byte_ids, n_gen, temperature, rep_backstop):
        net = self.net
        t0 = time.perf_counter()
        canon_mem = net._mem_m
        save_mem = canon_mem.clone()
        save_th = net._theta_bind.clone()
        static_in = byte_ids.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.raw(static_in, n_gen, temperature=temperature,
                         rep_backstop=rep_backstop)
        torch.cuda.current_stream().wait_stream(s)
        canon_mem.copy_(save_mem)
        net._mem_m = canon_mem
        net._theta_bind.copy_(save_th)
        pre = {k: v for k, v in net.__dict__.items() if torch.is_tensor(v)}
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.raw(static_in, n_gen, temperature=temperature,
                     rep_backstop=rep_backstop)
        snap = {}
        for k, v in net.__dict__.items():
            if torch.is_tensor(v) and (k not in pre or pre[k] is not v):
                snap[k] = v
        mem_static = net._mem_m
        g.replay()  # 捕获不执行内核 → 回放一次落实首调输出与状态推进
        if mem_static is not canon_mem:
            canon_mem.copy_(mem_static)
            net._mem_m = canon_mem
        self.capture_wall_s += time.perf_counter() - t0
        return {"in": static_in, "g": g, "snap": snap, "canon_mem": canon_mem,
                "mem_static": mem_static, "n": int(byte_ids.shape[1]) + int(n_gen)}

    def __call__(self, byte_ids, n_gen, temperature=None, rep_backstop=False):
        if self.disabled or not torch.is_tensor(temperature):
            return self.raw(byte_ids, n_gen, temperature=temperature,
                            rep_backstop=rep_backstop)
        net = self.net
        cur = self._struct_key()
        if self._struct != cur:
            self.graphs.clear()
            self._struct = cur
        key = (tuple(byte_ids.shape), int(n_gen), bool(rep_backstop),
               bool(getattr(net, "_entropy_sample", False)))
        ent = self.graphs.get(key)
        if ent is None:
            try:
                ent = self._capture(byte_ids, n_gen, temperature, rep_backstop)
                self.graphs[key] = ent
                return net._cont_cur[:, : ent["n"]]
            except RuntimeError:  # AcceleratorError ⊂ RuntimeError; 捕获失败即永久回退 eager
                self.disabled = True
                print("exp115_say: CUDA graph 捕获失败 → echo 路径回退 eager", flush=True)
                return self.raw(byte_ids, n_gen, temperature=temperature,
                                rep_backstop=rep_backstop)
        ent["in"].copy_(byte_ids)
        if (net._mem_m is not ent["canon_mem"]
                and net._mem_m.shape == ent["canon_mem"].shape):
            ent["canon_mem"].copy_(net._mem_m)
            net._mem_m = ent["canon_mem"]
        try:
            ent["g"].replay()
            for name, obj in ent["snap"].items():
                setattr(net, name, obj)
            ent["canon_mem"].copy_(ent["mem_static"])
            net._mem_m = ent["canon_mem"]
        except RuntimeError:
            self.graphs.clear()
            return self.raw(byte_ids, n_gen, temperature=temperature,
                            rep_backstop=rep_backstop)
        return net._cont_cur[:, : ent["n"]]


def _pct(v, q):
    if not v:
        return None
    vs = sorted(v)
    return vs[min(len(vs) - 1, int(q * len(vs)))]


class _SamplePrefetch:
    """数据预取线程 (P3-1 投喂管线): 后台 dataset 取样 + pinned 非阻塞 H2D.

    与旧内联口径同序同值 (idxs[(step-1) % n] 从 start 起); 队列有界, 预取深度 2;
    预取异常经队列在主循环 get() 处重抛, 不静默挂死.
    """

    def __init__(self, ds, idxs, dev, start=0, depth=2):
        self._q: queue.Queue[torch.Tensor | BaseException] = queue.Queue(maxsize=depth)
        self._th = threading.Thread(
            target=self._loop, args=(ds, idxs, dev, start), daemon=True)
        self._th.start()

    def _loop(self, ds, idxs, dev, start):
        n = len(idxs)
        pos = start
        while True:
            try:
                b = ds[idxs[pos % n]]
                x = b.unsqueeze(0).pin_memory().to(dev, non_blocking=True)
            except Exception as e:  # 预取失败 → 主线程 get() 处重抛, 不静默挂死
                self._q.put(e)
                return
            self._q.put(x)
            pos += 1

    def get(self) -> torch.Tensor:
        x = self._q.get()
        if isinstance(x, BaseException):
            raise x
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", default="exp115_say",
                    help="产物名前缀 (位于本次训练的版本目录 out/v{N}-{YYYYMMDD}-{HHMMSS}/ 内)")
    ap.add_argument("--init-ckpt", default="out/exp114_say.safetensors",
                    help="起始检查点 (P3-b 主线: out/exp115_p3c_ev.pt)")
    ap.add_argument("--init-sidecar", default="out/exp114_world_state.json")
    ap.add_argument("--base-step", type=int, default=27500,
                    help="起始全局步 (起始检查点对应步数; 报告 offset 用)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=1270000)
    ap.add_argument("--trigram-lines", type=int, default=None)
    ap.add_argument("--eager", action="store_true")
    args = ap.parse_args()
    tag_name = os.path.basename(args.tag)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = torch.device(args.device)
    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=S_MAX,
                        lm_freeze_w1=False)  # 115 唯一训练臂差异: W1 解冻

    t0 = time.time()
    print("exp115: 构建世界语言物理 (常用字 2 万行 + 3-gram 位图)...", flush=True)
    world = WorldLangPhysics(args.data, n_trigram_lines=args.trigram_lines)
    print(f"  常用字={len(world.common_set)} 3-gram 数={world.n_trigrams} "
          f"q_base={world.q_base:.4f} q_ref={world.q_ref:.4f} "
          f"(E_ref 已入体=0.0, 外部 q 仅报告口径) ({time.time()-t0:.0f}s)",
          flush=True)

    if args.resume:
        tag, sidecar = _resolve_resume(tag_name)
        with open(sidecar, encoding="utf-8") as f:
            st = json.load(f)
        step0, gt0 = world.load_state(st)
        load_path = f"{tag}_step{step0}.safetensors"
        if not os.path.exists(load_path):
            load_path = tag + ".safetensors"
        net = DensePCNet.load(load_path, cfg).to(dev)
        # P3-a: τ 是状态纯函数 (每回声步重算) — 侧车 τ 只打印不回填
        print(f"exp115_say: resume 权重={load_path} 侧车={sidecar} step={step0} "
              f"E(体内,来自检查点)= {float(net._metab_E.item()):.4f} τ(侧车,参考)={gt0:.3f} "
              f"cert={world.n_certified}", flush=True)
    else:
        tag = run_file(args.tag)  # 本次训练全部产物 → 新版本目录 out/v{N}-{YYYYMMDD}-{HHMMSS}/
        with open(args.init_sidecar, encoding="utf-8") as f:
            st = json.load(f)
        _, gt0 = world.load_state(st)
        net = DensePCNet.load(args.init_ckpt, cfg).to(dev)
        step0 = args.base_step
        print(f"exp115_say: init={args.init_ckpt} 114 终态侧车 step={st['step']} "
              f"τ={gt0:.3f} 锚={world.world_eps_ema} "
              f"cert={world.n_certified} (累计步 {args.base_step} 起) "
              f"lm_freeze_w1=False", flush=True)
    net._echo_entropy = False
    if not args.eager and dev.type == "cuda":
        net.forward_engine.continuation = GraphedContinuation(net.forward_engine, net)
        print("exp115_say: CUDA graph 快路径 ON (整段 echo continuation 一张图)", flush=True)

    tg = time.time()
    gauge, n_tail_b, n_trunc_b = build_gauge(args.data, dev)
    print(f"exp115: held-out 仪表集: tail={n_tail_b}B ({len(gauge['tail'])} 窗) "
          f"trunc={n_trunc_b}B ({len(gauge['trunc'])} 窗) "
          f"合计={(n_tail_b + n_trunc_b)/1048576:.3f}MB ({time.time()-tg:.0f}s)",
          flush=True)
    bpb_log = open(run_file("exp115_bpb.jsonl"), "a" if args.resume else "w",
                   encoding="utf-8", buffering=1)

    lazy = args.max_samples > 100000
    ds = ByteDataset(args.data, max_length=S_MAX,
                            max_samples=args.max_samples or None, lazy=lazy)
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    print(f"exp115_say: data n={len(ds)} lazy={lazy} (感知/echo 交替)", flush=True)
    prefetch = _SamplePrefetch(ds, idxs, dev, start=step0)  # P3-1: 取样+H2D 移出主循环

    tracked = ("W_lm", "W_lm_2", "W_04", "W_42", "W_diff", "W_bind",
               "W_t4", "W_t2", "W_t3", "W_t5", "W_t6", "W1")
    init_w = {n: getattr(net, n).detach().float().cpu().clone() for n in tracked}  # tensor-guard: report (冻结快照口径)

    logf = open(run_file(tag_name + ".log"), "a" if args.resume else "w",
                encoding="utf-8", buffering=1)
    jsonf = open(run_file(tag_name + ".jsonl"), "a" if args.resume else "w",
                 encoding="utf-8", buffering=1)
    temp_hist, r_hist, e_hist, leg_hist, q_hist = [], [], [], [], []
    w1_raw_hist, w1_disk_hist, wd_raw_hist = [], [], []
    w1_start = net.W1.data.float().clone()  # tensor-guard: report (冻结快照口径)
    last_q = 0.0
    last_sc = {"q": 0.0, "L": 0.0, "S": 0.0, "X": 0.0}  # 最近一次回声步的外部评分 (报告口径)
    last_gen = b""
    gen_dec = codecs.getincrementaldecoder("utf-8")(errors="replace")  # 生成流跨步续接
    gen_txt, gen_raw_hex = "", ""
    last_e, last_dm = 0.0, float("nan")
    last_death_rounds = 0
    step = step0
    nan_at = None
    first_abnormal = None

    for i in range(1, args.steps + 1):
        step = step0 + i
        do_report = step % RPT == 0
        w1_ref = net.W1.data.clone() if do_report else None  # 报告步起点快照 (区间位移口径)
        # P3-b 节律内生: 每步喂世界流, 看/说由模型内部门控决定 (step%2 节拍器已拆除,
        # 判词 §2.2); 首步 force_perception (开机先看); sync_gate 取路由镜像
        x = prefetch.get()  # 预取样本 (与旧内联口径同序同值, 已在 dev 上)
        exec_b = net._behavior_py  # 本步执行的决策 (上一步末门算)
        net.learn(x, force_perception=(i == 1))
        net.sync_gate()
        if do_report:
            # 遥测报告 (每 RPT=25 全局步一次; .item() 读数是同步排空主嫌 → 采样统计;
            # 事件行 (death/NaN) 另行, 不受此限)
            trace_norm = float(net.W_act_elig.norm().item())
            r_v = float(net._metab_R.item())  # 原语每步新鲜 (不依赖回声步刷新)
            # leg 口径 = ‖迹‖·|生存信号| = |原语| (生存信号在消费端已除迹)
            leg = (trace_norm * abs(float(net._survival_signal.item()))
                   if exec_b else float("nan"))  # leg 只在回声步定义
            if exec_b and hasattr(net, "_gen_bytes"):
                gen_bytes = bytes(int(v) for v in net._gen_bytes[0].tolist())
                last_gen = gen_bytes
                last_sc = world.score(gen_bytes)
                last_q = last_sc["q"]
                world.record(gen_bytes)
                gen_txt = gen_dec.decode(gen_bytes)  # 增量 UTF-8 状态机 (多字节跨步续接)
                gen_raw_hex = gen_bytes.hex()
            # 体内代谢遥测: E/R 账本已入体 (ΔF 型, P3-b 行为内分账), 训练路径零外部注入
            e_v = float(net._metab_E.item())
            # 报告口径 = 本步执行行为的 MAD
            dm_v = float((net._metab_df_mad_eco if exec_b else net._metab_df_mad_perc).item())
            fam_v = float(net._metab_famine_prog.item())
            sc_v = float(net._metab_starve_cnt.item())
            sl_v = float(net._metab_silence_cnt.item())
            dr_v = int(net._metab_death_round_cnt.item())
            st_v = float(getattr(net, "_metab_stress", torch.tensor(0.0)).item())
            cp_v = float(net._metab_cost_perc.item())
            cl_v = float(net._metab_cost_learn.item())
            cm_v = float(net._metab_cost_mem.item())
            cs_v = float(net._metab_cost_say.item())
            ct_v = float(net._metab_cost_tot.item())
            last_e, last_dm = e_v, dm_v

            eps_now = float(getattr(net, "_lm_eps", torch.tensor(0.0)).item())
            certified = world.certify(last_sc["q"], eps_now)  # 仅报告口径, 不再注入机体

            gt = getattr(net, "_gen_temp", None)
            gt_v = float(gt.item()) if gt is not None else (
                net.cfg.metab_tau_lo + net.cfg.metab_tau_hi) / 2.0
            ema = getattr(net, "_lang_eps_ema", None)
            ema_v = float(ema.item()) if ema is not None else float("nan")
            rep = getattr(net, "_rep_frac", None)
            rep_v = float(rep.item()) if rep is not None else 0.0

            temp_hist.append(gt_v)
            r_hist.append(r_v)
            e_hist.append(e_v)
            leg_hist.append(leg)
            q_hist.append(last_q)
            rec = {"step": step, "eps_lm": eps_now, "R": r_v,
                   "E": e_v, "df_mad": dm_v, "stress": st_v, "leg": leg,
                   "df_mad_perc": float(net._metab_df_mad_perc.item()),
                   "df_mad_eco": float(net._metab_df_mad_eco.item()),
                   "famine_prog": fam_v,
                   "cost_perc": cp_v, "cost_learn": cl_v,
                   "cost_mem": cm_v, "cost_say": cs_v, "cost_tot": ct_v,
                   "trace_norm": trace_norm,
                   "behavior": 1 if exec_b else 0,
                   "psay": float(net._metab_psay.item()),
                   "gain_perc": float(net._metab_gain_perc.item()),
                   "gain_eco": float(net._metab_gain_eco.item()),
                   "nov": float(net._metab_nov.item()),
                   "L": last_sc["L"], "S": last_sc["S"], "X": last_sc["X"], "q": last_q,
                   "gen_temp": gt_v, "anchor": ema_v, "certified": int(certified),
                   "rep_frac": rep_v,
                   "gen": gen_txt if exec_b else None,
                   "gen_raw": gen_raw_hex if exec_b else None,
                   "starve_cnt": sc_v, "silence_cnt": sl_v, "death_rounds": dr_v,
                   "mem_k": net._mem_m.shape[0],
                   "active_size_l4": net.active_size["l4"],
                   "active_size_l5": net.active_size["l5"],
                   "w1_absmax_raw": w1_raw_hist[-1] if w1_raw_hist else None,
                   "w1_disk": w1_disk_hist[-1] if w1_disk_hist else None,
                   "w_diff_absmax_raw": wd_raw_hist[-1] if wd_raw_hist else None}
            jsonf.write(json.dumps(rec) + "\n")

        if do_report:
            # W1 双口径遥测 (每 RPT 步; 区间位移 = 当前 vs 报告步起点快照; 跨死亡回合
            # 形状变化 → 位移无定义记 NaN; 位移早于可能发生的死亡回合计算不再是前提 —
            # 事件行与位移分置, 死亡裁剪不计入"学习位移"由 NaN 占位保证).
            raw = getattr(net, "_dW1_absmax_raw", None)
            w1_raw_hist.append(float(raw.item()) if raw is not None else float("nan"))
            if w1_ref is not None and w1_ref.shape == net.W1.shape:
                w1_disk_hist.append((net.W1.data - w1_ref).norm().item())  # fp16 范数 (禁 float 转读报告口径)
            else:
                w1_disk_hist.append(float("nan"))
            wd_raw = getattr(net, "_dW_diff_absmax_raw", None)
            wd_raw_hist.append(float(wd_raw.item()) if wd_raw is not None else float("nan"))

        # 代谢触发修剪 (P2 下探缺失 + P3-b 对称牙): 墙钟已退役 — 连续饥饿 (F 无下探) 或
        # 连续沉默 (只看不说) 达阈值 → 模型内死亡回合 (mem 饥荒击杀 + 拓扑修剪).
        # 轮询 8 步, 触发权完全在模型中; 返回触发原因供事件行遥测.
        death_cause = net.maybe_prune(net._step_counter)

        # 死亡事件遥测: 轮询点比较回合计数, 新回合追加 death_event 行 (被杀单元
        # 信息不可回读, 记当前态; active_size/mem_k 为 Python 侧零同步值).
        # 轮询口径与 maybe_prune 一致 (net._step_counter, 非全局 step — resume 后偏移不同)
        if net._step_counter % 8 == 0:
            dr = int(net._metab_death_round_cnt.item())
            if dr > last_death_rounds:
                last_death_rounds = dr
                jsonf.write(json.dumps({
                    "event": "death", "step": step,
                    "cause": death_cause,
                    "E": float(net._metab_E.item()),
                    "famine_prog": float(net._metab_famine_prog.item()),
                    "death_rounds": dr,
                    "mem_k": net._mem_m.shape[0],
                    "active_size_l4": net.active_size["l4"],
                    "active_size_l5": net.active_size["l5"],
                }) + "\n")
        if do_report:
            # 全参有限扫描 (每 RPT 步 — 原每步 ~30 次 isfinite().all() 同步, 利用率主嫌;
            # 25 步滞后对"中止"守卫足够, 立即中止由 W1 遥测非有限首异常兜底)
            if first_abnormal is None and not math.isfinite(w1_raw_hist[-1]):
                first_abnormal = {"step": step, "quantity": "_dW1_absmax_raw",
                                  "value": w1_raw_hist[-1]}
                print(f"[W1] step {step}: 缩放前 |dW1| absmax={w1_raw_hist[-1]} "
                      f"(非有限 — 首异常量)", flush=True)

            bad = [n for n, p in net.named_parameters() if not torch.isfinite(p).all()]
            if bad:
                nan_at = step
                if first_abnormal is None:
                    first_abnormal = {"step": step, "quantity": f"param:{bad[0]}"}
                hi = getattr(net, "_h_in_max", None)
                msg = (f"step {step}: NaN={bad[:3]} -> 中止 | 首异常={first_abnormal} | "
                       f"‖W1_elig‖={float(net.W1_elig.norm().item()):.3e} "
                       f"‖W_act_elig‖={float(net.W_act_elig.norm().item()):.3e} "
                       f"h_in_max={float(hi.item()) if hi is not None else 'NA'} "
                       f"E={last_e:.3f} τ={temp_hist[-1] if temp_hist else 'NA'}")
                print(msg, flush=True)
                logf.write(msg + "\n")
                break

        if step % 50 == 0:
            w50 = w1_disk_hist[-50:]
            r50 = [v for v in w1_raw_hist[-50:] if math.isfinite(v)]
            msg = (f"step {step}: τ={temp_hist[-1] if temp_hist else float('nan'):.2f} "
                   f"R={r_hist[-1] if r_hist else float('nan'):+.3f} "
                   f"MAD={last_dm:.4f} "
                   f"leg={leg_hist[-1] if leg_hist else float('nan'):.3f} "
                   f"E={e_hist[-1] if e_hist else float('nan'):.3f} "
                   f"q={last_q:.3f} | "
                   f"W1: raw_max={max(r50) if r50 else float('nan'):.1f} "
                   f"disk̄={sum(w50)/len(w50):.4f} "
                   f"disp={float((net.W1.data.float()-w1_start).norm().item()):.2f}")  # tensor-guard: report (摘要位移口径)
            if exec_b:
                msg += f" gen={last_gen[:24].decode('utf-8', errors='replace')!r} ({len(last_gen)}B)"
            print(msg, flush=True)
            logf.write(msg + "\n")
        if step % 500 == 0:
            net.save(tag + f"_step{step}.safetensors")
            _write_sidecar(world, step, net)
            # ── 仪表钩子: 独立评估实例 (零状态污染) + s 标定 + 名次 ──
            te = time.time()
            g = eval_gauge(tag + f"_step{step}.safetensors", cfg, gauge, dev)
            rec = {"step": step,
                   "bpb_tail": g["bpb_tail"], "bpb_trunc": g["bpb_trunc"],
                   "bpb_all": (g["bpb_tail"] * len(gauge["tail"]) * 255
                               + g["bpb_trunc"] * len(gauge["trunc"]) * 255)
                              / ((len(gauge["tail"]) + len(gauge["trunc"])) * 255),
                   "best_s_on_cal": g["best_s_on_cal"],
                   "cal_bpb_at_best_s": g["cal_bpb_at_best_s"],
                   "eval_bpb_s60": g["eval"]["bpb_s60"],
                   "eval_bpb_at_best_s": g["eval"]["bpb_at_best_s"],
                   "eval_bpb_s4": g["eval"]["bpb_per_s"]["4.0"],
                   "top1": g["eval"]["top1"], "top15": g["eval"]["top15"],
                   "rank_median": g["eval"]["rank_median"],
                   "n_win_tail": len(gauge["tail"]),
                   "n_win_trunc": len(gauge["trunc"]),
                   "eval_s": round(time.time() - te, 1)}
            bpb_log.write(json.dumps(rec) + "\n")
            print(f"  [bpb] step={step} s60_tail={g['bpb_tail']:.4f} "
                  f"trunc={g['bpb_trunc']:.4f} | s*={g['best_s_on_cal']} "
                  f"eval@best_s={g['eval']['bpb_at_best_s']:.3f} "
                  f"(s4={g['eval']['bpb_per_s']['4.0']:.3f}) "
                  f"top15={g['eval']['top15']:.4f} "
                  f"median={g['eval']['rank_median']:.0f} ({rec['eval_s']}s)",
                  flush=True)

    net.save(tag + ".safetensors")
    _write_sidecar(world, step, net)
    if nan_at is None or step % 500 != 0:
        te = time.time()
        g = eval_gauge(tag + ".safetensors", cfg, gauge, dev)
        rec = {"step": step,
               "bpb_tail": g["bpb_tail"], "bpb_trunc": g["bpb_trunc"],
               "bpb_all": (g["bpb_tail"] * len(gauge["tail"]) * 255
                           + g["bpb_trunc"] * len(gauge["trunc"]) * 255)
                          / ((len(gauge["tail"]) + len(gauge["trunc"])) * 255),
               "best_s_on_cal": g["best_s_on_cal"],
               "cal_bpb_at_best_s": g["cal_bpb_at_best_s"],
               "eval_bpb_s60": g["eval"]["bpb_s60"],
               "eval_bpb_at_best_s": g["eval"]["bpb_at_best_s"],
               "eval_bpb_s4": g["eval"]["bpb_per_s"]["4.0"],
               "top1": g["eval"]["top1"], "top15": g["eval"]["top15"],
               "rank_median": g["eval"]["rank_median"],
               "n_win_tail": len(gauge["tail"]),
               "n_win_trunc": len(gauge["trunc"]),
               "eval_s": round(time.time() - te, 1)}
        bpb_log.write(json.dumps(rec) + "\n")
        print(f"  [bpb] 终态 step={step} s60_tail={g['bpb_tail']:.4f} "
              f"trunc={g['bpb_trunc']:.4f} | s*={g['best_s_on_cal']} "
              f"eval@best_s={g['eval']['bpb_at_best_s']:.3f} "
              f"top15={g['eval']['top15']:.4f} ({rec['eval_s']}s)", flush=True)
    for f in os.listdir(run_dir()):
        if f.startswith(tag_name + "_step") and f.endswith(".safetensors"):
            os.remove(os.path.join(run_dir(), f))

    # frozen_moved 复测 (entrywise rel_frob, 阶段六口径)
    fin_w = {n: getattr(net, n).detach().float().cpu() for n in tracked}  # tensor-guard: report (摘要快照口径)
    moved = {}
    for n in tracked:
        w0, w1 = init_w[n], fin_w[n]
        moved[n] = ((w1 - w0).norm() / (w0.norm() + 1e-12)).item()
    n_echo = len(e_hist)  # 遥测样本数 (每 RPT 全局步一次; P3-b 后不再按回声步)
    half = n_echo // 2
    last100 = slice(max(0, n_echo - 100), n_echo)
    r_last = r_hist[last100]
    w1_fin = net.W1.data.float()  # tensor-guard: report (摘要位移口径)
    summary = {
        "steps_done": step,
        "nan_at": nan_at,
        "first_abnormal": first_abnormal,
        "report_n": n_echo,
        "tau_max": max(temp_hist) if temp_hist else None,
        "tau_mean": (sum(temp_hist) / len(temp_hist)) if temp_hist else None,
        "temp_first_half_mean": sum(temp_hist[:half]) / max(1, half),
        "temp_last_half_mean": sum(temp_hist[half:]) / max(1, n_echo - half),
        "E_first_half_mean": sum(e_hist[:half]) / max(1, half),
        "E_last_half_mean": sum(e_hist[half:]) / max(1, n_echo - half),
        "q_first_half_mean": sum(q_hist[:half]) / max(1, half),
        "q_last_half_mean": sum(q_hist[half:]) / max(1, n_echo - half),
        "r_mean_last100": sum(r_last) / max(1, len(r_last)),
        "r_std_last100": (sum((v - sum(r_last) / max(1, len(r_last))) ** 2
                              for v in r_last) / max(1, len(r_last))) ** 0.5,
        "frozen_moved_entrywise": moved,
        "W1_telemetry": {
            "n_steps": len(w1_disk_hist),
            "raw_absmax_mean": (sum(v for v in w1_raw_hist if math.isfinite(v))
                                / max(1, sum(1 for v in w1_raw_hist
                                             if math.isfinite(v)))),
            "raw_absmax_median": _pct([v for v in w1_raw_hist
                                       if math.isfinite(v)], 0.5),
            "raw_absmax_p95": _pct([v for v in w1_raw_hist
                                    if math.isfinite(v)], 0.95),
            "raw_absmax_max": max((v for v in w1_raw_hist
                                   if math.isfinite(v)), default=None),
            "raw_nonfinite_steps": sum(1 for v in w1_raw_hist
                                       if not math.isfinite(v)),
            "disk_norm_mean": sum(w1_disk_hist) / max(1, len(w1_disk_hist)),
            "disk_norm_median": _pct(w1_disk_hist, 0.5),
            "disk_norm_p95": _pct(w1_disk_hist, 0.95),
            "disk_norm_max": max(w1_disk_hist) if w1_disk_hist else None,
            "W1_displacement": float((w1_fin - w1_start).norm().item()),
            "W1_rel_frob": moved["W1"],
        },
        "W_diff_telemetry": {
            "n_steps": len(wd_raw_hist),
            "raw_absmax_median": _pct([v for v in wd_raw_hist
                                       if math.isfinite(v)], 0.5),
            "raw_absmax_p95": _pct([v for v in wd_raw_hist
                                    if math.isfinite(v)], 0.95),
            "raw_absmax_max": max((v for v in wd_raw_hist
                                   if math.isfinite(v)), default=None),
            "raw_nonfinite_steps": sum(1 for v in wd_raw_hist
                                       if not math.isfinite(v)),
            "W_diff_rel_frob": moved["W_diff"],
        },
    }
    print("frozen-check (entrywise rel_frob):", flush=True)
    for n in tracked:
        print(f"  {n:12s} {moved[n]:.3e}", flush=True)
    wt = summary["W1_telemetry"]
    print(f"[exp115_say] 段末 step={step} W1: raw_absmax(med/p95/max)="
          f"{wt['raw_absmax_median']:.1f}/{wt['raw_absmax_p95']:.1f}/"
          f"{wt['raw_absmax_max']:.1f} disk̄={wt['disk_norm_mean']:.4f} "
          f"disp={wt['W1_displacement']:.2f} rel_frob={moved['W1']:.3e} "
          f"nan_at={nan_at} 用时 {time.time()-t0:.0f}s", flush=True)
    with open(run_file(tag_name + "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    logf.close()
    jsonf.close()
    bpb_log.close()


if __name__ == "__main__":
    main()
