"""方案 A 实测: 把真实 echo 步 (continuation 63 次 _predict + learn) 罩进 CUDA Graph.

两档, 分开跑 (各自独立进程, 避免状态互相污染):
  --which cont : 整段 continuation(seed=16, n_gen=63) 捕获成一张图
  --which step : 整步 learn(None, free_run=False) 捕获成一张图

量: eager 墙钟 vs graph.replay() 墙钟; 活性检查 (换 seed 后回放输出须变).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from _probe_meta import config_meta

from dataset import DualChannelDataset  # noqa: E402
from model import DensePCNet  # noqa: E402
from pkg.cli.utils import run_file  # noqa: E402

SEED_N = 16


def _timeit(fn, reps: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def _capture(fn, warmup: int = 3):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    t0 = time.perf_counter()
    with torch.cuda.graph(g):
        out = fn()
    return g, out, time.perf_counter() - t0


def _snap(out) -> torch.Tensor:
    if isinstance(out, dict):
        out = next(v for v in out.values() if torch.is_tensor(v))
    return out.detach().clone()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["cont", "step", "both", "verify", "death"], required=True)
    ap.add_argument("--ckpt", default="out/exp115_p3c_ev.pt")
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--reps-eager", type=int, default=2)
    ap.add_argument("--reps-replay", type=int, default=5)
    ap.add_argument("--sync-debug", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or run_file("bench_graph_echo.json")

    torch.set_grad_enabled(False)
    dev = "cuda"
    net = DensePCNet.load(args.ckpt).to(dev)
    ds = DualChannelDataset(args.data, max_length=args.max_length, max_samples=1270000, lazy=True)

    seed_t = ds[101][0][-SEED_N:].unsqueeze(0).to(dev).contiguous()
    n_gen = net.cfg.free_run_window - 1  # 63

    def cont():
        return net.forward_engine.continuation(
            seed_t, n_gen, temperature=getattr(net, "_gen_temp", None), rep_backstop=False
        )

    def step():
        net._echo_seed = seed_t
        net.learn(None, free_run=False)

    # 真实预热 (同 bench_split_steps 口径)
    for i in (101, 103, 105, 107):
        net._echo_seed = ds[i][0][-SEED_N:].unsqueeze(0).to(dev)
        net.learn(None, free_run=False)

    rep: dict[str, object] = {
        "which": args.which,
        "seed_n": SEED_N,
        "n_gen": n_gen,
        "free_run_window": net.cfg.free_run_window,
        "gen_temp_is_tensor": torch.is_tensor(getattr(net, "_gen_temp", None)),
        "config": config_meta(probe="bench_graph_echo.py", args=vars(args)),
    }

    if args.which == "verify":
        # 数值对照: 同一初始状态 (state_dict 回滚) 下比较输出
        def _cmp(a, b):
            row = {"shapes_match": list(a.shape) == list(b.shape)}
            if row["shapes_match"]:
                d = (a.float() - b.float()).abs()
                row["exact_equal"] = bool(torch.equal(a, b))
                row["max_abs_diff"] = float(d.max().item())
                row["n_diff_elem"] = int((d > 0).sum().item())
                row["n_elem"] = int(d.numel())
            return row

        def _mk(tval):
            def c():
                return net.forward_engine.continuation(seed_t, n_gen, temperature=tval, rep_backstop=False)
            return c

        rows = []
        for tag, tval in (("greedy", None), ("sampled", getattr(net, "_gen_temp", None))):
            c = _mk(tval)
            sd0 = {k: v.detach().clone() for k, v in net.state_dict().items()}
            # 自检: 同状态两次 eager
            torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            out_e1 = _snap(c())
            net.load_state_dict(sd0)
            torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            out_e2 = _snap(c())
            net.load_state_dict(sd0)
            # 图: 捕获期 kernel 不执行, 必须 replay 才得结果; 且 _mem_m 在捕获期被重绑到
            # graph pool 张量 (Python 属性), 图节点读的仍是捕获前那个张量 → 需恢复引用
            mem_ref = net._mem_m
            mem_val = mem_ref.detach().clone()
            torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            g, out_g, cap_s = _capture(c, warmup=0)
            net._mem_m = mem_ref
            net.load_state_dict(sd0)
            net._mem_m = mem_ref
            mem_ref.copy_(mem_val)
            g.replay()
            torch.cuda.synchronize()
            out_g_s = _snap(out_g)
            rows.append({"mode": tag, "cmp": "eager_vs_eager", **_cmp(out_e1, out_e2)})
            rows.append({"mode": tag, "cmp": "eager_vs_graph", "capture_wall_s": round(cap_s, 3),
                         **_cmp(out_e1, out_g_s)})
            del g
        rep["rows"] = rows
        for r in rows:
            print(json.dumps(r, ensure_ascii=False), flush=True)
        Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    if args.which == "both":
        eager_step = _timeit(step, args.reps_eager)
        eager_cont = _timeit(cont, args.reps_eager)
        rep["eager_step_s"] = round(eager_step, 4)
        rep["eager_cont_s"] = round(eager_cont, 4)
        print(f"eager step {eager_step:.4f} s | eager cont {eager_cont:.4f} s", flush=True)
        g, out, cap_s = _capture(cont)
        rep["capture_wall_s"] = round(cap_s, 3)
        rep["replay_cont_s"] = round(_timeit(g.replay, args.reps_replay), 4)
        # 落地: 真实 GraphedContinuation (exp115_say, 单一来源) — 首次调用触发捕获
        from exp115_say import GraphedContinuation

        landed = GraphedContinuation(net.forward_engine, net)
        net.forward_engine.continuation = landed
        step()
        rep["landed_capture_wall_s"] = round(landed.capture_wall_s, 3)
        graph_step = _timeit(step, args.reps_eager)
        rep["graph_step_s"] = round(graph_step, 4)
        rep["step_speedup"] = round(eager_step / graph_step, 2)
        # 数值对照: 同状态 + 同 RNG 起点下 landed 回放 vs eager
        mem_val = net._mem_m.detach().clone()
        th_val = net._theta_bind.detach().clone()
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        out_e = _snap(landed.raw(seed_t, n_gen, temperature=net._gen_temp, rep_backstop=False))
        net._mem_m.copy_(mem_val)
        net._theta_bind.copy_(th_val)
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        out_g = _snap(landed(seed_t, n_gen, temperature=net._gen_temp, rep_backstop=False))
        rep["landed_vs_eager_exact"] = bool(torch.equal(out_e, out_g))
        rep["landed_vs_eager_max_abs_diff"] = float((out_e.float() - out_g.float()).abs().max().item())
        print(json.dumps({k: rep[k] for k in ("eager_step_s", "eager_cont_s", "capture_wall_s",
                                              "replay_cont_s", "landed_capture_wall_s", "graph_step_s",
                                              "step_speedup", "landed_vs_eager_exact",
                                              "landed_vs_eager_max_abs_diff")},
                         ensure_ascii=False), flush=True)
        Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    if args.which == "death":
        # 结构守卫合成验证: 剪枝会换掉权重张量 (旧图地址失效) → 同形状换参也须重捕且输出仍正确
        from exp115_say import GraphedContinuation

        landed = GraphedContinuation(net.forward_engine, net)
        net.forward_engine.continuation = landed
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        out0 = _snap(landed(seed_t, n_gen, temperature=net._gen_temp, rep_backstop=False))
        rep["captures"] = len(landed.graphs)
        cap0 = landed.capture_wall_s
        with torch.no_grad():
            net.W1 = torch.nn.Parameter(net.W1.detach().clone())
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        out1 = _snap(landed(seed_t, n_gen, temperature=net._gen_temp, rep_backstop=False))
        rep["recaptured"] = bool(landed.capture_wall_s > cap0)
        rep["recapture_deterministic"] = bool(torch.equal(out0, out1))
        # 重捕后是纯回放: 再对一次 RNG 起点 (捕获预热会吃掉 3 段 RNG, 不能拿重捕那次比)
        mem_val = net._mem_m.detach().clone()
        th_val = net._theta_bind.detach().clone()
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        out_e2 = _snap(landed.raw(seed_t, n_gen, temperature=net._gen_temp, rep_backstop=False))
        net._mem_m.copy_(mem_val)
        net._theta_bind.copy_(th_val)
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        out2 = _snap(landed(seed_t, n_gen, temperature=net._gen_temp, rep_backstop=False))
        rep["out_finite"] = bool(torch.isfinite(out2).all().item())
        rep["out_matches_eager"] = bool(torch.equal(out2, out_e2))
        print(json.dumps({k: rep[k] for k in ("captures", "recaptured", "recapture_deterministic",
                                              "out_finite", "out_matches_eager")},
                         ensure_ascii=False), flush=True)
        Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    fn = cont if args.which == "cont" else step
    eager_s = _timeit(fn, args.reps_eager)
    rep["eager_s"] = round(eager_s, 4)
    print(f"eager {args.which}: {eager_s:.4f} s/call", flush=True)

    try:
        if args.sync_debug:
            torch.cuda.set_sync_debug_mode("warn")
        g, out, cap_s = _capture(fn)
        rep["capture_ok"] = True
        rep["capture_wall_s"] = round(cap_s, 3)
        before = _snap(out)
        replay_s = _timeit(g.replay, args.reps_replay)
        rep["replay_s"] = round(replay_s, 4)
        rep["speedup_vs_eager"] = round(eager_s / replay_s, 2)
        # 活性: 换 seed 原地覆写后回放, 输出必须变化且有限
        seed_t.copy_(ds[137][0][-SEED_N:].unsqueeze(0).to(dev))
        g.replay()
        torch.cuda.synchronize()
        after = _snap(out)
        rep["out_changed_on_new_seed"] = bool(not torch.equal(before, after))
        rep["out_finite"] = bool(torch.isfinite(after).all().item())
        # RNG 活性: 同 seed 连续回放, 采样路径每次须给不同输出 (图内 philox 状态须推进)
        outs = []
        for _ in range(5):
            g.replay()
            torch.cuda.synchronize()
            outs.append(_snap(out))
        rep["same_seed_replays"] = len(outs)
        rep["same_seed_distinct"] = len({o.cpu().numpy().tobytes() for o in outs})
        print(json.dumps({k: rep[k] for k in ("capture_wall_s", "replay_s", "speedup_vs_eager",
                                               "out_changed_on_new_seed", "out_finite",
                                               "same_seed_replays", "same_seed_distinct")},
                         ensure_ascii=False), flush=True)
    except Exception as e:
        rep["capture_ok"] = False
        rep["err"] = f"{type(e).__name__}: {str(e)[:600]}"
        print(json.dumps({"capture_ok": False, "err": rep["err"]}, ensure_ascii=False), flush=True)

    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
