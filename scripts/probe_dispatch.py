"""
P0 相位归属派发普查 (静态图前置测量).

D1 版 (probe_dispatch.py) 只答"每步多少次派发"; 静态图策划要的是"派发花在哪个相位",
以及"哪些相位在等宿主" — 后者才是捕获的阻塞点. 本仪器三件事:

  1. 相位归属: 可重入栈式 TorchDispatchMode, 把一次 learn 的派发拆成互不重叠的 exclusive 计数
     (父相位扣除子相位), 覆盖 forward 族 = _predict 子树
     (_recurrent/_bind/_bind_sparse/_stp_update/corebrain.forward) + continuation 自身管道;
     其余即学习规则.
  2. 宿主同步计数: 逐相位数 aten::_local_scalar_dense (= .item() / bool(张量) / .tolist()),
     这是"捕获阻塞点"的可数形式, 与 ref 后端可对拍.
  3. 整步口径: 训练器一步 = learn + sync_gate + maybe_prune (exp115_say/exp123 线),
     另给 learn-only 口径.

只测量, 不判读, 不设门 (P0 无门)
compile 评估不在此列: 证据在 scripts/tmp/bench_step_214_*.json.
产物经 run_file 落 out/vN-*/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from _probe_meta import config_meta
from torch.utils._python_dispatch import TorchDispatchMode

from crates.cli.utils import run_file
from crates.corebrain.center import CoreBrain
from crates.hybrid.forward import ForwardEngine
from crates.model import HybridPCNet
from crates.utils.codec import ByteDataset

SEED_N = 16
SYNC_OP = "aten::_local_scalar_dense"  # .item() / bool(张量) / .tolist() 统一落到这一个算子
PHASE_PATCHES = (
    (ForwardEngine, "_predict", "_predict"),
    (ForwardEngine, "continuation", "continuation"),
    (ForwardEngine, "_recurrent", "_recurrent"),
    (ForwardEngine, "_bind", "_bind"),
    (ForwardEngine, "_bind_sparse", "_bind_sparse"),
    (ForwardEngine, "_stp_update", "_stp_update"),
    (ForwardEngine, "_precise", "_precise"),
    (CoreBrain, "forward", "corebrain.forward"),
)
FAMILY = ("_predict", "_recurrent", "_bind", "_bind_sparse", "_stp_update", "corebrain.forward")


def _opname(func: Any) -> str:
    try:
        return str(func._schema.name)
    except Exception:
        return str(func)


class PhaseCounter(TorchDispatchMode):
    """派发计数 + 可重入相位栈: exclusive 归属, 子相位派发自动从父相位扣除 (互不重叠)."""

    def __init__(self) -> None:
        self.n = 0
        self.kinds: Counter[str] = Counter()
        self.host_sync = 0
        self.frames: list[dict[str, Any]] = []
        self.calls: Counter[str] = Counter()
        self.incl: Counter[str] = Counter()
        self.excl: Counter[str] = Counter()
        self.sync: Counter[str] = Counter()
        self.wall: Counter[str] = Counter()

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        self.n += 1
        name = _opname(func)
        self.kinds[name] += 1
        if name == SYNC_OP:
            self.host_sync += 1
        return func(*args, **(kwargs or {}))

    @contextmanager
    def phase(self, name: str) -> Any:
        fr: dict[str, Any] = {"base": self.n, "sync_base": self.host_sync,
                              "t0": time.perf_counter(), "child": 0}
        self.frames.append(fr)
        try:
            yield
        finally:
            d = self.n - fr["base"]
            self.frames.pop()
            self.calls[name] += 1
            self.incl[name] += d
            self.excl[name] += d - fr["child"]
            self.sync[name] += self.host_sync - fr["sync_base"]
            self.wall[name] += time.perf_counter() - fr["t0"]
            if self.frames:
                self.frames[-1]["child"] += d


def _install_phases(counter: PhaseCounter) -> None:
    """按方法名打相位探针 (只加记账, 不改模型代码)."""
    for cls, attr, name in PHASE_PATCHES:
        orig = getattr(cls, attr)

        def wrapper(self: Any, *a: Any, _orig: Any = orig, _name: str = name, **k: Any) -> Any:
            with counter.phase(_name):
                return _orig(self, *a, **k)

        setattr(cls, attr, wrapper)


def _snap(counter: PhaseCounter) -> dict[str, Any]:
    return {"dispatch": counter.n, "host_sync": counter.host_sync,
            "calls": dict(counter.calls), "incl": dict(counter.incl),
            "excl": dict(counter.excl), "sync": dict(counter.sync)}


def _diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"dispatch": b["dispatch"] - a["dispatch"],
                           "host_sync": b["host_sync"] - a["host_sync"]}
    for key in ("calls", "incl", "excl", "sync"):
        out[key] = {k: v - a[key].get(k, 0) for k, v in b[key].items() if v - a[key].get(k, 0)}
    return out


def _finalize(rec: dict[str, Any]) -> None:
    family = sum(rec["excl"].get(k, 0) for k in FAMILY)
    cont_own = rec["excl"].get("continuation", 0)
    rec["capture_region_excl"] = family
    rec["continuation_own_excl"] = cont_own
    rec["forward_family_excl"] = family + cont_own
    rec["learn_rule_excl"] = rec["dispatch"] - family - cont_own
    rec["forward_share"] = round((family + cont_own) / max(rec["dispatch"], 1), 4)


def _means(recs: list[dict[str, Any]]) -> dict[str, Any]:
    if not recs:
        return {}
    n = len(recs)
    acc: dict[str, Counter[str]] = {k: Counter() for k in ("incl", "excl", "sync", "calls")}
    for r in recs:
        for key in acc:
            for k, v in r[key].items():
                acc[key][k] += v

    def avg(key: str) -> float:
        return round(sum(r[key] for r in recs) / n, 1)

    return {
        "n": n,
        "dispatch": avg("dispatch"), "host_sync": avg("host_sync"),
        "capture_region_excl": avg("capture_region_excl"),
        "continuation_own_excl": avg("continuation_own_excl"),
        "forward_family_excl": avg("forward_family_excl"),
        "learn_rule_excl": avg("learn_rule_excl"),
        "forward_share": round(sum(r["forward_share"] for r in recs) / n, 4),
        "phase_incl": {k: round(v / n, 1) for k, v in acc["incl"].most_common()},
        "phase_excl": {k: round(v / n, 1) for k, v in acc["excl"].most_common()},
        "phase_sync": {k: round(v / n, 2) for k, v in acc["sync"].most_common()},
        "phase_calls": {k: round(v / n, 2) for k, v in acc["calls"].most_common()},
    }


def _call_learn(net, ds, dev, i: int, last_tail: torch.Tensor, label: str) -> torch.Tensor:
    """bench_step._one_step 同款交替: sense = 真实窗, echo = 自续写 (byte_ids=None)."""
    if label == "sense":
        x = ds[i].unsqueeze(0).to(dev)
        net.learn(x)
        return x[0, -SEED_N:]
    net._echo_seed = last_tail.unsqueeze(0)
    net.learn(None, free_run=False)
    return last_tail


def _timeit(net, ds, dev, steps: int, start: int, last_tail) -> float:
    """墙钟仅报告; 计时在相位探针之外 (探针的 Python 记账会污染计时)."""
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(start, start + steps):
        last_tail = _call_learn(net, ds, dev, i, last_tail, "sense" if i % 2 == 1 else "echo")
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/exp115_p3c_ev.safetensors")
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--steps", type=int, default=6, help="learn 计数步数 (bench_step 同款交替, 起点 i=1)")
    ap.add_argument("--trainer-steps", type=int, default=3, help="训练器整步计数步数 (learn+sync_gate+maybe_prune)")
    ap.add_argument("--scan-s", type=int, nargs="+", default=[16, 32, 64, 256], help="_predict 单次逐 S 扫描")
    ap.add_argument("--time-steps", type=int, default=6, help="无探针的墙钟步数 (仅报告, 起点 i=20)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--prereg-sha", default="", help="预锁文件 sha256 (嵌入产物, 供证据链核对)")
    args = ap.parse_args()
    args.out = args.out or run_file("dispatch_census_p0.json")

    torch.set_grad_enabled(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = HybridPCNet.load(args.ckpt).to(dev)
    ds = ByteDataset(args.data, max_length=args.max_length, max_samples=1270000, lazy=True)
    last_tail = torch.zeros(SEED_N, dtype=torch.long, device=dev)

    counter = PhaseCounter()
    _install_phases(counter)

    for i in range(1, 3):  # 预热 (不计数)
        last_tail = _call_learn(net, ds, dev, i, last_tail, "sense" if i % 2 == 1 else "echo")

    s_frw = net.cfg.free_run_window
    report: dict[str, Any] = {
        "tag": "p0-dispatch-census",
        "caliber": {
            "learn_only": "步骤 i=1..steps 交替 sense/echo, 与 bench_step._one_step 同款; 计数起点 i=1",
            "trainer_step": "learn(x) + sync_gate() + maybe_prune(_step_counter), 见 exp115_say.py:742/889",
            "wall": "无探针 _timeit, start=20, 与 bench_step_214 同口径; 仅报告, 不作门",
            "warmup": "计数前 i=1..2 两步预热 (不计数)",
        },
        "steps": args.steps,
        "prereg_sha256": args.prereg_sha,
        "phase_patches": [n for _, _, n in PHASE_PATCHES],
        "sync_op": SYNC_OP,
        "family_phases": list(FAMILY),
        "free_run_window": s_frw,
        "ckpt": args.ckpt,
        "max_length": args.max_length,
    }
    learn_recs: list[dict[str, Any]] = []
    trainer_recs: list[dict[str, Any]] = []
    scans: list[dict[str, Any]] = []

    with counter:
        # ① learn-only 口径 (与 bench_step/bench_step_214 可比)
        for k in range(args.steps):
            i = 1 + k
            label = "sense" if i % 2 == 1 else "echo"
            x = ds[i].unsqueeze(0).to(dev)
            bp = bool(getattr(net, "_behavior_py", False))
            pre = _snap(counter)
            with counter.phase(f"learn_{label}"):
                if label == "sense":
                    net.learn(x)
                    last_tail = x[0, -SEED_N:]
                else:
                    net._echo_seed = last_tail.unsqueeze(0)
                    net.learn(None, free_run=False)
            rec = _diff(pre, _snap(counter))
            rec.update({"i": i, "label": label, "behavior_py_at_entry": bp})
            rec["route_echo"] = rec["calls"].get("continuation", 0) > 0
            _finalize(rec)
            learn_recs.append(rec)
        report["learn_steps"] = learn_recs

        # ② 训练器整步口径 (exp115_say/exp123 线的每步调用序列)
        for k in range(args.trainer_steps):
            i = 100 + k
            x = ds[i].unsqueeze(0).to(dev)
            bp = bool(getattr(net, "_behavior_py", False))
            pre = _snap(counter)
            with counter.phase("trainer_step"):
                with counter.phase("learn_sense"):
                    net.learn(x)
                with counter.phase("sync_gate"):
                    net.sync_gate()
                with counter.phase("maybe_prune"):
                    net.maybe_prune(net._step_counter)
            rec = _diff(pre, _snap(counter))
            rec.update({"i": i, "behavior_py_at_entry": bp})
            rec["route_echo"] = rec["calls"].get("continuation", 0) > 0
            _finalize(rec)
            trainer_recs.append(rec)
        report["trainer_steps"] = trainer_recs

        # ③ _predict 单次逐 S 扫描 (感知口径; 生成侧口径见 continuation 扫描的 _predict 子相位)
        for S in args.scan_s:
            x = ds[200][:S].unsqueeze(0).to(dev)
            pre = _snap(counter)
            with counter.phase(f"predict_S{S}"):
                net.forward_engine._predict(x, store_state=True)
            rec = _diff(pre, _snap(counter))
            rec.update({"S": S, "variant": "perception", "predict_calls": rec["calls"].get("_predict", 0)})
            _finalize(rec)
            scans.append(rec)

        pre = _snap(counter)
        with counter.phase("predict_free_run"):
            net.forward_engine._predict(None, store_state=True)
        rec = _diff(pre, _snap(counter))
        rec.update({"S": s_frw, "variant": "free_run", "predict_calls": rec["calls"].get("_predict", 0)})
        _finalize(rec)
        scans.append(rec)
        report["predict_scans"] = scans

        # ④ continuation 单次 (镜像回声路由口径: 温度张量/回退位与实际调用一致)
        seed = ds[210][:SEED_N].unsqueeze(0).to(dev)
        saved_entropy = getattr(net, "_entropy_sample", False)
        net._entropy_sample = getattr(net, "_echo_entropy", False)
        try:
            pre = _snap(counter)
            with counter.phase("continuation_echo_route"):
                out_t = net.forward_engine.continuation(
                    seed, s_frw - 1, temperature=getattr(net, "_gen_temp", None),
                    rep_backstop=getattr(net, "_echo_rep", False))
            cont = _diff(pre, _snap(counter))
        finally:
            net._entropy_sample = saved_entropy
        cont.update({"n_gen": s_frw - 1, "out_shape": list(out_t.shape),
                     "temp_is_tensor": torch.is_tensor(getattr(net, "_gen_temp", None)),
                     "predict_calls": cont["calls"].get("_predict", 0)})
        _finalize(cont)
        report["continuation_echo"] = cont

        report["counted_total_dispatch"] = counter.n
        report["counted_host_sync"] = counter.host_sync
        report["counted_kinds_top40"] = dict(counter.kinds.most_common(40))
        report["counted_n_distinct_ops"] = len(counter.kinds)
        report["phase_wall_s"] = {k: round(v, 4) for k, v in counter.wall.most_common()}

    base_s = _timeit(net, ds, dev, args.time_steps, 20, last_tail)
    report["baseline_step_s"] = round(base_s, 4)

    report["means"] = {"learn_only": _means(learn_recs), "trainer_step": _means(trainer_recs)}
    report["config"] = config_meta(probe="probe_dispatch.py (P0 相位归属版)", args={k: v for k, v in vars(args).items()})
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("config", "counted_kinds_top40")},
                     ensure_ascii=True, indent=1))


if __name__ == "__main__":
    main()
