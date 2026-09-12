"""
arch_guard 架构回归闸

为何用 CPU: 贪心 generate 与 learn 都是纯计算 (零采样随机), fp16 CPU
matmul 逐位可复现; CUDA 归约还原序不确定, 不足以作逐位闸.
为何不对照 exp115_say.log: 日志生成走 τ=1.0 随机采样 + 训练中演化状态,
没有可复现的 oracle. 因此锚定本闸自己的确定性输出.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from redline import scan_model_sources

from model import DensePCNet
from pkg.outver import ensure_run_dir

torch.set_grad_enabled(False)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).resolve().parent.parent / "out"


def _latest(pattern: str) -> Path | None:
    """out/ 版本目录 (v{N}-{YYYYMMDD}-{HHMMSS}) 内按 glob 取最新 — 零硬编码路径."""
    hits = sorted(OUT.glob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def _resolve_ckpt() -> str:
    p = _latest("*/migrated/exp115_say.safetensors")
    if p is None:
        raise FileNotFoundError("out/*/migrated/exp115_say.safetensors 不存在 — 先跑 scripts/migrate_ckpt.py")
    return str(p)


def _resolve_baseline() -> str:
    p = _latest("*/arch_guard_baseline.json")
    if p is None:  # 首跑锚定: 按版本目录规范新开 (pkg/outver)
        d = ensure_run_dir(OUT)
        return str(Path(d) / "arch_guard_baseline.json")
    return str(p)


CKPT = _resolve_ckpt()
BASELINE = _resolve_baseline()
FIXED_PROMPT = "春"
FIXED_N = 64
SEED = 117
LEARN_STEPS = 4
LEARN_W = ("W_04", "W_42", "W_23", "W_35", "W_56", "W_diff", "W1", "W_lm")


def _run_pytest() -> dict:
    """跑 tests/ 全量, 返回 pass/fail 计数."""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--no-cov", "-q"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    tail = (r.stdout or r.stderr).strip().splitlines()
    summary = tail[-1] if tail else ""
    return {"returncode": r.returncode, "summary": summary}


def _gen_signature(net, prompt: str, n: int, dev) -> dict:
    """CPU 贪心生成 → 解码字节 + SHA256 + 文本 + 结构签名."""
    torch.manual_seed(SEED)
    net = net.to(dev)
    out = net.generate(prompt, n_tokens=n, temperature=0.0, dev=dev)
    b = bytes(out)
    text = b.decode("utf-8", errors="replace")
    dg = hashlib.sha256(b).hexdigest()
    return {
        "prompt": prompt,
        "n_tokens": n,
        "n_bytes": len(b),
        "sha256": dg,
        "decoded": text,
        "W1_shape": list(net.W1.shape),
        "param_count": sum(p.numel() for p in net.parameters()),
    }


def _learn_signature(net, steps: int, dev) -> dict:
    """CPU 固定步纯感知 learn → 关键 W 增量 SHA256 (learn 数值锚).

    learn(byte_ids, closed_loop=False, free_run=False) 纯前向+Hebbian,
    固定 seed 随机输入, 无采样随机 — fp16 CPU 逐位可复现. 哈希各 W 的
    增量(终态 - 初态), 改名/改序若动数值, 增量哈希必变.
    """
    torch.manual_seed(SEED)
    net = net.to(dev)

    def wget(n):
        return getattr(net, n).data  # 权重是 nn.Parameter/Buffer, 走 getattr 而非 __dict__

    base = {w: wget(w).float().detach().cpu().clone() for w in LEARN_W}
    for _ in range(steps):
        x = torch.randint(0, 256, (1, 8), dtype=torch.long, device=dev)
        net.learn(x, closed_loop=False, free_run=False)
    h = hashlib.sha256()
    for w in LEARN_W:
        delta = (wget(w).float().cpu() - base[w]).detach()
        h.update(delta.contiguous().view(-1).numpy().tobytes())
    return {"learn_steps": steps, "sha256": h.hexdigest()}


def _redline_scan() -> dict:
    """静态红线扫描 (R0 红线固化轮): model/ 活动源码红线 token 必须零命中.

    数值锚 (generate/learn) 只能抓行为漂移, 抓不住"不碰锚定路径"的语义
    违规 (如 L 回灌复辟) — 本面补语义级守卫. 命中即 FAIL.
    """
    hits = scan_model_sources()
    return {"hits": hits}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--prompt", default=FIXED_PROMPT)
    ap.add_argument("--n", type=int, default=FIXED_N)
    ap.add_argument("--learn-steps", type=int, default=LEARN_STEPS)
    ap.add_argument("--no-learn", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-pytest", action="store_true")
    args = ap.parse_args()

    dev = torch.device(args.device)
    net = DensePCNet.load(args.ckpt).to(dev)
    sig = _gen_signature(net, args.prompt, args.n, dev)
    lsig = None if args.no_learn else _learn_signature(net, args.learn_steps, dev)
    rsig = _redline_scan()

    pytest_out = {"skipped": True}
    if not args.no_pytest:
        pytest_out = _run_pytest()

    if not os.path.exists(args.baseline):
        doc = {
            "anchored_at": {
                "ckpt": args.ckpt,
                "prompt": args.prompt,
                "n": args.n,
                "learn_steps": args.learn_steps,
                "device": args.device,
            },
            "pytest": pytest_out,
            "golden": sig,
            "learn_golden": lsig,
            "redline": rsig,
        }
        with open(args.baseline, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        print(f"[arch_guard] 列表锚定: {args.baseline}")
        print(f"  generate: n_bytes={sig['n_bytes']} sha256={sig['sha256']}")
        print(f"  decoded={sig['decoded']!r}")
        if lsig is not None:
            print(f"  learn: steps={lsig['learn_steps']} sha256={lsig['sha256']}")
        if pytest_out.get("summary") is not None:
            print(f"  pytest: {pytest_out['summary']}")
        print(f"  redline: {'CLEAN' if not rsig['hits'] else rsig['hits']}")
        print("[arch_guard] 锚定成功 (本闸首跑). 后续每轮重跑本闸做逐位对照.")
        return 0

    with open(args.baseline, encoding="utf-8") as f:
        doc = json.load(f)
    g = doc["golden"]
    ok = (
        sig["n_bytes"] == g["n_bytes"]
        and sig["sha256"] == g["sha256"]
        and sig["W1_shape"] == g["W1_shape"]
        and sig["param_count"] == g["param_count"]
    )
    print(
        f"[arch_guard] 对照: n_bytes {sig['n_bytes']} vs {g['n_bytes']} | "
        f"sha256 {sig['sha256']} vs {g['sha256']}"
    )
    print(f"  decoded={sig['decoded']!r}")
    lg = doc.get("learn_golden")
    learn_ok = True
    if lsig is not None and lg is not None:
        learn_ok = lsig["sha256"] == lg["sha256"]
        print(f"  learn: sha256 {lsig['sha256']} vs {lg['sha256']} ({'PASS' if learn_ok else 'FAIL'})")
    elif lsig is not None:
        print(f"  learn: sha256 {lsig['sha256']} (anchor 无 learn 锚 — 需重锚定)")
        learn_ok = False
    rl = doc.get("redline")
    redline_ok = not rsig["hits"]
    print(
        f"  redline: {'CLEAN' if redline_ok else rsig['hits']}"
        + ("" if rl is not None else " (anchor 无 redline 面 — 差异留档后建议重锚)")
    )
    if rl is not None and rl.get("hits"):
        redline_ok = False  # 锚定时就已命中 → 持续违规
    pv = pytest_out.get("summary", "skipped")
    pav = doc["pytest"].get("summary", "skipped")
    print(f"  pytest: {pv} (anchor: {pav})")
    if ok and learn_ok and redline_ok:
        print("[arch_guard] PASS — generate 与 learn 逐位与锚定一致, 结构签名不变, 红线干净.")
        return 0
    print("[arch_guard] FAIL — 与锚定逐位不一致 (结构移动破坏了行为) 或红线命中.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
