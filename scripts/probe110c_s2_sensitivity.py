"""probe110c: S2 敏感度 — 意图通路对发声的实际影响 (只读, 零改权重).

设计文档 §6 证伪线: "发声对内部状态扰动的敏感度 > 0". D1 后 W_act 降格为
意图调制器 (z_bind@W_act → logits ≤15% 偏置) — 本探针测该通路是否真实
影响生成流. 若影响率为零, "自我进入声音"失败, 方案证伪.

方法 (采样确定性控制):
  1. 固定 torch seed → continuation 正常 (W_act 原始) → stream_a
  2. 同 seed → W_act 临时置零 (意图通路断开) → stream_b
  3. 影响率 = mean(stream_a != stream_b)
  对照: 同 seed 两次正常生成 (验证采样确定性, 应 = 0)
       不同 seed 两次正常生成 (采样噪声底)

附: pot_int/W_lm logits 能量占比 (意图电位在 logits 中的统计占比).

用法: uv run python scripts/probe110c_s2_sensitivity.py [--ckpt out/exp110_say.pt]
"""
import argparse
import importlib.util
import sys

import torch

torch.set_grad_enabled(False)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from model.dense import DensePCConfig, DensePCNet

_spec = importlib.util.spec_from_file_location("probe110", "scripts/probe110_langnoise.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["probe110"] = _mod
_spec.loader.exec_module(_mod)
read_lines = _mod.read_lines
DEV = _mod.DEV

N_SEQ = 16
N_GEN = 63
TEMP = 4.0


def gen(net, seed_bytes, temp=TEMP):
    dev = next(net.parameters()).device
    seed = torch.tensor([list(seed_bytes)], dtype=torch.long, device=dev)
    out = net.forward_engine.continuation(seed, N_GEN, temperature=temp, rep_backstop=True)
    return bytes(int(v) for v in out[0, len(seed_bytes):].tolist())


def diff_rate(a, b):
    return sum(1 for x, y in zip(a, b) if x != y) / max(1, len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/exp110_say.pt")
    ap.add_argument("--data", default=_mod.DATA)
    args = ap.parse_args()

    cfg = DensePCConfig(d_input=256, d_act=256, max_seq_len=256)
    net = DensePCNet.load(args.ckpt, cfg).to(DEV)
    net.use_w_act = False
    net._entropy_sample = False

    texts = read_lines(args.data, 0, 100, 160)
    seeds = [t.encode("utf-8")[:16] for t in texts[:N_SEQ]]

    # 1. 采样确定性验证 (同 seed 两次正常生成)
    torch.manual_seed(123)
    a1 = [gen(net, s) for s in seeds]
    torch.manual_seed(123)
    a2 = [gen(net, s) for s in seeds]
    det = sum(diff_rate(x, y) for x, y in zip(a1, a2)) / N_SEQ

    # 2. 采样噪声底 (不同 seed)
    torch.manual_seed(999)
    a3 = [gen(net, s) for s in seeds]
    noise = sum(diff_rate(x, y) for x, y in zip(a1, a3)) / N_SEQ

    # 3. 意图影响 (同 seed, W_act 置零 — 意图通路断开)
    w_act_saved = net.W_act.detach().clone()
    net.W_act.data.zero_()
    torch.manual_seed(123)
    b1 = [gen(net, s) for s in seeds]
    net.W_act.data.copy_(w_act_saved)
    intent = sum(diff_rate(x, y) for x, y in zip(a1, b1)) / N_SEQ

    # 4. 意图电位能量占比 (logits 层面)
    ratio = []
    snap_ok = True
    try:
        for s in seeds[:8]:
            dev = next(net.parameters()).device
            seed = torch.tensor([list(s)], dtype=torch.long, device=dev)
            _ = net.forward_engine._predict(seed, store_state=True, is_inference=True)
            pot_int = net._bind_vec[:, -1] @ net.W_act
            ratio.append(float((pot_int.norm() / (pot_int.norm() + 60.0)).item()))
    except Exception as e:
        snap_ok = False
        print(f"(能量占比测量跳过: {e})", flush=True)

    print(f"采样确定性 (同seed×2):   diff={det:.4f}  (应=0)", flush=True)
    print(f"采样噪声底 (异seed):     diff={noise:.4f}", flush=True)
    print(f"意图影响 (W_act 置零):   diff={intent:.4f}  ← S2 敏感度", flush=True)
    if snap_ok:
        print(f"意图电位占比:            {sum(ratio)/len(ratio):.4f} (pot_int/(pot_int+60))", flush=True)

    verdict = "非零 — 意图通路有效" if intent > noise * 1.5 and intent > 0.05 else (
        "零/噪声级 — 方案证伪线触发" if intent < 0.05 else "弱 — 边缘状态")
    print(f"\n[probe110c] S2 判定: {verdict}", flush=True)


if __name__ == "__main__":
    main()
