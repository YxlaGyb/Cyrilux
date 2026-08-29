"""第 110 轮 P0 (D0): W_lm 裁判判别力测绘 — 纯只读, 零训练, 零改权重.

设计文档: docs/es/110_design.md §5. 测 W_lm (R 信号的实际来源) 是否具备
"语言 vs 噪声"判别力 — 修正交接文档两处口径错位:
  1. ε=1-P(下一字节) 高均值 ≠ 无区分力 (方向正确但 softmax 平的模型同样
     ε∈0.95-0.99); 判别力 = 分布差, 必须测 AUC 而非均值
  2. 107b 转移熵 3.199/3.200 测的是 z_bind (表示层上游), 本探针测喂给 R
     的 probs_lm (读出端) — 器官对位

口径: 逐行复刻 readout._build_lm_signal 前向 (z4→W_diff→zh→W1→h→W_lm→
能量调制→可打印掩码→softmax) — exp108 R 信号的产生现场, 非另造评估管线.
单次因果前向取全部位置 (时间核 shift 递归因果, 无跨位置泄漏; is_inference
关 ACh 噪声确定性).

只读保证: _bind 漂移 _theta_bind (EMA), _predict 写回 _mem_m (批均值) —
每序列前快照恢复; 权重参数零触碰, 结束时核对全部参数逐位不变.

七条件 (N_SEQ=24 序列 × S_LEN=160 字节):
  lang_train  数据集前 400 行 (训练分布内, 全部轮 max_samples≤2 万)
  lang_held   第 10 万行起 (纯 heldout)
  shuffle     lang_train 字节级打乱 (保字节频率, 破坏 UTF-8 结构+语法)
  rand_cn     top-700 常见字均匀随机排 (保多字节结构, 破坏语法)
  noise_uni   均匀随机字节 (频率+结构全破坏)
  noise_print 随机可打印 ASCII (合法字节域, 无结构)
  self_gen    W_lm 分支贪心自生成 (种子=真实文本 16 字节, 裸生成无阻断) —
              测复读吸引子概率: 若 p̄(self_gen) > p̄(lang), 差分 R 方向有毒

判据 (预注册, docs/es/110_design.md §5):
  主: AUC(lang_held vs noise_uni) >0.7 判别力存在 / ≈0.5 盲
  结构感知: AUC(lang vs shuffle) >0.7 才算看见序列结构而非频率
  毒性: p̄(self_gen) > p̄(lang) → ε_lm 差分 R 奖励自洽吸引子

用法: uv run python scripts/probe110_langnoise.py [--smoke]
"""
import argparse
import json
import math
import random
import sys
import time

import torch

torch.set_grad_enabled(False)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from model.dense import DensePCConfig, DensePCNet
from model.dense.forward import _rms

DEV = "cuda"
S_LEN = 160
N_SEQ = 24
DATA = "dataset/pretrain_t2t_mini.jsonl"
CKPTS = ["out/chat104b2.pt", "out/chat107_pool.pt", "out/exp108_say3.pt"]
FROZEN = ["W_lm", "W_lm_2", "W1", "W_04", "W_42", "W_diff", "W_bind", "W_bind_self"]


def load_common_set(path=DATA, n_lines=20000, top=700):
    import collections
    cnt = collections.Counter()
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            raw = json.loads(line).get("text", "")
            bs = raw.encode("utf-8")
            j = 0
            while j < len(bs):
                b = bs[j]
                if 0xE0 <= b <= 0xEF and j + 2 < len(bs) and 0x80 <= bs[j + 1] <= 0xBF and 0x80 <= bs[j + 2] <= 0xBF:
                    cnt[bs[j:j + 3]] += 1
                    j += 3
                else:
                    j += 1
    return list(c for c, _ in cnt.most_common(top))


def read_lines(path, start, count, min_bytes):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if len(out) >= count:
                break
            t = json.loads(line)["text"]
            if len(t.encode("utf-8")) >= min_bytes:
                out.append(t)
    return out


def snapshot_state(net):
    return {
        "_theta_bind": net._theta_bind.detach().clone(),
        "_mem_m": net._mem_m.detach().clone(),
    }


def restore_state(net, snap):
    net._theta_bind.copy_(snap["_theta_bind"])
    net._mem_m.copy_(snap["_mem_m"])


def p_target_series(net, ids):
    """一次因果前向 → [S-1] fp16: 位置 t 的 P(字节 t+1 | 字节 0..t).

    逐行复刻 readout._build_lm_signal (readout.py L40-116), 含能量调制与
    可打印掩码 — 与 exp108 wlm_err 同一产生现场.
    """
    _ = net.forward_engine._predict(ids, store_state=True, is_inference=True)
    a4 = net.active_size["l4"]
    z4 = net._z4
    d_h = net.d_h
    z4_n_ = z4 / (z4.norm(dim=-1, keepdim=True) + 1e-3)
    pred_delta_ = z4_n_ @ net.W_diff[:a4, :a4].T + net.b_diff[:a4].unsqueeze(0).unsqueeze(0)
    z4r = z4 + pred_delta_
    z4_lm = z4r / (1.0 + z4r.abs())
    z4_lm = _rms(z4_lm)
    z4_lm = z4_lm * (1.0 - 0.5 * z4_lm.pow(2))
    z4_lm = z4_lm / (1.0 + z4_lm.abs())
    zh = torch.cat([z4_lm, net._bind_vec, net._mem_out], dim=-1)
    zh = _rms(zh)
    h = zh @ net.W1
    h = h / (1.0 + h.abs())
    h = _rms(h)
    h = h * (1.0 - 0.5 * h.pow(2))
    inv_h = 1.0 / math.sqrt(d_h)
    logits_lm = (h @ net.W_lm + net.bias_lm) * inv_h
    logits_c = (logits_lm - logits_lm.mean(dim=-1, keepdim=True)) / (
        logits_lm.std(dim=-1, keepdim=True) + 1e-4
    )
    logits_lm = logits_c / logits_c.abs().max(dim=-1, keepdim=True).values * 60.0
    mask_print = torch.zeros(256, dtype=torch.float16, device=ids.device)
    mask_print[32:] = 1.0
    logits_lm = logits_lm + (1.0 - mask_print) * -1e4
    probs = torch.softmax(logits_lm, dim=-1)  # [1,S,256] fp16
    tgt = ids[0, 1:]  # [S-1]
    p = probs[0, :-1, :].gather(1, tgt.unsqueeze(1)).squeeze(1)  # [S-1]
    return p


def self_gen_stream(net, seed_bytes, n_gen=144):
    """W_lm 分支贪心自生成 (echo 语义: 裸生成, 无 rep/UTF-8 阻断)."""
    dev = next(net.parameters()).device
    seed = torch.tensor([list(seed_bytes)], dtype=torch.long, device=dev)
    saved = getattr(net, "use_w_act", False)
    net.use_w_act = False
    try:
        out = net.forward_engine.continuation(seed, n_gen, temperature=0.0, rep_backstop=False)
    finally:
        net.use_w_act = saved
    return bytes(int(v) for v in out[0].tolist())


def auc(xs, ys):
    """Mann-Whitney AUC: P(x>y) + 0.5·P(x=y). 纯 Python O(n²) (分析端)."""
    n_gt = sum(1 for x in xs for y in ys if x > y)
    n_eq = sum(1 for x in xs for y in ys if x == y)
    return (n_gt + 0.5 * n_eq) / (len(xs) * len(ys))


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def run_ckpt(net, conds, smoke=False):
    """跑全部条件, 返回 {cond: [序列级 p̄ ...]} + 窗级 {cond: [64字节窗 p̄ ...]}."""
    seq_stats = {}
    win_stats = {}
    for name, blobs in conds.items():
        seqs = blobs[:2] if smoke else blobs
        snap = snapshot_state(net)
        seq_means = []
        win_means = []
        for raw in seqs:
            ids = torch.tensor([list(raw)], dtype=torch.long, device=DEV)
            p = p_target_series(net, ids)
            restore_state(net, snap)
            pl = [float(v) for v in p.tolist()]
            seq_means.append(mean(pl))
            for w0 in range(0, len(pl) - 62, 64):
                win_means.append(mean(pl[w0:w0 + 63]))
        seq_stats[name] = seq_means
        win_stats[name] = win_means
    return seq_stats, win_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="out/probe110_langnoise.json")
    args = ap.parse_args()

    random.seed(7)
    torch.manual_seed(7)

    n_seq = 2 if args.smoke else N_SEQ
    print(f"probe110: 条件构造 (N_SEQ={n_seq}, S_LEN={S_LEN})...", flush=True)
    lang_train = [t.encode("utf-8")[:S_LEN] for t in read_lines(DATA, 0, 4000, S_LEN)][:n_seq]
    lang_held = [t.encode("utf-8")[:S_LEN] for t in read_lines(DATA, 100000, 4000, S_LEN)][:n_seq]
    common = load_common_set()
    rng = random.Random(7)

    def shuffled(b):
        l = list(b)
        rng.shuffle(l)
        return bytes(l)

    def rand_cn():
        n_char = S_LEN // 3
        return b"".join(common[rng.randrange(len(common))] for _ in range(n_char))

    def noise_uni():
        return bytes(rng.randrange(256) for _ in range(S_LEN))

    def noise_print():
        return bytes(rng.randrange(32, 127) for _ in range(S_LEN))

    conds = {
        "lang_train": lang_train,
        "lang_held": lang_held,
        "shuffle": [shuffled(b) for b in lang_train],
        "rand_cn": [rand_cn() for _ in range(n_seq)],
        "noise_uni": [noise_uni() for _ in range(n_seq)],
        "noise_print": [noise_print() for _ in range(n_seq)],
    }
    print(f"  lang_train={len(lang_train)} lang_held={len(lang_held)} "
          f"common={len(common)} (top700)", flush=True)

    report = {"s_len": S_LEN, "n_seq": n_seq, "checkpoints": {}}
    t0 = time.time()

    # 冻结不变性: chat107_pool vs exp108_say3 (108 世界模型全程冻结的验证)
    if not args.smoke:
        cfg = DensePCConfig(d_input=256, d_act=256, max_seq_len=256)
        n_a = DensePCNet.load("out/chat107_pool.pt", cfg).to(DEV)
        n_b = DensePCNet.load("out/exp108_say3.pt", cfg).to(DEV)
        diffs = {}
        for n_ in FROZEN:
            diffs[n_] = float((getattr(n_a, n_) - getattr(n_b, n_)).abs().max().item())
        report["freeze_check"] = diffs
        print(f"freeze-check (pool vs say3): " + ", ".join(
            f"{k}={v:.1e}" for k, v in diffs.items()), flush=True)
        del n_a, n_b
        torch.cuda.empty_cache()

    for ck in CKPTS:
        cfg = DensePCConfig(d_input=256, d_act=256, max_seq_len=256)
        net = DensePCNet.load(ck, cfg).to(DEV)
        net.use_w_act = False
        w0 = {n_: getattr(net, n_).detach().clone() for n_ in FROZEN}

        # self_gen: 每条件独立 (依赖该检查点的生成流)
        snap = snapshot_state(net)
        conds["self_gen"] = [self_gen_stream(net, lang_train[i][:16]) for i in range(n_seq)]
        restore_state(net, snap)

        tc = time.time()
        seq_stats, win_stats = run_ckpt(net, conds, smoke=args.smoke)

        # 只读不变量: 权重逐位不变
        moved = [n_ for n_ in FROZEN if not torch.equal(getattr(net, n_), w0[n_])]
        entry = {"seq": seq_stats, "win": win_stats, "params_moved": moved}
        aucs = {
            "main: lang_held>noise_uni": auc(seq_stats["lang_held"], seq_stats["noise_uni"]),
            "lang_train>noise_uni": auc(seq_stats["lang_train"], seq_stats["noise_uni"]),
            "structure: lang_train>shuffle": auc(seq_stats["lang_train"], seq_stats["shuffle"]),
            "structure: lang_held>shuffle": auc(seq_stats["lang_held"], seq_stats["shuffle"]),
            "freq: shuffle>noise_uni": auc(seq_stats["shuffle"], seq_stats["noise_uni"]),
            "freq: rand_cn>noise_uni": auc(seq_stats["rand_cn"], seq_stats["noise_uni"]),
            "print: noise_print>noise_uni": auc(seq_stats["noise_print"], seq_stats["noise_uni"]),
            "toxic: self_gen>lang_train": auc(seq_stats["self_gen"], seq_stats["lang_train"]),
            "toxic: self_gen>lang_held": auc(seq_stats["self_gen"], seq_stats["lang_held"]),
            "win main: lang_held>noise_uni": auc(win_stats["lang_held"], win_stats["noise_uni"]),
        }
        entry["auc"] = aucs
        report["checkpoints"][ck] = entry

        print(f"\n== {ck} ({time.time()-tc:.0f}s) ==", flush=True)
        for name in ("lang_train", "lang_held", "shuffle", "rand_cn", "noise_print", "noise_uni", "self_gen"):
            s = seq_stats[name]
            print(f"  {name:<12} p̄={mean(s):.4f} std={stdev(s):.4f} (n={len(s)})", flush=True)
        for k, v in aucs.items():
            print(f"  AUC {k:<34} {v:.3f}", flush=True)
        print(f"  params_moved={moved}", flush=True)
        del net
        torch.cuda.empty_cache()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\n[probe110] 完成 {time.time()-t0:.0f}s → {args.out}", flush=True)


if __name__ == "__main__":
    main()
