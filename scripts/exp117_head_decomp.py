"""
头 bpb 分解: 真文本 vs 模型自生成文本.

冻结 v244 终态. 同一读出头 (readout.py:79-108), 同一 bpb 口径, 分别在
(1) held-out 真实窗集 (2) 模型自身生成流 上打分, 并配 unigram/bigram 基线.

判别: exp116 已证头在真文本 bpb=6.73 > unigram 4.77 (学坏了). 本步隔离两假说 —
  (a) 学习规则太弱: 头连"自身生成分布"都拟合不了 → 头在自生成上也 > unigram_self;
  (b) 塌缩污染 target: 头拟合了自生成 (塌缩) 分布 → 头在自生成上 ≪ unigram_self,
      但在真文本上比边缘分布还差.
基线归属: unigram_T/bigram_T 拟合真实训练字节; unigram_S/bigram_S 拟合自生成字节.

口径:
  - 头读出方程与评分复用 exp116 (单一仪器), 与 v246 锚点 (6.7316) 逐位可比.
  - 两个头并列: mig = W1+W1_new (readout.py:79-108); red = 仅 W1 (exp115/_gen_logit_mode="old").
  - 自生成 = forward_engine.continuation (回声/训练自生成路径), 单流 N=1, 温度 tensor 罩 CUDA Graph.
  - 真实/自生成两次抽取各用全新实例 (零状态污染; exp115 _sample_dump 同款).
"""
import argparse
import json
import time

import torch
from exp115_say import N_EVAL_WIN, GraphedContinuation, GraphedPredict, build_gauge
from exp116_z4_probe import S_GRID, _md5, bpb_only, extract, score_logits

from crates.cli.utils import resolve_path, run_file
from crates.model import CyreneModel, HybridPCNet

torch.set_grad_enabled(False)


def load_fresh(ckpt, cfg, dev):
    net = HybridPCNet.load(ckpt, cfg).to(dev)
    net.eval()
    return net


def gen_one(net, dev, seed, s0, n_gen, temperature, gc=None):
    ids = seed[:s0].unsqueeze(0).to(dev).long()
    if gc is None:
        out = net.forward_engine.continuation(
            ids, n_gen, temperature=temperature, rep_backstop=False)
    else:
        out = gc(ids, n_gen, temperature=temperature, rep_backstop=False)
    return out[0, s0:].clone().cpu().long()


def gen_self_streams(net, dev, seed_wins, s0, n_gen, temperature, use_graph):
    gc = GraphedContinuation(net.forward_engine, net) if use_graph else None
    temp = torch.tensor([temperature], dtype=torch.float16, device=dev) if use_graph \
        else float(temperature)
    streams, t0 = [], time.time()
    for i, w in enumerate(seed_wins):
        streams.append(gen_one(net, dev, w, s0, n_gen, temp, gc))
        if (i + 1) % 16 == 0:
            print(f"  [self] {i + 1}/{len(seed_wins)} 流 ({time.time() - t0:.0f}s)", flush=True)
    return streams


def extract_fresh(ckpt, cfg, dev, wins, tag):
    net = load_fresh(ckpt, cfg, dev)
    gp = GraphedPredict(net.forward_engine, net)
    net.forward_engine._predict = gp
    try:
        _, y, RO = extract(net, dev, wins, tag, want_readout=True)
    finally:
        net.forward_engine._predict = gp.raw
        gp.graphs.clear()
        del net
        torch.cuda.empty_cache()
    return y, RO


def fit_unigram(y):
    cnt = torch.bincount(y, minlength=256).to(torch.float16)
    p = cnt / int(cnt.sum())
    p = p.clamp_min(1.0 / int(2 * cnt.sum()))
    return torch.log(p)


def fit_bigram(wins, k=0.1):
    cnt = torch.zeros(256 * 256, dtype=torch.long)
    for w in wins:
        idx = w[:-1].long() * 256 + w[1:].long()
        cnt += torch.bincount(idx, minlength=256 * 256)
    cnt = cnt.reshape(256, 256)
    c16 = cnt.to(torch.float16)
    p = (c16 + k) / (c16.sum(dim=1, keepdim=True) + 256 * k)
    return torch.log(p)


def score_unigram(uni_logits, y):
    lg = uni_logits.unsqueeze(0).expand(y.numel(), -1)
    return score_logits(lg, y, scale=1.0)


def score_bigram(logp2, wins):
    lgs, ys = [], []
    for w in wins:
        lgs.append(logp2[w[:-1].long()])
        ys.append(w[1:].long())
    return score_logits(torch.cat(lgs), torch.cat(ys), scale=1.0)


def with_s1(r):
    return {**r, "bpb_at_s1": r["bpb"]}


def head_scores(logits, y):
    s1 = score_logits(logits, y, scale=1.0)
    per_s = {str(s): bpb_only(logits, y, scale=s) for s in S_GRID}
    best_s = min(S_GRID, key=lambda s: per_s[str(s)])
    return {"bpb_at_s1": s1["bpb"], "top1": s1["top1"], "top15": s1["top15"],
            "rank_median": s1["rank_median"], "rank_mean": s1["rank_mean"], "n": s1["n"],
            "best_s": best_s, "bpb_at_best_s": per_s[str(best_s)], "bpb_per_s": per_s}


def pred_marginal(logits, topn=12):
    p = torch.softmax(logits, dim=-1).mean(dim=0)
    v, i = p.topk(topn)
    return {"bytes": i.tolist(), "probs": [round(float(x), 6) for x in v]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/v244-20261009-214217/e9f_bplus_readout_realign_500.safetensors")
    ap.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.85)
    ap.add_argument("--streams", type=int, default=64)
    ap.add_argument("--seed-len", type=int, default=16)
    ap.add_argument("--gen-len", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dev = torch.device(args.device)
    ckpt = resolve_path(args.ckpt)
    md5 = _md5(ckpt)
    print(f"exp117: ckpt={ckpt}\n  md5={md5}", flush=True)

    cfg = CyreneModel(d_input=256, d_act=256, max_seq_len=256,
                      lm_freeze_w1=False)
    net = load_fresh(ckpt, cfg, dev)
    print(f"exp117: dim_4={net.active_size['l4']} d_h={net.d_h} "
          f"migrate={getattr(net, '_migrate_expand', False)} w1v2={getattr(net, '_w1v2', False)} "
          f"cont_cur={tuple(net._cont_cur.shape)}", flush=True)

    if args.smoke:
        w = torch.arange(args.seed_len, dtype=torch.long)
        t0 = time.time()
        gen_one(net, dev, w, args.seed_len, 64, float(args.temperature), None)
        print(f"exp117 smoke: eager 64 步 {time.time() - t0:.2f}s", flush=True)
        gc = GraphedContinuation(net.forward_engine, net)
        temp = torch.tensor([args.temperature], dtype=torch.float16, device=dev)
        t0 = time.time()
        gen_one(net, dev, w, args.seed_len, 256, temp, gc)
        c1 = time.time() - t0
        t0 = time.time()
        gen_one(net, dev, w, args.seed_len, 256, temp, gc)
        print(f"exp117 smoke: graph 捕获+256 步 {c1:.2f}s, 回放 256 步 {time.time() - t0:.2f}s "
              f"disabled={gc.disabled}", flush=True)
        return

    tg = time.time()
    gauge, _, _ = build_gauge(args.data, dev)
    tail, trunc = gauge["tail"], gauge["trunc"]
    real_wins = tail[:N_EVAL_WIN]
    fit_pool = trunc + tail[N_EVAL_WIN:]
    order = torch.randperm(len(fit_pool), generator=torch.Generator().manual_seed(args.seed)).tolist()
    fit_pool = [fit_pool[i] for i in order]
    train_wins = fit_pool[: int(len(fit_pool) * args.train_frac)]
    print(f"exp117: gauge tail={len(tail)} trunc={len(trunc)} ({time.time() - tg:.0f}s); "
          f"real test={len(real_wins)} train={len(train_wins)}", flush=True)

    seed_src = tail[N_EVAL_WIN:N_EVAL_WIN + args.streams]
    assert len(seed_src) >= args.streams, "自生成种子窗不足"
    self_streams = gen_self_streams(net, dev, seed_src, args.seed_len, args.gen_len,
                                    args.temperature, not args.no_graph)
    del net
    torch.cuda.empty_cache()
    self_wins = self_streams
    self_bytes = torch.cat(self_wins)
    uniq_w = len({tuple(int(x) for x in w.tolist()) for w in self_wins})
    cnt_b = torch.bincount(self_bytes, minlength=256)
    top_b = int(cnt_b.argmax())
    q = cnt_b.to(torch.float16) / int(cnt_b.sum())
    ent = float(-(q.clamp_min(6e-8).log2() * q).sum())
    snippet = bytes(self_wins[0][:64].to(torch.uint8).numpy().tolist()).decode("utf-8", errors="replace")
    print(f"exp117: self-gen {args.streams} 流 × {args.gen_len}B → {len(self_wins)} 窗; "
          f"不同窗={uniq_w} 不同字节={int((cnt_b > 0).sum())} "
          f"最高频字节=0x{top_b:02X}({float(cnt_b[top_b]) / cnt_b.sum():.3f}) 熵={ent:.2f}bit",
          flush=True)
    print(f"exp117: self 样本 #{1}: {snippet!r}", flush=True)

    yr, ROr = extract_fresh(ckpt, cfg, dev, real_wins, "real")
    ys, ROs = extract_fresh(ckpt, cfg, dev, self_wins, "self")
    print(f"exp117: extract real={tuple(yr.shape)} self={tuple(ys.shape)}", flush=True)

    yr_tr = torch.cat([w[1:].long() for w in train_wins])
    uni_T, big_T = fit_unigram(yr_tr), fit_bigram(train_wins)
    uni_S, big_S = fit_unigram(ys), fit_bigram(self_wins)

    head_mig = {"real": head_scores(ROr["mig"], yr), "self": head_scores(ROs["mig"], ys)}
    head_red = {"real": head_scores(ROr["red"], yr), "self": head_scores(ROs["red"], ys)}
    uT_real, uT_self = with_s1(score_unigram(uni_T, yr)), with_s1(score_unigram(uni_T, ys))
    uS_real, uS_self = with_s1(score_unigram(uni_S, yr)), with_s1(score_unigram(uni_S, ys))
    bT_real, bT_self = with_s1(score_bigram(big_T, real_wins)), with_s1(score_bigram(big_T, self_wins))
    bS_real, bS_self = with_s1(score_bigram(big_S, real_wins)), with_s1(score_bigram(big_S, self_wins))

    def _line(tag, r, s, extra=""):
        print(f"exp117: {tag:13s} real bpb={r['bpb_at_s1']:.4f} top1={r['top1']:.4f} "
              f"| self bpb={s['bpb_at_s1']:.4f} top1={s['top1']:.4f} {extra}", flush=True)

    _line("head[W1+new]", head_mig["real"], head_mig["self"],
          f"| best_s real={head_mig['real']['best_s']}({head_mig['real']['bpb_at_best_s']:.4f}) "
          f"self={head_mig['self']['best_s']}({head_mig['self']['bpb_at_best_s']:.4f})")
    _line("head[W1 only]", head_red["real"], head_red["self"],
          f"| best_s real={head_red['real']['best_s']}({head_red['real']['bpb_at_best_s']:.4f}) "
          f"self={head_red['self']['best_s']}({head_red['self']['bpb_at_best_s']:.4f})")
    _line("unigram_T", uT_real, uT_self)
    _line("bigram_T", bT_real, bT_self)
    _line("unigram_S", uS_real, uS_self)
    _line("bigram_S", bS_real, bS_self)

    pm_head = pred_marginal(ROr["mig"])
    pm_uS = pred_marginal(uni_S.unsqueeze(0))
    pm_uT = pred_marginal(uni_T.unsqueeze(0))
    for tag, pm in (("head@real", pm_head), ("unigram_T", pm_uT), ("unigram_S", pm_uS)):
        print(f"exp117: {tag:10s} top: {pm['bytes'][:6]} p={[round(x, 3) for x in pm['probs'][:6]]}",
              flush=True)

    hm_real, hm_self = head_mig["real"], head_mig["self"]
    hr_real, hr_self = head_red["real"], head_red["self"]
    fits_self = min(hm_self["bpb_at_s1"], hr_self["bpb_at_s1"]) < uS_self["bpb_at_s1"] - 0.5
    bad_on_real = min(hm_real["bpb_at_s1"], hr_real["bpb_at_s1"]) > uT_real["bpb_at_s1"]
    if bad_on_real and fits_self:
        verdict = "b: 头贴合自生成(塌缩)分布, 但对真文本比边缘分布还差 → 塌缩污染 target"
    elif bad_on_real and not fits_self:
        verdict = "a: 头对自生成分布也不贴合 (head_self ≥ unigram_S), 两文本皆差 → 学习规则/路径太弱"
    else:
        verdict = "不明: head 未比 unigram_T 差, 与 exp116 锚点不符"

    out = {
        "ckpt": args.ckpt, "ckpt_md5": md5, "data": args.data, "seed": args.seed,
        "head_basis": {"mig": "readout.py:79-108 (W1 + W1_new) raw logits",
                       "red": "readout.py:82-94 仅 W1 (exp115 _readout_lc / _gen_logit_mode=old 同构)"},
        "real": {"n_windows": len(real_wins), "source": f"gauge.tail[0:{N_EVAL_WIN}]"},
        "self": {"n_streams": args.streams, "seed_len": args.seed_len, "gen_len": args.gen_len,
                 "temperature": args.temperature, "rep_backstop": False,
                 "seed_source": f"gauge.tail[{N_EVAL_WIN}:{N_EVAL_WIN + args.streams}] 前 {args.seed_len}B",
                 "basis": "forward_engine.continuation (回声/训练自生成路径), 丢弃种子",
                 "total_bytes": int(self_bytes.numel()), "n_windows": len(self_wins),
                 "distinct_windows": uniq_w, "distinct_bytes": int((cnt_b > 0).sum()),
                 "top_byte": top_b, "top_byte_freq": float(cnt_b[top_b]) / float(cnt_b.sum()),
                 "stream_entropy_bits": ent, "sample": snippet},
        "head_mig": head_mig, "head_red": head_red,
        "unigram_T": {"basis": "train split 目标字节计数 (exp116 clamp_min 同构)",
                      "real": uT_real, "self": uT_self},
        "bigram_T": {"basis": "train split (prev,next) 计数 k=0.1", "real": bT_real, "self": bT_self},
        "unigram_S": {"basis": "自生成目标字节计数", "real": uS_real, "self": uS_self},
        "bigram_S": {"basis": "自生成 (prev,next) 计数 k=0.1", "real": bS_real, "self": bS_self},
        "head_pred_marginal_on_real": pm_head,
        "unigram_T_top": pm_uT, "unigram_S_top": pm_uS,
        "verdict": {"head_fits_self": fits_self, "head_bad_on_real": bad_on_real, "text": verdict},
    }
    path = run_file("head_decomp.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"exp117: 写出 {path}", flush=True)
    print(f"exp117: 判定 = {verdict}", flush=True)


if __name__ == "__main__":
    main()
