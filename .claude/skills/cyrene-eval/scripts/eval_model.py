"""Cyrene 模型综合评估脚本.
Usage:
    python eval_model.py out/model8/final.pt
    python eval_model.py out/model8/final.pt --dataset dataset/agent_rl_math.jsonl
    python eval_model.py out/model8/final.pt --skip-ppl --skip-generation
"""

import argparse
import json
import math
import time
import torch
from model.model_cyrene import CyreneModel

torch.set_grad_enabled(False)


def load_model(path):
    t0 = time.perf_counter()
    m = CyreneModel.load(path)
    elapsed = time.perf_counter() - t0
    a = m.pool.alive
    layers = {}
    for L in sorted((m.pool.layer[a]).unique().tolist()):
        layers[L] = ((m.pool.layer == L) & a).sum().item()
    syns = m.pool.syn_alive.sum().item()
    print(
        f"  {path}: step={m._step} neurons={a.sum().item()} syn={syns} top_layer={
            m._top_layer
        } load={elapsed:.1f}s"
    )
    for L, n in layers.items():
        print(f"    L{L}: {n} neurons")

    # 检查 lm_bias 状态
    bias_nz = (m.pool.lm_bias != 0).sum().item()
    print(
        f"    lm_bias non-zero: {bias_nz}/256  lm_weight norm: {
            m.pool.lm_weight.norm(dim=0).mean():.4f}"
    )
    return m


def _byte_tensor(text: str, device):
    """将文本转为 [1, S] long tensor."""
    return torch.tensor([list(text.encode("utf-8"))], dtype=torch.long, device=device)


def test_sensitivity(m, texts=None):
    if texts is None:
        texts = [
            "Hello world! How are you today?",
            "import torch; def foo(x): return x*2",
            "123 + 456 = 579, 789 * 10 = 7890",
            "人工智能的未来在于",
        ]
    print("\n  [敏感度]")
    for text in texts:
        bv = _byte_tensor(text, m.device)
        m.reset_hidden_state()
        m.step(bv)
        logits_t = m.pool.forward.compute_lm_logits(m._top_layer, use_mu=True).float()
        probs = torch.softmax(logits_t, dim=-1)
        ent = -(probs * (probs + 1e-12).log()).sum().item()
        top3 = torch.topk(probs, 3)
        items = [
            (chr(i) if 32 <= i < 127 else f"0x{i:02x}", f"{p:.3f}")
            for i, p in zip(top3.indices.tolist(), top3.values.tolist())
        ]
        print(f"    {text[:40]:40s} ent={ent:.2f} top3={items}")


def test_ppl(m, data_path="dataset/sft_t2t.jsonl", max_samples=15, max_tokens=500):
    print(f"\n  [PPL] {data_path}")
    texts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(texts) >= max_samples:
                break
            item = json.loads(line)
            t = item.get("text", item.get("chosen", ""))
            if len(t) < 20:
                continue
            texts.append(t)

    if not texts:
        print("    no data -- skipped")
        return 0, 0, 0

    loss_sum = 0.0
    n = 0
    correct = 0
    t0 = time.perf_counter()
    for text in texts:
        m.reset_hidden_state()
        bs = text.encode("utf-8")
        if len(bs) < 14:
            continue
        for pos in range(1, min(len(bs), max_tokens - n + 1)):
            ctx = bs[:pos]
            if len(ctx) < 13:
                continue
            tgt = bs[pos]
            bv = torch.tensor([list(ctx)], dtype=torch.long, device=m.device)
            m.step(bv)
            logits_t = m.pool.forward.compute_lm_logits(m._top_layer, use_mu=True).float()
            loss_sum += float(
                torch.nn.functional.cross_entropy(
                    logits_t.unsqueeze(0), torch.tensor([tgt], device=m.device)
                ).item()
            )
            if int(logits_t.argmax().item()) == tgt:
                correct += 1
            n += 1
            if n >= max_tokens:
                break
        if n >= max_tokens:
            break

    elapsed = time.perf_counter() - t0
    ppl = math.exp(loss_sum / n) if loss_sum / n < 50 else float("inf")
    print(
        f"    tokens={n} PPL={ppl:.1f} CE={loss_sum / n:.2f} top1={100 * correct / n:.1f}%  speed={n / elapsed:.0f} tok/s"
    )
    return ppl, loss_sum / n, 100 * correct / n


def test_memory(m, text=None, repeats=3):
    if text is None:
        text = "Hello, world! This is a test of memory retention."
    print(f'\n  [记忆] "{text[:40]}..."')
    bs = list(text.encode("utf-8"))
    for rep in range(repeats):
        m.reset_hidden_state()
        correct = 0
        total = 0
        for i in range(min(len(bs) - 1, 25)):
            ctx = bs[: i + 1]
            if len(ctx) < 13:
                continue
            tgt = bs[i + 1]
            bv = torch.tensor([ctx], dtype=torch.long, device=m.device)
            m.step(bv)
            logits_t = m.pool.forward.compute_lm_logits(m._top_layer, use_mu=True).float()
            if int(logits_t.argmax().item()) == tgt:
                correct += 1
            total += 1
        print(f"    round {rep + 1}: {correct}/{total} = {100 * correct / total:.0f}%")


def test_generation(m, prompts=None, n_tokens=40):
    if prompts is None:
        prompts = ["Hello", "import ", "The quick"]
    print("\n  [生成]")
    for prompt in prompts:
        m.reset_hidden_state()
        gen = list(prompt.encode("utf-8"))
        for _ in range(n_tokens):
            bv = torch.tensor([gen[-64:]], dtype=torch.long, device=m.device)
            m.step(bv)
            logits_t = m.pool.forward.compute_lm_logits(m._top_layer, use_mu=True).float() / 0.7
            topv, _ = torch.topk(logits_t, min(15, 256))
            logits_t[logits_t < topv[-1]] = -float("inf")
            probs = torch.softmax(logits_t, dim=-1)
            gen.append(int(torch.multinomial(probs, 1).item()))
        text = bytes(gen).decode("utf-8", errors="replace")
        safe = text[:50].encode("ascii", errors="replace").decode("ascii")
        print(f'    "{prompt}" -> "{safe}"')


def test_selectivity(m, texts=None):
    if texts is None:
        texts = [
            ("EN", "hello world python code english text example"),
            ("CODE", "import torch def foo return x + 1 class Model"),
            ("CN", "人工智能的未来在于深度学习"),
        ]
    print("\n  [选择性]")
    results = {}
    for label, text in texts:
        m.reset_hidden_state()
        bv = _byte_tensor(text, m.device)
        m.step(bv)
        top_mask = (m.pool.layer == m._top_layer) & m.pool.alive
        z = m.pool.state[top_mask, 0].float().clone()
        mu = m.pool.state[top_mask, 1].float().clone()
        results[label] = (z, mu)

    labels = list(results.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            z1, mu1 = results[labels[i]]
            z2, mu2 = results[labels[j]]
            cos_z = torch.nn.functional.cosine_similarity(z1.unsqueeze(0), z2.unsqueeze(0)).item()
            cos_mu = torch.nn.functional.cosine_similarity(
                mu1.unsqueeze(0), mu2.unsqueeze(0)
            ).item()
            print(
                f"    {labels[i]} vs {labels[j]}: cos(Z)={cos_z:.4f} cos(MU)={cos_mu:.4f} gap={cos_z - cos_mu:+.4f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="模型路径")
    parser.add_argument("--dataset", default="dataset/sft_t2t.jsonl")
    parser.add_argument("--ppl-samples", type=int, default=15)
    parser.add_argument("--ppl-tokens", type=int, default=500)
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-selectivity", action="store_true")
    args = parser.parse_args()

    all_ppl = []
    for path in args.checkpoints:
        print(f"\n{'=' * 60}")
        print(f"评估: {path}")
        print(f"{'=' * 60}")
        m = load_model(path)

        if not args.skip_sensitivity:
            test_sensitivity(m)
        if not args.skip_ppl:
            ppl, ce, top1 = test_ppl(m, args.dataset, args.ppl_samples, args.ppl_tokens)
            all_ppl.append((path, m._step, ppl, ce, top1))
        if not args.skip_memory:
            test_memory(m)
        if not args.skip_generation:
            test_generation(m)
        if not args.skip_selectivity:
            test_selectivity(m)

    if len(all_ppl) > 1:
        print(f"\n{'=' * 60}")
        print("对比汇总")
        print(f"{'=' * 60}")
        print(f"{'Model':<30} {'Step':<10} {'PPL':<10} {'CE':<8} {'Top-1':<8}")
        print("-" * 60)
        for path, step, ppl, ce, top1 in all_ppl:
            name = path.split("/")[-1].replace(".pt", "")
            print(f"{name:<30} {step:<10} {ppl:<10.1f} {ce:<8.2f} {top1:<8.1f}%")


if __name__ == "__main__":
    main()
