"""DensePCNet evaluation — language ability, memory, selectivity."""
import math
import time
import torch
from model.pc.dense.core import DensePCNet, DensePCConfig

torch.set_grad_enabled(False)


def load_model(path):
    t0 = time.perf_counter()
    sd = torch.load(path, map_location="cpu", weights_only=True)
    # infer config from shapes
    d_pe = sd["pos_encoding"].shape[-1]
    d_l4 = sd["W_04"].shape[0]
    d_l2 = sd["W_42"].shape[0]
    d_l3 = sd["W_23"].shape[0]
    d_l5 = sd["W_35"].shape[0]
    d_l6 = sd["W_56"].shape[0]
    S = sd["pos_encoding"].shape[0]
    d_input = sd["W_04"].shape[1] - d_pe
    cfg = DensePCConfig(d_input=d_input, d_pe=d_pe,
                        d_l4=d_l4, d_l2=d_l2, d_l3=d_l3,
                        d_l5=d_l5, d_l6=d_l6, max_seq_len=S)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DensePCNet(cfg, max_seq_len=S).to(dev)
    # skip BCM shape mismatch
    for k in list(sd.keys()):
        if k in ("bcm_slope", "bcm_zero") and sd[k].shape != net.state_dict()[k].shape:
            sd.pop(k)
    net.load_state_dict(sd, strict=False)
    # infer active_size from actual pruned shapes
    net.active_size = {
        "l4": net.W_04.shape[0],
        "l2": net.W_42.shape[0],
        "l3": net.W_23.shape[0],
        "l5": net.W_35.shape[0],
        "l6": net.W_56.shape[0],
    }
    # reset death_row to match pruned sizes
    for k in ("l4", "l2", "l3", "l5", "l6"):
        a = net.active_size[k]
        net._death_row[k] = torch.zeros(a, dtype=torch.int8, device=dev)
        net._probation_counter[k] = torch.zeros(a, dtype=torch.int16, device=dev)
    elapsed = time.perf_counter() - t0
    d = cfg.dims()
    print(f"  {path}: L4={d['l4']} L2={d['l2']} L3={d['l3']} "
          f"L5={d['l5']} L6={d['l6']}  params={cfg.param_count():,}  "
          f"load={elapsed:.1f}s")
    bias_nz = (net.bias_lm != 0).sum().item()
    print(f"    lm_bias non-zero: {bias_nz}/256  lm_weight norm: {net.W_LM.float().norm(dim=0).mean():.4f}")
    as_sz = net.active_size
    print(f"    active_size: l4={as_sz['l4']} l2={as_sz['l2']} l3={as_sz['l3']} l5={as_sz['l5']} l6={as_sz['l6']}")
    return net, cfg


def _byte_tensor(text: str, dev) -> torch.Tensor:
    """[1, S] long."""
    return torch.tensor([list(text.encode("utf-8"))], dtype=torch.long, device=dev)


def test_sensitivity(net, texts=None):
    if texts is None:
        texts = [
            "Hello world! How are you today?",
            "import torch; def foo(x): return x*2",
            "123 + 456 = 579, 789 * 10 = 7890",
            "人工智能的未来在于",
        ]
    print("\n  [Sensitivity]")
    for text in texts:
        bv = _byte_tensor(text, next(net.parameters()).device)
        logits = net(bv)  # [1, S, 256]
        probs = torch.softmax(logits[0, -1].float(), dim=-1)
        ent = -(probs * (probs + 1e-12).log()).sum().item()
        top3 = torch.topk(probs, 3)
        items = [
            (chr(i) if 32 <= i < 127 else f"0x{i:02x}", f"{p:.3f}")
            for i, p in zip(top3.indices.tolist(), top3.values.tolist())
        ]
        # average entropy across all positions too
        all_probs = torch.softmax(logits[0].float(), dim=-1)  # [S, 256]
        avg_ent = (-(all_probs * (all_probs + 1e-12).log()).sum(dim=-1)).mean().item()
        print(f"    {text[:40]:40s} last_ent={ent:.2f} avg_ent={avg_ent:.2f} top3_last={items}")


def test_ppl(net, cfg, data_path="dataset/sft_t2t.jsonl", max_samples=15, max_tokens=500):
    print(f"\n  [PPL] {data_path}")
    texts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(texts) >= max_samples:
                break
            item = __import__("json").loads(line)
            t = ""
            if "conversations" in item:
                gt = item.get("gt", None)
                for m in item["conversations"]:
                    content = m.get("content", m.get("value", ""))
                    t += content
                if gt and isinstance(gt, list) and any(g for g in gt if g):
                    answers = "\n".join(str(g) for g in gt if g)
                    t += "\n" + answers
            if not t:
                t = item.get("text", item.get("chosen", ""))
            if len(t) < 10:
                continue
            texts.append(t)

    if not texts:
        print("    no data -- skipped")
        return 0, 0, 0

    dev = next(net.parameters()).device
    loss_sum = 0.0
    n = 0
    correct = 0
    t0 = time.perf_counter()
    for text in texts:
        bs = text.encode("utf-8")
        if len(bs) < 14:
            continue
        for pos in range(1, min(len(bs), max_tokens - n + 1)):
            ctx = bs[:pos]
            if len(ctx) < 13:
                continue
            # 截断到模型 pos_encoding 最大长度
            ctx = ctx[-cfg.max_seq_len:]
            tgt = bs[pos]
            bv = torch.tensor([list(ctx)], dtype=torch.long, device=dev)
            logits_t = net(bv)  # [1, S, 256]
            last_logits = logits_t[0, -1].float()
            loss_val = torch.nn.functional.cross_entropy(
                last_logits.unsqueeze(0),
                torch.tensor([tgt], device=dev)
            ).item()
            loss_sum += loss_val
            if int(last_logits.argmax().item()) == tgt:
                correct += 1
            n += 1
            if n >= max_tokens:
                break
        if n >= max_tokens:
            break

    elapsed = time.perf_counter() - t0
    ppl = math.exp(loss_sum / n) if loss_sum / n < 50 else float("inf")
    print(f"    tokens={n} PPL={ppl:.1f} CE={loss_sum / n:.2f} "
          f"top1={100 * correct / n:.1f}%  speed={n / elapsed:.0f} tok/s")
    return ppl, loss_sum / n, 100 * correct / n


def test_memory(net, text=None, repeats=3):
    if text is None:
        text = "Hello, world! This is a test of memory retention."
    print(f'\n  [Memory] "{text[:40]}..."')
    dev = next(net.parameters()).device
    bs = list(text.encode("utf-8"))
    for rep in range(repeats):
        correct = 0
        total = 0
        for i in range(min(len(bs) - 1, 25)):
            ctx = bs[: i + 1]
            if len(ctx) < 13:
                continue
            tgt = bs[i + 1]
            bv = torch.tensor([ctx], dtype=torch.long, device=dev)
            logits_t = net(bv)
            if int(logits_t[0, -1].argmax().item()) == tgt:
                correct += 1
            total += 1
        print(f"    round {rep + 1}: {correct}/{total} = {100 * correct / total:.0f}%")


def test_generation(net, prompts=None, n_tokens=40):
    if prompts is None:
        prompts = ["Hello", "import ", "The quick"]
    print("\n  [Generation]")
    dev = next(net.parameters()).device
    for prompt in prompts:
        gen = list(prompt.encode("utf-8"))
        for _ in range(n_tokens):
            bv = torch.tensor([gen[-64:]], dtype=torch.long, device=dev)
            logits_t = net(bv)  # [1, S, 256]
            last_logits = logits_t[0, -1].float() / 0.7
            topv, _ = torch.topk(last_logits, min(15, 256))
            last_logits[last_logits < topv[-1]] = -float("inf")
            probs = torch.softmax(last_logits, dim=-1)
            gen.append(int(torch.multinomial(probs, 1).item()))
        text = bytes(gen).decode("utf-8", errors="replace")
        safe = text[:50].encode("ascii", errors="replace").decode("ascii")
        print(f'    "{prompt}" -> "{safe}"')


def test_selectivity(net, texts=None):
    if texts is None:
        texts = [
            ("EN", "hello world python code english text example"),
            ("CODE", "import torch def foo return x + 1 class Model"),
            ("CN", "人工智能的未来在于深度学习"),
        ]
    print("\n  [Selectivity]")
    dev = next(net.parameters()).device
    results = {}
    for label, text in texts:
        bv = _byte_tensor(text, dev)
        _ = net(bv)
        results[label] = net._z5[0, -1].float().clone()

    labels = list(results.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            z1, z2 = results[labels[i]], results[labels[j]]
            cos_z = torch.nn.functional.cosine_similarity(
                z1.unsqueeze(0), z2.unsqueeze(0)
            ).item()
            print(f"    {labels[i]} vs {labels[j]}: cos(Z5)={cos_z:.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--dataset", default="dataset/sft_t2t_mini.jsonl")
    parser.add_argument("--ppl-samples", type=int, default=15)
    parser.add_argument("--ppl-tokens", type=int, default=500)
    args = parser.parse_args()

    all_ppl = []
    for path in args.checkpoints:
        print(f"\n{'=' * 60}")
        print(f"Eval: {path}")
        print(f"{'=' * 60}")
        net, cfg = load_model(path)
        dev = next(net.parameters()).device
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"    device={dev}  trainable_params={trainable:,}")

        test_sensitivity(net)
        ppl, ce, top1 = test_ppl(net, cfg, args.dataset, args.ppl_samples, args.ppl_tokens)
        all_ppl.append((path, trainable, ppl, ce, top1))
        test_memory(net)
        test_generation(net)
        test_selectivity(net)

    if len(all_ppl) > 1:
        print(f"\n{'=' * 60}")
        print("Comparison Summary")
        print(f"{'=' * 60}")
        print(f"{'Model':<30} {'Params':<10} {'PPL':<10} {'CE':<8} {'Top-1':<8}")
        print("-" * 60)
        for path, params, ppl, ce, top1 in all_ppl:
            name = path.split("/")[-1].replace(".pt", "")
            print(f"{name:<30} {params:<10} {ppl:<10.1f} {ce:<8.2f} {top1:<8.1f}%")


if __name__ == "__main__":
    main()
