"""综合模型评估: 语言能力 + 记忆能力.

Usage:
    python scripts/eval_model.py                    # 评估 final.pt
    python scripts/eval_model.py --ckpt ckpt_s20000.pt  # 评估特定 checkpoint
"""

import argparse, math, random, time, torch
from torch.utils.data import DataLoader
from model.model_cyrene import CyreneModel
from model.core.dataset import DualChannelDataset

torch.set_grad_enabled(False)


def load_model(path: str) -> CyreneModel:
    t0 = time.perf_counter()
    m = CyreneModel.load(path)
    m.bridge.set_warmup(0)  # 禁用 warmup
    elapsed = time.perf_counter() - t0
    stats = m.pool.query.get_activity_stats()
    print(f"模型: {path}")
    print(f"  加载耗时: {elapsed:.1f}s")
    print(f"  步数: {m._step}")
    print(f"  神经元: {stats['total_neurons']} (活跃), 突触: {stats['total_synapses']} (活跃)")
    print(f"  平均发放率: {stats['avg_firing_rate']:.4f}, 平均阈值: {stats['avg_threshold']:.4f}")
    return m


# ═══════════════════════════════════════════════════════════════
# Test 1: 困惑度 (Perplexity) — 语言建模能力
# ═══════════════════════════════════════════════════════════════
def test_perplexity(model: CyreneModel, data_path: str, max_samples: int = 100):
    print(f"\n{'=' * 60}")
    print("Test 1: Perplexity (语言建模能力)")
    print(f"数据集: {data_path} ({max_samples} 样本)")
    print(f"{'=' * 60}")

    ds = DualChannelDataset(data_path, max_length=128, max_samples=max_samples)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    total_loss = 0.0
    total_tokens = 0
    correct_top1 = 0
    correct_top5 = 0
    t0 = time.perf_counter()

    for batch_idx, (byte_seq, labels) in enumerate(loader):
        byte_seq = byte_seq.to(model.device)
        labels = labels.to(model.device)
        S = byte_seq.shape[-1]

        for pos in range(1, S):
            target = labels[0, pos - 1].item()
            if target == -100:  # padding
                continue

            context = byte_seq[:, :, :pos].contiguous()
            if context.shape[-1] < 13:
                continue

            model.step(context)

            logits = model.lm_head.predict_logits(model.pool, model._top_layer)
            loss = model.lm_head.cross_entropy_loss(logits, int(target))
            total_loss += loss
            total_tokens += 1

            # Top-1 / Top-5 准确率
            pred = max(range(256), key=lambda i: logits[i])
            top5 = sorted(range(256), key=lambda i: logits[i], reverse=True)[:5]
            if pred == target:
                correct_top1 += 1
            if target in top5:
                correct_top5 += 1

        if (batch_idx + 1) % 20 == 0:
            elapsed = time.perf_counter() - t0
            avg_loss = total_loss / max(total_tokens, 1)
            ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
            print(
                f"  [{batch_idx + 1}/{max_samples}] PPL={ppl:.2f} "
                f"top1={100 * correct_top1 / max(total_tokens, 1):.1f}% "
                f"top5={100 * correct_top5 / max(total_tokens, 1):.1f}% "
                f"tokens={total_tokens} ({total_tokens / elapsed:.0f} tok/s)"
            )

    elapsed = time.perf_counter() - t0
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")

    print("\n  结果:")
    print(f"  Perplexity: {ppl:.2f}")
    print(f"  Avg CE Loss: {avg_loss:.4f}")
    print(f"  Top-1 Accuracy: {100 * correct_top1 / total_tokens:.2f}%")
    print(f"  Top-5 Accuracy: {100 * correct_top5 / total_tokens:.2f}%")
    print(f"  Tokens evaluated: {total_tokens}")
    print(f"  Speed: {total_tokens / elapsed:.0f} tok/s")
    return {
        "ppl": ppl,
        "top1": 100 * correct_top1 / total_tokens,
        "top5": 100 * correct_top5 / total_tokens,
    }


# ═══════════════════════════════════════════════════════════════
# Test 2: 文本生成 — 连贯性
# ═══════════════════════════════════════════════════════════════
def test_generation(model: CyreneModel, prompts: list[str], max_tokens: int = 80):
    print(f"\n{'=' * 60}")
    print("Test 2: 文本生成 (连贯性与多样性)")
    print(f"{'=' * 60}")

    results = []
    for prompt in prompts:
        byte_seq = list(prompt.encode("utf-8"))
        generated_bytes = []

        for _ in range(max_tokens):
            byte_vals = torch.tensor(
                [[b / 128.0 - 1.0 for b in byte_seq + generated_bytes]],
                dtype=torch.half,
                device=model.device,
            )
            role_mask = torch.ones_like(byte_vals)
            seq = torch.stack([byte_vals, role_mask], dim=1)
            # 如果序列太长, 截断最后 128 个位置
            if seq.shape[-1] > 128:
                seq = seq[:, :, -128:]

            model.step(seq)
            logits = model.lm_head.predict_logits(model.pool, model._top_layer)

            # Top-k 采样 (temperature=0.8, top_k=20)
            logits_t = torch.tensor(logits, dtype=torch.float32)
            logits_t = logits_t / 0.8
            top_vals, top_indices = torch.topk(logits_t, min(20, 256))
            threshold = top_vals[-1]
            logits_t[logits_t < threshold] = -float("inf")
            probs = torch.softmax(logits_t, dim=-1)
            next_byte = int(torch.multinomial(probs, 1).item())
            generated_bytes.append(next_byte)

        generated = bytes(generated_bytes).decode("utf-8", errors="replace")
        results.append({"prompt": prompt, "generated": generated})
        # 截断到可控长度, 替换不可打印字符
        display_gen = generated[:80].encode("ascii", errors="replace").decode("ascii")
        print(f"\n  Prompt:    {prompt}")
        print(f"  Generated: {display_gen}")

    return results


# ═══════════════════════════════════════════════════════════════
# Test 3: 记忆能力 — 重复暴露后预测准确率变化
# ═══════════════════════════════════════════════════════════════
def test_memory(model: CyreneModel, text: str, repeats: int = 5):
    """重复喂同一文本, 观察 next-byte 预测准确率是否提升."""
    print(f"\n{'=' * 60}")
    print("Test 3: 记忆能力 (重复暴露后准确率变化)")
    print(f"文本: {text[:60]}...")
    print(f"重复次数: {repeats}")
    print(f"{'=' * 60}")

    byte_seq = list(text.encode("utf-8"))
    n_bytes = len(byte_seq) - 1  # 预测时用前 N 个预测后 1 个

    for rep in range(repeats):
        correct = 0
        total = 0
        t0 = time.perf_counter()

        for i in range(min(n_bytes, 100)):  # 最多 100 个字节
            context = byte_seq[: i + 1]
            if len(context) < 13:
                continue
            target = byte_seq[i + 1]

            byte_vals = torch.tensor(
                [[b / 128.0 - 1.0 for b in context]],
                dtype=torch.half,
                device=model.device,
            )
            role_mask = torch.ones_like(byte_vals)
            seq = torch.stack([byte_vals, role_mask], dim=1)

            model.step(seq)
            logits = model.lm_head.predict_logits(model.pool, model._top_layer)
            pred = max(range(256), key=lambda i: logits[i])

            if pred == target:
                correct += 1
            total += 1

        elapsed = time.perf_counter() - t0
        acc = 100 * correct / total if total > 0 else 0
        print(
            f"  第 {rep + 1} 轮: 准确率 {acc:.1f}% ({correct}/{total}), {total / elapsed:.0f} tok/s"
        )

    # 最终一轮的逐位置分解
    print("\n  逐位置预测 (最后一轮):")
    for i in range(min(n_bytes, 20)):
        context = byte_seq[: i + 1]
        if len(context) < 13:
            continue
        target = byte_seq[i + 1]
        byte_vals = torch.tensor(
            [[b / 128.0 - 1.0 for b in context]],
            dtype=torch.half,
            device=model.device,
        )
        role_mask = torch.ones_like(byte_vals)
        seq = torch.stack([byte_vals, role_mask], dim=1)

        model.step(seq)
        logits = model.lm_head.predict_logits(model.pool, model._top_layer)
        pred = max(range(256), key=lambda i: logits[i])
        target_char = chr(target) if 32 <= target < 127 else f"\\x{target:02x}"
        pred_char = chr(pred) if 32 <= pred < 127 else f"\\x{pred:02x}"
        match = "OK" if pred == target else "X"
        print(f"    位置 {i + 1:2d}: target='{target_char}' pred='{pred_char}' {match}")


# ═══════════════════════════════════════════════════════════════
# Test 4: final.pt vs ckpt_s20000.pt 对比
# ═══════════════════════════════════════════════════════════════
def test_compare_ckpts(path_a: str, path_b: str, data_path: str):
    """两个 checkpoint 的快速困惑度对比."""
    print(f"\n{'=' * 60}")
    print("Test 4: Checkpoint 对比")
    print(f"{'=' * 60}")

    for path in [path_a, path_b]:
        m = CyreneModel.load(path)
        m.bridge.set_warmup(0)
        ds = DualChannelDataset(data_path, max_length=64, max_samples=20)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

        total_loss = 0.0
        n = 0
        t0 = time.perf_counter()

        for byte_seq, labels in loader:
            byte_seq = byte_seq.to(m.device)
            S = byte_seq.shape[-1]
            for pos in range(1, S):
                target = labels[0, pos - 1].item()
                if target == -100:
                    continue
                context = byte_seq[:, :, :pos].contiguous()
                if context.shape[-1] < 13:
                    continue
                m.step(context)
                logits = m.lm_head.predict_logits(m.pool, m._top_layer)
                loss = m.lm_head.cross_entropy_loss(logits, int(target))
                total_loss += loss
                n += 1
                if n >= 200:
                    break
            if n >= 200:
                break

        avg_loss = total_loss / max(n, 1)
        ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
        name = path.split("/")[-1]
        print(f"  {name}: PPL={ppl:.2f}, CE={avg_loss:.4f}, tokens={n}")

        # 释放显存
        del m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="out/model1/final.pt")
    parser.add_argument("--dataset", default="dataset/sft_t2t.jsonl")
    parser.add_argument("--max-ppl-samples", type=int, default=100)
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--skip-gen", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Compare final.pt vs ckpt_s20000.pt")
    args = parser.parse_args()

    model = load_model(args.ckpt)

    if not args.skip_ppl:
        test_perplexity(model, args.dataset, max_samples=args.max_ppl_samples)

    if not args.skip_gen:
        prompts = [
            "人工智能的未来",
            "The meaning of life is",
            "def fibonacci(n):",
            "import torch\n\n# 定义一个简单的神经网络",
            "今天天气真不错",
        ]
        test_generation(model, prompts)

    if not args.skip_memory:
        test_memory(
            model,
            "Hello, world! This is a test of memory retention. "
            "The quick brown fox jumps over the lazy dog.",
        )

    if args.compare:
        ckpt_dir = "/".join(args.ckpt.split("/")[:-1])
        test_compare_ckpts(
            f"{ckpt_dir}/final.pt",
            f"{ckpt_dir}/ckpt_s20000.pt",
            args.dataset,
        )
