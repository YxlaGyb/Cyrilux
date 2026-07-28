"""统一评估模块 — StreamRunner Perplexity + 文本生成。

用法:
    from model.core.evaluation import (
        compute_perplexity, generate_text,
        run_full_evaluation, create_eval_runner_loader,
    )
"""

import math

import torch
from torch.utils.data import DataLoader

from model.core.dataset import DualChannelDataset
from model.model_cyrene import CyreneConfig, CyreneModel


def create_eval_runner_loader(
    data_path: str,
    max_length: int = 128,
    max_samples: int = 500,
    batch_size: int = 1,
) -> DataLoader:
    """为评估创建 DataLoader (batch_size=1, StreamRunner 单序列处理)."""
    ds = DualChannelDataset(data_path, max_length=max_length, max_samples=max_samples)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def _make_eval_runner(h_front: int = 64) -> CyreneModel:
    """创建并初始化一个评估用 CyreneModel."""
    cfg = CyreneConfig(hidden_size=h_front, warmup_steps=50)
    runner = CyreneModel(cfg)
    runner.add_hidden_layer(
        n_neurons=min(h_front * 4, 512),
        from_layer=0,
        to_layer=7,
        connection_density=0.2,
    )
    runner.warmup(20)
    return runner


# ═══════════════════════════════════════════════════════════════
# 1. Perplexity
# ═══════════════════════════════════════════════════════════════


@torch.no_grad()
def compute_perplexity(
    runner: CyreneModel,
    loader: DataLoader,
    max_batches: int = 20,
) -> tuple[float, float]:
    """计算 Perplexity = exp(mean CE loss).

    对序列逐位前馈, 读 lm_head logits 算 CE.
    """
    total_loss = 0.0
    total_tokens = 0

    for batch_idx, (byte_seq, labels) in enumerate(loader):
        if batch_idx >= max_batches:
            break

        runner.reset_hidden_state()

        S = byte_seq.shape[-1]

        for pos in range(1, S):
            context = byte_seq[:, :, :pos].contiguous()
            if context.shape[-1] < 13:
                continue

            runner.step(context)

            logits = runner.lm_head.predict_logits(runner.pool, runner._top_layer, use_mu=True)

            target = labels[0, pos - 1].item()
            if target == -100:
                continue

            loss_val = runner.lm_head.cross_entropy_loss(logits, int(target))
            total_loss += loss_val
            total_tokens += 1

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    return ppl, avg_loss


# ═══════════════════════════════════════════════════════════════
# 2. 文本生成
# ═══════════════════════════════════════════════════════════════


@torch.no_grad()
def generate_text(
    runner: CyreneModel,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 20,
) -> str:
    """字节级自回归文本生成 — StreamRunner lm_head."""
    runner.reset_hidden_state()
    byte_seq = list(prompt.encode("utf-8"))

    for _ in range(max_new_tokens):
        byte_vals = torch.tensor(
            [[b / 128.0 - 1.0 for b in byte_seq]],
            dtype=torch.half,
        )
        role_mask = torch.ones_like(byte_vals)
        seq = torch.stack([byte_vals, role_mask], dim=1)

        runner.step(seq)

        logits = runner.lm_head.predict_logits(runner.pool, runner._top_layer, use_mu=True)
        logits_t = torch.tensor(logits, dtype=torch.half)

        if temperature > 0:
            logits_t = logits_t / temperature

        if top_k > 0:
            top_vals, _ = torch.topk(logits_t, min(top_k, 256))
            threshold = top_vals[-1]
            logits_t[logits_t < threshold] = -float("inf")

        probs = torch.softmax(logits_t.float(), dim=-1).to(dtype=torch.half)
        next_byte = int(torch.multinomial(probs, 1).item())
        byte_seq.append(next_byte)

    return bytes(byte_seq).decode("utf-8", errors="replace")


# ═══════════════════════════════════════════════════════════════
# 3. 完整评估入口
# ═══════════════════════════════════════════════════════════════


def run_full_evaluation(
    runner: CyreneModel,
    loader: DataLoader,
    max_batches: int = 20,
    prompts: list[str] | None = None,
) -> dict:
    """全面评估: Perplexity + 文本生成。"""
    result: dict = {}

    print("Computing perplexity...")
    ppl, avg_loss = compute_perplexity(runner, loader, max_batches=max_batches)
    result["ppl"] = ppl
    result["avg_loss"] = avg_loss
    print(f"  PPL={ppl:.4f}  avg_CE={avg_loss:.4f}")

    generations = []
    if prompts:
        print("Generating text...")
        for p in prompts:
            gen = generate_text(runner, p, max_new_tokens=50)
            generations.append({"prompt": p, "generated": gen})
            print(f"  Prompt: {p}")
            print(f"  Generated: {gen[:100]}...")

    result["generations"] = generations
    return result
