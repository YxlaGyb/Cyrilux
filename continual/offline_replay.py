"""
离线生成式自巩固 — 利用 PC 模型自身预测能力做自我回放.

Phase 6 (Bonus): 在训练间隙, 让 PC 模型为 MemoryBank 中每个旧任务
生成合成训练数据, 再以纯 CE loss 回放. 完全不需要原始数据源.

原理:
  1. 从 bank 取每个旧任务的 exemplar, 截取其前 8 字节作为 prompt
  2. 调用 generate_with_pc() 自回归生成后续字节流
  3. 生成结果 → 标准的 (byte_tensor, label_tensor) → 纯 CE backward

Ponytail: 这是生成式回放的最简实现 — 每次生成只做一次前向解码.
"""
from __future__ import annotations

import torch
from torch import nn

from continual.memory_bank import MemoryBank


class OfflineReplayer:
    """PC 生成式自巩固回放器."""

    def __init__(self, memory_bank: MemoryBank, model: nn.Module):
        self.memory_bank = memory_bank
        self.model = model

    @torch.no_grad()
    def generate_for_task(
        self,
        task_id: str,
        n_samples: int = 100,
        max_new_tokens: int = 120,
        temperature: float = 0.7,
        top_k: int = 20,
        prompt_len: int = 8,
        device: str = 'cuda:0',
    ) -> list:
        """为指定任务生成合成训练数据.

        对 bank 中随机挑选的 exemplar, 取其前 prompt_len 字节作为 prompt,
        调用 generate_with_pc 生成后续字节流, 组装成 (byte_tensor, label_tensor).

        Returns: [(byte_tensor [128] uint8, label_tensor [128] long), ...]
        """
        self.model.eval()
        buf = self.memory_bank._store.get(task_id, [])
        if not buf:
            return []

        samples = []
        idx = torch.randperm(len(buf))[:min(n_samples, len(buf))].tolist()
        for i in idx:
            ex = buf[i]
            prompt_bytes = ex.byte_tensor[:prompt_len]  # [prompt_len]
            prompt = prompt_bytes.unsqueeze(0).to(device)  # [1, prompt_len]
            generated = self.model.generate_with_pc(
                prompt,
                max_new_tokens=max_new_tokens,
                T_infer=0,
                gamma=0.1,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=0x02,
            )
            # generated: [1, prompt_len + new_tokens]
            full_seq = generated[0]  # [seq]
            # 截断/填充到 128
            seq_len = full_seq.size(0)
            if seq_len < 128:
                padded = torch.cat([full_seq.cpu(), torch.zeros(128 - seq_len, dtype=torch.long)])
            else:
                padded = full_seq[:128].cpu()
            byte_t = padded.to(torch.uint8)
            label_t = byte_t.clone().to(torch.long)
            label_t[byte_t == 0x00] = -100
            samples.append((byte_t, label_t))
        return samples

    def replay_generated(
        self,
        optimizer,
        n_per_task: int = 100,
        batch_size: int = 32,
        temperature: float = 0.7,
        device: str = 'cuda:0',
    ) -> dict:
        """对 MemoryBank 中所有任务生成并回放合成数据.

        Returns: {task_id: avg_ce_loss_after_replay}
        """
        results = {}
        for task_id in self.memory_bank.tasks:
            samples = self.generate_for_task(
                task_id, n_samples=n_per_task, temperature=temperature, device=device,
            )
            if not samples:
                continue

            total_loss = 0.0
            n_batches = 0
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i + batch_size]
                byte_seq = torch.stack([s[0] for s in batch], dim=0).to(device)
                labels = torch.stack([s[1] for s in batch], dim=0).to(device)
                pos_emb = self.model.get_position_embeddings(byte_seq.size(1), device)
                _, ce_loss = self.model.forward_with_ce(byte_seq, labels, pos_emb)

                optimizer.zero_grad(set_to_none=True)
                ce_loss.backward()
                trainable = [p for p in self.model.parameters()
                             if p.requires_grad and p.grad is not None]
                if trainable:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()

                total_loss += ce_loss.item()
                n_batches += 1

            results[task_id] = total_loss / max(n_batches, 1)
        return results
