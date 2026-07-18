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

    def __init__(self, memory_bank: MemoryBank, model: nn.Module, world_model=None):
        self.memory_bank = memory_bank
        self.model = model
        self.world_model = world_model

    # ── 质量过滤 ──────────────────────────────────────────────────

    @torch.no_grad()
    def _compute_surprise(self, byte_t: torch.Tensor, device: str) -> float:
        """计算一条样本的世界模型 transition surprise."""
        if self.world_model is None:
            return 0.5
        x = byte_t.unsqueeze(0).to(device)
        pos = self.model.get_position_embeddings(x.size(-1), device)
        z_init, _ = self.model.forward_with_ce(x.float().unsqueeze(1), x.long(), pos)
        z_top = z_init[-1].detach()
        ctx = torch.zeros(1, 5, device=device)
        _, unc = self.world_model(z_top, ctx)
        return unc.mean().item()

    def filter_by_world_model(
        self,
        samples: list,
        device: str,
        low_percentile: float = 0.1,
        high_percentile: float = 0.9,
    ) -> list:
        """用世界模型 surprise 过滤样本."""
        if self.world_model is None or not samples:
            return samples
        surprises = [self._compute_surprise(s[0], device) for s in samples]
        if len(surprises) < 5:
            return samples
        sorted_s = sorted(surprises)
        lo = sorted_s[int(len(sorted_s) * low_percentile)]
        hi = sorted_s[min(len(sorted_s) - 1, int(len(sorted_s) * high_percentile))]
        filtered = [s for s, sp in zip(samples, surprises) if lo <= sp <= hi]
        return filtered if filtered else samples

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
        enable_wm_filter: bool = True,
        enable_wm_temperature: bool = True,
    ) -> list:
        """为指定任务生成合成训练数据.

        对 bank 中随机挑选的 exemplar, 取其前 prompt_len 字节作为 prompt,
        调用 generate_with_pc 生成后续字节流, 组装成 (byte_tensor, label_tensor).

        Args:
            enable_wm_filter: 若 True, 用世界模型过滤低质量样本
            enable_wm_temperature: 若 True, 用世界模型 uncertainty 调制温度

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

            # 世界模型 uncertainty 调制温度 (3b)
            eff_temp = temperature
            if enable_wm_temperature and self.world_model is not None:
                # 对 prompt 计算世界模型 uncertainty
                pos = self.model.get_position_embeddings(prompt.size(-1), device)
                z_init, _ = self.model.forward_with_ce(
                    prompt.float().unsqueeze(1), prompt.long(), pos)
                z_top = z_init[-1].detach()
                ctx = torch.zeros(1, 5, device=device)
                _, unc = self.world_model(z_top, ctx)
                wm_conf = 1.0 - unc.mean().item()
                # 高 confidence → 低温度 (更确定性的生成)
                # 低 confidence → 高温度 (探索更多变体)
                eff_temp = temperature * (1.5 - wm_conf)
                eff_temp = max(0.3, min(1.5, eff_temp))

            generated = self.model.generate_with_pc(
                prompt,
                max_new_tokens=max_new_tokens,
                T_infer=0,
                gamma=0.1,
                temperature=eff_temp,
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

        # 世界模型质量过滤 (3a)
        if enable_wm_filter:
            samples = self.filter_by_world_model(samples, device)

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
        # 3c: 按世界模型 uncertainty 排序任务 — 更不确定的任务先回放更多
        tasks = list(self.memory_bank.tasks)
        if self.world_model is not None and len(tasks) > 1:
            uncertainties = []
            for tid in tasks:
                buf = self.memory_bank._store.get(tid, [])
                if buf:
                    ex = buf[0]
                    x = ex.byte_tensor.unsqueeze(0).to(device)
                    pos = self.model.get_position_embeddings(x.size(-1), device)
                    pos_emb = self.model.get_position_embeddings(x.size(-1), device)
                    z_init, _ = self.model.forward_with_ce(
                        x.float().unsqueeze(1), x.long(), pos_emb)
                    z_top = z_init[-1].detach()
                    ctx = torch.zeros(1, 5, device=device)
                    _, unc = self.world_model(z_top, ctx)
                    uncertainties.append((tid, unc.mean().item()))
                else:
                    uncertainties.append((tid, 0.5))
            # 按 uncertainty 降序排列
            uncertainties.sort(key=lambda x: -x[1])
            tasks = [t[0] for t in uncertainties]

        results = {}
        for task_id in tasks:
            # uncertainty 高的任务生成更多样本
            eff_n = n_per_task
            if self.world_model is not None and len(tasks) > 1:
                for tid, unc in uncertainties:
                    if tid == task_id:
                        eff_n = int(n_per_task * (0.5 + unc))
                        eff_n = max(n_per_task // 2, min(n_per_task * 2, eff_n))
                        break

            samples = self.generate_for_task(
                task_id, n_samples=eff_n, temperature=temperature, device=device,
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
