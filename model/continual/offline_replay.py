"""
离线生成式自巩固
利用 PC 模型自身预测能力做自我回放.

Phase 6 (Bonus): 在训练间隙, 让 PC 模型为 MemoryBank 中每个旧任务
生成合成训练数据. 完全不需要原始数据源.

原理:
  1. 从 bank 取每个旧任务的 exemplar, 截取其前 8 字节作为 prompt
  2. 调用 generate_with_pc() 自回归生成后续字节流
  3. 生成结果 → 标准的 (byte_tensor, label_tensor)

注意: 仅做生成 (inference), 不含任何 backward 训练.
Hebbian 回放训练由 trainer._maybe_replay/_maybe_abstraction_replay 接管.
"""

from __future__ import annotations

import torch

from model.continual.memory_bank import MemoryBank
from model.continual.abstraction_bank import AbstractionBank


class OfflineReplayer:
    """PC 生成式自巩固回放器 — 纯生成, 零 backward."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer=None,
        memory_bank: MemoryBank = None,
        abstraction_bank: AbstractionBank = None,
        for_token_free: bool = True,
        world_model=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.memory_bank = memory_bank
        self.abstraction_bank = abstraction_bank
        self.for_token_free = for_token_free
        self.world_model = world_model

    # ── 质量过滤 ──────────────────────────────────────────────────

    @torch.no_grad()
    def _compute_surprise(self, byte_t: torch.Tensor, device: str) -> float:
        """计算一条样本的世界模型 transition surprise."""
        if self.world_model is None:
            return 0.5
        x = byte_t.unsqueeze(0).to(device)
        pos = self.model.get_position_embeddings(x.size(-1), device)
        z_init, _ = self.model.forward_with_ce(x.half().unsqueeze(1), x.long(), pos)
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
        max_length: int = 64,
        temperature: float = 0.8,
        top_k: int = 20,
        prompt_len: int = 8,
        device: str = "cuda:0",
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
        idx = torch.randperm(len(buf))[: min(n_samples, len(buf))].tolist()
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
                    prompt.half().unsqueeze(1), prompt.long(), pos
                )
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
                max_new_tokens=max_length,
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
                padded = torch.cat(
                    [full_seq.cpu(), torch.zeros(128 - seq_len, dtype=torch.long)]
                )
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
