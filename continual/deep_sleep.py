"""
深度 SLEEP 引擎 — 吸引子景观维护 (Attractor Landscape Maintenance).

不是简单回放，而是 3 阶段的吸引子雕刻:
  1. Pattern Completion:    遮掩序列后半段 → PC 补全 → 比较完整序列 → backward
  2. Noise Broadening:      完整序列加噪声 → PC infer → 与原始比较 → backward
  3. Competitive Balance:   跨概念混合回放 → 防止单一吸引子过度生长

核心理念:
  SLEEP 不是"复习", 而是"雕刻吸引子盆地":
  - 补全 → 加深盆地 (学会从碎片恢复完整)
  - 噪声 → 加宽盆地 (增强鲁棒性)
  - 平衡 → 防止一个盆地吞噬另一个
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn.functional as F


class SleepEngine:
    """深层睡眠引擎 — 吸引子景观维护。"""

    def __init__(
        self,
        num_sub_layers: int = 12,
        pattern_completion_steps: int = 20,    # Phase 1 步数
        noise_broadening_steps: int = 20,      # Phase 2 步数
        competitive_steps: int = 10,           # Phase 3 步数
        noise_levels: List[float] = None,      # Phase 2 噪声级别
        completion_mask_ratio: float = 0.5,    # Phase 1 遮掩比例
        lr_scale: float = 0.5,                 # SLEEP 时学习率缩放
        grad_clip: float = 1.0,
        T_infer_sleep: int = 2,                # SLEEP 时推理步数 (减少)
        gamma_sleep: float = 0.05,             # SLEEP 时 gamma (减弱)
    ):
        self.num_sub_layers = num_sub_layers
        self.pattern_completion_steps = pattern_completion_steps
        self.noise_broadening_steps = noise_broadening_steps
        self.competitive_steps = competitive_steps
        self.noise_levels = noise_levels or [0.05, 0.1, 0.15]
        self.completion_mask_ratio = completion_mask_ratio
        self.lr_scale = lr_scale
        self.grad_clip = grad_clip
        self.T_infer_sleep = T_infer_sleep
        self.gamma_sleep = gamma_sleep

        # 统计
        self.stats: dict = {
            'completion_loss': [],
            'noise_loss': [],
            'competitive_loss': [],
        }

    @torch.enable_grad()
    def run(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        memory_bank,
        abstraction_bank,
        device: str = 'cuda:0',
        phases: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """执行完整的 SLEEP 循环。

        Args:
            phases: None = 全部, 或 ['completion', 'noise', 'competitive']

        Returns:
            {'completion_avg_loss': float, 'noise_avg_loss': float, 'competitive_avg_loss': float}
        """
        if phases is None:
            phases = ['completion', 'noise', 'competitive']

        results = {}
        original_lr = optimizer.param_groups[0]['lr']
        sleep_lr = original_lr * self.lr_scale
        for pg in optimizer.param_groups:
            pg['lr'] = sleep_lr

        try:
            if 'completion' in phases and memory_bank.total > 0:
                avg_loss = self._phase_pattern_completion(model, optimizer, memory_bank, device)
                results['completion_avg_loss'] = avg_loss

            if 'noise' in phases and memory_bank.total > 0:
                avg_loss = self._phase_noise_broadening(model, optimizer, memory_bank, device)
                results['noise_avg_loss'] = avg_loss

            if 'competitive' in phases and memory_bank.total > 0:
                avg_loss = self._phase_competitive_balance(model, optimizer, memory_bank, abstraction_bank, device)
                results['competitive_avg_loss'] = avg_loss
        finally:
            for pg in optimizer.param_groups:
                pg['lr'] = original_lr

        return results

    # ── Phase 1: 模式补全 ──────────────────────────────────────────

    @torch.enable_grad()
    def _phase_pattern_completion(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        memory_bank,
        device: str,
    ) -> float:
        """模式补全: 遮掩序列后半段 → PC 推理补全 → 与完整序列对比。

        机制:
          1. 从 MemoryBank 采样完整序列
          2. 遮掩后半段 (masked_positions 替换为 0)
          3. PC 推理 T 步尝试恢复完整表示
          4. 比较恢复的表示与原始表示的差异 → backward

        效果: 加深吸引子 — 学会从局部线索重构整体。
        """
        losses = []
        model.train()

        for _ in range(self.pattern_completion_steps):
            exemplars = memory_bank.sample(1, strategy='uniform')
            if not exemplars:
                break

            ex = exemplars[0]
            full_byte = ex.byte_tensor.squeeze().unsqueeze(0).to(device)   # [1, seq]
            full_label = ex.label_tensor.squeeze().unsqueeze(0).to(device) # [1, seq]
            seq_len = full_byte.size(-1)
            mask_len = max(1, int(seq_len * self.completion_mask_ratio))

            # 遮掩后半段: 后半段置 0 (沿最后一维)
            masked_byte = full_byte.clone()
            masked_byte[..., -mask_len:] = 0

            pos_emb = model.get_position_embeddings(seq_len, device)

            # 用遮掩序列初始化 z
            z_init, _ = model.forward_with_ce(masked_byte, full_label, pos_emb)

            # PC inference 尝试恢复完整表示
            z_detached = [z.detach() for z in z_init]
            z_conv, *_ = model.spatiotemporal_infer(
                z_detached, pos_emb,
                gamma=self.gamma_sleep, T=self.T_infer_sleep,
                return_errors=False, return_pred_loss=False,
            )

            # 完整序列的表示作为目标
            z_full, _ = model.forward_with_ce(full_byte, full_label, pos_emb)
            z_full_detached = [z.detach() for z in z_full]

            # 被遮掩位置的表示差异
            completion_loss = 0.0
            for ℓ in range(1, self.num_sub_layers + 1):
                # 只比较遮掩部分的表示
                z_recovered = z_conv[ℓ][:, -mask_len:, :]
                z_target = z_full_detached[ℓ][:, -mask_len:, :]
                completion_loss += F.mse_loss(z_recovered, z_target)

            # 加入 CE loss — 帮助字节级别恢复
            z_h = model.model.norm(z_conv[self.num_sub_layers])
            logits = model.model.lm_head(z_h.to(dtype=model.model.lm_head.weight.dtype))
            s_logits = logits[..., :-1, :].contiguous()
            s_labels = full_label[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                s_logits.float().view(-1, s_logits.size(-1)),
                s_labels.view(-1),
                ignore_index=-100,
            )

            total_loss = completion_loss + 0.3 * ce_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            if trainable:
                torch.nn.utils.clip_grad_norm_(trainable, self.grad_clip)
            optimizer.step()

            losses.append(total_loss.item())

        avg_loss = sum(losses) / max(len(losses), 1) if losses else 0.0
        self.stats['completion_loss'].append(avg_loss)
        return avg_loss

    # ── Phase 2: 噪声加宽 ──────────────────────────────────────────

    @torch.enable_grad()
    def _phase_noise_broadening(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        memory_bank,
        device: str,
    ) -> float:
        """噪声加宽: 完整序列加噪声 → PC infer → 与原始比较。

        机制:
          1. 从 MemoryBank 采完整序列
          2. 在字节级别加随机替换噪声
          3. PC 推理收敛
          4. 收敛后的表示与无噪声时的表示对比

        效果: 加宽吸引子盆地 — 学会容忍输入噪声。
        """
        losses = []
        model.train()

        for _ in range(self.noise_broadening_steps):
            exemplars = memory_bank.sample(1, strategy='uniform')
            if not exemplars:
                break

            ex = exemplars[0]
            clean_byte = ex.byte_tensor.squeeze().unsqueeze(0).to(device)
            clean_label = ex.label_tensor.squeeze().unsqueeze(0).to(device)
            seq_len = clean_byte.size(-1)

            # 随机选一个噪声级别
            noise_scale = random.choice(self.noise_levels)

            # 字节级噪声: 随机替换字节
            noisy_byte = clean_byte.clone()
            n_noise = max(1, int(seq_len * noise_scale))
            noise_pos = torch.randperm(seq_len, device=device)[:n_noise]
            noise_shape = list(noisy_byte.shape[:-1]) + [n_noise]
            noisy_byte[..., noise_pos] = torch.randint(0, 256, noise_shape, device=device).to(noisy_byte.dtype)

            pos_emb = model.get_position_embeddings(seq_len, device)

            # 加噪声版本 forward
            z_noisy, ce_loss = model.forward_with_ce(noisy_byte, clean_label, pos_emb)
            z_noisy_det = [z.detach() for z in z_noisy]
            z_conv, *_ = model.spatiotemporal_infer(
                z_noisy_det, pos_emb,
                gamma=self.gamma_sleep, T=self.T_infer_sleep,
                return_errors=False, return_pred_loss=False,
            )

            # 干净版本 forward (作为目标)
            with torch.no_grad():
                z_clean, _ = model.forward_with_ce(clean_byte, clean_label, pos_emb)
                z_clean_det = [z.detach() for z in z_clean]

            # 表示一致性: 噪声收敛后的表示应接近干净表示
            repr_loss = 0.0
            for ℓ in range(1, self.num_sub_layers + 1):
                repr_loss += F.mse_loss(z_conv[ℓ], z_clean_det[ℓ])

            total_loss = repr_loss + 0.5 * ce_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            if trainable:
                torch.nn.utils.clip_grad_norm_(trainable, self.grad_clip)
            optimizer.step()

            losses.append(total_loss.item())

        avg_loss = sum(losses) / max(len(losses), 1) if losses else 0.0
        self.stats['noise_loss'].append(avg_loss)
        return avg_loss

    # ── Phase 3: 竞争平衡 ──────────────────────────────────────────

    @torch.enable_grad()
    def _phase_competitive_balance(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        memory_bank,
        abstraction_bank,
        device: str,
    ) -> float:
        """竞争平衡: 跨概念混合回放。

        机制:
          1. 从多个不同概念中各采 exemplar
          2. 混合成一个 batch
          3. 联合 forward_with_ce → backbone update

        效果: 防止某个概念吸引子过度生长, 保持跨概念平衡。
        """
        if memory_bank.total < 2:
            return 0.0

        losses = []
        model.train()

        # 找到 bank 中的不同 concept_id / task_id
        concept_groups: Dict[str, List] = {}
        for buf_list in memory_bank._store.values():
            for ex in buf_list:
                key = ex.concept_id if ex.concept_id else ex.task_id
                if key not in concept_groups:
                    concept_groups[key] = []
                concept_groups[key].append(ex)

        group_ids = list(concept_groups.keys())
        if len(group_ids) < 2:
            # 只有一个概念 → 使用不同任务的 exemplar
            group_ids = list(memory_bank._store.keys())

        if len(group_ids) < 2:
            return 0.0

        for _ in range(self.competitive_steps):
            # 从 2-3 个不同组各采 1 条
            n_groups = min(len(group_ids), random.randint(2, 3))
            chosen = random.sample(group_ids, n_groups)

            batch_bytes = []
            batch_labels = []
            for gid in chosen:
                pool = concept_groups.get(gid, [])
                if not pool:
                    # fallback: 从 MemoryBank._store 取
                    for buf in memory_bank._store.values():
                        if buf:
                            pool = buf
                            break
                if pool:
                    ex = random.choice(pool)
                    batch_bytes.append(ex.byte_tensor.squeeze().unsqueeze(0))
                    batch_labels.append(ex.label_tensor.squeeze().unsqueeze(0))

            if len(batch_bytes) < 2:
                continue

            mix_byte = torch.cat(batch_bytes, dim=0).to(device)
            mix_label = torch.cat(batch_labels, dim=0).to(device)
            seq_len = mix_byte.size(-1)
            pos_emb = model.get_position_embeddings(seq_len, device)

            _, ce_loss = model.forward_with_ce(mix_byte, mix_label, pos_emb)

            optimizer.zero_grad(set_to_none=True)
            ce_loss.backward()
            trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            if trainable:
                torch.nn.utils.clip_grad_norm_(trainable, self.grad_clip)
            optimizer.step()

            losses.append(ce_loss.item())

        avg_loss = sum(losses) / max(len(losses), 1) if losses else 0.0
        self.stats['competitive_loss'].append(avg_loss)
        return avg_loss

    # ── 工具 ────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {'stats': self.stats}

    def load_state_dict(self, state: dict):
        self.stats = state.get('stats', {'completion_loss': [], 'noise_loss': [], 'competitive_loss': []})
