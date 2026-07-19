"""
深度 SLEEP 引擎 — 纯 Hebbian 吸引子景观维护 (Attractor Landscape Maintenance).

不是简单回放，而是 3 阶段的吸引子雕刻，全部使用局部 Hebbian 学习 (零反向传播):
  1. Pattern Completion:    遮掩序列后半段 → PC 补全 → 局部预测误差驱动 Hebbian 更新
  2. Noise Broadening:      完整序列加噪声 → PC infer → 误差信号驱动噪声容忍学习
  3. Competitive Balance:   跨概念混合回放 → Hebbian 竞争平衡

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

from model.local_updates import (
    compute_all_hebbian_updates,
    apply_hebbian_updates,
    BCMState,
)


class SleepEngine:
    """深层睡眠引擎 — 纯 Hebbian 吸引子景观维护 (零反向传播)。"""

    def __init__(
        self,
        num_sub_layers: int = 12,
        pattern_completion_steps: int = 20,    # Phase 1 步数
        noise_broadening_steps: int = 20,      # Phase 2 步数
        competitive_steps: int = 10,           # Phase 3 步数
        noise_levels: List[float] = None,      # Phase 2 噪声级别
        completion_mask_ratio: float = 0.5,    # Phase 1 遮掩比例
        T_infer_sleep: int = 2,                # SLEEP 时推理步数 (减少)
        gamma_sleep: float = 0.05,             # SLEEP 时 gamma (减弱)
        hebbian_base_eta: float = 5e-5,        # Hebbian 学习率 (训练期 1e-5, 睡眠需略高但不过量)
        hebbian_lambda_min: float = 0.01,      # Decoder 约束最小权重
        dopamine_gamma: float = 0.3,           # RPE 调制系数
    ):
        self.num_sub_layers = num_sub_layers
        self.pattern_completion_steps = pattern_completion_steps
        self.noise_broadening_steps = noise_broadening_steps
        self.competitive_steps = competitive_steps
        self.noise_levels = noise_levels or [0.05, 0.1, 0.15]
        self.completion_mask_ratio = completion_mask_ratio
        self.T_infer_sleep = T_infer_sleep
        self.gamma_sleep = gamma_sleep
        self.hebbian_base_eta = hebbian_base_eta
        self.hebbian_lambda_min = hebbian_lambda_min
        self.dopamine_gamma = dopamine_gamma

        # BCM 滑动阈值 (慢速 tau=0.005 适合离线巩固)
        self.bcm_state = BCMState(n_layers=num_sub_layers, tau=0.005)

        # 统计
        self.stats: dict = {
            'completion_loss': [],
            'noise_loss': [],
            'competitive_loss': [],
        }

    # ── Hebbian 更新工具 ──────────────────────────────────────────

    @torch.no_grad()
    def _hebbian_step(self, model: torch.nn.Module, byte_seq: torch.Tensor,
                      labels: torch.Tensor, device: str):
        """对 (byte_seq, labels) 执行一次纯 Hebbian 权重更新 (零 backward)。

        流程:
          1. forward_with_ce → z_init (起始潜在变量)
          2. spatiotemporal_infer → z_conv, ε_list (收敛 + 局部预测误差)
          3. compute_all_hebbian_updates → ΔW (基于 ε 的 Hebbian 更新)
          4. apply_hebbian_updates → 权重写入
        """
        seq_len = byte_seq.size(-1)
        pos_emb = model.get_position_embeddings(seq_len, device)
        z_init, _ = model.forward_with_ce(byte_seq, labels, pos_emb)

        z_det = [z.detach() for z in z_init]
        z_conv, _, _, _, ε_list = model.spatiotemporal_infer(
            z_det, pos_emb,
            gamma=self.gamma_sleep, T=self.T_infer_sleep,
            return_errors=True, return_pred_loss=False,
            ach_value=0.5, return_ε=True,
        )

        # 固定调制基线 (睡眠阶段不需要在线多巴胺/乙酰胆碱调节)
        updates = compute_all_hebbian_updates(
            ε_list, z_init, byte_seq, model, self,
            D=0.5, ACh=0.5, modulation=0.5,
            λ=self.hebbian_lambda_min,
            decoder=model.decoder,
            target_byte_embed=None,
            bcm_state=self.bcm_state,
        )
        apply_hebbian_updates(updates, model,
                              max_delta=0.02, weight_bound=4.0)

    def run(
        self,
        model: torch.nn.Module,
        memory_bank,
        abstraction_bank,
        device: str = 'cuda:0',
        phases: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """执行完整的 SLEEP 循环 (纯 Hebbian, 零优化器/反向传播)。

        Args:
            phases: None = 全部, 或 ['completion', 'noise', 'competitive']

        Returns:
            {'completion_avg_loss': float, 'noise_avg_loss': float, 'competitive_avg_loss': float}
        """
        if phases is None:
            phases = ['completion', 'noise', 'competitive']

        results = {}
        if 'completion' in phases and memory_bank.total > 0:
            avg_loss = self._phase_pattern_completion(model, memory_bank, device)
            results['completion_avg_loss'] = avg_loss

        if 'noise' in phases and memory_bank.total > 0:
            avg_loss = self._phase_noise_broadening(model, memory_bank, device)
            results['noise_avg_loss'] = avg_loss

        if 'competitive' in phases and memory_bank.total > 0:
            avg_loss = self._phase_competitive_balance(model, memory_bank, abstraction_bank, device)
            results['competitive_avg_loss'] = avg_loss

        return results

    # ── Phase 1: 模式补全 ──────────────────────────────────────────

    @torch.no_grad()
    def _phase_pattern_completion(
        self,
        model: torch.nn.Module,
        memory_bank,
        device: str,
    ) -> float:
        """模式补全: 遮掩序列 → 局部预测误差驱动 Hebbian 更新。

        机制:
          1. 从 MemoryBank 采样完整序列
          2. 遮掩后半段 (置 0)
          3. forward + spatiotemporal_infer → 局部预测误差
          4. 误差驱动 Hebbian 权重更新 — 自动学习时序补全

        Hebbian 下的效果: 遮掩位置产生大预测误差 → temp_proj 权重更新 →
          学会从上下文预测遮掩部分的表示 → 加深吸引子盆地。
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

            # 遮掩后半段: 后半段置 0
            masked_byte = full_byte.clone()
            masked_byte[..., -mask_len:] = 0

            # Hebbian 更新: 用遮掩序列驱动局部预测误差 → 自动补全学习
            self._hebbian_step(model, masked_byte, full_label, device)

            # 记录 loss (仅监控, 不参与更新)
            pos_emb = model.get_position_embeddings(seq_len, device)
            with torch.no_grad():
                z_init, _ = model.forward_with_ce(masked_byte, full_label, pos_emb)
                z_det = [z.detach() for z in z_init]
                z_conv, *_ = model.spatiotemporal_infer(
                    z_det, pos_emb,
                    gamma=self.gamma_sleep, T=self.T_infer_sleep,
                    return_errors=False, return_pred_loss=False,
                )
                z_full, _ = model.forward_with_ce(full_byte, full_label, pos_emb)
                monitor_loss = 0.0
                for ℓ in range(1, self.num_sub_layers + 1):
                    monitor_loss += F.mse_loss(
                        z_conv[ℓ][:, -mask_len:, :],
                        z_full[ℓ][:, -mask_len:, :].detach(),
                    )
                losses.append(monitor_loss.item())

        avg_loss = sum(losses) / max(len(losses), 1) if losses else 0.0
        self.stats['completion_loss'].append(avg_loss)
        return avg_loss

    # ── Phase 2: 噪声加宽 ──────────────────────────────────────────

    @torch.no_grad()
    def _phase_noise_broadening(
        self,
        model: torch.nn.Module,
        memory_bank,
        device: str,
    ) -> float:
        """噪声加宽: 加噪序列 → 局部预测误差驱动 Hebbian 更新。

        机制:
          1. 从 MemoryBank 采完整序列
          2. 字节级随机替换噪声
          3. forward + spatiotemporal_infer → 局部预测误差
          4. 误差驱动 Hebbian 更新 — 自动学会容忍噪声

        Hebbian 下的效果: 噪声位置产生大预测误差 → 权重更新 →
          学会从噪声上下文预测正确表示 → 加宽吸引子盆地。
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

            # Hebbian 更新: 用加噪序列驱动局部预测误差 → 自动噪声容忍学习
            self._hebbian_step(model, noisy_byte, clean_label, device)

            # 记录 loss (仅监控)
            pos_emb = model.get_position_embeddings(seq_len, device)
            with torch.no_grad():
                z_noisy, _ = model.forward_with_ce(noisy_byte, clean_label, pos_emb)
                z_noisy_det = [z.detach() for z in z_noisy]
                z_conv, *_ = model.spatiotemporal_infer(
                    z_noisy_det, pos_emb,
                    gamma=self.gamma_sleep, T=self.T_infer_sleep,
                    return_errors=False, return_pred_loss=False,
                )
                z_clean, _ = model.forward_with_ce(clean_byte, clean_label, pos_emb)
                monitor_loss = 0.0
                for ℓ in range(1, self.num_sub_layers + 1):
                    monitor_loss += F.mse_loss(z_conv[ℓ], z_clean[ℓ].detach())
                losses.append(monitor_loss.item())

        avg_loss = sum(losses) / max(len(losses), 1) if losses else 0.0
        self.stats['noise_loss'].append(avg_loss)
        return avg_loss

    # ── Phase 3: 竞争平衡 ──────────────────────────────────────────

    @torch.no_grad()
    def _phase_competitive_balance(
        self,
        model: torch.nn.Module,
        memory_bank,
        abstraction_bank,
        device: str,
    ) -> float:
        """竞争平衡: 跨概念混合回放 → Hebbian 竞争平衡。

        机制:
          1. 从多个不同概念中各采 exemplar
          2. 混合成一个 batch
          3. 联合 forward + Hebbian 更新 — 跨概念误差平衡

        Hebbian 下的效果: 多概念 batch 产生不同模式的预测误差 →
          权重更新平衡各概念吸引子强度, 防止单一吸引子过度生长。
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

            # Hebbian 更新: 混合概念 batch → 跨概念误差驱动平衡
            self._hebbian_step(model, mix_byte, mix_label, device)

            # 记录 CE loss (仅监控)
            with torch.no_grad():
                _, ce_loss = model.forward_with_ce(mix_byte, mix_label,
                    model.get_position_embeddings(mix_byte.size(-1), device))
                losses.append(ce_loss.item())

        avg_loss = sum(losses) / max(len(losses), 1) if losses else 0.0
        self.stats['competitive_loss'].append(avg_loss)
        return avg_loss

    # ── 工具 ────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {'stats': self.stats}

    def load_state_dict(self, state: dict):
        self.stats = state.get('stats', {'completion_loss': [], 'noise_loss': [], 'competitive_loss': []})
