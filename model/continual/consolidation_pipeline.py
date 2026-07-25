"""
持续巩固管道 (Consolidation Pipeline).

不做查表检索。核心职责:
  1. 维护一个临时经验缓冲 (ContinuousBuffer) — 环形 FIFO
  2. 根据吸引子景观选择"值得巩固"的经验
  3. 自动触发的 MemoryBank / AbstractionBank 写入 (不再只靠 task_end)
  4. 根据景观熵调度 SLEEP 阶段

核心理念:
  - 不是"记下所有内容", 而是"选择能加深深化吸引子的经验"
  - 薄弱吸引子周边的高信息增益样本 → 优先写入
  - 稳固吸引子周边的样本 → 跳过 (不浪费容量)
"""

from __future__ import annotations

from typing import Any, List, Tuple, Optional, Dict

from dataclasses import dataclass

import torch

from model.continual.attractor_landscape import AttractorLandscape
from model.continual.abstraction_bank import compute_layer_importance


@dataclass
class ZSample:
    """单条 PC latent 表示缓存条目。"""

    z_states: List[torch.Tensor]  # [13] × [1, seq, hidden]
    byte_tensor: torch.Tensor  # [128] uint8
    label_tensor: torch.Tensor  # [128] long
    task_id: str
    concept_id: str
    information_gain: float  # ICM 信息增益
    dopamine_score: float
    step: int


class ContinuousBuffer:
    """环形 FIFO 缓冲 — 暂存近期 z 表示。

    区别于 MemoryBank: 这里不按 task/concept 分组,
    只是原始序列的近期快照。由 ConsolidationPipeline 决定
    哪些进入长期存储。
    """

    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self._buffer: List[ZSample] = []
        self._head: int = 0  # 写入位置

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) >= self.capacity

    def add(self, sample: ZSample):
        if len(self._buffer) < self.capacity:
            self._buffer.append(sample)
        else:
            self._buffer[self._head] = sample
            self._head = (self._head + 1) % self.capacity

    def sample_recent(self, n: int) -> List[ZSample]:
        """返回最近添加的 N 条 (不采样, 取最新)。"""
        if not self._buffer:
            return []
        n = min(n, len(self._buffer))
        return self._buffer[-n:]

    def sample_by_priority(self, n: int) -> List[ZSample]:
        """按 information_gain × dopamine 加权采样。"""
        if not self._buffer:
            return []
        scores = []
        for s in self._buffer:
            ig = s.information_gain
            if ig is None or (isinstance(ig, float) and (ig != ig)):
                ig = 0.01
            ds = s.dopamine_score
            if ds is None or (isinstance(ds, float) and (ds != ds)):
                ds = 0.1
            w = max(ig, 0.01) * max(ds, 0.1)
            scores.append(max(w, 0.01))
        weights = torch.tensor(scores, dtype=torch.float16)
        weights = torch.nan_to_num(weights, nan=0.01)
        wsum = weights.sum().clamp_min(1e-8)
        if wsum <= 0 or not torch.isfinite(wsum):
            weights = torch.ones_like(weights) / len(weights)
        else:
            weights = weights / wsum
        n = min(n, len(self._buffer))
        idx = torch.multinomial(weights, n, replacement=False).tolist()
        return [self._buffer[i] for i in idx]

    def clear(self):
        self._buffer.clear()
        self._head = 0


class ConsolidationPipeline:
    """持续巩固调度器。

    工作流:
      1. 每步接收 train_step 的 z_states (→ ContinuousBuffer)
      2. 每 write_interval 步: 检查吸引子景观 → 选择经验 → 写入长期存储
      3. 每 sleep_check_interval 步: 决定是否需要 SLEEP
    """

    def __init__(
        self,
        buffer_capacity: int = 500,
        memory_write_interval: int = 50,  # 每 N 步写 MemoryBank
        abstraction_write_interval: int = 200,  # 每 N 步写 AbstractionBank
        sleep_check_interval: int = 500,  # 每 N 步检查是否需要 SLEEP
        sleep_entropy_threshold: float = 0.3,  # 归一化熵低于此值 → 需要 SLEEP
        collapse_threshold: float = 0.2,  # 坍缩比高于此值 → 需要 SLEEP
        memory_batch_size: int = 32,  # 每次写入的样本数
        abstraction_batch_size: int = 16,
        num_sub_layers: int = 12,
        min_info_gain_for_write: float = 0.05,  # 只有高于此信息增益才写入
    ):
        self.buffer = ContinuousBuffer(buffer_capacity)
        self.landscape = AttractorLandscape(num_sub_layers=num_sub_layers)

        self.memory_write_interval = memory_write_interval
        self.abstraction_write_interval = abstraction_write_interval
        self.sleep_check_interval = sleep_check_interval
        self.memory_batch_size = memory_batch_size
        self.abstraction_batch_size = abstraction_batch_size
        self.min_info_gain_for_write = min_info_gain_for_write

        self._last_sleep_check: int = 0
        self._last_memory_write: int = 0
        self._last_abstraction_write: int = 0

        # Stride 加速
        self.stride: int = 1

        # 统计
        self.stats: dict = {
            "n_memory_writes": 0,
            "n_abstraction_writes": 0,
            "n_sleep_requests": 0,
            "last_landscape_report": {},
        }

    # ── 核心: 每步调度 ────────────────────────────────────────────────

    def observe(
        self,
        z_states: List[torch.Tensor],
        byte_tensor: torch.Tensor,
        label_tensor: torch.Tensor,
        task_id: str,
        concept_id: str,
        information_gain: float,
        dopamine_score: float,
        step: int,
    ):
        """每步接收一个样本 (已在 train_step 中准备好)。"""
        # 低信息增益的直接过滤 — 不值得缓存
        if information_gain < self.min_info_gain_for_write:
            return

        sample = ZSample(
            z_states=[z.detach().cpu() for z in z_states],
            byte_tensor=byte_tensor.cpu(),
            label_tensor=label_tensor.cpu(),
            task_id=task_id,
            concept_id=concept_id,
            information_gain=information_gain,
            dopamine_score=dopamine_score,
            step=step,
        )
        self.buffer.add(sample)

    def tick(
        self,
        step: int,
        model: torch.nn.Module,
        memory_bank,
        abstraction_bank,
        device: str = "cuda:0",
        dopamine_score: Optional[float] = None,
    ) -> dict[str, Any]:
        """每步调用 — 检查是否需要触发写入或 SLEEP。

        多巴胺 D (RPE) 来自 PC 推理循环的自组织信号:
        D → 1 = 自由能快速下降 = 稳定状态, 适合巩固;
        D → 0 = 自由能上升 = 新奇/不稳定, 推迟巩固.

        Returns:
            {'triggered': str | None, 'stats': dict}
        """
        result = {"triggered": None, "stats": {}}

        # ── MemoryBank 写入 ──
        if (step - self._last_memory_write) >= self.memory_write_interval:
            n_written = self._write_to_memory_bank(model, memory_bank, device)
            self._last_memory_write = step
            self.stats["n_memory_writes"] += 1
            if n_written > 0:
                result["triggered"] = f"memory_write:{n_written}"

        # ── AbstractionBank 写入 ──
        if (step - self._last_abstraction_write) >= self.abstraction_write_interval:
            n_written = self._write_to_abstraction_bank(model, abstraction_bank, device)
            self._last_abstraction_write = step
            self.stats["n_abstraction_writes"] += 1
            if n_written > 0:
                prev = result["triggered"] or ""
                result["triggered"] = f"{prev}+abstraction_write:{n_written}"

        # ── 睡眠检查 (误差比率驱动) ──
        if (step - self._last_sleep_check) >= self.sleep_check_interval:
            self._last_sleep_check = step
            need_sleep = self._check_sleep_needed(abstraction_bank, dopamine_score)
            if need_sleep:
                self.stats["n_sleep_requests"] += 1
                result["triggered"] = "sleep_needed"

        result["stats"] = dict(self.stats)
        return result

    # ── 内部: 写入逻辑 ────────────────────────────────────────────────

    def _write_to_memory_bank(
        self,
        model: torch.nn.Module,
        memory_bank,
        device: str,
    ) -> int:
        """从 buffer 选择高价值样本写入 MemoryBank。"""
        if self.buffer.size < self.memory_batch_size:
            return 0

        # 选高信息增益 × 高多巴胺的样本
        samples = self.buffer.sample_by_priority(self.memory_batch_size)
        if not samples:
            return 0

        # 按 task_id 分组写入
        by_task: Dict[
            str, List[Tuple[torch.Tensor, torch.Tensor, float, float, float, str]]
        ] = {}
        for s in samples:
            tid = s.task_id
            if tid not in by_task:
                by_task[tid] = []
            by_task[tid].append(
                (
                    s.byte_tensor,
                    s.label_tensor,
                    s.dopamine_score,
                    s.information_gain,
                    s.information_gain,
                    s.concept_id,
                )
            )

        n_total = 0
        for tid, entries in by_task.items():
            # ── Stride 下采样 ──
            if self.stride > 1:
                entries_subsampled = []
                for bt, lt, ds, ig, iv, cid in entries:
                    idx = torch.arange(0, bt.size(-1), self.stride, device=bt.device)
                    bt_s = bt[..., idx]
                    lt_s = lt[..., idx] if lt is not None else lt
                    entries_subsampled.append((bt_s, lt_s, ds, ig, iv, cid))
                entries = entries_subsampled

            pairs = [(e[0], e[1]) for e in entries]
            d_scores = [e[2] for e in entries]
            info_gains = [e[3] for e in entries]
            concept_ids = [e[5] for e in entries]

            # 用平均多巴胺和信息增益
            avg_d = sum(d_scores) / len(d_scores)
            avg_ig = sum(info_gains) / len(info_gains)
            # concept_id 取众数 (出现最多的)
            concept_id = (
                max(set(concept_ids), key=concept_ids.count) if concept_ids else ""
            )

            memory_bank.add_samples(
                tid,
                pairs,
                dopamine_score=avg_d,
                baseline_loss=0.0,  # 后续 forgot sniffer 会更新
                transition_surprise=0.0,
                intrinsic_value=avg_ig,
                concept_id=concept_id,
            )
            n_total += len(pairs)

        return n_total

    def _write_to_abstraction_bank(
        self,
        model: torch.nn.Module,
        abstraction_bank,
        device: str,
    ) -> int:
        """从 buffer 选择高价值 z 表示写入 AbstractionBank。"""
        if self.buffer.size < self.abstraction_batch_size:
            return 0

        samples = self.buffer.sample_by_priority(self.abstraction_batch_size)
        if not samples:
            return 0

        # 按 task_id 分组
        by_task: Dict[str, List[ZSample]] = {}
        for s in samples:
            tid = s.task_id
            if tid not in by_task:
                by_task[tid] = []
            by_task[tid].append(s)

        n_total = 0
        for tid, group in by_task.items():
            # ── Stride 下采样 z_states ──
            if self.stride > 1:
                for s in group:
                    s.z_states = [z[..., :: self.stride] for z in s.z_states]

            z_states_list = [s.z_states for s in group]
            avg_d = sum(s.dopamine_score for s in group) / len(group)
            avg_ig = sum(s.information_gain for s in group) / len(group)
            concept_ids = [s.concept_id for s in group]
            concept_id = (
                max(set(concept_ids), key=concept_ids.count) if concept_ids else ""
            )

            # 计算层重要性 (用第一个样本)
            if group:
                layer_imp = compute_layer_importance(
                    group[0].z_states,
                    model,
                    (None, None),
                    dopamine_D=avg_d,
                    eta=1.0,
                )
            else:
                layer_imp = None

            abstraction_bank.add_z_samples(
                tid,
                z_states_list,
                layer_importance=layer_imp,
                dopamine_score=avg_d,
                world_model_surprise=0.0,
                concept_id=concept_id,
                information_gain=avg_ig,
                group_by_concept=True if concept_id else False,
            )
            n_total += len(z_states_list)

        return n_total

    # ── 内部: 睡眠调度 ────────────────────────────────────────────────

    def _check_sleep_needed(
        self, abstraction_bank, dopamine_score: Optional[float] = None
    ) -> bool:
        """检查是否需要深度睡眠。

        使用多巴胺 D (RPE): D → 1 (自由能快速下降) → 稳定状态 → 适合巩固.
        D 与 error_ratio 正相关 (都在 F 快速下降时触发), 但无需多步推理.
        """
        if abstraction_bank.total_prototypes < 4:
            return False

        if dopamine_score is not None:
            # D 高 → 自由能快速下降 → 表示空间收敛 → 触发睡眠巩固
            self.stats["last_dopamine_score"] = dopamine_score
            need_sleep = dopamine_score > 0.70
            self.stats["last_sleep_decision"] = {
                "dopamine_score": dopamine_score,
                "need_sleep": need_sleep,
            }
            return need_sleep

        return False  # 无 dopamine_score 时, 不由 pipeline 触发睡眠

    def force_consolidate(
        self,
        step: int,
        model: torch.nn.Module,
        memory_bank,
        abstraction_bank,
        device: str = "cuda:0",
    ) -> dict[str, Any]:
        """外部触发 (来自 training.py): 立即执行一次 MemoryBank + AbstractionBank 写入。

        用于低误差比率持续窗口时的强制巩固。
        """
        result: dict[str, Any] = {"triggered": None}
        n_mem = self._write_to_memory_bank(model, memory_bank, device)
        n_abs = self._write_to_abstraction_bank(model, abstraction_bank, device)
        if n_mem > 0 or n_abs > 0:
            result["triggered"] = f"force:mem={n_mem},abs={n_abs}"
        return result
