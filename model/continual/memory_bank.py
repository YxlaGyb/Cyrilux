"""
多巴胺效用驱动的 Memory Bank — 存储旧任务 exemplars 用于回放.

每个 Exemplar 保存 byte_tensor + label_tensor + task_id + dopamine_score + baseline_loss
MemoryBank 按 task_id 分组, 每任务容量上限 max_per_task, FIFO 淘汰.
采样策略: dopamine 加权 (高分值样本被回放的概率更高).

Ponytail: 存张量而非文本 — 反序列化零解析开销, 直接喂模型.
"""

from __future__ import annotations
import math

import dataclasses
from typing import List, Tuple

import torch


@dataclasses.dataclass
class Exemplar:
    """单条记忆样本 — 字节级张量, 无需 tokenizer."""

    byte_tensor: torch.Tensor  # [128] uint8
    label_tensor: torch.Tensor  # [128] long, -100 for padding
    task_id: str
    dopamine_score: float = 0.5
    baseline_loss: float = 0.0  # 刚存入时的 CE loss (forgetting sniffer 基线)
    transition_surprise: float = 0.0  # 世界模型预测误差/惊讶度
    replay_priority: float = 0.0  # 由世界模型和多巴胺联合打分的回放优先级
    # ── 内在动机扩展 ──
    concept_id: str = ""  # 所属概念 ID (ConceptDiscovery 分配)
    intrinsic_value: float = 0.0  # 内在价值 (ICM information_gain)
    age: int = 0  # 已存储的更新步数


class MemoryBank:
    """多任务/多概念记忆银行 — 按 task 或 concept 分组的 priority 淘汰 buffer."""

    def __init__(
        self, max_per_task: int = 2000, enable_intrinsic_eviction: bool = True
    ):
        self.max_per_task = max_per_task
        self.enable_intrinsic_eviction = enable_intrinsic_eviction
        self._store: dict[str, List[Exemplar]] = {}
        self._global_age: int = 0

    # ── 属性 ──────────────────────────────────────────────────────────

    @property
    def tasks(self) -> List[str]:
        return list(self._store.keys())

    @property
    def total(self) -> int:
        return sum(len(v) for v in self._store.values())

    def __len__(self):
        return self.total

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._store

    # ── 核心操作 ──────────────────────────────────────────────────────

    def add_samples(
        self,
        task_id: str,
        samples: List[Tuple[torch.Tensor, torch.Tensor]],
        dopamine_score: float,
        baseline_loss: float,
        transition_surprise: float = 0.0,
        intrinsic_value: float = 0.0,
        concept_id: str = "",
    ):
        """追加一批 exemplars 到指定任务/概念.

        Args:
            intrinsic_value: ICM 信息增益 (用于 Level 2 保留门控)
            concept_id: ConceptDiscovery 分配的概念 ID
        """
        if task_id not in self._store:
            self._store[task_id] = []
        buf = self._store[task_id]
        for byte_t, label_t in samples:
            self._global_age += 1
            buf.append(
                Exemplar(
                    byte_tensor=byte_t.clone(),
                    label_tensor=label_t.clone(),
                    task_id=task_id,
                    dopamine_score=dopamine_score,
                    baseline_loss=baseline_loss,
                    transition_surprise=transition_surprise,
                    replay_priority=max(dopamine_score, 0.1)
                    + max(transition_surprise, 0.0)
                    + intrinsic_value,
                    intrinsic_value=intrinsic_value,
                    concept_id=concept_id,
                    age=self._global_age,
                )
            )
        # 淘汰(priority-based 或 FIFO 回退)
        self._evict(task_id)

    def _evict(self, task_id: str):
        """按优先级淘汰超出容量的旧 exemplars。"""
        buf = self._store.get(task_id, [])
        if len(buf) <= self.max_per_task:
            return
        if self.enable_intrinsic_eviction:
            # priority × decay 排序淘汰
            current_step = self._global_age
            scored = []
            for i, ex in enumerate(buf):
                age = current_step - ex.age
                decay = math.exp(
                    -self.max_per_task
                    * 0.001
                    * (1.0 / max(ex.intrinsic_value, 0.01))
                    * max(age, 1)
                )
                score = ex.replay_priority * decay
                scored.append((score, i))
            scored.sort(key=lambda x: x[0])
            n_evict = len(buf) - self.max_per_task
            evict_indices = set(i for _, i in scored[:n_evict])
            buf[:] = [ex for i, ex in enumerate(buf) if i not in evict_indices]
        else:
            # FIFO 回退
            while len(buf) > self.max_per_task:
                buf.pop(0)

    def sample(self, batch_size: int, strategy: str = "dopamine") -> List[Exemplar]:
        """按策略采样 exemplars.

        Args:
            batch_size: 采样数量
            strategy: 'dopamine' → 按 dopamine_score 加权;
                      'world_model' → 按 replay_priority;
                      'intrinsic' → 按 intrinsic_value × replay_priority 联合加权;
                      'uniform' → 等概率

        Returns:
            采样的 exemplar 列表 (可能短于 batch_size 若 bank 为空)
        """
        if self.total == 0:
            return []
        all_ex = []
        weights = []
        for buf in self._store.values():
            for ex in buf:
                all_ex.append(ex)
                if strategy == "dopamine":
                    w = (
                        max(ex.dopamine_score, 0.1)
                        + max(ex.transition_surprise, 0.0) * 0.5
                    )
                    w = 0.1 if (math.isnan(w) or math.isinf(w)) else w
                    weights.append(w)
                elif strategy == "world_model":
                    w = max(ex.replay_priority, 0.1)
                    w = 0.1 if (math.isnan(w) or math.isinf(w)) else w
                    weights.append(w)
                elif strategy == "intrinsic":
                    w = max(ex.intrinsic_value, 0.01) * max(ex.replay_priority, 0.01)
                    w = 0.1 if (math.isnan(w) or math.isinf(w) or w < 1e-8) else w
                    weights.append(w)
                else:
                    weights.append(1.0)
        if not all_ex:
            return []
        w = torch.tensor(weights, dtype=torch.float16)
        # NaN/inf 防护
        if not torch.isfinite(w).all():
            w = torch.where(
                torch.isfinite(w), w, torch.tensor(0.1, dtype=torch.float16)
            )
        w = w / w.sum()
        idx = torch.multinomial(w, min(batch_size, len(all_ex)), replacement=False)
        return [all_ex[i] for i in idx.tolist()]

    # ── 序列化 ────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        """序列化 — 用于 checkpoint 持久化."""
        state = {}
        for task_id, buf in self._store.items():
            if not buf:
                continue
            state[task_id] = {
                "byte_tensor": torch.stack([ex.byte_tensor for ex in buf], dim=0).cpu(),
                "label_tensor": torch.stack(
                    [ex.label_tensor for ex in buf], dim=0
                ).cpu(),
                "dopamine_scores": [ex.dopamine_score for ex in buf],
                "baseline_losses": [ex.baseline_loss for ex in buf],
                "transition_surprises": [ex.transition_surprise for ex in buf],
                "replay_priorities": [ex.replay_priority for ex in buf],
                "concept_ids": [ex.concept_id for ex in buf],
                "intrinsic_values": [ex.intrinsic_value for ex in buf],
            }
        return state

    def load_state_dict(self, state: dict):
        """反序列化 — 从 checkpoint 恢复."""
        self._store = {}
        for task_id, data in state.items():
            buf = []
            for i in range(data["byte_tensor"].size(0)):
                buf.append(
                    Exemplar(
                        byte_tensor=data["byte_tensor"][i],
                        label_tensor=data["label_tensor"][i],
                        task_id=task_id,
                        dopamine_score=data["dopamine_scores"][i],
                        baseline_loss=data["baseline_losses"][i],
                        transition_surprise=data.get("transition_surprises", [0.0])[i],
                        concept_id=(data.get("concept_ids") or [""])[i]
                        if "concept_ids" in data
                        else "",
                        intrinsic_value=(data.get("intrinsic_values") or [0.0])[i]
                        if "intrinsic_values" in data
                        else 0.0,
                    )
                )
            self._store[task_id] = buf

    def clear_task(self, task_id: str):
        """清除指定任务的记忆."""
        self._store.pop(task_id, None)
