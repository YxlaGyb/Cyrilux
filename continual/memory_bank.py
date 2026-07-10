"""
多巴胺效用驱动的 Memory Bank — 存储旧任务 exemplars 用于回放.

每个 Exemplar 保存 byte_tensor + label_tensor + task_id + dopamine_score + baseline_loss.
MemoryBank 按 task_id 分组, 每任务容量上限 max_per_task, FIFO 淘汰.
采样策略: dopamine 加权 (高分值样本被回放的概率更高).

Ponytail: 存张量而非文本 — 反序列化零解析开销, 直接喂模型.
"""
from __future__ import annotations

import dataclasses
from typing import List, Tuple, Optional

import torch


@dataclasses.dataclass
class Exemplar:
    """单条记忆样本 — 字节级张量, 无需 tokenizer."""
    byte_tensor: torch.Tensor   # [128] uint8
    label_tensor: torch.Tensor  # [128] long, -100 for padding
    task_id: str
    dopamine_score: float = 0.5
    baseline_loss: float = 0.0  # 刚存入时的 CE loss (forgetting sniffer 基线)


class MemoryBank:
    """多任务记忆银行 — 按 task 分组的 FIFO buffer."""

    def __init__(self, max_per_task: int = 2000):
        self.max_per_task = max_per_task
        self._store: dict[str, List[Exemplar]] = {}

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
    ):
        """追加一批 exemplars 到指定任务. FIFO 淘汰超出容量的旧样本."""
        if task_id not in self._store:
            self._store[task_id] = []
        buf = self._store[task_id]
        for byte_t, label_t in samples:
            buf.append(Exemplar(
                byte_tensor=byte_t.clone(),
                label_tensor=label_t.clone(),
                task_id=task_id,
                dopamine_score=dopamine_score,
                baseline_loss=baseline_loss,
            ))
        # FIFO 淘汰
        while len(buf) > self.max_per_task:
            buf.pop(0)

    def sample(self, batch_size: int, strategy: str = 'dopamine') -> List[Exemplar]:
        """按策略采样 exemplars.

        Args:
            batch_size: 采样数量
            strategy: 'dopamine' → 按 dopamine_score 加权; 'uniform' → 等概率

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
                if strategy == 'dopamine':
                    weights.append(max(ex.dopamine_score, 0.1))
                else:
                    weights.append(1.0)
        if not all_ex:
            return []
        w = torch.tensor(weights, dtype=torch.float)
        w = w / w.sum()
        idx = torch.multinomial(w, min(batch_size, len(all_ex)), replacement=False)
        return [all_ex[i] for i in idx.tolist()]

    # ── 评估 ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, model, device: str, N: int = 32) -> dict:
        """纯前向评估 bank 中各任务的 CE loss (无梯度, T=0).

        Returns:
            {task_id: {'avg_ce': float, 'baseline_ce': float, 'ratio': float}}
        """
        if self.total == 0:
            return {}
        model.eval()
        results = {}
        for task_id, buf in self._store.items():
            if not buf:
                continue
            idx = torch.randperm(len(buf))[:min(N, len(buf))].tolist()
            losses = []
            for i in idx:
                ex = buf[i]
                x = ex.byte_tensor.unsqueeze(0).to(device)
                y = ex.label_tensor.unsqueeze(0).to(device)
                z = model.init_z(x)
                h = model.model.norm(z[model.num_sub_layers])
                logits = model.model.lm_head(h)
                s_logits = logits[..., :-1, :].contiguous()
                s_labels = y[..., 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    s_logits.view(-1, s_logits.size(-1)),
                    s_labels.view(-1),
                    ignore_index=-100,
                    reduction='sum',
                )
                n_tokens = (s_labels != -100).sum().item()
                losses.append(loss.item() / max(n_tokens, 1))
            avg_ce = sum(losses) / len(losses)
            # baseline_loss 平均值 (只统计 >0 的有效值)
            baselines = [ex.baseline_loss for ex in buf if ex.baseline_loss > 0]
            baseline_ce = sum(baselines) / max(len(baselines), 1)
            ratio = avg_ce / max(baseline_ce, 1e-8)
            results[task_id] = {'avg_ce': avg_ce, 'baseline_ce': baseline_ce, 'ratio': ratio}
        return results

    # ── 序列化 ────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        """序列化 — 用于 checkpoint 持久化."""
        state = {}
        for task_id, buf in self._store.items():
            if not buf:
                continue
            state[task_id] = {
                'byte_tensor': torch.stack([ex.byte_tensor for ex in buf], dim=0).cpu(),
                'label_tensor': torch.stack([ex.label_tensor for ex in buf], dim=0).cpu(),
                'dopamine_scores': [ex.dopamine_score for ex in buf],
                'baseline_losses': [ex.baseline_loss for ex in buf],
            }
        return state

    def load_state_dict(self, state: dict):
        """反序列化 — 从 checkpoint 恢复."""
        self._store = {}
        for task_id, data in state.items():
            buf = []
            for i in range(data['byte_tensor'].size(0)):
                buf.append(Exemplar(
                    byte_tensor=data['byte_tensor'][i],
                    label_tensor=data['label_tensor'][i],
                    task_id=task_id,
                    dopamine_score=data['dopamine_scores'][i],
                    baseline_loss=data['baseline_losses'][i],
                ))
            self._store[task_id] = buf

    def clear_task(self, task_id: str):
        """清除指定任务的记忆."""
        self._store.pop(task_id, None)
