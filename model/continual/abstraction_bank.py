"""抽象记忆银行 — StreamRunner 适配版。

核心转变:
  MemoryBank 存"模型见过什么"(原始字节),
  AbstractionBank 存"模型学会了什么"(神经元稳态表示 + 原型).

架构:
  AbstractionEntry  — 单条抽象记忆 (神经元快照)
  AbstractionBank   — 多任务原型银行, 支持 k-means 压缩 + 表示级回放
  VariationalReplayer — 结构变体生成 + 吸引子强度测试
  AbstractionSniffer  — 抽象级遗忘检测 (余弦距离到原型)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from model.model_cyrene import CyreneModel


@torch.no_grad()
def compute_layer_importance(
    runner: CyreneModel,
    dopamine_D: float = 0.0,
    eta: float = 1.0,
) -> torch.Tensor:
    """计算神经元分组预测误差 -> layer_importance 向量 [n_layers].

    CyreneModel: 按 layer_groups 分组, 每组内计算平均 |epsilon|.
    公式: pi_ell = 1 + eta * D * mean(|eps|)

    Args:
        runner: CyreneModel
        dopamine_D: 当前多巴胺 D in [0,1]
        eta: 精度调制强度

    Returns:
        layer_importance: [max_layer+1] float tensor (fp16)
    """
    if not runner.pool.layer_groups:
        return torch.tensor([1.0], dtype=torch.half)

    max_layer = max(runner.pool.layer_groups.keys())
    importance = []

    for layer in range(max_layer + 1):
        nids = runner.pool.layer_groups.get(layer, set())
        if not nids:
            importance.append(0.0)
            continue
        eps_vals = [
            abs(runner.pool.neurons[nid].epsilon)
            for nid in nids
            if nid in runner.pool.neurons
        ]
        mean_eps = sum(eps_vals) / max(len(eps_vals), 1)
        importance.append(mean_eps)

    pi = torch.tensor(importance, dtype=torch.half)
    pi_min, pi_max = pi.min(), pi.max()
    if pi_max > pi_min:
        pi = (pi - pi_min) / (pi_max - pi_min)
    else:
        pi = torch.zeros_like(pi)
    pi = 1.0 + eta * dopamine_D * pi
    return pi


@torch.no_grad()
def _kmeans_cosine(
    points: torch.Tensor,
    k: int,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cosine-distance k-means 聚类."""
    n = points.size(0)
    if n <= k:
        return points, torch.arange(n, device=points.device)

    points = F.normalize(points, dim=-1)
    idx = torch.randperm(n, device=points.device)[:k]
    centers = points[idx].clone()

    for _ in range(max_iter):
        sim = points @ centers.T
        labels = sim.argmax(dim=-1)
        new_centers = torch.stack(
            [points[labels == c].mean(dim=0) for c in range(k)]
        )
        new_centers = F.normalize(new_centers, dim=-1)
        if (centers - new_centers).norm() < tol:
            centers = new_centers
            break
        centers = new_centers

    return centers, labels


class AbstractionEntry:
    """单条抽象记忆 — 神经元活动快照."""

    def __init__(self, z_states: Dict[int, float], task_id: str, step: int):
        self.z_states = z_states
        self.task_id = task_id
        self.step = step
        self.visit_count = 0
        self.prototype_similarity: Optional[float] = None

    def to_vector(self, all_nids: List[int]) -> torch.Tensor:
        vals = [self.z_states.get(nid, 0.0) for nid in all_nids]
        return torch.tensor(vals, dtype=torch.half)

    def update_prototype_similarity(
        self, prototype: torch.Tensor, all_nids: List[int]
    ):
        vec = self.to_vector(all_nids).unsqueeze(0)
        sim = F.cosine_similarity(vec, prototype.unsqueeze(0)).item()
        self.prototype_similarity = sim


class AbstractionBank:
    """多任务抽象记忆银行 — CyreneModel 神经元快照."""

    def __init__(self, max_per_task: int = 1000, k_prototypes: int = 16):
        self.max_per_task = max_per_task
        self.k_prototypes = k_prototypes
        self.entries: Dict[str, List[AbstractionEntry]] = {}
        self.prototypes: Dict[str, torch.Tensor] = {}
        self._all_nids: List[int] = []

    def update_neuron_ids(self, runner: CyreneModel):
        self._all_nids = sorted(runner.pool.neurons.keys())

    def add_entry(self, entry: AbstractionEntry):
        if entry.task_id not in self.entries:
            self.entries[entry.task_id] = []
        self.entries[entry.task_id].append(entry)
        if len(self.entries[entry.task_id]) > self.max_per_task:
            self.entries[entry.task_id].pop(0)

    def snapshot(
        self, runner: CyreneModel, task_id: str, step: int
    ) -> AbstractionEntry:
        self.update_neuron_ids(runner)
        z_snapshot = {
            nid: runner.pool.neurons[nid].z
            for nid in self._all_nids
            if nid in runner.pool.neurons
        }
        entry = AbstractionEntry(z_snapshot, task_id, step)
        self.add_entry(entry)
        return entry

    def compute_prototypes(self, task_id: str) -> Optional[torch.Tensor]:
        entries = self.entries.get(task_id, [])
        if not entries or not self._all_nids:
            return None
        vecs = torch.stack([e.to_vector(self._all_nids) for e in entries])
        k = min(self.k_prototypes, len(entries))
        centers, _ = _kmeans_cosine(vecs, k=k)
        self.prototypes[task_id] = centers.mean(dim=0)
        return self.prototypes[task_id]

    def get_prototype(self, task_id: str) -> Optional[torch.Tensor]:
        return self.prototypes.get(task_id)

    def replay(
        self, runner: CyreneModel, task_id: str, n_samples: int = 4
    ) -> List[AbstractionEntry]:
        """从任务的原型记忆中采样, 通过 runner 回放."""
        entries = self.entries.get(task_id, [])
        if not entries:
            return []
        idx = torch.randperm(len(entries))[:n_samples].tolist()
        replayed = []
        for i in idx:
            entry = entries[i]
            for nid, z_val in entry.z_states.items():
                if nid in runner.pool.neurons:
                    runner.pool.neurons[nid].z = z_val
            replayed.append(entry)
        return replayed


class VariationalReplayer:
    """结构变体生成 + 吸引子强度测试 — CyreneModel 版."""

    def __init__(self, bank: AbstractionBank, noise_scale: float = 0.05):
        self.bank = bank
        self.noise_scale = noise_scale

    @torch.no_grad()
    def test_attractor_strength(
        self, runner: CyreneModel, task_id: str, n_samples: int = 4
    ) -> float:
        """测试吸引子强度: 扰动后与原型相似度."""
        proto = self.bank.get_prototype(task_id)
        if proto is None:
            return 0.0
        entries = self.bank.entries.get(task_id, [])
        if not entries:
            return 0.0
        sims = []
        idx = torch.randperm(len(entries))[:n_samples].tolist()
        for i in idx:
            entry = entries[i]
            vec = entry.to_vector(self.bank._all_nids)
            noisy = vec + torch.randn_like(vec) * self.noise_scale
            sim = F.cosine_similarity(
                noisy.unsqueeze(0), proto.unsqueeze(0)
            ).item()
            sims.append(sim)
        return sum(sims) / max(len(sims), 1)

    @torch.no_grad()
    def replay_variations(
        self, runner: CyreneModel, task_id: str, n_variations: int = 2
    ) -> List[AbstractionEntry]:
        """从原型生成变体并通过 runner 回放."""
        proto = self.bank.get_prototype(task_id)
        if proto is None:
            return []
        entries = self.bank.entries.get(task_id, [])
        if not entries:
            return []
        replayed = []
        idx = torch.randperm(len(entries))[:n_variations].tolist()
        for i in idx:
            entry = entries[i]
            for nid, z_val in entry.z_states.items():
                if nid in runner.pool.neurons:
                    noise = torch.randn(1).item() * self.noise_scale
                    runner.pool.neurons[nid].z = z_val + noise
            replayed.append(entry)
        return replayed


class AbstractionSniffer:
    """抽象级遗忘检测 — 原型余弦距离."""

    def __init__(
        self,
        abstraction_bank: AbstractionBank,
        threshold: float = 0.3,
        window: int = 100,
    ):
        self.bank = abstraction_bank
        self.threshold = threshold
        self.window = window
        self._history: Dict[str, List[float]] = {}

    def check(self, runner: CyreneModel, task_id: str) -> bool:
        proto = self.bank.get_prototype(task_id)
        if proto is None:
            return False
        if task_id not in self._history:
            self._history[task_id] = []

        current = AbstractionBank()
        current.update_neuron_ids(runner)
        entry = current.snapshot(runner, task_id, 0)
        sim = F.cosine_similarity(
            entry.to_vector(current._all_nids).unsqueeze(0),
            proto.unsqueeze(0),
        ).item()
        self._history[task_id].append(sim)
        if len(self._history[task_id]) > self.window:
            self._history[task_id].pop(0)

        if len(self._history[task_id]) >= 10:
            recent = sum(self._history[task_id][-5:]) / 5
            return recent < self.threshold
        return False
