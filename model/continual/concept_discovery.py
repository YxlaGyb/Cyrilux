"""
无监督概念发现 (Unsupervised Concept Discovery)

在 PC latent space 中对 z_states 做在线流式聚类，自动发现数据中的自然概念/技能，
彻底摆脱外部 task_id 依赖。

架构:
  - _concepts: dict[str, Concept] — 自动命名的概念
  - _stream_buffer: list[z_state] — 待聚类缓冲
  - 自适应阈值: 初始 0.85 → 随概念数衰减到 0.65
  - 合并: cosine > 0.95 的相似概念自动合并
  - 淘汰: 支持度 < 5 的脆弱概念

与现有机制打配合:
  - MemoryBank.add_samples() 可调用 concept_discovery.observe() 自动确定概念
  - AbstractionBank.consolidate() 可在概念级别执行
  - CuriositySampler 使用脆弱概念作为生成目标
"""

from __future__ import annotations
from typing import Tuple, Optional

import torch


@torch.no_grad()
def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """余弦相似度, a=[D], b=[K, D] or [D]。"""
    if b.dim() == 1:
        b = b.unsqueeze(0)
    a_n = a / (a.norm() + 1e-8)
    b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return a_n @ b_n.T  # [K]


class Concept:
    """单个自动发现的概念。"""

    def __init__(self, concept_id: str, centroid: torch.Tensor):
        self.id = concept_id
        self.centroid = centroid.clone()  # [D]
        self.support: int = 1  # 分配的样本数
        self.avg_intrinsic_value: float = 0.0  # 平均内在价值
        self.last_seen_step: int = 0
        self.created_step: int = 0
        self.n_merges: int = 0
        self._momentum_buffer: torch.Tensor = centroid.clone()

    def update(self, z: torch.Tensor, intrinsic_value: float = 0.0, step: int = 0):
        """增量更新质心 (EMA)。"""
        self.support += 1
        self.last_seen_step = step
        lr = 1.0 / max(self.support, 1)
        self.centroid = self.centroid * (1 - lr) + z.squeeze() * lr
        # 指数平滑内在价值
        self.avg_intrinsic_value = (
            self.avg_intrinsic_value * 0.95 + intrinsic_value * 0.05
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "support": self.support,
            "avg_intrinsic_value": self.avg_intrinsic_value,
            "last_seen_step": self.last_seen_step,
            "created_step": self.created_step,
            "n_merges": self.n_merges,
        }


class ConceptDiscovery:
    """在线流式概念发现。"""

    def __init__(
        self,
        initial_threshold: float = 0.85,
        min_threshold: float = 0.65,
        max_concepts: int = 20,
        merge_threshold: float = 0.95,
        min_support: int = 5,
        consolidation_interval: int = 200,
        feature_dim: int = 256,
    ):
        self.initial_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_concepts = max_concepts
        self.merge_threshold = merge_threshold
        self.min_support = min_support
        self.consolidation_interval = consolidation_interval
        self.feature_dim = feature_dim

        self._concepts: dict[str, Concept] = {}
        self._stream_buffer: list[
            tuple[torch.Tensor, float, int]
        ] = []  # [(z, intrinsic_value, step)]
        self._concept_counter: int = 0
        self._global_step: int = 0
        self._n_observations: int = 0

    # ── 属性 ──

    @property
    def n_concepts(self) -> int:
        return len(self._concepts)

    @property
    def concept_ids(self) -> list[str]:
        return list(self._concepts.keys())

    @property
    def alive_concepts(self) -> list[tuple[str, Concept]]:
        return [
            (k, v) for k, v in self._concepts.items() if v.support >= self.min_support
        ]

    @property
    def fragile_concepts(self) -> list[tuple[str, Concept]]:
        return [
            (k, v) for k, v in self._concepts.items() if v.support < self.min_support
        ]

    def get_threshold(self) -> float:
        """自适应阈值: 概念越多阈值越低。"""
        t = self.initial_threshold - (self.n_concepts / self.max_concepts) * (
            self.initial_threshold - self.min_threshold
        )
        return max(t, self.min_threshold)

    # ── 核心 ──

    def observe(
        self,
        z_state: torch.Tensor,
        feature_embedding: torch.Tensor | None = None,
        intrinsic_value: float = 0.0,
    ) -> str:
        """观察一个 z_state, 返回分配的概念 ID。

        Args:
            z_state: [1, seq, D] PC 收敛后的顶层表示
            feature_embedding: [D] ICM 特征 (可选, 用于增强相似度)
            intrinsic_value: ICM 信息增益 (用于概念更新权重)
        """
        self._global_step += 1
        self._n_observations += 1

        # 使用均值池化作为概念表示
        z_pooled = z_state.mean(dim=1).squeeze(0)  # [D]

        # 如果提供了 feature_embedding, 拼接
        if feature_embedding is not None:
            z_pooled = torch.cat([z_pooled, feature_embedding.view(-1)], dim=0)

        # 查找最接近的概念
        best_id, best_sim = self._find_nearest(z_pooled)

        if best_id is not None and best_sim >= self.get_threshold():
            # 分配到已有概念
            self._concepts[best_id].update(z_pooled, intrinsic_value, self._global_step)
            self._stream_buffer.append((z_pooled, intrinsic_value, self._global_step))
            return best_id
        else:
            # 创建新概念
            return self._create_concept(z_pooled, intrinsic_value)

    def _find_nearest(self, z_pooled: torch.Tensor) -> Tuple[Optional[str], float]:
        """找到最接近的概念。"""
        if not self._concepts:
            return None, 0.0
        best_id = None
        best_sim = -1.0
        for cid, concept in self._concepts.items():
            sim = _cosine_similarity(z_pooled, concept.centroid).item()
            if sim > best_sim:
                best_sim = sim
                best_id = cid
        return best_id, best_sim

    def _create_concept(
        self, z_pooled: torch.Tensor, intrinsic_value: float = 0.0
    ) -> str:
        """创建新概念。"""
        if len(self._concepts) >= self.max_concepts:
            # 淘汰最脆弱的概念
            self._evict_fragile()
        cid = f"concept_{self._concept_counter}"
        self._concept_counter += 1
        c = Concept(cid, z_pooled)
        c.avg_intrinsic_value = intrinsic_value
        c.created_step = self._global_step
        c.last_seen_step = self._global_step
        self._concepts[cid] = c
        self._stream_buffer.append((z_pooled, intrinsic_value, self._global_step))
        return cid

    def _evict_fragile(self):
        """淘汰最脆弱的概念 (支持度最小 + 最久未见)。"""
        if not self._concepts:
            return
        # 按 (support, -last_seen_step) 排序
        sorted_concepts = sorted(
            self._concepts.items(),
            key=lambda x: (x[1].support, -x[1].last_seen_step),
        )
        # 淘汰支持度最小的
        evict_id, _ = sorted_concepts[0]
        del self._concepts[evict_id]

    def consolidate(self):
        """合并相似概念 + 淘汰脆弱概念。"""
        if self.n_concepts < 2:
            # 仅淘汰
            self._prune_fragile()
            return

        # 合并相似概念
        merged = set()
        cids = list(self._concepts.keys())
        for i in range(len(cids)):
            if cids[i] in merged:
                continue
            for j in range(i + 1, len(cids)):
                if cids[j] in merged:
                    continue
                ci = self._concepts[cids[i]]
                cj = self._concepts[cids[j]]
                sim = _cosine_similarity(ci.centroid, cj.centroid).item()
                if sim >= self.merge_threshold:
                    # 合并 j → i
                    total = ci.support + cj.support
                    ci.centroid = (
                        ci.centroid * ci.support + cj.centroid * cj.support
                    ) / total
                    ci.support = total
                    ci.avg_intrinsic_value = (
                        ci.avg_intrinsic_value * ci.support
                        + cj.avg_intrinsic_value * cj.support
                    ) / total
                    ci.n_merges += 1
                    merged.add(cids[j])

        for mid in merged:
            del self._concepts[mid]

        # 淘汰脆弱概念
        self._prune_fragile()

    def _prune_fragile(self):
        """淘汰长期未见、支持度不足的脆弱概念。"""
        evict = []
        for cid, concept in self._concepts.items():
            if (
                concept.support < self.min_support
                and self._global_step - concept.created_step > 500
            ):
                evict.append(cid)
        for cid in evict:
            del self._concepts[cid]

    # ── 查询 ──

    def get_concept_summary(self, concept_id: str) -> Optional[dict]:
        c = self._concepts.get(concept_id)
        if c is None:
            return None
        return c.to_dict()

    def get_all_summaries(self) -> dict[str, dict]:
        return {cid: c.to_dict() for cid, c in self._concepts.items()}

    def get_centroids(self, device: str = "cpu") -> dict[str, torch.Tensor]:
        return {cid: c.centroid.to(device) for cid, c in self._concepts.items()}

    def get_most_uncertain_concept(self, n: int = 1) -> list[tuple[str, float]]:
        """返回 avg_intrinsic_value 最高 (最不确定) 的前 N 个概念。"""
        scored = [(cid, c.avg_intrinsic_value) for cid, c in self._concepts.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def get_fragile_concept_ids(self) -> list[str]:
        return [
            cid for cid, c in self._concepts.items() if c.support < self.min_support
        ]

    # ── 序列化 ──

    def state_dict(self) -> dict:
        centroids = {}
        meta = {}
        for cid, c in self._concepts.items():
            centroids[cid] = c.centroid.cpu()
            meta[cid] = c.to_dict()
        return {
            "centroids": centroids,
            "meta": meta,
            "concept_counter": self._concept_counter,
            "global_step": self._global_step,
            "n_observations": self._n_observations,
        }

    def load_state_dict(self, state: dict):
        self._concepts = {}
        for cid, cent in state.get("centroids", {}).items():
            meta = state["meta"].get(cid, {})
            c = Concept(cid, cent)
            c.support = meta.get("support", 1)
            c.avg_intrinsic_value = meta.get("avg_intrinsic_value", 0.0)
            c.last_seen_step = meta.get("last_seen_step", 0)
            c.created_step = meta.get("created_step", 0)
            c.n_merges = meta.get("n_merges", 0)
            self._concepts[cid] = c
        self._concept_counter = state.get("concept_counter", 0)
        self._global_step = state.get("global_step", 0)
        self._n_observations = state.get("n_observations", 0)
