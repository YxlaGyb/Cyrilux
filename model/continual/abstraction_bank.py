"""
抽象记忆银行 — PC 表示空间中的原型压缩 + 表示级回放 + 吸引子检测.

核心转变:
  MemoryBank 存"模型见过什么"(原始字节),
  AbstractionBank 存"模型学会了什么"(PC 收敛后的 z 表示 + 原型).

架构:
  AbstractionEntry  — 单条抽象记忆 (z_states × 13 层)
  AbstractionBank   — 多任务原型银行, 支持 k-means 压缩 + 表示级回放
  VariationalReplayer — 结构变体生成 + 吸引子强度测试
  AbstractionSniffer  — 抽象级遗忘检测 (余弦距离到原型)
"""
from __future__ import annotations

import math
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# 工具: 层重要性计算

@torch.no_grad()
def compute_layer_importance(
    z_conv: List[torch.Tensor],
    model: nn.Module,
    pos_emb: Tuple,
    dopamine_D: float = 0.0,
    eta: float = 1.0,
) -> torch.Tensor:
    """计算每子层的预测误差平方 → layer_importance 向量 [num_sub_layers].

    公式: π_ℓ = 1 + η · D · ‖ε_ℓ‖ (归一化后)

    Args:
        z_conv: PC 收敛后的全层表示 [13] × [1, seq, hidden]
        model: PCLocalDynamicMiniMind
        pos_emb: (None, None) — 局部 Conv 模型无位置编码
        dopamine_D: 当前多巴胺信号 D ∈ [0, 1]
        eta: 精度调制强度

    Returns:
        layer_importance: [num_sub_layers=12] float tensor
    """
    L = model.num_sub_layers  # = 12
    ε_sq = []

    for ℓ in range(1, L + 1):
        # 自下而上预测
        μ_bu = model.predict(ℓ, z_conv[ℓ - 1], pos_emb)
        μ_bu_res = μ_bu + z_conv[ℓ - 1]

        # 时序预测
        seq_len = z_conv[ℓ].size(1)
        if seq_len > 1:
            z_prev_t = z_conv[ℓ][:, :-1, :]
            z_temp = model.temporal_proj[ℓ - 1](z_prev_t)
            μ_temp = torch.cat([torch.zeros_like(z_conv[ℓ][:, :1, :]), z_temp], dim=1)
        else:
            μ_temp = torch.zeros_like(z_conv[ℓ])

        # 自上而下预测
        if ℓ < L and seq_len > 1:
            z_down_prev = z_conv[ℓ + 1][:, :-1, :]
            z_down = model.topdown_proj[ℓ - 1](z_down_prev)
            μ_down = torch.cat([torch.zeros_like(z_conv[ℓ][:, :1, :]), z_down], dim=1)
        else:
            μ_down = torch.zeros_like(z_conv[ℓ])

        μ_total = μ_bu_res + μ_temp + μ_down
        ε = z_conv[ℓ] - μ_total
        ε_sq.append((ε ** 2).mean().item())

    π = torch.tensor(ε_sq, dtype=torch.float)
    # 归一化到 [0, 1] 区间, 保留相对比例
    π_min, π_max = π.min(), π.max()
    if π_max > π_min:
        π = (π - π_min) / (π_max - π_min)
    else:
        π = torch.zeros_like(π)
    # 调制: π_ℓ = 1 + η · D · ε_norm
    π = 1.0 + eta * dopamine_D * π
    return π  # [12]


# AbstractionEntry — 单条抽象记忆

@torch.no_grad()
def _kmeans_cosine(
    points: torch.Tensor,
    k: int,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cosine-distance k-means 聚类.

    Args:
        points: [N, d] float tensor
        k: 聚类数
        max_iter: 最大迭代
        tol: 收敛容忍度

    Returns:
        centroids: [k, d]
        assignments: [N] long
    """
    N, d = points.shape
    if N <= k:
        # 样本数不足 k, 返回所有样本作为原型
        return points, torch.arange(N, device=points.device)

    # k-means++ 初始化
    centroids = []
    centroids.append(points[torch.randint(0, N, (1,))].squeeze(0))  # [d]
    for _ in range(1, k):
        dists = torch.cdist(points, torch.stack(centroids), p=2).min(dim=1)[0]
        # 数值稳定性: 处理全零距离 → 退化为均匀采样
        dist_sum = dists.sum()
        if dist_sum < 1e-12:
            prob = torch.ones_like(dists) / dists.size(0)
        else:
            prob = dists / dist_sum
        # 确保无 nan/inf
        prob = torch.nan_to_num(prob, nan=1.0 / dists.size(0))
        centroids.append(points[torch.multinomial(prob, 1)].squeeze(0))  # [d]
    centroids = torch.stack(centroids)  # [k, d]

    for it in range(max_iter):
        # 余弦距离 = 1 - cosine_similarity
        norms = points.norm(dim=1, keepdim=True) + 1e-8
        c_norms = centroids.norm(dim=1, keepdim=True) + 1e-8
        cos_sim = (points @ centroids.T) / (norms @ c_norms.T)  # [N, k]
        assignments = cos_sim.argmax(dim=1)  # [N]

        new_centroids = []
        for i in range(k):
            mask = assignments == i
            if mask.any():
                new_centroids.append(points[mask].mean(dim=0))
            else:
                new_centroids.append(centroids[i])
        new_centroids = torch.stack(new_centroids)

        shift = (new_centroids - centroids).norm().item()
        centroids = new_centroids
        if shift < tol:
            break

    return centroids, assignments


# AbstractionBank — 抽象记忆银行

class AbstractionBank:
    """表示级抽象记忆银行.

    不存原始字节, 存 PC 推理收敛后的 z 表示.
    每任务/概念自动做 k-means 原型压缩, 压缩比 ~32000:1.

    支持:
      - task_id 分组 (旧)
      - concept_id 分组 (新, 内在动机)
      - information_gain 加权 consolidate
    """

    def __init__(
        self,
        max_entries_per_task: int = 2000,
        n_prototypes: int = 8,
        consolidation_frequency: int = 1,
        enable_concept_grouping: bool = True,
    ):
        self.max_entries_per_task = max_entries_per_task
        self.n_prototypes = n_prototypes
        self.consolidation_frequency = consolidation_frequency
        self.enable_concept_grouping = enable_concept_grouping

        # 原始 z 存储 (按任务)
        self._store: Dict[str, List[dict]] = {}

        # 按概念分组的存储 (与 _store 并存, 优先使用)
        self._store_by_concept: Dict[str, List[dict]] = {}

        # 压缩后的原型 (按任务/概念)
        self._prototypes: Dict[str, torch.Tensor] = {}  # id → [k, hidden]
        self._prototype_importances: Dict[str, torch.Tensor] = {}  # id → [k, 12]
        self._prototype_scores: Dict[str, torch.Tensor] = {}  # id → [k] retention score
        self._meta: Dict[str, dict] = {}  # id → {n_entries, n_consolidations, ...}
        self._consolidation_counter: Dict[str, int] = {}

    # ── 属性 ──────────────────────────────────────────────────────────

    @property
    def tasks(self) -> List[str]:
        return list(set(self._store.keys()) | set(self._prototypes.keys()))

    @property
    def total_entries(self) -> int:
        return sum(len(v) for v in self._store.values())

    @property
    def total_prototypes(self) -> int:
        return sum(v.size(0) for v in self._prototypes.values())

    def get_prototypes(self, task_id: str) -> Optional[torch.Tensor]:
        """返回 [k, hidden] 原型张量."""
        return self._prototypes.get(task_id, None)

    def get_num_prototypes(self, task_id: str) -> int:
        p = self._prototypes.get(task_id, None)
        return p.size(0) if p is not None else 0

    # ── 核心: 存储 ────────────────────────────────────────────────────

    @torch.no_grad()
    def add_z_samples(
        self,
        task_id: str,
        z_states_list: List[List[torch.Tensor]],
        layer_importance: Optional[torch.Tensor] = None,
        dopamine_score: float = 0.0,
        world_model_surprise: float = 0.0,
        # 内在动机扩展
        concept_id: str = '',
        information_gain: float = 0.0,
        group_by_concept: bool = False,
    ):
        """追加一批收敛后的 z 表示到指定任务/概念.

        Args:
            task_id: 任务 ID
            concept_id: 概念 ID (ConceptDiscovery 分配)
            information_gain: ICM 信息增益 (用于加权)
            group_by_concept: True → 按 concept_id 分组存储 (而非 task_id)
        """
        group_key = concept_id if (self.enable_concept_grouping and group_by_concept and concept_id) else task_id
        store = self._store_by_concept if (self.enable_concept_grouping and group_by_concept and concept_id) else self._store

        if group_key not in store:
            store[group_key] = []
            self._consolidation_counter[group_key] = 0
            self._meta[group_key] = {'n_entries': 0, 'n_consolidations': 0}

        buf = store[group_key]
        for z_states in z_states_list:
            if layer_importance is None:
                π = torch.ones(12, dtype=torch.float)
            else:
                π = layer_importance.clone().cpu()

            buf.append({
                'z_states': [z.clone().cpu() for z in z_states],
                'layer_importance': π,
                'dopamine_score': dopamine_score,
                'world_model_surprise': float(world_model_surprise),
                'information_gain': float(information_gain),
                'retention_weight': 1.0 + max(float(world_model_surprise), 0.0) + float(information_gain),
                'seq_len': z_states[0].size(1),
                'task_id': task_id,
                'concept_id': concept_id,
            })

        # FIFO 淘汰
        while len(buf) > self.max_entries_per_task:
            buf.pop(0)

        self._meta[group_key]['n_entries'] = len(buf)

        # 自动触发异步 consolidate
        self._consolidation_counter[group_key] += 1
        if self._consolidation_counter[group_key] >= self.consolidation_frequency:
            self.consolidate(group_key)

    # ── 核心: 原型压缩 ─────────────────────────────────────────────────

    @torch.no_grad()
    def consolidate(self, group_key: Optional[str] = None, n_prototypes: Optional[int] = None):
        """对指定组 (任务/概念) 或全部组做 k-means 原型压缩.

        使用 information_gain 加权 retention_weight.
        """
        n_proto = n_prototypes or self.n_prototypes
        # 同时检查 _store 和 _store_by_concept
        all_stores = {'_store': self._store, '_store_by_concept': self._store_by_concept}

        if group_key:
            # 找到 group_key 所属的 store
            for store_name, store in all_stores.items():
                if group_key in store:
                    self._consolidate_one(store, group_key, n_proto)
                    return
        else:
            for store_name, store in all_stores.items():
                for gk in list(store.keys()):
                    self._consolidate_one(store, gk, n_proto)

    @torch.no_grad()
    def _consolidate_one(self, store: Dict, group_key: str, n_proto: int):
        """对一个组的条目做原型压缩."""
        if group_key not in store or not store[group_key]:
            return

        entries = store[group_key]

        # 收集顶层表示
        z_tops = []
        importances = []
        retention_weights = []
        info_gains = []
        for e in entries:
            z_top = e['z_states'][-1]
            z_tops.append(z_top.squeeze(0))
            importances.append(e['layer_importance'])
            rw = e.get('retention_weight', 1.0 + max(e.get('world_model_surprise', 0.0), 0.0)
                        + e.get('information_gain', 0.0))
            retention_weights.append(rw)
            info_gains.append(e.get('information_gain', 0.0))

        all_z = torch.cat(z_tops, dim=0)
        hidden = all_z.size(-1)
        k = min(n_proto, all_z.size(0))
        if k < 1:
            return

        centroids, assignments = _kmeans_cosine(all_z, k)

        # information_gain 加权评分
        n_total = all_z.size(0)
        seq_len = max(entries[0]['z_states'][-1].size(1), 1)
        point_entry_idx = torch.arange(n_total, device=assignments.device) // seq_len
        imp_stack = torch.stack(importances)
        retention_stack = torch.tensor(retention_weights, dtype=torch.float)
        info_gain_stack = torch.tensor(info_gains, dtype=torch.float)
        proto_importances = torch.zeros(k, 12, dtype=torch.float)
        proto_scores = torch.zeros(k, dtype=torch.float)
        proto_info_gains = torch.zeros(k, dtype=torch.float)
        for i in range(k):
            mask = assignments == i
            if mask.any():
                entry_indices = point_entry_idx[mask].unique().long()
                matched = imp_stack[entry_indices.clamp(0, len(importances) - 1)]
                entry_weights = retention_stack[entry_indices.clamp(0, len(retention_weights) - 1)]
                proto_importances[i] = (matched * entry_weights.unsqueeze(1)).mean(dim=0)
                proto_scores[i] = entry_weights.mean()
                proto_info_gains[i] = info_gain_stack[entry_indices.clamp(0, len(info_gains) - 1)].mean()
            else:
                proto_scores[i] = 1.0

        self._prototypes[group_key] = centroids
        self._prototype_importances[group_key] = proto_importances
        self._prototype_scores[group_key] = proto_scores

        # 2c: 淘汰低 retention 原型 — 结合 info_gain 加权
        effective_scores = proto_scores * (1.0 + 0.5 * proto_info_gains)
        keep_mask = effective_scores >= 0.3
        if keep_mask.any() and not keep_mask.all():
            n_pruned = (~keep_mask).sum().item()
            self._prototypes[group_key] = centroids[keep_mask]
            self._prototype_importances[group_key] = proto_importances[keep_mask]
            self._prototype_scores[group_key] = proto_scores[keep_mask]
            self._meta[group_key]['n_pruned'] = self._meta[group_key].get('n_pruned', 0) + n_pruned
        elif not keep_mask.any():
            best_idx = effective_scores.argmax()
            self._prototypes[group_key] = centroids[best_idx:best_idx+1]
            self._prototype_importances[group_key] = proto_importances[best_idx:best_idx+1]
            self._prototype_scores[group_key] = proto_scores[best_idx:best_idx+1]
            self._meta[group_key]['n_pruned'] = self._meta[group_key].get('n_pruned', 0) + (k - 1)

        self._meta[group_key]['n_consolidations'] += 1
        self._meta[group_key]['compression_ratio'] = (
            len(entries) * entries[0]['seq_len'] / max(k, 1)
        )
        self._consolidation_counter[group_key] = 0

    # ── 核心: 采样原型 ────────────────────────────────────────────────

    def sample_prototypes(
        self,
        batch_size: int = 16,
        device: str = 'cuda:0',
    ) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        """从所有任务的原型中采样一批.

        Returns:
            [(task_id, proto_z [1, hidden], proto_imp [12])] × batch_size
        """
        if self.total_prototypes == 0:
            return []

        # 收集所有 (task_id, prototype_vector, importance, weight)
        all_protos: List[Tuple[str, torch.Tensor, torch.Tensor, float]] = []
        for tid, protos in self._prototypes.items():
            imp = self._prototype_importances.get(tid, torch.ones(12, dtype=torch.float))
            score = self._prototype_scores.get(tid, torch.ones(protos.size(0), dtype=torch.float))
            for i in range(protos.size(0)):
                weight = max(float(score[i]) if i < score.size(0) else 1.0, 0.1)
                all_protos.append((tid, protos[i: i + 1], imp[i], weight))

        if not all_protos:
            return []

        weights = torch.tensor([item[3] for item in all_protos], dtype=torch.float32)
        weights = weights / weights.sum().clamp_min(1e-8)
        n = min(batch_size, len(all_protos))
        idx = torch.multinomial(weights, n, replacement=False).tolist()
        result = []
        for i in idx:
            tid, pz, imp, _ = all_protos[i]
            result.append((tid, pz.to(device), imp.to(device)))
        return result

    # ── 核心: 表示级回放 ─────────────────────────────────────────────

    def replay_loss(
        self,
        model: nn.Module,
        batch_size: int = 16,
        device: str = 'cuda:0',
        pos_emb: Tuple = (None, None),
    ) -> Optional[torch.Tensor]:
        """表示级回放 loss.

        不同于 CE 回放 (重新预测字节), 这个 loss 是:
          F_replay = Σ_{ℓ} ½ · π_ℓ · ‖z_proto_ℓ - μ_total(ℓ)‖²

        本质: 告诉模型"对于这类问题, 你的第 ℓ 层表示应该接近原型".
        梯度主要走 temporal_proj + topdown_proj → 固化层间预测结构.

        Returns:
            scalar tensor, 或 None (无原型)
        """
        if self.total_prototypes == 0:
            return None

        proto_batch = self.sample_prototypes(batch_size, device)
        if not proto_batch:
            return None

        total_loss = 0.0
        L = model.num_sub_layers

        for _task_id, z_proto, proto_imp in proto_batch:
            # z_proto: [1, hidden] — 单个原型向量
            # 把它展开成一个 "伪序列": [1, seq=1, hidden]
            z_prev = z_proto.unsqueeze(0)  # [1, 1, hidden]

            ℓ_loss = 0.0
            for ℓ in range(1, L + 1):
                μ_bu = model.predict(ℓ, z_prev, pos_emb)  # [1, 1, hidden]
                μ_bu_res = μ_bu + z_prev

                # 单步时序 (t=1, 无 t-1, 所以 μ_temp = 0)
                # 单步 top-down (无 t-1, 所以 μ_down = 0)
                μ_total = μ_bu_res

                # proto_imp[ℓ-1] ∈ [0.1, 3.0] — 该层精度权重
                π_ℓ = proto_imp[ℓ - 1].item()
                π_ℓ = max(0.1, min(3.0, π_ℓ))

                ε = z_proto - μ_total  # [1, 1, hidden]
                ℓ_loss += 0.5 * π_ℓ * (ε ** 2).sum()

                # 更新 z_prev = 当前层的原型 (用于下一层的 bottom-up)
                # 下一层预测 z_{ℓ+1} 需要 z_ℓ 作为输入
                if ℓ < L:
                    z_prev = z_proto.unsqueeze(0)  # 保持 [1, 1, hidden]

            total_loss += ℓ_loss

        return total_loss / max(len(proto_batch), 1)

    # ── 序列级回放 (替代方案: 用原型重建伪序列) ─────────────────────

    def replay_loss_sequence(
        self,
        model: nn.Module,
        batch_size: int = 4,
        seq_len: int = 8,
        device: str = 'cuda:0',
        pos_emb: Tuple = (None, None),
    ) -> Optional[torch.Tensor]:
        """序列级表示回放 — 用多个原型拼接成伪序列.

        每个原型对应序列中一个位置的表示, 让模型学习
        层间 + 时序 + 自上而下三重预测.
        """
        if self.total_prototypes == 0:
            return None

        # 采样原型 (可能来自多个任务)
        proto_batch = self.sample_prototypes(batch_size * seq_len, device)
        if not proto_batch:
            return None

        # 重组为 batch 个伪序列
        n_available = len(proto_batch)
        n_seq = min(batch_size, n_available // max(seq_len, 1))
        if n_seq < 1:
            return None

        total_loss = 0.0
        L = model.num_sub_layers

        for seq_idx in range(n_seq):
            start = seq_idx * seq_len
            seq_protos = proto_batch[start: start + seq_len]

            # 构建伪序列张量: [1, seq_len, hidden]
            pseudo_seq = torch.cat([p[1] for p in seq_protos], dim=1)  # [1, seq_len, hidden]
            seq_imp = torch.stack([p[2] for p in seq_protos], dim=0)   # [seq_len, 12]

            # 逐层计算预测误差
            ℓ_loss = 0.0
            z_prev = pseudo_seq  # [1, seq_len, hidden] — z_0 位置

            for ℓ in range(1, L + 1):
                # 自下而上: sublayer(z_{ℓ-1}) + residual
                μ_bu = model.predict(ℓ, z_prev, pos_emb)
                μ_bu_res = μ_bu + z_prev

                # 时序: z_ℓ(t-1) → z_ℓ(t)
                if seq_len > 1:
                    z_prev_t = pseudo_seq[:, :-1, :]
                    z_temp = model.temporal_proj[ℓ - 1](z_prev_t)
                    μ_temp = torch.cat([torch.zeros_like(pseudo_seq[:, :1, :]), z_temp], dim=1)
                else:
                    μ_temp = torch.zeros_like(pseudo_seq)

                # 自上而下: z_{ℓ+1}(t-1) → z_ℓ(t)
                if ℓ < L and seq_len > 1:
                    z_down_prev = pseudo_seq[:, :-1, :]
                    z_down = model.topdown_proj[ℓ - 1](z_down_prev)
                    μ_down = torch.cat([torch.zeros_like(pseudo_seq[:, :1, :]), z_down], dim=1)
                else:
                    μ_down = torch.zeros_like(pseudo_seq)

                μ_total = μ_bu_res + μ_temp + μ_down

                # 逐位置精度加权
                errors = (pseudo_seq - μ_total) ** 2  # [1, seq_len, hidden]
                for t in range(seq_len):
                    π_t = seq_imp[t, ℓ - 1].item()
                    π_t = max(0.1, min(3.0, π_t))
                    ℓ_loss += 0.5 * π_t * errors[0, t].sum()

                # 更新 z_prev = 当前层的表示 (所有层用同一个伪序列)
                z_prev = pseudo_seq

            total_loss += ℓ_loss

        return total_loss / max(n_seq, 1)

    # ── 序列级回放（完整多层版本）─────────────────────────────────

    def replay_loss_full(
        self,
        model: nn.Module,
        z_conv_exemplar: List[torch.Tensor],
        layer_importance: torch.Tensor,
        pos_emb: Tuple = (None, None),
    ) -> torch.Tensor:
        """对一条完整 z_conv 做表示级回放.

        z_conv_exemplar: [13] × [1, seq, hidden] — 从 AbstractionBank 取出的一条 z 记录
        layer_importance: [12] — 每层精度
        """
        L = model.num_sub_layers
        total_loss = 0.0

        # 使用存储的 z_states 作为目标
        z_targets = z_conv_exemplar  # [13] × [1, seq, hidden]
        seq_len = z_targets[0].size(1)

        for ℓ in range(1, L + 1):
            z_ℓ = z_targets[ℓ]
            z_ℓ_minus_1 = z_targets[ℓ - 1]

            # 自下而上
            μ_bu = model.predict(ℓ, z_ℓ_minus_1, pos_emb)
            μ_bu_res = μ_bu + z_ℓ_minus_1

            # 时序
            if seq_len > 1:
                z_prev_t = z_ℓ[:, :-1, :]
                z_temp = model.temporal_proj[ℓ - 1](z_prev_t)
                μ_temp = torch.cat([torch.zeros_like(z_ℓ[:, :1, :]), z_temp], dim=1)
            else:
                μ_temp = torch.zeros_like(z_ℓ)

            # 自上而下
            if ℓ < L and seq_len > 1:
                z_down_prev = z_targets[ℓ + 1][:, :-1, :]
                z_down = model.topdown_proj[ℓ - 1](z_down_prev)
                μ_down = torch.cat([torch.zeros_like(z_ℓ[:, :1, :]), z_down], dim=1)
            else:
                μ_down = torch.zeros_like(z_ℓ)

            μ_total = μ_bu_res + μ_temp + μ_down

            π_ℓ = layer_importance[ℓ - 1].item()
            π_ℓ = max(0.1, min(3.0, π_ℓ))

            ε = z_ℓ - μ_total
            total_loss += 0.5 * π_ℓ * (ε ** 2).sum()

        return total_loss

    # ── 序列化 ────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        """返回可序列化的状态字典."""
        proto_cpu = {tid: p.cpu() for tid, p in self._prototypes.items()}
        imp_cpu = {tid: imp.cpu() for tid, imp in self._prototype_importances.items()}
        proto_score_cpu = {tid: score.cpu() for tid, score in self._prototype_scores.items()}

        return {
            'prototypes': proto_cpu,
            'prototype_importances': imp_cpu,
            'prototype_scores': proto_score_cpu,
            'meta': self._meta,
            'config': {
                'max_entries_per_task': self.max_entries_per_task,
                'n_prototypes': self.n_prototypes,
                'enable_concept_grouping': self.enable_concept_grouping,
            },
        }

    def load_state_dict(self, state: dict):
        """从状态字典恢复."""
        self._prototypes = {tid: p.clone() for tid, p in state.get('prototypes', {}).items()}
        self._prototype_importances = {
            tid: imp.clone() for tid, imp in state.get('prototype_importances', {}).items()
        }
        self._prototype_scores = {
            tid: score.clone() for tid, score in state.get('prototype_scores', {}).items()
        }
        self._meta = state.get('meta', {})
        cfg = state.get('config', {})
        self.max_entries_per_task = cfg.get('max_entries_per_task', self.max_entries_per_task)
        self.n_prototypes = cfg.get('n_prototypes', self.n_prototypes)
        self.enable_concept_grouping = cfg.get('enable_concept_grouping', self.enable_concept_grouping)

        self._store.clear()
        self._store_by_concept.clear()

    def sample_replay_batch(
        self,
        batch_size: int = 16,
        device: str = 'cuda:0',
    ):
        """从原始存储(非原型)采样一批 z_states, 用于 Hebbian 回放.

        Returns:
            (z_init_batch, seq_len) | None
            z_init_batch: list[tensor] — 13 层 z stack [B, S, H]
            seq_len: int
        """
        all_entries: List[dict] = []
        for store in [self._store, self._store_by_concept]:
            for group_key in store:
                all_entries.extend(store[group_key])

        if not all_entries:
            return None

        n = min(batch_size, len(all_entries))
        idx = torch.randperm(len(all_entries))[:n].tolist()
        chosen = [all_entries[i] for i in idx]

        # 取第一项的 seq_len (假设 batch 内一致)
        seq_len = chosen[0]['seq_len']

        # stack z_states: 每个 entry 有 13 个 [1, S, H] 层
        z_stack = []
        for layer_idx in range(13):
            layer_tensors = []
            for entry in chosen:
                z = entry['z_states'][layer_idx].to(device)  # [1, S, H]
                layer_tensors.append(z)
            z_stack.append(torch.cat(layer_tensors, dim=0))  # [B, S, H]

        return z_stack, seq_len

    def __len__(self):
        return self.total_prototypes

    def __repr__(self):
        tasks_str = ', '.join(
            f'{tid}: {self.get_num_prototypes(tid)} proto'
            for tid in self._prototypes
        )
        return f'AbstractionBank({tasks_str})'


# VariationalReplayer — 结构变体生成 + 吸引子测试

class VariationalReplayer:
    """结构变体生成器.

    从 AbstractionBank 取原型 z_proto, 加扰动后重新 PC infer,
    看模型能否稳定收敛到附近的吸引子.

    能 → 该区域学到了真正的结构 (吸引盆宽)
    不能 → 只是记忆了一个点 (吸引盆窄)
    """

    def __init__(self, model: nn.Module, bank: AbstractionBank):
        self.model = model
        self.bank = bank

    @torch.no_grad()
    def generate_variants(
        self,
        prototype_z: torch.Tensor,        # [1, hidden]
        n_variants: int = 5,
        noise_scale: float = 0.1,
        T_infer: int = 4,
        gamma: float = 0.1,
        pos_emb: Tuple = (None, None),
    ) -> List[dict]:
        """从单个原型生成结构变体.

        Args:
            prototype_z: 原型向量 [1, hidden]
            n_variants: 变体数
            noise_scale: 扰动标准差
            T_infer: PC 推理步数
            gamma: 推理步长

        Returns:
            [{
                'z_reconverged': List[Tensor] — [13] × [1, 1, hidden]
                'convergence_dist': float — ‖z_new - z_proto‖
                'F_drop': float — F[0] - F[T-1] (能量下降)
                'converged': bool — dist < threshold
            }]
        """
        # 用原型构建全 13 层表示: z_0..z_12
        # 对于局部 Conv 模型, 各层表示维度相同 = [1, 1, hidden]
        hidden = prototype_z.size(-1)
        results = []

        for _ in range(n_variants):
            # 对各层加独立的噪声
            z_noised = []
            for layer_idx in range(13):
                noise = torch.randn_like(prototype_z) * noise_scale
                z_noised.append((prototype_z + noise).unsqueeze(0))

            # 从扰动后的 z 做 PC 推理
            z_reconverged, errors_hist, F_hist, _ = self.model.spatiotemporal_infer(
                z_noised, pos_emb, gamma=gamma, T=T_infer,
                return_errors=False, return_pred_loss=False,
            )

            convergence_dist = (z_reconverged[-1] - prototype_z.unsqueeze(0)).norm().item()
            F_drop = F_hist[0] - F_hist[-1] if len(F_hist) > 1 else 0.0
            converged = convergence_dist < 0.5  # 启发式阈值

            results.append({
                'z_reconverged': z_reconverged,
                'convergence_dist': convergence_dist,
                'F_drop': F_drop,
                'converged': converged,
            })

        return results

    @torch.no_grad()
    def evaluate_basin(
        self,
        task_id: str,
        n_variants: int = 10,
        noise_scales: List[float] = [0.05, 0.1, 0.2, 0.5],
        T_infer: int = 4,
        gamma: float = 0.1,
        device: str = 'cuda:0',
        pos_emb: Tuple = (None, None),
    ) -> dict:
        """评估指定任务的吸引子强度.

        Returns:
            {
                'basin_attraction': {noise: ratio} — 各噪声级别下的收敛率
                'avg_convergence_dist': {noise: float}
                'avg_F_drop': {noise: float}
            }
        """
        prototypes = self.bank.get_prototypes(task_id)
        if prototypes is None or prototypes.size(0) == 0:
            return {'basin_attraction': {}, 'avg_convergence_dist': {}, 'avg_F_drop': {}}

        results = {}
        for noise in noise_scales:
            n_converged = 0
            total_dist = 0.0
            total_F_drop = 0.0
            n_total = 0

            for proto_idx in range(prototypes.size(0)):
                proto_z = prototypes[proto_idx: proto_idx + 1].to(device)  # [1, hidden]
                variants = self.generate_variants(
                    proto_z, n_variants=n_variants,
                    noise_scale=noise, T_infer=T_infer, gamma=gamma, pos_emb=pos_emb,
                )
                for v in variants:
                    n_total += 1
                    if v['converged']:
                        n_converged += 1
                    total_dist += v['convergence_dist']
                    total_F_drop += v['F_drop']

            results[noise] = {
                'basin_attraction': n_converged / max(n_total, 1),
                'avg_convergence_dist': total_dist / max(n_total, 1),
                'avg_F_drop': total_F_drop / max(n_total, 1),
            }

        return results


# AbstractionSniffer — 抽象级遗忘检测

class AbstractionSniffer:
    """抽象级遗忘检测.

    检测的不是"还记不记得字节序列", 而是"原型表示是否偏移":
    - 用 AbstractionBank 中的原型作为锚点
    - 让模型重新 infer 对应 task 的数据
    - 计算收敛后的 z 与存储的原型之间的 cosine 距离
    - cosine < 0.7 → 抽象已偏移 → 触发回放

    世界模型集成:
    - 同时计算 transition surprise 作为第二维度
    - surprise 高 + cosine 低 → 双重确认漂移
    - surprise 高但 cosine 正常 → 可能是学到了新的变异, 不触发漂移
    """

    def __init__(
        self,
        bank: AbstractionBank,
        model: nn.Module,
        check_interval: int = 200,
        drift_threshold: float = 0.7,
        repair_steps: int = 10,
        repair_lr_factor: float = 0.3,
        world_model=None,
        world_model_context_dim: int = 5,
    ):
        self.bank = bank
        self.model = model
        self.check_interval = check_interval
        self.drift_threshold = drift_threshold
        self.repair_steps = repair_steps
        self.repair_lr_factor = repair_lr_factor
        self._repairing = False
        self._repair_counter = 0
        self._last_similarities: Dict[str, float] = {}
        self._last_wm_surprises: Dict[str, float] = {}
        self.world_model = world_model
        self.world_model_context_dim = world_model_context_dim

    # ── 属性 ──────────────────────────────────────────────────────────

    @property
    def is_repairing(self) -> bool:
        return self._repairing

    @property
    def last_similarities(self) -> dict:
        return dict(self._last_similarities)

    # ── 核心检测 ──────────────────────────────────────────────────────

    @torch.no_grad()
    def check_abstraction_drift(
        self,
        task_id: str,
        device: str,
        n_samples: int = 16,
        pos_emb: Tuple = (None, None),
    ) -> float:
        """检测指定任务的抽象是否漂移.

        流程:
          1. 从 bank 获取 task 的原型
          2. 从 bank 获取原始 z 存储, 取一条 re-infer
          3. 计算收敛后的顶层表示与所有原型的 max cosine 相似度
          4. 如果世界模型可用, 计算 transition surprise 作为第二维度

        Returns:
            effective_sim ∈ [-1, 1], < drift_threshold → 遗忘
            (world model surprise 高时会降低 effective_sim)
        """
        prototypes = self.bank.get_prototypes(task_id)
        if prototypes is None or prototypes.size(0) == 0:
            return 1.0  # 无原型可比较 → 无遗忘

        # 从 bank 中取原始 z 存储
        if task_id not in self.bank._store or not self.bank._store[task_id]:
            return 1.0

        entries = self.bank._store[task_id]
        n = min(n_samples, len(entries))
        idx = torch.randperm(len(entries))[:n].tolist()

        proto_device = prototypes.device
        prototypes_norm = F.normalize(prototypes, dim=-1)  # [k, hidden]

        similarities = []
        wm_surprises = []
        for i in idx:
            entry = entries[i]
            z_top_stored = entry['z_states'][-1]  # [1, seq, hidden]

            # 当前模型对同一样本重新 infer
            z_init = [z.clone().to(device) for z in entry['z_states']]
            z_reconv, _, _, _ = self.model.spatiotemporal_infer(
                z_init, pos_emb, gamma=0.1, T=4,
                return_errors=False, return_pred_loss=False,
            )

            # 取顶层平均池化
            z_current = z_reconv[-1].mean(dim=1)  # [1, hidden]
            z_current_norm = F.normalize(z_current, dim=-1)

            # 与所有原型的余弦相似度
            cos_sim = (z_current_norm @ prototypes_norm.T)  # [1, k]
            max_sim = cos_sim.max().item()
            similarities.append(max_sim)

            # 世界模型 transition surprise
            if self.world_model is not None:
                ctx = torch.zeros(1, self.world_model_context_dim, device=device)
                _, uncertainty = self.world_model(z_current, ctx)
                wm_surprises.append(uncertainty.mean().item())

        avg_sim = sum(similarities) / max(len(similarities), 1)
        self._last_similarities[task_id] = avg_sim

        if wm_surprises:
            avg_wm = sum(wm_surprises) / len(wm_surprises)
            self._last_wm_surprises[task_id] = avg_wm
            # 世界模型 surprise 高时降低 effective_sim
            # (高 surprise 意味着 transition dynamics 不稳定 → 可能已漂移)
            wm_penalty = min(avg_wm * 0.5, 0.3)  # 最多扣 0.3
            effective_sim = avg_sim - wm_penalty
        else:
            effective_sim = avg_sim

        return effective_sim

    def check(
        self,
        global_step: int,
        device: str,
        pos_emb: Tuple = (None, None),
    ) -> Optional[List[str]]:
        """多任务抽象漂移嗅探.

        Returns:
            漂移的任务 ID 列表, 或 None
        """
        if self.bank.total_prototypes == 0:
            return None

        if not self._repairing and global_step % self.check_interval != 0:
            return None

        drifted = []
        for tid in self.bank._prototypes:
            sim = self.check_abstraction_drift(tid, device, pos_emb=pos_emb)
            if sim < self.drift_threshold:
                drifted.append(tid)

        return drifted if drifted else None

    # ── 修复管理 ──────────────────────────────────────────────────────

    def repair_begin(self, optimizer, current_lr: float) -> float:
        """进入修复模式: 降低 LR."""
        self._repairing = True
        self._repair_counter = 0
        repair_lr = current_lr * self.repair_lr_factor
        for pg in optimizer.param_groups:
            pg['lr'] = repair_lr
        return repair_lr

    def repair_end(self, optimizer, restore_lr: float):
        """退出修复模式: 恢复 LR."""
        self._repairing = False
        self._repair_counter = 0
        for pg in optimizer.param_groups:
            pg['lr'] = restore_lr

    def get_replay_batch(
        self,
        batch_size: int,
        device: str,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """获取修复用的回放 batch — 从 AbstractionBank 采样原型."""
        proto_batch = self.bank.sample_prototypes(batch_size, device)
        if not proto_batch:
            return None
        # 返回原型 z 和重要性 (用于 replay_loss)
        # 注意: 这不是 byte 级回放, 而是表示级回放
        # 调用方应使用 bank.replay_loss() 而非 CE loss
        return proto_batch

    def get_drift_report(self) -> str:
        """生成漂移检测文本报告."""
        lines = ['[AbstractionSniffer] Drift Report:']
        for tid, sim in sorted(self._last_similarities.items()):
            status = '✓' if sim >= self.drift_threshold else '✗ DRIFT'
            lines.append(f'  {tid}: cosine={sim:.4f} {status}')
        return '\n'.join(lines)
