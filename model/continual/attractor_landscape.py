"""
吸引子景观分析工具 (Attractor Landscape Analyzer).

不存储任何记忆，只对 PC latent space 做纯测量:
  - basin_depth:    从原型出发加噪声 → 收敛回同一 attractor 的成功率
  - basin_width:    能容忍多少扰动仍收敛到同一 attractor
  - basin_collapse: 检测某 attractor 是否被后续学习覆盖
  - landscape_entropy: 整个 latent space 的吸引子分布熵

核心理念:
  记忆 = 吸引子盆地。强度 = basin 的深度 × 宽度。
  不查表，用 PC inference 重构。
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn.functional as F

from model.continual.abstraction_bank import AbstractionBank, compute_layer_importance


class AttractorLandscape:
    """吸引子景观分析器 — 只读测量，不修改任何状态。"""

    def __init__(
        self,
        num_sub_layers: int = 12,
        n_noise_levels: int = 5,
        n_variants_per_proto: int = 5,
        T_infer: int = 4,
    ):
        self.num_sub_layers = num_sub_layers
        self.n_noise_levels = n_noise_levels
        self.n_variants_per_proto = n_variants_per_proto
        self.T_infer = T_infer

    # 核心测量

    @torch.no_grad()
    def estimate_basin_depth(
        self,
        model: torch.nn.Module,
        prototype: torch.Tensor,
        noise_levels: Optional[List[float]] = None,
        pos_emb: Tuple = (None, None),
        gamma: float = 0.1,
        device: str = "cuda:0",
        similarity_threshold: float = 0.85,
    ) -> Dict[str, float]:
        """测量吸引子深度: 加噪声后 PC infer 收敛回原点的成功率。

        Args:
            prototype: [1, hidden] 原型向量 (作为 attractor 中心)
            noise_levels: 噪声标准差列表 (如 [0.05, 0.1, 0.2, 0.5, 1.0])

        Returns:
            {noise_str: convergence_rate}
        """
        if noise_levels is None:
            noise_levels = [0.05, 0.1, 0.2, 0.5, 1.0]

        proto = prototype.to(device)
        hidden = proto.size(-1)

        # 将 [1, hidden] 扩展为 [1, 1, hidden] 用于 PC inference
        z_proto = [
            torch.zeros(1, 1, hidden, device=device, dtype=torch.float16)
            for _ in range(self.num_sub_layers + 1)
        ]
        z_proto[-1] = proto.unsqueeze(1)  # [1, 1, hidden]

        results = {}
        for noise in noise_levels:
            n_converged = 0
            n_total = 0

            for _ in range(self.n_variants_per_proto):
                # 在顶层加噪声
                z_noisy = [z.clone() for z in z_proto]
                z_noisy[-1] = z_noisy[-1] + torch.randn_like(z_noisy[-1]) * noise

                # PC inference: 看噪声状态能否收敛回原型
                z_conv, *_ = model.spatiotemporal_infer(
                    z_noisy,
                    pos_emb,
                    gamma=gamma,
                    T=self.T_infer,
                    return_errors=False,
                    return_pred_loss=False,
                )
                z_final = z_conv[-1]  # [1, 1, hidden]

                # 余弦相似度到原始原型
                sim = F.cosine_similarity(z_final.squeeze(0), proto, dim=-1).item()
                if sim >= similarity_threshold:
                    n_converged += 1
                n_total += 1

            results[f"{noise:.2f}"] = n_converged / max(n_total, 1)

        return results

    @torch.no_grad()
    def estimate_basin_width(
        self,
        model: torch.nn.Module,
        prototype: torch.Tensor,
        pos_emb: Tuple = (None, None),
        gamma: float = 0.1,
        device: str = "cuda:0",
        similarity_threshold: float = 0.85,
    ) -> float:
        """估计吸引子宽度: 二分搜索找到最大噪声容忍度。

        Returns:
            max_noise: 仍能收敛的最大噪声标准差
        """
        lo, hi = 0.01, 2.0
        best = 0.0
        for _ in range(8):  # 二分 8 轮
            mid = (lo + hi) / 2.0
            depth = self.estimate_basin_depth(
                model,
                prototype,
                noise_levels=[mid],
                pos_emb=pos_emb,
                gamma=gamma,
                device=device,
                similarity_threshold=similarity_threshold,
            )
            rate = depth.get(f"{mid:.2f}", 0.0)
            if rate >= 0.5:
                best = mid
                lo = mid
            else:
                hi = mid
        return best

    @torch.no_grad()
    def detect_basin_collapse(
        self,
        model: torch.nn.Module,
        abstraction_bank: AbstractionBank,
        group_key: str,
        pos_emb: Tuple = (None, None),
        gamma: float = 0.1,
        device: str = "cuda:0",
    ) -> Dict[str, float]:
        """检测某组原型的吸引子是否坍缩。

        坍缩指标:
          - 低噪声 (0.05) 收敛率 < 0.5 → 严重坍缩
          - 中等噪声 (0.2) 收敛率 < 0.3 → 中度坍缩

        Returns:
            {'collapse_level': float (0-1), 'n_prototypes_collapsed': int, 'details': {...}}
        """
        prototypes = abstraction_bank.get_prototypes(group_key)
        if prototypes is None or prototypes.size(0) == 0:
            return {"collapse_level": 1.0, "n_prototypes_collapsed": 0, "details": {}}

        collapsed = 0
        details = {}
        for i in range(prototypes.size(0)):
            proto = prototypes[i : i + 1, :]  # [1, hidden]
            depth_low = self.estimate_basin_depth(
                model,
                proto,
                noise_levels=[0.05, 0.2],
                pos_emb=pos_emb,
                gamma=gamma,
                device=device,
            )
            low_rate = depth_low.get("0.05", 0.0)
            mid_rate = depth_low.get("0.20", 0.0)
            is_collapsed = low_rate < 0.5 or mid_rate < 0.3
            if is_collapsed:
                collapsed += 1
            details[f"proto_{i}"] = {
                "low_noise_rate": low_rate,
                "mid_noise_rate": mid_rate,
                "collapsed": is_collapsed,
            }

        collapse_level = collapsed / max(prototypes.size(0), 1)
        return {
            "collapse_level": collapse_level,
            "n_prototypes_collapsed": collapsed,
            "total_prototypes": prototypes.size(0),
            "details": details,
        }

    @torch.no_grad()
    def compute_landscape_entropy(
        self,
        abstraction_bank: AbstractionBank,
    ) -> Dict[str, float]:
        """计算 latent space 的吸引子分布熵。

        原理:
          - 所有原型 centroid 的归一化 → 高维单位球面上的分布
          - 计算 pairwise cosine 相似度的直方图熵
          - 熵高 → 吸引子分散 (多样化记忆)
          - 熵低 → 吸引子聚集 (同质化记忆)

        Returns:
            {'entropy': float, 'max_entropy': float, 'normalized_entropy': float, 'n_groups': int}
        """
        all_centroids = []
        for tid, protos in abstraction_bank._prototypes.items():
            for i in range(protos.size(0)):
                all_centroids.append(protos[i : i + 1, :])

        if len(all_centroids) < 2:
            return {
                "entropy": 0.0,
                "max_entropy": 0.0,
                "normalized_entropy": 0.0,
                "n_groups": len(all_centroids),
            }

        all_z = torch.cat(all_centroids, dim=0)  # [N, hidden]
        all_z = F.normalize(all_z, dim=-1)
        sim_matrix = all_z @ all_z.T  # [N, N]

        # 将相似度离散化为 16 个 bin
        bins = torch.linspace(-1, 1, 17, device=sim_matrix.device)
        triu = torch.triu_indices(sim_matrix.size(0), sim_matrix.size(1), offset=1)
        vals = sim_matrix[triu[0], triu[1]]
        hist = torch.histc(vals, bins=16, min=-1, max=1)
        hist = hist / hist.sum().clamp_min(1e-8)

        # 熵
        entropy = -(hist * torch.log(hist.clamp_min(1e-8))).sum().item()
        max_entropy = math.log(16)
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0

        return {
            "entropy": entropy,
            "max_entropy": max_entropy,
            "normalized_entropy": normalized,
            "n_groups": len(all_centroids),
        }

    @torch.no_grad()
    def find_weakest_basin(
        self,
        model: torch.nn.Module,
        abstraction_bank: AbstractionBank,
        pos_emb: Tuple = (None, None),
        gamma: float = 0.1,
        device: str = "cuda:0",
        top_k: int = 3,
    ) -> List[Tuple[str, int, float]]:
        """找出最脆弱的 top-k 吸引子 (宽度最小)。

        Returns:
            [(group_key, proto_idx, basin_width), ...]
        """
        candidates = []
        for gk in abstraction_bank._prototypes:
            protos = abstraction_bank._prototypes[gk]
            for i in range(protos.size(0)):
                proto = protos[i : i + 1, :]
                width = self.estimate_basin_width(
                    model,
                    proto,
                    pos_emb=pos_emb,
                    gamma=gamma,
                    device=device,
                )
                candidates.append((gk, i, width))

        candidates.sort(key=lambda x: x[2])  # 按宽度升序
        return candidates[:top_k]

    @torch.no_grad()
    def full_landscape_report(
        self,
        model: torch.nn.Module,
        abstraction_bank: AbstractionBank,
        pos_emb: Tuple = (None, None),
        gamma: float = 0.1,
        device: str = "cuda:0",
    ) -> dict:
        """生成完整的景观报告 — 用于 GUI 展示和调度决策。

        Returns:
            dict with keys:
              - n_prototypes_total
              - n_groups
              - avg_basin_depth (各噪声级别平均)
              - collapse_ratio
              - entropy_metrics
              - group_reports: {group_key: {...}}
        """
        report = {
            "n_prototypes_total": abstraction_bank.total_prototypes,
            "n_groups": len(abstraction_bank._prototypes),
            "collapse_ratio": 0.0,
            "entropy_metrics": self.compute_landscape_entropy(abstraction_bank),
            "group_reports": {},
        }

        if abstraction_bank.total_prototypes == 0:
            return report

        collapsed_total = 0
        proto_total = 0
        depth_accum = {}

        for gk in abstraction_bank._prototypes:
            protos = abstraction_bank._prototypes[gk]
            if protos.size(0) == 0:
                continue

            grp_report = {"n_prototypes": protos.size(0), "basin_depths": {}}
            for i in range(protos.size(0)):
                proto = protos[i : i + 1, :]
                depths = self.estimate_basin_depth(
                    model,
                    proto,
                    pos_emb=pos_emb,
                    gamma=gamma,
                    device=device,
                )
                for noise_str, rate in depths.items():
                    if noise_str not in depth_accum:
                        depth_accum[noise_str] = []
                    depth_accum[noise_str].append(rate)

                # 坍缩检查
                low_rate = depths.get("0.05", 0.0)
                if low_rate < 0.5:
                    collapsed_total += 1
                proto_total += 1

            report["group_reports"][gk] = grp_report

        report["collapse_ratio"] = collapsed_total / max(proto_total, 1)
        report["avg_basin_depths"] = {k: sum(v) / len(v) for k, v in depth_accum.items()}
        return report
