"""
LearningEngine
Hebbian 可塑性 + 稳态 + 阈值调节.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from model.constants import (
    F_BCM_SLOPE,
    F_BCM_ZERO,
    F_EPS,
    F_FIRING_RATE,
    F_MU,
    F_PI,
    F_THRESHOLD,
    F_Z,
    F_Z_PREV,
)

if TYPE_CHECKING:
    from .tensor_pool import TensorNeuronPool


def compute_precision_scales(
    pool: TensorNeuronPool,
    D: float,
    ACh: float,
    eta: float = 1.0,
) -> None:
    """逐神经元精度权重: pi = 1 + eta*D*|eps| + eta*ACh*|eps| (批量 tensor)."""
    pool.learning.compute_precision_scales(D, ACh, eta)


class LearningEngine:
    """学习与稳态引擎
    Hebbian 更新、LM head 学习、阈值调节、稳态步骤.

    通过 self.pool 引用访问共享张量, 扩容后自动指向新张量.
    """

    def __init__(self, pool):
        self.pool = pool

    def hebbian_pass(
        self,
        active_nids: torch.Tensor,
        eta: float,
        oja_alpha: float,
        dopamine: float,
    ) -> float:
        """批量 Hebbian + Oja 更新活跃神经元的入突触.

        dw = eta * D * pi * eps * z_pre - oja_alpha * D * eps^2 * w
        """
        A = active_nids.shape[0]
        if A == 0:
            return 0.0

        in_s = self.pool.in_ptrs[active_nids.long()]  # [A, K]
        valid = (in_s >= 0) & self.pool.syn_alive[in_s.long()]
        in_s = in_s.long()
        pre_ids = self.pool.pre_id[in_s]
        valid &= pre_ids >= 0

        w = self.pool.weight[in_s]  # [A, K]
        eps_post = self.pool.state[active_nids.long(), F_EPS].unsqueeze(-1)  # [A, 1]
        pi_post = self.pool.state[active_nids.long(), F_PI].unsqueeze(-1)  # [A, 1]
        z_pre = torch.where(valid, self.pool.state[pre_ids.long(), F_Z], torch.zeros_like(w))

        eta_eff = eta * dopamine * pi_post

        # Hebb: eta * eps * z_pre
        hebb = eta_eff * eps_post * z_pre
        # BCM 滑动阈值 (个性化)
        bcm_avg = self.pool.state[active_nids.long(), F_FIRING_RATE].unsqueeze(-1)
        bcm_slope = self.pool.state[active_nids.long(), F_BCM_SLOPE].unsqueeze(-1)
        bcm_zero = self.pool.state[active_nids.long(), F_BCM_ZERO].unsqueeze(-1)
        bcm_gain = 1.0 - bcm_slope * (bcm_avg - bcm_zero)
        bcm_gain = torch.clamp(bcm_gain, -1.0, 1.0)
        hebb = hebb * bcm_gain
        # Oja: -alpha * D * eps^2 * w — 仅在 |w| > 0.8 时介入
        oja_trigger = (w.abs() > 0.8).to(torch.float16)
        oja = -oja_alpha * dopamine * (eps_post**2) * w * oja_trigger

        dw = hebb + oja
        dw = torch.where(valid, dw, torch.zeros_like(dw))

        self.pool.weight.scatter_add_(
            0,
            in_s[valid].long(),
            dw[valid],
        )
        self.pool.trace[in_s[valid].long()] = self.pool.trace[in_s[valid].long()] * 0.9 + z_pre[valid].abs() * 0.1
        self.pool.syn_age[in_s[valid].long()] += 1

        return float(dw.abs().sum().item())

    def hebbian_temporal(
        self,
        active_nids: torch.Tensor,
        eta: float,
        dopamine: float,
    ) -> float:
        """
        时序权重的 Hebbian 更新: dw = eta * D * eps * z_prev - 0.01 * w.
        """
        A = active_nids.shape[0]
        if A == 0:
            return 0.0

        connected = self.pool.t_connected[active_nids.long()]
        if not connected.any():
            return 0.0

        nids_c = active_nids[connected]
        eps = self.pool.state[nids_c.long(), F_EPS]
        z_prev = self.pool.state[nids_c.long(), F_Z_PREV]

        dw = eta * dopamine * eps * z_prev
        self.pool.t_weight[nids_c.long()] += dw
        return float(dw.abs().sum().item())

    def hebbian_topdown(
        self,
        active_nids: torch.Tensor,
        eta: float,
        dopamine: float,
    ) -> float:
        """Topdown 权重的 Hebbian 更新 (向量化, 零逐元素 .item())."""
        if active_nids.shape[0] == 0:
            return 0.0

        td_alive = self.pool.td_alive
        if not td_alive.any():
            return 0.0

        td_indices = torch.where(td_alive)[0]
        post = self.pool.td_post[td_indices].long()
        pre = self.pool.td_pre[td_indices].long()

        active_mask = torch.isin(post, active_nids) & self.pool.alive[pre]
        if not active_mask.any():
            return 0.0

        idx = td_indices[active_mask]
        eps = self.pool.state[post[active_mask], F_EPS]
        z_pre_val = self.pool.state[pre[active_mask], F_Z]
        w = self.pool.td_weight[idx]
        dw = eta * dopamine * eps * z_pre_val
        self.pool.td_weight[idx] = w + dw
        return float(dw.abs().sum().item())

    def hebbian_lm_head(
        self,
        top_layer: int,
        target_byte: int,
        eta: float,
        dopamine: float,
        pred_byte: int = -1,
        use_mu: bool = False,
    ) -> float:
        """
        误差门控 Hebbian:
        +dw[target]=gate*z, -dw[pred]=gate*z.
        """
        top_mask = (self.pool.layer == top_layer) & self.pool.alive
        n_top = int(top_mask.sum().item())
        if n_top == 0:
            return 0.0

        state_col = F_MU if use_mu else F_Z
        z_top = self.pool.state[top_mask, state_col]
        z_use = z_top
        eta_eff = eta * dopamine

        if pred_byte < 0:
            logits = self.pool.lm_weight[:, top_mask] @ z_top
            pred_byte = int(logits.argmax().item())

        error = pred_byte != target_byte
        gate = 1.0 if error else 0.1

        # symmetric ±dw: 创造目标/预测列之间的竞争
        dw = torch.zeros(256, n_top, dtype=torch.float16, device=self.pool.device)
        dw[target_byte] = eta_eff * gate * z_use
        if error:
            dw[pred_byte] = -eta_eff * gate * z_use

        self.pool.lm_weight.data[:, top_mask] += dw

        self.pool.lm_bias.data[target_byte] = torch.clamp(self.pool.lm_bias.data[target_byte] + 1e-4, min=-1.0, max=1.0)
        if error:
            self.pool.lm_bias.data[pred_byte] = torch.clamp(self.pool.lm_bias.data[pred_byte] - 1e-4, min=-1.0, max=1.0)

        return 0.0

    def hebbian_lm_head_batched(
        self,
        top_layer: int,
        targets: torch.Tensor,
        eta: float,
        dopamine: float,
        use_mu: bool = False,
    ) -> float:
        """
        批量 LM head Hebbian
        零循环, 零 .item().
        """
        top_mask = (self.pool.layer == top_layer) & self.pool.alive
        n_top = int(top_mask.sum().item())
        if n_top == 0 or targets.shape[0] == 0:
            return 0.0

        state_col = F_MU if use_mu else F_Z
        z_top = self.pool.state[top_mask, state_col]
        eta_eff = eta * dopamine

        logits = self.pool.lm_weight[:, top_mask] @ z_top + self.pool.lm_bias
        pred = logits.argmax()
        n_e = (pred != targets).sum().item()

        dw = torch.zeros(256, n_top, dtype=torch.float16, device=self.pool.device)
        gate = torch.where(pred != targets, 1.0, 0.1)
        pos = (eta_eff * gate.unsqueeze(1) * z_top.unsqueeze(0)).to(torch.float16)
        dw.index_add_(0, targets, pos)
        if n_e > 0:
            neg = (-eta_eff * z_top.unsqueeze(0).expand(int(n_e), -1)).to(torch.float16)
            dw.index_add_(0, pred.expand(int(n_e)), neg)

        self.pool.lm_weight.data[:, top_mask] += dw
        self.pool.lm_bias.data[targets] += 1e-4
        return 0.0

    def adjust_thresholds(self, target_rate: float = 0.01, rate_eta: float = 0.01):
        """每步阈值调节 (稳态 homeostatic plasticity)."""
        alive = self.pool.alive
        if alive.any():
            delta = rate_eta * (self.pool.state[alive, F_FIRING_RATE] - target_rate)
            self.pool.state[alive, F_THRESHOLD] = torch.clamp(
                self.pool.state[alive, F_THRESHOLD] + delta, min=1e-4, max=0.5
            )

    def compute_precision_scales(self, D: float, ACh: float, eta: float = 1.0):
        """更新精度权重 pi = 1 + eta*D*|eps| + eta*ACh*|eps|."""
        alive = self.pool.alive
        if alive.any():
            e_abs = self.pool.state[alive, F_EPS].abs()
            self.pool.state[alive, F_PI] = 1.0 + eta * D * e_abs + eta * ACh * e_abs

    def finalize_step(self):
        """保存 z 到 z_prev (供下一步 temporal 预测)."""
        self.pool.state[:, F_Z_PREV] = self.pool.state[:, F_Z]

    def homeostasis_lm_head(self, top_layer: int):
        """
        LM head 稳态:
        逐列 L2 上限约束 (保留列间幅度差异, 只防爆炸).
        """
        top_mask = (self.pool.layer == top_layer) & self.pool.alive
        if not top_mask.any():
            return
        w_sub = self.pool.lm_weight[:, top_mask]  # [256, n_top]
        col_norm = w_sub.norm(dim=0)  # [n_top]
        over = col_norm > 8.0
        if over.any():
            scale = torch.where(over, 8.0 / (col_norm + 1e-6), 1.0)
            self.pool.lm_weight[:, top_mask] = w_sub * scale.unsqueeze(0)

    def homeostasis_step(
        self,
        current_step: int,
        target_rate: float = 0.01,
        rate_eta: float = 0.001,
        prune_interval: int = 100,
        grow_interval: int = 200,
        max_prune: int = 10,
        max_grow: int = 2,
        max_inactive: int = 1000,
    ) -> dict:
        """
        周期性维护:
        修剪 + 生长 (阈值调节已移至每步 adjust_thresholds).
        """
        stats: dict = {"threshold_adjusted": 0}

        alive = self.pool.alive

        # ── 内存预算压力调节 ──
        usage = self.pool._storage.usage_ratio()
        _prune_interval = prune_interval
        _max_prune = max_prune
        _grow_interval = grow_interval
        _max_grow = max_grow

        if usage > 0.7:
            _prune_interval = max(10, prune_interval // 2)
            _max_prune = max_prune * 2
            _grow_interval = grow_interval * 2
            _max_grow = max(0, max_grow - 1)
        if usage > 0.9:
            _max_grow = 0

        # 1. 修剪, 包含孤儿检查 + 不活跃检查
        if current_step % _prune_interval == 0:
            alive_ids = torch.where(alive)[0]
            age = current_step - self.pool.created_at[alive_ids]
            inactive_for = current_step - self.pool.last_active[alive_ids]
            in_counts = self.pool._in_counts[alive_ids.cpu()].to(self.pool.device)

            layer_of_alive = self.pool.layer[alive_ids]
            min_age = torch.where(layer_of_alive > 0, 500, 100)

            orphan_mask = (in_counts == 0) & (age > 500)
            inactive_mask = (
                (inactive_for > max_inactive) & (self.pool.state[alive_ids, F_FIRING_RATE] < 0.001) & (age > min_age)
            )
            prune_mask = orphan_mask | inactive_mask
            candidates = alive_ids[prune_mask][:_max_prune]
            for nid in candidates.tolist():
                self.pool.neuron.prune_neuron(int(nid))
            stats["pruned"] = len(candidates)
        else:
            stats["pruned"] = 0

        # 1.5 突触 turnover (每 500 步: 替换 2% 最弱突触)
        if current_step % 500 == 0:
            stats["turnover"] = self.pool.synapse.synapse_turnover(current_step)

        # 2. 生长 (分裂高误差神经元)
        if _max_grow > 0 and current_step % _grow_interval == 0:
            alive_ids = torch.where(alive)[0]
            high_err = self.pool.state[alive_ids, F_EPS].abs() > self.pool.state[alive_ids, F_THRESHOLD] * 3.0
            candidates = alive_ids[high_err][:_max_grow]
            grown = 0
            for nid in candidates.tolist():
                if self.pool.neuron.split_neuron(int(nid)) is not None:
                    grown += 1
            stats["grown"] = grown
        else:
            stats["grown"] = 0

        return stats
