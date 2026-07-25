"""神经发生控制器 — 结构自组织 (Phase C).

基于 SalienceGate 的活性追踪 + 剪枝 + 生长, 实现动态网络结构调整。
所有操作不物理删除权重, 只通过 gate logits 控制通道活性。

核心流程:
  ActivationTracker.update(z, ε)     # 每步更新 EMA
  ChannelPruner.prune(model, gates)  # 每 N 步关闭低活性通道
  ChannelGrowth.grow(model, gates, tracker, ε)  # 每 N 步复活/生长

决策记录:
  - 不物理删除权重, 只关 gate (保留参数以便复活)
  - 分裂时复制活跃通道权重乘以 0.5 + 噪声到 dormant 通道
  - 生长间隔 > 剪枝间隔, 防止振荡
"""

from typing import List, Tuple

import torch
from torch import nn


class ActivationTracker:
    """跨层通道激活 EMA 追踪器。

    从 SalienceGate 的 activation_ema buffer 收集每层活性数据,
    提供统一接口给 ChannelPruner / ChannelGrowth。
    """

    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        # per-gate-index 缓存, 避免每步重新收集
        self._cached_ema: List[torch.Tensor] = []
        self._cached_gate_values: List[torch.Tensor] = []

    @torch.no_grad()
    def update(self, gates: nn.ModuleList):
        """从 salience_gates ModuleList 收集当前 EMA 和 gate 值。"""
        self._cached_ema = [g.activation_ema.clone() for g in gates]
        self._cached_gate_values = [g.get_gate_values() for g in gates]

    @torch.no_grad()
    def get_dead_channels(
        self,
        gates: nn.ModuleList,
        threshold_act: float = 0.001,
        threshold_gate: float = 0.05,
    ) -> List[Tuple[int, int]]:
        """返回 (gate_idx, channel_idx) 对, 标识需要剪枝的通道。

        Args:
            gates: salience_gates ModuleList
            threshold_act: 激活 EMA 低于此值视为死亡
            threshold_gate: gate 值低于此值视为关闭

        Returns:
            dead_list: [(gate_idx, channel_idx), ...]
        """
        dead = []
        for g_idx, gate in enumerate(gates):
            ema = (
                self._cached_ema[g_idx]
                if g_idx < len(self._cached_ema)
                else gate.activation_ema
            )
            gv = (
                self._cached_gate_values[g_idx]
                if g_idx < len(self._cached_gate_values)
                else gate.get_gate_values()
            )
            dead_mask = (ema < threshold_act) & (gv < threshold_gate)
            for c_idx in dead_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
                dead.append((g_idx, c_idx))
        return dead

    @torch.no_grad()
    def get_active_channels(
        self,
        gates: nn.ModuleList,
        threshold_gate: float = 0.5,
    ) -> List[Tuple[int, int, float]]:
        """返回 (gate_idx, channel_idx, activation_ema) 列表, 按 activation_ema 降序。
        """
        active = []
        for g_idx, gate in enumerate(gates):
            gv = (
                self._cached_gate_values[g_idx]
                if g_idx < len(self._cached_gate_values)
                else gate.get_gate_values()
            )
            ema = (
                self._cached_ema[g_idx]
                if g_idx < len(self._cached_ema)
                else gate.activation_ema
            )
            active_mask = gv > threshold_gate
            for c_idx in active_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
                active.append((g_idx, c_idx, ema[c_idx].item()))
        active.sort(key=lambda x: x[2], reverse=True)
        return active

    @torch.no_grad()
    def get_dormant_channels(
        self,
        gates: nn.ModuleList,
        threshold_gate: float = 0.05,
    ) -> List[Tuple[int, int]]:
        """返回 (gate_idx, channel_idx) 对, gate 已关闭的通道。"""
        dormant = []
        for g_idx, gate in enumerate(gates):
            gv = (
                self._cached_gate_values[g_idx]
                if g_idx < len(self._cached_gate_values)
                else gate.get_gate_values()
            )
            dormant_mask = gv < threshold_gate
            for c_idx in dormant_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
                dormant.append((g_idx, c_idx))
        return dormant


class ChannelPruner:
    """低活性通道剪枝器 — 关闭 (非删除) 低 salience 通道。

    策略: 若通道激活 EMA < threshold_act 且 gate 值 < threshold_gate,
    则视为已死亡, 将 logits 设为 -10 (σ(-10) ≈ 0)。
    """

    def __init__(self, threshold_act: float = 0.001, threshold_gate: float = 0.05):
        self.threshold_act = threshold_act
        self.threshold_gate = threshold_gate

    @torch.no_grad()
    def prune(self, gates: nn.ModuleList, tracker: ActivationTracker) -> int:
        """执行剪枝, 返回本次剪枝的通道数。"""
        dead = tracker.get_dead_channels(gates, self.threshold_act, self.threshold_gate)
        if not dead:
            return 0
        for g_idx, c_idx in dead:
            gates[g_idx].logits.data[c_idx] = -10.0
        return len(dead)


class ChannelGrowth:
    """通道生长器 — 基于预测误差复活 dormant 通道或分裂活跃通道。

    策略:
      1. 检查每层预测误差
      2. 若有 dormant 通道 → 复活 (随机化权重 + 打开 gate)
      3. 若误差极高且无 dormant → 分裂最活跃通道到 dormant 通道
    """

    def __init__(
        self,
        error_threshold_high: float = 2.0,
        noise_scale: float = 0.1,
    ):
        self.error_threshold_high = error_threshold_high
        self.noise_scale = noise_scale

    @torch.no_grad()
    def grow(
        self,
        model: nn.Module,
        gates: nn.ModuleList,
        tracker: ActivationTracker,
        ε_list: List[torch.Tensor],
        max_grow: int = 8,
    ) -> Tuple[int, int]:
        """执行生长, 返回 (n_resurrected, n_split)。

        Args:
            model: StreamRunner 实例
            gates: salience_gates ModuleList
            tracker: ActivationTracker
            ε_list: 每层预测误差列表 [ε₁, ε₂, ..., ε_L], 每个 [B,S,H]
            max_grow: 单步最大生长通道数

        Returns:
            (n_resurrected, n_split)
        """
        n_resurrected = 0
        n_split = 0

        # 误差按层聚合
        L = len(ε_list)
        layer_errors = []
        for ℓ in range(L):
            err = ε_list[ℓ].abs().mean().item()
            layer_errors.append(err)

        # 找出误差最高的层
        high_error_layers = [
            ℓ for ℓ, err in enumerate(layer_errors) if err > self.error_threshold_high
        ]

        for ℓ in high_error_layers:
            if n_resurrected + n_split >= max_grow:
                break

            # 该层对应的两个 gate 索引
            g_conv = 2 * ℓ  # conv gate
            g_mlp = 2 * ℓ + 1  # MLP gate

            # 尝试从两个 gate 中找 dormant 通道
            for g_idx in [g_conv, g_mlp]:
                if g_idx >= len(gates):
                    continue
                if n_resurrected + n_split >= max_grow:
                    break

                dormant = tracker.get_dormant_channels(gates)
                dormant_here = [(gi, ci) for gi, ci in dormant if gi == g_idx]

                if dormant_here:
                    # 复活: 随机化该通道权重 + 打开 gate
                    self._resurrect_channel(model, gates, g_idx, dormant_here[0][1])
                    n_resurrected += 1
                else:
                    # 无 dormant → 分裂: 复制最活跃通道到最不活跃
                    active = tracker.get_active_channels(gates)
                    active_here = [
                        (gi, ci, ema) for gi, ci, ema in active if gi == g_idx
                    ]
                    if len(active_here) >= 2:
                        # 找 gate 最低的通道 (可能只是略高于阈值)
                        all_gv = gates[g_idx].get_gate_values()
                        min_c = all_gv.argmin().item()
                        max_c = active_here[0][1]  # 最活跃通道
                        if min_c != max_c:
                            self._split_channel(model, gates, g_idx, max_c, min_c)
                            n_split += 1

        return n_resurrected, n_split

    @torch.no_grad()
    def _resurrect_channel(
        self,
        model: nn.Module,
        gates: nn.ModuleList,
        g_idx: int,
        c_idx: int,
    ):
        """复活指定通道: 随机化其权重 + 打开 gate。"""
        block_idx = g_idx // 2
        is_conv = g_idx % 2 == 0

        # 随机化权重
        if is_conv and block_idx < len(model.model.layers):
            # conv weight [H, H, 3]: 输出通道 c 是 [c, :, :]
            conv_w = model.model.layers[block_idx].local_conv.weight
            fan_in = conv_w.size(1) * conv_w.size(2)
            std = (2.0 / fan_in) ** 0.5
            conv_w.data[c_idx] = torch.randn_like(conv_w[c_idx]) * std
        elif not is_conv and block_idx < len(model.model.layers):
            # MLP down_proj [H, inter]: 输出通道 c 是 [c, :]
            mlp = model.model.layers[block_idx].mlp
            for proj in [mlp.down_proj]:
                std = (2.0 / proj.weight.size(1)) ** 0.5
                proj.weight.data[c_idx] = torch.randn_like(proj.weight[c_idx]) * std

        # 打开 gate: logit = 0 (σ(0) = 0.5, 中性)
        gates[g_idx].logits.data[c_idx] = 0.0
        # 重置 activation EMA
        gates[g_idx].activation_ema[c_idx] = 0.0

    @torch.no_grad()
    def _split_channel(
        self,
        model: nn.Module,
        gates: nn.ModuleList,
        g_idx: int,
        src_c: int,
        dst_c: int,
    ):
        """分裂: 复制源通道权重 * 0.5 + noise 到目标通道。"""
        block_idx = g_idx // 2
        is_conv = g_idx % 2 == 0

        if is_conv and block_idx < len(model.model.layers):
            conv_w = model.model.layers[block_idx].local_conv.weight
            noise = torch.randn_like(conv_w[src_c]) * self.noise_scale
            conv_w.data[dst_c] = conv_w[src_c] * 0.5 + noise
        elif not is_conv and block_idx < len(model.model.layers):
            mlp = model.model.layers[block_idx].mlp
            for proj in [mlp.gate_proj, mlp.up_proj, mlp.down_proj]:
                if proj.weight.dim() == 2:
                    # down_proj [H, inter]: 按输出通道复制
                    noise = torch.randn_like(proj.weight[src_c]) * self.noise_scale
                    proj.weight.data[dst_c] = proj.weight[src_c] * 0.5 + noise

        # 打开 gate
        gates[g_idx].logits.data[dst_c] = 3.0
        gates[g_idx].activation_ema[dst_c] = 0.0


class NeurogenesisController:
    """神经发生控制器统一入口。

    管理 ActivationTracker + ChannelPruner + ChannelGrowth,
    按指定间隔执行剪枝和生长。
    """

    def __init__(
        self,
        hidden_size: int,
        prune_interval: int = 100,
        grow_interval: int = 300,
        prune_threshold_act: float = 0.001,
        prune_threshold_gate: float = 0.05,
        grow_error_threshold: float = 2.0,
        max_grow_per_step: int = 8,
    ):
        self.tracker = ActivationTracker(hidden_size)
        self.pruner = ChannelPruner(prune_threshold_act, prune_threshold_gate)
        self.grower = ChannelGrowth(grow_error_threshold)
        self.prune_interval = prune_interval
        self.grow_interval = grow_interval
        self.max_grow_per_step = max_grow_per_step

    @torch.no_grad()
    def step(
        self,
        model: nn.Module,
        ε_list: List[torch.Tensor],
        global_step: int,
    ) -> dict:
        """每一步调用的入口。

        Args:
            model: StreamRunner 实例 (需有 salience_gates 属性)
            ε_list: 每层预测误差
            global_step: 当前训练步

        Returns:
            dict: {n_pruned, n_resurrected, n_split, active_ratio}
        """
        gates = getattr(model, "salience_gates", None)
        if gates is None or len(gates) == 0:
            return {
                "n_pruned": 0,
                "n_resurrected": 0,
                "n_split": 0,
                "active_ratio": 1.0,
            }

        # 更新追踪器
        self.tracker.update(gates)

        n_pruned = 0
        n_resurrected = 0
        n_split = 0

        # 剪枝 (每 prune_interval 步)
        if global_step % self.prune_interval == 0 and global_step > 0:
            n_pruned = self.pruner.prune(gates, self.tracker)

        # 生长 (每 grow_interval 步)
        if global_step % self.grow_interval == 0 and global_step > 0 and ε_list:
            n_resurrected, n_split = self.grower.grow(
                model,
                gates,
                self.tracker,
                ε_list,
                max_grow=self.max_grow_per_step,
            )

        # 活性统计
        active_ratios = [g.get_active_ratio() for g in gates]
        mean_active = sum(active_ratios) / len(active_ratios) if active_ratios else 1.0

        return {
            "n_pruned": n_pruned,
            "n_resurrected": n_resurrected,
            "n_split": n_split,
            "active_ratio": mean_active,
        }
