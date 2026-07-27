"""ForwardEngine — 前向传播: 预测、误差计算、logits、发射."""

from __future__ import annotations

import torch

from .constants import F_EPS, F_MU, F_THRESHOLD, F_Z


class ForwardEngine:
    """前向传播引擎: 全网预测、层预测、误差、logits、兴奋发射.

    通过 self.pool 引用访问共享张量, 扩容后自动指向新张量.
    """

    def __init__(self, pool):
        self.pool = pool

    def predict_all(self) -> None:
        """全网 scatter-add 预测: mu = 1/sqrt(K) * scatter_add(weight * z_pre, post)."""
        alive_syn = self.pool.syn_alive
        if not alive_syn.any():
            return

        syn_mask = alive_syn
        pre = self.pool.pre_id[syn_mask]
        post = self.pool.post_id[syn_mask]
        w = self.pool.weight[syn_mask]
        z_pre = self.pool.state[pre.long(), F_Z]

        # z_pre RMSNorm (投影前归一化, 防止上游激活值爆炸)
        rms = (z_pre * z_pre).mean().sqrt()
        rms = rms + 1e-6
        z_pre = z_pre / rms

        contrib = w * z_pre
        mu = torch.zeros(self.pool.N, dtype=torch.float16, device=self.pool.device)
        mu.scatter_add_(0, post.long(), contrib)

        # 1/sqrt(K) 缩放 (K = 每神经元入连接数)
        alive = self.pool.alive
        fan_in = (self.pool.in_ptrs[alive] >= 0).sum(dim=-1, keepdim=True).to(torch.float16)
        scale = torch.rsqrt(fan_in + 1e-6)
        self.pool.state[alive, F_MU] = mu[alive] * scale.squeeze(-1)
        self.pool.state[alive, F_EPS] = self.pool.state[alive, F_Z] - self.pool.state[alive, F_MU]

        # 非感官层神经元: z 向 mu 靠拢但不瞬间跟随 (膜电位时间常数)
        non_sensory = alive & (self.pool.layer > 0)
        if non_sensory.any():
            z_old = self.pool.state[non_sensory, F_Z]
            mu_ns = self.pool.state[non_sensory, F_MU]
            z_new = 0.3 * z_old + 0.7 * mu_ns  # 30%历史 + 70%当前输入

            # 微小噪声打破对称性
            noise = torch.randn_like(z_new) * 0.005
            z_new = z_new + noise

            # 逐层 k-WTA 侧抑制
            for L in self.pool.layer[non_sensory].unique().tolist():
                if L <= 0:
                    continue
                in_layer = self.pool.layer[non_sensory] == L
                n_L = in_layer.sum().item()
                if n_L <= 4:
                    continue
                z_L = z_new[in_layer]
                k = max(4, n_L // 10)
                _, top_idx = torch.topk(z_L.abs(), k)
                winners = torch.zeros(n_L, dtype=torch.bool, device=self.pool.device)
                winners[top_idx] = True
                z_new[in_layer] = torch.where(winners, z_L, z_L * 0.1)

            self.pool.state[non_sensory, F_Z] = z_new
            self.pool.state[non_sensory, F_EPS] = z_new - mu_ns

        # BCM 滑动阈值: 每个神经元追踪自己的近期平均活动量 (|z|)
        self.pool.state[:, 4] = (
            0.995 * self.pool.state[:, 4] + 0.005 * self.pool.state[:, F_Z].abs()
        )  # F_FIRING_RATE=4

    def predict_neurons(self, nids: torch.Tensor) -> torch.Tensor:
        """批量预测指定神经元的 mu."""
        E = nids.shape[0]
        if E == 0:
            return torch.zeros(0, dtype=torch.float16, device=self.pool.device)

        in_s = self.pool.in_ptrs[nids.long()]  # [E, K]
        valid = (in_s >= 0) & self.pool.syn_alive[in_s.long()]
        in_s = in_s.long()
        pre_ids = self.pool.pre_id[in_s]  # [E, K]
        valid &= pre_ids >= 0
        w = self.pool.weight[in_s]  # [E, K]
        z_pre = torch.where(valid, self.pool.state[pre_ids.long(), F_Z], torch.zeros_like(w))
        w_masked = torch.where(valid, w, torch.zeros_like(w))

        mu = (w_masked * z_pre).sum(dim=-1)  # [E]
        return mu

    def update_batch(self, nids: torch.Tensor, z_new: torch.Tensor) -> None:
        """批量更新 z, 重算 mu 和 eps."""
        if nids.shape[0] == 0:
            return
        self.pool.state[nids.long(), F_Z] = z_new
        mu = self.predict_neurons(nids)
        self.pool.state[nids.long(), F_MU] = mu
        self.pool.state[nids.long(), F_EPS] = z_new - mu

    def temporal_topdown_pass(self, top_layer: int):
        """时序预测 + 自上而下预测, 更新 mu/epsilon."""
        alive = self.pool.alive

        # 时序预测: mu += t_weight * z_prev
        connected = self.pool.t_connected & alive
        if connected.any():
            mu_temp = self.pool.t_weight[connected] * self.pool.state[connected, 6]  # F_Z_PREV=6
            self.pool.state[connected, F_MU] += mu_temp
            self.pool.state[connected, F_EPS] = (
                self.pool.state[connected, F_Z] - self.pool.state[connected, F_MU]
            )

        # 自上而下预测: 对感觉层 (layer==0) 做 scatter_add
        if top_layer <= 0:
            return

        td_alive = self.pool.td_alive
        if not td_alive.any():
            return

        td_mask = td_alive
        td_pre = self.pool.td_pre[td_mask].long()
        td_post = self.pool.td_post[td_mask].long()
        td_w = self.pool.td_weight[td_mask]
        z_upper = self.pool.state[td_pre, F_Z]

        contrib = td_w * z_upper
        mu_td = torch.zeros(self.pool.N, dtype=torch.float16, device=self.pool.device)
        mu_td.scatter_add_(0, td_post, contrib)

        sensory = (self.pool.layer == 0) & alive
        if sensory.any():
            self.pool.state[sensory, F_MU] += mu_td[sensory]
            self.pool.state[sensory, F_EPS] = (
                self.pool.state[sensory, F_Z] - self.pool.state[sensory, F_MU]
            )

    def compute_free_energy(self) -> torch.Tensor:
        """F = sum(epsilon^2) over alive neurons."""
        return (self.pool.state[self.pool.alive, F_EPS] ** 2).sum()

    def compute_lm_logits(self, top_layer: int, use_mu: bool = False) -> torch.Tensor:
        """计算 256 个字节 logits. logits = lm_weight @ z[top_layer_mask]."""
        top_mask = (self.pool.layer == top_layer) & self.pool.alive
        n_top = int(top_mask.sum().item())
        if n_top == 0:
            return torch.zeros(256, dtype=torch.float16, device=self.pool.device)

        state_col = F_MU if use_mu else F_Z
        z_top = self.pool.state[top_mask, state_col]  # [n_top]
        w_sub = self.pool.lm_weight[:, top_mask]  # [256, n_top]
        logits = w_sub @ z_top  # [256]
        return logits

    def compute_cross_entropy(self, logits: torch.Tensor, target_byte: int) -> torch.Tensor:
        """计算交叉熵损失."""
        return torch.nn.functional.cross_entropy(
            logits.unsqueeze(0).float(),
            torch.tensor([target_byte], device=self.pool.device),
        )

    def emit_active(self, current_time: int) -> torch.Tensor:
        """扫描全网络, 返回活跃神经元索引. firing: |eps| > threshold."""
        firing = (
            self.pool.state[:, F_EPS].abs() > self.pool.state[:, F_THRESHOLD]
        ) & self.pool.alive
        if firing.any():
            self.pool.last_active[firing] = current_time
        return torch.where(firing)[0]
