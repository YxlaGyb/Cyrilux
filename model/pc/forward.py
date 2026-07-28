"""ForwardEngine — 前向传播: 预测、误差计算、logits、发射."""

from __future__ import annotations

import torch

from .constants import F_EPS, F_MU, F_THRESHOLD, F_Z


class ForwardEngine:
    """前向传播引擎:
    全网预测、层预测、误差、logits、兴奋发射.

    通过 self.pool 引用访问共享张量, 扩容后自动指向新张量.
    """

    def __init__(self, pool):
        self.pool = pool

        # k-WTA 查表 (CPU 侧, 避免每步 float() GPU sync)
        self._kwta_cpu = [0.0] * 16
        self._kwta_cpu[10] = 0.10
        self._kwta_cpu[11] = 0.10
        self._kwta_cpu[12] = 0.15
        self._kwta_cpu[13] = 0.20
        self._kwta_cpu[14] = 0.10

        # 持久 scatter_add 缓冲区 (懒分配, N 增长时自动重建)
        self._mu_buf = None
        self._mu_td_buf = None

        # k-WTA 缓存 (层 ID 在 warmup 后不变, 每 50 步刷新)
        self._cached_layer_list = None
        self._layer_refresh_counter = 0

        # 持久 k-WTA winners 缓冲 + 噪声模板缓存
        self._kwta_winners = None
        self._noise_cache = None
        self._noise_idx = 0
        self._NOISE_TEMPLATES = 8

    def _ensure_buffers(self):
        """
        确保持久缓冲区大小 ≥ pool.N (扩容后自动重建).
        """
        N = self.pool.N
        dev = self.pool.device
        if self._mu_buf is None or self._mu_buf.shape[0] < N:
            self._mu_buf = torch.zeros(N, dtype=torch.float16, device=dev)
        if self._mu_td_buf is None or self._mu_td_buf.shape[0] < N:
            self._mu_td_buf = torch.zeros(N, dtype=torch.float16, device=dev)
        if self._kwta_winners is None or self._kwta_winners.shape[0] < N:
            self._kwta_winners = torch.zeros(N, dtype=torch.bool, device=dev)
        if self._noise_cache is None or self._noise_cache.shape[0] < N:
            self._noise_cache = torch.stack([
                torch.randn(N, dtype=torch.float16, device=dev) * 0.005
                for _ in range(self._NOISE_TEMPLATES)
            ], dim=0)  # [8, N]

    def predict_all(self) -> None:
        """
        全网 scatter-add 预测:
        mu = 1/sqrt(K) * scatter_add(weight * z_pre, post).
        """
        alive_syn = self.pool.syn_alive
        if not alive_syn.any():
            return

        syn_mask = alive_syn
        pre = self.pool.pre_id[syn_mask]
        post = self.pool.post_id[syn_mask]
        w = self.pool.weight[syn_mask]
        z_pre = self.pool.state[pre.long(), F_Z]

        # z_pre RMSNorm (跳过感官层 L0 — 值固定为 ±1, 稀疏时会过度放大)
        # 非感官层 z_pre 归一化防止上游激活值爆炸
        non_sensory_mask = self.pool.layer[pre.long()] > 0
        if non_sensory_mask.any():
            z_sub = z_pre[non_sensory_mask]
            rms = (z_sub * z_sub).mean().sqrt()
            rms = rms + 1e-6
            z_pre = z_pre.clone()  # avoid in-place
            z_pre[non_sensory_mask] = z_sub / rms

        contrib = w * z_pre
        self._ensure_buffers()
        self._mu_buf.zero_()
        mu = self._mu_buf
        mu.scatter_add_(0, post.long(), contrib)

        # 1/sqrt(K) 缩放 — 使用缓存的扇入数
        alive = self.pool.alive
        if getattr(self.pool, '_fan_in_dirty', True):
            fan_in = (self.pool.in_ptrs[alive] >= 0).sum(dim=-1).to(torch.float16)
            self.pool._fan_in_cache[alive] = fan_in
            self.pool._fan_in_dirty = False
        scale = torch.rsqrt(self.pool._fan_in_cache[alive] + 1e-6)
        self.pool.state[alive, F_MU] = mu[alive] * scale
        self.pool.state[alive, F_EPS] = self.pool.state[alive, F_Z] - self.pool.state[alive, F_MU]

        # 非感官层神经元: z 向 mu 靠拢但不瞬间跟随 (膜电位时间常数)
        non_sensory = alive & (self.pool.layer > 0)
        if non_sensory.any():
            z_old = self.pool.state[non_sensory, F_Z]
            mu_ns = self.pool.state[non_sensory, F_MU]
            z_new = 0.3 * z_old + 0.7 * mu_ns  # 30%历史 + 70%当前输入

            # 噪声轮换: 免 randn kernel launch
            self._noise_idx = (self._noise_idx + 1) % self._NOISE_TEMPLATES
            z_new = z_new + self._noise_cache[self._noise_idx][non_sensory]

            # 逐层 k-WTA 侧抑制 — 持久 winners 缓冲
            layer_ids = self.pool.layer[non_sensory]
            self._layer_refresh_counter += 1
            if self._layer_refresh_counter % 50 == 1 or self._cached_layer_list is None:
                unique_layers = layer_ids.unique()
                self._cached_layer_list = unique_layers.tolist()

            for L in self._cached_layer_list:
                in_layer = layer_ids == L
                z_L = z_new[in_layer]
                n_L = z_L.shape[0]
                if L <= 0 or n_L <= 4:
                    continue
                k = max(4, int(n_L * self._kwta_cpu[L]))
                _, top_idx = torch.topk(z_L.abs(), k)
                self._kwta_winners[:n_L].zero_()
                self._kwta_winners[:n_L][top_idx] = True
                z_new[in_layer] = torch.where(self._kwta_winners[:n_L], z_L, z_L * 0.15)

            self.pool.state[non_sensory, F_Z] = z_new
            self.pool.state[non_sensory, F_EPS] = z_new - mu_ns

        # BCM 滑动阈值: 每个神经元追踪自己的近期平均活动量 (|z|)
        self.pool.state[:, 4] = (
            0.995 * self.pool.state[:, 4] + 0.005 * self.pool.state[:, F_Z].abs()
        )  # F_FIRING_RATE=4

    def predict_neurons(self, nids: torch.Tensor) -> torch.Tensor:
        """
        批量预测指定神经元的 mu.
        """
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
        """
        批量更新 z, 重算 mu 和 eps.
        """
        if nids.shape[0] == 0:
            return
        self.pool.state[nids.long(), F_Z] = z_new
        mu = self.predict_neurons(nids)
        self.pool.state[nids.long(), F_MU] = mu
        self.pool.state[nids.long(), F_EPS] = z_new - mu

    def temporal_topdown_pass(self, top_layer: int):
        """
        时序预测 + 自上而下预测, 更新 mu/epsilon.
        """
        alive = self.pool.alive

        # 时序预测: mu += t_weight * z_prev
        connected = self.pool.t_connected & alive
        if connected.any():
            mu_temp = self.pool.t_weight[connected] * self.pool.state[connected, 6]  # F_Z_PREV=6
            self.pool.state[connected, F_MU] += mu_temp
            self.pool.state[connected, F_EPS] = (
                self.pool.state[connected, F_Z] - self.pool.state[connected, F_MU]
            )

        # 自上而下预测: 只对感觉层 (layer==0) 做 scatter_add
        if top_layer <= 0:
            return

        td_alive = self.pool.td_alive
        if not td_alive.any():
            return

        # 只取 topdown 到 L0 的连接 (td_post 指向感觉层)
        td_mask = td_alive
        td_post_all = self.pool.td_post[td_mask].long()
        td_to_sensory = self.pool.layer[td_post_all] == 0
        if not td_to_sensory.any():
            return

        # 编译缩减: 只对感光层的连接做 scatter_add
        td_idx_all = torch.where(td_mask)[0]
        td_idx = td_idx_all[td_to_sensory]
        td_pre = self.pool.td_pre[td_idx].long()
        td_post = self.pool.td_post[td_idx].long()
        td_w = self.pool.td_weight[td_idx]
        z_upper = self.pool.state[td_pre, F_Z]

        contrib = td_w * z_upper
        self._ensure_buffers()
        self._mu_td_buf.zero_()
        mu_td = self._mu_td_buf
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

    def compute_lm_logits(self, top_layer: int, use_mu: bool = True) -> torch.Tensor:
        """计算 256 个字节 logits. logits = lm_weight @ z[top_layer_mask]."""
        top_mask = (self.pool.layer == top_layer) & self.pool.alive
        if not top_mask.any():
            return torch.zeros(256, dtype=torch.float16, device=self.pool.device)

        state_col = F_MU if use_mu else F_Z
        z_top = self.pool.state[top_mask, state_col]
        w_sub = self.pool.lm_weight[:, top_mask]
        logits = w_sub @ z_top + self.pool.lm_bias
        return logits

    def compute_cross_entropy(self, logits: torch.Tensor, target_byte: int) -> torch.Tensor:
        """计算交叉熵损失."""
        return torch.nn.functional.cross_entropy(
            logits.unsqueeze(0).float(),
            torch.tensor([target_byte], device=self.pool.device),
        )

    def emit_active(self, current_time: int) -> torch.Tensor:
        """
        扫描全网络, 返回活跃神经元索引. firing: |eps| > threshold.
        """
        firing = (
            self.pool.state[:, F_EPS].abs() > self.pool.state[:, F_THRESHOLD]
        ) & self.pool.alive
        if firing.any():
            self.pool.last_active[firing] = current_time
        return torch.where(firing)[0]
