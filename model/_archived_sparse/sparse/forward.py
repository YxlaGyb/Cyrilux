"""
ForwardEngine

前向传播:
    预测、误差计算、logits、发射.
"""

from __future__ import annotations

import torch

from model.constants import F_EPS, F_FIRING_RATE, F_MU, F_THRESHOLD, F_Z


class ForwardEngine:
    def __init__(self, pool):
        self.pool = pool
        self._mu_buf: torch.Tensor | None = None
        self._mu_td_buf: torch.Tensor | None = None
        self._noise_cache = None
        self._noise_idx = 0
        self._NOISE_TEMPLATES = 8

    def _ensure_buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        N = self.pool.N
        dev = self.pool.device
        if self._mu_buf is None or self._mu_buf.shape[0] < N:
            self._mu_buf = torch.zeros(N, dtype=torch.float16, device=dev)
        if self._mu_td_buf is None or self._mu_td_buf.shape[0] < N:
            self._mu_td_buf = torch.zeros(N, dtype=torch.float16, device=dev)
        if self._noise_cache is None or self._noise_cache.shape[0] < N:
            self._noise_cache = torch.stack(
                [torch.randn(N, dtype=torch.float16, device=dev) * 0.005 for _ in range(self._NOISE_TEMPLATES)], dim=0
            )
        return self._mu_buf, self._mu_td_buf

    def predict_all(self) -> None:
        alive_syn = self.pool.syn_alive
        if not alive_syn.any():
            return
        syn_mask = alive_syn
        pre = self.pool.pre_id[syn_mask]
        post = self.pool.post_id[syn_mask]
        w = self.pool.weight[syn_mask]
        z_pre = self.pool.state[pre.long(), F_Z]
        self._ensure_buffers()
        mu_buf, _ = self._ensure_buffers()
        mu_buf.zero_()
        mu_buf.scatter_add_(0, post.long(), w * z_pre)
        alive = self.pool.alive
        if getattr(self.pool, "_fan_in_dirty", True):
            fan_in = (self.pool.in_ptrs[alive] >= 0).sum(dim=-1).to(torch.float16)
            self.pool._fan_in_cache[alive] = fan_in
            self.pool._fan_in_dirty = False
        scale = torch.rsqrt(self.pool._fan_in_cache[alive] + 1e-6)
        self.pool.state[alive, F_MU] = mu_buf[alive] * scale
        self.pool.state[alive, F_EPS] = self.pool.state[alive, F_Z] - self.pool.state[alive, F_MU]
        non_sensory = alive & (self.pool.layer > 0)
        if non_sensory.any():
            z_new = 0.3 * self.pool.state[non_sensory, F_Z] + 0.7 * self.pool.state[non_sensory, F_MU]
            self._noise_idx = (self._noise_idx + 1) % self._NOISE_TEMPLATES
            z_new = z_new + self._noise_cache[self._noise_idx][non_sensory]
            self.pool.state[non_sensory, F_Z] = z_new
            self.pool.state[non_sensory, F_EPS] = z_new - self.pool.state[non_sensory, F_MU]
        self.pool.state[:, F_FIRING_RATE] = (
            0.995 * self.pool.state[:, F_FIRING_RATE] + 0.005 * self.pool.state[:, F_Z].abs()
        )

    def temporal_topdown_pass(self, top_layer: int):
        alive = self.pool.alive
        connected = self.pool.t_connected & alive
        if connected.any():
            self.pool.state[connected, F_MU] += self.pool.t_weight[connected] * self.pool.state[connected, 6]
            self.pool.state[connected, F_EPS] = self.pool.state[connected, F_Z] - self.pool.state[connected, F_MU]
        if top_layer <= 0:
            return
        td_alive = self.pool.td_alive
        if not td_alive.any():
            return
        td_post_all = self.pool.td_post[td_alive].long()
        td_to_sensory = self.pool.layer[td_post_all] == 0
        if not td_to_sensory.any():
            return
        td_idx_all = torch.where(td_alive)[0]
        td_idx = td_idx_all[td_to_sensory]
        td_pre = self.pool.td_pre[td_idx].long()
        td_post = self.pool.td_post[td_idx].long()
        td_w = self.pool.td_weight[td_idx]
        z_upper = self.pool.state[td_pre, F_Z]
        self._ensure_buffers()
        _, mu_td_buf = self._ensure_buffers()
        mu_td_buf.zero_()
        mu_td_buf.scatter_add_(0, td_post, td_w * z_upper)
        sensory = (self.pool.layer == 0) & alive
        if sensory.any():
            self.pool.state[sensory, F_MU] += mu_td_buf[sensory]
            self.pool.state[sensory, F_EPS] = self.pool.state[sensory, F_Z] - self.pool.state[sensory, F_MU]

    def predict_neurons(self, nids):
        if nids.shape[0] == 0:
            return torch.zeros(0, dtype=torch.float16, device=self.pool.device)
        in_s = self.pool.in_ptrs[nids.long()]
        valid = (in_s >= 0) & self.pool.syn_alive[in_s.long()]
        in_s = in_s.long()
        pre_ids = self.pool.pre_id[in_s]
        valid &= pre_ids >= 0
        w = self.pool.weight[in_s]
        z_pre = torch.where(valid, self.pool.state[pre_ids.long(), F_Z], torch.zeros_like(w))
        return (torch.where(valid, w, torch.zeros_like(w)) * z_pre).sum(dim=-1)

    def update_batch(self, nids, z_new):
        if nids.shape[0] == 0:
            return
        self.pool.state[nids.long(), F_Z] = z_new
        mu = self.predict_neurons(nids)
        self.pool.state[nids.long(), F_MU] = mu
        self.pool.state[nids.long(), F_EPS] = z_new - mu

    def compute_free_energy(self):
        return (self.pool.state[self.pool.alive, F_EPS] ** 2).sum()

    def compute_lm_logits(self, top_layer, use_mu=True):
        top_mask = (self.pool.layer == top_layer) & self.pool.alive
        if not top_mask.any():
            return torch.zeros(256, dtype=torch.float16, device=self.pool.device)
        z_top = self.pool.state[top_mask, F_MU if use_mu else F_Z]
        return self.pool.lm_weight[:, top_mask] @ z_top + self.pool.lm_bias

    @staticmethod
    def sample_topk_uniform(logits, k=8, temperature=1.0):
        """纯侧抑制采样: top-k 公平抽签."""
        probs = torch.softmax((logits.float() / temperature), dim=-1)
        vals, idxs = probs.topk(k)
        chosen = torch.randint(0, k, (1,)).item()
        return int(idxs[chosen].item())

    def compute_cross_entropy(self, logits, target_byte):
        return torch.nn.functional.cross_entropy(
            logits.unsqueeze(0).float(), torch.tensor([target_byte], device=self.pool.device)
        )

    def emit_active(self, current_time):
        firing = (self.pool.state[:, F_EPS].abs() > self.pool.state[:, F_THRESHOLD]) & self.pool.alive
        if firing.any():
            self.pool.last_active[firing] = current_time
        return torch.where(firing)[0]
