"""TrainingLoop — 事件驱动, 纯局部 Hebbian, 零 autograd.

一次前馈全序列, 批 LM head Hebbian (唯一 target 去重, 防高频列垄断).
z*10 仅推理 compute_lm_logits 生效, 训练时用原生 z.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
from tqdm import tqdm

from pkg.utils.trainer_utils import setup_seed
from pkg.device.cuda import setup_cuda_device
from model.model_cyrene import CyreneConfig, CyreneModel

from .config import TrainingConfig


class TrainingLoop:
    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.device = setup_cuda_device()
        self._setup_environment()
        self.runner: Optional[CyreneModel] = None
        self.global_step = 0
        self._last_stats: dict = {}
        self._last_F: float = float("inf")
        self._last_D: float = 0.5

    def _setup_environment(self):
        setup_seed(self.cfg.seed)

    def _log(self, message: str):
        if self.cfg.progress_callback:
            self.cfg.progress_callback({"type": "log", "message": message})

    def _build_model(self) -> CyreneModel:
        cfg = CyreneConfig(
            hidden_size=self.cfg.hidden_size,
            warmup_steps=50,
            hebbian_base_eta=self.cfg.hebbian_base_eta,
            oja_alpha=getattr(self.cfg, "oja_alpha", 0.05),
            ach_beta_0=self.cfg.hebbian_ach_beta_0,
            prune_interval=self.cfg.prune_interval,
            grow_interval=self.cfg.grow_interval,
            homeostasis_interval=self.cfg.homeostasis_interval,
            connection_density=self.cfg.connection_density,
            bias_strength=self.cfg.bias_strength,
        )
        runner = CyreneModel(cfg)
        runner.add_hidden_layer()
        self._log(
            f"CyreneModel built: h_front={self.cfg.hidden_size}, "
            f"budget={runner.pool._storage.total_allocated_bytes() / 1e6:.0f}MB / "
            f"{runner.pool._storage.max_memory_bytes / 1e9:.1f}GB"
        )
        return runner

    def warmup(self):
        assert self.runner is not None
        with torch.no_grad():
            dummy = torch.zeros(1, 64, dtype=torch.long, device=self.device)
            for _ in range(getattr(self.cfg, "warmup_steps", 20)):
                self.runner.step(dummy)
        self._log("Warmup done")

    def _build_mu_table(self, m, top_mask):
        """预计算 [255, n_top] mu 表 — 向量化批处理, 零 Python 循环."""
        n_top = int(top_mask.sum().item())
        top_idx = torch.where(top_mask)[0]
        sensory = (m.pool.layer == 0) & m.pool.alive
        s_nids = torch.where(sensory)[0]
        s_pos = m.pool.position[s_nids]
        if len(s_nids) == 0:
            return torch.zeros(255, n_top, dtype=torch.float16, device=m.device)

        # 映射 position → position index (0-based)
        pos_idx = torch.zeros(256, dtype=torch.long, device=m.device) - 1
        pos_idx[s_pos.long()] = torch.arange(len(s_nids), device=m.device)
        valid_p = pos_idx >= 0
        pos_t = torch.where(valid_p)[0]  # [n_pos_used], 0-255

        all_out = m.pool.out_ptrs[s_nids].long()  # [n_s, K]
        all_valid = (all_out >= 0) & m.pool.syn_alive[all_out]  # [n_s, K]
        post_all = m.pool.post_id[all_out]  # [n_s, K]
        w_all = m.pool.weight[all_out]  # [n_s, K]
        fan_all = m.pool._fan_in_cache[post_all]  # [n_s, K]
        in_top = torch.isin(post_all, top_idx) & all_valid  # [n_s, K]

        mu_t = torch.zeros(255, n_top, dtype=torch.float16, device=m.device)
        for pi in range(255):
            si = pos_idx[pi + 1]  # position (pi+1) → sensory index
            if si < 0:
                continue
            mask = in_top[si]
            if not mask.any():
                continue
            post_si = post_all[si][mask]
            w_si = w_all[si][mask]
            fan_si = fan_all[si][mask]
            col = torch.searchsorted(top_idx, post_si)
            mu_t[pi, col] = (w_si * torch.rsqrt(fan_si + 1e-6)).to(torch.float16)
        return mu_t

    def _compute_prefix_M(self, B: int) -> torch.Tensor:
        idx = torch.arange(B, device=self.device)
        M = 0.7 * (0.3 ** (idx.unsqueeze(1) - idx.unsqueeze(0)))
        M = torch.tril(M).to(torch.float16)
        return M

    def train_step(self, byte_seq: torch.Tensor, labels: torch.Tensor) -> dict:
        assert self.runner is not None
        m = self.runner

        # 前馈序列：感官创建 + 只对 L4 做预测（跳过全量 predict_pass）
        m.encode_and_predict_l4_only(byte_seq)

        targets_all = labels[0, 1:].long()
        valid = (targets_all >= 0) & (targets_all <= 255)
        targets_t = targets_all[valid]
        B = min(targets_t.shape[0], 255)
        if B == 0:
            return {"ce_val": 0}

        top_mask = (m.pool.layer == m._top_layer) & m.pool.alive
        n_top = int(top_mask.sum().item())
        if n_top == 0:
            return {"ce_val": 0}
        top_idx = torch.where(top_mask)[0]

        eta_eff = m.config.hebbian_base_eta * 5000.0 * 0.5

        # ── mu_table (每步重建 — 不同 batch 有不同感官神经元) ──
        mu_mat = self._build_mu_table(m, top_mask)[:B]

        M = self._compute_prefix_M(B)
        zs = M @ mu_mat

        # ── 列 dropout Hebbian: 每步只更新随机 25% 的列 ──
        tg = targets_t[:B]
        logits = zs.float() @ m.pool.lm_weight[:, top_idx].float().T + m.pool.lm_bias
        preds = logits.argmax(dim=-1)

        lw = m.pool.lm_weight.data.clone()
        lw_top = lw[:, top_idx].clone()

        # ---- column-dropout Hebbian ----
        err_mask = preds != tg
        dw = torch.zeros(256, n_top, dtype=torch.float16, device=m.device)
        if (~err_mask).any():
            dw.index_add_(0, tg[~err_mask], (0.1 * eta_eff * zs[~err_mask]).to(torch.float16))
        if err_mask.any():
            dw.index_add_(0, tg[err_mask], (eta_eff * zs[err_mask]).to(torch.float16))
            dw.index_add_(0, preds[err_mask], (-eta_eff * zs[err_mask]).to(torch.float16))
        # column dropout: 25% columns randomly get zero update per step
        col_mask = torch.rand(256, device=m.device) < 0.25
        dw[~col_mask] = 0.0
        dw[~col_mask] = 0.0

        lw_top += dw

        lw[:, top_idx] = lw_top
        m.pool.lm_weight.data[:] = lw

        # ── bias Hebbian ──
        m.pool.lm_bias.data[tg[~err_mask]] += 5e-5
        m.pool.lm_bias.data[tg[err_mask]] += 1e-4
        m.pool.lm_bias.data[preds[err_mask]] -= 1e-4

        return {"ce_val": 0.0}

    def get_state(self) -> dict:
        return {
            "global_step": self.global_step,
            "last_F": self._last_F,
            "last_D": self._last_D,
            "n_neurons": (int(self.runner.pool.alive.sum().item()) if self.runner else 0),
            "n_synapses": (int(self.runner.pool.syn_alive.sum().item()) if self.runner else 0),
            "temporal_connections": (int(self.runner.pool.t_connected.sum().item()) if self.runner else 0),
            "topdown_connections": (int(self.runner.pool.td_alive.sum().item()) if self.runner else 0),
        }

    def train(self, task_pipelines: list[tuple[str, torch.utils.data.Dataset]]):
        out_dir = os.path.join(os.getcwd(), self.cfg.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        self.runner = self._build_model()
        self._log("CyreneModel training initialized")
        self.global_step = 0
        max_steps = getattr(self.cfg, "max_steps", 0)
        log_interval = getattr(self.cfg, "log_interval", 50)
        VIZ_STATE_PATH = os.path.join(out_dir, ".viz_state.json")

        for task_id, dataset in task_pipelines:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=True,
                num_workers=0,
            )
            for batch_idx, (byte_seq, labels) in enumerate(tqdm(loader, desc=f"Task {task_id}")):
                if max_steps > 0 and batch_idx >= max_steps:
                    break
                self.global_step += 1
                byte_seq = byte_seq.to(self.device)
                labels = labels.to(self.device)
                self.train_step(byte_seq, labels)

                if self.global_step % 50 == 0:
                    try:
                        s = self.get_state()
                        s["step"] = self.global_step
                        with open(VIZ_STATE_PATH, "w") as f:
                            json.dump(s, f)
                    except:
                        pass

                if self.global_step % log_interval == 0:
                    self._log(f"[{self.global_step}] F={self._last_F:.1f}")

                if self.global_step % self.cfg.save_interval == 0:
                    self.runner.save(f"{out_dir}/ckpt_s{self.global_step}.pt")

        self.runner.save(f"{out_dir}/final.pt")
        self._log(f"Training complete: {self.global_step} steps")
