"""TrainingLoop
事件驱动, 纯局部 Hebbian, 零 autograd.

CyreneModel.step() 内部已包含: 感官前端 -> 事件驱动传播 -> 自由能 ->
D/ACh/pi 调制 -> Hebbian 更新. 训练循环仅负责数据注入 + 监控.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from tqdm import tqdm

from pkg.utils.trainer_utils import setup_seed
from model.model_cyrene import CyreneConfig, CyreneModel

from .config import TrainingConfig


class TrainingLoop:
    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_environment()
        self.runner: Optional[CyreneModel] = None
        self.global_step = 0
        self._last_stats: dict = {}
        self._last_F: float = float("inf")
        self._last_D: float = 0.5

    def _setup_environment(self):
        setup_seed(self.cfg.seed)
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("medium")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True

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
        )
        runner = CyreneModel(cfg)
        runner.frontend = runner.frontend.to(self.device)
        runner.add_hidden_layer(
            n_neurons=min(self.cfg.hidden_size * 4, 512),
            from_layer=0,
            to_layer=7,
            connection_density=0.2,
        )
        self._log(f"CyreneModel built: h_front={self.cfg.hidden_size}")
        return runner

    def warmup(self):
        assert self.runner is not None
        with torch.no_grad():
            dummy = torch.zeros(1, 2, 64, dtype=torch.half, device=self.device)
            for _ in range(getattr(self.cfg, 'warmup_steps', 20)):
                self.runner.step(dummy)
        self._log("Warmup done")

    def train_step(self, byte_seq: torch.Tensor, labels: torch.Tensor) -> dict:
        assert self.runner is not None
        stats = self.runner.step(byte_seq)
        F_curr = stats.get("free_energy", 0.0)
        D = stats.get("D", 0.5)
        modulation = stats.get("modulation", 0.5)
        lm_loss = stats.get("lm_loss", 0.0)

        self._last_F = F_curr
        self._last_D = D

        result = {
            "ce_val": lm_loss,
            "F_val": F_curr,
            "D": D,
            "ACh": stats.get("ACh", 0.5),
            "modulation": modulation,
            "uncertainty": stats.get("uncertainty", 0.5),
            "lr": self.cfg.hebbian_base_eta * modulation,
            "firing_rate": stats.get("firing_rate", 0.0),
            "n_neurons": stats.get("n_neurons", 0),
            "n_synapses": stats.get("n_synapses", 0),
            "warmup": stats.get("warmup", False),
            "phase": "event_driven",
        }
        pred_byte = stats.get("pred_byte", -1)
        if pred_byte >= 0:
            result["pred_byte"] = pred_byte

        self._last_stats = result
        return result

    def get_state(self) -> dict:
        return {
            "global_step": self.global_step,
            "last_F": self._last_F,
            "last_D": self._last_D,
            "n_neurons": (
                self.runner.pool.get_total_neurons() if self.runner else 0
            ),
            "n_synapses": (
                self.runner.pool.get_total_synapses() if self.runner else 0
            ),
            "temporal_connections": len(self.runner.temporal) if self.runner else 0,
            "topdown_connections": len(self.runner.topdown) if self.runner else 0,
            "lm_head_connections": len(self.runner.lm_head) if self.runner else 0,
        }

    def train(
        self,
        task_pipelines: list[tuple[str, torch.utils.data.Dataset]],
    ):
        out_dir = os.path.join(os.getcwd(), self.cfg.out_dir)
        os.makedirs(out_dir, exist_ok=True)

        self.runner = self._build_model()
        self._log("CyreneModel training initialized")
        self.global_step = 0

        max_steps = getattr(self.cfg, "max_steps", 0)
        log_interval = getattr(self.cfg, "log_interval", 50)

        for task_id, dataset in task_pipelines:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=True,
                num_workers=0,
            )

            for batch_idx, (byte_seq, labels) in enumerate(
                tqdm(loader, desc=f"Task {task_id}")
            ):
                if max_steps > 0 and batch_idx >= max_steps:
                    break
                self.global_step += 1

                byte_seq = byte_seq.to(self.device)
                labels = labels.to(self.device)
                self.train_step(byte_seq, labels)

                if self.global_step % log_interval == 0:
                    self._log(f"[{self.global_step}] F={self._last_F:.1f}")

                if self.global_step % self.cfg.save_interval == 0:
                    self.runner.save(f"{out_dir}/ckpt_s{self.global_step}.pt")

        self.runner.save(f"{out_dir}/final.pt")
        self._log(f"Training complete: {self.global_step} steps")
