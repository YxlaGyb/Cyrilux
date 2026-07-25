"""StreamRunner 训练循环 — 事件驱动, 纯局部 Hebbian, 零 autograd.

替换旧世界的 CyrenePC 5-phase bp_free 训练环.
StreamRunner.step() 内部已包含: 感官前端 -> 事件驱动传播 -> 自由能 ->
D/ACh/pi 调制 -> Hebbian 更新. 训练循环仅负责数据注入 + 监控 + 持续学习.

Callback 架构保留: CheckpointCallback, ContinualCallback, IntrinsicCallback 等.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
from tqdm import tqdm

from model.continual.hippocampus_buffer import HippocampusBuffer
from model.core.globals import DEVICE
from pkg.utils.trainer_utils import setup_seed
from model.model_cyrene import CyreneConfig, CyreneModel

from .callback_base import CallbackBase
from .callbacks import (
    CheckpointCallback,
    ContinualCallback,
    IntrinsicCallback,
    LoggingCallback,
    PipelineCallback,
    SleepCallback,
)
from .config import TrainingConfig

_LOG2 = math.log(2)


class TrainingLoop:
    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.device = DEVICE
        self._setup_environment()
        self.runner: Optional[CyreneModel] = None
        self.global_step = 0
        self._total_steps = 0
        self._last_stats: dict = {}
        self.forgetting_log: list[dict] = []
        self._last_world_surprise: float = 0.0
        self._last_world_loss: Optional[float] = None
        self._F_trend_buffer: list[float] = []
        self._surprise_buffer: list[float] = []
        self._current_task_id: str = ""
        self._trained_tasks: list[str] = []
        self._last_F: float = float("inf")
        self._last_D: float = 0.5
        self._wm_metrics: dict[str, list[float]] = {
            "transition_error": [], "uncertainty": [], "fp_rate": []
        }
        self._wm_fp_count: int = 0
        self._wm_high_surprise_count: int = 0
        self._last_ce_for_fp: float = 0.0
        self._novelty_boost_steps: int = 0
        self._novelty_surprise_injected: float = 0.0
        self._fallback_state: Optional[dict] = None
        self.memory_bank = None
        self.sniffer = None
        self.abstraction_bank = None
        self.abstraction_sniffer = None
        self.hippocampus = HippocampusBuffer(capacity=200, min_info_gain=0.03)
        self.icm = None
        self.concept_discovery = None
        self.memory_gate = None
        self._icm_output: Optional[dict] = None
        self._intrinsic_stats: dict[str, list] = {
            "pred_loss": [],
            "inverse_loss": [],
            "information_gain": [],
            "uncertainty": [],
            "n_concepts": [],
        }
        self.world_model = None
        self.landscape = None
        self.consolidation_pipeline = None
        self.sleep_engine = None
        self.callbacks: list[CallbackBase] = []

    def _emit(self, event: str, **kwargs):
        for cb in self.callbacks:
            try:
                fn = getattr(cb, event, None)
                if fn is not None:
                    fn(loop=self, **kwargs)
            except Exception as e:
                self._log(f"[Callback] {type(cb).__name__}.{event} error: {e}")

    def _build_default_callbacks(self):
        self.callbacks = [
            ContinualCallback(),
            IntrinsicCallback(),
            PipelineCallback(),
            SleepCallback(),
            CheckpointCallback(),
            LoggingCallback(),
        ]

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
        ACh = stats.get("ACh", 0.5)
        modulation = stats.get("modulation", 0.5)
        uncertainty = stats.get("uncertainty", 0.5)
        lm_loss = stats.get("lm_loss", 0.0)
        pred_byte = stats.get("pred_byte", -1)
        n_neurons = stats.get("n_neurons", 0)
        n_synapses = stats.get("n_synapses", 0)
        firing_rate = stats.get("firing_rate", 0.0)
        is_warmup = stats.get("warmup", False)

        self._last_F = F_curr
        self._last_D = D
        self._F_trend_buffer.append(F_curr)

        if self.sniffer is not None and math.isfinite(F_curr):
            self.sniffer.observe_free_energy(F_curr)

        result = {
            "ce_val": lm_loss,
            "F_val": F_curr,
            "D": D,
            "ACh": ACh,
            "modulation": modulation,
            "uncertainty": uncertainty,
            "lr": self.cfg.hebbian_base_eta * modulation,
            "firing_rate": firing_rate,
            "n_neurons": n_neurons,
            "n_synapses": n_synapses,
            "warmup": is_warmup,
            "phase": "event_driven",
        }
        if pred_byte >= 0:
            result["pred_byte"] = pred_byte

        info_gain = (
            self._icm_output.get("information_gain", 0.0)
            if self._icm_output
            else 0.0
        )
        if info_gain > self.hippocampus.min_info_gain:
            self.hippocampus.add(
                z_states=None,
                byte_tensor=byte_seq[0].detach(),
                label_tensor=labels[0].detach(),
                info_gain=info_gain,
                step=self.global_step,
            )

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
            "forgetting_log": self.forgetting_log[-10:],
        }

    def train(
        self,
        task_pipelines: list[
            tuple[str, torch.utils.data.Dataset, Optional[torch.utils.data.DataLoader]]
        ],
    ):
        out_dir = os.path.join(os.getcwd(), self.cfg.out_dir)
        os.makedirs(out_dir, exist_ok=True)

        self.runner = self._build_model()
        self._log("StreamRunner training initialized")

        self.neurogenesis = None
        self.world_model = None
        self.icm = None
        self.concept_discovery = None
        self.memory_gate = None
        self.landscape = None
        self.consolidation_pipeline = None

        self._total_steps = sum(
            min(len(ds), self.cfg.max_steps) if hasattr(ds, "__len__") else self.cfg.max_steps
            for _, ds, _ in task_pipelines
        )

        self._emit("on_training_start")
        self.global_step = 0

        for task_id, dataset, _loader_override in task_pipelines:
            self._current_task_id = task_id
            self._trained_tasks.append(task_id)

            loader = (
                _loader_override
                if _loader_override is not None
                else torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.cfg.batch_size,
                    shuffle=True,
                    num_workers=0,
                )
            )
            self._emit("on_task_start", task_id=task_id)

            for batch_idx, (byte_seq, labels) in enumerate(
                tqdm(loader, desc=f"Task {task_id}")
            ):
                if self.cfg.max_steps > 0 and batch_idx >= self.cfg.max_steps:
                    break
                self.global_step += 1

                byte_seq = byte_seq.to(self.device)
                labels = labels.to(self.device)
                result = self.train_step(byte_seq, labels)

                if self.global_step % self.cfg.log_interval == 0:
                    self._emit("on_log", stats=result, step=self.global_step)

                if self.global_step % self.cfg.save_interval == 0:
                    self._emit("on_checkpoint", step=self.global_step)

            self._emit("on_task_end", task_id=task_id)

        self._emit("on_training_end")
        self._log(f"Training complete: {self.global_step} steps")
