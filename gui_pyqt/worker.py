"""
后台 Worker 线程 — 将 core.* 模块封装为 PyQt6 QThread.

4 种 Worker 类型:
  - TrainingWorker:  训练 (ThreadedTrainer)
  - EvalWorker:      评估 (run_full_evaluation)
  - AutonomousWorker: 持续自主运行 (AutonomousMind)
  - ScanWorker:      文件/检查点扫描 (非阻塞)

用法:
    worker = TrainingWorker(config, pipelines)
    worker.progress.connect(update_ui)
    worker.finished.connect(on_done)
    worker.start()
    ...
    worker.requestInterruption()  # 停止
"""

from __future__ import annotations

import os
import json
import time
import glob
import traceback
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from core.training import TrainingConfig
from core.threaded_trainer import ThreadedTrainer
from core.evaluation import run_full_evaluation, create_eval_loader, load_with_remap
from core.autonomous_mind import AutonomousMind, DEFAULT_CFG
from model.pc_layers import PCLocalDynamicMiniMind
from model.model_minimind import MiniMindConfig


# ═══════════════════════════════════════════════════════════════════
# 工具: 进度回调 → pyqtSignal 适配器
# ═══════════════════════════════════════════════════════════════════


class _ProgressAdapter:
    """将普通 callback 函数转换为 pyqtSignal 发射."""

    def __init__(self, signal: pyqtSignal):
        self._signal = signal

    def __call__(self, data: dict):
        self._signal.emit(data)


# ═══════════════════════════════════════════════════════════════════
# TrainingWorker
# ═══════════════════════════════════════════════════════════════════


class TrainingWorker(QThread):
    """后台训练线程 — 封装 ThreadedTrainer."""

    progress = pyqtSignal(dict)   # 训练进度 (与 ProgressCallback 格式一致)
    finished = pyqtSignal(dict)   # {"status": "ok"|"error", "message": str, "state": dict}
    checkpoint_saved = pyqtSignal(str)  # 检查点路径

    def __init__(self, config: TrainingConfig,
                 task_pipelines: list,
                 parent=None):
        super().__init__(parent)
        self._config = config
        self._pipelines = task_pipelines
        self._trainer: ThreadedTrainer | None = None

    def run(self) -> None:
        try:
            adapter = _ProgressAdapter(self.progress)
            self._trainer = ThreadedTrainer(self._config, progress_callback=adapter)

            # 将 isInterruptionRequested 桥接到 stop flag
            original_run = self._trainer._run_training

            def _patched_run():
                # 每步检查 QThread 中断请求
                orig_check = self._trainer._check_stop
                self._trainer._check_stop = lambda: (
                    orig_check() or self.isInterruptionRequested()
                )
                original_run()
                self._trainer._check_stop = orig_check

            self._trainer._run_training = _patched_run
            self._trainer._custom_pipelines = self._pipelines
            self._trainer.start()
            self._trainer.wait()

            final = self._trainer.get_final_state()
            self.finished.emit({"status": "ok", "message": "训练完成", "state": final})

        except Exception as e:
            self.finished.emit({
                "status": "error",
                "message": f"{type(e).__name__}: {e}",
                "state": {},
            })

    def request_pause(self) -> None:
        """暂停训练."""
        if self._trainer:
            self._trainer.pause()

    def request_resume(self) -> None:
        """恢复训练."""
        if self._trainer:
            self._trainer.resume()

    def is_paused(self) -> bool:
        return self._trainer is not None and self._trainer.is_paused()


# ═══════════════════════════════════════════════════════════════════
# EvalWorker
# ═══════════════════════════════════════════════════════════════════


class EvalWorker(QThread):
    """后台评估线程 — 封装 run_full_evaluation."""

    progress = pyqtSignal(dict)   # 评估进度
    finished = pyqtSignal(dict)   # {"status": "ok"|"error", "results": dict}
    log = pyqtSignal(str)         # 日志消息

    def __init__(self, checkpoint_path: str,
                 data_path: str = "dataset/sft_t2t.jsonl",
                 hidden_size: int = 256,
                 num_layers: int = 4,
                 max_seq_len: int = 128,
                 batch_size: int = 8,
                 max_samples: int = 500,
                 max_batches: int = 20,
                 gamma: float = 0.1,
                 T: int = 2,
                 prompts: list[str] | None = None,
                 parent=None):
        super().__init__(parent)
        self._ckpt = checkpoint_path
        self._data = data_path
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._max_seq_len = max_seq_len
        self._batch_size = batch_size
        self._max_samples = max_samples
        self._max_batches = max_batches
        self._gamma = gamma
        self._T = T
        self._prompts = prompts or ["人工智能的未来在于", "小明今天去了公园，他看到", "深度学习是一种"]

    def run(self) -> None:
        try:
            import torch

            self.log.emit(f"加载模型: {self._ckpt}")
            lm_cfg = MiniMindConfig(
                hidden_size=self._hidden_size,
                num_hidden_layers=self._num_layers,
                use_moe=False,
            )
            model = PCLocalDynamicMiniMind(lm_cfg)
            model = load_with_remap(model, self._ckpt)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()

            self.log.emit(f"加载数据: {self._data}")
            loader = create_eval_loader(
                self._data,
                max_length=self._max_seq_len,
                max_samples=self._max_samples,
                batch_size=self._batch_size,
            )
            pos = model.get_position_embeddings(self._max_seq_len, device)

            models_dict = {"model": (model, pos)}
            results = run_full_evaluation(
                models_dict,
                loader,
                gamma=self._gamma,
                T=self._T,
                max_batches=self._max_batches,
                prompts=self._prompts,
            )
            self.finished.emit({"status": "ok", "results": results})

        except Exception as e:
            self.finished.emit({
                "status": "error",
                "message": f"{type(e).__name__}: {e}",
                "results": {},
            })


# ═══════════════════════════════════════════════════════════════════
# AutonomousWorker
# ═══════════════════════════════════════════════════════════════════


class AutonomousWorker(QThread):
    """后台持续自主运行线程 — 封装 AutonomousMind."""

    progress = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    log = pyqtSignal(str)

    def __init__(self,
                 checkpoint: str | None = None,
                 out_dir: str = "out_autonomous",
                 hidden_size: int = 256,
                 num_layers: int = 4,
                 wake_steps: int = 20,
                 play_steps: int = 100,
                 sleep_interval: int = 500,
                 batch_size: int = 16,
                 lr: float = 1e-4,
                 gamma: float = 0.05,
                 T_infer: int = 1,
                 data_dir: str = "dataset",
                 parent=None):
        super().__init__(parent)
        self._cfg = {
            **DEFAULT_CFG,
            "checkpoint": checkpoint,
            "out_dir": out_dir,
            "wake_steps": wake_steps,
            "play_steps": play_steps,
            "sleep_interval": sleep_interval,
            "batch_size": batch_size,
            "lr": lr,
            "gamma": gamma,
            "T_infer": T_infer,
            "data_dir": data_dir,
        }
        self._lm_cfg = MiniMindConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            use_moe=False,
        )
        self._mind: AutonomousMind | None = None

    def run(self) -> None:
        try:
            self.log.emit("启动持续自主运行...")
            self._mind = AutonomousMind(lm_config=self._lm_cfg, cfg=self._cfg)

            # 每步检查中断
            original_step = self._mind._step

            def _patched_step(*args, **kwargs):
                if self.isInterruptionRequested():
                    raise InterruptedError("用户请求停止")
                return original_step(*args, **kwargs)

            self._mind._step = _patched_step
            self._mind.run_forever()
            self.finished.emit({"status": "ok", "message": "自主运行已完成"})

        except InterruptedError:
            self.finished.emit({"status": "stopped", "message": "用户停止"})
        except Exception as e:
            self.finished.emit({
                "status": "error",
                "message": f"{type(e).__name__}: {e}",
            })

    def request_stop(self) -> None:
        self.requestInterruption()


# ═══════════════════════════════════════════════════════════════════
# ScanWorker
# ═══════════════════════════════════════════════════════════════════


class ScanWorker(QThread):
    """后台扫描线程 — 扫描检查点 / 数据集 / 模板."""

    finished = pyqtSignal(dict)  # {"type": "checkpoints"|"datasets"|"templates", "items": list}

    def __init__(self, scan_type: str, base_dir: str = "ola_out", parent=None):
        super().__init__(parent)
        self._type = scan_type
        self._base = base_dir

    def run(self) -> None:
        try:
            if self._type == "checkpoints":
                items = self._scan_pt_files()
            elif self._type == "datasets":
                items = self._scan_datasets()
            elif self._type == "templates":
                items = self._scan_templates()
            else:
                items = []
            self.finished.emit({"type": self._type, "items": items})
        except Exception as e:
            self.finished.emit({"type": self._type, "items": [], "error": str(e)})

    def _scan_pt_files(self) -> list[dict]:
        entries = []
        if not os.path.isdir(self._base):
            return entries
        for fpath in glob.glob(os.path.join(self._base, "**", "*.pt"), recursive=True):
            try:
                stat = os.stat(fpath)
                rel = os.path.relpath(fpath, self._base)
                parts = rel.replace("\\", "/").split("/")
                fname = os.path.splitext(os.path.basename(fpath))[0]
                step = 0
                if "s" in fname:
                    try:
                        step = int(fname.split("s")[-1])
                    except ValueError:
                        pass
                entries.append({
                    "path": fpath,
                    "dir": parts[0] if len(parts) > 1 else "",
                    "filename": os.path.basename(fpath),
                    "size_kb": stat.st_size / 1024,
                    "step": step,
                    "mtime": stat.st_mtime,
                })
            except OSError:
                pass
        entries.sort(key=lambda e: e["mtime"], reverse=True)
        return entries

    def _scan_datasets(self) -> list[dict]:
        ds_dir = "dataset"
        if not os.path.isdir(ds_dir):
            return []
        files = []
        for f in sorted(os.listdir(ds_dir)):
            if f.endswith(".jsonl"):
                fpath = os.path.join(ds_dir, f)
                size_kb = os.path.getsize(fpath) / 1024
                files.append({"name": f, "path": fpath, "size_kb": round(size_kb, 1)})
        return files

    def _scan_templates(self) -> list[dict]:
        tmpl_dir = os.path.join(self._base, "configs")
        items = []
        if not os.path.isdir(tmpl_dir):
            return items
        for fname in sorted(os.listdir(tmpl_dir)):
            if fname.endswith(".json"):
                try:
                    fpath = os.path.join(tmpl_dir, fname)
                    with open(fpath, "r") as f:
                        data = json.load(f)
                    items.append({
                        "name": os.path.splitext(fname)[0],
                        "path": fpath,
                        "config": data,
                    })
                except Exception:
                    pass
        return items
