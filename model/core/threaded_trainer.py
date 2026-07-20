"""
线程安全的训练管理器 — 用于 Tkinter GUI 后台训练。

在独立线程中运行 TrainingLoop，通过回调推送进度到 GUI。

用法:
    from model.core.threaded_trainer import ThreadedTrainer, TkProgressCallback

    trainer = ThreadedTrainer(config, progress_callback=TkProgressCallback(text_widget))
    trainer.start()
    trainer.stop()

    # 训练完成后获取模型做 Phase 2
    model = trainer.get_model()
"""
import os, json, math, threading, traceback
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import DataLoader

from model.core.training import TrainingLoop, TrainingConfig
from model.core.dataset import DualChannelDataset, load_datasets
from model.core.trainer_utils import Logger


# ─── Tkinter 进度回调 ──────────────────────────────────────────

class TkProgressCallback:
    """将训练进度推送到 Tkinter Text 组件。"""

    def __init__(self, text_widget, root=None):
        """
        Args:
            text_widget: tkinter.Text 或 scrolledtext.ScrolledText 实例
            root: tkinter.Tk 根窗口 (用于线程安全调度)
        """
        self.text = text_widget
        self.root = root

    def __call__(self, data: dict):
        msg_type = data.get('type', 'log')
        message = data.get('message', '')

        if self.root:
            self.root.after(0, self._do_update, data)
        else:
            self._do_update(data)

    def _do_update(self, data: dict):
        msg_type = data.get('type', 'log')
        message = data.get('message', '')

        if msg_type == 'log':
            self._insert(f'[INFO] {message}\n')
        elif msg_type == 'progress':
            step = data.get('step', 0)
            total = data.get('total_steps', 0)
            ce = data.get('ce_loss', 0)
            F = data.get('F', 0)
            D = data.get('D', 0)
            lr = data.get('lr', 0)
            self._insert(
                f'[Step {step}/{total}] CE={ce:.4f} F={F:.1f} D={D:.3f} lr={lr:.2e}\n'
            )
        elif msg_type == 'phase':
            pname = data.get('phase_name', '')
            self._insert(f'\n── [{pname}] {message} ──\n')
        elif msg_type == 'checkpoint':
            ckpt_path = data.get('checkpoint_path', '')
            self._insert(f'[CHECKPOINT] {ckpt_path}\n')
        elif msg_type == 'done':
            self._insert(f'\n✅ {message}\n')
        elif msg_type == 'error':
            self._insert(f'\n❌ {message}\n')

    def _insert(self, text: str):
        self.text.insert('end', text)
        self.text.see('end')
        self.text.update_idletasks()


# ─── 线程安全训练管理器 ────────────────────────────────────────

class ThreadedTrainer:
    """
    后台训练管理器。在独立线程中运行 TrainingLoop，通过回调推送到 GUI。

    用法:
        trainer = ThreadedTrainer(config, progress_callback=my_callback)
        trainer.start()
        ...
        trainer.stop()
        trainer.wait()
        model = trainer.get_model()
    """

    def __init__(self, config: TrainingConfig, progress_callback=None):
        self.config = config
        self.callback = progress_callback or (lambda x: None)
        self._stop_flag = threading.Event()
        self._pause_flag = threading.Event()
        self._pause_flag.set()  # 默认未暂停 (set = 运行中)
        self._thread: Optional[threading.Thread] = None
        self._trained_loop: Optional[TrainingLoop] = None  # 训练后的训练器 (含模型)
        self._final_state: dict = {}

    # ── 公开接口 ──

    def start(self):
        """启动后台训练 (非阻塞)。"""
        if self._thread and self._thread.is_alive():
            self._log('训练已在运行中')
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run_training, daemon=True)
        self._thread.start()
        self._log('训练线程已启动')

    def stop(self):
        """请求停止训练。"""
        self._stop_flag.set()
        self._pause_flag.set()  # 解除暂停阻塞, 让线程快速退出
        self._log('正在停止训练...')

    def pause(self):
        """暂停训练。"""
        self._pause_flag.clear()
        self._log('⏸ 训练已暂停')

    def resume(self):
        """恢复训练。"""
        self._pause_flag.set()
        self._log('▶ 训练已恢复')

    def is_paused(self) -> bool:
        return not self._pause_flag.is_set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_model(self):
        """获取训练后的模型 (供 Phase 2 使用)。"""
        if self._trained_loop:
            return self._trained_loop.model
        return None

    def get_loop(self):
        """获取 TrainingLoop 实例 (含模型、优化器等)。"""
        return self._trained_loop

    def get_final_state(self) -> dict:
        return self._final_state

    def wait(self, timeout=None):
        """等待训练完成。"""
        if self._thread:
            self._thread.join(timeout)

    # ── 内部 ──

    def _log(self, msg: str):
        self.callback({'type': 'log', 'message': msg})

    def _progress(self, **kwargs):
        kwargs['type'] = 'progress'
        self.callback(kwargs)

    def _emit(self, **kwargs):
        self.callback(kwargs)

    def _check_stop(self):
        """检查停止标志。"""
        return self._stop_flag.is_set()

    def _build_task_pipelines(self, task_specs: list) -> list:
        """
        从任务规格构建 (task_id, dataset) 列表。

        task_specs: [(task_id, data_path_or_ds, max_samples?), ...]
        """
        import json
        pipelines = []
        for spec in task_specs:
            task_id = spec[0]
            data_source = spec[1]

            if isinstance(data_source, torch.utils.data.Dataset):
                pipelines.append((task_id, data_source))
            elif isinstance(data_source, list):
                # 文件路径列表 → 多文件合并
                ds = load_datasets(
                    data_source,
                    max_length=self.config.max_seq_len,
                    max_samples=spec[2] if len(spec) > 2 else 0,
                )
                pipelines.append((task_id, ds))
            elif isinstance(data_source, str):
                # 单文件路径
                ds = DualChannelDataset(
                    data_source,
                    max_length=self.config.max_seq_len,
                    max_samples=spec[2] if len(spec) > 2 else 0,
                )
                pipelines.append((task_id, ds))

        return pipelines

    def _run_training(self):
        """内部训练循环 (在独立线程中运行)。"""
        try:
            loop = TrainingLoop(self.config)

            # ── 确保 self.callback 指向真实回调 ──
            # 优先: GUI 模式通过 config.progress_callback 传入
            real_cb = loop.cfg.progress_callback or self.callback
            self.callback = real_cb  # 使 _log / _progress / _emit 能从队列到达 GUI

            # ── 包装 progress_callback 以支持暂停/停止 ──
            orig_cb = real_cb

            def progress_wrapper(data: dict):
                data_type = data.get('type', '')
                if data_type == 'progress':
                    # 暂停阻塞: 等待 pause_flag 被 set (仅当 paused 时阻塞)
                    self._pause_flag.wait()
                    # 停止检查
                    if self._check_stop():
                        return
                # 所有事件直接转发到 GUI 队列
                orig_cb(data)

            loop.cfg.progress_callback = progress_wrapper

            # ── 构建任务流水线 ──
            # 优先使用 GUI 传入的 _custom_pipelines (选中文件)
            if hasattr(self, '_custom_pipelines') and self._custom_pipelines:
                task_pipelines = self._build_task_pipelines(self._custom_pipelines)
            else:
                # 回退: 从 datasets/ 目录自动发现
                from pathlib import Path
                data_dir = Path(os.getcwd()) / 'datasets'
                if data_dir.exists():
                    jsonl_files = sorted(data_dir.glob('*.jsonl'))
                else:
                    jsonl_files = []
                if jsonl_files:
                    pipelines = [('default', str(jsonl_files[0]), self.config.subset if self.config.subset > 0 else 0)]
                    task_pipelines = self._build_task_pipelines(pipelines)
                else:
                    self._emit(type='error', message='未找到数据文件 (请先在 GUI 中选择数据集文件)')
                    return

            # ── 训练 ──
            loop.train(task_pipelines)

            self._trained_loop = loop
            self._final_state = {
                'model': loop.model,
                'config': self.config.to_dict(),
                'steps': loop.global_step,
            }

            self._emit(type='done', message='训练完成！')

        except Exception as e:
            tb = traceback.format_exc()
            self._emit(type='error', message=f'训练出错: {str(e)}\n{tb}')
            self._log(f'错误: {str(e)}')

    def set_task_pipelines(self, pipelines: list):
        """设置自定义任务流水线。"""
        self._custom_pipelines = pipelines


# ─── 独立运行入口 ──────────────────────────────────────────────

def run_training_standalone(config: TrainingConfig, task_specs: list = None):
    """
    同步运行训练 (非 GUI 模式)。

    Args:
        config: TrainingConfig 配置
        task_specs: [(task_id, data_path_or_ds, max_samples?), ...]
    """
    import sys
    mgr = ThreadedTrainer(config, progress_callback=lambda x: print(
        f'[{x.get("type","?")}] {x.get("message","")}' 
        if x.get('type') in ('log', 'phase', 'checkpoint', 'done', 'error')
        else f'[Step {x.get("step","?")}/{x.get("total_steps","?")}] CE={x.get("ce_loss",0):.4f} F={x.get("F",0):.1f} D={x.get("D",0):.3f}'
    ))

    if task_specs:
        mgr.set_task_pipelines(mgr._build_task_pipelines(task_specs))

    mgr.start()
    mgr.wait()
    return mgr.get_model()
