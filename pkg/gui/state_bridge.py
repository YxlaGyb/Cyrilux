"""
状态桥接 — 将旧 gui.state.ExperimentState 通过 pyqtSignal 桥接到 PyQt6.

使用:
    signals = ExperimentStateSignals()
    signals.config_changed.connect(lambda k, v: print(k, v))

    bridge = bridge_state(signals)
    # bridge 代理 ExperimentState 的所有更新
"""

from __future__ import annotations

import os
import json
import glob
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from pkg.gui.state import ExperimentState


# ═══════════════════════════════════════════════════════════════════
# PyQt6 信号桥接
# ═══════════════════════════════════════════════════════════════════


class ExperimentStateSignals(QObject):
    """跨线程安全的 PyQt6 信号 — 对应 ExperimentState 状态变更."""

    config_changed = pyqtSignal(str, str)        # key, value
    checkpoints_updated = pyqtSignal(list)        # list[dict]
    templates_updated = pyqtSignal(dict)          # {name: config}
    training_progress = pyqtSignal(dict)          # 训练进度 data
    task_pipelines_changed = pyqtSignal(list)
    out_dir_changed = pyqtSignal(str)
    status_message = pyqtSignal(str)              # 状态栏消息


# ═══════════════════════════════════════════════════════════════════
# ExperimentState 桥接代理
# ═══════════════════════════════════════════════════════════════════


class StateBridge:
    """代理 ExperimentState，将所有变更通过 signals 广播."""

    def __init__(self, state: ExperimentState, signals: ExperimentStateSignals):
        self._state = state
        self.signals = signals

    # ── config 代理 ──

    def config_get(self, key: str, default: str = "") -> str:
        return self._state.config_get(key, default)

    def config_set(self, key: str, value: str) -> None:
        self._state.config_set(key, value)
        self.signals.config_changed.emit(key, value)

    def config_to_kwargs(self) -> dict:
        return self._state.config_to_kwargs()

    def config_update_from(self, d: dict) -> None:
        """批量更新 config 并逐个广播."""
        for k, v in d.items():
            old = self._state.config.get(k)
            sv = str(v)
            if old != sv:
                self._state.config[k] = sv
                self.signals.config_changed.emit(k, sv)

    # ── 检查点 ──

    def scan_checkpoints(self, base_dir: str | None = None) -> list[dict]:
        result = self._state.scan_checkpoints(base_dir)
        self.signals.checkpoints_updated.emit(result)
        return result

    @property
    def checkpoint_registry(self) -> list[dict]:
        return self._state.checkpoint_registry

    # ── 配置模板 ──

    def load_templates(self, template_dir: str = "ola_out/configs") -> dict:
        result = self._state.load_templates(template_dir)
        self.signals.templates_updated.emit(result)
        return result

    def save_template(self, name: str, template_dir: str = "ola_out/configs") -> None:
        self._state.save_template(name, template_dir)
        self.signals.templates_updated.emit(dict(self._state.config_templates))

    @property
    def config_templates(self) -> dict:
        return self._state.config_templates

    # ── 任务管线 ──

    @property
    def task_pipelines(self) -> list:
        return self._state.task_pipelines

    @task_pipelines.setter
    def task_pipelines(self, val: list) -> None:
        self._state.task_pipelines = val
        self.signals.task_pipelines_changed.emit(val)

    # ── 输出目录 ──

    @property
    def out_dir(self) -> str:
        return self._state.out_dir

    @out_dir.setter
    def out_dir(self, val: str) -> None:
        self._state.out_dir = val
        self.signals.out_dir_changed.emit(val)

    # ── 原始 state 访问 ──

    @property
    def raw(self) -> ExperimentState:
        return self._state


def bridge_state(state: ExperimentState | None = None,
                 signals: ExperimentStateSignals | None = None) -> StateBridge:
    """便捷工厂: 创建或复用 StateBridge."""
    if state is None:
        state = ExperimentState()
    if signals is None:
        signals = ExperimentStateSignals()
    return StateBridge(state, signals)
