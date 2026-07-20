"""训练页面 — 参数配置 + 数据集 + 实时图表 + 日志."""

from __future__ import annotations

import os

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QSplitter, QScrollArea,
                             QGroupBox, QFormLayout, QLineEdit,
                             QDoubleSpinBox, QSpinBox, QCheckBox,
                             QComboBox, QProgressBar, QFrame, QSizePolicy,
                             QMessageBox)

from pkg.gui.theme import HackerTheme
from pkg.gui.state_bridge import ExperimentStateSignals
from pkg.gui.widgets.realtime_chart import RealtimeChart
from pkg.gui.widgets.log_viewer import LogViewer
from pkg.gui.widgets.metric_card import MetricCard
from pkg.gui.worker import TrainingWorker
from model.core.train import TrainingConfig


def _make_spin(lo: int, hi: int, val: int, name: str = "paramSpin") -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setObjectName(name)
    return s


class _ParamGroup(QGroupBox):
    """可折叠参数组."""

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(True)
        self._form = QFormLayout(self)
        self._form.setSpacing(4)
        self._form.setContentsMargins(8, 20, 8, 8)
        self._widgets: dict[str, QWidget] = {}

    def add_row(self, key: str, label: str, widget: QWidget, default: str = "") -> None:
        self._widgets[key] = widget
        self._form.addRow(label, widget)

    def get(self, key: str, default: str = "") -> str:
        w = self._widgets.get(key)
        if w is None:
            return default
        if isinstance(w, QLineEdit):
            return w.text() or default
        if isinstance(w, (QDoubleSpinBox, QSpinBox)):
            return str(w.value())
        if isinstance(w, QCheckBox):
            return "1" if w.isChecked() else "0"
        if isinstance(w, QComboBox):
            return w.currentText() or default
        return default

    def set_val(self, key: str, val: str) -> None:
        w = self._widgets.get(key)
        if w is None:
            return
        if isinstance(w, QLineEdit):
            w.setText(val)
        elif isinstance(w, QDoubleSpinBox):
            w.setValue(float(val))
        elif isinstance(w, QSpinBox):
            w.setValue(int(val))
        elif isinstance(w, QCheckBox):
            w.setChecked(val in ("1", "true", "True"))
        elif isinstance(w, QComboBox):
            idx = w.findText(val)
            if idx >= 0:
                w.setCurrentIndex(idx)

    def to_dict(self) -> dict[str, str]:
        return {k: self.get(k) for k in self._widgets}


class TrainingPage(QWidget):
    """训练 — 参数/数据集/图表/日志."""

    def __init__(self, bridge, theme: HackerTheme, signals: ExperimentStateSignals):
        super().__init__()
        self._bridge = bridge
        self._theme = theme
        self._signals = signals
        self._worker: TrainingWorker | None = None
        self._timeline: list[dict] = []
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(6)

        heading = QLabel(">>> 训练")
        heading.setObjectName("accent")
        outer.addWidget(heading)

        # ── 主分割: 左参数 | 中-右图表 ──
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── 左: 可滚动参数面板 ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        param_panel = QWidget()
        param_layout = QVBoxLayout(param_panel)
        param_layout.setSpacing(6)
        param_layout.setContentsMargins(0, 0, 0, 0)

        # 1) 模型参数
        self._g_model = _ParamGroup("模型")
        self._g_model.add_row("hidden_size", "hidden_size:", _make_spin(64, 1024, 256))
        self._g_model.add_row("num_hidden_layers", "layers:", _make_spin(1, 24, 4))
        self._g_model.add_row("use_moe", "use_moe:", QCheckBox())
        param_layout.addWidget(self._g_model)

        # 2) 训练参数
        self._g_train = _ParamGroup("训练")
        self._g_train.add_row("lr", "lr:", QLineEdit("0.001", objectName="paramInput"))
        self._g_train.add_row("batch_size", "batch_size:", _make_spin(1, 512, 48))
        self._g_train.add_row("epochs", "epochs:", _make_spin(1, 100, 3))
        self._g_train.add_row("max_seq_len", "max_seq_len:", _make_spin(16, 2048, 128))
        self._g_train.add_row("seed", "seed:", _make_spin(0, 999999, 42))
        self._g_train.add_row("subset", "subset:", _make_spin(-1, 99999, -1))
        self._g_train.add_row("grad_clip", "grad_clip:", QLineEdit("1.0", objectName="paramInput"))
        param_layout.addWidget(self._g_train)

        # 3) PC 参数
        self._g_pc = _ParamGroup("PC (预测编码)")
        self._g_pc.add_row("T_infer", "T_infer:", _make_spin(1, 64, 16))
        self._g_pc.add_row("gamma", "gamma:", QLineEdit("0.1", objectName="paramInput"))
        self._g_pc.add_row("max_beta", "max_beta:", QLineEdit("2.0", objectName="paramInput"))
        param_layout.addWidget(self._g_pc)

        # 4) 多巴胺参数
        self._g_dopa = _ParamGroup("多巴胺")
        self._g_dopa.add_row("dopamine_eta", "eta:", QLineEdit("1.0", objectName="paramInput"))
        self._g_dopa.add_row("dopamine_beta", "beta:", QLineEdit("0.5", objectName="paramInput"))
        self._g_dopa.add_row("dopamine_gamma", "gamma:", QLineEdit("0.3", objectName="paramInput"))
        param_layout.addWidget(self._g_dopa)

        # 5) CL 参数
        self._g_cl = _ParamGroup("持续学习")
        self._g_cl.add_row("replay_ratio", "replay_ratio:", _make_spin(1, 100, 5))
        self._g_cl.add_row("bank_size", "bank_size:", _make_spin(100, 50000, 2000))
        self._g_cl.add_row("sniff_interval", "sniff_interval:", _make_spin(10, 5000, 200))
        param_layout.addWidget(self._g_cl)

        param_layout.addStretch()
        scroll.setWidget(param_panel)
        main_splitter.addWidget(scroll)

        # ── 右: 4 个图表 + 底栏 ──
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 0, 0, 0)
        right_layout.setSpacing(4)

        # 数据集选择 & 输出目录
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("数据集:"))
        self._dataset_combo = QComboBox()
        self._dataset_combo.setObjectName("paramInput")
        self._dataset_combo.setEditable(True)
        self._dataset_combo.addItems([
            "dataset/sft_t2t.jsonl", "dataset/sft_t2t_mini.jsonl",
            "dataset/agent_rl.jsonl", "dataset/dpo.jsonl",
            "dataset/pretrain_t2t.jsonl", "dataset/lora_identity.jsonl",
        ])
        top_row.addWidget(self._dataset_combo, 1)
        top_row.addWidget(QLabel("输出:"))
        self._out_dir_edit = QLineEdit("ola_out/out_train", objectName="paramInput")
        top_row.addWidget(self._out_dir_edit, 1)
        right_layout.addLayout(top_row)

        # 4 个实时图表 (2x2 网格)
        chart_grid = QHBoxLayout()
        chart_left = QVBoxLayout()
        self._chart_ce = RealtimeChart()
        self._chart_ce.set_title("CE Loss")
        chart_left.addWidget(self._chart_ce, 1)
        self._chart_F = RealtimeChart()
        self._chart_F.set_title("F (预测误差)")
        chart_left.addWidget(self._chart_F, 1)
        chart_grid.addLayout(chart_left)

        chart_right = QVBoxLayout()
        self._chart_D = RealtimeChart()
        self._chart_D.set_title("D (多巴胺)")
        chart_right.addWidget(self._chart_D, 1)
        self._chart_lr = RealtimeChart()
        self._chart_lr.set_title("LR")
        chart_right.addWidget(self._chart_lr, 1)
        chart_grid.addLayout(chart_right)
        right_layout.addLayout(chart_grid, 1)

        # ── 控制栏 + 进度 ──
        ctrl_row = QHBoxLayout()
        self._start_btn = QPushButton("▶ 开始")
        self._start_btn.clicked.connect(self._start_training)
        ctrl_row.addWidget(self._start_btn)
        self._pause_btn = QPushButton("⏸ 暂停")
        self._pause_btn.clicked.connect(self._toggle_pause)
        self._pause_btn.setEnabled(False)
        ctrl_row.addWidget(self._pause_btn)
        self._stop_btn = QPushButton("⏹ 停止")
        self._stop_btn.clicked.connect(self._stop_training)
        self._stop_btn.setEnabled(False)
        ctrl_row.addWidget(self._stop_btn)
        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("progressBar")
        self._progress_bar.setRange(0, 100)
        ctrl_row.addWidget(self._progress_bar, 1)
        right_layout.addLayout(ctrl_row)

        # 日志
        self._log = LogViewer()
        right_layout.addWidget(self._log, 2)

        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([320, 480])
        outer.addWidget(main_splitter, 1)

    def _connect_signals(self) -> None:
        self._signals.config_changed.connect(self._on_config_changed)
        self._signals.status_message.connect(self._log.info)

    def _on_config_changed(self, key: str, value: str) -> None:
        for g in (self._g_model, self._g_train, self._g_pc, self._g_dopa, self._g_cl):
            g.set_val(key, value)

    def _collect_config(self) -> TrainingConfig:
        d = {}
        for g in (self._g_model, self._g_train, self._g_pc, self._g_dopa, self._g_cl):
            d.update(g.to_dict())
        # dataset_path 不传给 TrainingConfig (单独通过 pipelines 传入)
        d.pop("dataset_path", None)
        return TrainingConfig(**{k: self._coerce(v) for k, v in d.items()})

    @staticmethod
    def _coerce(v: str | None):
        if v is None:
            return None
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        return v

    def _start_training(self) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "提示", "训练已在运行中")
            return
        cfg = self._collect_config()
        subset = int(self._g_train.get("subset", "-1"))
        pipelines = [
            ("task_1", self._dataset_combo.currentText(),
             subset if subset > 0 else 0),
        ]
        self._worker = TrainingWorker(cfg, pipelines)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.checkpoint_saved.connect(
            lambda p: self._log.info(f"检查点已保存: {p}"))

        self._timeline.clear()
        for chart in (self._chart_ce, self._chart_F, self._chart_D, self._chart_lr):
            chart.clear_series()

        self._worker.start()
        self._start_btn.setEnabled(False)
        self._pause_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._log.info("训练已启动")

    def _on_progress(self, data: dict) -> None:
        t = data.get("type", "")
        if t == "progress":
            step = data.get("step", 0)
            total = data.get("total_steps", 1)
            self._progress_bar.setValue(int(step / total * 100) if total else 0)
            self._chart_ce.append("ce", data.get("ce_loss", 0))
            self._chart_F.append("F", data.get("F", 0))
            self._chart_D.append("D", data.get("D", 0))
            self._chart_lr.append("lr", data.get("lr", 0))
        elif t == "log":
            self._log.info(data.get("message", ""))
        elif t == "phase":
            self._log.info(f"▶ 阶段 {data.get('phase', '?')}")
        elif t == "checkpoint":
            self._log.ok(f"💾 检查点: {data.get('checkpoint_path', '')}")
        elif t == "task_done":
            self._log.ok(f"✔ 任务完成: {data.get('task_id', '')}")
        elif t == "done":
            self._log.ok("✔ 训练完成")
        elif t == "error":
            self._log.error(f"✖ 错误: {data.get('message', '')}")

    def _on_finished(self, result: dict) -> None:
        self._start_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._progress_bar.setValue(100)
        msg = result.get("message", "")
        if result.get("status") == "ok":
            self._log.ok(f"✔ {msg}")
        else:
            self._log.error(f"✖ {msg}")

    def _toggle_pause(self) -> None:
        if not self._worker:
            return
        if self._worker.is_paused():
            self._worker.request_resume()
            self._pause_btn.setText("⏸ 暂停")
            self._log.info("恢复训练")
        else:
            self._worker.request_pause()
            self._pause_btn.setText("▶ 恢复")
            self._log.info("训练已暂停")

    def _stop_training(self) -> None:
        if self._worker:
            self._worker.requestInterruption()
            self._log.warn("⏹ 请求停止训练")

    def on_theme_changed(self, theme) -> None:
        self._theme = theme
