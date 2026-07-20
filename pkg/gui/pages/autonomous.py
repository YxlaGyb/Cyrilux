"""自主运行页面 — WAKE/PLAY/SLEEP 阶段指示器 + 实时文本 + 曲线."""

from __future__ import annotations

import os
import json

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QSplitter, QGroupBox,
                             QFormLayout, QLineEdit, QSpinBox,
                             QTextEdit, QFrame, QProgressBar,
                             QMessageBox, QSizePolicy)

from pkg.gui.theme import HackerTheme, DARK_PALETTE
from pkg.gui.widgets.realtime_chart import RealtimeChart
from pkg.gui.widgets.log_viewer import LogViewer
from pkg.gui.widgets.metric_card import MetricCard
from pkg.gui.worker import AutonomousWorker


def _make_spin(lo: int, hi: int, val: int, name: str = "paramSpin") -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setObjectName(name)
    return s


PHASE_COLORS = {
    "WAKE": "#00ff41",
    "PLAY": "#ffd700",
    "SLEEP": "#00bfff",
}


class _PhaseBadge(QLabel):
    """阶段指示器徽章."""

    def __init__(self, parent=None):
        super().__init__("● 待命", parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            f"background:{DARK_PALETTE.card_bg};color:{DARK_PALETTE.fg_secondary};"
            f"font-size:18px;font-weight:bold;padding:8px 16px;"
            f"border:1px solid {DARK_PALETTE.border};border-radius:4px;")
        self.setFixedHeight(48)

    def set_phase(self, phase: str) -> None:
        color = PHASE_COLORS.get(phase, DARK_PALETTE.accent)
        text = phase if phase in PHASE_COLORS else "待命"
        self.setText(f"● {text}")
        self.setStyleSheet(
            f"background:{DARK_PALETTE.card_bg};color:{color};"
            f"font-size:18px;font-weight:bold;padding:8px 16px;"
            f"border:1px solid {color};border-radius:4px;")


class AutonomousPage(QWidget):
    """自主运行 — 连接 core.autonomous_mind.AutonomousMind."""

    def __init__(self, bridge, theme: HackerTheme):
        super().__init__()
        self._bridge = bridge
        self._theme = theme
        self._worker: AutonomousWorker | None = None
        self._phase_badge = _PhaseBadge()
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(6)

        heading = QLabel(">>> 自主运行")
        heading.setObjectName("accent")
        outer.addWidget(heading)

        # ── 顶栏: 阶段指示器 + 状态 ──
        top_row = QHBoxLayout()
        self._phase_badge.setMinimumWidth(160)
        top_row.addWidget(self._phase_badge)
        top_row.addStretch()

        self._card_step = MetricCard("总步数", "0", "运行总步数")
        top_row.addWidget(self._card_step)
        self._card_gen = MetricCard("生成 Tokens", "0", "生成文本量")
        top_row.addWidget(self._card_gen)
        self._card_curiosity = MetricCard("好奇度", "0.00", "Curiosity 采样分数")
        top_row.addWidget(self._card_curiosity)
        outer.addLayout(top_row)

        # ── 主分割: 左配置 | 中图表 | 右文本 ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左: 配置
        cfg_group = QGroupBox("自主配置")
        cfg_form = QFormLayout(cfg_group)
        cfg_form.setSpacing(4)

        self._ckpt_edit = QLineEdit("", objectName="paramInput")
        self._ckpt_edit.setPlaceholderText("(可选) 初始检查点路径")
        cfg_form.addRow("checkpoint:", self._ckpt_edit)

        self._out_dir_edit = QLineEdit("out_autonomous", objectName="paramInput")
        cfg_form.addRow("out_dir:", self._out_dir_edit)

        self._wake_spin = _make_spin(1, 999, 20)
        cfg_form.addRow("wake_steps:", self._wake_spin)

        self._play_spin = _make_spin(1, 999, 100)
        cfg_form.addRow("play_steps:", self._play_spin)

        self._sleep_spin = _make_spin(1, 9999, 500)
        cfg_form.addRow("sleep_interval:", self._sleep_spin)

        self._batch_spin = _make_spin(1, 256, 16)
        cfg_form.addRow("batch_size:", self._batch_spin)

        self._gamma_edit = QLineEdit("0.05", objectName="paramInput")
        cfg_form.addRow("gamma:", self._gamma_edit)

        self._t_spin = _make_spin(1, 16, 1)
        cfg_form.addRow("T_infer:", self._t_spin)

        self._data_dir_edit = QLineEdit("dataset", objectName="paramInput")
        cfg_form.addRow("data_dir:", self._data_dir_edit)

        self._hs_spin = _make_spin(64, 1024, 256)
        cfg_form.addRow("hidden_size:", self._hs_spin)

        self._nl_spin = _make_spin(1, 24, 4)
        cfg_form.addRow("num_layers:", self._nl_spin)

        splitter.addWidget(cfg_group)

        # 中: 图表
        chart_panel = QWidget()
        chart_layout = QVBoxLayout(chart_panel)
        chart_layout.setContentsMargins(4, 0, 4, 0)

        self._chart_dopamine = RealtimeChart()
        self._chart_dopamine.set_title("多巴胺 D(t)")
        self._chart_dopamine.add_series("dopamine", DARK_PALETTE.accent)
        chart_layout.addWidget(self._chart_dopamine, 1)

        self._chart_curiosity = RealtimeChart()
        self._chart_curiosity.set_title("好奇度 C(t)")
        self._chart_curiosity.add_series("curiosity", "#ffd700")
        chart_layout.addWidget(self._chart_curiosity, 1)

        self._chart_loss = RealtimeChart()
        self._chart_loss.set_title("Loss")
        self._chart_loss.add_series("loss", "#ff6b6b")
        chart_layout.addWidget(self._chart_loss, 1)

        splitter.addWidget(chart_panel)

        # 右: 生成文本
        text_panel = QWidget()
        text_layout = QVBoxLayout(text_panel)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.addWidget(QLabel("生成文本预览"))
        self._gen_preview = QTextEdit()
        self._gen_preview.setReadOnly(True)
        self._gen_preview.setObjectName("logOutput")
        self._gen_preview.setPlaceholderText("自主生成的文本将显示在这里...")
        text_layout.addWidget(self._gen_preview, 1)

        splitter.addWidget(text_panel)
        splitter.setSizes([200, 300, 260])
        outer.addWidget(splitter, 1)

        # ── 控制栏 ──
        ctrl_row = QHBoxLayout()
        self._start_btn = QPushButton("▶ 启动自主运行")
        self._start_btn.clicked.connect(self._start_auto)
        ctrl_row.addWidget(self._start_btn)
        self._stop_btn = QPushButton("⏹ 停止")
        self._stop_btn.clicked.connect(self._stop_auto)
        self._stop_btn.setEnabled(False)
        ctrl_row.addWidget(self._stop_btn)
        ctrl_row.addStretch()
        outer.addLayout(ctrl_row)

        # 日志
        self._log = LogViewer()
        outer.addWidget(self._log, 2)

    def _start_auto(self) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "提示", "自主运行已在运行中")
            return

        ckpt = self._ckpt_edit.text().strip() or None
        self._worker = AutonomousWorker(
            checkpoint=ckpt,
            out_dir=self._out_dir_edit.text(),
            hidden_size=self._hs_spin.value(),
            num_layers=self._nl_spin.value(),
            wake_steps=self._wake_spin.value(),
            play_steps=self._play_spin.value(),
            sleep_interval=self._sleep_spin.value(),
            batch_size=self._batch_spin.value(),
            gamma=float(self._gamma_edit.text() or "0.05"),
            T_infer=self._t_spin.value(),
            data_dir=self._data_dir_edit.text(),
        )
        self._worker.progress.connect(self._on_auto_progress)
        self._worker.finished.connect(self._on_auto_finished)
        self._worker.log.connect(self._log.info)

        # 清空旧数据
        self._chart_dopamine.clear_series()
        self._chart_curiosity.clear_series()
        self._chart_loss.clear_series()
        self._gen_preview.clear()
        self._card_step.set_value("0")
        self._card_gen.set_value("0")
        self._card_curiosity.set_value("0.00")

        self._phase_badge.set_phase("WAKE")
        self._worker.start()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._log.info("自主运行已启动")

    def _on_auto_progress(self, data: dict) -> None:
        # 阶段
        phase = data.get("phase", "")
        if phase in PHASE_COLORS:
            self._phase_badge.set_phase(phase)

        # 步数
        step = data.get("step", 0)
        self._card_step.set_value(str(step))

        # 指标
        dopamine = data.get("dopamine", data.get("D", None))
        if dopamine is not None:
            self._chart_dopamine.append("dopamine", float(dopamine))

        curiosity = data.get("curiosity", data.get("C", None))
        if curiosity is not None:
            self._chart_curiosity.append("curiosity", float(curiosity))
            self._card_curiosity.set_value(f"{float(curiosity):.4f}")

        loss = data.get("loss", data.get("ce_loss", None))
        if loss is not None:
            self._chart_loss.append("loss", float(loss))

        # 生成文本
        gen = data.get("generation", data.get("gen_text", ""))
        if gen:
            self._gen_preview.append(f"\n── Step {step} ──\n{gen}")
            self._card_gen.set_value(str(sum(len(g) for g in [gen])))

        # 日志
        msg = data.get("message", "")
        if msg:
            self._log.info(msg)

    def _on_auto_finished(self, result: dict) -> None:
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._phase_badge.set_phase("")
        status = result.get("status", "")
        msg = result.get("message", "")
        if status == "ok":
            self._log.ok(f"✔ {msg}")
        elif status == "stopped":
            self._log.warn(f"⏹ {msg}")
        else:
            self._log.error(f"✖ {msg}")

    def _stop_auto(self) -> None:
        if self._worker:
            self._worker.requestInterruption()
            self._log.warn("⏹ 请求停止自主运行")

    def on_theme_changed(self, theme) -> None:
        self._theme = theme
