"""评估页面 — 检查点选择 + 配置 + EvalWorker + 结果展示."""

from __future__ import annotations

import os
import glob
import json

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QSplitter, QGroupBox,
                             QFormLayout, QLineEdit, QSpinBox,
                             QComboBox, QTabWidget, QTableWidget,
                             QTableWidgetItem, QTextEdit, QHeaderView,
                             QFrame, QProgressBar, QCheckBox,
                             QMessageBox, QSizePolicy)

from gui.theme import HackerTheme
from gui.widgets.log_viewer import LogViewer


def _make_spin(lo: int, hi: int, val: int, name: str = "paramSpin") -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setObjectName(name)
    return s
from gui.worker import EvalWorker


class EvaluationPage(QWidget):
    """评估 — 连接 core.evaluation.run_full_evaluation."""

    def __init__(self, bridge, theme: HackerTheme):
        super().__init__()
        self._bridge = bridge
        self._theme = theme
        self._worker: EvalWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(6)

        heading = QLabel(">>> 评估")
        heading.setObjectName("accent")
        outer.addWidget(heading)

        # ── 顶栏: 检查点 + 配置 ──
        top_row = QHBoxLayout()

        # 检查点选择
        top_row.addWidget(QLabel("检查点:"))
        self._ckpt_combo = QComboBox()
        self._ckpt_combo.setObjectName("paramInput")
        self._ckpt_combo.setMinimumWidth(260)
        self._ckpt_combo.setEditable(True)
        top_row.addWidget(self._ckpt_combo, 1)
        self._scan_ckpt_btn = QPushButton("↻ 扫描")
        self._scan_ckpt_btn.clicked.connect(self._scan_checkpoints)
        top_row.addWidget(self._scan_ckpt_btn)

        top_row.addWidget(QLabel("数据集:"))
        self._data_combo = QComboBox()
        self._data_combo.setObjectName("paramInput")
        self._data_combo.setEditable(True)
        self._data_combo.addItems([
            "dataset/sft_t2t.jsonl", "dataset/sft_t2t_mini.jsonl",
            "dataset/pretrain_t2t.jsonl", "dataset/agent_rl.jsonl",
            "dataset/dpo.jsonl",
        ])
        top_row.addWidget(self._data_combo, 1)

        outer.addLayout(top_row)

        # ── 配置行 ──
        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("hidden_size:"))
        self._hs_spin = _make_spin(64, 1024, 256)
        cfg_row.addWidget(self._hs_spin)
        cfg_row.addWidget(QLabel("layers:"))
        self._nl_spin = _make_spin(1, 24, 4)
        cfg_row.addWidget(self._nl_spin)
        cfg_row.addWidget(QLabel("max_seq_len:"))
        self._msl_spin = _make_spin(16, 2048, 128)
        cfg_row.addWidget(self._msl_spin)
        cfg_row.addWidget(QLabel("gamma:"))
        self._gamma_edit = QLineEdit("0.1", objectName="paramInput")
        self._gamma_edit.setMaxLength(12)
        cfg_row.addWidget(self._gamma_edit)
        cfg_row.addWidget(QLabel("T:"))
        self._t_spin = _make_spin(1, 64, 2)
        cfg_row.addWidget(self._t_spin)
        cfg_row.addWidget(QLabel("max_batches:"))
        self._mb_spin = _make_spin(1, 999, 20)
        cfg_row.addWidget(self._mb_spin)
        cfg_row.addStretch()
        outer.addLayout(cfg_row)

        # ── 操作行 ──
        op_row = QHBoxLayout()
        self._run_btn = QPushButton("▶ 运行评估")
        self._run_btn.clicked.connect(self._run_eval)
        self._run_btn.setObjectName("primaryBtn")
        op_row.addWidget(self._run_btn)
        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("progressBar")
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setVisible(False)
        op_row.addWidget(self._progress_bar, 1)
        outer.addLayout(op_row)

        # ── 结果 Tab ──
        self._result_tabs = QTabWidget()

        # PPL 表
        self._ppl_table = QTableWidget()
        self._ppl_table.setColumnCount(2)
        self._ppl_table.setHorizontalHeaderLabels(["模型/任务", "PPL"])
        self._ppl_table.horizontalHeader().setStretchLastSection(True)
        self._result_tabs.addTab(self._ppl_table, "PPL")

        # 生成文本
        self._gen_text = QTextEdit()
        self._gen_text.setReadOnly(True)
        self._gen_text.setObjectName("logOutput")
        self._result_tabs.addTab(self._gen_text, "生成文本")

        # 自监督
        self._ss_table = QTableWidget()
        self._ss_table.setColumnCount(2)
        self._ss_table.setHorizontalHeaderLabels(["指标", "值"])
        self._ss_table.horizontalHeader().setStretchLastSection(True)
        self._result_tabs.addTab(self._ss_table, "自监督")

        outer.addWidget(self._result_tabs, 1)

        # 日志
        self._log = LogViewer()
        outer.addWidget(self._log, 1)

        # 加载时自动扫描
        QTimer.singleShot(200, self._scan_checkpoints)

    def _scan_checkpoints(self) -> None:
        """扫描 ola_out 下所有 .pt 文件."""
        self._ckpt_combo.clear()
        base = "ola_out"
        if not os.path.isdir(base):
            self._log.warn("ola_out 目录不存在")
            return
        files = []
        for d in sorted(os.listdir(base)):
            dir_path = os.path.join(base, d)
            if os.path.isdir(dir_path):
                for f in sorted(os.listdir(dir_path)):
                    if f.endswith(".pt"):
                        files.append(os.path.join(dir_path, f))
        if not files:
            # 也搜索一级子目录
            files = sorted(glob.glob("ola_out/**/*.pt", recursive=True))
        for fp in files:
            self._ckpt_combo.addItem(fp)
        self._log.info(f"扫描到 {len(files)} 个检查点文件")

    def _run_eval(self) -> None:
        ckpt = self._ckpt_combo.currentText().strip()
        if not ckpt or not os.path.isfile(ckpt):
            QMessageBox.warning(self, "提示", "请先选择有效的检查点文件")
            return
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "提示", "评估已在运行中")
            return

        self._worker = EvalWorker(
            checkpoint_path=ckpt,
            data_path=self._data_combo.currentText(),
            hidden_size=self._hs_spin.value(),
            num_layers=self._nl_spin.value(),
            max_seq_len=self._msl_spin.value(),
            max_batches=self._mb_spin.value(),
            gamma=float(self._gamma_edit.text() or "0.1"),
            T=self._t_spin.value(),
        )
        self._worker.progress.connect(self._on_eval_progress)
        self._worker.finished.connect(self._on_eval_finished)
        self._worker.log.connect(self._log.info)

        self._run_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._log.info(f"评估启动: {ckpt}")
        self._worker.start()

    def _on_eval_progress(self, data: dict) -> None:
        msg = data.get("message", "")
        if msg:
            self._log.info(msg)

    def _on_eval_finished(self, result: dict) -> None:
        self._run_btn.setEnabled(True)
        self._progress_bar.setVisible(False)
        if result.get("status") != "ok":
            self._log.error(f"评估失败: {result.get('message', '')}")
            return

        results = result.get("results", {})
        self._log.ok("评估完成")

        # PPL
        ppl_data = results.get("results_ppl", results.get("ppl", {}))
        if isinstance(ppl_data, dict):
            self._ppl_table.setRowCount(len(ppl_data))
            for i, (k, v) in enumerate(ppl_data.items()):
                self._ppl_table.setItem(i, 0, QTableWidgetItem(str(k)))
                self._ppl_table.setItem(i, 1, QTableWidgetItem(f"{float(v):.4f}" if isinstance(v, (int, float)) else str(v)))
        elif isinstance(ppl_data, list):
            self._ppl_table.setRowCount(len(ppl_data))
            for i, v in enumerate(ppl_data):
                self._ppl_table.setItem(i, 0, QTableWidgetItem(str(i)))
                self._ppl_table.setItem(i, 1, QTableWidgetItem(f"{v:.4f}" if isinstance(v, (int, float)) else str(v)))

        # 生成文本
        gen_data = results.get("results_gen", results.get("generation", results.get("gen_text", "")))
        if isinstance(gen_data, str):
            self._gen_text.setPlainText(gen_data)
        elif isinstance(gen_data, (list, dict)):
            self._gen_text.setPlainText(json.dumps(gen_data, ensure_ascii=False, indent=2))
        elif gen_data:
            self._gen_text.setPlainText(str(gen_data))

        # 自监督
        ss_data = results.get("results_ss", results.get("self_supervised", {}))
        if isinstance(ss_data, dict):
            self._ss_table.setRowCount(len(ss_data))
            for i, (k, v) in enumerate(ss_data.items()):
                self._ss_table.setItem(i, 0, QTableWidgetItem(str(k)))
                self._ss_table.setItem(i, 1, QTableWidgetItem(f"{float(v):.6f}" if isinstance(v, (int, float)) else str(v)))
        elif isinstance(ss_data, list):
            self._ss_table.setRowCount(len(ss_data))
            for i, v in enumerate(ss_data):
                self._ss_table.setItem(i, 0, QTableWidgetItem(str(i)))
                self._ss_table.setItem(i, 1, QTableWidgetItem(f"{v:.6f}" if isinstance(v, (int, float)) else str(v)))

    def on_theme_changed(self, theme) -> None:
        self._theme = theme
