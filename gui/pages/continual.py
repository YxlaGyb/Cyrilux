"""持续学习页面 — 任务管线 + 遗忘矩阵 + Memory Bank."""

from __future__ import annotations

import os
import json

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QSplitter, QTreeWidget,
                             QTreeWidgetItem, QFrame, QGroupBox,
                             QListWidget, QListWidgetItem, QMessageBox,
                             QInputDialog, QSizePolicy)

from gui.theme import HackerTheme
from gui.widgets.forgetting_matrix import ForgettingMatrix
from gui.widgets.log_viewer import LogViewer
from gui.widgets.metric_card import MetricCard


TASK_COLORS = ["#00ff41", "#ffd700", "#00bfff", "#ff6b6b", "#da70d6"]


class ContinualPage(QWidget):
    """持续学习 — 任务管线 + 遗忘热力图 + 记忆库."""

    def __init__(self, bridge, theme: HackerTheme):
        super().__init__()
        self._bridge = bridge
        self._theme = theme
        self._tasks: list[dict] = []
        self._build_ui()
        self._load_forgetting_data()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(6)

        heading = QLabel(">>> 持续学习")
        heading.setObjectName("accent")
        outer.addWidget(heading)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── 左: 任务管线 ──
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 4, 0)

        left_layout.addWidget(QLabel("任务管线"))
        self._task_tree = QTreeWidget()
        self._task_tree.setHeaderLabel("任务 / 数据集 / 检查点")
        self._task_tree.setAlternatingRowColors(False)
        self._task_tree.setDragDropMode(
            QTreeWidget.DragDropMode.InternalMove)
        left_layout.addWidget(self._task_tree, 1)

        btn_row = QHBoxLayout()
        self._add_task_btn = QPushButton("+ 添加任务")
        self._add_task_btn.clicked.connect(self._add_task)
        btn_row.addWidget(self._add_task_btn)
        self._remove_task_btn = QPushButton("− 删除")
        self._remove_task_btn.clicked.connect(self._remove_task)
        btn_row.addWidget(self._remove_task_btn)
        self._scan_btn = QPushButton("↻ 扫描遗忘")
        self._scan_btn.clicked.connect(self._load_forgetting_data)
        btn_row.addWidget(self._scan_btn)
        left_layout.addLayout(btn_row)

        splitter.addWidget(left_panel)

        # ── 中: 遗忘矩阵 + 状态卡 ──
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(4, 0, 4, 0)

        center_layout.addWidget(QLabel("遗忘矩阵 (Forgetting Heatmap)"))
        self._fmat = ForgettingMatrix()
        self._fmat.setMinimumWidth(280)
        self._fmat.setMinimumHeight(280)
        center_layout.addWidget(self._fmat, 2)

        # 状态卡片
        card_row = QHBoxLayout()
        self._card_mem = MetricCard("Memory Bank", "0", "存储的 exemplar 数量")
        card_row.addWidget(self._card_mem)
        self._card_abs = MetricCard("Abstraction Bank", "0", "抽象规则数")
        card_row.addWidget(self._card_abs)
        self._card_sniff = MetricCard("遗忘嗅探", "待命", "最后扫描结果")
        card_row.addWidget(self._card_sniff)
        center_layout.addLayout(card_row)

        splitter.addWidget(center_panel)

        # ── 右: 详情日志 ──
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.addWidget(QLabel("遗忘日志"))
        self._log_viewer = LogViewer()
        right_layout.addWidget(self._log_viewer, 1)

        # 快速查看遗忘日志文件
        self._view_flog_btn = QPushButton("📄 读取遗忘日志文件")
        self._view_flog_btn.clicked.connect(self._load_forgetting_from_log)
        right_layout.addWidget(self._view_flog_btn)

        splitter.addWidget(right_panel)
        splitter.setSizes([240, 360, 200])
        outer.addWidget(splitter, 1)

    def _add_task(self) -> None:
        name, ok = QInputDialog.getText(self, "添加任务", "任务名称:")
        if not ok or not name.strip():
            return
        ds, ok2 = QInputDialog.getText(self, "数据集", "数据集路径:", text="dataset/sft_t2t.jsonl")
        if not ok2:
            return
        item = QTreeWidgetItem([name])
        ds_item = QTreeWidgetItem([f"📄 {ds}"])
        item.addChild(ds_item)
        self._task_tree.addTopLevelItem(item)
        self._task_tree.expandAll()
        self._tasks.append({"name": name, "dataset": ds})
        self._log_viewer.info(f"添加任务: {name} ({ds})")

    def _remove_task(self) -> None:
        item = self._task_tree.currentItem()
        if item is None:
            return
        parent = item.parent()
        top = parent if parent else item
        root = self._task_tree.invisibleRootItem()
        for i in range(root.childCount()):
            if root.child(i) is top:
                name = top.text(0)
                root.removeChild(top)
                self._tasks = [t for t in self._tasks if t["name"] != name]
                self._log_viewer.info(f"删除任务: {name}")
                return

    def _find_forgetting_log(self) -> str | None:
        """在 ola_out 各子目录中查找 forgetting_log.json."""
        base = "ola_out"
        if not os.path.isdir(base):
            return None
        for d in sorted(os.listdir(base)):
            fp = os.path.join(base, d, "forgetting_log.json")
            if os.path.isfile(fp):
                return fp
        return None

    def _load_forgetting_data(self) -> None:
        fp = self._find_forgetting_log()
        if fp is None:
            self._log_viewer.warn("未找到 forgetting_log.json")
            self._card_sniff.set_value("无日志")
            return
        self._load_forgetting_from_file(fp)

    def _load_forgetting_from_log(self) -> None:
        fp = self._find_forgetting_log()
        if fp:
            self._load_forgetting_from_file(fp)
        else:
            QMessageBox.information(self, "提示", "未找到 forgetting_log.json")

    def _load_forgetting_from_file(self, fp: str) -> None:
        try:
            with open(fp, encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            self._log_viewer.error(f"读取遗忘日志失败: {e}")
            return

        # 兼容顶层为 list 或 dict
        if isinstance(raw, dict):
            data = raw
        elif isinstance(raw, list):
            if raw and isinstance(raw[0], (int, float)):
                data = {"matrix": [raw]}
            elif raw and isinstance(raw[-1], dict):
                data = raw[-1]
            else:
                data = {"matrix": raw}
        else:
            data = {}

        matrix = data.get("matrix", data.get("forgetting_matrix", []))
        if not matrix and "history" in data:
            history = data["history"]
            n = len(history)
            if n > 0:
                matrix = [[0.0] * n for _ in range(n)]
                for i, h in enumerate(history):
                    for j, val in enumerate(history):
                        matrix[i][j] = abs(h.get(str(j), h.get(j, 0)))

        labels = data.get("tasks", data.get("labels", []))
        if not labels and matrix:
            labels = [f"Task {i}" for i in range(len(matrix))]

        if matrix:
            self._fmat.set_data(matrix, labels)
            self._log_viewer.ok(f"遗忘矩阵: {len(matrix)}×{len(matrix[0]) if isinstance(matrix[0], list) else 0}")

        bank_size = data.get("bank_size", data.get("memory_bank", 0))
        self._card_mem.set_value(str(bank_size))
        abs_size = data.get("abstraction_size", data.get("abstraction_bank", 0))
        self._card_abs.set_value(str(abs_size))
        sniff = data.get("sniffer", data.get("anomaly_count", "待命"))
        self._card_sniff.set_value(str(sniff))

        self._log_viewer.info(f"遗忘日志来源: {fp}")

    def on_theme_changed(self, theme) -> None:
        self._theme = theme
