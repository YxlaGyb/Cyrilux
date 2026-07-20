"""数据管理页面 — FileBrowser + JSONL 预览 + 分割/转换/任务准备."""

from __future__ import annotations

import os
import json

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QSplitter, QPlainTextEdit,
                             QGroupBox, QSpinBox, QRadioButton,
                             QFileDialog, QMessageBox)

from pkg.gui.theme import HackerTheme
from pkg.gui.widgets.file_browser import FileBrowser


class _JSONLPreview(QPlainTextEdit):
    """JSONL 内容预览."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setObjectName("logViewer")
        self.setFont(__import__("PyQt6.QtGui", fromlist=["QFont"]).QFont("Consolas", 9))
        self.setMaximumBlockCount(5000)
        self.setPlaceholderText("选择一个 .jsonl 文件预览内容...")

    def load_file(self, path: str) -> None:
        self.clear()
        if not path or not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 200:
                        self.appendPlainText("... (截断, 仅显示前 200 行)")
                        break
                    try:
                        obj = json.loads(line)
                        pretty = json.dumps(obj, ensure_ascii=False, indent=2)
                        self.appendPlainText(f"[{i}] {pretty[:500]}")
                        if len(pretty) > 500:
                            self.appendPlainText("  ... (截断)")
                    except json.JSONDecodeError:
                        self.appendPlainText(f"[{i}] {line.rstrip()}")
                    self.appendPlainText("—" * 40)
        except Exception as e:
            self.setPlainText(f"读取失败: {e}")


class DataManagerPage(QWidget):
    """数据管理 — 浏览/预览/分割/转换/任务准备."""

    def __init__(self, bridge, theme: HackerTheme):
        super().__init__()
        self._bridge = bridge
        self._theme = theme
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(8)

        heading = QLabel(">>> 数据管理")
        heading.setObjectName("accent")
        outer.addWidget(heading)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左: 文件浏览器
        fb_wrap = QWidget()
        fb_layout = QVBoxLayout(fb_wrap)
        fb_layout.setContentsMargins(0, 0, 0, 0)
        fb_label = QLabel("数据集目录")
        fb_label.setObjectName("secondary")
        fb_layout.addWidget(fb_label)
        self._file_browser = FileBrowser("dataset")
        fb_layout.addWidget(self._file_browser, 1)
        refresh_btn = QPushButton("↻ 刷新")
        refresh_btn.clicked.connect(self._refresh_browser)
        fb_layout.addWidget(refresh_btn)
        splitter.addWidget(fb_wrap)

        # 右: 预览 + 工具
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(8)

        preview_label = QLabel("内容预览")
        preview_label.setObjectName("secondary")
        right_layout.addWidget(preview_label)
        self._preview = _JSONLPreview()
        right_layout.addWidget(self._preview, 3)

        # 工具面板
        tool_group = QGroupBox("工具")
        tool_layout = QVBoxLayout(tool_group)

        split_row = QHBoxLayout()
        split_row.addWidget(QLabel("分割大小:"))
        self._split_size = QSpinBox()
        self._split_size.setRange(100, 100000)
        self._split_size.setValue(10000)
        self._split_size.setObjectName("paramSpin")
        split_row.addWidget(self._split_size)
        self._split_btn = QPushButton("分割文件")
        self._split_btn.clicked.connect(self._do_split)
        split_row.addWidget(self._split_btn)
        tool_layout.addLayout(split_row)

        convert_row = QHBoxLayout()
        self._convert_btn = QPushButton("转换文件 (jsonl→text)")
        self._convert_btn.clicked.connect(self._do_convert)
        convert_row.addWidget(self._convert_btn)
        tool_layout.addLayout(convert_row)

        prep_row = QHBoxLayout()
        prep_row.addWidget(QLabel("准备任务:"))
        self._prep_4task = QRadioButton("4-Task")
        self._prep_4task.setChecked(True)
        self._prep_hetero = QRadioButton("Hetero (5-Task)")
        prep_row.addWidget(self._prep_4task)
        prep_row.addWidget(self._prep_hetero)
        self._prep_btn = QPushButton("运行")
        self._prep_btn.clicked.connect(self._do_prepare)
        prep_row.addWidget(self._prep_btn)
        prep_row.addStretch()
        tool_layout.addLayout(prep_row)

        right_layout.addWidget(tool_group, 0)
        splitter.addWidget(right_panel)
        splitter.setSizes([280, 520])
        outer.addWidget(splitter, 1)

        self._file_browser._tree.itemClicked.connect(self._on_file_clicked)

    def _refresh_browser(self) -> None:
        self._file_browser.load_directory("dataset")

    def _on_file_clicked(self, item, column) -> None:
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path and path.endswith(".jsonl"):
            self._preview.load_file(path)

    def _do_split(self) -> None:
        selected = self._file_browser.selected_paths()
        jsonl_files = [p for p in selected if p.endswith(".jsonl")]
        if not jsonl_files:
            QMessageBox.information(self, "提示", "请先选择一个 .jsonl 文件")
            return
        from pkg.utils.data_splitter import split_file
        path = jsonl_files[0]
        chunk_size = self._split_size.value()
        try:
            result = split_file(path, chunk_size=chunk_size)
            QMessageBox.information(
                self, "分割完成",
                f"文件: {os.path.basename(path)}\n"
                f"分块: {result['n_chunks']}\n"
                f"总行数: {result['total_lines']}\n"
                f"输出: {result['output_files']}")
        except Exception as e:
            QMessageBox.critical(self, "分割失败", str(e))

    def _do_convert(self) -> None:
        selected = self._file_browser.selected_paths()
        jsonl_files = [p for p in selected if p.endswith(".jsonl")]
        if not jsonl_files:
            QMessageBox.information(self, "提示", "请先选择一个 .jsonl 文件")
            return
        from pkg.utils.data_converter import convert_file
        path = jsonl_files[0]
        try:
            out_path = convert_file(path)
            QMessageBox.information(self, "转换完成", f"输出: {out_path}")
        except Exception as e:
            QMessageBox.critical(self, "转换失败", str(e))

    def _do_prepare(self) -> None:
        from pkg.utils.prepare_tasks import prepare_4tasks, prepare_hetero
        try:
            if self._prep_4task.isChecked():
                result = prepare_4tasks()
                label = "4-Task"
            else:
                result = prepare_hetero()
                label = "Hetero"
            summary = "\n".join(f"  {k}: {v}" for k, v in result.items())
            QMessageBox.information(
                self, f"{label} 完成", f"任务准备结果:\n{summary}")
        except Exception as e:
            QMessageBox.critical(self, "任务准备失败", f"{type(e).__name__}: {e}")

    def on_theme_changed(self, theme) -> None:
        self._theme = theme
