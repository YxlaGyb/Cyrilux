"""检查点页面 — .pt 文件扫描 / 排序表格 / 右键操作."""

from __future__ import annotations

import os
import re

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QHeaderView, QAbstractItemView, QMenu,
                             QMessageBox, QInputDialog)

from gui_pyqt.theme import HackerTheme


def _format_size(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size //= 1024
    return f"{size}TB"


def _format_mtime(t: float) -> str:
    from datetime import datetime
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")


def _get_step(name: str) -> int:
    m = re.search(r'_s?(\d+)\.pt$', name)
    return int(m.group(1)) if m else 0


class CheckpointsPage(QWidget):
    """检查点管理 — 扫描 ola_out 目录, 展示排序表格."""

    def __init__(self, bridge, theme: HackerTheme):
        super().__init__()
        self._bridge = bridge
        self._theme = theme
        self._checkpoints: list[dict] = []
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(8)

        heading = QLabel(">>> 检查点")
        heading.setObjectName("accent")
        outer.addWidget(heading)

        toolbar = QHBoxLayout()
        self._refresh_btn = QPushButton("↻ 扫描")
        self._refresh_btn.clicked.connect(self._refresh)
        toolbar.addWidget(self._refresh_btn)

        self._out_dir_label = QLabel()
        self._out_dir_label.setObjectName("dim")
        toolbar.addWidget(self._out_dir_label, 1)
        outer.addLayout(toolbar)

        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["目录", "文件名", "Step", "大小", "修改时间"])
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_context_menu)
        self._table.setSortingEnabled(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        outer.addWidget(self._table, 1)

    def _refresh(self) -> None:
        try:
            registry = self._bridge.scan_checkpoints()
        except Exception:
            registry = self._scan_fallback()
        self._checkpoints = list(registry) if registry else []
        self._out_dir_label.setText(
            f"输出目录: {getattr(self._bridge, '_out_dir', 'ola_out')}  "
            f"共 {len(self._checkpoints)} 个检查点")
        self._populate_table()

    def _scan_fallback(self) -> list[dict]:
        results = []
        base = "ola_out"
        if not os.path.isdir(base):
            return results
        for root, dirs, files in os.walk(base):
            for fn in files:
                if fn.endswith(".pt"):
                    fp = os.path.join(root, fn)
                    try:
                        sz = os.path.getsize(fp)
                        mt = os.path.getmtime(fp)
                    except OSError:
                        continue
                    results.append({
                        "path": fp,
                        "dir": os.path.relpath(root, base),
                        "filename": fn,
                        "size": sz,
                        "step": _get_step(fn),
                        "mtime": mt,
                    })
        results.sort(key=lambda x: x["mtime"], reverse=True)
        return results

    def _populate_table(self) -> None:
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(self._checkpoints))
        for i, cp in enumerate(self._checkpoints):
            self._table.setItem(i, 0, QTableWidgetItem(cp.get("dir", "")))
            self._table.setItem(i, 1, QTableWidgetItem(cp.get("filename", "")))
            step_item = QTableWidgetItem(str(cp.get("step", 0)))
            self._table.setItem(i, 2, step_item)
            self._table.setItem(i, 3, QTableWidgetItem(_format_size(cp.get("size", 0))))
            self._table.setItem(i, 4, QTableWidgetItem(
                _format_mtime(cp.get("mtime", 0))))
            self._table.item(i, 0).setData(Qt.ItemDataRole.UserRole, cp.get("path", ""))
        self._table.setSortingEnabled(True)

    def _show_context_menu(self, pos) -> None:
        row = self._table.rowAt(pos.y())
        if row < 0:
            return
        path = self._table.item(row, 0).data(Qt.ItemDataRole.UserRole) or ""
        filename = self._table.item(row, 1).text() if self._table.item(row, 1) else ""

        menu = QMenu(self)
        reveal = QAction("在资源管理器中显示", self)
        reveal.triggered.connect(lambda: self._reveal(path))
        menu.addAction(reveal)
        copy = QAction("复制路径", self)
        copy.triggered.connect(lambda: self._copy_path(path))
        menu.addAction(copy)
        if path.endswith(".pt"):
            menu.addSeparator()
            rename = QAction("重命名...", self)
            rename.triggered.connect(lambda: self._rename_checkpoint(row, path))
            menu.addAction(rename)
            delete = QAction("删除", self)
            delete.triggered.connect(lambda: self._delete_checkpoint(row, path))
            menu.addAction(delete)
        menu.exec(self._table.viewport().mapToGlobal(pos))

    @staticmethod
    def _reveal(path: str) -> None:
        if not path:
            return
        import subprocess
        subprocess.Popen(f'explorer /select,"{os.path.abspath(path)}"')

    @staticmethod
    def _copy_path(path: str) -> None:
        from PyQt6.QtCore import QMimeData
        from PyQt6.QtWidgets import QApplication
        clip = QApplication.clipboard()
        mime = QMimeData()
        mime.setText(path)
        clip.setMimeData(mime)

    def _rename_checkpoint(self, row: int, old_path: str) -> None:
        new_name, ok = QInputDialog.getText(
            self, "重命名", "新文件名:", text=os.path.basename(old_path))
        if ok and new_name:
            new_path = os.path.join(os.path.dirname(old_path), new_name)
            try:
                os.rename(old_path, new_path)
                self._refresh()
            except OSError as e:
                QMessageBox.critical(self, "重命名失败", str(e))

    def _delete_checkpoint(self, row: int, path: str) -> None:
        reply = QMessageBox.question(
            self, "确认删除", f"确定删除 {os.path.basename(path)}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            try:
                os.remove(path)
                self._refresh()
            except OSError as e:
                QMessageBox.critical(self, "删除失败", str(e))

    def on_theme_changed(self, theme) -> None:
        self._theme = theme
