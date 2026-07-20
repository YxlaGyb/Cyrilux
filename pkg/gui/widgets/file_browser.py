"""
文件浏览器 — 目录树 + 文件列表, 支持过滤和右键菜单.
"""

from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QTreeWidget, QTreeWidgetItem,
                             QMenu, QAbstractItemView, QApplication)
from PyQt6.QtGui import QAction


class FileBrowser(QWidget):
    """文件/目录浏览器.

    用法:
        fb = FileBrowser()
        fb.load_directory("dataset")  # 加载目录
        fb.set_filter("*.jsonl")       # 只显示 jsonl
        fb.file_selected.connect(callback)
    """

    def __init__(self, root_path: str = ".", parent=None):
        super().__init__(parent)
        self._root_path = root_path
        self._filter = "*"
        self._show_hidden = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # 路径栏
        path_row = QHBoxLayout()
        self._path_label = QLabel(root_path)
        self._path_label.setObjectName("secondary")
        path_row.addWidget(self._path_label, 1)
        layout.addLayout(path_row)

        # 树
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["名称", "大小", "修改时间"])
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._tree.setColumnWidth(0, 240)
        self._tree.setColumnWidth(1, 80)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)
        # 双击展开目录
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self._tree, 1)

    def load_directory(self, path: str) -> None:
        """加载指定目录."""
        self._root_path = path
        self._path_label.setText(os.path.abspath(path))
        self._tree.clear()
        if not os.path.isdir(path):
            return
        try:
            items = sorted(os.listdir(path))
        except PermissionError:
            return
        for name in items:
            if not self._show_hidden and name.startswith("."):
                continue
            full = os.path.join(path, name)
            if os.path.isdir(full):
                item = QTreeWidgetItem([name + "/", "", ""])
                item.setData(0, Qt.ItemDataRole.UserRole, full)
                item.setChildIndicatorPolicy(
                    QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
            elif self._filter == "*" or name.endswith(self._filter.replace("*", "")):
                size = self._format_size(os.path.getsize(full))
                mtime = self._format_time(os.path.getmtime(full))
                item = QTreeWidgetItem([name, size, mtime])
                item.setData(0, Qt.ItemDataRole.UserRole, full)
            else:
                continue
            self._tree.addTopLevelItem(item)

    def set_filter(self, pattern: str) -> None:
        """设置文件过滤器 (如 *.jsonl, *.pt)."""
        self._filter = pattern
        self.load_directory(self._root_path)

    def selected_paths(self) -> list[str]:
        """获取当前选中项路径列表."""
        paths = []
        for item in self._tree.selectedItems():
            p = item.data(0, Qt.ItemDataRole.UserRole)
            if p:
                paths.append(p)
        return paths

    # ── 内部 ──

    def _on_item_double_clicked(self, item, column) -> None:
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path and os.path.isdir(path):
            self.load_directory(path)

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)
        # 只在选中项时显示操作
        selected = self._tree.selectedItems()
        if selected:
            copy_name = QAction("复制名称", self)
            copy_path = QAction("复制路径", self)
            copy_name.triggered.connect(lambda: self._copy_text(selected[0].text(0)))
            copy_path.triggered.connect(
                lambda: self._copy_text(selected[0].data(0, Qt.ItemDataRole.UserRole) or ""))
            menu.addAction(copy_name)
            menu.addAction(copy_path)
            menu.addSeparator()
        refresh = QAction("刷新", self)
        refresh.triggered.connect(lambda: self.load_directory(self._root_path))
        menu.addAction(refresh)
        menu.exec(self._tree.viewport().mapToGlobal(pos))

    @staticmethod
    def _copy_text(text: str) -> None:
        from PyQt6.QtCore import QMimeData
        clip = QApplication.clipboard()
        mime = QMimeData()
        mime.setText(text)
        clip.setMimeData(mime)

    @staticmethod
    def _format_size(size: int) -> str:
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size}{unit}"
            size //= 1024
        return f"{size}TB"

    @staticmethod
    def _format_time(t: float) -> str:
        from datetime import datetime
        return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
