"""
日志查看器 — 带颜色标签的滚动文本区.
支持按级别着色: INFO/灰色, WARN/黄, ERROR/红, 其他/绿.
"""

from __future__ import annotations

import re

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QTextCursor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPlainTextEdit


class LogViewer(QPlainTextEdit):
    """语法高亮的日志查看器.

    用法:
        log = LogViewer()
        log.info("训练开始")
        log.warn("学习率偏高")
        log.error("CUDA OOM")
    """

    _LEVEL_COLORS = {
        "INFO":    "#7a9a7a",
        "WARN":    "#ffb000",
        "WARNING": "#ffb000",
        "ERROR":   "#ff3333",
        "DEBUG":   "#3a5a3a",
        "OK":      "#00ff41",
        "DONE":    "#00ff41",
    }

    def __init__(self, max_lines: int = 10000, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setObjectName("logViewer")
        self.setMaximumBlockCount(max_lines)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setFont(__import__("PyQt6.QtGui", fromlist=["QFont"]).QFont("Consolas", 9))

    def _append_colored(self, text: str, color: str) -> None:
        self.moveCursor(QTextCursor.MoveOperation.End)
        fmt = self.currentCharFormat()
        fmt.setForeground(QColor(color))
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text + "\n", fmt)
        self.ensureCursorVisible()

    def info(self, msg: str) -> None:
        self._append_colored(f"[INFO] {msg}", self._LEVEL_COLORS["INFO"])

    def warn(self, msg: str) -> None:
        self._append_colored(f"[WARN] {msg}", self._LEVEL_COLORS["WARN"])

    def error(self, msg: str) -> None:
        self._append_colored(f"[ERROR] {msg}", self._LEVEL_COLORS["ERROR"])

    def ok(self, msg: str) -> None:
        self._append_colored(f"[OK] {msg}", self._LEVEL_COLORS["OK"])

    def write(self, msg: str) -> None:
        """兼容 callback 接口: 自动检测级别."""
        if not msg:
            return
        for level, color in self._LEVEL_COLORS.items():
            if level in msg.upper()[:20]:
                self._append_colored(msg.rstrip(), color)
                return
        self._append_colored(msg.rstrip(), "#d0e8d0")

    def clear_log(self) -> None:
        self.clear()
