"""
骇客像素风 Metric 卡片 — 带标签/值/可选的趋势指示.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel


class MetricCard(QFrame):
    """数值概览卡片: 标签 + 大号值 + 可选脚注."""

    def __init__(self, label: str, value: str = "—", footnote: str = "",
                 parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)
        self.setMinimumHeight(80)
        self.setObjectName("metricCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        self._label = QLabel(label)
        self._label.setObjectName("secondary")
        layout.addWidget(self._label)

        self._value = QLabel(value)
        self._value.setObjectName("accent")
        self._value.setStyleSheet("font-size: 18pt; font-weight: bold;")
        layout.addWidget(self._value)

        self._footnote = QLabel(footnote)
        self._footnote.setObjectName("dim")
        if not footnote:
            self._footnote.setVisible(False)
        layout.addWidget(self._footnote)

    def set_value(self, value: str) -> None:
        self._value.setText(value)

    def set_footnote(self, text: str) -> None:
        self._footnote.setText(text)
        self._footnote.setVisible(bool(text))
