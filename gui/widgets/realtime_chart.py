"""
实时折线图组件 — 纯 QPainter 绘制, 无额外依赖.
支持固定窗口滚动、多条曲线、网格、标签.
"""

from __future__ import annotations

import math
from typing import Sequence

from PyQt6.QtCore import Qt, QRectF, QPointF, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QFontMetrics
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel


class RealtimeChart(QWidget):
    """轻量实时折线图 — 支持 N 条曲线滚动更新.

    用法:
        chart = RealtimeChart(title="CE Loss", max_points=200)
        chart.add_series("train", color="#00ff41")
        chart.add_series("val",   color="#ffb000")
        chart.append("train", 0.5)
        chart.append("val", 0.6)
    """

    def __init__(self, title: str = "", max_points: int = 200,
                 y_range: tuple[float, float] | None = None,
                 parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.setObjectName("chartWidget")

        self._title = title
        self._max_points = max_points
        self._y_range = y_range  # (ymin, ymax), None=auto
        self._series: dict[str, dict] = {}  # name -> {color, data: []}
        self._margin = 8

    def add_series(self, name: str, color: str = "#00ff41") -> None:
        """添加一条曲线."""
        if name not in self._series:
            self._series[name] = {"color": color, "data": []}

    def append(self, name: str, value: float) -> None:
        """追加一个数据点."""
        if name not in self._series:
            self.add_series(name)
        data = self._series[name]["data"]
        data.append(value)
        if len(data) > self._max_points:
            data[:] = data[-self._max_points:]
        self.update()

    def extend(self, name: str, values: Sequence[float]) -> None:
        """批量追加."""
        if name not in self._series:
            self.add_series(name)
        data = self._series[name]["data"]
        data.extend(values)
        if len(data) > self._max_points:
            data[:] = data[-self._max_points:]
        self.update()

    def clear_series(self, name: str | None = None) -> None:
        """清空指定或全部曲线."""
        if name:
            if name in self._series:
                self._series[name]["data"].clear()
        else:
            for s in self._series.values():
                s["data"].clear()
        self.update()

    def set_title(self, title: str) -> None:
        """动态设置标题."""
        self._title = title
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        if w < 20 or h < 20:
            painter.end()
            return

        # 背景
        bg = self.palette().window().color()
        painter.fillRect(0, 0, w, h, bg)

        # 标题
        if self._title:
            painter.setPen(QColor(self.palette().text().color()))
            title_font = QFont("Consolas", 9)
            painter.setFont(title_font)
            painter.drawText(4, 4, w - 8, 16,
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             self._title)

        # 绘图区
        top = 22 if self._title else 8
        left = 8
        pw = w - 2 * left
        ph = h - top - 8
        if pw < 10 or ph < 10:
            painter.end()
            return

        # 收集所有数据找 y 范围
        all_vals = []
        for s in self._series.values():
            all_vals.extend(s["data"])
        if not all_vals:
            painter.end()
            return

        if self._y_range:
            y_min, y_max = self._y_range
        else:
            y_min = min(all_vals)
            y_max = max(all_vals)
            if y_max - y_min < 1e-8:
                y_min -= 0.1
                y_max += 0.1
            pad = (y_max - y_min) * 0.1
            y_min -= pad
            y_max += pad

        # 网格线
        grid_pen = QPen(QColor(self.palette().mid().color()), 1)
        n_grid = 4
        for i in range(n_grid + 1):
            gy = top + ph * i / n_grid
            painter.setPen(grid_pen)
            painter.drawLine(int(left), int(gy), int(left + pw), int(gy))
            # 刻度标签
            val = y_max - (y_max - y_min) * i / n_grid
            painter.setPen(QColor(self.palette().text().color()))
            label_font = QFont("Consolas", 7)
            painter.setFont(label_font)
            painter.drawText(QRectF(0, gy - 6, left - 2, 12),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             f"{val:.2f}")

        # 绘制曲线
        for s_name, s_data in self._series.items():
            data = s_data["data"]
            if len(data) < 2:
                continue
            color = QColor(s_data["color"])
            pen = QPen(color, 1.5)
            painter.setPen(pen)

            path = []
            n = len(data)
            for i, val in enumerate(data):
                x = left + pw * i / max(n - 1, 1)
                y = top + ph * (1 - (val - y_min) / (y_max - y_min))
                # 钳位到绘图区
                y = max(top, min(top + ph, y))
                path.append(QPointF(x, y))

            # 连线
            for i in range(1, len(path)):
                painter.drawLine(path[i - 1], path[i])

            # 最后点高亮
            if path:
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(path[-1], 3, 3)

        painter.end()
