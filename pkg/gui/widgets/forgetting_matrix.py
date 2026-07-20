"""
遗忘矩阵热力图 — 展示 N×N 跨任务 CE 矩阵.
支持颜色映射、单元格数值、行列标签.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel


class ForgettingMatrix(QWidget):
    """N×N 热力图矩阵.

    用法:
        matrix = ForgettingMatrix(task_labels=["A", "B", "C", "D"])
        matrix.set_data([
            [2.5, 3.1, 4.0, 4.8],
            [2.6, 2.3, 3.5, 4.2],
            [2.7, 2.4, 2.2, 3.8],
            [2.8, 2.5, 2.3, 2.1],
        ])
    """

    def __init__(self, task_labels: list[str] | None = None, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setObjectName("forgettingMatrix")

        self._labels = task_labels or []
        self._data: list[list[float]] = []
        self._vmin: float = 0.0
        self._vmax: float = 5.0

    def set_data(self, matrix: list[list[float]]) -> None:
        """设置 N×N 数据并刷新."""
        self._data = matrix
        if matrix:
            vals = [v for row in matrix for v in row]
            self._vmin = min(vals) if vals else 0.0
            self._vmax = max(vals) if vals else 5.0
            if not self._labels or len(self._labels) != len(matrix):
                self._labels = [chr(65 + i) for i in range(len(matrix))]
        self.update()

    def _interpolate_color(self, t: float) -> QColor:
        """0→绿, 0.5→黄, 1→红."""
        t = max(0.0, min(1.0, t))
        if t < 0.5:
            # 绿→黄
            r = int(255 * t * 2)
            g = 255
            b = 0
        else:
            # 黄→红
            r = 255
            g = int(255 * (1 - (t - 0.5) * 2))
            b = 0
        return QColor(r, g, b)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        if w < 40 or h < 40 or not self._data:
            painter.drawText(QRectF(0, 0, w, h),
                             Qt.AlignmentFlag.AlignCenter, "无数据")
            painter.end()
            return

        n = len(self._data)
        # 留出行列标签空间
        label_w = 40
        label_h = 24
        cell_w = (w - label_w) / max(n, 1)
        cell_h = (h - label_h) / max(n, 1)
        cell_size = min(cell_w, cell_h)
        # 居中
        total_w = label_w + cell_size * n
        total_h = label_h + cell_size * n
        ox = (w - total_w) / 2
        oy = (h - total_h) / 2

        cell_font = QFont("Consolas", 8)
        label_font = QFont("Consolas", 9)

        # 绘制热力单元格
        for i in range(n):
            for j in range(n):
                x = ox + label_w + j * cell_size
                y = oy + label_h + i * cell_size
                val = self._data[i][j] if i < len(self._data) and j < len(self._data[i]) else 0
                t = (val - self._vmin) / max(self._vmax - self._vmin, 1e-8)
                color = self._interpolate_color(t)
                painter.fillRect(QRectF(int(x), int(y), int(cell_size), int(cell_size)), color)

                # 数值
                painter.setPen(Qt.GlobalColor.white if t > 0.4 else Qt.GlobalColor.black)
                painter.setFont(cell_font)
                painter.drawText(QRectF(int(x), int(y), int(cell_size), int(cell_size)),
                                 Qt.AlignmentFlag.AlignCenter, f"{val:.2f}")

        # 列标签 (任务名)
        painter.setPen(QColor(self.palette().text().color()))
        painter.setFont(label_font)
        for j in range(n):
            x = ox + label_w + j * cell_size
            lbl = self._labels[j] if j < len(self._labels) else f"T{j}"
            painter.drawText(QRectF(int(x), int(oy), int(cell_size), int(label_h)),
                             Qt.AlignmentFlag.AlignCenter, lbl)

        # 行标签
        for i in range(n):
            y = oy + label_h + i * cell_size
            lbl = self._labels[i] if i < len(self._labels) else f"T{i}"
            painter.drawText(QRectF(int(ox), int(y), int(label_w), int(cell_size)),
                             Qt.AlignmentFlag.AlignCenter, lbl)

        painter.end()
