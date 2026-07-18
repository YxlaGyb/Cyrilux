"""实时曲线图 — 基于 matplotlib FigureCanvasTkAgg 嵌入 maliang Canvas."""

from __future__ import annotations

import tkinter as tk
from typing import Optional

import matplotlib
import maliang

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui.theme import ThemeManager


class RealtimeChart:
    """实时更新的 matplotlib 曲线组件 (嵌入 maliang Canvas)."""

    def __init__(self, master: maliang.Canvas,
                 position: tuple[int, int],
                 size: tuple[int, int],
                 title: str = "", ylabel: str = "",
                 max_points: int = 500,
                 theme_mgr: Optional[ThemeManager] = None):
        self._master = master
        self.title = title
        self.ylabel = ylabel
        self.max_points = max_points
        self._theme_mgr = theme_mgr

        self._series: dict[str, tuple[list[float], list[float]]] = {}
        self._colors = ["#7c5cfc", "#4ade80", "#fbbf24", "#f87171",
                        "#60a5fa", "#c084fc", "#34d399", "#fb923c"]

        p = theme_mgr.palette if theme_mgr else None
        bg = p.card_bg if p else "#ffffff"
        fg = p.fg if p else "#1a1a2e"
        grid_c = p.border if p else "#e8e8f0"
        line_c = p.border if p else "#d0d0de"

        self._frame = tk.Frame(master, bg=bg)
        master.create_window(position[0], position[1],
                             window=self._frame,
                             width=size[0], height=size[1],
                             anchor="nw")

        self._fig = Figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
        self._fig.patch.set_facecolor(bg)

        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor(bg)
        self._ax.set_title(self.title, color=fg, fontsize=9, pad=4)
        self._ax.set_ylabel(self.ylabel, color=fg, fontsize=8)
        self._ax.tick_params(colors=fg, labelsize=7)
        self._ax.grid(True, color=grid_c, linewidth=0.3)
        self._ax.spines["top"].set_visible(False)
        self._ax.spines["right"].set_visible(False)
        for spine in self._ax.spines.values():
            spine.set_color(line_c)

        self._canvas = FigureCanvasTkAgg(self._fig, master=self._frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def add_series(self, label: str):
        if label not in self._series:
            self._series[label] = ([], [])

    def add_point(self, label: str, x: float, y: float):
        if label not in self._series:
            self._series[label] = ([], [])
        xs, ys = self._series[label]
        xs.append(x)
        ys.append(y)
        if len(xs) > self.max_points:
            xs.pop(0)
            ys.pop(0)

    def clear(self):
        for label in self._series:
            self._series[label] = ([], [])
        self._redraw()

    def set_data(self, series_data: dict[str, tuple[list[float], list[float]]]):
        self._series = series_data
        self._redraw()

    def redraw(self):
        self._redraw()

    def _redraw(self):
        self._ax.clear()
        self._ax.set_title(self.title, fontsize=9, pad=4)
        self._ax.set_ylabel(self.ylabel, fontsize=8)

        p = self._theme_mgr.palette if self._theme_mgr else None
        fg = p.fg if p else "#1a1a2e"
        legend_bg = p.card_bg if p else "#ffffff"
        legend_edge = p.border if p else "#d0d0de"

        for i, (label, (xs, ys)) in enumerate(self._series.items()):
            if xs and ys:
                color = self._colors[i % len(self._colors)]
                self._ax.plot(xs, ys, label=label, color=color, linewidth=1.0, alpha=0.9)

        if self._series:
            self._ax.legend(fontsize=6, loc="upper right",
                            facecolor=legend_bg, edgecolor=legend_edge,
                            labelcolor=fg)
        self._ax.grid(True, linewidth=0.3, alpha=0.5)
        self._ax.tick_params(labelsize=7)
        self._fig.tight_layout()
        self._canvas.draw_idle()
