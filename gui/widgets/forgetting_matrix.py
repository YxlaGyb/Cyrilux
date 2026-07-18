"""遗忘矩阵热力图 — 基于 matplotlib 嵌入 maliang Canvas."""

from __future__ import annotations

import tkinter as tk
from typing import Optional

import matplotlib
import maliang

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui.theme import ThemeManager


class ForgettingMatrix:
    """遗忘矩阵热力图 (嵌入 maliang Canvas)."""

    def __init__(self, master: maliang.Canvas,
                 position: tuple[int, int],
                 size: tuple[int, int],
                 data: Optional[list] = None,
                 theme_mgr: Optional[ThemeManager] = None):
        self._master = master
        self.data = data or []
        self._theme_mgr = theme_mgr
        self._metric = "ce"
        self._position = position
        self._size = size

        p = theme_mgr.palette if theme_mgr else None
        bg = p.card_bg if p else "#ffffff"
        fg = p.fg if p else "#1a1a2e"

        self._frame = tk.Frame(master, bg=bg)
        master.create_window(position[0], position[1],
                             window=self._frame,
                             width=size[0], height=size[1],
                             anchor="nw")

        self._fig = Figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
        self._fig.patch.set_facecolor(bg)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor(bg)

        self._canvas = FigureCanvasTkAgg(self._fig, master=self._frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 指标切换按钮
        toolbar = tk.Frame(self._frame, bg=bg)
        toolbar.pack(fill=tk.X, pady=(2, 0))
        self._ce_btn = tk.Button(toolbar, text="CE", bg=bg, fg=fg,
                                 command=lambda: self._switch("ce"))
        self._ce_btn.pack(side=tk.LEFT, padx=2)
        self._ppl_btn = tk.Button(toolbar, text="PPL", bg=bg, fg=fg,
                                  command=lambda: self._switch("ppl"))
        self._ppl_btn.pack(side=tk.LEFT, padx=2)

        if self.data:
            self._render()

    def _switch(self, metric: str):
        self._metric = metric
        self._render()

    def set_data(self, data: list):
        self.data = data
        self._render()

    def _render(self):
        p = self._theme_mgr.palette if self._theme_mgr else None
        fg = p.fg if p else "#1a1a2e"
        fg2 = p.fg_secondary if p else "#6b6b8a"
        bg = p.card_bg if p else "#ffffff"

        self._ax.clear()

        if not self.data:
            self._ax.text(0.5, 0.5, "暂无数据", ha="center", va="center",
                          color=fg2, fontsize=12)
            self._canvas.draw_idle()
            return

        import numpy as np
        n_tasks = len(self.data)
        task_ids = sorted({k for d in self.data for k in d.get("results", {})},
                          key=lambda x: (int(x) if isinstance(x, str) and x.isdigit() else x))
        n_cols = len(task_ids)

        matrix = np.full((n_tasks, n_cols), np.nan)
        for i, d in enumerate(self.data):
            for j, tid in enumerate(task_ids):
                entry = d.get("results", {}).get(tid, {})
                val = entry.get(self._metric, np.nan)
                if val is not None:
                    matrix[i, j] = val

        cmap = "YlOrRd" if self._metric == "ce" else "YlGnBu"
        im = self._ax.imshow(matrix, cmap=cmap, aspect="auto", interpolation="nearest")

        self._ax.set_xticks(range(n_cols))
        self._ax.set_xticklabels([f"T{tid}" for tid in task_ids], color=fg, fontsize=8)
        self._ax.set_yticks(range(n_tasks))
        self._ax.set_yticklabels([f"After T{d['after_task']}" for d in self.data],
                                 color=fg, fontsize=8)
        self._ax.set_xlabel("评估任务", color=fg2, fontsize=9)
        self._ax.set_ylabel("训练后", color=fg2, fontsize=9)

        for i in range(n_tasks):
            for j in range(n_cols):
                val = matrix[i, j]
                if not np.isnan(val):
                    mean_val = matrix[~np.isnan(matrix)].mean() if np.any(~np.isnan(matrix)) else 0
                    text_color = "white" if val > mean_val else "black"
                    self._ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                  color=text_color, fontsize=7)

        metric_label = "Cross-Entropy" if self._metric == "ce" else "Perplexity"
        self._ax.set_title(f"遗忘矩阵 ({metric_label})", color=fg, fontsize=10)

        cbar = self._fig.colorbar(im, ax=self._ax, shrink=0.75)
        cbar.ax.yaxis.set_tick_params(color=fg, labelsize=7)
        for t in cbar.ax.get_yticklabels():
            t.set_color(fg)

        self._fig.tight_layout()
        self._canvas.draw_idle()
