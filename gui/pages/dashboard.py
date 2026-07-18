"""控制台首页 — 基于 maliang Canvas 的项目总览 + KPI 卡片 + 快速入口."""

from __future__ import annotations

import os
import glob
import tkinter as tk
from tkinter import ttk

import maliang

from gui.theme import FONT_FAMILY, ThemeManager
from gui.state import ExperimentState
from gui.worker import async_task
from gui.widgets.metric_card import MetricCard


class Page(maliang.Canvas):
    """控制台首页 (maliang Canvas)."""

    def __init__(self, parent, state: ExperimentState, app, **kw):
        super().__init__(parent, expand="xy", **kw)
        self.state = state
        self.app = app
        self.theme_mgr = state.theme_mgr

        self._cards: dict[str, MetricCard] = {}
        self._info_labels: dict[str, tk.Label] = {}
        self._tree: Optional[ttk.Treeview] = None

        p = self.theme_mgr.palette
        self._inner = maliang.Canvas(self, expand="", bg=p.bg, highlightthickness=0)
        self._inner_id = self.create_window(0, 0, window=self._inner, anchor="nw")
        self.bind("<Configure>", self._on_resize)
        self._build()

    def _on_resize(self, event):
        try:
            self.itemconfigure(self._inner_id, width=event.width, height=event.height)
        except tk.TclError:
            pass

    def _build(self):
        inner = self._inner
        p = self.theme_mgr.palette

        # ── 标题 ──
        tk.Label(inner, text="控制台", font=(FONT_FAMILY, 16, "bold"),
                 bg=p.bg, fg=p.fg, anchor="w").pack(fill=tk.X, padx=20, pady=(16, 4))
        tk.Label(inner, text="virtuosov2 — 基于预测编码的本地动态小语言模型",
                 font=(FONT_FAMILY, 10), fg=p.fg_secondary, bg=p.bg,
                 anchor="w").pack(fill=tk.X, padx=20, pady=(0, 16))

        # ── 项目信息 ──
        info_frame = tk.Frame(inner, bg=p.bg)
        info_frame.pack(fill=tk.X, padx=20, pady=(0, 16))
        for key, default in [("框架", "PyTorch 2.12+"), ("模型架构", "Predictive Coding"),
                             ("Python", "3.12+"), ("目标硬件", "GTX 1650 Ti 4GB")]:
            row = tk.Frame(info_frame, bg=p.bg)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=key, font=(FONT_FAMILY, 9), width=12, anchor="w",
                     fg=p.fg_secondary, bg=p.bg).pack(side=tk.LEFT)
            lbl = tk.Label(row, text=default, font=(FONT_FAMILY, 9), anchor="w",
                           fg=p.fg, bg=p.bg)
            lbl.pack(side=tk.LEFT)
            self._info_labels[key] = lbl

        # ── KPI 卡片 (每个放在子 Canvas 上) ──
        card_frame = maliang.Canvas(inner, expand="", bg=p.bg, highlightthickness=0)
        card_frame.pack(fill=tk.X, padx=20, pady=(0, 16))

        card_data = [
            ("checkpoints", "已训练步数", "--"),
            ("outdirs", "输出目录", "--"),
            ("datasets", "数据集统计", "--"),
        ]
        for i, (key, title, val) in enumerate(card_data):
            cf = maliang.Canvas(card_frame, bg=p.bg, width=200, height=90)
            cf.pack(side=tk.LEFT, padx=(0 if i == 0 else 10, 0))
            card = MetricCard(cf, position=(0, 0), size=(200, 90),
                              title=title, value=val, theme_mgr=self.theme_mgr)
            self._cards[key] = card

        # 延迟扫描
        self.after(300, self._scan_stats)

        # ── 快速开始 ──
        tk.Label(inner, text="快速开始", font=(FONT_FAMILY, 12, "bold"),
                 bg=p.bg, fg=p.fg, anchor="w").pack(fill=tk.X, padx=20, pady=(0, 8))

        btn_frame = tk.Frame(inner, bg=p.bg)
        btn_frame.pack(fill=tk.X, padx=20)

        tk.Button(btn_frame, text="开始训练",
                  command=lambda: self.app.show_page("training"),
                  font=(FONT_FAMILY, 10), bg="#7c5cfc", fg="white",
                  padx=16, pady=4).pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(btn_frame, text="数据管理",
                  command=lambda: self.app.show_page("data"),
                  font=(FONT_FAMILY, 10), bg=p.card_bg, fg=p.fg,
                  padx=16, pady=4).pack(side=tk.LEFT, padx=8)

        tk.Button(btn_frame, text="管理检查点",
                  command=lambda: self.app.show_page("checkpoints"),
                  font=(FONT_FAMILY, 10), bg=p.card_bg, fg=p.fg,
                  padx=16, pady=4).pack(side=tk.LEFT, padx=8)

        # ── 最近检查点 ──
        recent_frame = tk.Frame(inner, bg=p.bg)
        recent_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(16, 0))

        tk.Label(recent_frame, text="最近检查点", font=(FONT_FAMILY, 12, "bold"),
                 bg=p.bg, fg=p.fg, anchor="w").pack(fill=tk.X, pady=(0, 8))

        cols = ("name", "dir", "size", "time")
        tree = ttk.Treeview(recent_frame, columns=cols, show="headings", height=6)
        tree.heading("name", text="文件名")
        tree.heading("dir", text="目录")
        tree.heading("size", text="大小")
        tree.heading("time", text="修改时间")
        tree.column("name", width=200)
        tree.column("dir", width=120)
        tree.column("size", width=80, anchor="e")
        tree.column("time", width=140)
        tree.pack(fill=tk.BOTH, expand=True)
        self._tree = tree

        # 填充最近检查点
        cps = self.state.checkpoint_registry or []
        for cp in sorted(cps, key=lambda x: x.get("mtime", 0), reverse=True)[:5]:
            tree.insert("", tk.END, values=(
                cp.get("filename", ""),
                cp.get("dir", ""),
                f"{cp.get('size_kb', 0):.0f} KB",
                cp.get("mtime_str", ""),
            ))

    def _scan_stats(self):
        cps = self.state.checkpoint_registry or []
        steps = max((cp.get("step", 0) for cp in cps), default=0)
        if "checkpoints" in self._cards:
            self._cards["checkpoints"].set_value(str(steps))

        async_task(self._inner, worker=self._scan_stats_worker,
                   on_done=self._scan_stats_done)

    def _scan_stats_worker(self) -> tuple[int, int, float]:
        out_dir_count = 0
        if os.path.isdir("ola_out"):
            for d in sorted(os.listdir("ola_out")):
                if os.path.isdir(os.path.join("ola_out", d)):
                    out_dir_count += 1

        dataset_files = 0
        dataset_size = 0
        if os.path.isdir("dataset"):
            for f in glob.glob("dataset/*.jsonl"):
                dataset_files += 1
                dataset_size += os.path.getsize(f)
        return out_dir_count, dataset_files, dataset_size / (1024 * 1024)

    def _scan_stats_done(self, result: tuple[int, int, float]):
        out_dir_count, dataset_files, size_mb = result
        if "outdirs" in self._cards:
            self._cards["outdirs"].set_value(f"{out_dir_count} 目录")
        if "datasets" in self._cards:
            self._cards["datasets"].set_value(f"{dataset_files} 文件, {size_mb:.1f} MB")
