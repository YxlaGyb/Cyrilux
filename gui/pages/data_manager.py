"""数据管理页 — 文件浏览 / 预览 / 统计 / 工具."""

from __future__ import annotations

import os
import json
import tkinter as tk
from tkinter import ttk, filedialog

import maliang

from gui.theme import FONT_FAMILY
from gui.state import ExperimentState
from gui.worker import async_task
from gui.widgets.file_browser import FileBrowser


class Page(maliang.Canvas):
    """数据管理页面 (maliang Canvas)."""

    def __init__(self, parent, state: ExperimentState, app, **kw):
        super().__init__(parent, expand="xy", **kw)
        self.state = state
        self.app = app
        self.theme_mgr = state.theme_mgr

        p = self.theme_mgr.palette
        self._inner = tk.Frame(self, bg=p.bg)
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

        inner.grid_rowconfigure(0, weight=0)  # title
        inner.grid_rowconfigure(1, weight=1)  # browser + tabs
        inner.grid_rowconfigure(2, weight=0)  # tools
        inner.grid_columnconfigure(0, weight=1)

        # 标题
        tk.Label(inner, text="数据管理", font=(FONT_FAMILY, 16, "bold"),
                 bg=p.bg, fg=p.fg, anchor="w").grid(row=0, column=0, sticky="ew",
                                                      padx=20, pady=(16, 8))

        # ── 文件浏览器 + 预览/统计 ──
        main = tk.Frame(inner, bg=p.bg)
        main.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 8))
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=1)

        # 文件浏览器 (外层 LabelFrame 内嵌 Canvas 容器)
        browser_frame = tk.LabelFrame(main, text="文件", font=(FONT_FAMILY, 10, "bold"),
                                      bg=p.bg, fg=p.fg, padx=4, pady=4)
        browser_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        browser_frame.grid_rowconfigure(0, weight=1)
        browser_frame.grid_columnconfigure(0, weight=1)

        browser_canvas = tk.Canvas(browser_frame, highlightthickness=0, bg=p.bg)
        browser_canvas.grid(row=0, column=0, sticky="nsew")
        self._browser_canvas = browser_canvas
        self._browser_built = False
        browser_canvas.bind("<Map>", lambda e: self._build_browser(browser_canvas))

        # 右侧标签页
        nb = ttk.Notebook(main)
        nb.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        # 预览
        preview_frame = tk.Frame(nb, bg=p.bg)
        nb.add(preview_frame, text="预览")
        self._preview_text = tk.Text(preview_frame, font=(FONT_FAMILY, 9),
                                     wrap=tk.WORD, state=tk.DISABLED,
                                     bg=p.input_bg, fg=p.fg, relief=tk.FLAT)
        self._preview_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # 统计
        stats_frame = tk.Frame(nb, bg=p.bg)
        nb.add(stats_frame, text="统计")

        self._stats_tree = ttk.Treeview(stats_frame, columns=("key", "value"),
                                        show="headings", height=8)
        self._stats_tree.heading("key", text="指标")
        self._stats_tree.heading("value", text="数值")
        self._stats_tree.column("key", width=100)
        self._stats_tree.column("value", width=100, anchor="e")
        self._stats_tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ── 工具面板 ──
        tools = tk.LabelFrame(inner, text="工具", font=(FONT_FAMILY, 10, "bold"),
                              bg=p.bg, fg=p.fg, padx=8, pady=4)
        tools.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 8))

        # 数据分割
        split_frame = tk.Frame(tools, bg=p.bg)
        split_frame.pack(fill=tk.X, pady=2)
        tk.Label(split_frame, text="数据分割:", font=(FONT_FAMILY, 9),
                 bg=p.bg, fg=p.fg).pack(side=tk.LEFT)
        tk.Label(split_frame, text="Train:", font=(FONT_FAMILY, 9),
                 bg=p.bg, fg=p.fg).pack(side=tk.LEFT, padx=(8, 2))
        self._split_train = tk.Entry(split_frame, width=6,
                                     bg=p.input_bg, fg=p.fg, relief=tk.FLAT)
        self._split_train.insert(0, "0.8")
        self._split_train.pack(side=tk.LEFT)
        tk.Label(split_frame, text="Val:", font=(FONT_FAMILY, 9),
                 bg=p.bg, fg=p.fg).pack(side=tk.LEFT, padx=(8, 2))
        self._split_val = tk.Entry(split_frame, width=6,
                                   bg=p.input_bg, fg=p.fg, relief=tk.FLAT)
        self._split_val.insert(0, "0.1")
        self._split_val.pack(side=tk.LEFT)
        tk.Label(split_frame, text="Test:", font=(FONT_FAMILY, 9),
                 bg=p.bg, fg=p.fg).pack(side=tk.LEFT, padx=(8, 2))
        self._split_test = tk.Entry(split_frame, width=6,
                                    bg=p.input_bg, fg=p.fg, relief=tk.FLAT)
        self._split_test.insert(0, "0.1")
        self._split_test.pack(side=tk.LEFT)
        tk.Button(split_frame, text="执行分割", command=self._run_split,
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(side=tk.LEFT, padx=8)

        # 格式转换 / 任务准备
        btn_frame = tk.Frame(tools, bg=p.bg)
        btn_frame.pack(fill=tk.X, pady=2)
        tk.Button(btn_frame, text="格式转换", command=self._run_convert,
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="任务准备 (4task)", command=lambda: self._run_prepare("4task"),
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="任务准备 (Hetero)", command=lambda: self._run_prepare("hetero"),
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(side=tk.LEFT, padx=2)

    def _build_browser(self, canvas: tk.Canvas):
        if self._browser_built:
            return
        self._browser_built = True
        canvas.update_idletasks()
        w = canvas.winfo_width() or 300
        h = canvas.winfo_height() or 200
        self._browser = FileBrowser(canvas, position=(0, 0),
                                    size=(w, h), directory="dataset",
                                    on_select=self._on_file_select,
                                    theme_mgr=self.theme_mgr)

    def _on_file_select(self, entry: dict):
        """文件选中回调 — 异步预览/统计."""
        path = entry.get("path", "")

        # 显示加载中
        self._preview_text.configure(state=tk.NORMAL)
        self._preview_text.delete("1.0", tk.END)
        self._preview_text.insert(tk.END, "加载中...")
        self._preview_text.configure(state=tk.DISABLED)

        for row in self._stats_tree.get_children():
            self._stats_tree.delete(row)

        async_task(self, worker=lambda: self._preview_worker(path),
                   on_done=self._preview_done,
                   on_error=self._preview_error)

    def _preview_worker(self, path: str) -> tuple[list[str], int, int]:
        """后台线程: 读取前5行 + 统计行数/大小."""
        preview_lines = []
        line_count = 0
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < 5:
                    try:
                        obj = json.loads(line)
                        preview_lines.append(json.dumps(obj, ensure_ascii=False)[:200])
                    except json.JSONDecodeError:
                        preview_lines.append(line[:200])
                line_count += 1
        size = os.path.getsize(path)
        return preview_lines, line_count, size

    def _preview_done(self, result: tuple[list[str], int, int]):
        """主线程: 更新预览和统计."""
        preview_lines, line_count, size = result

        self._preview_text.configure(state=tk.NORMAL)
        self._preview_text.delete("1.0", tk.END)
        for line in preview_lines:
            self._preview_text.insert(tk.END, line + "\n")
        self._preview_text.configure(state=tk.DISABLED)

        self._stats_tree.insert("", tk.END, values=("行数", str(line_count)))
        self._stats_tree.insert("", tk.END, values=("大小", f"{size / 1024:.1f} KB"))

    def _preview_error(self, exc: Exception):
        """主线程: 显示错误."""
        self._preview_text.configure(state=tk.NORMAL)
        self._preview_text.delete("1.0", tk.END)
        self._preview_text.insert(tk.END, f"无法读取: {exc}")
        self._preview_text.configure(state=tk.DISABLED)

    def _run_split(self):
        self.app.status_message("数据分割通过 CLI 执行: python main.py data split ...")

    def _run_convert(self):
        self.app.status_message("格式转换通过 CLI 执行: python main.py data convert ...")

    def _run_prepare(self, mode: str):
        self.app.status_message(f"任务准备 ({mode}) 通过 CLI 执行: python main.py prepare ...")
