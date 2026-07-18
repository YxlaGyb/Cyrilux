"""持续学习页 — 任务管线 + 遗忘矩阵 + 记忆浏览器."""

from __future__ import annotations

import os
import json
import tkinter as tk
from tkinter import ttk, filedialog

import maliang

from gui.theme import FONT_FAMILY
from gui.state import ExperimentState
from gui.worker import async_task
from gui.widgets.forgetting_matrix import ForgettingMatrix


class Page(maliang.Canvas):
    """持续学习页面 (maliang Canvas)."""

    def __init__(self, parent, state: ExperimentState, app, **kw):
        super().__init__(parent, expand="xy", **kw)
        self.state = state
        self.app = app
        self.theme_mgr = state.theme_mgr
        self._forgetting_data = []

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
        inner.grid_rowconfigure(1, weight=1)  # main
        inner.grid_columnconfigure(0, weight=1)
        inner.grid_columnconfigure(1, weight=1)

        # 标题
        tk.Label(inner, text="持续学习", font=(FONT_FAMILY, 16, "bold"),
                 bg=p.bg, fg=p.fg, anchor="w").grid(row=0, column=0, columnspan=2,
                                  sticky="ew", padx=20, pady=(16, 8))

        # ── 左: 任务管线 ──
        left = tk.Frame(inner, bg=p.bg)
        left.grid(row=1, column=0, sticky="nsew", padx=(20, 8), pady=(0, 8))

        tk.Label(left, text="任务管线", font=(FONT_FAMILY, 12, "bold"),
                 bg=p.bg, fg=p.fg, anchor="w").pack(fill=tk.X)

        self._task_tree = ttk.Treeview(left, columns=("path",), show="headings", height=8)
        self._task_tree.heading("path", text="数据路径")
        self._task_tree.column("path", width=280)
        self._task_tree.pack(fill=tk.BOTH, expand=True, pady=4)

        btn_row = tk.Frame(left, bg=p.bg)
        btn_row.pack(fill=tk.X, pady=2)
        tk.Button(btn_row, text="添加任务", command=self._add_task,
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="上移", command=self._move_up,
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="下移", command=self._move_down,
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="移除", command=self._remove_task,
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(side=tk.LEFT, padx=2)

        # ── 右: 遗忘矩阵 + 记忆浏览器 ──
        right = tk.Frame(inner, bg=p.bg)
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 20), pady=(0, 8))
        right.grid_rowconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # 遗忘矩阵
        matrix_frame = tk.Frame(right, bg=p.bg)
        matrix_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        matrix_frame.grid_rowconfigure(1, weight=1)
        matrix_frame.grid_columnconfigure(0, weight=1)

        header_row = tk.Frame(matrix_frame, bg=p.bg)
        header_row.grid(row=0, column=0, sticky="ew")
        tk.Label(header_row, text="遗忘矩阵", font=(FONT_FAMILY, 12, "bold"),
                 bg=p.bg, fg=p.fg, anchor="w").pack(side=tk.LEFT)
        tk.Button(header_row, text="加载 JSON", command=self._load_forgetting_json,
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(side=tk.RIGHT)

        # ForgettingMatrix 用子 Canvas 占位
        matrix_canvas = tk.Canvas(matrix_frame, highlightthickness=0, bg=p.bg)
        matrix_canvas.grid(row=1, column=0, sticky="nsew")
        self._matrix_canvas = matrix_canvas
        self._matrix_built = False
        matrix_canvas.bind("<Map>", self._build_forgetting_matrix)

        # 记忆浏览器 (标签页)
        nb = ttk.Notebook(right)
        nb.grid(row=1, column=0, sticky="nsew")

        # MemoryBank 标签
        mem_frame = tk.Frame(nb, bg=p.bg)
        nb.add(mem_frame, text="Memory Bank")
        self._mem_tree = ttk.Treeview(mem_frame, columns=("task", "count", "score"),
                                      show="headings", height=8)
        self._mem_tree.heading("task", text="任务")
        self._mem_tree.heading("count", text="Exemplar 数")
        self._mem_tree.heading("score", text="多巴胺分数")
        self._mem_tree.column("task", width=80)
        self._mem_tree.column("count", width=100, anchor="e")
        self._mem_tree.column("score", width=100, anchor="e")
        self._mem_tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # AbstractionBank 标签
        abs_frame = tk.Frame(nb, bg=p.bg)
        nb.add(abs_frame, text="Abstraction Bank")
        self._abs_label = tk.Label(abs_frame, text="原型数量: --", font=(FONT_FAMILY, 9),
                                   bg=p.bg, fg=p.fg, anchor="w")
        self._abs_label.pack(fill=tk.X, padx=8, pady=4)

        # 嗅探状态
        sniff_frame = tk.Frame(right, bg=p.bg)
        sniff_frame.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        tk.Label(sniff_frame, text="嗅探状态:", font=(FONT_FAMILY, 9),
                 bg=p.bg, fg=p.fg).pack(side=tk.LEFT)
        self._sniff_var = tk.StringVar(value="正常")
        tk.Label(sniff_frame, textvariable=self._sniff_var,
                 font=(FONT_FAMILY, 9), bg=p.bg, fg="#4ade80").pack(side=tk.LEFT, padx=4)

        # 自动加载遗忘日志 (延迟异步)
        self.after(300, self._deferred_load_forgetting)

    def _build_forgetting_matrix(self, event=None):
        if self._matrix_built:
            return
        self._matrix_built = True
        c = self._matrix_canvas
        c.update_idletasks()
        w = c.winfo_width() or 400
        h = c.winfo_height() or 200
        self._forgetting_matrix = ForgettingMatrix(
            c, position=(0, 0), size=(w, h), data=self._forgetting_data,
            theme_mgr=self.theme_mgr)

    def _add_task(self):
        path = filedialog.askopenfilename(
            title="选择数据集",
            filetypes=[("JSONL", "*.jsonl"), ("所有文件", "*.*")],
            initialdir="dataset",
        )
        if path:
            tid = f"T{self._task_tree.get_children().__len__() + 1}"
            self._task_tree.insert("", tk.END, values=(path,), iid=tid)
            self.state.task_pipelines.append((tid, path))
            self.app.status_message(f"已添加任务 {tid}")

    def _remove_task(self):
        sel = self._task_tree.selection()
        if sel:
            self._task_tree.delete(sel[0])
            self.state.task_pipelines = [(t, p) for t, p in self.state.task_pipelines if t != sel[0]]

    def _move_up(self):
        sel = self._task_tree.selection()
        if not sel:
            return
        idx = self._task_tree.index(sel[0])
        if idx > 0:
            self._task_tree.move(sel[0], "", idx - 1)

    def _move_down(self):
        sel = self._task_tree.selection()
        if not sel:
            return
        idx = self._task_tree.index(sel[0])
        children = self._task_tree.get_children()
        if idx < len(children) - 1:
            self._task_tree.move(sel[0], "", idx + 1)

    def _load_forgetting_json(self):
        path = filedialog.askopenfilename(
            title="选择遗忘日志",
            filetypes=[("JSON", "*.json"), ("所有文件", "*.*")],
            initialdir="ola_out",
        )
        if path:
            self._parse_forgetting_json(path)

    def _deferred_load_forgetting(self):
        """延迟启动异步加载遗忘日志."""
        async_task(self, worker=self._load_worker,
                   on_done=self._load_done,
                   on_error=lambda e: self.app.status_message(f"遗忘日志加载失败", 3000))

    def _load_worker(self):
        """后台线程: 查找并读取遗忘日志 JSON."""
        for candidate in [
            "ola_out/out_continual_5task/forgetting_log.json",
            "ola_out/out_hetero_pc/forgetting_log.json",
            "ola_out/out_hetero_fast/forgetting_log.json",
        ]:
            if os.path.isfile(candidate):
                with open(candidate, encoding="utf-8") as f:
                    return json.load(f)
        return None

    def _load_done(self, data):
        """主线程: 更新遗忘矩阵."""
        if data is None:
            return
        try:
            self._forgetting_data = data
            if hasattr(self, "_forgetting_matrix"):
                self._forgetting_matrix.set_data(data)
            self.app.status_message("遗忘矩阵已自动加载")
        except tk.TclError:
            pass
