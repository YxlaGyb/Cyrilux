"""评估页 — 加载模型 + PPL / 生成 / 自监督评估."""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk, filedialog

import maliang

from gui.theme import FONT_FAMILY
from gui.state import ExperimentState


class Page(maliang.Canvas):
    """模型评估页面 (maliang Canvas)."""

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
        inner.grid_rowconfigure(1, weight=0)  # load model
        inner.grid_rowconfigure(2, weight=0)  # eval config
        inner.grid_rowconfigure(3, weight=1)  # results
        inner.grid_columnconfigure(0, weight=1)

        # 标题
        tk.Label(inner, text="评估", font=(FONT_FAMILY, 16, "bold"),
                 bg=p.bg, fg=p.fg, anchor="w").grid(row=0, column=0, sticky="ew",
                                                      padx=20, pady=(16, 8))

        # ── 加载模型 ──
        load_frame = tk.Frame(inner, bg=p.bg)
        load_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 8))

        tk.Label(load_frame, text="检查点:", font=(FONT_FAMILY, 9),
                 bg=p.bg, fg=p.fg).pack(side=tk.LEFT)
        self._ckpt_var = tk.StringVar()
        e = tk.Entry(load_frame, textvariable=self._ckpt_var, font=(FONT_FAMILY, 9),
                     width=50, bg=p.input_bg, fg=p.fg, relief=tk.FLAT)
        e.pack(side=tk.LEFT, padx=4)
        tk.Button(load_frame, text="浏览", command=self._browse_ckpt,
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(side=tk.LEFT, padx=2)

        # 评估配置
        cfg_frame = tk.LabelFrame(inner, text="评估配置", font=(FONT_FAMILY, 10, "bold"),
                                  bg=p.bg, fg=p.fg, padx=8, pady=4)
        cfg_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 8))

        row1 = tk.Frame(cfg_frame, bg=p.bg)
        row1.pack(fill=tk.X, pady=2)
        tk.Label(row1, text="数据:", font=(FONT_FAMILY, 9),
                 bg=p.bg, fg=p.fg).pack(side=tk.LEFT)
        self._eval_data_var = tk.StringVar(value="dataset/sft_t2t_mini.jsonl")
        e = tk.Entry(row1, textvariable=self._eval_data_var, font=(FONT_FAMILY, 9),
                     width=40, bg=p.input_bg, fg=p.fg, relief=tk.FLAT)
        e.pack(side=tk.LEFT, padx=4)
        tk.Button(row1, text="浏览", command=lambda: self._browse("eval_data"),
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(side=tk.LEFT)

        row2 = tk.Frame(cfg_frame, bg=p.bg)
        row2.pack(fill=tk.X, pady=2)
        for label, key, default in [("T:", "T_infer", "2"), ("Gamma:", "gamma", "0.1"),
                                     ("Max batches:", "", ""), ("Max samples:", "", "")]:
            tk.Label(row2, text=label, font=(FONT_FAMILY, 9),
                     bg=p.bg, fg=p.fg).pack(side=tk.LEFT, padx=(8, 2))
            e = tk.Entry(row2, font=(FONT_FAMILY, 9), width=8,
                         bg=p.input_bg, fg=p.fg, relief=tk.FLAT)
            e.insert(0, default)
            e.pack(side=tk.LEFT, padx=2)

        # Prompt 列表
        row3 = tk.Frame(cfg_frame, bg=p.bg)
        row3.pack(fill=tk.X, pady=2)
        tk.Label(row3, text="Prompts (每行一个):", font=(FONT_FAMILY, 9),
                 bg=p.bg, fg=p.fg).pack(side=tk.LEFT)
        self._prompt_text = tk.Text(row3, height=3, font=(FONT_FAMILY, 9),
                                    width=60, bg=p.input_bg, fg=p.fg, relief=tk.FLAT)
        self._prompt_text.pack(side=tk.LEFT, padx=4)
        self._prompt_text.insert("1.0", "Hello\nWhat is AI?\n")

        tk.Button(cfg_frame, text="开始评估", command=self._run_eval,
                  font=(FONT_FAMILY, 10), bg="#7c5cfc", fg="white",
                  padx=12, pady=2).pack(anchor="e", pady=4)

        # ── 结果 (标签页) ──
        self._result_nb = ttk.Notebook(inner)
        self._result_nb.grid(row=3, column=0, sticky="nsew", padx=20, pady=(0, 8))

        for tab_name in ["PPL", "生成", "自监督", "汇总"]:
            frame = tk.Frame(self._result_nb, bg=p.bg)
            self._result_nb.add(frame, text=tab_name)
            tk.Label(frame, text=f"{tab_name} 结果将在此显示",
                     font=(FONT_FAMILY, 10), fg=p.fg_secondary, bg=p.bg).pack(padx=20, pady=20)

    def _browse_ckpt(self):
        path = filedialog.askopenfilename(
            title="选择检查点",
            filetypes=[("PyTorch", "*.pt"), ("所有文件", "*.*")],
            initialdir="ola_out",
        )
        if path:
            self._ckpt_var.set(path)

    def _browse(self, target):
        path = filedialog.askopenfilename(
            title="选择评估数据",
            filetypes=[("JSONL", "*.jsonl"), ("所有文件", "*.*")],
            initialdir="dataset",
        )
        if path:
            if target == "eval_data":
                self._eval_data_var.set(path)

    def _run_eval(self):
        self.app.status_message("评估功能通过 CLI 执行: python main.py eval ...")
        # TODO: 连接 CLI eval 子进程
