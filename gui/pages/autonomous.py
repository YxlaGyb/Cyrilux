"""自主运行页 — WAKE/PLAY/SLEEP 阶段监控."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog

import maliang

from gui.theme import FONT_FAMILY
from gui.state import ExperimentState
from gui.widgets.log_viewer import LogViewer
from gui.widgets.realtime_chart import RealtimeChart


class Page(maliang.Canvas):
    """自主运行页面 (maliang Canvas)."""

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
        inner.grid_rowconfigure(1, weight=0)  # config
        inner.grid_rowconfigure(2, weight=1)  # status + text
        inner.grid_rowconfigure(3, weight=0)  # log
        inner.grid_columnconfigure(0, weight=1)

        # 标题
        tk.Label(inner, text="自主运行", font=(FONT_FAMILY, 16, "bold"),
                 bg=p.bg, fg=p.fg, anchor="w").grid(row=0, column=0, sticky="ew",
                                                      padx=20, pady=(16, 8))

        # ── 配置 ──
        cfg = tk.LabelFrame(inner, text="运行配置", font=(FONT_FAMILY, 10, "bold"),
                            bg=p.bg, fg=p.fg, padx=8, pady=4)
        cfg.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 8))

        row1 = tk.Frame(cfg, bg=p.bg)
        row1.pack(fill=tk.X, pady=2)
        fields = [("Wake 步数:", "500"), ("Play 步数:", "500"), ("Sleep 步数:", "500"),
                  ("温度:", "0.8"), ("Top-K:", "40"), ("Max New:", "128")]
        for i, (label, default) in enumerate(fields):
            tk.Label(row1, text=label, font=(FONT_FAMILY, 9),
                     bg=p.bg, fg=p.fg).pack(side=tk.LEFT, padx=(8, 2))
            e = tk.Entry(row1, font=(FONT_FAMILY, 9), width=8,
                         bg=p.input_bg, fg=p.fg, relief=tk.FLAT)
            e.insert(0, default)
            e.pack(side=tk.LEFT, padx=2)

        row2 = tk.Frame(cfg, bg=p.bg)
        row2.pack(fill=tk.X, pady=2)
        tk.Label(row2, text="数据目录:", font=(FONT_FAMILY, 9),
                 bg=p.bg, fg=p.fg).pack(side=tk.LEFT)
        self._data_dir_var = tk.StringVar(value="dataset")
        e = tk.Entry(row2, textvariable=self._data_dir_var, font=(FONT_FAMILY, 9),
                     width=30, bg=p.input_bg, fg=p.fg, relief=tk.FLAT)
        e.pack(side=tk.LEFT, padx=4)
        tk.Button(row2, text="浏览", command=self._browse_data,
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(side=tk.LEFT)

        tk.Label(row2, text="检查点:", font=(FONT_FAMILY, 9),
                 bg=p.bg, fg=p.fg).pack(side=tk.LEFT, padx=(16, 2))
        self._ckpt_var = tk.StringVar()
        e = tk.Entry(row2, textvariable=self._ckpt_var, font=(FONT_FAMILY, 9),
                     width=20, bg=p.input_bg, fg=p.fg, relief=tk.FLAT)
        e.pack(side=tk.LEFT, padx=4)
        tk.Button(row2, text="浏览", command=self._browse_ckpt,
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(side=tk.LEFT)

        # 控制按钮
        row3 = tk.Frame(cfg, bg=p.bg)
        row3.pack(fill=tk.X, pady=4)
        self._btn_start = tk.Button(row3, text="开始自主运行",
                                    command=self._start, font=(FONT_FAMILY, 10),
                                    bg="#4ade80", fg="black", padx=12, pady=2)
        self._btn_start.pack(side=tk.LEFT, padx=2)
        self._btn_stop = tk.Button(row3, text="停止", command=self._stop,
                                   font=(FONT_FAMILY, 10), state=tk.DISABLED,
                                   bg=p.card_bg, fg=p.fg, padx=12, pady=2)
        self._btn_stop.pack(side=tk.LEFT, padx=2)

        # ── 阶段指示器 + 生成文本 + 曲线 ──
        main = tk.Frame(inner, bg=p.bg)
        main.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 8))
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=1)

        # 阶段指示器
        stage_frame = tk.Frame(main, bg=p.bg)
        stage_frame.grid(row=0, column=0, sticky="ns", padx=(0, 8))

        self._stage_labels = {}
        for stage in ["WAKE", "PLAY", "SLEEP"]:
            lbl = tk.Label(stage_frame, text=stage, font=(FONT_FAMILY, 12, "bold"),
                           bg=p.bg, fg=p.fg_secondary, padx=12, pady=8)
            lbl.pack(pady=4)
            self._stage_labels[stage] = lbl

        # 区域: 生成文本 + 曲线
        right = tk.Frame(main, bg=p.bg)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # 文本预览
        text_frame = tk.LabelFrame(right, text="生成文本预览",
                                   font=(FONT_FAMILY, 10, "bold"),
                                   bg=p.bg, fg=p.fg, padx=4, pady=4)
        text_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        self._gen_text = tk.Text(text_frame, font=(FONT_FAMILY, 9), wrap=tk.WORD,
                                 state=tk.DISABLED, bg=p.input_bg, fg=p.fg, height=8)
        self._gen_text.pack(fill=tk.BOTH, expand=True)

        # 曲线 — 使用 maliang.Canvas 作为 RealtimeChart 的容器
        curve_frame = tk.LabelFrame(right, text="学习曲线",
                                    font=(FONT_FAMILY, 10, "bold"),
                                    bg=p.bg, fg=p.fg, padx=4, pady=4)
        curve_frame.grid(row=1, column=0, sticky="nsew")
        curve_frame.grid_rowconfigure(0, weight=1)
        curve_frame.grid_columnconfigure(0, weight=1)
        curve_canvas = maliang.Canvas(curve_frame, highlightthickness=0, bg=p.bg)
        curve_canvas.grid(row=0, column=0, sticky="nsew")
        self._chart = RealtimeChart(curve_canvas, position=(0, 0),
                                    size=(400, 120), title="Loss",
                                    ylabel="loss", max_points=200,
                                    theme_mgr=self.theme_mgr)

        # ── 日志 ──
        log_frame = tk.Frame(inner, bg=p.bg, height=120)
        log_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 8))
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_propagate(False)
        self._log_frame = log_frame
        self._log_built = False
        log_frame.bind("<Map>", self._build_logviewer)

    def _build_logviewer(self, event=None):
        if self._log_built:
            return
        self._log_built = True
        self._log_frame.update_idletasks()
        w = self._log_frame.winfo_width() or 400
        h = self._log_frame.winfo_height() or 120
        # LogViewer 需要 Canvas 作为 master, 在 Frame 内嵌一个
        log_canvas = maliang.Canvas(self._log_frame, highlightthickness=0,
                                    bg=self.theme_mgr.palette.bg)
        log_canvas.pack(fill=tk.BOTH, expand=True)
        self._log = LogViewer(log_canvas, position=(0, 0),
                              size=(w, h), theme_mgr=self.theme_mgr)

    def _browse_data(self):
        d = filedialog.askdirectory(initialdir=self._data_dir_var.get())
        if d:
            self._data_dir_var.set(d)

    def _browse_ckpt(self):
        path = filedialog.askopenfilename(
            title="选择检查点",
            filetypes=[("PyTorch", "*.pt"), ("所有文件", "*.*")],
            initialdir="ola_out",
        )
        if path:
            self._ckpt_var.set(path)

    def _start(self):
        self._btn_start.configure(state=tk.DISABLED)
        self._btn_stop.configure(state=tk.NORMAL)
        self._stage_labels["WAKE"].configure(fg="#4ade80")
        if hasattr(self, "_log") and self._log:
            self._log.info("自主运行已启动 (WAKE 阶段)")
        self.app.set_status("自主运行中")
        # TODO: 启动 autonomous_mind

    def _stop(self):
        self._btn_start.configure(state=tk.NORMAL)
        self._btn_stop.configure(state=tk.DISABLED)
        for lbl in self._stage_labels.values():
            lbl.configure(fg=self.theme_mgr.palette.fg_secondary)
        if hasattr(self, "_log") and self._log:
            self._log.info("自主运行已停止")
        self.app.set_status("就绪")
