"""训练监控页 — 基于 maliang Canvas 的数据选择 / 配置预设 / 实时曲线 / 控制栏 / 日志."""

from __future__ import annotations

import os
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

import maliang

from gui.theme import FONT_FAMILY, ThemeManager
from gui.state import ExperimentState
from gui.widgets.realtime_chart import RealtimeChart
from gui.widgets.log_viewer import LogViewer

# ── 预设集 ──
QUICK_PRESETS: dict[str, dict] = {
    "tiny": {
        "batch_size": "8", "hidden_size": "128", "num_hidden_layers": "2",
        "lr": "3e-4", "epochs": "1", "max_seq_len": "64", "T_infer": "1",
        "gamma": "0.1", "max_beta": "1.0", "grad_clip": "1.0",
        "subset": "200",
    },
    "small": {
        "batch_size": "24", "hidden_size": "256", "num_hidden_layers": "4",
        "lr": "3e-4", "epochs": "1", "max_seq_len": "128", "T_infer": "2",
        "gamma": "0.1", "max_beta": "2.0", "grad_clip": "1.0",
        "subset": "0",
    },
    "medium": {
        "batch_size": "48", "hidden_size": "512", "num_hidden_layers": "6",
        "lr": "2e-4", "epochs": "2", "max_seq_len": "256", "T_infer": "3",
        "gamma": "0.05", "max_beta": "3.0", "grad_clip": "0.5",
        "subset": "0",
    },
}

# 参数分组定义 (用于折叠的"高级参数")
PARAM_GROUPS = [
    ("基础参数", [
        ("batch_size", "Batch Size"),
        ("max_seq_len", "Max Seq Len"),
        ("lr", "学习率"),
        ("epochs", "Epochs"),
        ("subset", "Subset"),
        ("seed", "Seed"),
        ("split_size", "Split Size"),
    ]),
    ("模型 / PC", [
        ("hidden_size", "Hidden Size"),
        ("num_hidden_layers", "层数"),
        ("T_infer", "T Infer"),
        ("gamma", "Gamma"),
        ("max_beta", "Max Beta"),
        ("max_beta_conv", "Max Beta Conv"),
        ("grad_clip", "Grad Clip"),
    ]),
    ("多巴胺", [
        ("dopamine_eta", "ETA"),
        ("dopamine_beta", "Beta"),
        ("dopamine_gamma", "Gamma"),
    ]),
    ("持续学习", [
        ("replay_ratio", "Replay Ratio"),
        ("bank_size", "Bank Size"),
        ("sniff_interval", "Sniff Interval"),
        ("repair_threshold", "Repair Threshold"),
        ("repair_steps", "Repair Steps"),
        ("eval_samples", "Eval Samples"),
        ("n_prototypes", "N Prototypes"),
        ("abstraction_replay_interval", "Abstraction Replay"),
    ]),
]


class Page(maliang.Canvas):
    """训练监控页面 (maliang Canvas)."""

    def __init__(self, parent, state: ExperimentState, app, **kw):
        super().__init__(parent, expand="xy", **kw)
        self.state = state
        self.app = app
        self.theme_mgr = state.theme_mgr

        # 训练相关
        self._trainer = None
        self._running = False
        self._paused = False

        # 延迟任务句柄
        self._after_tokens: list[str] = []

        # 数据集文件列表
        self._dataset_files: list[str] = []
        self._dataset_vars: list[tk.BooleanVar] = []

        # 高级参数展开状态
        self._advanced_visible = False
        self._param_widgets: dict[str, tuple[tk.StringVar, tk.Entry]] = {}

        # 内部布局 Frame
        p = self.theme_mgr.palette
        self._inner = tk.Frame(self, bg=p.bg)
        self._inner_id = self.create_window(0, 0, window=self._inner, anchor="nw")
        self.bind("<Configure>", self._on_resize)

        self._build()

        # 延迟扫描
        tok = self.after(200, self._scan_dataset_files)
        self._after_tokens.append(tok)
        tok = self.after(300, self._refresh_checkpoints)
        self._after_tokens.append(tok)

    def _on_resize(self, event):
        try:
            self.itemconfigure(self._inner_id, width=event.width, height=event.height)
        except tk.TclError:
            pass

    def destroy(self):
        """清理: 取消延迟任务 + 停止训练."""
        for tok in self._after_tokens:
            try:
                self.after_cancel(tok)
            except Exception:
                pass
        self._after_tokens.clear()
        if self._trainer and self._trainer.is_running():
            self._trainer.stop()
        super().destroy()

    # ═══════════════════════════════════════════════════════════════
    # UI 构建
    # ═══════════════════════════════════════════════════════════════

    def _build(self):
        p = self.theme_mgr.palette
        inner = self._inner

        inner.grid_rowconfigure(0, weight=0)  # title
        inner.grid_rowconfigure(1, weight=1)  # main split
        inner.grid_rowconfigure(2, weight=0)  # log
        inner.grid_columnconfigure(0, weight=0)  # left config
        inner.grid_columnconfigure(1, weight=1)  # right charts+ctrl

        # 标题
        tk.Label(inner, text="训练", font=(FONT_FAMILY, 16, "bold"),
                 bg=p.bg, fg=p.fg, anchor="w").grid(row=0, column=0, columnspan=2,
                                  sticky="ew", padx=20, pady=(16, 8))

        # ── 左: 配置面板 ──
        self._build_left_config(inner, p)

        # ── 右: 实时监控 ──
        self._build_monitor(inner, p)

        # ── 底部全宽: 日志 ──
        self._build_log(inner, p)

    # ── 左: 配置面板 ──

    def _build_left_config(self, inner, p):
        container = tk.Frame(inner, width=320, bg=p.bg)
        container.grid(row=1, column=0, sticky="nsw", padx=(20, 8), pady=(0, 8))
        container.grid_propagate(False)

        canvas = tk.Canvas(container, width=310, highlightthickness=0, bg=p.bg)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=p.bg)

        scrollable.bind("<Configure>",
                        lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event):
            try:
                canvas.yview_scroll(-1 * (event.delta // 120), "units")
            except tk.TclError:
                pass

        canvas.bind("<MouseWheel>", _on_mousewheel)
        self._scroll_canvas = canvas

        self._build_data_section(scrollable, p)
        self._build_preset_section(scrollable, p)
        self._build_advanced_section(scrollable, p)
        self._build_template_buttons(scrollable, p)

    def _build_data_section(self, parent, p):
        gb = tk.LabelFrame(parent, text="数据与输出",
                           font=(FONT_FAMILY, 10, "bold"),
                           padx=8, pady=4, bg=p.bg, fg=p.fg)
        gb.pack(fill=tk.X, padx=4, pady=4)

        tk.Label(gb, text="数据集文件:", font=(FONT_FAMILY, 9),
                 bg=p.bg, fg=p.fg, anchor="w").pack(fill=tk.X, pady=(4, 2))

        list_frame = tk.Frame(gb, height=80, bg=p.bg)
        list_frame.pack(fill=tk.X, padx=2, pady=2)
        list_frame.pack_propagate(False)

        self._dataset_canvas = tk.Canvas(list_frame, height=76,
                                         highlightthickness=0, bg=p.bg)
        ds_scroll = ttk.Scrollbar(list_frame, orient="vertical",
                                  command=self._dataset_canvas.yview)
        ds_inner = tk.Frame(self._dataset_canvas, bg=p.bg)

        ds_inner.bind("<Configure>", lambda e: self._dataset_canvas.configure(
            scrollregion=self._dataset_canvas.bbox("all")))
        self._dataset_canvas.create_window((0, 0), window=ds_inner, anchor="nw")
        self._dataset_canvas.configure(yscrollcommand=ds_scroll.set)

        self._dataset_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ds_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._dataset_inner = ds_inner

        btn_row = tk.Frame(gb, bg=p.bg)
        btn_row.pack(fill=tk.X, pady=(2, 4))
        tk.Button(btn_row, text="刷新", command=self._scan_dataset_files,
                  font=(FONT_FAMILY, 8), bg=p.card_bg, fg=p.fg).pack(side=tk.LEFT, padx=2)
        self._ds_status = tk.Label(btn_row, text="扫描中...",
                                   font=(FONT_FAMILY, 8), fg=p.fg_secondary, bg=p.bg)
        self._ds_status.pack(side=tk.LEFT, padx=4)

        out_row = tk.Frame(gb, bg=p.bg)
        out_row.pack(fill=tk.X, pady=2)
        tk.Label(out_row, text="输出目录:", font=(FONT_FAMILY, 9),
                 bg=p.bg, fg=p.fg, width=9, anchor="w").pack(side=tk.LEFT)
        self._out_dir_var = tk.StringVar(value=self.state.out_dir)
        tk.Entry(out_row, textvariable=self._out_dir_var,
                 font=(FONT_FAMILY, 9), width=14,
                 bg=p.input_bg, fg=p.fg, insertbackground=p.fg).pack(side=tk.LEFT, padx=2)
        tk.Button(out_row, text="浏览", font=(FONT_FAMILY, 8),
                  bg=p.card_bg, fg=p.fg,
                  command=self._browse_out_dir).pack(side=tk.LEFT)

        ckpt_row = tk.Frame(gb, bg=p.bg)
        ckpt_row.pack(fill=tk.X, pady=2)
        tk.Label(ckpt_row, text="恢复检查点:", font=(FONT_FAMILY, 9),
                 bg=p.bg, fg=p.fg, width=9, anchor="w").pack(side=tk.LEFT)
        self._ckpt_var = tk.StringVar()
        self._ckpt_combo = ttk.Combobox(ckpt_row, textvariable=self._ckpt_var,
                                        font=(FONT_FAMILY, 8), width=20, state="readonly")
        self._ckpt_combo.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        self._ckpt_combo.bind("<<ComboboxSelected>>", self._on_ckpt_selected)

    def _build_preset_section(self, parent, p):
        gb = tk.LabelFrame(parent, text="快速预设",
                           font=(FONT_FAMILY, 10, "bold"),
                           padx=8, pady=4, bg=p.bg, fg=p.fg)
        gb.pack(fill=tk.X, padx=4, pady=4)

        preset_names = ["自定义"] + list(QUICK_PRESETS.keys())
        self._preset_var = tk.StringVar(value="自定义")
        combo = ttk.Combobox(gb, textvariable=self._preset_var,
                             values=preset_names, font=(FONT_FAMILY, 9),
                             state="readonly", width=16)
        combo.pack(fill=tk.X, padx=2, pady=2)
        combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

        tk.Label(gb, text="选择预设后自动填入参数, 可继续在高级参数中微调",
                 font=(FONT_FAMILY, 8), fg=p.fg_secondary, bg=p.bg,
                 wraplength=260, anchor="w").pack(fill=tk.X, padx=2, pady=(0, 4))

    def _build_advanced_section(self, parent, p):
        self._adv_frame = tk.LabelFrame(parent, text="高级参数",
                                        font=(FONT_FAMILY, 10, "bold"),
                                        padx=8, pady=4, bg=p.bg, fg=p.fg)
        self._adv_frame.pack(fill=tk.X, padx=4, pady=4)

        toggle_row = tk.Frame(self._adv_frame, bg=p.bg)
        toggle_row.pack(fill=tk.X)
        self._toggle_btn = tk.Button(
            toggle_row, text="展开 ▼", font=(FONT_FAMILY, 8),
            command=self._toggle_advanced, padx=8, cursor="hand2",
            bg=p.card_bg, fg=p.fg)
        self._toggle_btn.pack(side=tk.RIGHT)

        self._adv_content = tk.Frame(self._adv_frame, bg=p.bg)
        self._adv_content_packed = False

    def _toggle_advanced(self):
        self._advanced_visible = not self._advanced_visible
        if self._advanced_visible:
            if not self._adv_content_packed:
                p = self.theme_mgr.palette
                self._build_adv_param_groups(self._adv_content, p)
                self._adv_content_packed = True
            self._adv_content.pack(fill=tk.X, pady=(4, 0))
            self._toggle_btn.configure(text="收起 ▲")
        else:
            self._adv_content.pack_forget()
            self._toggle_btn.configure(text="展开 ▼")

    def _build_adv_param_groups(self, parent, p):
        for group_name, fields in PARAM_GROUPS:
            gb = tk.LabelFrame(parent, text=group_name,
                               font=(FONT_FAMILY, 9, "bold"),
                               padx=6, pady=2, bg=p.bg, fg=p.fg)
            gb.pack(fill=tk.X, pady=2)

            for key, label in fields:
                row = tk.Frame(gb, bg=p.bg)
                row.pack(fill=tk.X, pady=1)
                tk.Label(row, text=label, font=(FONT_FAMILY, 8),
                         bg=p.bg, fg=p.fg, width=14, anchor="w").pack(side=tk.LEFT)
                var = tk.StringVar(value=self.state.config_get(key))
                entry = tk.Entry(row, textvariable=var, font=(FONT_FAMILY, 8), width=10,
                                 bg=p.input_bg, fg=p.fg, insertbackground=p.fg)
                entry.pack(side=tk.RIGHT)
                self._param_widgets[key] = (var, entry)

    def _build_template_buttons(self, parent, p):
        btn_row = tk.Frame(parent, bg=p.bg)
        btn_row.pack(fill=tk.X, padx=4, pady=8)
        tk.Button(btn_row, text="保存模板", command=self._save_template,
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="加载模板", command=self._load_template,
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(side=tk.LEFT, padx=2)

    # ── 右: 监控区 ──

    def _build_monitor(self, inner, p):
        right = tk.Frame(inner, bg=p.bg)
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 20), pady=(0, 8))
        right.grid_rowconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)
        right.grid_columnconfigure(1, weight=1)

        # 4 个子 Canvas 占位作为 RealtimeChart 的 master
        self._charts = {}
        chart_configs = [
            ("chart_ce", "CE Loss", "CE"),
            ("chart_F", "Free Energy", "F"),
            ("chart_D", "Dopamine", "D"),
            ("chart_lr", "Learning Rate", "LR"),
        ]
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for (key, title, ylabel), (r, c) in zip(chart_configs, positions):
            sub_canvas = tk.Canvas(right, highlightthickness=0, bg=p.bg)
            sub_canvas.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
            chart = RealtimeChart(sub_canvas, position=(0, 0),
                                  size=(200, 150), title=title,
                                  ylabel=ylabel, theme_mgr=self.theme_mgr)
            chart.add_series(key)
            self._charts[key] = chart

        # ── 控制栏 ──
        ctrl = tk.Frame(right, bg=p.bg)
        ctrl.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        self._btn_start = tk.Button(ctrl, text="开始训练", command=self._start,
                                    font=(FONT_FAMILY, 10), bg="#4ade80",
                                    fg="black", padx=12, pady=2)
        self._btn_start.pack(side=tk.LEFT, padx=2)

        self._btn_stop = tk.Button(ctrl, text="停止", command=self._stop,
                                   font=(FONT_FAMILY, 10), state=tk.DISABLED,
                                   bg="#f87171", fg="black", padx=12, pady=2)
        self._btn_stop.pack(side=tk.LEFT, padx=2)

        self._btn_pause = tk.Button(ctrl, text="暂停", command=self._pause,
                                    font=(FONT_FAMILY, 10), state=tk.DISABLED,
                                    bg=p.card_bg, fg=p.fg, padx=12, pady=2)
        self._btn_pause.pack(side=tk.LEFT, padx=2)

        self._progress = ttk.Progressbar(ctrl, mode="determinate", length=200)
        self._progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        self._step_var = tk.StringVar(value="0 / 0")
        tk.Label(ctrl, textvariable=self._step_var,
                 font=(FONT_FAMILY, 9), bg=p.bg, fg=p.fg).pack(side=tk.LEFT, padx=4)

    # ── 日志 ──

    def _build_log(self, inner, p):
        """日志区域: tk.Canvas 占位 + 延迟创建 LogViewer."""
        log_container = tk.Frame(inner, bg=p.bg, height=130)
        log_container.grid(row=2, column=0, columnspan=2, sticky="nsew",
                           padx=20, pady=(0, 8))
        log_container.grid_propagate(False)

        sub_canvas = tk.Canvas(log_container, highlightthickness=0, bg=p.bg)
        sub_canvas.pack(fill=tk.BOTH, expand=True)

        self._log_container = log_container
        self._log_sub_canvas = sub_canvas
        self._log_built = False
        sub_canvas.bind("<Map>", self._build_log_delayed)

    def _build_log_delayed(self, event=None):
        if self._log_built:
            return
        self._log_built = True
        c = self._log_sub_canvas
        c.update_idletasks()
        w = c.winfo_width() or 600
        h = c.winfo_height() or 120
        self._log = LogViewer(c, position=(0, 0), size=(w, h),
                              theme_mgr=self.theme_mgr)

    # ═══════════════════════════════════════════════════════════════
    # 数据 / 检查点扫描
    # ═══════════════════════════════════════════════════════════════

    def _scan_dataset_files(self):
        """扫描 dataset/ 目录下的 .jsonl 文件."""
        self._ds_status.configure(text="扫描中...")
        # 清空旧 checkbox
        for w in self._dataset_inner.winfo_children():
            w.destroy()
        self._dataset_vars.clear()
        self._dataset_files.clear()

        files = sorted(glob.glob("dataset/*.jsonl"))
        if not files:
            # 也检查 datasets/
            files = sorted(glob.glob("datasets/*.jsonl"))
        if not files:
            self._ds_status.configure(text="未找到数据集文件")
            return

        self._dataset_files = files
        for fpath in files:
            var = tk.BooleanVar(value=False)
            fname = os.path.basename(fpath)
            cb = tk.Checkbutton(self._dataset_inner, text=fname, variable=var,
                                font=(FONT_FAMILY, 8), anchor="w",
                                selectcolor="#1e1e2e")
            cb.pack(fill=tk.X, padx=2, pady=1)
            self._dataset_vars.append(var)

        self._ds_status.configure(text=f"{len(files)} 个文件")

    def _refresh_checkpoints(self):
        """从 state 注册表刷新检查点下拉."""
        registrations = self.state.checkpoint_registry
        if not registrations:
            self._ckpt_combo.configure(values=["(无)"])
            self._ckpt_var.set("(无)")
            return
        labels = []
        for e in registrations:
            d = e.get("dir", "")
            f = e.get("filename", "")
            s = e.get("step", 0)
            label = f"{d}/{f} (step {s})" if s else f"{d}/{f}"
            labels.append(label)
        self._ckpt_combo.configure(values=labels)
        self._ckpt_var.set(labels[0] if labels else "(无)")

    def _on_ckpt_selected(self, event=None):
        idx = self._ckpt_combo.current()
        if idx >= 0 and idx < len(self.state.checkpoint_registry):
            entry = self.state.checkpoint_registry[idx]
            self.state.current_model_path = entry["path"]

    def _browse_out_dir(self):
        d = filedialog.askdirectory(initialdir=self._out_dir_var.get() or ".")
        if d:
            self._out_dir_var.set(d)

    # ═══════════════════════════════════════════════════════════════
    # 预设
    # ═══════════════════════════════════════════════════════════════

    def _on_preset_selected(self, event=None):
        name = self._preset_var.get()
        if name == "自定义" or name not in QUICK_PRESETS:
            return
        preset = QUICK_PRESETS[name]
        for key, val in preset.items():
            self.state.config_set(key, val)
            if key in self._param_widgets:
                self._param_widgets[key][0].set(val)
        self.app.set_status(f"预设 '{name}' 已应用")

    # ═══════════════════════════════════════════════════════════════
    # 配置同步 / 模板
    # ═══════════════════════════════════════════════════════════════

    def _sync_params(self):
        """将界面参数同步到 state."""
        for key, (var, _) in self._param_widgets.items():
            self.state.config_set(key, var.get())

    def _save_template(self):
        from tkinter import simpledialog
        name = simpledialog.askstring("保存模板", "模板名称:")
        if name:
            self._sync_params()
            self.state.config_templates[name] = dict(self.state.config)
            self.app.set_status(f"模板 '{name}' 已保存")

    def _load_template(self):
        if not self.state.config_templates:
            self.app.set_status("暂无保存的模板")
            return
        from tkinter import simpledialog
        names = list(self.state.config_templates.keys())
        name = simpledialog.askstring("加载模板",
                                      f"可用模板: {', '.join(names)}\n输入名称:")
        if name and name in self.state.config_templates:
            for key, val in self.state.config_templates[name].items():
                self.state.config_set(key, val)
                if key in self._param_widgets:
                    self._param_widgets[key][0].set(val)
            self.app.set_status(f"模板 '{name}' 已加载")

    # ═══════════════════════════════════════════════════════════════
    # ★ 修复2: ThreadedTrainer 控制
    # ═══════════════════════════════════════════════════════════════

    def _get_selected_datasets(self) -> list[tuple[str, str, int]]:
        """获取选中的数据集文件列表 → [(task_id, path, max_samples), ...]."""
        selected = []
        for i, var in enumerate(self._dataset_vars):
            if var.get():
                fpath = self._dataset_files[i]
                task_id = os.path.splitext(os.path.basename(fpath))[0]
                subset = int(self.state.config_get("subset", "0"))
                selected.append((task_id, fpath, subset))
        return selected

    def _start(self):
        self._sync_params()

        # ── 检查数据集 ──
        pipelines = self._get_selected_datasets()
        if not pipelines:
            messagebox.showwarning("警告", "请先选择至少一个数据集文件")
            return

        # ── 检查输出目录 ──
        out_dir = self._out_dir_var.get().strip()
        if not out_dir:
            messagebox.showwarning("警告", "请设置输出目录")
            return

        # ── 导入训练组件 (延迟避免循环) ──
        from core.training import TrainingConfig
        from core.threaded_trainer import ThreadedTrainer

        # 构建配置
        kwargs = self.state.config_to_kwargs()
        kwargs["out_dir"] = out_dir
        # 检查点恢复
        ckpt = self.state.current_model_path
        if ckpt and os.path.isfile(ckpt):
            kwargs["checkpoint_path"] = ckpt

        config = TrainingConfig(**kwargs)

        # ── TkProgressCallback: 日志输出 ──
        from core.threaded_trainer import TkProgressCallback
        tk_cb = TkProgressCallback(self._log._text, root=self.app.root)

        # ── 创建 Trainer ──
        self._trainer = ThreadedTrainer(config, progress_callback=tk_cb)
        self._trainer.set_task_pipelines(pipelines)

        # 替换回调: 同时更新图表 + 进度条
        def combined_cb(data: dict):
            tk_cb(data)
            try:
                self.app.root.after(0, self._on_progress, data)
            except Exception:
                pass

        self._trainer.callback = combined_cb

        # ── UI 状态 ──
        self._running = True
        self._paused = False
        self._btn_start.configure(state=tk.DISABLED, text="训练中...")
        self._btn_stop.configure(state=tk.NORMAL)
        self._btn_pause.configure(state=tk.NORMAL, text="暂停")

        # 清空旧图表
        for ch in self._charts.values():
            ch.clear()

        self._progress["value"] = 0
        self._step_var.set("0 / 0")
        self._log.info("=" * 50)
        self._log.info("训练开始...")
        self.app.set_status("训练运行中")

        # ── 启动 ──
        self._trainer.start()

    def _stop(self):
        if self._trainer:
            self._trainer.stop()
        self._running = False
        self._btn_start.configure(state=tk.NORMAL, text="开始训练")
        self._btn_stop.configure(state=tk.DISABLED)
        self._btn_pause.configure(state=tk.DISABLED, text="暂停")
        self._log.info("训练已停止")
        self.app.set_status("就绪")

    def _pause(self):
        if not self._trainer:
            return
        if self._paused:
            self._trainer.resume()
            self._paused = False
            self._btn_pause.configure(text="暂停")
            self._log.info("▶ 训练已恢复")
        else:
            self._trainer.pause()
            self._paused = True
            self._btn_pause.configure(text="继续")
            self._log.info("⏸ 训练已暂停")

    def _on_progress(self, data: dict):
        """在主线程中处理进度回调 (图表 / 进度条 / 步数)."""
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return

        if data.get("type") == "progress":
            step = data.get("step", 0)
            total = data.get("total_steps", 1)
            ce = data.get("ce_loss")
            F_val = data.get("F")
            D_val = data.get("D")
            lr = data.get("lr")

            # 图表
            if ce is not None:
                self._charts["chart_ce"].add_point("chart_ce", step, ce)
                self._charts["chart_ce"].redraw()
            if F_val is not None:
                self._charts["chart_F"].add_point("chart_F", step, F_val)
                self._charts["chart_F"].redraw()
            if D_val is not None:
                self._charts["chart_D"].add_point("chart_D", step, D_val)
                self._charts["chart_D"].redraw()
            if lr is not None:
                self._charts["chart_lr"].add_point("chart_lr", step, lr)
                self._charts["chart_lr"].redraw()

            # 进度条
            if total > 0:
                pct = min(step / total * 100, 100)
                self._progress["value"] = pct
            self._step_var.set(f"{step} / {total}")

        elif data.get("type") == "done":
            self._running = False
            self._btn_start.configure(state=tk.NORMAL, text="开始训练")
            self._btn_stop.configure(state=tk.DISABLED)
            self._btn_pause.configure(state=tk.DISABLED, text="暂停")
            self._progress["value"] = 100
            self.app.set_status("训练完成 ✅")

        elif data.get("type") == "error":
            self._running = False
            self._btn_start.configure(state=tk.NORMAL, text="开始训练")
            self._btn_stop.configure(state=tk.DISABLED)
            self._btn_pause.configure(state=tk.DISABLED, text="暂停")
            self.app.set_status("训练出错 ❌")
