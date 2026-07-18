"""
virtuosov2 GUI 主窗口 — 导航框架 + 页面路由.

布局:
  ┌─ Title Bar ──────────────────────────────────────┐
  │  virtuosov2 v0.2.0              [深色主题] [设置] │
  ├──────────┬──────────────────────────────────────┤
  │          │  BREADCRUMB                          │
  │  NAV     ├──────────────────────────────────────┤
  │  PANEL   │  CONTENT AREA                        │
  │  200px   │  (当前页面)                           │
  │          │                                      │
  ├──────────┴──────────────────────────────────────┤
  │  Status Bar                                     │
  └─────────────────────────────────────────────────┘

线程安全: 训练线程通过 root.after() 队列推送更新到主线程.
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

import maliang

from gui.theme import ThemeManager, FONT_FAMILY, FONT_FAMILY_MONO, Theme
from gui.state import ExperimentState

# 页面导入 (延迟到 show_page 时, 避免循环依赖)
_page_registry = {}


def _lazy_page(name: str):
    """延迟导入并缓存页面类."""
    if name in _page_registry:
        return _page_registry[name]
    module_map = {
        "dashboard": "gui.pages.dashboard",
        "training": "gui.pages.training",
        "continual": "gui.pages.continual",
        "evaluation": "gui.pages.evaluation",
        "autonomous": "gui.pages.autonomous",
        "data": "gui.pages.data_manager",
        "checkpoints": "gui.pages.checkpoints",
        "templates": "gui.pages.config_templates",
    }
    if name not in module_map:
        raise ValueError(f"未知页面: {name}")
    import importlib
    mod = importlib.import_module(module_map[name])
    cls = getattr(mod, "Page")
    _page_registry[name] = cls
    return cls


# ═══════════════════════════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════════════════════════


class MainWindow:
    """应用主窗口."""

    def __init__(self, root: tk.Tk, state: ExperimentState, dpi_scale: float = 1.25):
        self.root = root
        self.state = state
        self.theme_mgr = state.theme_mgr

        # DPI 感知
        self._init_dpi(dpi_scale)

        # 窗口配置
        root.title("virtuosov2 v0.2.0")
        root.geometry(size=(1280, 860))
        root.minsize(1024, 720)

        # 当前页面跟踪
        self._current_page: Optional[tk.Widget] = None
        self._current_page_name: str = ""

        # 构建 UI
        self._build_ui()

        # 应用主题
        self.theme_mgr.apply(root)
        self.theme_mgr.on_toggle(self._on_theme_changed)

    # ── DPI ──

    def _init_dpi(self, dpi_scale: float):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        try:
            self.root.tk.call("tk", "scaling", self.root.tk.call("tk", "scaling") * dpi_scale)
        except Exception:
            pass

    # ── UI 构建 ──

    def _build_ui(self):
        self.root.grid_rowconfigure(0, weight=0)  # title
        self.root.grid_rowconfigure(1, weight=0)  # breadcrumb
        self.root.grid_rowconfigure(2, weight=1)  # nav + content
        self.root.grid_rowconfigure(3, weight=0)  # status
        self.root.grid_columnconfigure(0, weight=0)  # nav
        self.root.grid_columnconfigure(1, weight=1)  # content

        self._build_title_bar()
        self._build_breadcrumb()
        self._build_nav_panel()
        self._build_content_area()
        self._build_status_bar()

        # 默认页
        self.show_page("dashboard")

    def _build_title_bar(self):
        p = self.theme_mgr.palette
        frame = tk.Frame(self.root, height=40, bg=p.accent)
        frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        frame.grid_propagate(False)

        tk.Label(
            frame,
            text="virtuosov2 v0.2.0",
            font=(FONT_FAMILY, 13, "bold"),
            bg=p.accent,
            fg="#ffffff",
        ).pack(side=tk.LEFT, padx=16, pady=8)

        # 主题切换
        self._theme_btn = tk.Label(
            frame,
            text=f"{self.theme_mgr.theme_name()}主题",
            font=(FONT_FAMILY, 10),
            bg=p.card_bg,
            fg=p.fg,
            cursor="hand2",
            padx=10,
            pady=2,
        )
        self._theme_btn.pack(side=tk.RIGHT, padx=8, pady=8)
        self._theme_btn.bind("<Button-1>", lambda e: self._toggle_theme())

    def _build_breadcrumb(self):
        p = self.theme_mgr.palette
        frame = tk.Frame(self.root, height=24, bg=p.bg)
        frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        frame.grid_propagate(False)

        self._breadcrumb_var = tk.StringVar(value="控制台")
        tk.Label(
            frame,
            textvariable=self._breadcrumb_var,
            font=(FONT_FAMILY, 9),
            bg=p.bg,
            fg=p.fg_secondary,
        ).pack(side=tk.LEFT, padx=16, pady=2)

    def _build_nav_panel(self):
        p = self.theme_mgr.palette
        frame = tk.Frame(self.root, width=180, bg=p.card_bg)
        frame.grid(row=2, column=0, sticky="nsw")
        frame.grid_propagate(False)

        # 导航标题
        tk.Label(
            frame,
            text="导航",
            font=(FONT_FAMILY, 10, "bold"),
            bg=p.card_bg,
            fg=p.fg,
            padx=16,
            pady=8,
        ).pack(fill=tk.X)

        self._nav_buttons: dict[str, tk.Widget] = {}
        nav_items = [
            ("dashboard", "控制台"),
            ("training", "训练"),
            ("continual", "持续学习"),
            ("evaluation", "评估"),
            ("autonomous", "自主运行"),
            ("data", "数据"),
            ("checkpoints", "检查点"),
            ("templates", "模板"),
        ]
        for name, label in nav_items:
            btn = tk.Label(
                frame,
                text=label,
                font=(FONT_FAMILY, 10),
                bg=p.card_bg,
                fg=p.fg_secondary,
                padx=16,
                pady=6,
                cursor="hand2",
                anchor="w",
            )
            btn.pack(fill=tk.X)
            btn.bind("<Button-1>", lambda e, n=name: self.show_page(n))
            self._nav_buttons[name] = btn

    def _build_content_area(self):
        p = self.theme_mgr.palette
        self._content_frame = maliang.Canvas(self.root, bg=p.bg, highlightthickness=0)
        self._content_frame.grid(row=2, column=1, sticky="nsew", padx=(0, 1), pady=(0, 1))
        self._content_frame.grid_rowconfigure(0, weight=1)
        self._content_frame.grid_columnconfigure(0, weight=1)

    def _build_status_bar(self):
        p = self.theme_mgr.palette
        frame = tk.Frame(self.root, height=24, bg=p.card_bg)
        frame.grid(row=3, column=0, columnspan=2, sticky="ew")
        frame.grid_propagate(False)

        self._status_var = tk.StringVar(value="就绪")
        tk.Label(
            frame,
            textvariable=self._status_var,
            font=(FONT_FAMILY, 9),
            bg=p.card_bg,
            fg=p.fg_secondary,
        ).pack(side=tk.LEFT, padx=16, pady=2)

        # GPU 状态 (延迟探测, 不阻塞主线程)
        self._gpu_var = tk.StringVar(value="GPU: 探测中...")
        tk.Label(
            frame,
            textvariable=self._gpu_var,
            font=(FONT_FAMILY, 9),
            bg=p.card_bg,
            fg=p.fg_secondary,
        ).pack(side=tk.RIGHT, padx=16, pady=2)
        self.root.after(500, self._probe_gpu)

    # ── 页面路由 ──

    def show_page(self, name: str):
        """切换到指定页面."""
        if name == self._current_page_name:
            return

        # 销毁当前页面
        if self._current_page:
            self._current_page.destroy()

        # 更新面包屑
        page_labels = {
            "dashboard": "控制台",
            "training": "训练",
            "continual": "持续学习",
            "evaluation": "评估",
            "autonomous": "自主运行",
            "data": "数据",
            "checkpoints": "检查点",
            "templates": "模板",
        }
        self._breadcrumb_var.set(f"首页 > {page_labels.get(name, name)}")

        # 高亮导航
        p = self.theme_mgr.palette
        for n, btn in self._nav_buttons.items():
            btn.configure(fg=p.fg if n == name else p.fg_secondary)

        # 创建新页面
        try:
            PageCls = _lazy_page(name)
            page = PageCls(self._content_frame, self.state, self)
            page.grid(row=0, column=0, sticky="nsew")
            self._current_page = page
            self._current_page_name = name
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._show_error_page(str(e))
            # 重置状态防止重复销毁已失效的页面
            self._current_page = None
            self._current_page_name = None

    def _show_error_page(self, msg: str):
        frame = ttk.Frame(self._content_frame, padding=40)
        frame.grid(row=0, column=0, sticky="nsew")
        ttk.Label(frame, text="页面加载失败", font=(FONT_FAMILY, 16)).pack()
        ttk.Label(frame, text=msg, font=(FONT_FAMILY, 10)).pack(pady=8)

    # ── 主题 ──

    def _toggle_theme(self):
        self.theme_mgr.toggle()
        p = self.theme_mgr.palette
        self.root.configure(bg=p.bg)
        self.theme_mgr.apply(self.root)
        self._theme_btn.configure(text=f"{self.theme_mgr.theme_name()}主题")

    def _on_theme_changed(self, theme: Theme):
        pass  # handled in _toggle_theme

    # ── GPU 探测 ──

    def _probe_gpu(self):
        """延迟探测 GPU 状态 (在事件循环中异步执行)."""
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)[:24]
                mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self._gpu_var.set(f"GPU: {name} | {mem:.0f}GB")
            else:
                self._gpu_var.set("GPU: 未检测到")
        except Exception:
            self._gpu_var.set("GPU: --")

    # ── 状态栏 ──

    def set_status(self, text: str):
        self._status_var.set(text)

    def status_message(self, msg: str, timeout_ms: int = 5000):
        """显示临时状态消息."""
        prev = self._status_var.get()
        self._status_var.set(msg)
        if timeout_ms > 0:
            self.root.after(timeout_ms, lambda: self._status_var.set(prev))


# ═══════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════


def launch_gui(dpi_scale: float = 1.25):
    """启动 GUI 主窗口."""
    root = maliang.Tk()
    root.minsize(1024, 720)

    state = ExperimentState()
    MainWindow(root, state, dpi_scale)

    def _on_close():
        root.destroy()
        os._exit(0)

    root.protocol("WM_DELETE_WINDOW", _on_close)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        root.destroy()
        os._exit(0)
