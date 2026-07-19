"""
骇客像素风主窗口 — 左导航 + ToolBar + ContentStack + StatusBar.

启动:
    from gui import launch_gui
    launch_gui()
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime

from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QStatusBar, QFrame,
    QSplitter, QSizePolicy,
)
from PyQt6.QtGui import QFont, QAction, QIcon, QPalette, QColor, QFontDatabase

from gui.theme import HackerTheme, FONT_FAMILY, FONT_SIZE_BASE
from gui.state_bridge import ExperimentStateSignals, bridge_state
from gui.state import ExperimentState


# ═══════════════════════════════════════════════════════════════════
# 导航项
# ═══════════════════════════════════════════════════════════════════

NAV_ITEMS = [
    ("ctrl",   "≡",  "控制台",     "dashboard"),
    ("train",  "◈",  "训练",       "training"),
    ("cl",     "⟁",  "持续学习",    "continual"),
    ("eval",   "⌘",  "评估",       "evaluation"),
    ("auto",   "⚡",  "自主运行",    "autonomous"),
    ("data",   "⎔",  "数据管理",    "data_manager"),
    ("ckpt",   "◉",  "检查点",     "checkpoints"),
    ("config_templates", "☰",  "配置模板",    "config_templates"),
]


# ═══════════════════════════════════════════════════════════════════
# 左侧导航
# ═══════════════════════════════════════════════════════════════════


class NavItem(QPushButton):
    """单个导航按钮."""

    def __init__(self, key: str, icon: str, label: str, parent=None):
        super().__init__(parent)
        self._key = key
        self.setCheckable(True)
        self.setFixedHeight(38)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setText(f"  {icon}  {label}")
        self.setFont(QFont(FONT_FAMILY, 10))
        self.setObjectName("navItem")


class LeftNav(QWidget):
    """左侧导航面板 — 220px 宽, 可折叠."""

    active_changed = pyqtSignal(str)  # key

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: dict[str, NavItem] = {}
        self._active_key: str | None = None
        self._collapsed = False
        self.setFixedWidth(220)
        self.setObjectName("leftNav")

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(2)

        # 标题
        title = QLabel("  NAV")
        title.setFont(QFont(FONT_FAMILY, 9))
        title.setObjectName("secondary")
        title.setFixedHeight(24)
        self._layout.addWidget(title)

        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setObjectName("navSep")
        self._layout.addWidget(sep)
        self._layout.addSpacing(4)

        # 导航按钮
        for key, icon, label, _ in NAV_ITEMS:
            btn = NavItem(key, icon, label)
            btn.clicked.connect(lambda checked, k=key: self._on_click(k))
            self._layout.addWidget(btn)
            self._items[key] = btn

        self._layout.addStretch()

        # 默认选中 (用 button_key 而非 page_key)
        self.set_active("ctrl")

    def _on_click(self, key: str) -> None:
        self.set_active(key)
        self.active_changed.emit(key)

    def set_active(self, key: str) -> None:
        """切换活跃项."""
        if self._active_key:
            old = self._items.get(self._active_key)
            if old:
                old.setChecked(False)
        self._active_key = key
        btn = self._items.get(key)
        if btn:
            btn.setChecked(True)

    def toggle_collapse(self) -> None:
        """折叠/展开."""
        self._collapsed = not self._collapsed
        w = 60 if self._collapsed else 220
        self.setFixedWidth(w)


# ═══════════════════════════════════════════════════════════════════
# ToolBar
# ═══════════════════════════════════════════════════════════════════


class ToolBar(QWidget):
    """顶栏 — 面包屑(当前页名) + 右侧主题切换."""

    theme_toggled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setObjectName("toolBar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)

        self._breadcrumb = QLabel("控制台")
        self._breadcrumb.setFont(QFont(FONT_FAMILY, 11))
        self._breadcrumb.setObjectName("breadcrumb")
        layout.addWidget(self._breadcrumb)

        layout.addStretch()

        self._theme_btn = QPushButton("☾")
        self._theme_btn.setFixedSize(32, 28)
        self._theme_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._theme_btn.clicked.connect(self.theme_toggled.emit)
        layout.addWidget(self._theme_btn)

    def set_title(self, title: str) -> None:
        self._breadcrumb.setText(title)

    def set_theme_icon(self, dark: bool) -> None:
        self._theme_btn.setText("☀" if dark else "☾")


# ═══════════════════════════════════════════════════════════════════
# ContentStack
# ═══════════════════════════════════════════════════════════════════


class ContentStack(QStackedWidget):
    """页面容器 — 支持 fade 切换 (通过 QTimer 分步透明)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("contentStack")


# ═══════════════════════════════════════════════════════════════════
# GPU 探测
# ═══════════════════════════════════════════════════════════════════


def _probe_gpu() -> str:
    """非阻塞探测 GPU 信息."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            return f"GPU: {name}  {mem:.0f}GB"
        return "GPU: N/A"
    except Exception:
        return "GPU: N/A"


# ═══════════════════════════════════════════════════════════════════
# MainWindow
# ═══════════════════════════════════════════════════════════════════


class MainWindow(QMainWindow):
    """骇客像素风主窗口."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("virtuosov2 — PC 局部动态小语言模型")
        self.setMinimumSize(1200, 720)
        self.resize(1440, 900)

        # ── 主题 ──
        self._theme = HackerTheme(dark=True)

        # ── 状态桥接 ──
        self._state = ExperimentState()
        self._signals = ExperimentStateSignals()
        self._bridge = bridge_state(self._state, self._signals)

        # ── 中央控件 ──
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 左侧导航
        self._nav = LeftNav()
        main_layout.addWidget(self._nav)

        # 右侧内容区 (ToolBar + 页面 + StatusBar)
        right_side = QVBoxLayout()
        right_side.setContentsMargins(0, 0, 0, 0)
        right_side.setSpacing(0)

        self._toolbar = ToolBar()
        right_side.addWidget(self._toolbar)

        self._content = ContentStack()
        right_side.addWidget(self._content, 1)

        main_layout.addLayout(right_side, 1)

        # ── 状态栏 ──
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status_label = QLabel("就绪")
        self._status_label.setObjectName("secondary")
        self._status.addWidget(self._status_label, 1)

        self._gpu_label = QLabel("GPU: 检测中...")
        self._gpu_label.setObjectName("secondary")
        self._status.addPermanentWidget(self._gpu_label)

        # 时钟
        self._clock_label = QLabel()
        self._clock_label.setObjectName("secondary")
        self._status.addPermanentWidget(self._clock_label)

        # ── 信号连接 ──
        self._nav.active_changed.connect(self._on_nav_change)
        self._toolbar.theme_toggled.connect(self._toggle_theme)
        self._signals.status_message.connect(self._status_label.setText)

        # ── 初始化 ──
        self._register_pages()
        self._apply_theme()
        self._nav.set_active("ctrl")
        self._content.setCurrentIndex(0)

        # ── 后初始化 ──
        QTimer.singleShot(500, self._on_startup)
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(10000)
        self._update_clock()

    # ── 页面注册 ──

    def _register_pages(self) -> None:
        """注册 8 个页面到 ContentStack (逐页 try/except 兜底)."""
        self._pages: dict[str, QWidget] = {}
        _page_specs: list[tuple[str, str, tuple]] = [
            ("dashboard", "gui.pages.dashboard", ("DashboardPage", self._bridge, self._theme)),
            ("training",   "gui.pages.training",   ("TrainingPage", self._bridge, self._theme, self._signals)),
            ("continual",  "gui.pages.continual",  ("ContinualPage", self._bridge, self._theme)),
            ("evaluation", "gui.pages.evaluation", ("EvaluationPage", self._bridge, self._theme)),
            ("autonomous", "gui.pages.autonomous", ("AutonomousPage", self._bridge, self._theme)),
            ("data_manager","gui.pages.data_manager",("DataManagerPage", self._bridge, self._theme)),
            ("checkpoints","gui.pages.checkpoints",("CheckpointsPage", self._bridge, self._theme)),
            ("config_templates","gui.pages.config_templates",("ConfigTemplatesPage", self._bridge, self._theme)),
        ]
        for page_key, mod_path, (cls_name, *args) in _page_specs:
            try:
                import importlib
                mod = importlib.import_module(mod_path)
                cls = getattr(mod, cls_name)
                page = cls(*args)
                self._pages[page_key] = page
                self._content.addWidget(page)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                # 创建兜底 fallback 页面
                fallback = QWidget()
                fb_layout = QVBoxLayout(fallback)
                err_label = QLabel(f"⚠ 页面加载失败: {cls_name}\n{exc}")
                err_label.setObjectName("error")
                fb_layout.addWidget(err_label)
                self._pages[page_key] = fallback
                self._content.addWidget(fallback)

    # ── 导航切换 ──

    def _on_nav_change(self, key: str) -> None:
        """导航项切换 → 更新面包屑 + 页面 (button_key → page_key 映射)."""
        # button_key → page_key 映射
        _btn_to_page = {k: p for k, _, _, p in NAV_ITEMS}
        page_key = _btn_to_page.get(key, key)
        # 面包屑
        titles = {k: label for k, _, label, _ in NAV_ITEMS}
        self._toolbar.set_title(titles.get(key, page_key))
        # 页面切换
        page_keys = list(self._pages.keys())
        if page_key in page_keys:
            self._content.setCurrentIndex(page_keys.index(page_key))

    # ── 主题切换 ──

    def _toggle_theme(self) -> None:
        self._theme.toggle()
        self._apply_theme()
        self._toolbar.set_theme_icon(self._theme.is_dark)
        # 通知各页面
        for page in self._pages.values():
            if hasattr(page, "on_theme_changed"):
                page.on_theme_changed(self._theme)

    def _apply_theme(self) -> None:
        qss = self._theme.qss()
        self.setStyleSheet(qss)
        self._toolbar.set_theme_icon(self._theme.is_dark)

    # ── 启动 ──

    def _on_startup(self) -> None:
        """启动后异步任务 (try/except 防止闪退)."""
        try:
            # GPU 探测
            gpu_info = _probe_gpu()
            self._gpu_label.setText(gpu_info)
            # 触发 dashboard 数据加载
            if "dashboard" in self._pages:
                page = self._pages["dashboard"]
                if hasattr(page, "refresh"):
                    page.refresh()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._status_label.setText(f"⚠ 启动加载异常: {exc}")

    def _update_clock(self) -> None:
        self._clock_label.setText(datetime.now().strftime("%H:%M:%S"))

    # ── 属性 ──

    @property
    def bridge(self):
        return self._bridge

    @property
    def signals(self):
        return self._signals

    @property
    def theme(self):
        return self._theme


# ═══════════════════════════════════════════════════════════════════
# 启动入口
# ═══════════════════════════════════════════════════════════════════


_APP: QApplication | None = None


def _exception_hook(exc_type, exc_value, exc_tb):
    """全局异常钩子 — 打印 traceback 到 stderr 防止 PyQt6 静默吞异常."""
    import traceback
    traceback.print_exception(exc_type, exc_value, exc_tb)


def launch_gui() -> None:
    """启动 PyQt6 骇客像素风 GUI."""
    sys.excepthook = _exception_hook

    global _APP
    if _APP is None:
        _APP = QApplication(sys.argv)
        _APP.setStyle("Fusion")
        # 设置应用字体
        font = QFont(FONT_FAMILY, FONT_SIZE_BASE)
        _APP.setFont(font)

    window = MainWindow()
    window.show()

    if _APP is not None:
        _APP.exec()
