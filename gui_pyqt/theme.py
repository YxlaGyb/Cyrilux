"""
骇客像素风主题系统 — 深色/浅色 双主题 + QSS 生成.

使用:
    theme = HackerTheme(dark=True)
    app.setStyleSheet(theme.qss())

    # 运行时切换
    theme.toggle()
    app.setStyleSheet(theme.qss())

HackerPalette 18 色系统:
  Dark:  bg #0a0e0a  green #00ff41  amber #ffb000  red #ff3333  cyan #00d4ff
  Light: bg #e8eae0  green #007a20  amber #b07000  red #cc2222  cyan #007a8a
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HackerPalette:
    """骇客像素风单主题色板 — 18 色系统."""

    # 背景层
    bg: str               # 最底层背景
    card_bg: str          # 卡片/面板背景
    input_bg: str         # 输入框背景
    border: str           # 边框
    border_light: str     # 浅边框 / hover

    # 前景层
    fg: str               # 主文字
    fg_secondary: str     # 次要文字
    fg_dim: str           # 禁用/占位文字

    # 语义色
    accent: str           # 主题绿 — 主要操作
    accent_dim: str       # 暗绿 — 非活跃状态
    warning: str          # 琥珀 — 警告
    error: str            # 红 — 错误/删除
    cyan: str             # 青 — 信息/链接

    # 功能色
    success: str          # 成功状态
    progress_bg: str      # 进度条背景
    progress_fg: str      # 进度条前景
    select_bg: str        # 选中背景
    select_fg: str        # 选中文字
    highlight: str        # 高亮/搜索匹配


# ── 双主题 18 色板 ────────────────────────────────────────────────

DARK_PALETTE = HackerPalette(
    bg="#0a0e0a",
    card_bg="#0f1a0f",
    input_bg="#141e14",
    border="#1a3a1a",
    border_light="#2a5a2a",
    fg="#d0e8d0",
    fg_secondary="#7a9a7a",
    fg_dim="#3a5a3a",
    accent="#00ff41",
    accent_dim="#007a20",
    warning="#ffb000",
    error="#ff3333",
    cyan="#00d4ff",
    success="#00cc44",
    progress_bg="#1a2a1a",
    progress_fg="#00ff41",
    select_bg="#00ff41",
    select_fg="#0a0e0a",
    highlight="#ffb000",
)

LIGHT_PALETTE = HackerPalette(
    bg="#e8eae0",
    card_bg="#ffffff",
    input_bg="#f0f2ec",
    border="#c8d0c0",
    border_light="#a8b8a0",
    fg="#1a201a",
    fg_secondary="#6a7a6a",
    fg_dim="#9aaa9a",
    accent="#007a20",
    accent_dim="#3a8a4a",
    warning="#b07000",
    error="#cc2222",
    cyan="#007a8a",
    success="#008830",
    progress_bg="#d0d8c8",
    progress_fg="#007a20",
    select_bg="#007a20",
    select_fg="#ffffff",
    highlight="#b07000",
)


# ── 字体常量 ──────────────────────────────────────────────────────

FONT_FAMILY = "Consolas, 'Courier New', monospace"
FONT_SIZE_BASE = 10
FONT_SIZE_SM = 9
FONT_SIZE_LG = 12
FONT_SIZE_XL = 16


# ── QSS 生成器 ────────────────────────────────────────────────────

def _c(hex_color: str) -> str:
    """确保颜色格式为 #rrggbb."""
    return hex_color if hex_color.startswith("#") else f"#{hex_color}"


class HackerTheme:
    """骇客像素风主题 — 管理调色板及 QSS 字符串."""

    def __init__(self, dark: bool = True):
        self._dark = dark

    @property
    def palette(self) -> HackerPalette:
        return DARK_PALETTE if self._dark else LIGHT_PALETTE

    @property
    def is_dark(self) -> bool:
        return self._dark

    def toggle(self) -> None:
        self._dark = not self._dark

    def set_dark(self, dark: bool) -> None:
        self._dark = dark

    def qss(self) -> str:
        """生成完整应用 QSS."""
        p = self.palette
        bg = _c(p.bg)
        card = _c(p.card_bg)
        inp = _c(p.input_bg)
        border = _c(p.border)
        border_lt = _c(p.border_light)
        fg = _c(p.fg)
        fg_sec = _c(p.fg_secondary)
        fg_dim = _c(p.fg_dim)
        accent = _c(p.accent)
        accent_dim = _c(p.accent_dim)
        warn = _c(p.warning)
        err = _c(p.error)
        cyan = _c(p.cyan)
        ok = _c(p.success)
        prog_bg = _c(p.progress_bg)
        prog_fg = _c(p.progress_fg)
        sel_bg = _c(p.select_bg)
        sel_fg = _c(p.select_fg)
        hl = _c(p.highlight)

        return f"""
/* ═══════════════════════════════════════════════════════════════
   骇客像素风 — 全局 QSS
   ═══════════════════════════════════════════════════════════════ */

QWidget {{
    background-color: {bg};
    color: {fg};
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_BASE}pt;
}}

/* ── 主窗口 ── */
QMainWindow {{
    background-color: {bg};
}}

/* ── 菜单栏 ── */
QMenuBar {{
    background-color: {card};
    border-bottom: 1px solid {border};
    padding: 2px;
}}
QMenuBar::item {{
    padding: 4px 12px;
    color: {fg_sec};
}}
QMenuBar::item:selected {{
    background-color: {accent_dim};
    color: {fg};
}}
QMenu {{
    background-color: {card};
    border: 1px solid {border};
    padding: 2px;
}}
QMenu::item {{
    padding: 4px 24px;
}}
QMenu::item:selected {{
    background-color: {accent_dim};
    color: {accent};
}}

/* ── 按钮 ── */
QPushButton {{
    background-color: {accent_dim};
    color: {accent};
    border: 1px solid {border};
    padding: 6px 16px;
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_SM}pt;
    min-height: 20px;
}}
QPushButton:hover {{
    background-color: {border};
    border-color: {accent};
    color: {accent};
}}
QPushButton:pressed {{
    background-color: {accent};
    color: {sel_fg};
}}
QPushButton:disabled {{
    background-color: {card};
    color: {fg_dim};
    border-color: {border};
}}
QPushButton#danger {{
    color: {err};
    border-color: {err};
}}
QPushButton#danger:hover {{
    background-color: {err};
    color: {sel_fg};
}}

/* ── 输入框 ── */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {inp};
    color: {fg};
    border: 1px solid {border};
    padding: 4px 8px;
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_BASE}pt;
    selection-background-color: {accent};
    selection-color: {sel_fg};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {accent};
}}
QLineEdit:disabled {{
    background-color: {card};
    color: {fg_dim};
}}

/* ── 下拉框 ── */
QComboBox {{
    background-color: {inp};
    color: {fg};
    border: 1px solid {border};
    padding: 4px 8px;
    min-height: 20px;
}}
QComboBox:hover {{
    border-color: {accent};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid {accent};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {card};
    color: {fg};
    border: 1px solid {border};
    selection-background-color: {accent_dim};
    selection-color: {accent};
}}

/* ── 复选框 ── */
QCheckBox {{
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {border};
    background-color: {inp};
}}
QCheckBox::indicator:checked {{
    background-color: {accent};
    border-color: {accent};
}}

/* ── 标签页 ── */
QTabWidget::pane {{
    border: 1px solid {border};
    background-color: {card};
}}
QTabBar::tab {{
    background-color: {card};
    color: {fg_sec};
    border: 1px solid {border};
    border-bottom: none;
    padding: 6px 16px;
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_SM}pt;
}}
QTabBar::tab:selected {{
    background-color: {bg};
    color: {accent};
    border-bottom: 1px solid {bg};
}}
QTabBar::tab:hover {{
    color: {accent};
}}

/* ── 树/列表 ── */
QTreeWidget, QTreeView, QListWidget, QListView {{
    background-color: {card};
    color: {fg};
    border: 1px solid {border};
    alternate-background-color: {inp};
    selection-background-color: {accent_dim};
    selection-color: {accent};
}}
QTreeWidget::item, QTreeView::item {{
    padding: 4px 6px;
    border-bottom: 1px solid {border};
}}

/* ── 表格 ── */
QTableWidget, QTableView {{
    background-color: {card};
    color: {fg};
    border: 1px solid {border};
    gridline-color: {border};
    selection-background-color: {accent_dim};
    selection-color: {accent};
}}
QHeaderView::section {{
    background-color: {inp};
    color: {fg_sec};
    border: none;
    border-bottom: 1px solid {border};
    padding: 4px 8px;
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_SM}pt;
}}

/* ── 滚动条 ── */
QScrollBar:vertical {{
    background: {bg};
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {border};
    min-height: 30px;
    border-radius: 2px;
}}
QScrollBar::handle:vertical:hover {{
    background: {accent_dim};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {bg};
    height: 10px;
}}
QScrollBar::handle:horizontal {{
    background: {border};
    min-width: 30px;
    border-radius: 2px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {accent_dim};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── 进度条 ── */
QProgressBar {{
    background-color: {prog_bg};
    border: 1px solid {border};
    text-align: center;
    color: {fg};
    font-size: {FONT_SIZE_SM}pt;
    min-height: 16px;
}}
QProgressBar::chunk {{
    background-color: {prog_fg};
}}

/* ── 状态栏 ── */
QStatusBar {{
    background-color: {card};
    border-top: 1px solid {border};
    color: {fg_sec};
    font-size: {FONT_SIZE_SM}pt;
}}

/* ── 分组框 ── */
QGroupBox {{
    border: 1px solid {border};
    margin-top: 12px;
    padding: 12px 8px 8px;
    font-family: {FONT_FAMILY};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: {accent};
    font-size: {FONT_SIZE_SM}pt;
}}

/* ── 分割器 ── */
QSplitter::handle {{
    background-color: {border};
    width: 2px;
}}

/* ── 标签 ── */
QLabel {{
    color: {fg};
    background: transparent;
}}
QLabel#secondary {{
    color: {fg_sec};
    font-size: {FONT_SIZE_SM}pt;
}}
QLabel#accent {{
    color: {accent};
}}
QLabel#error {{
    color: {err};
}}

/* ── 工具提示 ── */
QToolTip {{
    background-color: {card};
    color: {fg};
    border: 1px solid {accent};
    padding: 4px 8px;
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_SM}pt;
}}

/* ── 滑块 ── */
QSlider::groove:horizontal {{
    background: {prog_bg};
    height: 4px;
}}
QSlider::handle:horizontal {{
    background: {accent};
    width: 12px;
    height: 12px;
    margin: -4px 0;
}}
QSlider::sub-page:horizontal {{
    background: {accent};
}}

/* ── 对话框 ── */
QDialog {{
    background-color: {bg};
}}
"""
