"""
双主题系统 (深色 / 浅色) — 基于 maliang 主题系统.

用法:
    theme_mgr = ThemeManager()
    theme_mgr.apply()               # 初始应用 (浅色)
    theme_mgr.toggle()              # 切换主题

颜色常量在 PALETTES[Theme] 中, 供自定义组件使用.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass

from maliang.theme import set_color_mode


class Theme(Enum):
    LIGHT = "light"
    DARK = "dark"


@dataclass
class ThemePalette:
    """单主题色板."""

    bg: str
    card_bg: str
    border: str
    fg: str
    fg_secondary: str
    select_bg: str
    select_fg: str
    accent: str
    accent_light: str
    success: str
    error: str
    warning: str
    input_bg: str
    disabled_fg: str
    progress_bg: str
    progress_fg: str


# ── 色板定义 ─────────────────────────────────────────────────────

PALETTES = {
    Theme.LIGHT: ThemePalette(
        bg="#f5f5f8",
        card_bg="#ffffff",
        border="#d0d0de",
        fg="#1a1a2e",
        fg_secondary="#6b6b8a",
        select_bg="#6c5ce7",
        select_fg="#ffffff",
        accent="#6c5ce7",
        accent_light="#8b7cf7",
        success="#22c55e",
        error="#ef4444",
        warning="#eab308",
        input_bg="#ffffff",
        disabled_fg="#a0a0b8",
        progress_bg="#e0e0ee",
        progress_fg="#6c5ce7",
    ),
    Theme.DARK: ThemePalette(
        bg="#1e1e2e",
        card_bg="#2a2a3e",
        border="#3e3e5e",
        fg="#e0e0f0",
        fg_secondary="#9090b0",
        select_bg="#7c5cfc",
        select_fg="#ffffff",
        accent="#7c5cfc",
        accent_light="#9b7ffc",
        success="#4ade80",
        error="#f87171",
        warning="#fbbf24",
        input_bg="#363654",
        disabled_fg="#606080",
        progress_bg="#363654",
        progress_fg="#7c5cfc",
    ),
}

# ── 字体 ──────────────────────────────────────────────────────────

FONT_FAMILY = "Segoe UI"
FONT_FAMILY_MONO = "Consolas"


# ── 主题管理器 ────────────────────────────────────────────────────


class ThemeManager:
    """管理 maliang 主题切换与自定义颜色."""

    def __init__(self, initial: Theme = Theme.LIGHT):
        self._theme = initial
        self._callbacks: list[callable] = []

    # ── 属性 ──

    @property
    def palette(self) -> ThemePalette:
        return PALETTES[self._theme]

    @property
    def theme(self) -> Theme:
        return self._theme

    def is_dark(self) -> bool:
        return self._theme == Theme.DARK

    # ── 观察者 ──

    def on_toggle(self, cb: callable):
        """注册主题切换回调."""
        self._callbacks.append(cb)

    # ── 核心 ──

    def apply(self, root=None):
        """将当前主题应用到窗口."""
        set_color_mode(self._theme.value)
        for cb in self._callbacks:
            try:
                cb(self._theme)
            except Exception:
                pass

    def toggle(self):
        """切换深色/浅色主题."""
        self._theme = Theme.LIGHT if self._theme == Theme.DARK else Theme.DARK
        self.apply()

    def theme_name(self) -> str:
        return "浅色" if not self.is_dark() else "深色"
