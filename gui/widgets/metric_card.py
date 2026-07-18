"""KPI 卡片组件 — 基于 maliang Widget + shapes/Text."""

from __future__ import annotations

from typing import Optional

import maliang
from maliang import shapes
from maliang.core.virtual import Widget

from gui.theme import FONT_FAMILY, ThemeManager


class MetricCard:
    """带标题/数值/增量/描述的可复用 KPI 卡片 (绘制在 maliang Widget 上)."""

    def __init__(self, master: maliang.Canvas,
                 position: tuple[int, int],
                 size: tuple[int, int],
                 title: str = "", value: str = "--",
                 delta: Optional[str] = None,
                 delta_positive: Optional[bool] = None,
                 subtitle: str = "",
                 theme_mgr: Optional[ThemeManager] = None):
        self._master = master
        self._theme_mgr = theme_mgr
        self._position = position
        self._size = size

        p = theme_mgr.palette if theme_mgr else None
        card_bg = p.card_bg if p else "#ffffff"
        fg = p.fg if p else "#1a1a2e"
        fg2 = p.fg_secondary if p else "#6b6b8a"
        border = p.border if p else "#d0d0de"

        # 创建虚拟 Widget 作为形状和文本的容器
        self._widget = Widget(master, position=position, size=size)

        # 卡片背景圆角矩形 (相对于 Widget 左上角)
        # 注意: RoundedRectangle 内部 display() 硬编码了 outline="",
        # 因此不能通过 kwargs 传 outline, 以免冲突。
        shapes.RoundedRectangle(
            self._widget,
            relative_position=(0, 0),
            size=size,
            radius=8,
            fill=card_bg,
        )

        # 标题 (相对于 Widget 左上角的偏移)
        self._title = maliang.Text(
            self._widget, relative_position=(12, 8),
            text=title, fontsize=9, fill=fg,
        )

        # 数值
        self._value_label = maliang.Text(
            self._widget, relative_position=(12, 28),
            text=value, fontsize=22, weight="bold", fill=fg,
        )

        # 增量行
        if delta is not None:
            delta_fg = "#4ade80" if delta_positive else "#f87171"
            if delta_positive is None:
                delta_fg = fg2
            self._delta_label = maliang.Text(
                self._widget, relative_position=(12, size[1] - 24),
                text=delta, fontsize=9, fill=delta_fg,
            )

        # 副标题
        if subtitle:
            maliang.Text(
                self._widget, relative_position=(12, size[1] - 12),
                text=subtitle, fontsize=8, fill=fg2,
            )

    def set_value(self, value: str):
        self._value_label.text = value

    def set_delta(self, delta: str, positive: Optional[bool] = None):
        """更新增量值."""
        if hasattr(self, '_delta_label'):
            self._delta_label.text = delta
