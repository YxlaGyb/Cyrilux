"""结构化日志查看器 — 基于 tkinter.Text 嵌入 maliang Canvas."""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from typing import Optional

import maliang
from gui.theme import FONT_FAMILY_MONO, ThemeManager


class LogViewer:
    """带颜色分级的日志查看器 (嵌入在 maliang Canvas 上)."""

    MAX_LINES = 500

    LEVELS = {
        "DEBUG": ("log_debug", "#8080a0"),
        "INFO": ("log_info", "#c0c0e0"),
        "SUCCESS": ("log_success", "#4ade80"),
        "WARN": ("log_warn", "#fbbf24"),
        "WARNING": ("log_warn", "#fbbf24"),
        "ERROR": ("log_error", "#f87171"),
    }

    def __init__(self, master: maliang.Canvas,
                 position: tuple[int, int],
                 size: tuple[int, int],
                 theme_mgr: Optional[ThemeManager] = None):
        self._master = master
        self._theme_mgr = theme_mgr
        self._line_count = 0

        p = theme_mgr.palette if theme_mgr else None
        bg = p.card_bg if p else "#ffffff"
        fg = p.fg if p else "#1a1a2e"
        border = p.border if p else "#d0d0de"

        # Frame 嵌入 Canvas
        self._frame = tk.Frame(master, bg=bg)
        master.create_window(position[0], position[1],
                             window=self._frame,
                             width=size[0], height=size[1],
                             anchor="nw")

        self._font = tkfont.Font(family=FONT_FAMILY_MONO, size=9)
        self._text = tk.Text(self._frame, font=self._font, wrap=tk.WORD, state=tk.DISABLED,
                             bg=bg, fg=fg, insertbackground=fg,
                             border=0, highlightthickness=1, highlightbackground=border,
                             padx=6, pady=4)
        self._text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sb = tk.Scrollbar(self._frame, command=self._text.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._text.configure(yscrollcommand=sb.set)

        for level, (tag, color) in self.LEVELS.items():
            self._text.tag_configure(tag, foreground=color)
        self._text.tag_configure("timestamp", foreground="#606080")

        self._menu = tk.Menu(self._text, tearoff=0)
        self._menu.add_command(label="复制", command=self._copy_selection)
        self._menu.add_command(label="清空", command=self.clear)
        self._menu.add_command(label="全部复制", command=self._copy_all)
        self._text.bind("<Button-3>", self._show_menu)
        self._text.bind("<Control-c>", lambda e: self._copy_selection())

    def info(self, msg: str):
        self._append("INFO", msg)

    def success(self, msg: str):
        self._append("SUCCESS", msg)

    def warn(self, msg: str):
        self._append("WARN", msg)

    def error(self, msg: str):
        self._append("ERROR", msg)

    def debug(self, msg: str):
        self._append("DEBUG", msg)

    def clear(self):
        self._text.configure(state=tk.NORMAL)
        self._text.delete("1.0", tk.END)
        self._line_count = 0
        self._text.configure(state=tk.DISABLED)

    def get_text(self) -> str:
        return self._text.get("1.0", tk.END).strip()

    def _append(self, level: str, msg: str):
        tag = self.LEVELS.get(level, ("log_info", "#c0c0e0"))[0]
        self._text.configure(state=tk.NORMAL)
        if self._line_count >= self.MAX_LINES:
            self._text.delete("1.0", "2.0 linestart")
            self._line_count -= 1
        self._text.insert(tk.END, msg + "\n", tag)
        self._line_count += 1
        self._text.configure(state=tk.DISABLED)
        self._text.see(tk.END)

    def _copy_selection(self):
        try:
            sel = self._text.selection_get()
            self._frame.clipboard_clear()
            self._frame.clipboard_append(sel)
        except tk.TclError:
            pass

    def _copy_all(self):
        self._frame.clipboard_clear()
        self._frame.clipboard_append(self.get_text())

    def _show_menu(self, e):
        self._menu.tk_popup(e.x_root, e.y_root)
