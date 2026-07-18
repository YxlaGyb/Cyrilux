"""增强文件浏览器 — Treeview 嵌入 maliang Canvas."""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

import maliang
from gui.theme import FONT_FAMILY_MONO, ThemeManager
from gui.worker import async_task


class FileBrowser:
    """文件浏览器 (嵌入 maliang Canvas)."""

    def __init__(self, master: maliang.Canvas,
                 position: tuple[int, int],
                 size: tuple[int, int],
                 directory: str = "dataset",
                 on_select: Optional[Callable] = None,
                 theme_mgr: Optional[ThemeManager] = None):
        self._master = master
        self.directory = directory
        self.on_select_cb = on_select
        self._theme_mgr = theme_mgr
        self._entries: list[dict] = []

        p = theme_mgr.palette if theme_mgr else None
        bg = p.card_bg if p else "#ffffff"

        # Frame 嵌入 Canvas
        self._frame = tk.Frame(master, bg=bg)
        master.create_window(position[0], position[1],
                             window=self._frame,
                             width=size[0], height=size[1],
                             anchor="nw")

        self._frame.grid_rowconfigure(1, weight=1)
        self._frame.grid_columnconfigure(0, weight=1)

        # 工具栏
        toolbar = tk.Frame(self._frame, bg=bg)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self._path_var = tk.StringVar(value=directory)
        tk.Button(toolbar, text="浏览", command=self._browse_dir).pack(side=tk.RIGHT, padx=2)
        tk.Button(toolbar, text="扫描", command=self.scan).pack(side=tk.RIGHT, padx=2)
        e = tk.Entry(toolbar, textvariable=self._path_var, font=(FONT_FAMILY_MONO, 9))
        e.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Treeview
        cols = ("name", "size", "lines", "format")
        self._tree = ttk.Treeview(self._frame, columns=cols, show="headings",
                                  selectmode="browse", height=12)
        self._tree.heading("name", text="文件名")
        self._tree.heading("size", text="大小")
        self._tree.heading("lines", text="行数")
        self._tree.heading("format", text="格式")
        self._tree.column("name", width=240, minwidth=120)
        self._tree.column("size", width=80, anchor="e", minwidth=60)
        self._tree.column("lines", width=60, anchor="e", minwidth=40)
        self._tree.column("format", width=80, anchor="c", minwidth=60)
        self._tree.grid(row=1, column=0, sticky="nsew")

        sb = ttk.Scrollbar(self._frame, orient=tk.VERTICAL, command=self._tree.yview)
        sb.grid(row=1, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        # 延迟扫描
        self._frame.after(200, self.scan)

    def scan(self):
        for row in self._tree.get_children():
            self._tree.delete(row)
        self._entries = []
        base = self._path_var.get()
        if not os.path.isdir(base):
            return
        async_task(self._frame, worker=lambda: self._scan_worker(base),
                   on_done=self._scan_done)

    def _scan_worker(self, base: str) -> list[dict]:
        entries = []
        for fname in sorted(os.listdir(base)):
            fpath = os.path.join(base, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                size_kb = os.path.getsize(fpath) / 1024
                ext = os.path.splitext(fname)[1].lower()
                lines = self._count_lines(fpath, ext)
                entries.append({
                    "name": fname, "path": fpath,
                    "size_kb": size_kb, "lines": lines, "ext": ext,
                })
            except OSError:
                pass
        return entries

    def _scan_done(self, entries: list[dict]):
        try:
            self._entries = entries
            total_size = 0
            for e in entries:
                size_kb = e["size_kb"]
                size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
                total_size += size_kb
                fmt = e["ext"][1:].upper() if e["ext"] else "未知"
                self._tree.insert("", tk.END, values=(e["name"], size_str, e["lines"], fmt))
        except tk.TclError:
            pass

    def _count_lines(self, fpath: str, ext: str) -> int:
        try:
            if ext in (".jsonl", ".txt"):
                with open(fpath, encoding="utf-8") as f:
                    return sum(1 for _ in f)
            return 0
        except Exception:
            return 0

    def _browse_dir(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(initialdir=self._path_var.get())
        if d:
            self._path_var.set(d)
            self.scan()

    def _on_select(self, e):
        sel = self._tree.selection()
        if sel:
            idx = self._tree.index(sel[0])
            if idx < len(self._entries):
                if self.on_select_cb:
                    self.on_select_cb(self._entries[idx])
