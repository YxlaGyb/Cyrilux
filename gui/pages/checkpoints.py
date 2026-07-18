"""检查点管理页 — 浏览 / 排序 / 加载 / 删除."""

from __future__ import annotations

import os
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import maliang

from gui.theme import FONT_FAMILY
from gui.state import ExperimentState
from gui.worker import async_task


class Page(maliang.Canvas):
    """检查点管理页面 (maliang Canvas)."""

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
        # 延迟扫描, 避免阻塞界面
        self.after(300, self._scan)

    def _on_resize(self, event):
        try:
            self.itemconfigure(self._inner_id, width=event.width, height=event.height)
        except tk.TclError:
            pass

    def _build(self):
        inner = self._inner
        p = self.theme_mgr.palette

        inner.grid_rowconfigure(0, weight=0)  # title
        inner.grid_rowconfigure(1, weight=0)  # toolbar
        inner.grid_rowconfigure(2, weight=1)  # table
        inner.grid_rowconfigure(3, weight=0)  # detail
        inner.grid_columnconfigure(0, weight=1)

        # 标题
        tk.Label(inner, text="检查点", font=(FONT_FAMILY, 16, "bold"),
                 bg=p.bg, fg=p.fg, anchor="w").grid(row=0, column=0, sticky="ew",
                                                      padx=20, pady=(16, 8))

        # 工具栏
        toolbar = tk.Frame(inner, bg=p.bg)
        toolbar.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 8))

        tk.Label(toolbar, text="输出目录:", font=(FONT_FAMILY, 9),
                 bg=p.bg, fg=p.fg).pack(side=tk.LEFT)
        self._dir_var = tk.StringVar(value="ola_out")
        e = tk.Entry(toolbar, textvariable=self._dir_var, font=(FONT_FAMILY, 9),
                     width=40, bg=p.input_bg, fg=p.fg, relief=tk.FLAT)
        e.pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="浏览", command=self._browse_dir,
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="扫描", command=self._scan,
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(side=tk.LEFT, padx=2)

        self._scan_status = tk.Label(toolbar, text="", font=(FONT_FAMILY, 9),
                                     bg=p.bg, fg=p.fg_secondary)
        self._scan_status.pack(side=tk.RIGHT, padx=4)

        # 检查点表格
        table_frame = tk.Frame(inner, bg=p.bg)
        table_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 8))
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        cols = ("dir", "filename", "size", "step", "mtime")
        self._tree = ttk.Treeview(table_frame, columns=cols, show="headings",
                                  selectmode="browse")
        self._tree.heading("dir", text="目录")
        self._tree.heading("filename", text="文件名")
        self._tree.heading("size", text="大小")
        self._tree.heading("step", text="步数")
        self._tree.heading("mtime", text="修改时间")
        self._tree.column("dir", width=100)
        self._tree.column("filename", width=200)
        self._tree.column("size", width=80, anchor="e")
        self._tree.column("step", width=60, anchor="e")
        self._tree.column("mtime", width=140)
        self._tree.grid(row=0, column=0, sticky="nsew")

        sb = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self._tree.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=sb.set)

        # 右键菜单
        self._context_menu = tk.Menu(self._tree, tearoff=0, bg=p.card_bg, fg=p.fg)
        self._context_menu.add_command(label="加载到评估", command=self._load_to_eval)
        self._context_menu.add_command(label="加载到训练", command=self._load_to_train)
        self._context_menu.add_separator()
        self._context_menu.add_command(label="删除", command=self._delete_checkpoint)
        self._context_menu.add_command(label="打开所在文件夹", command=self._open_folder)
        self._context_menu.add_command(label="重命名", command=self._rename_checkpoint)
        self._tree.bind("<Button-3>", self._show_context_menu)

        # 详情面板
        detail_frame = tk.LabelFrame(inner, text="详情", font=(FONT_FAMILY, 10, "bold"),
                                     bg=p.bg, fg=p.fg, padx=8, pady=4)
        detail_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 8))

        self._detail_var = tk.StringVar(value="选择检查点查看详情")
        tk.Label(detail_frame, textvariable=self._detail_var,
                 font=(FONT_FAMILY, 9), bg=p.bg, fg=p.fg,
                 wraplength=700).pack(fill=tk.X)

        self._tree.bind("<<TreeviewSelect>>", self._on_select)

    def _scan(self):
        """异步扫描检查点目录 (后台线程)."""
        for row in self._tree.get_children():
            self._tree.delete(row)

        base = self._dir_var.get()
        if not os.path.isdir(base):
            self._scan_status.configure(text="目录不存在")
            return

        self._scan_status.configure(text="扫描中...")
        async_task(self, worker=lambda: self._scan_worker(base),
                   on_done=self._scan_done)

    def _scan_worker(self, base: str) -> list[dict]:
        """后台线程: 扫描 .pt 文件 + 统计."""
        self.state.scan_checkpoints(base)
        return list(self.state.checkpoint_registry)

    def _scan_done(self, entries: list[dict]):
        """主线程: 填充 Treeview."""
        try:
            import datetime
            for cp in sorted(entries, key=lambda x: x.get("step", 0), reverse=True):
                mtime_str = ""
                if cp.get("mtime"):
                    mtime_str = datetime.datetime.fromtimestamp(cp["mtime"]).strftime("%Y-%m-%d %H:%M")
                    cp["mtime_str"] = mtime_str
                self._tree.insert("", tk.END, values=(
                    cp.get("dir", ""),
                    cp.get("filename", ""),
                    f"{cp.get('size_kb', 0):.0f} KB",
                    str(cp.get("step", "")),
                    mtime_str,
                ))

            self._scan_status.configure(text=f"共 {len(entries)} 个检查点")
        except tk.TclError:
            pass

    def _browse_dir(self):
        d = filedialog.askdirectory(initialdir=self._dir_var.get())
        if d:
            self._dir_var.set(d)
            self._scan()

    def _on_select(self, e):
        sel = self._tree.selection()
        if not sel:
            return
        values = self._tree.item(sel[0], "values")
        if values:
            self._detail_var.set(f"文件: {values[0]}/{values[1]} | 步数: {values[3]} | 大小: {values[2]}")

    def _show_context_menu(self, e):
        sel = self._tree.selection()
        if sel:
            self._context_menu.tk_popup(e.x_root, e.y_root)

    def _get_selected_path(self) -> str | None:
        sel = self._tree.selection()
        if not sel:
            return None
        values = self._tree.item(sel[0], "values")
        if not values:
            return None
        base = self._dir_var.get()
        return os.path.join(base, str(values[0]), str(values[1]))

    def _load_to_eval(self):
        path = self._get_selected_path()
        if path:
            self.state.current_model_path = path
            self.app.status_message(f"检查点已加载: {os.path.basename(path)}")
            self.app.show_page("evaluation")

    def _load_to_train(self):
        path = self._get_selected_path()
        if path:
            self.state.current_model_path = path
            self.app.status_message(f"检查点已加载到训练: {os.path.basename(path)}")
            self.app.show_page("training")

    def _delete_checkpoint(self):
        path = self._get_selected_path()
        if path and messagebox.askyesno("确认删除", f"确定删除 {os.path.basename(path)}?"):
            try:
                os.remove(path)
                self._scan()
                self.app.status_message("已删除")
            except Exception as e:
                messagebox.showerror("错误", str(e))

    def _open_folder(self):
        path = self._get_selected_path()
        if path:
            folder = os.path.dirname(path)
            os.startfile(folder)

    def _rename_checkpoint(self):
        path = self._get_selected_path()
        if not path:
            return
        from tkinter import simpledialog
        new_name = simpledialog.askstring("重命名", "新文件名:",
                                          initialvalue=os.path.basename(path))
        if new_name:
            new_path = os.path.join(os.path.dirname(path), new_name)
            try:
                os.rename(path, new_path)
                self._scan()
                self.app.status_message("已重命名")
            except Exception as e:
                messagebox.showerror("错误", str(e))
