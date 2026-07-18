"""配置模板管理页 — 保存 / 加载 / 导入 / 导出参数模板."""

from __future__ import annotations

import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import maliang

from gui.theme import FONT_FAMILY
from gui.state import ExperimentState
from gui.worker import async_task


TEMPLATE_DIR = "ola_out/configs"


class Page(maliang.Canvas):
    """配置模板管理页面 (maliang Canvas)."""

    def __init__(self, parent, state: ExperimentState, app, **kw):
        super().__init__(parent, expand="xy", **kw)
        self.state = state
        self.app = app
        self.theme_mgr = state.theme_mgr
        self._current_name: str | None = None

        # 确保模板目录存在
        os.makedirs(TEMPLATE_DIR, exist_ok=True)

        p = self.theme_mgr.palette
        self._inner = tk.Frame(self, bg=p.bg)
        self._inner_id = self.create_window(0, 0, window=self._inner, anchor="nw")
        self.bind("<Configure>", self._on_resize)
        self._build()
        # 延迟扫描, 不阻塞页面切换
        self.after(200, self._deferred_refresh)

    def _on_resize(self, event):
        try:
            self.itemconfigure(self._inner_id, width=event.width, height=event.height)
        except tk.TclError:
            pass

    def _build(self):
        inner = self._inner
        p = self.theme_mgr.palette

        inner.grid_rowconfigure(0, weight=0)  # title
        inner.grid_rowconfigure(1, weight=1)  # list + detail
        inner.grid_rowconfigure(2, weight=0)  # buttons
        inner.grid_columnconfigure(0, weight=0)
        inner.grid_columnconfigure(1, weight=1)

        # 标题
        tk.Label(inner, text="配置模板", font=(FONT_FAMILY, 16, "bold"),
                 bg=p.bg, fg=p.fg, anchor="w").grid(row=0, column=0, columnspan=2,
                                                      sticky="ew", padx=20, pady=(16, 8))

        # ── 左侧: 模板列表 ──
        list_frame = tk.Frame(inner, bg=p.bg)
        list_frame.grid(row=1, column=0, sticky="nsew", padx=(20, 8), pady=(0, 8))

        tk.Label(list_frame, text="模板列表", font=(FONT_FAMILY, 10, "bold"),
                 bg=p.bg, fg=p.fg, anchor="w").pack(fill=tk.X)

        self._list_tree = ttk.Treeview(list_frame, columns=("group",), show="tree", height=15)
        self._list_tree.pack(fill=tk.BOTH, expand=True, pady=4)

        # 分组
        self._groups = {}
        for group in ["训练", "PC", "持续学习", "QAT", "全量"]:
            node = self._list_tree.insert("", tk.END, text=group, open=True)
            self._groups[group] = node

        self._list_tree.bind("<<TreeviewSelect>>", self._on_select)

        # ── 右侧: 模板详情 ──
        detail_frame = tk.Frame(inner, bg=p.bg)
        detail_frame.grid(row=1, column=1, sticky="nsew", padx=(8, 20), pady=(0, 8))
        detail_frame.grid_rowconfigure(1, weight=1)
        detail_frame.grid_columnconfigure(0, weight=1)

        tk.Label(detail_frame, text="参数详情", font=(FONT_FAMILY, 10, "bold"),
                 bg=p.bg, fg=p.fg, anchor="w").grid(row=0, column=0, sticky="ew", pady=(0, 4))

        # 参数表格
        cols = ("key", "value")
        self._detail_tree = ttk.Treeview(detail_frame, columns=cols, show="headings", height=12)
        self._detail_tree.heading("key", text="参数名")
        self._detail_tree.heading("value", text="数值")
        self._detail_tree.column("key", width=120)
        self._detail_tree.column("value", width=100, anchor="e")
        self._detail_tree.grid(row=1, column=0, sticky="nsew")

        sb = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=self._detail_tree.yview)
        sb.grid(row=1, column=1, sticky="ns")
        self._detail_tree.configure(yscrollcommand=sb.set)

        # 描述
        tk.Label(detail_frame, text="描述:", font=(FONT_FAMILY, 9),
                 bg=p.bg, fg=p.fg).grid(row=2, column=0, sticky="ew", pady=(4, 0))
        self._desc_text = tk.Text(detail_frame, height=3, font=(FONT_FAMILY, 9),
                                  bg=p.input_bg, fg=p.fg, relief=tk.FLAT)
        self._desc_text.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(2, 0))

        # ── 底部按钮 ──
        btn_frame = tk.Frame(inner, bg=p.bg)
        btn_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 8))

        tk.Button(btn_frame, text="保存当前配置", command=self._save_from_state,
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="加载到训练页", command=self._load_to_training,
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="导出 JSON", command=self._export_json,
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="导入 JSON", command=self._import_json,
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="删除", command=self._delete_template,
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(side=tk.LEFT, padx=2)

    def _deferred_refresh(self):
        """延迟启动异步刷新."""
        self._refresh_list()

    def _refresh_list(self):
        """异步刷新模板列表 (后台线程)."""
        # 清除子节点
        for group in self._groups.values():
            for child in self._list_tree.get_children(group):
                self._list_tree.delete(child)

        if not os.path.isdir(TEMPLATE_DIR):
            return

        async_task(self, worker=self._refresh_worker,
                   on_done=self._refresh_done)

    def _refresh_worker(self) -> list[tuple[str, str, str, str]]:
        """后台线程: 扫描模板目录, 返回 [(name, group, fpath), ...]."""
        results = []
        for fname in sorted(os.listdir(TEMPLATE_DIR)):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(TEMPLATE_DIR, fname), encoding="utf-8") as f:
                        data = json.load(f)
                    name = data.get("name", fname[:-5])
                    group = data.get("group", "全量")
                    fpath = os.path.join(TEMPLATE_DIR, fname)
                    results.append((name, group, fpath))
                except Exception:
                    pass
        return results

    def _refresh_done(self, results: list[tuple[str, str, str, str]]):
        """主线程: 填充 Treeview."""
        for name, group, fpath in results:
            parent = self._groups.get(group, self._groups["全量"])
            self._list_tree.insert(parent, tk.END, text=name,
                                   values=(fpath,))

    def _on_select(self, e):
        sel = self._list_tree.selection()
        if not sel:
            return
        filepath = self._list_tree.item(sel[0], "values")
        if not filepath:
            return
        path = filepath[0]
        if not os.path.isfile(path):
            return

        # 异步加载
        async_task(self, worker=lambda: self._load_worker(path),
                   on_done=self._load_done,
                   on_error=lambda ex: messagebox.showerror("错误", f"无法加载模板: {ex}"))

    def _load_worker(self, path: str) -> tuple[str, dict, str]:
        """后台线程: 加载 JSON 模板."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        name = data.get("name", os.path.basename(path)[:-5])
        return name, data.get("params", {}), data.get("description", "")

    def _load_done(self, result: tuple[str, dict, str]):
        """主线程: 更新参数表和描述."""
        name, params, description = result
        self._current_name = name

        for row in self._detail_tree.get_children():
            self._detail_tree.delete(row)
        for key, val in params.items():
            self._detail_tree.insert("", tk.END, values=(key, str(val)))

        self._desc_text.delete("1.0", tk.END)
        self._desc_text.insert("1.0", description)

    def _save_from_state(self):
        """将当前 state 保存为模板."""
        name = self._ask_name("保存模板", "模板名称:")
        if not name:
            return

        p = self.theme_mgr.palette

        # 选择分组
        group_win = tk.Toplevel(self, bg=p.bg)
        group_win.title("选择分组")
        group_win.geometry("300x150")
        group_var = tk.StringVar(value="全量")

        tk.Label(group_win, text="选择分组:", font=(FONT_FAMILY, 9),
                 bg=p.bg, fg=p.fg).pack(pady=8)
        for g in ["训练", "PC", "持续学习", "QAT", "全量"]:
            tk.Radiobutton(group_win, text=g, variable=group_var, value=g,
                           font=(FONT_FAMILY, 9),
                           bg=p.bg, fg=p.fg).pack(anchor="w", padx=20)
        tk.Button(group_win, text="确定", command=group_win.destroy,
                  font=(FONT_FAMILY, 9), bg=p.card_bg, fg=p.fg).pack(pady=8)
        self.wait_window(group_win)

        desc = self._desc_text.get("1.0", tk.END).strip()

        data = {
            "name": name,
            "group": group_var.get(),
            "description": desc,
            "params": dict(self.state.config),
        }

        fpath = os.path.join(TEMPLATE_DIR, f"{name}.json")
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self._refresh_list()
        self.app.status_message(f"模板 '{name}' 已保存")

    def _load_to_training(self):
        """加载当前模板到训练页配置."""
        if not self._current_name:
            messagebox.showinfo("提示", "请先选择一个模板")
            return
        # 更新 state
        for row in self._detail_tree.get_children():
            key, val = self._detail_tree.item(row, "values")
            if key in self.state.config:
                self.state.config_set(key, val)
        self.app.status_message(f"模板 '{self._current_name}' 已加载")
        self.app.show_page("training")

    def _export_json(self):
        if not self._current_name:
            messagebox.showinfo("提示", "请先选择一个模板")
            return
        src = os.path.join(TEMPLATE_DIR, f"{self._current_name}.json")
        if not os.path.isfile(src):
            messagebox.showerror("错误", "模板文件不存在")
            return
        dst = filedialog.asksaveasfilename(
            title="导出模板",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile=f"{self._current_name}.json",
        )
        if dst:
            import shutil
            shutil.copy(src, dst)
            self.app.status_message("已导出")

    def _import_json(self):
        path = filedialog.askopenfilename(
            title="导入模板",
            filetypes=[("JSON", "*.json")],
        )
        if path:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if "params" not in data:
                    data = {"name": os.path.basename(path)[:-5], "params": data, "group": "全量"}
                name = data.get("name", os.path.basename(path)[:-5])
                dst = os.path.join(TEMPLATE_DIR, f"{name}.json")
                with open(dst, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                self._refresh_list()
                self.app.status_message(f"已导入: {name}")
            except Exception as e:
                messagebox.showerror("错误", str(e))

    def _delete_template(self):
        if not self._current_name:
            messagebox.showinfo("提示", "请先选择一个模板")
            return
        if messagebox.askyesno("确认删除", f"删除模板 '{self._current_name}'?"):
            fpath = os.path.join(TEMPLATE_DIR, f"{self._current_name}.json")
            if os.path.isfile(fpath):
                os.remove(fpath)
            self._current_name = None
            self._detail_tree.delete(*self._detail_tree.get_children())
            self._desc_text.delete("1.0", tk.END)
            self._refresh_list()
            self.app.status_message("已删除")

    def _ask_name(self, title: str, prompt: str) -> str | None:
        from tkinter import simpledialog
        return simpledialog.askstring(title, prompt)
