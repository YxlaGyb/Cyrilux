"""异步工作线程池 — 将阻塞 I/O 转移到后台线程，不阻塞 GUI 主线程.

用法:
    from gui.worker import async_task

    def on_scan(self):
        self.status_var.set("扫描中...")
        async_task(self, worker=self._scan_worker, on_done=self._scan_done)

    def _scan_worker(self):
        # 在后台线程执行 — 不要操作 tkinter 组件
        return glob.glob("**/*.pt", recursive=True)

    def _scan_done(self, result):
        # 在主线程更新 UI
        self.tree.insert(...)
"""

from __future__ import annotations

import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

_EXECUTOR = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="gui_worker",
)


def async_task(
    root: tk.Misc,
    *,
    worker: Callable[[], Any],
    on_done: Callable[[Any], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
):
    """在后台线程执行 worker(), 完成后通过 root.after() 在主线程回调.

    参数:
        root: 任意 tk widget (用于 after 调度到主线程)
        worker:  后台线程执行的函数, 返回值传给 on_done
        on_done: 可选, 主线程回调接收 worker 返回值
        on_error: 可选, 主线程错误回调接收 Exception
    """
    def _run():
        try:
            result = worker()
            if on_done is not None:
                root.after(0, on_done, result)
        except Exception as e:
            if on_error is not None:
                root.after(0, on_error, e)
            else:
                import traceback
                traceback.print_exc()

    _EXECUTOR.submit(_run)
