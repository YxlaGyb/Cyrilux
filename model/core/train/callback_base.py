"""Callback 基类 — 所有事件方法默认空实现."""


class CallbackBase:
    """训练事件回调基类. 子类按需覆盖方法."""

    def on_fit_start(self, loop) -> None:
        """训练开始时调用 — 模型和组件已创建完毕."""

    def on_task_start(self, loop, task_id: str, dataset) -> None:
        """每个任务开始时调用 — DataLoader 已创建."""

    def on_epoch_start(self, loop, epoch: int, total_epochs: int) -> None:
        """每个 epoch 开始时调用."""

    def on_step_end(self, loop, result: dict, pbar, epoch: int, task_id: str) -> None:
        """每个 train_step 之后调用."""

    def on_task_end(self, loop, task_id: str, dataset) -> None:
        """每个任务完成后调用."""

    def on_fit_end(self, loop, task_pipelines: list) -> None:
        """全部任务完成后调用 — 最终保存之前."""
