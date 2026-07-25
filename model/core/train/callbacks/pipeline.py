"""持续巩固管道 Callback
ConsolidationPipeline tick/force 调度.
"""

from model.core.train.callback_base import CallbackBase


class PipelineCallback(CallbackBase):
    """ConsolidationPipeline tick + force_consolidate 降频调度."""

    def __init__(self):
        """初始化多巴胺窗口."""
        self._dopamine_window: list[float] = []

    def on_step_end(self, loop, result, pbar, epoch, task_id):
        """每步后 tick consolidation pipeline."""
        if loop.consolidation_pipeline is None:
            return
        if loop.global_step % loop.cfg.consolidation_pipeline_interval != 0:
            return

        m = result
        current_D = m.get("D", 0.0)
        if isinstance(current_D, float) and (current_D != current_D):
            current_D = 0.0

        self._dopamine_window.append(current_D)
        if len(self._dopamine_window) > 30:
            self._dopamine_window.pop(0)

        tick_result = {"triggered": None}
        try:
            tick_result = loop.consolidation_pipeline.tick(
                loop.global_step,
                loop.model,
                loop.memory_bank,
                loop.abstraction_bank,
                device=loop.device,
                dopamine_score=current_D,
            )
        except Exception as pipe_err:
            loop._log(f"[Pipeline] tick 忽略异常: {pipe_err}")

        if tick_result.get("triggered") and loop.global_step % 500 == 0:
            loop._log(f"[Pipeline] {tick_result['triggered']}")

        mean_D = sum(self._dopamine_window) / max(len(self._dopamine_window), 1)
        mean_D_trigger = 0.70
        if mean_D > mean_D_trigger and len(self._dopamine_window) >= 20:
            force_result = {}
            try:
                force_result = loop.consolidation_pipeline.force_consolidate(
                    loop.global_step,
                    loop.model,
                    loop.memory_bank,
                    loop.abstraction_bank,
                    device=loop.device,
                )
                self._dopamine_window.clear()
            except Exception as pipe_err:
                loop._log(f"[Pipeline] force_consolidate 忽略异常: {pipe_err}")
            if force_result.get("triggered") and loop.global_step % 500 == 0:
                loop._log(f"[Pipeline] force:{force_result['triggered']}")
