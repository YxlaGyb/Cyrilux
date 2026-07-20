"""检查点 Callback — 间隔保存、任务保存、最终保存."""

import os

import torch

from model.core.train.callback_base import CallbackBase
from model.model_cyrene import CyreneConfig


class CheckpointCallback(CallbackBase):
    """训练检查点 + unified_final 保存."""

    def __init__(self):
        """初始化检查点回调."""
        self.out_dir: str = ""

    def on_step_end(self, loop, result, pbar, epoch, task_id):
        """每步后按间隔保存检查点."""
        if loop.cfg.save_interval <= 0:
            return
        self.out_dir = os.path.join(os.getcwd(), loop.cfg.out_dir)

        if loop.global_step % loop.cfg.save_interval == 0 or loop.global_step == 1:
            ckpt_path = os.path.join(
                self.out_dir, f"unified_ckpt_s{loop.global_step}.pt"
            )
            self.save_checkpoint(loop, ckpt_path, epoch, task_id)
            if loop.cfg.progress_callback:
                loop.cfg.progress_callback(
                    {
                        "type": "checkpoint",
                        "step": loop.global_step,
                        "checkpoint_path": ckpt_path,
                    }
                )

    def on_task_end(self, loop, task_id, dataset):
        """任务结束时保存任务最终检查点."""
        self.out_dir = os.path.join(os.getcwd(), loop.cfg.out_dir)
        self.save_checkpoint(
            loop,
            os.path.join(self.out_dir, f"task_{task_id}_final.pt"),
            loop.cfg.epochs - 1,
            task_id,
        )

    def on_fit_end(self, loop, task_pipelines):
        """全部任务结束时保存 unified_final.pt."""
        self.out_dir = os.path.join(os.getcwd(), loop.cfg.out_dir)
        # 保存 unified_final.pt
        loop.model.cpu()
        fp = os.path.join(self.out_dir, "unified_final.pt")
        torch.save(loop.model.state_dict(), fp)
        loop._log(
            f"unified_final saved → {fp} ({os.path.getsize(fp) // 1024 // 1024}MB)"
        )

    # ── 内部 ────────────────────────────────────────────────

    def save_checkpoint(self, loop, path: str, epoch: int = 0, task_id: str = None):
        """保存检查点到磁盘."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self._build_ckpt(loop, epoch, task_id), path)
        loop._log(f"Checkpoint saved → {path}")

    def _build_ckpt(self, loop, epoch, task_id=None, metrics=None) -> dict:
        ckpt = {
            "epoch": epoch,
            "step": loop.global_step,
            "model_state": loop.model.state_dict(),
            "lm_config": CyreneConfig(
                hidden_size=loop.cfg.hidden_size,
                num_hidden_layers=loop.cfg.num_hidden_layers,
                use_moe=loop.cfg.use_moe,
            ),
            "config": loop.cfg.to_dict(),
        }
        if loop.cfg.enable_world_model and loop.world_model is not None:
            ckpt["world_model_state"] = loop.world_model.state_dict()
        if loop.cfg.enable_intrinsic_motivation and loop.icm is not None:
            ckpt["icm_state"] = loop.icm.state_dict()
        if metrics:
            ckpt.update(metrics)
        ckpt["memory_bank"] = loop.memory_bank.state_dict()
        ckpt["abstraction_bank"] = loop.abstraction_bank.state_dict()
        if loop.sleep_engine is not None:
            ckpt["sleep_engine"] = loop.sleep_engine.state_dict()
        if task_id:
            ckpt["task_id"] = task_id
        return ckpt
