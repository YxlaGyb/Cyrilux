"""睡眠 Consolidation Callback — 6a WM 驱动合并 + 6b 深度睡眠."""

import math

import torch

from model.continual.offline_replay import OfflineReplayer
from model.core.train.callback_base import CallbackBase


class SleepCallback(CallbackBase):
    """任务后 WM 驱动合并 (6a) + 全部任务后深度睡眠 (6b)."""

    def on_task_end(self, loop, task_id, dataset):
        """任务后 WM 驱动合并 (6a)."""
        # 6a: WM 驱动合并后巩固
        if (
            loop.cfg.enable_world_model
            and loop.cfg.sleep_consolidation
            and loop.world_model is not None
        ):
            self._sleep_consolidation(loop, task_id)

    def on_fit_end(self, loop, task_pipelines):
        """全部任务后深度睡眠 (6b)."""
        # 6b: Full Sleep
        if loop.cfg.enable_world_model and loop.cfg.full_sleep_after_all:
            self._full_sleep_phase(loop, task_pipelines)

    # ── 6a ──────────────────────────────────────────────────

    def _sleep_consolidation(self, loop, task_id):
        """用世界模型 transition_error 调整 store retention_weights → 重新合并."""
        if not hasattr(loop, "abstraction_bank") or loop.abstraction_bank is None:
            return
        entries = loop.abstraction_bank._store.get(task_id, [])
        if not entries:
            return

        device = loop.device
        n_adjusted = 0
        with torch.no_grad():
            wm_ctx = loop._build_world_model_context(1)
            for e in entries:
                z_top = e["z_states"][-1]
                seq_len = z_top.size(1)
                mid_idx = seq_len // 2
                z_rep = z_top[:, mid_idx : mid_idx + 1, :].to(device)
                z_pred, _ = loop.world_model(z_rep, wm_ctx)
                t_err = torch.mean((z_pred - z_rep) ** 2).item()
                decay = math.exp(-t_err * 2.0)
                old_w = e.get("retention_weight", 1.0)
                new_w = old_w * (0.5 + 0.5 * decay)
                e["retention_weight"] = new_w
                n_adjusted += 1

        if n_adjusted:
            loop.abstraction_bank.consolidate(task_id)
            loop._log(
                f"[Sleep/6a] WM-driven reconsolidation Task {task_id}: "
                f"adjusted {n_adjusted} entries"
            )

    # ── 6b ──────────────────────────────────────────────────

    def _full_sleep_phase(self, loop, task_pipelines):
        """全部任务后: 深度 SLEEP — 使用 SleepEngine."""
        if not loop.cfg.enable_deep_sleep or loop.sleep_engine is None:
            loop._log("[Sleep] Deep sleep disabled, fallback to standard replay")
            self._standard_sleep_phase(loop, task_pipelines)
            return

        loop._log("[Sleep/Deep] 开始深度睡眠 — 吸引子景观维护...")
        loop.model.train()

        if loop.landscape is not None:
            report = loop.landscape.full_landscape_report(
                loop.model,
                loop.abstraction_bank,
                pos_emb=(None, None),
                gamma=loop.cfg.gamma,
                device=loop.device,
            )
            loop._log(
                f"[Sleep] 景观报告: {report['n_prototypes_total']} prototypes, "
                f"entropy={report['entropy_metrics']['normalized_entropy']:.3f}, "
                f"collapse_ratio={report['collapse_ratio']:.3f}"
            )

        if loop.memory_bank.total < 4:
            loop._log("[Sleep] MemoryBank 样本不足, 跳过深度睡眠")
            return

        phases = ["completion", "noise", "competitive"]
        results = loop.sleep_engine.run(
            model=loop.model,
            memory_bank=loop.memory_bank,
            abstraction_bank=loop.abstraction_bank,
            device=loop.device,
            phases=phases,
        )

        for phase_name, avg_loss in results.items():
            loop._log(f"[Sleep] Phase {phase_name}: avg_loss={avg_loss:.4f}")

        for gk in list(loop.abstraction_bank._store.keys()):
            loop.abstraction_bank.consolidate(gk)
        for gk in list(loop.abstraction_bank._store_by_concept.keys()):
            loop.abstraction_bank.consolidate(gk)

        loop._log("[Sleep/Deep] 深度睡眠完成")

    def _standard_sleep_phase(self, loop, task_pipelines):
        """标准 SLEEP 回退: WM-filtered OfflineReplayer replay."""
        if not loop.cfg.enable_world_model or loop.world_model is None:
            return
        loop._log("[Sleep/6b] Full sleep phase — generating WM-filtered replay...")

        task_uncertainties: list[tuple[str, float]] = []
        for task_id, _ in task_pipelines:
            protos = loop.abstraction_bank.get_prototypes(task_id)
            if not protos:
                task_uncertainties.append((task_id, 0.5))
                continue
            errors = []
            device = loop.device
            with torch.no_grad():
                wm_ctx = loop._build_world_model_context(1)
                for z_proto in protos[:20]:
                    z_t = torch.as_tensor(z_proto, device=device).unsqueeze(0)
                    z_pred, _ = loop.world_model(z_t, wm_ctx)
                    errors.append(torch.mean((z_pred - z_t) ** 2).item())
            task_uncertainties.append((task_id, sum(errors) / max(len(errors), 1)))

        task_uncertainties.sort(key=lambda x: x[1], reverse=True)
        top_k = min(loop.cfg.sleep_replay_tasks, len(task_uncertainties))

        replayer = OfflineReplayer(
            model=loop.model,
            tokenizer=None,
            memory_bank=loop.memory_bank,
            abstraction_bank=loop.abstraction_bank,
            for_token_free=True,
            world_model=loop.world_model,
        )

        for task_id, unc in task_uncertainties[:top_k]:
            loop._log(
                f"[Sleep/6b] Replaying task {task_id} "
                f"(uncertainty={unc:.4f}), generating "
                f"{loop.cfg.sleep_replay_samples} samples..."
            )
            replayer.generate_for_task(
                task_id,
                n_samples=loop.cfg.sleep_replay_samples,
                max_length=64,
                temperature=0.8,
                enable_wm_filter=True,
                enable_wm_temperature=True,
            )
        loop._log("[Sleep/6b] Done")
