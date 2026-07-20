"""内在动机 Callback — ICM 后处理、概念 consolidation、MemoryGate 自适应."""

from model.core.train.callback_base import CallbackBase


class IntrinsicCallback(CallbackBase):
    """ICM / ConceptDiscovery / MemoryGate 步后处理与任务终化."""

    def __init__(self):
        """初始化内在动机统计."""
        self._intrinsic_stats: dict[str, list] = {
            "pred_loss": [],
            "inverse_loss": [],
            "information_gain": [],
            "uncertainty": [],
            "n_concepts": [],
        }

    def on_step_end(self, loop, result, pbar, epoch, task_id):
        """每步后记录 ICM 指标、概念 consolidation."""
        if not loop.cfg.enable_intrinsic_motivation or loop.icm is None:
            return

        # 概念 consolidation (每 500 步)
        if loop.global_step % 500 == 0 and loop.concept_discovery is not None:
            loop.concept_discovery.consolidate()

        # 记忆门控自适应
        if (
            loop.memory_gate is not None
            and loop.global_step % loop.cfg.consolidation_pipeline_interval == 0
        ):
            loop.memory_gate.adapt_thresholds()

        # 收集统计 (每步)
        if loop._icm_output:
            for k in ["pred_loss", "inverse_loss", "information_gain", "uncertainty"]:
                v = loop._icm_output.get(k)
                if v is not None:
                    self._intrinsic_stats[k].append(v)
            if loop.concept_discovery:
                self._intrinsic_stats["n_concepts"].append(
                    loop.concept_discovery.n_concepts
                )

    def on_task_end(self, loop, task_id, dataset):
        """任务结束时做概念 consolidation 并打印统计."""
        if not loop.cfg.enable_intrinsic_motivation or loop.concept_discovery is None:
            return

        # 概念 consolidation
        loop.concept_discovery.consolidate()
        n_concepts = loop.concept_discovery.n_concepts
        fragile = len(loop.concept_discovery.get_fragile_concept_ids())
        loop._log(
            f"[Intrinsic] After {task_id}: {n_concepts} concepts ({fragile} fragile)"
        )
        for cid, c in loop.concept_discovery.alive_concepts[:10]:
            loop._log(
                f"  Concept {cid}: support={c.support}, "
                f"avg_IG={c.avg_intrinsic_value:.4f}"
            )

        # 统计
        ig_mean = sum(self._intrinsic_stats["information_gain"][-500:]) / max(
            len(self._intrinsic_stats["information_gain"][-500:]), 1
        )
        loop._log(
            f"[Intrinsic Stats] IG_mean_500={ig_mean:.4f}, "
            f"n_concepts_peak={max(self._intrinsic_stats.get('n_concepts') or [0])}"
        )

        if loop.memory_gate is not None:
            gate_stats = loop.memory_gate.get_stats()
            loop._log(
                f"[MemoryGate] threshold_low={gate_stats['threshold_low']:.4f}, "
                f"threshold_high={gate_stats['threshold_high']:.4f}, "
                f"storage_ratio={gate_stats['storage_ratio']:.3f}"
            )
