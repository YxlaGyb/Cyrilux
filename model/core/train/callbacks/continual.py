"""持续学习 Callback — 记忆回放、遗忘嗅探、抽象漂移、任务终化、跨任务评估."""

import json
import math
import os

import torch

from model.continual.abstraction_bank import compute_layer_importance
from model.core.train.callback_base import CallbackBase


class ContinualCallback(CallbackBase):
    """MemoryBank 回放、遗忘嗅探+修复、抽象漂移检测、任务终化、跨任务遗忘评估."""

    def __init__(self):
        """初始化持续学习回调."""
        self._surprise_buffer: list[float] = []
        self.forgetting_log: list[dict] = []

    def on_step_end(self, loop, result, pbar, epoch, task_id):
        """每步后: 回放、嗅探、抽象更新."""
        m = result

        # Sniffer 更新
        loop.sniffer.update_surprise(m.get("world_surprise", 0.0))

        # 记录 surprise 采样
        if loop.global_step % 100 == 0:
            self._surprise_buffer.append(m.get("world_surprise", 0.0))

        # ── 持续学习: 记忆回放 ──
        self._maybe_replay(loop)
        # ── AbstractionBank 回放 ──
        self._maybe_abstraction_replay(loop)
        # ── 遗忘嗅探 + 修复 ──
        self._maybe_sniff_forgetting(loop)
        # ── 抽象漂移检测 ──
        self._maybe_sniff_abstraction(loop)

    def on_task_end(self, loop, task_id, dataset):
        """任务结束时: 终化、抽象合并、跨任务评估."""
        # ── 任务完成 → exemplar 采样 ──
        self.finalize_task(loop, task_id, dataset)

        # ── 遗忘评估 ──
        loop._trained_tasks.append(task_id)
        eval_ds_list = [
            (tid, ds) for tid, ds in loop._task_pipelines if tid in loop._trained_tasks
        ]
        eval_results = self.evaluate_cross_tasks(loop, eval_ds_list)
        loop._log(f"[Eval] After Task {task_id}:")
        for tid, metrics in eval_results.items():
            marker = " ← trained" if tid == task_id else ""
            ce = metrics["ce"]
            ppl = metrics["ppl"]
            loop._log(f"  Task {tid}: CE={ce:.4f}, PPL={ppl:.2f}{marker}")

        # ── forgetting_log ──
        surprise_timeline = []
        if loop.cfg.enable_world_model and self._surprise_buffer:
            surprise_timeline = list(self._surprise_buffer)

        self.forgetting_log.append(
            {
                "after_task": task_id,
                "results": eval_results,
                "avg_world_surprise": (
                    sum(surprise_timeline) / max(len(surprise_timeline), 1)
                    if surprise_timeline
                    else None
                ),
                "surprise_timeline": surprise_timeline,
            }
        )
        self._surprise_buffer.clear()

        out_dir = os.path.join(os.getcwd(), loop.cfg.out_dir)
        with open(os.path.join(out_dir, "forgetting_log.json"), "w") as f:
            json.dump(self.forgetting_log, f, indent=2)

    # ── 内部方法 ─────────────────────────────────────────────

    def _maybe_replay(self, loop):
        if loop.memory_bank.total <= 0:
            return

        # 海马体快速回放
        if (
            loop.hippocampus.size > 0
            and loop.global_step % (loop.cfg.replay_ratio * 2) == 0
            and not (
                loop.sniffer.is_repairing
                if hasattr(loop.sniffer, "is_repairing")
                else False
            )
        ):
            hc_batch = loop.hippocampus.sample_for_replay(
                loop.cfg.batch_size // 4, device=loop.device
            )
            if hc_batch is not None:
                replay_byte_hc, replay_label_hc = hc_batch
                replay_byte_hc = torch.stack(
                    [
                        replay_byte_hc.float(),
                        torch.full_like(
                            replay_byte_hc, 2.0, dtype=torch.float, device=loop.device
                        ),
                    ],
                    dim=1,
                )
                loop._hebbian_update_on_data(
                    replay_byte_hc, replay_label_hc, stride=loop.cfg.replay_stride
                )

        if loop.global_step % loop.cfg.replay_ratio != 0:
            return
        if loop.sniffer.is_repairing:
            return

        # 采样策略
        if loop.cfg.enable_intrinsic_motivation and loop.icm is not None:
            strategy = "intrinsic"
        elif loop.cfg.enable_world_model and loop.world_model is not None:
            strategy = (
                "world_model"
                if loop._last_world_surprise >= loop.cfg.world_model_surprise_threshold
                else "dopamine"
            )
        else:
            strategy = "dopamine"

        replay_ex = loop.memory_bank.sample(loop.cfg.batch_size, strategy=strategy)
        if not replay_ex:
            return

        replay_byte = torch.stack([ex.byte_tensor for ex in replay_ex], dim=0).to(
            loop.device
        )
        replay_label = torch.stack([ex.label_tensor for ex in replay_ex], dim=0).to(
            loop.device
        )
        loop._hebbian_update_on_data(
            replay_byte, replay_label, stride=loop.cfg.replay_stride
        )

        # 刷新 replay_priority
        if loop.cfg.enable_world_model and loop.world_model is not None:
            with torch.no_grad():
                pos_emb = loop.model.get_position_embeddings(
                    replay_byte.size(-1), loop.device
                )
                z_rp, _ = loop.model.forward_with_ce(replay_byte, replay_label, pos_emb)
                z_top = z_rp[-1].detach()
                ctx = loop._build_world_model_context(replay_byte.size(0))
                _, uncertainty = loop.world_model(z_top, ctx)
                new_surprise = uncertainty.mean().item()
            for ex in replay_ex:
                ex.transition_surprise = new_surprise
                ex.replay_priority = (
                    max(ex.dopamine_score, 0.1)
                    + max(new_surprise, 0.0)
                    + ex.intrinsic_value
                )
        elif loop.cfg.enable_intrinsic_motivation and loop.icm is not None:
            if loop._icm_output:
                info_gain = loop._icm_output.get("information_gain", 0.0)
                for ex in replay_ex:
                    ex.replay_priority = max(ex.dopamine_score, 0.1) + info_gain

    def _maybe_abstraction_replay(self, loop):
        gs = loop.global_step
        if gs % loop.cfg.abstraction_replay_interval != 0:
            return

        replay_data = loop.abstraction_bank.sample_replay_batch(
            batch_size=16, device=loop.device
        )
        if replay_data is not None:
            z_batch, _seqlen = replay_data
            loop._hebbian_update_on_data(z_init=z_batch)
            loop._log(f"[AbstractionBank] Hebbian replay step {gs}")

        if gs % (loop.cfg.abstraction_replay_interval * 5) == 0:
            for tid in loop.abstraction_bank._store:
                loop.abstraction_bank.consolidate(tid)
                n_p = loop.abstraction_bank.get_num_prototypes(tid)
                loop._log(f"[AbstractionBank] Consolidated {tid}: {n_p} prototypes")

    def _maybe_sniff_forgetting(self, loop):
        forgotten = loop.sniffer.check(loop.global_step, loop.device)
        if not forgotten:
            return

        loop._log(f"[Sniffer] FORGOTTEN: {forgotten} — Hebbian repair")

        wm_factor = 1.0
        strategy = "dopamine"
        if loop.cfg.enable_world_model and loop.world_model is not None:
            wm_surprise = loop._last_world_surprise
            if wm_surprise > loop.cfg.world_model_surprise_threshold:
                wm_factor = 1.0 + min(wm_surprise, 1.0)
                strategy = "world_model"
            else:
                wm_factor = max(0.5, 1.0 - wm_surprise * 2)

        effective_steps = max(1, int(loop.cfg.repair_steps * wm_factor))
        loop._log(
            f"[Sniffer]  wm_surprise={loop._last_world_surprise:.3f} "
            f"factor={wm_factor:.2f} steps={effective_steps} strategy={strategy}"
        )

        for _ in range(effective_steps):
            replay_data = loop.sniffer.get_replay_batch(
                loop.cfg.batch_size, loop.device, strategy=strategy
            )
            if replay_data is None:
                break
            rp_byte, rp_label = replay_data
            loop._hebbian_update_on_data(rp_byte, rp_label)

        loop._log("[Sniffer] Hebbian repair complete")

    def _maybe_sniff_abstraction(self, loop):
        drifted = loop.abstraction_sniffer.check(
            loop.global_step, loop.device, pos_emb=(None, None)
        )
        if not drifted:
            return

        loop._log(f"[AbstractionSniffer] DRIFT detected: {drifted} — Hebbian repair")

        wm_factor = 1.0
        if loop.cfg.enable_world_model and loop.world_model is not None:
            wm_surprise = loop._last_world_surprise
            if wm_surprise > loop.cfg.world_model_surprise_threshold:
                wm_factor = 1.0 + min(wm_surprise, 1.0)
            else:
                wm_factor = max(0.5, 1.0 - wm_surprise * 2)

        effective_steps = max(1, int(loop.abstraction_sniffer.repair_steps * wm_factor))
        loop._log(
            f"[AbstractionSniffer] wm_surprise={loop._last_world_surprise:.3f} "
            f"factor={wm_factor:.2f} steps={effective_steps}"
        )

        for _ in range(effective_steps):
            replay_data = loop.abstraction_bank.sample_replay_batch(
                batch_size=16, device=loop.device
            )
            if replay_data is None:
                break
            z_batch, _seqlen = replay_data
            loop._hebbian_update_on_data(z_init=z_batch)

        loop._log("[AbstractionSniffer] Hebbian repair complete")

    # ── 任务终化 ─────────────────────────────────────────────

    def finalize_task(self, loop, task_id, task_dataset, show_progress: bool = True):
        """采样 exemplars → MemoryBank + AbstractionBank."""
        n_samples = min(200, len(task_dataset))
        idx = torch.randperm(len(task_dataset))[:n_samples].tolist()

        samples = []
        total_bl = 0.0
        with torch.no_grad():
            for i in idx:
                bt, lt = task_dataset[i]
                samples.append((bt, lt))
                x = bt.unsqueeze(0).to(loop.device)
                y = lt.unsqueeze(0).to(loop.device)
                p = loop.model.get_position_embeddings(x.size(-1), loop.device)
                _, bl = loop.model.forward_with_ce(x, y, p)
                total_bl += bl.item()

        avg_bl = total_bl / max(len(idx), 1)
        D_score = loop._last_D if hasattr(loop, "_last_D") else 0.5

        info_gain_val = 0.0
        concept_id_val = ""
        if loop.cfg.enable_intrinsic_motivation and loop._icm_output:
            info_gain_val = loop._icm_output.get("information_gain", 0.0)

        loop.memory_bank.add_samples(
            task_id,
            samples,
            D_score,
            avg_bl,
            transition_surprise=loop._last_world_surprise,
            intrinsic_value=info_gain_val,
            concept_id=concept_id_val,
        )
        loop._log(
            f"[Continual] Task {task_id}: {n_samples} exemplars → bank "
            f"(D={D_score:.3f}, baseline_CE={avg_bl:.4f})"
            f" — bank total: {loop.memory_bank.total}"
        )

        # AbstractionBank
        z_collected = []
        for bt, lt in samples[:100]:
            x = bt.unsqueeze(0).to(loop.device)
            with torch.no_grad():
                z_init = loop.model.init_z(x)
            z_conv, *_ = loop.model.spatiotemporal_infer(
                z_init,
                pos_emb=(None, None),
                gamma=loop.cfg.gamma,
                T=4,
                return_errors=False,
                return_pred_loss=False,
            )
            z_collected.append(z_conv)

        if z_collected:
            layer_imp = compute_layer_importance(
                z_collected[0],
                loop.model,
                (None, None),
                dopamine_D=D_score,
                eta=1.0,
            )
            loop.abstraction_bank.add_z_samples(
                task_id,
                z_collected,
                layer_importance=layer_imp,
                dopamine_score=D_score,
                world_model_surprise=loop._last_world_surprise,
                concept_id=concept_id_val,
                information_gain=info_gain_val,
                group_by_concept=False,
            )
            loop.abstraction_bank.consolidate(task_id)
            n_protos = loop.abstraction_bank.get_num_prototypes(task_id)
            loop._log(
                f"[AbstractionBank] Task {task_id}: "
                f"{len(z_collected)} z_states → {n_protos} prototypes"
            )

    # ── 跨任务评估 ───────────────────────────────────────────

    @torch.no_grad()
    def evaluate_cross_tasks(self, loop, task_ds_list, n_samples=None):
        """评估模型在所有已学任务上的 CE/PPL."""
        results = {}
        loop.model.eval()
        n = n_samples or loop.cfg.eval_samples
        for tid, ds in task_ds_list:
            n_eval = min(n, len(ds))
            total_ce = 0.0
            for i in range(n_eval):
                bt, lt = ds[i]
                x = bt.unsqueeze(0).to(loop.device)
                y = lt.unsqueeze(0).to(loop.device)
                p = loop.model.get_position_embeddings(x.size(-1), loop.device)
                _, ce = loop.model.forward_with_ce(x, y, p)
                total_ce += ce.item()
            avg_ce = total_ce / max(n_eval, 1)
            results[tid] = {"ce": avg_ce, "ppl": math.exp(min(avg_ce, 20))}
        loop.model.train()
        return results
