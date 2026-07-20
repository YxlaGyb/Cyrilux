"""日志 Callback — tqdm 进度条、callback_dict 转发、Hebbian 诊断、100 步日志."""

from model.core.train.callback_base import CallbackBase


class LoggingCallback(CallbackBase):
    """训练日志: 进度条 postfix、progress_callback 发射、诊断输出、WM 指标."""

    def __init__(self):
        """初始化日志统计."""
        self._wm_metrics: dict[str, list[float]] = {
            "transition_error": [],
            "uncertainty": [],
            "fp_rate": [],
        }
        self._wm_fp_count: int = 0
        self._wm_high_surprise_count: int = 0
        self._last_ce_for_fp: float = 0.0

    def on_step_end(self, loop, result, pbar, epoch, task_id):
        """每步后: 进度条 postfix、callback 转发、诊断输出."""
        m = result

        # ── Hebbian 诊断输出 ──
        hebb = m.get("hebb_diag")
        if hebb is not None and loop.global_step % 50 == 0:
            pbar.write(
                f"  [Hebb] oja_α={hebb['oja_alpha']:.4f} | "
                f"mean|ΔW|={hebb['avg_growth']:.6f} | "
                f"updates={hebb['n_params']}"
                + (f" | ⚠ {hebb['n_inf']} inf跳过" if hebb["n_inf"] > 0 else "")
            )

        # ── WM 滚动指标 (7a) ──
        if loop.cfg.enable_world_model:
            wl = m.get("world_loss")
            ws = m.get("world_surprise", 0.0)
            if wl is not None:
                self._wm_metrics["transition_error"].append(wl)
            self._wm_metrics["uncertainty"].append(ws)
            ce_now = m["ce_val"]
            if (
                m.get("update_mode") == "full"
                and ws > loop.cfg.world_model_surprise_threshold
            ):
                self._wm_high_surprise_count += 1
                if ce_now <= self._last_ce_for_fp:
                    self._wm_fp_count += 1
            if loop.global_step % 100 == 0 and self._wm_high_surprise_count > 0:
                fp_rate = self._wm_fp_count / max(self._wm_high_surprise_count, 1)
                self._wm_metrics["fp_rate"].append(fp_rate)
            self._last_ce_for_fp = ce_now

        # ── 进度条 postfix ──
        postfix = {
            "CE": f"{m['ce_val']:.4f}",
            "F": f"{m['F_final']:.1f}",
            "D": f"{m['D']:.3f}",
            "W": f"{m.get('world_surprise', 0.0):.3f}",
        }
        if loop.cfg.enable_intrinsic_motivation and loop._icm_output:
            postfix["IG"] = f"{loop._icm_output.get('information_gain', 0.0):.4f}"
            if loop.concept_discovery:
                postfix["C"] = f"{loop.concept_discovery.n_concepts}"
        pbar.set_postfix(**postfix)

        # ── callback_dict → progress_callback ──
        callback_dict: dict = {
            "type": "progress",
            "step": loop.global_step,
            "total_steps": loop._total_steps,
            "ce_loss": m["ce_val"],
            "F": m["F_final"],
            "temp_loss": m.get("temp_loss_val", 0.0),
            "BPB": m.get("bpb", 0.0),
            "bpb_pred": m.get("bpb_pred", 0.0),
            "D": m["D"],
            "lr": m["lr"],
        }
        if loop.cfg.enable_world_model:
            callback_dict.update(
                {
                    "world_surprise": m.get("world_surprise", 0.0),
                    "world_loss": m.get("world_loss"),
                    "wm_metrics": {
                        k: (v[-1] if v else 0.0) for k, v in self._wm_metrics.items()
                    },
                }
            )
        if loop.cfg.enable_intrinsic_motivation and loop._icm_output:
            callback_dict.update(
                {
                    "information_gain": loop._icm_output.get("information_gain", 0.0),
                    "icm_pred_loss": loop._icm_output.get("pred_loss", 0.0),
                    "n_concepts": loop.concept_discovery.n_concepts
                    if loop.concept_discovery
                    else 0,
                }
            )
        if loop.cfg.progress_callback:
            loop.cfg.progress_callback(callback_dict)

        # ── 100 步日志 ──
        if loop.global_step % 100 == 0:
            log = (
                f"[Step {loop.global_step}/{loop._total_steps}] "
                f"F={m['F_final']:.1f} CE={m['ce_val']:.4f} "
                f"TL={m.get('temp_loss_val', 0.0):.4f} "
                f"BPB={m.get('bpb', 0.0):.2f} "
                f"D={m['D']:.3f} lr={m['lr']:.2e} "
                f"W={m.get('world_surprise', 0.0):.3f}"
            )
            if loop.cfg.enable_intrinsic_motivation and loop._icm_output:
                n_c = (
                    loop.concept_discovery.n_concepts
                    if loop.concept_discovery
                    else 0
                )
                log += (
                    f" IG={loop._icm_output.get('information_gain', 0.0):.4f}"
                    f" ICML={m.get('icm_loss', 0.0):.4f}"
                    f" C={n_c}"
                )
            if m["π"]:
                π_str = ",".join(f"{p:.2f}" for p in m["π"])
                log += f" π=[{π_str}]"
            if loop.cfg.enable_world_model and loop.global_step % 500 == 0:
                te = self._wm_metrics["transition_error"]
                uq = self._wm_metrics["uncertainty"]
                fp = self._wm_metrics["fp_rate"]
                log += (
                    f" | WM: TE={(te[-1] if te else 0):.4f} "
                    f"U={(sum(uq[-100:]) / max(len(uq[-100:]), 1)):.4f} "
                    f"FP={(fp[-1] if fp else 0):.3f}"
                )
            if loop.cfg.progress_callback:
                loop._log(log)
            else:
                pbar.write(log)

    def on_fit_end(self, loop, task_pipelines):
        """全部训练结束时发送 done 信号."""
        if loop.cfg.progress_callback:
            loop.cfg.progress_callback({"type": "done", "message": "Training complete"})
