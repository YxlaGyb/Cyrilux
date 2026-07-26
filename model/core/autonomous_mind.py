"""持续自主运行 — AutonomousMind

无限循环: WAKE → PLAY → SLEEP, 永不停止.

StreamRunner 适配版: 所有 Hebbian 更新由 StreamRunner.step() 内部完成.

架构:
  WAKE  (好奇生成): CuriositySampler 通过 runner.step() 自回归生成字节
  PLAY  (在线交互): 回放数据送入 runner.step(), 内部触发 Hebbian 可塑性
  SLEEP (离线巩固): 批量回放 replay buffer 数据

用法:
    from model.core.autonomous_mind import AutonomousMind
    mind = AutonomousMind(runner)
    mind.run_forever()
"""

import json
import math
import os
import threading
import time
import traceback
from typing import Callable, Optional

import torch

from model.continual.concept_discovery import ConceptDiscovery
from model.model_cyrene import CyreneModel
from pkg.utils.trainer_utils import Logger

# ═══════════════════════════════════════════════════════════════════
# 默认配置
# ═══════════════════════════════════════════════════════════════════

DEFAULT_CFG = {
    "wake_steps": 20,
    "play_steps": 100,
    "sleep_interval": 500,
    "gen_max_new": 64,
    "gen_temperature": 0.8,
    "gen_top_k": 40,
    "gen_prompt_len": 32,
    "batch_size": 16,
    "max_seq_len": 128,
    "gamma": 0.05,
    "T_infer": 1,
    "dopamine_eta": 1.0,
    "dopamine_beta": 0.3,
    "dopamine_gamma": 0.2,
    "dopamine_threshold": 0.05,
    "max_replay_buffer": 2000,
    "replay_batch_size": 16,
    "replay_ratio": 3,
    "save_interval": 10000,
    "out_dir": "out_autonomous",
    "checkpoint": None,
    "data_dir": "datasets",
    "data_rotate_interval": 500,
}

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════
# 好奇心采样器
# ═══════════════════════════════════════════════════════════════════


class CuriositySampler:
    """基于模型自身熵的好奇心驱动采样器.

    使用 StreamRunner.step() + lm_head.predict_logits() 自回归生成.
    """

    def __init__(self, runner: CyreneModel, cfg: dict):
        self.runner = runner
        self.cfg = cfg
        self.device = runner.frontend.byte_proj.weight.device
        self.temperature = cfg.get("gen_temperature", 0.8)
        self.top_k = cfg.get("gen_top_k", 40)
        self.max_new = cfg.get("gen_max_new", 64)

    def sample(self, prompt_bytes: torch.Tensor, n_generations: int = 4) -> list:
        """从 prompt 种子生成多样化的 continuation.

        Returns:
            [(generated_bytes, entropy_score), ...]
        """
        results = []
        prompt_len = prompt_bytes.size(0)
        for _ in range(n_generations):
            gen = self._generate(prompt_bytes)
            if gen is None or gen.size(0) <= prompt_len:
                continue
            new_tokens = gen[prompt_len:]
            entropy = self._compute_entropy(new_tokens)
            results.append((gen.clone(), entropy))
        return results

    def _generate(self, prompt_bytes: torch.Tensor) -> Optional[torch.Tensor]:
        """自回归生成, 使用 runner.step() + lm_head."""
        try:
            seq = prompt_bytes.clone().to(self.device)
            byte_list = seq.tolist()
            max_seq_len = self.cfg.get("max_seq_len", 128)

            for _ in range(self.max_new):
                if len(byte_list) >= max_seq_len:
                    break
                # 编码为 [1, 2, S] fp16
                byte_vals = torch.tensor(
                    [[b / 128.0 - 1.0 for b in byte_list]], dtype=torch.half, device=self.device
                ).unsqueeze(0)
                mask = torch.ones_like(byte_vals)
                inp = torch.cat([byte_vals, mask], dim=1)

                self.runner.step(inp)
                logits = self.runner.lm_head.predict_logits(
                    self.runner.pool, self.runner._top_layer
                )
                logits = [v / self.temperature for v in logits]

                if self.top_k > 0:
                    sorted_l = sorted(logits, reverse=True)
                    threshold = sorted_l[min(self.top_k, len(sorted_l)) - 1]
                    logits = [v if v >= threshold else float("-inf") for v in logits]

                max_l = max(logits)
                exp_l = [math.exp(v - max_l) for v in logits]
                s = sum(exp_l)
                r = torch.rand(1, device=self.device).item()
                cum = 0.0
                next_byte = 0
                for i, p in enumerate(exp_l):
                    cum += p / s
                    if r < cum:
                        next_byte = i
                        break
                byte_list.append(next_byte)

            return torch.tensor(byte_list, device=self.device)
        except Exception as e:
            Logger(f"[CuriositySampler] Generation error: {e}")
            return None

    @staticmethod
    def _compute_entropy(tokens: torch.Tensor) -> float:
        if tokens.numel() == 0:
            return 0.0
        return tokens.unique().numel() / max(tokens.numel(), 1)

    def batch_generate(self, prompts: list, n_per_prompt: int = 2) -> list:
        all_results = []
        for i, p in enumerate(prompts):
            for gen_bytes, ent in self.sample(p, n_per_prompt):
                all_results.append((gen_bytes, ent, i))
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results


# ═══════════════════════════════════════════════════════════════════
# 经验回放缓冲区
# ═══════════════════════════════════════════════════════════════════


class ExperienceReplayBuffer:
    """FIFO 经验回放缓冲区."""

    def __init__(self, max_size: int = 2000):
        self.max_size = max_size
        self.buffer = []
        self._dopamine_sum = 0.0

    def add(self, byte_seq: torch.Tensor, labels: torch.Tensor, D: float = 0.5):
        self.buffer.append((byte_seq.cpu(), labels.cpu(), D))
        self._dopamine_sum += D
        if len(self.buffer) > self.max_size:
            removed = self.buffer.pop(0)
            self._dopamine_sum -= removed[2]

    def add_batch(self, byte_seq: torch.Tensor, labels: torch.Tensor, D: float = 0.5):
        for i in range(byte_seq.size(0)):
            self.add(byte_seq[i], labels[i], D)

    def sample(self, batch_size: int, device: str = "cpu") -> Optional[tuple]:
        if len(self.buffer) < batch_size:
            return None
        if self._dopamine_sum > 0 and len(self.buffer) > batch_size * 2:
            weights = torch.tensor([ex[2] + 0.01 for ex in self.buffer], dtype=torch.float16)
            idx = torch.multinomial(weights, batch_size, replacement=False)
        else:
            idx = torch.randperm(len(self.buffer))[:batch_size]
        batch_bytes = torch.stack([self.buffer[i][0] for i in idx]).to(device)
        batch_labels = torch.stack([self.buffer[i][1] for i in idx]).to(device)
        return batch_bytes, batch_labels

    @property
    def size(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════
# 元控制器, 阶段调度
# ═══════════════════════════════════════════════════════════════════


class MetaController:
    """调度 WAKE / PLAY / SLEEP 阶段, 检测高原."""

    PHASES = ["IDLE", "WAKE", "PLAY", "SLEEP"]

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.phase = "IDLE"
        self.total_steps = 0
        self.wake_count = 0
        self.play_count = 0
        self.sleep_count = 0
        self._loss_history = []
        self._plateau_streak = 0
        self._last_plateau_step = 0
        self.current_dopamine_threshold = cfg.get("dopamine_threshold", 0.05)
        self.current_gamma = cfg.get("gamma", 0.05)

    def next_phase(self) -> str:
        self.total_steps += 1
        if self.phase in ("IDLE", "SLEEP"):
            self.phase = "WAKE"
            self.wake_count += 1
        elif self.phase == "WAKE":
            if self.wake_count >= self.cfg.get("wake_steps", 20):
                self.phase = "PLAY"
                self.wake_count = 0
        elif self.phase == "PLAY":
            self.play_count += 1
            if self.total_steps % self.cfg.get("sleep_interval", 500) == 0:
                self.phase = "SLEEP"
                self.play_count = 0
            elif self.play_count >= self.cfg.get("play_steps", 100):
                self.phase = "WAKE"
                self.play_count = 0
        return self.phase

    def report_loss(self, loss: float):
        self._loss_history.append(loss)
        if len(self._loss_history) > 100:
            self._loss_history.pop(0)
        if len(self._loss_history) >= 20:
            recent = sum(self._loss_history[-10:]) / 10
            older = sum(self._loss_history[-20:-10]) / 10
            if older > 0 and abs(recent - older) / older < 0.01:
                self._plateau_streak += 1
            else:
                self._plateau_streak = 0
        if self._plateau_streak >= 3 and self.total_steps - self._last_plateau_step > 200:
            self._break_plateau()
            self._plateau_streak = 0
            self._last_plateau_step = self.total_steps

    def _break_plateau(self):
        old_g = self.current_gamma
        old_d = self.current_dopamine_threshold
        self.current_gamma = min(self.current_gamma * 1.5, 0.5)
        self.current_dopamine_threshold = max(self.current_dopamine_threshold * 0.8, 0.001)
        Logger(
            f"[MetaController] Plateau: gamma {old_g:.3f}->{self.current_gamma:.3f}, "
            f"D_thr {old_d:.3f}->{self.current_dopamine_threshold:.3f}"
        )

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "total_steps": self.total_steps,
            "wake_count": self.wake_count,
            "play_count": self.play_count,
            "sleep_count": self.sleep_count,
            "plateau_streak": self._plateau_streak,
            "dopamine_threshold": self.current_dopamine_threshold,
            "gamma": self.current_gamma,
        }


# ═══════════════════════════════════════════════════════════════════
# 外部数据轮换
# ═══════════════════════════════════════════════════════════════════


class DataRotator:
    """在 dataset/ 目录下轮换读取 jsonl 文件."""

    def __init__(self, data_dir: str, max_seq_len: int = 128, max_samples: int = 200):
        self.data_dir = os.path.join(ROOT_DIR, data_dir)
        self.max_seq_len = max_seq_len
        self.max_samples = max_samples
        self.files = []
        self._current_file_idx = -1
        self._current_data = []
        self._pos = 0
        self._scan_files()
        self._load_next()

    def _scan_files(self):
        if not os.path.isdir(self.data_dir):
            self.files = []
            return
        self.files = sorted(
            f for f in os.listdir(self.data_dir) if f.endswith(".jsonl") and not f.startswith("_")
        )
        conv = [f for f in self.files if "_converted" in f]
        if conv:
            self.files = conv + [f for f in self.files if f not in conv]

    def _load_next(self):
        if not self.files:
            self._current_data = []
            return
        self._current_file_idx = (self._current_file_idx + 1) % len(self.files)
        fpath = os.path.join(self.data_dir, self.files[self._current_file_idx])
        self._current_data = []
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= self.max_samples:
                        break
                    sample = json.loads(line)
                    text = sample.get("text", "")
                    if not text and "conversations" in sample:
                        parts = [
                            f"<|{t.get('role', 'user')}|>{t.get('content', '')}"
                            for t in sample["conversations"]
                        ]
                        text = "\n".join(parts) + "<|end|>"
                    if not text:
                        continue
                    byte_seq = text.encode("utf-8")[: self.max_seq_len]
                    padded = byte_seq.ljust(self.max_seq_len, b"\x00")
                    t = torch.frombuffer(bytearray(padded), dtype=torch.uint8).clone()
                    lbl = t.clone()
                    lbl[t == 0x00] = -100
                    self._current_data.append((t, lbl.to(torch.long)))
        except Exception as e:
            Logger(f"[DataRotator] Load error: {e}")
        Logger(
            f"[DataRotator] Loaded {len(self._current_data)} "
            f"from {self.files[self._current_file_idx]}"
        )
        self._pos = 0

    def get_batch(self, batch_size: int) -> Optional[tuple]:
        if not self._current_data and self.files:
            self._load_next()
        if not self._current_data:
            return None
        batch = []
        for _ in range(batch_size):
            if self._pos >= len(self._current_data):
                self._load_next()
                if not self._current_data:
                    break
            batch.append(self._current_data[self._pos])
            self._pos = (self._pos + 1) % len(self._current_data)
        if not batch:
            return None
        return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])


# ═══════════════════════════════════════════════════════════════════
# 自主意识核心
# ═══════════════════════════════════════════════════════════════════


class AutonomousMind:
    """持续自主运行 — WAKE → PLAY → SLEEP 循环 (StreamRunner)."""

    def __init__(
        self,
        runner: CyreneModel,
        cfg: dict = None,
        log_callback: Callable = None,
    ):
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.log_callback = log_callback or (lambda msg: Logger(msg))
        self._stop_flag = threading.Event()
        self.runner = runner
        self.device = runner.frontend.byte_proj.weight.device
        self.sampler = CuriositySampler(self.runner, self.cfg)
        self.replay_buffer = ExperienceReplayBuffer(self.cfg["max_replay_buffer"])
        self.controller = MetaController(self.cfg)
        self.data_rotator = DataRotator(
            self.cfg["data_dir"],
            max_seq_len=self.cfg["max_seq_len"],
            max_samples=200,
        )
        self.total_steps = 0
        self.current_phase = "IDLE"
        self.last_save_step = 0
        self._log(
            f"AutonomousMind init, device={self.device}, "
            f"neurons={self.runner.pool.get_total_neurons()}"
        )

    def _log(self, msg: str):
        self.log_callback(msg)

    # ══════════════════════════════════════════════════════════════
    # 主循环
    # ══════════════════════════════════════════════════════════════

    def run_forever(self):
        self._log("Entering perpetual run mode")
        self._stop_flag.clear()
        while not self._stop_flag.is_set():
            try:
                self._step()
            except Exception as e:
                self._log(f"Step error: {e}")
                self._log(traceback.format_exc()[-500:])
                time.sleep(1.0)

    def stop(self):
        self._stop_flag.set()

    def _step(self):
        phase = self.controller.next_phase()
        self.current_phase = phase
        if phase == "WAKE":
            self._wake_step()
        elif phase == "PLAY":
            self._play_step()
        elif phase == "SLEEP":
            self._sleep_step()
        self.total_steps = self.controller.total_steps
        if self.total_steps - self.last_save_step >= self.cfg["save_interval"]:
            self._save_checkpoint(async_save=True)
            self.last_save_step = self.total_steps

    def _wake_step(self):
        prompts = self._get_prompts(n_prompts=4)
        if not prompts:
            time.sleep(0.1)
            return
        for gen_bytes, entropy, _ in self.sampler.batch_generate(prompts, n_per_prompt=2):
            if gen_bytes is None or gen_bytes.numel() < 4:
                continue
            labels = gen_bytes.clone()
            labels[1:] = gen_bytes[:-1]
            labels[0] = -100
            labels = labels.to(torch.long)
            self.replay_buffer.add(gen_bytes, labels, D=0.3 + min(entropy, 1.0) * 0.7)
        self._log(f"[WAKE] buffer={self.replay_buffer.size}")

    def _get_prompts(self, n_prompts: int = 4) -> list:
        prompts = []
        plen = self.cfg["gen_prompt_len"]
        result = self.replay_buffer.sample(n_prompts, "cpu")
        if result is not None:
            for i in range(min(n_prompts, result[0].size(0))):
                seq = result[0][i]
                nz = seq[seq != 0]
                if nz.numel() >= plen:
                    prompts.append(nz[:plen].clone())
        if len(prompts) < n_prompts:
            ext = self.data_rotator.get_batch(n_prompts - len(prompts))
            if ext is not None:
                for i in range(ext[0].size(0)):
                    seq = ext[0][i]
                    nz = seq[seq != 0]
                    if nz.numel() >= plen:
                        prompts.append(nz[:plen].clone())
        return prompts

    def _play_step(self):
        """PLAY: runner.step() — Hebbian handled by StreamRunner internally."""
        batch = self.replay_buffer.sample(self.cfg["batch_size"], self.device)
        if batch is None:
            batch = self.data_rotator.get_batch(self.cfg["batch_size"])
        if batch is None:
            time.sleep(0.2)
            return
        byte_seq, _ = batch
        byte_vals = (byte_seq.half() / 128.0 - 1.0).unsqueeze(1)
        mask = torch.ones_like(byte_vals)
        inp = torch.cat([byte_vals, mask], dim=1)
        for i in range(byte_seq.size(0)):
            self.runner.step(inp[i : i + 1])
        if self.total_steps % 10 == 0:
            s = self.runner.get_state()
            self._log(
                f"[PLAY] Step {self.total_steps} | F={s['free_energy']:.1f} "
                f"D={s['D']:.3f} n={self.runner.pool.get_total_neurons()} "
                f"buf={self.replay_buffer.size}"
            )

    def _sleep_step(self):
        """SLEEP: replay buffer consolidation via StreamRunner."""
        self._log(f"[SLEEP] consolidate (buf={self.replay_buffer.size})")
        for _ in range(min(50, self.replay_buffer.size // 4)):
            batch = self.replay_buffer.sample(self.cfg["replay_batch_size"], self.device)
            if batch is None:
                break
            byte_seq, _ = batch
            byte_vals = (byte_seq.half() / 128.0 - 1.0).unsqueeze(1)
            mask = torch.ones_like(byte_vals)
            inp = torch.cat([byte_vals, mask], dim=1)
            for i in range(byte_seq.size(0)):
                self.runner.step(inp[i : i + 1])
        self._log("[SLEEP] done, rotating data")
        self.data_rotator = DataRotator(
            self.cfg["data_dir"], max_seq_len=self.cfg["max_seq_len"], max_samples=200
        )

    def _save_checkpoint(self, async_save: bool = False):
        def _save():
            try:
                out_dir = os.path.join(ROOT_DIR, self.cfg["out_dir"])
                os.makedirs(out_dir, exist_ok=True)
                ckpt = {
                    "runner_state": self.runner.get_state(),
                    "controller_state": self.controller.to_dict(),
                    "total_steps": self.total_steps,
                    "cfg": self.cfg,
                }
                path = os.path.join(out_dir, f"autonomous_s{self.total_steps}.pt")
                torch.save(ckpt, path)
                torch.save(ckpt, os.path.join(out_dir, "autonomous_latest.pt"))
                self._log(f"Saved: {path}")
            except Exception as e:
                self._log(f"Save error: {e}")

        if async_save:
            threading.Thread(target=_save, daemon=True).start()
        else:
            _save()


# ═══════════════════════════════════════════════════════════════════
# 内在动机元控制器
# ═══════════════════════════════════════════════════════════════════


class IntrinsicMetaController(MetaController):
    """支持内在动机信号的元控制器."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.curiosity_drive: float = 0.5
        self.competence_drive: float = 0.5
        self.boredom_signal: float = 0.0
        self._icm_history: list[dict] = []
        self.concept_discovery: Optional[ConceptDiscovery] = None

    def update_intrinsic(self, icm_output: dict):
        self._icm_history.append(icm_output)
        if len(self._icm_history) > 100:
            self._icm_history.pop(0)
        ig_ema = sum(d.get("information_gain", 0) for d in self._icm_history[-20:])
        ig_ema /= max(len(self._icm_history[-20:]), 1)
        self.curiosity_drive = min(1.0, ig_ema * 5.0)
        if len(self._icm_history) >= 20:
            recent = sum(d.get("inverse_loss", 0) for d in self._icm_history[-10:]) / 10
            older = sum(d.get("inverse_loss", 0) for d in self._icm_history[-20:-10]) / 10
            if older > 0:
                self.competence_drive = min(1.0, max(0.0, (older - recent) / older * 2))
        self.boredom_signal = min(
            1.0, max(0.0, (1 - self.curiosity_drive) * (1 - self.competence_drive) * 2)
        )

    def next_phase(self) -> str:
        phase = super().next_phase()
        if phase == "PLAY" and self.boredom_signal > 0.6:
            self.phase = "WAKE"
            self.play_count = 0
            self.wake_count = 0
            return "WAKE"
        if (
            phase == "WAKE"
            and self.curiosity_drive > 0.7
            and self.wake_count < self.cfg.get("wake_steps", 20) // 2
        ):
            self.phase = "PLAY"
            self.wake_count = 0
            return "PLAY"
        return phase

    def get_fragile_replay_targets(self) -> list[str]:
        return self.concept_discovery.get_fragile_concept_ids() if self.concept_discovery else []
