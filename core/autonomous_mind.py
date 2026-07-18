"""
持续自主运行 — AutonomousMind

无限循环: WAKE → PLAY → SLEEP, 永不停止.

架构:
  WAKE  (好奇生成): 模型自主生成多样化的字节序列 (Prompt → Continuation)
  PLAY  (在线交互): 模型处理生成文本, 在线学习, 多巴胺调制
  SLEEP (离线巩固): 记忆回放, 长程巩固

MetaController:
  - 阶段调度: 动态分配 WAKE/PLAY/SLEEP 占比
  - 高原检测: loss 停滞 → 提升 gamma / 降低 dopamine_threshold
  - 保底: 所有异常 try/except, 永不崩溃退出
  - 后台检查点: 每 N 步自动保存

用法:
    from core.autonomous_mind import AutonomousMind
    mind = AutonomousMind(model, lm_config)
    mind.run_forever()
"""
import os, json, math, time, random, threading, traceback
from pathlib import Path
from typing import Optional, Callable

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from model.pc_layers import PCLocalDynamicMiniMind
from model.model_minimind import MiniMindConfig
from model.pc_core import DopamineSignal
from core.trainer_utils import get_lr, Logger, setup_seed
from core.globals import DEVICE_STR

# 内在动机模块
from continual.intrinsic_curiosity import IntrinsicCuriosityModule
from continual.concept_discovery import ConceptDiscovery
from continual.memory_gating import MemoryGate


# ═══════════════════════════════════════════════════════════════════
# 默认配置
# ═══════════════════════════════════════════════════════════════════

DEFAULT_CFG = {
    # 阶段调度
    'wake_steps': 20,          # 每次 WAKE 生成步数
    'play_steps': 100,         # 每次 PLAY 训练步数
    'sleep_interval': 500,     # 每 N 总步执行一次 SLEEP

    # 生成
    'gen_max_new': 64,         # 每次生成最大新 token 数
    'gen_temperature': 0.8,    # 生成温度
    'gen_top_k': 40,           # Top-K 采样
    'gen_prompt_len': 32,      # Prompt 种子长度 (从 memory 采样)

    # 训练
    'batch_size': 16,
    'max_seq_len': 128,
    'lr': 1e-4,
    'gamma': 0.05,
    'T_infer': 1,
    'grad_clip': 1.0,

    # 多巴胺
    'dopamine_eta': 1.0,
    'dopamine_beta': 0.3,
    'dopamine_gamma': 0.2,
    'dopamine_threshold': 0.05,  # D 低于此值 → 提升好奇度

    # 记忆
    'max_replay_buffer': 2000,
    'replay_batch_size': 16,
    'replay_ratio': 3,         # 每 N 步插入 1 步回放

    # 检查点
    'save_interval': 200,
    'out_dir': 'out_autonomous',
    'checkpoint': None,        # 初始检查点路径

    # 外部数据 (持续学习)
    'data_dir': 'datasets',
    'data_rotate_interval': 500,  # 每 N 步换一个数据集文件
}


# 工作目录 (项目根目录)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════
# 好奇心采样器
# ═══════════════════════════════════════════════════════════════════

class CuriositySampler:
    """
    基于模型自身熵的好奇心驱动采样器.

    策略:
      - 从 replay buffer 采样 prompt 种子
      - 用模型生成 continuation
      - 计算生成序列的 entropy (预测不确定性)
      - 高 entropy → 高好奇心 → 优先用于 PLAY 训练
    """
    def __init__(self, model: PCLocalDynamicMiniMind, cfg: dict):
        self.model = model
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self.temperature = cfg.get('gen_temperature', 0.8)
        self.top_k = cfg.get('gen_top_k', 40)
        self.max_new = cfg.get('gen_max_new', 64)

    def sample(self, prompt_bytes: torch.Tensor, n_generations: int = 4) -> list:
        """
        从 prompt 种子生成多样化的 continuation.

        参数:
          prompt_bytes: [seq_len] uint8 tensor
          n_generations: 每个 prompt 生成的变体数

        返回:
          [(generated_bytes, entropy_score), ...]
        """
        results = []
        prompt_len = prompt_bytes.size(0)
        device = self.device

        with torch.no_grad():
            for _ in range(n_generations):
                gen = self._generate(prompt_bytes, device)
                if gen is None or gen.size(0) <= prompt_len:
                    continue

                new_tokens = gen[prompt_len:]
                entropy = self._compute_entropy(new_tokens)
                results.append((gen.clone(), entropy))

        return results

    def _generate(self, prompt_bytes: torch.Tensor, device) -> Optional[torch.Tensor]:
        """自回归生成。"""
        try:
            self.model.eval()
            seq = prompt_bytes.clone().to(device)  # [seq_len]
            max_new = self.max_new
            max_seq_len = self.cfg.get('max_seq_len', 128)

            for _ in range(max_new):
                if seq.size(0) >= max_seq_len:
                    break
                ctx = seq[-max_seq_len:].unsqueeze(0)  # [1, ctx_len]
                pos = self.model.get_position_embeddings(ctx.size(1), device)

                logits, _ = self.model.forward_with_ce(ctx, None, pos)
                logits = logits[:, -1, :]  # [1, vocab]
                logits = logits / self.temperature

                if self.top_k > 0:
                    top_k_vals, _ = torch.topk(logits, self.top_k, dim=-1)
                    threshold = top_k_vals[:, -1].unsqueeze(-1)
                    logits[logits < threshold] = -float('inf')

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
                seq = torch.cat([seq, next_token.squeeze(0)], dim=0)

            self.model.train()
            return seq
        except Exception as e:
            Logger(f'[CuriositySampler] Generation error: {e}')
            self.model.train()
            return None

    def _compute_entropy(self, tokens: torch.Tensor) -> float:
        """计算 token 序列的熵 (多样性指标)。"""
        if tokens.numel() == 0:
            return 0.0
        unique = tokens.unique().numel()
        return unique / max(tokens.numel(), 1)

    def batch_generate(self, prompts: list, n_per_prompt: int = 2) -> list:
        """
        批量生成。

        返回:
          [(generated_bytes, entropy, prompt_idx), ...]
        """
        all_results = []
        for i, p in enumerate(prompts):
            results = self.sample(p, n_per_prompt)
            for gen_bytes, ent in results:
                all_results.append((gen_bytes, ent, i))
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results


# ═══════════════════════════════════════════════════════════════════
# 经验回放缓冲区
# ═══════════════════════════════════════════════════════════════════

class ExperienceReplayBuffer:
    """
    FIFO 经验回放缓冲区.

    每条经验 = (byte_seq, byte_label, dopamine_score)
    采样策略: dopamine_weighted → 高 D 的样本更可能被回放.
    """
    def __init__(self, max_size: int = 2000):
        self.max_size = max_size
        self.buffer = []       # [(bytes, labels, D)]
        self._dopamine_sum = 0.0

    def add(self, byte_seq: torch.Tensor, labels: torch.Tensor, D: float = 0.5):
        """添加一条经验。"""
        self.buffer.append((byte_seq.cpu(), labels.cpu(), D))
        self._dopamine_sum += D
        if len(self.buffer) > self.max_size:
            removed = self.buffer.pop(0)
            self._dopamine_sum -= removed[2]

    def add_batch(self, byte_seq: torch.Tensor, labels: torch.Tensor, D: float = 0.5):
        """批量添加。"""
        for i in range(byte_seq.size(0)):
            self.add(byte_seq[i], labels[i], D)

    def sample(self, batch_size: int, device: str = 'cpu') -> Optional[tuple]:
        """多巴胺加权采样。"""
        if len(self.buffer) < batch_size:
            return None

        if self._dopamine_sum > 0 and len(self.buffer) > batch_size * 2:
            weights = torch.tensor([ex[2] + 0.01 for ex in self.buffer], dtype=torch.float)
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
# 元控制器 — 阶段调度
# ═══════════════════════════════════════════════════════════════════

class MetaController:
    """
    调度 WAKE / PLAY / SLEEP 阶段, 检测高原, 动态调整参数.

    阶段状态机:
      IDLE → WAKE (生成) → PLAY (训练) → (每 sleep_interval) → SLEEP (巩固) → IDLE
    """
    PHASES = ['IDLE', 'WAKE', 'PLAY', 'SLEEP']

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.phase = 'IDLE'
        self.total_steps = 0
        self.wake_count = 0
        self.play_count = 0
        self.sleep_count = 0

        # 高原检测
        self._loss_history = []
        self._plateau_streak = 0
        self._last_plateau_step = 0

        # 多巴胺自适应
        self.current_dopamine_threshold = cfg.get('dopamine_threshold', 0.05)
        self.current_gamma = cfg.get('gamma', 0.05)

    def next_phase(self) -> str:
        """决定下一阶段。"""
        self.total_steps += 1

        if self.phase in ('IDLE', 'SLEEP'):
            self.phase = 'WAKE'
            self.wake_count += 1
        elif self.phase == 'WAKE':
            if self.wake_count >= self.cfg.get('wake_steps', 20):
                self.phase = 'PLAY'
                self.wake_count = 0
        elif self.phase == 'PLAY':
            self.play_count += 1
            if self.total_steps % self.cfg.get('sleep_interval', 500) == 0:
                self.phase = 'SLEEP'
                self.play_count = 0
            elif self.play_count >= self.cfg.get('play_steps', 100):
                self.phase = 'WAKE'
                self.play_count = 0

        return self.phase

    def report_loss(self, loss: float):
        """报告当前 loss, 用于高原检测。"""
        self._loss_history.append(loss)
        if len(self._loss_history) > 100:
            self._loss_history.pop(0)

        if len(self._loss_history) >= 20:
            recent = self._loss_history[-10:]
            older = self._loss_history[-20:-10]
            if len(recent) >= 10 and len(older) >= 10:
                recent_avg = sum(recent) / len(recent)
                older_avg = sum(older) / len(older)
                if older_avg > 0 and abs(recent_avg - older_avg) / older_avg < 0.01:
                    self._plateau_streak += 1
                else:
                    self._plateau_streak = 0

        if self._plateau_streak >= 3 and self.total_steps - self._last_plateau_step > 200:
            self._break_plateau()
            self._plateau_streak = 0
            self._last_plateau_step = self.total_steps

    def _break_plateau(self):
        """突破高原: 提升 gamma, 降低 dopamine_threshold。"""
        old_gamma = self.current_gamma
        old_thresh = self.current_dopamine_threshold
        self.current_gamma = min(self.current_gamma * 1.5, 0.5)
        self.current_dopamine_threshold = max(self.current_dopamine_threshold * 0.8, 0.001)
        Logger(f'[MetaController] ⛰ Plateau broken: '
               f'γ {old_gamma:.3f}→{self.current_gamma:.3f}, '
               f'D_thr {old_thresh:.3f}→{self.current_dopamine_threshold:.3f}')

    def to_dict(self) -> dict:
        return {
            'phase': self.phase,
            'total_steps': self.total_steps,
            'wake_count': self.wake_count,
            'play_count': self.play_count,
            'sleep_count': self.sleep_count,
            'plateau_streak': self._plateau_streak,
            'dopamine_threshold': self.current_dopamine_threshold,
            'gamma': self.current_gamma,
        }


# ═══════════════════════════════════════════════════════════════════
# 外部数据轮换
# ═══════════════════════════════════════════════════════════════════

class DataRotator:
    """
    在 dataset/ 目录下轮换读取 jsonl 文件, 持续提供新鲜数据.
    支持格式检测 (conversations → text 自动转换).
    """
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
        """扫描 jsonl 文件。"""
        if not os.path.isdir(self.data_dir):
            self.files = []
            return
        self.files = sorted([
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.endswith('.jsonl') and not f.startswith('_')
        ])
        converted = [f for f in self.files if '_converted' in f]
        if converted:
            self.files = converted + [f for f in self.files if f not in converted]

    def _load_next(self):
        """加载下一个文件。"""
        if not self.files:
            self._current_data = []
            return

        self._current_file_idx = (self._current_file_idx + 1) % len(self.files)
        fpath = self.files[self._current_file_idx]
        fname = os.path.basename(fpath)

        self._current_data = []
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= self.max_samples:
                        break
                    sample = json.loads(line)

                    if 'text' in sample:
                        text = sample['text']
                    elif 'conversations' in sample:
                        parts = []
                        for turn in sample['conversations']:
                            role = turn.get('role', 'user')
                            content = turn.get('content', '')
                            parts.append(f'<|{role}|>{content}')
                        text = '\n'.join(parts) + '<|end|>'
                    else:
                        continue

                    byte_seq = text.encode('utf-8')[:self.max_seq_len]
                    padded = byte_seq.ljust(self.max_seq_len, b'\x00')
                    t = torch.frombuffer(bytearray(padded), dtype=torch.uint8).clone()
                    lbl = t.clone()
                    lbl[t == 0x00] = -100
                    self._current_data.append((t, lbl.to(torch.long)))
        except Exception as e:
            Logger(f'[DataRotator] Load {fname} error: {e}')

        Logger(f'[DataRotator] Loaded {len(self._current_data)} samples from {fname}')
        self._pos = 0

    def get_batch(self, batch_size: int) -> Optional[tuple]:
        """获取一批数据。"""
        if not self._current_data and self.files:
            self._load_next()

        if len(self._current_data) == 0:
            return None

        batch = []
        for _ in range(batch_size):
            if self._pos >= len(self._current_data):
                self._load_next()
                if len(self._current_data) == 0:
                    break
            batch.append(self._current_data[self._pos])
            self._pos = (self._pos + 1) % len(self._current_data)

        if not batch:
            return None

        bytes_t = torch.stack([b[0] for b in batch])
        labels_t = torch.stack([b[1] for b in batch])
        return bytes_t, labels_t


# ═══════════════════════════════════════════════════════════════════
# 自主意识核心
# ═══════════════════════════════════════════════════════════════════

class AutonomousMind:
    """
    持续自主运行核心 — 永不停止的 WAKE → PLAY → SLEEP 循环.

    用法:
        model = PCLocalDynamicMiniMind(lm_config)
        mind = AutonomousMind(model, lm_config)
        mind.run_forever()   # 无限循环, 永不返回
    """
    def __init__(
        self,
        model: PCLocalDynamicMiniMind = None,
        lm_config: MiniMindConfig = None,
        cfg: dict = None,
        log_callback: Callable = None,
    ):
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.log_callback = log_callback or (lambda msg: Logger(msg))
        self.device = DEVICE_STR
        self._stop_flag = threading.Event()

        # ── 创建 / 加载模型 ──
        self.lm_config = lm_config or MiniMindConfig(
            hidden_size=256, num_hidden_layers=4, use_moe=False,
        )
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = PCLocalDynamicMiniMind(self.lm_config).to(self.device)
            self._try_load_checkpoint()

        self.model.train()

        # ── 优化器 ──
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg['lr'], betas=(0.9, 0.95), fused=True,
        )

        # ── 子模块 ──
        self.dopamine = DopamineSignal(
            η=self.cfg['dopamine_eta'],
            threshold=self.cfg['dopamine_threshold'],
        )
        self.sampler = CuriositySampler(self.model, self.cfg)
        self.replay_buffer = ExperienceReplayBuffer(self.cfg['max_replay_buffer'])
        self.controller = MetaController(self.cfg)
        self.data_rotator = DataRotator(
            self.cfg['data_dir'],
            max_seq_len=self.cfg['max_seq_len'],
            max_samples=200,
        )

        # ── 状态 ──
        self.total_steps = 0
        self.current_phase = 'IDLE'
        self.last_save_step = 0
        self._gen_prompts_cache = []

        self._log(f'🧠 AutonomousMind 初始化完成')
        self._log(f'   Device: {self.device}')
        self._log(f'   Params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6:.2f}M')
        self._log(f'   Config: {json.dumps(self.cfg, indent=2)}')

    def _log(self, msg: str):
        self.log_callback(msg)

    def _try_load_checkpoint(self):
        """尝试从检查点加载。自动处理 byte_proj 输入通道变化 (1→2)。"""
        def _fix_shapes(sd, model_sd):
            """修复 byte_proj 权重形状不匹配。"""
            for key in list(sd.keys()):
                if key.endswith('byte_proj.weight') and key in model_sd:
                    expected = model_sd[key].shape
                    actual = sd[key].shape
                    if actual != expected and actual[1] == 1 and expected[1] == 2:
                        sd[key] = sd[key].expand(-1, 2, -1) / 2
            return sd

        def _safe_exists(path) -> bool:
            try:
                return os.path.exists(path)
            except OSError:
                return False

        try:
            ckpt_path = self.cfg.get('checkpoint')
            if ckpt_path and _safe_exists(ckpt_path):
                try:
                    ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                    _fix_shapes(ckpt['model_state'], self.model.state_dict())
                    self.model.load_state_dict(ckpt['model_state'], strict=False)
                    self._log(f'Loaded checkpoint: {ckpt_path}')
                    if 'optimizer_state' in ckpt and hasattr(self, 'optimizer'):
                        try:
                            self.optimizer.load_state_dict(ckpt['optimizer_state'])
                        except Exception:
                            pass
                    return True
                except Exception as e:
                    self._log(f'Checkpoint load failed: {e}')

            default_path = os.path.join(ROOT_DIR, 'out_pc_unified', 'unified_final.pt')
            if _safe_exists(default_path):
                try:
                    ckpt = torch.load(default_path, map_location=self.device, weights_only=False)
                    _fix_shapes(ckpt['model_state'], self.model.state_dict())
                    self.model.load_state_dict(ckpt['model_state'], strict=False)
                    self._log(f'Loaded default checkpoint: {default_path}')
                    return True
                except Exception as e:
                    self._log(f'Default checkpoint load failed: {e}')
        except Exception as e:
            self._log(f'Checkpoint loading error: {type(e).__name__}: {e}')
        return False

    # ══════════════════════════════════════════════════════════════
    # 主循环
    # ══════════════════════════════════════════════════════════════

    def run_forever(self):
        """
        无限循环: 永不返回.
        所有异常都被捕获, 确保持续运行.
        """
        self._log('🚀 进入持续运行模式 — 永不停止')
        self._stop_flag.clear()

        while not self._stop_flag.is_set():
            try:
                self._step()
            except Exception as e:
                self._log(f'❌ 步骤异常: {e}')
                self._log(traceback.format_exc()[-500:])
                time.sleep(1.0)
                continue

        self._log('⏹ 持续运行已停止')

    def stop(self):
        """设置停止标志 (线程安全)。"""
        self._stop_flag.set()
        self._log('⏹ 正在停止...')

    def _step(self):
        """单步执行当前阶段的动作。"""
        phase = self.controller.next_phase()
        self.current_phase = phase

        if phase == 'WAKE':
            self._wake_step()
        elif phase == 'PLAY':
            self._play_step()
        elif phase == 'SLEEP':
            self._sleep_step()

        self.total_steps = self.controller.total_steps
        if self.total_steps - self.last_save_step >= self.cfg['save_interval']:
            self._save_checkpoint(async_save=True)
            self.last_save_step = self.total_steps

    # ══════════════════════════════════════════════════════════════
    # WAKE 阶段: 好奇生成
    # ══════════════════════════════════════════════════════════════

    def _wake_step(self):
        """一次 WAKE 动作。"""
        prompts = self._get_prompts(n_prompts=4)
        if not prompts:
            time.sleep(0.1)
            return

        gen_results = self.sampler.batch_generate(prompts, n_per_prompt=2)

        added = 0
        for gen_bytes, entropy, _ in gen_results:
            if gen_bytes is None or gen_bytes.numel() < 4:
                continue
            labels = gen_bytes.clone()
            labels[1:] = gen_bytes[:-1]
            labels[0] = -100
            labels = labels.to(torch.long)

            D_bonus = min(entropy, 1.0)
            self.replay_buffer.add(gen_bytes, labels, D=0.3 + D_bonus * 0.7)
            added += 1

        if added:
            self._log(f'[WAKE] Generated {added} sequences, buffer={self.replay_buffer.size}')

    def _get_prompts(self, n_prompts: int = 4) -> list:
        """从 replay buffer 或外部数据获取 prompt 种子。"""
        prompts = []
        prompt_len = self.cfg['gen_prompt_len']

        result = self.replay_buffer.sample(n_prompts, 'cpu')
        if result is not None:
            batch_bytes, _ = result
            for i in range(min(n_prompts, batch_bytes.size(0))):
                seq = batch_bytes[i]
                non_zero = seq[seq != 0]
                if non_zero.numel() >= prompt_len:
                    prompts.append(non_zero[:prompt_len].clone())

        if len(prompts) < n_prompts:
            ext = self.data_rotator.get_batch(n_prompts - len(prompts))
            if ext is not None:
                batch_bytes, _ = ext
                for i in range(batch_bytes.size(0)):
                    seq = batch_bytes[i]
                    non_zero = seq[seq != 0]
                    if non_zero.numel() >= prompt_len:
                        prompts.append(non_zero[:prompt_len].clone())

        return prompts

    # ══════════════════════════════════════════════════════════════
    # PLAY 阶段: 在线训练
    # ══════════════════════════════════════════════════════════════

    def _play_step(self):
        """一次 PLAY 训练步骤。"""
        batch = self.replay_buffer.sample(self.cfg['batch_size'], self.device)

        if batch is None:
            batch = self.data_rotator.get_batch(self.cfg['batch_size'])
            if batch is None:
                time.sleep(0.2)
                return

        byte_seq, labels = batch
        bsz, seq_len = byte_seq.shape

        try:
            pos_emb = self.model.get_position_embeddings(seq_len, self.device)
            z_init, ce_loss = self.model.forward_with_ce(byte_seq, labels, pos_emb)
        except Exception as e:
            self._log(f'[PLAY] Forward error: {e}')
            return

        z_detached = [z.detach() for z in z_init]
        _, errors_hist, F_hist, F_pred = self.model.spatiotemporal_infer(
            z_detached, pos_emb,
            gamma=self.controller.current_gamma,
            T=self.cfg['T_infer'],
            return_errors=True,
            return_pred_loss=True,
        )

        D = self.dopamine.update(F_pred.item())
        D = max(D, 0.01)

        β_local = min(1.0, 0.1 + self.total_steps / 1000)
        β_conv = min(0.5, self.total_steps / 2000)
        β_local = β_local * (1.0 + self.cfg['dopamine_gamma'] * D)
        β_conv = β_conv * (1.0 + self.cfg['dopamine_gamma'] * D)

        ce_local_sum = ce_loss * (bsz * seq_len)
        scale_local = (F_pred.detach() / (ce_local_sum.detach() + 1e-8)).clamp(0.1, 10.0)
        total_loss = F_pred + β_local * scale_local * ce_local_sum

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        trainable = [p for p in self.model.parameters()
                     if p.requires_grad and p.grad is not None]
        if trainable:
            torch.nn.utils.clip_grad_norm_(trainable, self.cfg['grad_clip'])
        current_lr = self.cfg['lr'] * (1.0 + self.cfg['dopamine_beta'] * D)
        for pg in self.optimizer.param_groups:
            pg['lr'] = current_lr
        self.optimizer.step()

        self.replay_buffer.add_batch(byte_seq.detach(), labels.detach(), D=D)

        ce_val = ce_loss.item()
        F_val = F_pred.item()
        self.controller.report_loss(ce_val)

        if self.total_steps % 10 == 0:
            self._log(
                f'[PLAY] Step {self.total_steps} | '
                f'CE={ce_val:.4f} F={F_val:.1f} D={D:.3f} '
                f'lr={current_lr:.2e} buf={self.replay_buffer.size}'
            )

    # ══════════════════════════════════════════════════════════════
    # SLEEP 阶段: 离线巩固
    # ══════════════════════════════════════════════════════════════

    def _sleep_step(self):
        """SLEEP 离线巩固。"""
        self._log(f'[SLEEP] 开始离线巩固 (buffer={self.replay_buffer.size})')

        replay_steps = min(50, self.replay_buffer.size // 4)
        replay_losses = []

        for i in range(replay_steps):
            batch = self.replay_buffer.sample(self.cfg['replay_batch_size'], self.device)
            if batch is None:
                break

            byte_seq, labels = batch
            bsz, seq_len = byte_seq.shape

            try:
                pos_emb = self.model.get_position_embeddings(seq_len, self.device)
                z_init, ce_loss = self.model.forward_with_ce(byte_seq, labels, pos_emb)

                z_detached = [z.detach() for z in z_init]
                _, _, _, F_pred = self.model.spatiotemporal_infer(
                    z_detached, pos_emb,
                    gamma=self.controller.current_gamma * 0.5,
                    T=max(1, self.cfg['T_infer']),
                    return_errors=False,
                    return_pred_loss=True,
                )

                D = self.dopamine.update(F_pred.item())
                total_loss = F_pred + 0.5 * ce_loss

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                trainable = [p for p in self.model.parameters()
                             if p.requires_grad and p.grad is not None]
                if trainable:
                    torch.nn.utils.clip_grad_norm_(trainable, self.cfg['grad_clip'])
                self.optimizer.step()

                replay_losses.append(ce_loss.item())

            except Exception as e:
                self._log(f'[SLEEP] Replay error: {e}')
                continue

        if replay_losses:
            avg_loss = sum(replay_losses) / len(replay_losses)
            self._log(f'[SLEEP] 巩固完成: {len(replay_losses)} 步, avg_CE={avg_loss:.4f}')

        self._log(f'[SLEEP] 轮换外部数据源')
        self.data_rotator = DataRotator(
            self.cfg['data_dir'],
            max_seq_len=self.cfg['max_seq_len'],
            max_samples=200,
        )

    # ══════════════════════════════════════════════════════════════
    # 检查点保存
    # ══════════════════════════════════════════════════════════════

    def _save_checkpoint(self, async_save: bool = False):
        """保存模型状态。"""
        def _save():
            try:
                out_dir = os.path.join(ROOT_DIR, self.cfg['out_dir'])
                os.makedirs(out_dir, exist_ok=True)
                ckpt = {
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'controller_state': self.controller.to_dict(),
                    'total_steps': self.total_steps,
                    'lm_config': self.lm_config,
                    'cfg': self.cfg,
                }
                path = os.path.join(out_dir, f'autonomous_s{self.total_steps}.pt')
                torch.save(ckpt, path)
                latest = os.path.join(out_dir, 'autonomous_latest.pt')
                torch.save(ckpt, latest)
                self._log(f'💾 Checkpoint saved: {path}')
            except Exception as e:
                self._log(f'💾 Save error: {e}')

        if async_save:
            t = threading.Thread(target=_save, daemon=True)
            t.start()
        else:
            _save()


# ═══════════════════════════════════════════════════════════════════
# 内在动机元控制器
# ═══════════════════════════════════════════════════════════════════

class IntrinsicMetaController(MetaController):
    """支持内在动机信号 (curiosity_drive, competence_drive, boredom_signal) 的元控制器。

    扩展 MetaController:
      - curiosity_drive: 由 ICM information_gain 驱动
      - competence_drive: 由 inverse_loss 降低速度驱动
      - boredom_signal: 低 IG + 低 loss → 促进阶段切换
      - fragile_concept_targeting: 高优先级回放脆弱概念
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.curiosity_drive: float = 0.5   # [0, 1]
        self.competence_drive: float = 0.5  # [0, 1]
        self.boredom_signal: float = 0.0    # [0, 1]
        self._icm_history: list[dict] = []
        self.concept_discovery: Optional[ConceptDiscovery] = None

    def update_intrinsic(self, icm_output: dict):
        """每一步接收 ICM 输出信号。"""
        self._icm_history.append({
            'ig': icm_output.get('information_gain', 0.0),
            'pred_loss': icm_output.get('pred_loss', 0.0),
            'inverse_loss': icm_output.get('inverse_loss', 0.0),
            'uncertainty': icm_output.get('uncertainty', 0.0),
        })
        if len(self._icm_history) > 100:
            self._icm_history.pop(0)

        # curiosity_drive: 高 information_gain → 高好奇
        ig_ema = sum(d['ig'] for d in self._icm_history[-20:]) / max(len(self._icm_history[-20:]), 1)
        self.curiosity_drive = min(1.0, ig_ema * 5.0)

        # competence_drive: inverse_loss 下降速度
        if len(self._icm_history) >= 20:
            recent_inv = sum(d['inverse_loss'] for d in self._icm_history[-10:]) / 10
            older_inv = sum(d['inverse_loss'] for d in self._icm_history[-20:-10]) / 10
            if older_inv > 0:
                drop = (older_inv - recent_inv) / older_inv
                self.competence_drive = min(1.0, max(0.0, drop * 2.0))

        # boredom_signal: 低 IG + competence 高 → 无聊 → 切换阶段
        self.boredom_signal = min(1.0, max(0.0,
            (1.0 - self.curiosity_drive) * (1.0 - self.competence_drive) * 2.0
        ))

    def next_phase(self) -> str:
        """内在动机增强的阶段决策。

        覆盖 MetaController.next_phase 的部分行为:
          - boredom_signal > 0.6 → 强制从 PLAY 切换到 WAKE
          - curiosity_drive > 0.8 → 延长 PLAY (继续探索)
        """
        phase = super().next_phase()

        # boredom 打断
        if phase == 'PLAY' and self.boredom_signal > 0.6:
            self.phase = 'WAKE'
            self.play_count = 0
            self.wake_count = 0
            return 'WAKE'

        # 高好奇延长 PLAY
        if phase == 'WAKE' and self.curiosity_drive > 0.7:
            if self.wake_count < self.cfg.get('wake_steps', 20) // 2:
                self.phase = 'PLAY'  # 跳过剩余生成
                self.wake_count = 0
                return 'PLAY'

        return phase

    def get_fragile_replay_targets(self) -> list[str]:
        """获取脆弱概念 ID 列表用于回放。"""
        if self.concept_discovery is None:
            return []
        return self.concept_discovery.get_fragile_concept_ids()
