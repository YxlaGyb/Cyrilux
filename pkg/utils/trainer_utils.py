"""
训练工具函数: Logger / 学习率调度 / 随机种子 / 模型参数统计
"""
import os, sys, math, random, time
import numpy as np
import torch


# ── 日志 ──

class Logger:
    """简单日志类，支持同时输出到 stdout 和文件。"""
    def __init__(self, path: str = None, mode: str = 'a'):
        self.terminal = sys.stdout
        self.file = None
        if path:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            self.file = open(path, mode, encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        if self.file:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        self.terminal.flush()
        if self.file:
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


# ── 学习率 ──

def get_lr(global_step: int, total_steps: int, lr: float,
           warmup_ratio: float = 0.1) -> float:
    """
    Cosine 学习率调度 (含 warmup)。

    公式:
      - warmup 阶段: lr * (step / warmup_steps)
      - cosine 阶段: lr * (0.1 + 0.45 * (1 + cos(π * (step - warmup) / (total - warmup))))
    """
    warmup_steps = int(total_steps * warmup_ratio)
    if global_step <= warmup_steps:
        return lr * global_step / max(warmup_steps, 1)

    progress = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr * (0.1 + 0.45 * (1.0 + math.cos(math.pi * progress)))


# ── 随机种子 ──

def setup_seed(seed: int = 42):
    """固定所有随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── 参数统计 ──
# count_parameters / count_budget 已删除 — 页面化池不使用 nn.Module.parameters().
# effective_params / effective_capacity 保留, 用于分析池内权重稀疏度.


def effective_params(tensor: torch.Tensor, eps: float = 1e-4) -> int:
    """统计张量中当前"有效"的参数数。

    使用相对阈值: |w| > max(|w|) * eps 的权重计入有效参数。
    eps=1e-4 意味着"至少是最大权重的 0.01%"。
    Oja 的第二项 -ε^T·ε·W_old 会将弱连接推向零,
    自然低于阈值 → 有效参数减少。
    """
    w = tensor.detach()
    w_max = w.abs().max().item()
    if w_max == 0:
        return 0
    return (w.abs() > w_max * eps).sum().item()


def effective_capacity(model: torch.nn.Module, eps: float = 1e-4) -> float:
    """统计模型当前有效参数数 (百万单位)。"""
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += effective_params(p, eps)
    return total / 1e6
