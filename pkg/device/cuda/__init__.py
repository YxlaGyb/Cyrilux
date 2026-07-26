"""CUDA 设备初始化 — cuDNN, tf32, matmul precision 配置."""

from __future__ import annotations

import torch


def setup_cuda_device() -> torch.device:
    """统一 CUDA 初始化入口.

    设置 cuDNN benchmark, tf32, matmul precision.
    从 TrainingLoop._setup_environment() 移入.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
    return device
