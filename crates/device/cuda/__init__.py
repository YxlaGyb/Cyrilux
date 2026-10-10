"""
CUDA 设备初始化

cuDNN benchmark 配置.
"""

from __future__ import annotations

import torch


def setup_cuda_device() -> torch.device:
    """统一 CUDA 初始化入口.

    设置 cuDNN benchmark.
    从 TrainingLoop._setup_environment() 移入.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return device
