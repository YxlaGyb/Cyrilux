"""
Cyrilux 全局共享常量。

集中管理跨模块重复出现的硬编码值:
  - DEVICE / DEVICE_STR   设备选择 (统一 cuda:0 / cpu)
  - 后续可扩展: DTYPE, SEED 等
"""
import torch

DEVICE: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
"""torch device 对象, 供 model.to(device) 等使用."""

DEVICE_STR: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
"""设备字符串, 供日志/参数传递使用."""
