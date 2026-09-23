"""pkg.device — 物理设备抽象层.

CPU 和 GPU 的所有设备代码集中于此.

模块结构:
  - event_bridge.py — CPU ↔ GPU 事件队列桥接
  - stream.py — 持续运行循环入口 (仅兼容别名)
"""

from .event_bridge import EventBridge, SensoryEventBatch

__all__ = [
    "EventBridge",
    "SensoryEventBatch",
]
