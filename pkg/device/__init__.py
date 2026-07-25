"""pkg.device — 物理设备抽象层。

CPU 和 GPU 的所有设备代码集中于此。

模块结构:
  - sensory_frontend.py  — GPU Conv1D 感官前端
  - event_bridge.py      — CPU ↔ GPU 事件队列桥接
  - stream.py            — 持续运行循环入口 (仅兼容别名)
"""

from .sensory_frontend import SensoryConvBlock, SensoryFrontend
from .event_bridge import EventBridge, SensoryEventQueue, NetworkEventQueue

__all__ = [
    "SensoryConvBlock",
    "SensoryFrontend",
    "EventBridge",
    "SensoryEventQueue",
    "NetworkEventQueue",
]
