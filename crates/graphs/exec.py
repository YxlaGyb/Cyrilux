"""
图求值
两后端消费同一份声明: ref (解释执行) 与 capture (CUDA Graph 捕获回放)

本层只依赖标准库: 捕获是设备机制, 由 ``crates.device.cuda.capture`` 承接, 它把同一份
声明录成图; 捕获期任何 host 同步都会显式抛错, 不得静默回落.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any

from .spec import Graph, Op, validate_graph

REF = "ref"
CAPTURE = "capture"

_backend = CAPTURE
_capture_failed = 0


class CaptureHostSync(RuntimeError):
    """捕获期出现宿主同步, 图不再是静态的"""


def backend() -> str:
    return _backend


def set_backend(name: str) -> str:
    global _backend
    if name not in (REF, CAPTURE):
        raise ValueError(f"未知后端: {name!r}")
    _backend = name
    return _backend


def capture_failed() -> int:
    return _capture_failed


def note_capture_failed() -> int:
    """捕获会话失败计数, 由后端实现登记; 失败必须冒泡, 不许静默回落"""
    global _capture_failed
    _capture_failed += 1
    return _capture_failed


def reset_capture_failed() -> None:
    global _capture_failed
    _capture_failed = 0


def run_ref(
    graph: Graph,
    env: MutableMapping[str, Any],
    on_op: Callable[[Op], None] | None = None,
) -> MutableMapping[str, Any]:
    """按声明顺序求值, 结果就地写进 env，逐帧调用不产生新容器, 返回 env 本身"""
    validate_graph(graph)
    for op in graph.ops:
        if on_op is not None:
            on_op(op)
        result = op.fn(*[env[name] for name in op.inputs])
        if not op.outputs:
            continue
        if len(op.outputs) == 1:
            env[op.outputs[0]] = result
        elif isinstance(result, tuple) and len(result) == len(op.outputs):
            for name, value in zip(op.outputs, result):
                env[name] = value
        else:
            raise ValueError(f"{graph.name}:{op.name}: 期望 {len(op.outputs)} 个返回值")
    return env


def apply_writes(graph: Graph, env: MutableMapping[str, Any], net: Any) -> None:
    """
    按声明的写回契约把结果落到宿主对象上
    属性写是宿主协议, 图不持 net
    同形同型则原位覆盖: 捕获图的入参按地址绑定, 换对象即让后续回放读旧值
    """
    for name, attr in graph.writes:
        value = env[name]
        current = getattr(net, attr, None)
        if _same_slot(current, value):
            target = current.detach() if getattr(current, "requires_grad", False) else current
            target.copy_(value)
            continue
        setattr(net, attr, value)


def _same_slot(current: Any, value: Any) -> bool:
    """既有属性与结果同形同型同设备才能原位覆盖; 只看张量协议, 不认框架"""
    if current is None or current is value:
        return False
    if not hasattr(current, "copy_") or not hasattr(value, "copy_"):
        return False
    return (
        current.shape == value.shape
        and current.dtype == value.dtype
        and current.device == value.device
    )
