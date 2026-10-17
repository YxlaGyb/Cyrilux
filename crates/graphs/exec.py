"""
图求值
在名字环境上按声明顺序解释.

目前只有 ref 后端，同一张声明图的解释器
capture 后端 (CUDA Graph 录制与回放) 落到同一模块的同一份声明上
捕获期任何 host 同步都会显式抛错, 不得静默回落.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from .spec import Graph, validate_graph


def run_ref(graph: Graph, env: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """按声明顺序求值, 结果就地写进 env，逐帧调用不产生新容器, 返回 env 本身"""
    validate_graph(graph)
    for op in graph.ops:
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
    """
    for name, attr in graph.writes:
        setattr(net, attr, env[name])
