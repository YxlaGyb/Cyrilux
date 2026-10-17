"""
静态图声明层

Op / Graph

图只持结构 (算子序列、名字接线、写回契约、变体键), 不持张量
张量一律经 env 传入, 算子一律是模块级函数身份
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

NET = "net"
_TAG = b"cyrilux.graph.v1\0"


@dataclass(frozen=True)
class Op:
    """
    一次具名调用
    fn 是既有函数身份, inputs / outputs 是环境中的张量名.
    """

    name: str
    fn: Callable[..., Any]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    @property
    def source(self) -> str:
        return f"{getattr(self.fn, '__module__', '?')}.{getattr(self.fn, '__qualname__', '?')}"


@dataclass(frozen=True)
class Graph:
    """
    一段计算的唯一事实源
    按序表 + 写回契约 + 变体键
    """

    name: str
    ops: tuple[Op, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    writes: tuple[tuple[str, str], ...] = ()
    variant: tuple[tuple[str, Any], ...] = ()

    def fingerprint(self, env: Mapping[str, Any] | None = None) -> str:
        return graph_fingerprint(self, env)


def validate_graph(graph: Graph) -> None:
    """
    结构自检
    名字接线闭合、算子名唯一、算子必须是模块级函数
    """
    if not graph.name:
        raise ValueError("图必须有名字")
    defined: set[str] = set(graph.inputs)
    if len(defined) != len(graph.inputs):
        raise ValueError(f"{graph.name}: 输入名重复")
    seen: set[str] = set()
    for op in graph.ops:
        if not op.name or op.name in seen:
            raise ValueError(f"{graph.name}: 算子名缺失或重复: {op.name!r}")
        seen.add(op.name)
        _check_fn(graph, op)
        if not op.inputs:
            raise ValueError(f"{graph.name}:{op.name}: 输入名不得为空")
        for src in op.inputs:
            if src not in defined:
                raise ValueError(f"{graph.name}:{op.name}: 读未定义名 {src}")
        for dst in op.outputs:
            if dst in defined:
                raise ValueError(f"{graph.name}:{op.name}: 名 {dst} 被重复定义")
            defined.add(dst)
    for name in graph.outputs:
        if name not in defined:
            raise ValueError(f"{graph.name}: 输出名 {name} 未定义")
    for name, attr in graph.writes:
        if name not in defined:
            raise ValueError(f"{graph.name}: 写回名 {name} 未定义")
        if not attr:
            raise ValueError(f"{graph.name}: 写回属性名为空")


def defined_names(graph: Graph) -> frozenset[str]:
    """图中所有已定义名 (输入 ∪ 各算子输出)."""
    defined = set(graph.inputs)
    for op in graph.ops:
        defined.update(op.outputs)
    return frozenset(defined)


def _check_fn(graph: Graph, op: Op) -> None:
    if not callable(op.fn):
        raise ValueError(f"{graph.name}:{op.name}: 算子不可调用")
    qual = getattr(op.fn, "__qualname__", "")
    if "<lambda>" in qual or "<locals>" in qual:
        raise ValueError(f"{graph.name}:{op.name}: 算子必须是模块级函数: {qual}")


_SRC: dict[Callable[..., Any], str] = {}


def source_digest(fn: Callable[..., Any]) -> str:
    """
    算子源码摘要
    声明体一动, 指纹即变
    """
    digest = _SRC.get(fn)
    if digest is None:
        try:
            text = inspect.getsource(fn)
        except (OSError, TypeError):
            text = f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', '?')}"
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
        _SRC[fn] = digest
    return digest


def _shape_dtype(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return type(value).__name__
    return f"{tuple(shape)}:{getattr(value, 'dtype', '?')}"


def graph_fingerprint(graph: Graph, env: Mapping[str, Any] | None = None) -> str:
    """
    架构指纹
    算子序列 + 算子源码摘要 + 名字接线 + 变体键 (+ 绑定后的形状 / dtype).

    不含权重值: 架构静态、状态动态, 指纹必须与训练步数无关.
    env 传入时必须完整绑定全图所有名字, 否则指纹会随调用时机变化, 不再是架构事实.
    """
    if env is not None:
        missing = sorted(defined_names(graph) - set(env))
        if missing:
            raise ValueError(f"{graph.name}: 指纹需要完整环境, 缺 {missing[:4]}")
    h = hashlib.blake2b(digest_size=16)
    h.update(_TAG)
    h.update(graph.name.encode("utf-8"))
    h.update(f"|writes={graph.writes!r}|out={graph.outputs!r}".encode())
    for key, value in graph.variant:
        h.update(f"|{key}={value!r}".encode())
    for op in graph.ops:
        wiring = f"\x1e{op.name}<{op.source}>({','.join(op.inputs)})->({','.join(op.outputs)})"
        h.update(wiring.encode("utf-8"))
        h.update(b"\x1f" + source_digest(op.fn).encode("ascii"))
        if env is None:
            continue
        for src in op.inputs:
            h.update(f"\x1d{src}={_shape_dtype(env[src])}".encode())
    return h.hexdigest()
