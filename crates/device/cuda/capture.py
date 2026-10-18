"""
CUDA Graph 捕获后端

声明层只声明 (``crates.graphs``), 设备侧把声明录成一张图再回放: 捕获与回放是
设备机制, 因此张量快照、同步陷阱、私有内存池与结构纪元都住在这里.

捕获期任何 host 同步都会显式抛错, 不得静默回落.
"""

from __future__ import annotations

import time
from collections.abc import Hashable, Mapping, MutableMapping
from typing import Any

import torch

from crates.graphs import CaptureHostSync, Graph, note_capture_failed, run_ref, validate_graph

_WARMUP = 3


def capture_available(subject: Any = None) -> bool:
    """捕获靠 CUDA Graph, 无 CUDA 或对象不在 CUDA 上时 ref 是唯一后端"""
    if not torch.cuda.is_available():
        return False
    if subject is None:
        return True
    device = getattr(subject, "device", None)
    if device is None and hasattr(subject, "parameters"):
        param = next(subject.parameters(), None)
        device = param.device if param is not None else None
    return device is not None and device.type == "cuda"


def _blocked(name: str, where: str) -> Any:
    def _forbid(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        raise CaptureHostSync(f"{where}: 捕获期宿主同步 Tensor.{name}()")

    return _forbid


class _SyncTrap:
    """捕获期封死张量到宿主的读口"""

    _FORBID = ("item", "cpu", "tolist", "numpy", "__bool__", "__int__", "__float__", "__index__")

    def __init__(self, where: str) -> None:
        self._where = where
        self._saved: dict[str, Any] = {}

    def __enter__(self) -> _SyncTrap:
        for name in self._FORBID:
            self._saved[name] = getattr(torch.Tensor, name)
            setattr(torch.Tensor, name, _blocked(name, self._where))
        return self

    def __exit__(self, *exc: Any) -> bool:
        for name, fn in self._saved.items():
            setattr(torch.Tensor, name, fn)
        self._saved.clear()
        return False


def _rng_state() -> tuple[Any, list[Any]]:
    return torch.get_rng_state(), torch.cuda.get_rng_state_all()


def _restore_rng(state: tuple[Any, list[Any]]) -> None:
    cpu, gpus = state
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state_all(gpus)


class _Snapshot:
    """
    捕获前后的宿主快照

    预热与捕获各推进一次宿主状态, 但两者的还原语义不同:
    - 预热不能被后续看见: 属性若被换成新对象, 必须绑回原对象并删掉新属性,
      否则捕获帧读到的是预热帧造的图 (分支判据按属性存在性决定);
    - 捕获必须被后续看见: 图写的是捕获帧造出来的那些对象, 只能还原值不能换回对象,
      否则回放写进没人读的旧张量.
    """

    __slots__ = ("slots", "live", "frozen")

    def __init__(self, env: Mapping[str, Any], net: Any) -> None:
        self.slots: list[tuple[dict, str, torch.Tensor, torch.Tensor]] = []
        self.live: list[dict] = []
        self.frozen: list[set[str]] = []
        containers: list[dict] = [env] if isinstance(env, dict) else []
        if net is not None:
            for module in net.modules():
                for attr in ("_parameters", "_buffers"):
                    container = getattr(module, attr, None)
                    if isinstance(container, dict):
                        containers.append(container)
                fields = vars(module)
                containers.append(fields)
                self.live.append(fields)
                self.frozen.append(set(fields))
        seen: set[int] = set()
        for container in containers:
            for name, value in list(container.items()):
                if isinstance(value, torch.Tensor) and id(value) not in seen:
                    seen.add(id(value))
                    self.slots.append((container, name, value, value.clone()))

    def size_bytes(self) -> int:
        return sum(clone.numel() * clone.element_size() for _, _, _, clone in self.slots)

    def unbind(self) -> None:
        """把预热期的对象与属性全部撤掉, 让捕获帧看不见预热"""
        for container, name, original, clone in self.slots:
            if container.get(name) is not original:
                container[name] = original
            original.copy_(clone)
        for fields, known in zip(self.live, self.frozen):
            for name in [k for k in fields if k not in known]:
                del fields[name]

    def rewind(self) -> None:
        """把捕获帧当前绑定的对象倒回捕获前的值, 首帧回放才与 ref 同步"""
        for container, name, _, clone in self.slots:
            current = container.get(name)
            if (isinstance(current, torch.Tensor) and current.shape == clone.shape
                    and current.dtype == clone.dtype):
                current.copy_(clone)


def _struct_epoch(net: Any) -> int:
    return getattr(net, "_struct_epoch", 0)


def _owned_ids(net: Any) -> set[int]:
    """net 名下的参数与缓冲, 这些对象的地址在训练循环里保持不变"""
    if net is None:
        return set()
    return {id(t) for t in list(net.parameters()) + list(net.buffers())}


class CaptureSession:
    """
    一个变体的一张捕获图: 私有内存池 + 捕获期环境

    捕获前的权重/缓冲快照在捕获结束后原位还原, 使首步回放与 ref 的资产同源.
    非 net 持有的张量入参由会话私有缓冲承接: 外部每步换张量, 地址仍须固定.
    """

    __slots__ = ("graph", "env", "capture_ms", "snapshot_bytes", "replays", "_g", "_staged")

    @torch.no_grad()
    def __init__(self, graph: Graph, env: MutableMapping[str, Any], net: Any = None) -> None:
        validate_graph(graph)
        owned = _owned_ids(net)
        self._staged: dict[str, torch.Tensor] = {}
        for name in graph.inputs:
            value = env.get(name)
            if not isinstance(value, torch.Tensor) or id(value) in owned:
                continue
            buf = torch.empty_like(value)
            buf.copy_(value)
            env[name] = buf
            self._staged[name] = buf
        snap = _Snapshot(env, net)
        self.snapshot_bytes = snap.size_bytes()
        rng = _rng_state()
        started = time.perf_counter()
        try:
            with _SyncTrap(graph.name):
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(_WARMUP):
                        run_ref(graph, env)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                _restore_rng(rng)
                snap.unbind()
                graph_obj = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph_obj):
                    run_ref(graph, env)
        finally:
            snap.rewind()
            _restore_rng(rng)
        self.graph = graph
        self._g = graph_obj
        self.env = dict(env)
        self.capture_ms = (time.perf_counter() - started) * 1000.0
        self.replays = 0
        self._g.replay()
        self.replays += 1

    def stage(self, env: Mapping[str, Any]) -> None:
        """把本次入参搬进会话私有缓冲, 只搬非 net 持有的那些"""
        for name, buf in self._staged.items():
            value = env.get(name)
            if not isinstance(value, torch.Tensor):
                continue
            if value.shape != buf.shape or value.dtype != buf.dtype:
                raise ValueError(f"{self.graph.name}: 入参 {name} 形状 {tuple(value.shape)} 与捕获时 {tuple(buf.shape)} 不符")
            buf.copy_(value)

    def replay(self, strict: bool = False) -> MutableMapping[str, Any]:
        if strict:
            with _SyncTrap(f"{self.graph.name}:replay"):
                self._g.replay()
        else:
            self._g.replay()
        self.replays += 1
        return self.env


class CaptureBank:
    """
    按变体键缓存的捕获会话集
    变体键命中即回放; 结构纪元一变, 该族会话全部作废并重捕
    """

    __slots__ = ("name", "net", "captures", "_sessions", "_epoch")

    def __init__(self, name: str, net: Any) -> None:
        self.name = name
        self.net = net
        self._sessions: dict[Hashable, CaptureSession] = {}
        self._epoch = _struct_epoch(net)
        self.captures = 0

    def __len__(self) -> int:
        return len(self._sessions)

    @property
    def capture_ms(self) -> float:
        return sum(session.capture_ms for session in self._sessions.values())

    def drop(self) -> None:
        self._sessions.clear()
        self._epoch = _struct_epoch(self.net)

    def step(self, key: Hashable, graph: Graph, env: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        epoch = _struct_epoch(self.net)
        if epoch != self._epoch:
            self._sessions.clear()
            self._epoch = epoch
        session = self._sessions.get(key)
        if session is not None:
            session.stage(env)
            return session.replay()
        try:
            session = CaptureSession(graph, env, net=self.net)
        except (CaptureHostSync, RuntimeError):
            note_capture_failed()
            raise
        self._sessions[key] = session
        self.captures += 1
        return session.env
