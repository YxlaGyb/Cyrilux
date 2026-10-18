"""
静态图
声明层 (Op / Graph) 与求值层 (ref 解释器; capture 后端同源)
"""

from .exec import (
    CAPTURE,
    REF,
    CaptureHostSync,
    apply_writes,
    backend,
    capture_failed,
    note_capture_failed,
    reset_capture_failed,
    run_ref,
    set_backend,
)
from .spec import NET, Graph, Op, defined_names, graph_fingerprint, source_digest, validate_graph

__all__ = [
    "CAPTURE",
    "NET",
    "REF",
    "CaptureHostSync",
    "Graph",
    "Op",
    "apply_writes",
    "backend",
    "capture_failed",
    "defined_names",
    "graph_fingerprint",
    "note_capture_failed",
    "reset_capture_failed",
    "run_ref",
    "set_backend",
    "source_digest",
    "validate_graph",
]
