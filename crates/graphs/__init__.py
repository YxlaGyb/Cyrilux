"""
静态图
声明层 (Op / Graph) 与求值层 (ref 解释器; capture 后端同源)
"""

from .exec import apply_writes, run_ref
from .spec import NET, Graph, Op, defined_names, graph_fingerprint, source_digest, validate_graph

__all__ = [
    "NET",
    "Graph",
    "Op",
    "apply_writes",
    "defined_names",
    "graph_fingerprint",
    "run_ref",
    "source_digest",
    "validate_graph",
]
