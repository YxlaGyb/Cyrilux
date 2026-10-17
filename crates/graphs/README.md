---
description: "the declaration layer that turns every stepwise tensor computation into one static graph."
kind: "package-reference"
---

# graphs

English | [中文](README.zh.md)

## Summary

`crates.graphs` is the declaration layer for static graphs. It holds `Op` and `Graph`: the operator sequence, the name wiring, the write-back contract and the variant key that together are the single source of truth for a stretch of computation. It depends only on the standard library, holds no tensor and no network reference, and never performs arithmetic itself: an operator is the identity of a module-level function that already exists elsewhere. Two layers: `spec` declares and self-checks, `exec` interprets a declaration on a name environment. The same declaration will drive the CUDA Graph capture backend.

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Further exploration](#further-exploration)

-----

<a id="use-this-package"></a>
## Use this package

Import the declaration names and the reference interpreter from the package root.

```python
from crates.kernels import rms_norm
from crates.graphs import Graph, Op, apply_writes, run_ref, validate_graph

graph = Graph(
    name="hidden",
    ops=(Op("norm", rms_norm, ("pre",), ("post",)),),
    inputs=("pre",),
    outputs=("post",),
    writes=(("post", "_post"),),
    variant=(("S", 8),),
)
validate_graph(graph)
run_ref(graph, env)
apply_writes(graph, env, net)
```

| Export | Module | Shape | Meaning |
|---|---|---|---|
| `Op(name, fn, inputs, outputs)` | `spec` | declaration | One named call: the identity of `fn`, plus the environment names it reads and writes. |
| `Graph(name, ops, inputs, outputs, writes, variant)` | `spec` | declaration | The ordered operator sequence, the write-back contract and the variant key. |
| `validate_graph(graph)` | `spec` | `None` or raises | Structural self-check: closed wiring, unique operator names, module-level functions only. |
| `defined_names(graph)` | `spec` | `frozenset[str]` | Every name the graph defines: inputs plus operator outputs. |
| `source_digest(fn)` | `spec` | hex | Digest of an operator's source text, cached per function. |
| `graph_fingerprint(graph, env)` | `spec` | 32 hex | Architecture fingerprint; weights excluded, environment optional but all or nothing. |
| `Graph.fingerprint(env)` | `spec` | 32 hex | The same fingerprint as a method on the declaration. |
| `run_ref(graph, env)` | `exec` | `env` | The reference interpreter: evaluates in declaration order, writes results in place. |
| `apply_writes(graph, env, net)` | `exec` | `None` | Applies the write-back contract to the host object. |
| `NET` | `spec` | `str` | The reserved environment name (`"net"`) for the host object. |

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

Reproducibility is the first-order gain of this shape. A declaration is data: the same operators, wiring and variant key expand into the same graph on any machine, and the fingerprint recorded beside a run says which graph produced it. Weights and step counts are state, not architecture, so they never enter the fingerprint.

The graph holds structure only. Tensors arrive through a name environment, an operator is the identity of a function that lives in another crate, and the declaration must not introduce arithmetic of its own: a computation that is not expressed as an operator call does not belong in a graph. `validate_graph` enforces what can be checked mechanically: every read name is defined before use, no output name is defined twice, operator names are unique, an operator is callable and is not a lambda or a local function, and every write-back name is defined. `writes` pairs an environment name with a host attribute and is the only point where a graph touches the host object, which keeps the graph itself free of any network reference.

`run_ref` evaluates the declaration in order, stores each result back into the environment and returns that same environment, so a per-frame call allocates no new container. An operator with several outputs must return a tuple of that arity, and anything else raises. The write-back step is separate by design: a segment can be evaluated and fingerprinted without touching the host.

The fingerprint is a 128-bit BLAKE2 digest over a version tag, the graph name, the write-back contract, the output names, the variant key, and then per operator its name, its source digest, its wiring and, when an environment is bound, the shape and dtype of each input. Binding is all or nothing: `graph_fingerprint` refuses an incomplete environment, because a partially bound graph would make the fingerprint depend on when it was taken instead of on the architecture. `source_digest` is cached per function, so editing a declaration body changes the fingerprint while an untouched one stays stable.

Two red lines are checked mechanically by the coverage census rather than by convention: this package imports no `torch` and holds no tensor, and no graph contains arithmetic that is not an operator call. The capture backend lands on the same declarations in the same module: capturing records what the declaration already says, and a host synchronization during capture raises instead of falling back to eager execution.

-----

<a id="further-exploration"></a>
## Further exploration

- [kernels](../kernels/README.md): the operator library the declarations call, and the only place new arithmetic is added.
- [dense](../hybrid/README.md): the forward path whose segments are declared as graphs.
- [learning](../hybrid/learning/README.md): the learning path that receives the same treatment.
