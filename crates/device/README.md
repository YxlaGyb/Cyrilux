---
description: "抽象物理设备代码，提供 GPU → CPU 事件桥、批量感官事件容器、CUDA 初始化与流原语。"
kind: "package-reference"
---

# device

English | [中文](README.zh.md)

## Summary

`crates.device` centralizes all CPU and GPU device code. It exposes `EventBridge`, a GPU side top-k filter and event-batch manager, plus `SensoryEventBatch`, a tensor-backed container for sensory events. It also owns CUDA initialization via `setup_cuda_device` and the `TensorRingBuffer` and `StreamPool` primitives under the `cuda` subpackage. Events cross the bridge as aligned tensors, never as per-element `.item()` calls. The module holds no model weights and no training loop; it is the transport and device plumbing that other crates consume.

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Further exploration](#further-exploration)

-----

<a id="use-this-package"></a>
## Use this package

Represent and move events.

| 导出 | 形状 | 含义 |
|---|---|---|
| `SensoryEventBatch(pos, ch, val, layer, block_id)` | Event container | Immutable-ish container of event fields, each a length-`E` tensor (`pos` int32, `ch` int16, `val` fp16, `layer` int8, `block_id` int8). `len()` is the number of events; `.device` follows the tensors. |
| `bridge.ingest_hlist(h_list, top_k=0)` | GPU top-k filter | Extracts a sensory event batch from a `SensoryFrontend` hidden-state list on the GPU. `top_k` keeps the strongest channels per position; with `0` it uses a threshold mask. Returns a batch or `None` when there are no events. |
| `bridge.push_network_events(nids, eps)` | Network event producer | Pushes active-neuron network events onto the CPU-side tensor ring buffer, dropping nothing silently (the overflow is counted). |
| `bridge.pop_network_events(max_events=10)` | Network event consumer | Consumes up to `max_events` network events and returns the `(nids, eps)` tensor pair. |

Set up the device.

| 导出 | 形状 | 含义 |
|---|---|---|
| `setup_cuda_device()` | Device selector | Returns a `torch.device` (CUDA when available, else CPU) and, on CUDA, enables cuDNN benchmark. |

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

`event_bridge.py` defines both public symbols. `SensoryEventBatch` is a lightweight struct with `__slots__`; it carries five aligned tensors and exposes `__len__` and a `device` property. `EventBridge` keeps two fixed-size tensor ring buffers, one for sensory events and one for network events, and preallocates every backing tensor in the constructor so hot-path updates never allocate.

`ingest_hlist` does the top-k selection on the GPU using `torch.topk` and `torch.where`, which avoids per-element GPU-to-CPU transfers. During the warmup window it lowers the threshold and keeps all channels, then it converges to the configured `sensory_threshold` or a per-position top-k count. The returned batch is truncated to `max_sensory_events`, and the overrun is counted in `sensory_dropped`.

`push_network_events` and `pop_network_events` manage the CPU-side network ring manually. The producer truncates to capacity and records dropped events, and the consumer reconstructs rows across a ring wrap when needed. `get_stats` exposes the cumulative totals and current drop counters plus the remaining warmup steps.

The `cuda` subpackage holds three helpers. `setup_cuda_device` is the single CUDA tuning entry point. `TensorRingBuffer` is a general fixed-capacity tensor ring with `push`, `pop`, and `__len__`. `StreamPool` is a stub: it creates the requested streams lazily.

-----

<a id="further-exploration"></a>
## Further exploration

- [hybrid](../hybrid/README.md): the device bridge feeds event batches into the dense perception pipeline.
- [architecture](../../docs/architecture.md)：top-k 与稀疏、事件驱动传输设计。