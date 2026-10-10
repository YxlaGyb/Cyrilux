---
description: "device：抽象物理设备代码，提供 GPU→CPU 事件桥、批量感官事件容器、CUDA 初始化与流原语。"
kind: "package-reference"
---

# device

[English](README.md) | 中文

## 概述

`crates.device` 集中了所有 CPU 与 GPU 设备代码。它暴露 `EventBridge`（GPU 侧 top-k 过滤与事件批管理器）以及 `SensoryEventBatch`（张量化的感官事件容器）。它还通过 `setup_cuda_device` 拥有 CUDA 初始化，并在 `cuda` 子包下提供 `TensorRingBuffer` 与 `StreamPool` 原语。事件以对齐张量跨越桥，而非逐元素 `.item()` 调用。本模块不含模型权重，也没有训练循环；它是其他 crate 消费的传输与设备管道。

## 目录

- [使用本包](#use-this-package)
- [理解实现](#understand-the-implementation)
- [进一步探索](#further-exploration)

-----

<a id="use-this-package"></a>
## 使用本包

表示并搬运事件。

| 导出 | 形状 | 含义 |
|---|---|---|
| `SensoryEventBatch(pos, ch, val, layer, block_id)` | 事件容器 | 近乎不可变的事件字段容器，每个字段是长度为 `E` 的张量（`pos` int32、`ch` int16、`val` fp16、`layer` int8、`block_id` int8）。`len()` 为事件数；`.device` 跟随张量。 |
| `bridge.ingest_hlist(h_list, top_k=0)` | GPU top-k 过滤 | 在 GPU 上从 `SensoryFrontend` 隐藏状态列表提取感官事件批。`top_k` 保留每个位置最强的通道；为 `0` 时使用阈值掩码。若无事件则返回 `None`。 |
| `bridge.push_network_events(nids, eps)` | 网络事件生产者 | 把活跃神经元网络事件推入 CPU 侧张量环形缓冲，溢出被计数而非静默丢弃。 |
| `bridge.pop_network_events(max_events=10)` | 网络事件消费者 | 消费至多 `max_events` 条网络事件并返回 `(nids, eps)` 张量对。 |

配置设备。

| 导出 | 形状 | 含义 |
|---|---|---|
| `setup_cuda_device()` | 设备选择 | 返回 `torch.device`（有 CUDA 用 CUDA，否则 CPU），在 CUDA 上启用 cuDNN benchmark。 |

-----

<a id="understand-the-implementation"></a>
## 理解实现

`event_bridge.py` 定义了两个公开符号。`SensoryEventBatch` 用 `__slots__` 声明为轻量结构，携带五个对齐张量并暴露 `__len__` 与 `device` 属性。`EventBridge` 维护两个固定大小张量环形缓冲（感官事件一个、网络事件一个），并在构造器中预分配所有后备张量，使热路径更新永不分配内存。

`ingest_hlist` 用 `torch.topk` 与 `torch.where` 在 GPU 上完成 top-k 选择，避免逐元素 GPU→CPU 传输。在 warmup 窗口内它降低阈值并保留全部通道，随后收敛到配置的 `sensory_threshold` 或逐位置的 top-k 数量。返回的批被截断到 `max_sensory_events`，超出量计入 `sensory_dropped`。

`push_network_events` 与 `pop_network_events` 手工管理 CPU 侧网络环形缓冲。生产者截断到容量并记录丢弃事件，消费者在需要时跨环回绕重建行。`get_stats` 暴露累计总数、当前丢弃计数以及剩余 warmup 步数。

`cuda` 子包包含三个辅助。`setup_cuda_device` 是唯一的 CUDA 调优入口。`TensorRingBuffer` 是通用固定容量张量环形缓冲，带 `push`、`pop`、`__len__`。`StreamPool` 是 stub：惰性创建请求的流。

-----

<a id="further-exploration"></a>
## 进一步探索

- [hybrid](../hybrid/README.zh.md)：设备桥把事件批馈入密集感知管线。
- [architecture](../../docs/architecture.md)：top-k 与稀疏、事件驱动传输设计。