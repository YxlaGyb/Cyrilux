---
description: "密集感知-预测-行动闭环的引擎，前馈、推理、生成与声明式前向图。"
kind: "package-reference"
---

# hybrid

[English](README.md) | 中文

## 概述

`crates.hybrid` 实现密集 PPA（感知-预测-行动）闭环的运行时引擎。`forward.py` 拥有 `ForwardEngine`，即入口：`_predict` 装配环境并运行 `forward_graph.py` 声明的图。该图给出从字节输入沿 L0 到 L6 的激活路径、时间递归核、稀疏概念槽绑定，并执行自回归续写与生成。动态神经元机制（拓扑修剪与记忆单元群的生死）放在同级 crate `crates.neuron`。逐域学习更新放在同级子 crate `crates.hybrid.learning`。这些引擎共同消费 `crates.model` 声明的网络与配置，并以迹调制、无反向传播的方式驱动它。

## 目录

- [使用本包](#use-this-package)
- [理解实现](#understand-the-implementation)
- [进一步探索](#further-exploration)

-----

<a id="use-this-package"></a>
## 使用本包

引擎构造函数接受一个网络并持有其引用。

| 导出 | 形状 | 含义 |
|---|---|---|
| `ForwardEngine(net)` | 引擎 | 持有网络并驱动感知、推理、绑定与生成。由 `HybridPCNet.__init__` 自动构造，暴露为 `net.forward_engine`。 |

前馈与生成。

| 导出 | 形状 | 含义 |
|---|---|---|
| `engine.forward(byte_ids)` | 前馈 | 推理前馈，返回未来预测偏差，关闭 ACh，完全确定。 |
| `engine._predict(byte_ids, store_state, is_inference)` | 核心前馈 | 薄壳：装配环境并运行声明图。感知、Foldiak 去相关、去中心化与增量预测已作为声明算子搬到 `forward_graph.py`。 |
| `engine.generate(prompt, n_tokens, temperature, dev)` | 生成 | 让模型在提示字符串上前进并返回生成的字节。 |
| `engine.continuation(byte_ids, n_gen, temperature, rep_backstop)` | 自回归续写 | 训练 rollout 与验证共用。对输入前缀批量处理并返回追加的 token。 |

-----

<a id="understand-the-implementation"></a>
## 理解实现

`ForwardEngine._predict` 是唯一的激活路径。它在 L0 对字节序列做 one-hot 编码，通过 `_recurrent`（带逐层归一化的纯 matmul）应用时间核 `W_t*`，逐层经过 L4、L2、L3、L5、L6，在 L5 应用 Foldiak 反希伯去相关，并把去中心化的 z5 送到读出端，而原始 z5 则喂给 `W_diff` 产生世界模型预测误差。在自由运行模式下外部输入被置零，活动完全由内部递归与三个加性振荡器驱动，可选的 STP 资源动力学对递归做门控。`_bind` 把 z4 经 `W_bind` 投影到带 top-k 胜者全取的稀疏概念槽，并通过 `W_act` 暴露由此产生的动作电位。

`forward_graph.py` 是声明层。`ForwardGraphs.declare()` 产出一个 `crates.graphs.Graph`，其算子就是激活路径上具名的每一步（其中 25 个来自 `crates.kernels.cascade`，4 个来自 `crates.kernels.align` 的视图），`_inputs()` 点名图读取的每一个张量，`writes` 列出运行后由 `crates.graphs.apply_writes` 写回的网络属性。非自由档声明 213 个算子（含 `mu3_noise` 步时为 214），自由运行档 237 个（238 个），其中 63 个记忆步与 63 个新奇度步各自展开成节点。图按“变体 + 参数与缓冲名字序列”缓存并记录指纹，因此形状与权重相同的两次调用复用一个图；指纹只是可复现性记录，没有任何学习信号读它。`ForwardGraphs.clear()` 已实现但尚无调用点：结构事件（形状、行数或缓冲拓扑变化）目前会保留缓存图，直到接线落地。`writes` 的短表（`store_state=False`）同样在当前调用点不可达，调用点全部传 `store_state=True`；保留它是为了与签名对齐，不是为兼容。

`continuation` 为训练与评估复用同一条 logits 管线。温度可以传张量从而在 GPU 上直除、零标量同步，还带可选的重复问题兜底（屏蔽周期性字节），并用预建掩码屏蔽非法续字节以强制 UTF-8 合法性。多步承诺遥测由一个可选开关控制。

`crates.neuron` 拥有拓扑重塑与记忆单元生命周期：一个重塑回合如何把某层的行映射权重与其下游列用同一置换一起重塑，以及记忆单元如何出生与死亡。本包不含任何重塑逻辑：编排层轮询 `net.maybe_prune(step)`，网络再委托给 `net.pruner.decide()` 与 `net.pruner.compact()`。

-----

<a id="further-exploration"></a>
## 进一步探索

- [graphs](../graphs/README.zh.md)：声明层，拥有 `Graph`、`Op`、参考执行器与写回器。
- [model](../model/README.zh.md)：这些引擎操作的权重与配置声明。
- [learning](learning/README.zh.md)：逐域学习更新。
- [neuron](../neuron/README.zh.md)：动态神经元机制，主链修剪与记忆单元群生死。