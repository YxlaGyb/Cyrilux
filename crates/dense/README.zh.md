---
description: "dense：密集感知-预测-行动闭环的引擎，前馈、推理、生成与动态修剪。"
kind: "package-reference"
---

# dense

[English](README.md) | 中文

## 概述

`crates.dense` 实现密集 PPA（感知-预测-行动）闭环的运行时引擎。`forward.py` 拥有 `ForwardEngine`，它从字节输入沿 L0 到 L6 逐层走过激活路径，应用时间递归核，绑定稀疏概念槽，并执行自回归续写与生成。`pruning.py` 拥有 `PruningEngine`，它通过淘汰不再有贡献的神经元来重塑拓扑。逐域学习更新放在同级子 crate `crates.dense.learning`。这些引擎共同消费 `crates.model` 声明的网络与配置，并以纯希伯、无反向传播的方式驱动它。

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
| `ForwardEngine(net)` | 引擎 | 持有网络并驱动感知、推理、绑定与生成。由 `DensePCNet.__init__` 自动构造，暴露为 `net.forward_engine`。 |
| `PruningEngine(net)` | 引擎 | 持有网络并驱动拓扑重塑。自动构造为 `net.pruner`。 |

前馈与生成。

| 导出 | 形状 | 含义 |
|---|---|---|
| `engine.forward(byte_ids)` | 前馈 | 推理前馈，返回未来预测偏差，关闭 ACh，完全确定。 |
| `engine._predict(byte_ids, store_state, is_inference)` | 核心前馈 | 感知、Foldiak 去相关、去中心化与增量预测。 |
| `engine.generate(prompt, n_tokens, temperature, dev)` | 生成 | 让模型在提示字符串上前进并返回生成的字节。 |
| `engine.continuation(byte_ids, n_gen, temperature, rep_backstop)` | 自回归续写 | 训练 rollout 与验证共用。对输入前缀批量处理并返回追加的 token。 |

修剪。

| 导出 | 形状 | 含义 |
|---|---|---|
| `engine._prune()` | 拓扑重塑 | 执行一次：发育期守卫、死缓二级判决、相对排名淘汰。 |

-----

<a id="understand-the-implementation"></a>
## 理解实现

`ForwardEngine._predict` 是唯一的激活路径。它在 L0 对字节序列做 one-hot 编码，通过 `_recurrent`（带逐层归一化的纯 matmul）应用时间核 `W_t*`，逐层经过 L4、L2、L3、L5、L6，在 L5 应用 Foldiak 反希伯去相关，并把去中心化的 z5 送到读出端，而原始 z5 则喂给 `W_diff` 产生世界模型预测误差。在自由运行模式下外部输入被置零，活动完全由内部递归与三个加性振荡器驱动，可选的 STP 资源动力学对递归做门控。`_bind` 把 z4 经 `W_bind` 投影到带 top-k 胜者全取的稀疏概念槽，并通过 `W_act` 暴露由此产生的动作电位。

`continuation` 为训练与评估复用同一条 logits 管线。温度可以传张量从而在 GPU 上直除、零标量同步，还带可选的重复问题兜底（屏蔽周期性字节），并用预建掩码屏蔽非法续字节以强制 UTF-8 合法性。多步承诺遥测由一个可选开关控制。

`PruningEngine` 以严格顺序用一个置换重塑，使某层的行映射权重与其下游列保持对齐。每层的前馈权重、时间核、偏置、资格迹、死亡簿与 STP 资源都被置换并按存活宽度收缩，L4 专属辅助量（`W1`、`W_diff`、`W_state_pred`、记忆单元）也按同一置换 gather。淘汰先经死缓二级判决，再按行范数相对排名（L4 额外计入 W1 贡献），一直降到配置的下限尺寸。

-----

<a id="further-exploration"></a>
## 进一步探索

- [model](../model/README.zh.md)：这些引擎操作的权重与配置声明。
- [learning](learning/README.zh.md)：逐域学习更新。