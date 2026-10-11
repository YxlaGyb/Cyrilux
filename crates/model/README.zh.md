---
description: "model：稠密感知-预测-行动网络的配置与权重声明，加上共享神经调制与数值常量。"
kind: "package-reference"
---

# model

[English](README.md) | 中文

## 概述

`crates.model` 承担稠密感知-预测-行动（PPA）闭环的配置与权重声明面。它导出 `CyreneModel`，一个列举全部学习、修剪、代谢和振荡器超参数的 dataclass，以及 `HybridPCNet`，一个声明完整权重与 buffer 布局并把前馈、学习、修剪引擎委托出去的 `nn.Module`。该 crate 还承载定义层角色和信任域的数值常量。它不含训练循环，所有动力学都放在 `crates.hybrid`。

## 目录

- [使用本包](#use-this-package)
- [理解实现](#understand-the-implementation)
- [进一步探索](#further-exploration)

-----

<a id="use-this-package"></a>
## 使用本包

配置并构建网络。

| 导出 | 形状 | 含义 |
|---|---|---|
| `CyreneModel(...)` | 超参数 dataclass | 调用 `dims()` 获得层尺寸映射，调用 `param_count()` 获得预估参数量（前馈加生成与时间核）。 |
| `HybridPCNet(config=None)` | `nn.Module` | 声明全部参数与 buffer，完成权重初始化，并挂载 `forward_engine`、`learning_engine`、`pruner` 与 `mem_pool`。`config` 缺省为 `CyreneModel()`。 |
| `HybridPCNet.load(path, config=None)` | 检查点装载 | 从完整检查点重建网络。维度字段一律取自检查点，`config` 只补充非维度字段。对仅有权重的快照或未知键抛错。 |

读写检查点。

| 导出 | 形状 | 含义 |
|---|---|---|
| `net.save(path)` | 写 | 将参数、buffer、游离运行时张量与 CPU 计数器写为 safetensors，并在 metadata 中写入 `state_ver` 与 `config` 快照。 |
| `HybridPCNet.config_from_checkpoint(path)` | 读 | 在不加载张量的情况下读取 `config` 快照，适合 `list --detail` 之类的诊断。 |

运行模型。以下均为对稠密引擎的薄委托。

| 导出 | 形状 | 含义 |
|---|---|---|
| `net.forward(byte_ids)` | 前馈 | 推理前馈，返回未来预测偏差，确定性，关闭 ACh。 |
| `net.generate(prompt, n_tokens, temperature, dev)` | 生成 | 从提示字符串自回归生成，返回字节。 |
| `net.learn(byte_ids, closed_loop, free_run, force_perception)` | 学习 | 无反向传播、无教师目标的迹调制学习步。内部代谢决定该步走感知还是回声。 |
| `net.maybe_prune(step)` | 修剪 | 由代谢触发的重塑接缝。每 8 步轮询一次，返回触发原因或 None。判决委托给 `net.pruner.decide()`，物理折叠委托给 `net.pruner.compact()`。有些步只判决；当有死亡回合触发，或 `compact_interval` 已到且存在可释放工位时才做一次压实。 |
| `net.sync_gate()` | 路由 | 每步一次的路由交接，读取 P3-b 行为门并镜像到 CPU 标志。 |

-----

<a id="understand-the-implementation"></a>
## 理解实现

`CyreneModel` 声明全部超参数。`dims()` 与 `param_count()` 仅由配置直接推导声明的架构。

`HybridPCNet.__init__` 分成三部分职责。其一，权重声明：前馈矩阵（`W_04` 到 `W_56`）、世界模型预测器（`W_diff`、`W_state_pred`、`W_pred_54`、`W_pred_43`）、竞争性概念绑定层（`W_bind`、`W_bind_self`）、时间核（`W_t*`）、动作读出（`W_act`）与非线性混合头（`W1`、`W_lm`、`W_lm_2`）。其二，buffer 声明：活动 EMA、STP 资源、代谢账本与死亡契约计数器等慢变量注册为 buffer，并跨 `save` 与 `load` 持久化。其三，引擎委托：模型构造 `crates.hybrid` 的 `ForwardEngine` 与 `LearningEngine`，以及 `crates.neuron` 的 `PruningEngine` 与 `MemoryPool`，并把公开调用转发给它们。

`save` 写入 `state_ver` 锚点、`config` 快照，并把游离运行时张量放到 `attr.` 命名空间、CPU 计数器放到 `py.` 命名空间。`load` 拒绝缺少 `state_ver` 的快照；维度字段一律跟随检查点形状（修剪会物理切片矩阵）；未知持久化键一律抛错而不是静默丢弃。

原先分散在本 crate 与学习路径中的纯算子现已归入 `crates.kernels`：激活、归一化、神经调制与正则化。本 crate 只保留数值 `constants.py` 与权重及 buffer 布局，因此学习路径不再为数学而反指编排层。

`constants.py` 定义神经元状态列索引、六层皮层角色、每层配置（神经元数、惯性、k-WTA 比例）、连接类型，以及学习域消费的信任域与资格迹折扣。

-----

<a id="further-exploration"></a>
## 进一步探索

- [architecture](../../docs/architecture.md)：本 crate 的设计意图记录在这里。
- [hybrid](../hybrid/README.zh.md)：本模型委托的正向、生成与学习引擎。
- [neuron](../neuron/README.zh.md)：本模型委托的动态神经元引擎。
- [learning](../hybrid/learning/README.zh.md)：逐域学习子 crate。