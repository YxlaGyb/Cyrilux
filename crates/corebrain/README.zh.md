---
description: "corebrain：密集 PPA 之上的平行协调层，两视图、槽位竞争、跨视图预测与窗级修正。"
kind: "package-reference"
---

# corebrain

[English](README.md) | 中文

## 概述

`crates.corebrain` 承载中心脑：密集 PPA 网络旁边的一层平行结构。它读原生的 `z4` 状态，池化出两个视图（窗口的后四分之一与整个窗口），把两者投影到同一组槽位空间，并在每个视图内部各跑一次 top-k 胜者全取。一个小预测器负责把短视图激活映射到由长视图构成的慢速目标上，这份预测误差是该层唯一的学习信号。前向输出是一个窗级修正量，在 L2 归一化之前叠加到 L4 状态上。该层默认关闭。

## 目录

- [使用本包](#use-this-package)
- [理解实现](#understand-the-implementation)
- [进一步探索](#further-exploration)

-----

<a id="use-this-package"></a>
## 使用本包

通过网络构造该层，使它保持为已注册的子模块。

| 导出 | 形状 | 含义 |
|---|---|---|
| `CoreBrainConfig` | dataclass | 参数级默认值：槽位数、top-k、学习率、EMA 速率、信任域、正则权重、离差下限、修正幅度。 |
| `CoreBrain(cfg)` | 模块 | 该层本身。持有两组视图投影、槽位预测器与慢速目标缓冲。 |
| `DensePCNet.enable_core_brain(**overrides)` | 模块 | 构造该层、挂到网络上并返回。开关未开时不调用，因此默认跑不增加参数、不增加 buffer。 |

调用该层。

| 导出 | 形状 | 含义 |
|---|---|---|
| `CoreBrain.forward(z4)` | `[N, d_in]` | 窗级修正量。调用方把它平铺到各序列位置。 |
| `CoreBrain.learn()` | dict | 对该层自身权重做一次更新，随后返回下方诊断读数。 |
| `CoreBrain.observe(z4=None)` | dict | 传 `z4` 时跑一次前向加一次更新并返回读数；不传时返回最近一次读数。 |

契约接受的输入。

| 输入 | 形状 | 含义 |
|---|---|---|
| `z4` | `[N, S, d_in]`，fp16 | 原生 L4 状态。该层读的是网络其余部分同样使用的那个张量，且从不写回。 |
| `n_slots` | int，默认 32 | 该层自有槽位空间的宽度，与概念绑定槽位无关。 |
| `top_k` | int，默认 4 | 每个视图竞争后保留的胜出槽位数。 |

`learn` 与 `observe` 返回的诊断读数。

| 键 | 含义 |
|---|---|
| `xv_energy` | 跨视图预测误差的均方值。 |
| `xv_energy_shuffled` | 把短视图激活沿批维滚动一位后重算的同一误差。它与 `xv_energy` 拉开差距，说明预测携带了与样本有关的内容。 |
| `effective_rank` | 长视图槽位激活的参与比。接近 1 表示激活坍缩到单个槽位。 |
| `slot_overlap` | 两个视图同时赢下的槽位比例。 |
| `delta_cos` | 当前与上一窗级修正量之间的余弦。 |
| `delta_absmax` | 修正量的最大绝对值，即正反馈读数。 |

<a id="understand-the-implementation"></a>
## 理解实现

该层位于前向链旁边，不进入链内。在 `ForwardEngine._predict` 里，L4 递归产出 `z4` 之后，该层读取这个原生张量，为每个批元素返回一个修正量，修正量在 `l2_norm` 送入 L2 之前叠加。`z4` 的其他消费者（世界模型差分、绑定层、记忆单元）仍然读原生张量，因此修正只触及一条边。

两个视图取自同一个窗口。`crates/corebrain/center.py::_views` 取序列位置的后四分之一作为短视图、取全部位置作为长视图，因此短视图不携带更早的历史，而长视图携带。两者都只在位置上池化，不引入新参数，也不需要新数据。

每个视图各有自己的投影，落到同一组槽位空间。`crates/corebrain/project.py` 持有 `P_short` 与 `P_long`，用局部 delta 规则学习，配信任域与软行范数界。`P_long_ema` 是 `P_long` 的慢速副本，以指数滑动平均跟随，且不直接接收误差，这使得目标不会像在线侧那样快速移动。

竞争是硬的，且发生在每个视图内部。`crates/corebrain/compete.py::topk_wta` 每行保留数值最大的 `top_k` 项并置零其余，同时返回胜出掩码。两个视图共用同一组槽位索引，因此一个槽位可以被任一视图赢下。

预测器由误差训练，投影经由胜出掩码训练。`crates/corebrain/predict.py` 持有 `Q`，一个槽位到槽位的方阵，用局部 delta 规则更新。由于胜者全取不可导，误差以直通规则到达投影：`crates/corebrain/project.py::learn` 把误差乘以胜出掩码，只有产生胜出槽位的那几列收到更新。

防坍缩不依赖梯度。`crates/corebrain/regularize.py` 对槽位激活而非权重施加两条直接规则：一条反希伯去相关，去掉槽位协方差的非对角项；一条逐槽增益，把离差低于下限的槽整列重标。增益被限制在 `[1, 2]` 区间，因此不需要 clamp。

修正是把预测出的长视图编码读回 L4 空间。`CoreBrain.forward` 先把短视图激活经 `Q` 映射，再经 `P_long` 的转置，最后缩放。该层从不解码回字节。

该层只从自身的误差学习。`LearningEngine.learn` 在逐域更新之后调用 `CoreBrain.learn`，且只在感知相调用；自由运行与冻结的回声相会跳过。字节损失从不进入该层，因此修正不受字节流塑形。

<a id="further-exploration"></a>
## 进一步探索

- [dense](../dense/README.md)：前向链，以及修正量叠加的位置。
- [model](../model/README.md)：网络声明，以及构造该层的开关。
