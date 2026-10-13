---
description: "attention 调制 CoreBrain 跨视图预测误差的误差精度注意力层。"
kind: "package-reference"
---

# attention

[English](README.md) | 中文

## 概述

`crates.attention` 承载位于 `crates.corebrain` 之上的误差精度注意力层。它读取短视图槽位、长视图槽位和当前跨视图预测误差，维护独立的慢速激活状态，并返回经过精度调制的误差。它是 dense PPA 的强制组成部分，不是可选开关。

## 目录

- [使用本包](#use-this-package)
- [理解实现](#understand-the-implementation)
- [进一步探索](#further-exploration)

-----

<a id="use-this-package"></a>
## 使用本包

该层由 `CoreBrain` 自动构造。

| 导出 | 形状 | 含义 |
|---|---|---|
| `AttentionConfig` | dataclass | `ema_tau`、`temperature`、`floor` 与共享的 `top_k`。 |
| `ErrorAttention(n_slots, cfg)` | 模块 | 持有独立慢状态并计算精度增益。 |
| `ErrorAttention.forward(a_short, a_long, delta)` | `[N, K]` | 返回包含有效误差与诊断张量的 `AttentionOutput`。 |

`AttentionOutput` 字段：

| 字段 | 含义 |
|---|---|
| `effective_error` | `gain * delta`，CoreBrain 的 Q 与 P 更新使用的误差。 |
| `gain` | 均值为 1、带正下界的逐槽位精度分布。 |
| `salience` | `abs(a_short - a_long_ema)`，自下而上显著性。 |
| `top_down` | `topk_wta(abs(a_long_ema))`，稀疏慢速目标。 |
| `a_long_ema` | 独立慢速激活状态。 |

-----

<a id="understand-the-implementation"></a>
## 理解实现

`a_long_ema` 在每次感知学习时用长视图激活的 batch 均值更新。它与 `P_long_ema` 独立：`P_long_ema` 是 CoreBrain 使用的慢速投影目标，`a_long_ema` 只服务注意力。

自下而上信号为 `abs(a_short - a_long_ema)`。自上而下信号为 `a_long_ema` 的 top-k 幅度支持。两路相加后，softmax 生成正值分布，再通过仿射映射

```text
g = (1 - floor) * K * softmax(g_raw / temperature) + floor
```

在不使用 clamp 的前提下保持均值为 1、下界为 `floor`。

CoreBrain 继续使用现有目标 `topk_wta(long @ P_long_ema)`。注意力不替换该目标，只计算 `delta_eff = g * delta`，并把有效误差同时送入预测器与投影器更新。投影更新先经 `Q.T` 映射，再应用现有 WTA 掩码。

该层只在感知学习阶段运行。自由运行与回声冻结阶段不更新 `a_long_ema`。

-----

<a id="further-exploration"></a>
## 进一步探索

- [corebrain](../corebrain/README.zh.md)：被调制的承载层。
- [model](../model/README.zh.md)：网络构造与生命周期。
