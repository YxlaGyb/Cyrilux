---
description: "rtr：迹调制的资格可塑性，每个稠密权重更新如今都经由它完成信用分配。"
kind: "package-reference"
---

# rtr

[English](README.md) | 中文

## 概述

`crates.rtr` 承载迹调制的资格规则，它取代了每个稠密权重上的纯共现外积。每个更新点先由瞬时共活动标签累积出一条资格迹 `E`，再用逐状态残差 `m = r - V[state]` 调制后写入权重，其中 `r` 是体内自由能货币，`V` 是分桶的状态价值表。本包包含核心算子、迹缓冲的适配层，以及唯一的生存信号写点。它只依赖 `torch` 与 `crates.model` 的常量。

## 目录

- [使用本包](#use-this-package)
- [理解实现](#understand-the-implementation)
- [进一步探索](#further-exploration)
- [已知限制与后续工作](#known-limitations-and-deferred-work)

-----

<a id="use-this-package"></a>
## 使用本包

从包根导入算子。

```python
from crates.rtr import rtr_apply, trace, update_survival_signal
```

| 导出 | 模块 | 形状 | 含义 |
|---|---|---|---|
| `elig_accum(E, dW_raw, gamma)` | `kernel` | `[out, in]` | 资格迹 `E = gamma * E + dW_raw`，就地写回迹缓冲。 |
| `rtr_residual(v_row, index, r, eta_v)` | `kernel` | `[1]` | 三阶残差 `m = r - V[index]`，并就地追踪 `V[index] += eta_v * m`。 |
| `rtr_apply(E, m)` | `kernel` | `[out, in]` | 迹调制更新 `m * E / (norm(E) + 1e-6)`。 |
| `trace(net, wname, dW_raw)` | `trace` | `[out, in]` | 适配器：取 `net.{wname}_elig` 并用 `ELIG_GAMMA` 累积。 |
| `update_survival_signal(net, ctx)` | `signals` | `None` | 每步由 `net._metab_R` 与 `net._rtr_v` 写一次 `net._survival_signal`。 |

每个更新点都是同样三行：构造瞬时共活动标签，折进迹，再把调制后的迹写进权重。

```python
dW_raw = (pre.transpose(-2, -1) @ rms_norm(post)).mean(dim=0)
E = trace(net, "W_42", dW_raw)
dW = rtr_apply(E, net._survival_signal)
net.W_42[:dim].data += dW * eta
```

-----

<a id="understand-the-implementation"></a>
## 理解实现

`elig_accum` 把迹保持为瞬时标签 `dW_raw` 的指数遗忘累积。标签本身只是突触前后活动的裸外积：它不携带任何信用，也永不直接进入权重。跨步存活的、随后被残差调制的是迹。

`rtr_residual` 承担信用分配。`r` 是生存货币（`net._metab_R`，自由能变化的 tanh 压缩值），`V` 是分桶的逐状态价值表。桶号由 L4 激活的均值池化经固定、带种子的 `+/-1` 矩阵投影后取 `argmax` 得到。残差 `m = r - V[state]` 是三阶统计量：它度量当前状态比其滑动预期更好还是更差。`V` 是注册在模型上的缓冲，随 L4 其余状态一起参与逐行修剪。

`rtr_apply` 逐点归一化迹并按残差缩放。归一化放在这里而不是信号写点，因此同一个标量残差可以驱动每个点各自尺度的迹，且不会出现除零。

`update_survival_signal` 是 `net._survival_signal` 的唯一写点，由 `_update_metabolism` 在 `net._metab_R` 结算后无条件调用。由于它位于该步的代谢项内，感知步同样产出信号，而不只是闭环与回声步。两个行桶把感知货币与回声货币分开，因为两者的自由能变化尺度相差约一个数量级。

所有运算都保持在 fp16。桶号是常驻设备的长整型张量，因此 gather 与 `index_add_` 全程不离开 GPU，也不会触发同步。

-----

<a id="further-exploration"></a>
## 进一步探索

- [kernels](../kernels/README.md)：叶层算子，含仍在迹旁运行的稳态规则。
- [learning](../hybrid/learning/README.md)：逐领域调用 `trace` 与 `rtr_apply` 的更新点。
- [model](../model/README.md)：`_rtr_v` 与 `_rtr_proj` 缓冲以及数值常量。

-----

<a id="known-limitations-and-deferred-work"></a>
## 已知限制与后续工作

价值表较粗：两行共 64 桶，且桶号取自 `argmax`。相邻状态可能共用同一桶，且投影是固定的而非学习得到的。更细的索引，或以软混合替代 `argmax`，要等粗表被证明过于粗糙时再考虑。
