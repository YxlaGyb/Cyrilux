---
description: "kernels：dense、corebrain 与 attention 各 crate 共享的叶层算子库。"
kind: "package-reference"
---

# kernels

[English](README.md) | 中文

## 概述

`crates.kernels` 是叶层算子库。它承载学习与前向路径共享的每一个无梯度、与设备无关的函数：激活、归一化、稀疏竞争、神经调制、正则化、直通估计、塑性规则与谱守卫。它只依赖 `torch`，不含状态、不持网络引用，因此 `crates.model`、`crates.dense`、`crates.corebrain` 与 `crates.attention` 都能依赖它而不产生环。

## 目录

- [使用本包](#use-this-package)
- [理解实现](#understand-the-implementation)
- [进一步探索](#further-exploration)

-----

<a id="use-this-package"></a>
## 使用本包

从包根导入算子。

```python
from crates.kernels import rms_norm, saturate, topk_wta
```

| 导出 | 模块 | 形状 | 含义 |
|---|---|---|---|
| `saturate(x)` | `activation` | 同输入 | `x / (1 + abs(x))`，共享的分流抑制。 |
| `cubic(x)` | `activation` | 同输入 | `x * (1 - 0.5 * x**2)`，零反向传播的多项式激活。 |
| `rms_norm(x)` | `normalize` | 同输入 | 带掩码分母、保护零向量情形的 RMS 归一化。 |
| `l2_norm(x)` | `normalize` | 同输入 | 带掩码分母、保护零向量情形的 L2 归一化。 |
| `soft_norm_preserve(W)` | `normalize` | 就地 | 结构化、非 clamp 的行范数保持。 |
| `rel_norm(raw, ema, eps)` | `normalize` | 标量 | 自归一化比值 `raw / (ema + raw + eps)`，取值 `[0, 1)`。 |
| `topk_wta(x, k)` | `sparse` | `[N, K]` | 每行保留最大的 `k` 项，返回激活与胜出掩码。 |
| `ste_delta(view, e, win)` | `estimator` | `[N, K]` | 经胜者掩码到达投影的直通误差。 |
| `compute_free_energy(...)` | `modulation` | 标量 | 作为生存货币的绝对误差能量。 |
| `compute_dopamine(...)` | `modulation` | 标量 | 带符号自由能变化的 sigmoid。 |
| `compute_ach(...)` | `modulation` | 标量 | 不确定性响应。 |
| `compute_uncertainty(...)` | `modulation` | 标量 | 近期误差的不确定性估计。 |
| `combine_modulation(...)` | `modulation` | 标量 | 把各调制通道合成一个增益。 |
| `gate_psay(...)` | `modulation` | 标量 | P3-b 行为门：说与看。 |
| `foldiak_decorrelate(...)` | `regularize` | 就地 | 对槽位协方差做反希伯去相关。 |
| `variance_floor(...)` | `regularize` | 就地 | 把低于下限的离差整列重标的逐槽增益，限制在 `[1, 2]`。 |
| `participation_ratio(...)` | `regularize` | 标量 | 离差诊断量。 |
| `activity_baseline(post, ema, initialized)` | `plasticity` | `[out]` | 窗内活动²及相对慢 EMA 的超额部分。 |
| `energy_constraint(W, dW, post, ema, initialized, oja_alpha, oja_elasticity)` | `plasticity` | `[out, in]` | 真实 Oja 项加活动相关遗忘。 |
| `elig_accum(E, dW_raw, gamma)` | `plasticity` | `[out, in]` | 资格迹 `E = gamma * E + dW_raw`。 |
| `rho_control(dW, W_ref, adaptive)` | `plasticity` | `[out, in]` | 通道级塑性控制，返回缩放后的 `dW`、`rho` 与 `s`。 |
| `decorr_W(W, E, coef, max_delta_ratio, learn_boost)` | `spectral` | 就地 | 以范数信任域移除 top-1 主方向。 |
| `spectral_radius_guard(W, rescale, bound)` | `spectral` | 就地 | 在估计谱半径发散时约束递归。 |

-----

<a id="understand-the-implementation"></a>
## 理解实现

本 crate 是那些原先散落在 `crates.model`、`crates.dense`、`crates.corebrain` 与 `crates.attention` 中的算子的唯一归属。把它们移到这里，打断了从学习路径反向指回 `crates.model` 的依赖，因为纯数学从不需要编排层。

每个函数都是纯函数：读取参数并返回结果，就地正则化器（`soft_norm_preserve`、`foldiak_decorrelate`、`variance_floor`、`decorr_W`、`spectral_radius_guard`）例外，它们直接改写传入张量。没有函数持有网络引用。当算子需要网络状态（慢 EMA 缓冲、冷启动标志、`cfg` 系数）时，`crates.dense.learning._common` 的调用侧读取该状态并传入，使核保持可独立测试。

稀疏与调制路径保持 fp16 契约。`rms_norm` 与 `l2_norm` 用掩码式分母而非 clamp 保护零向量情形，`saturate` 与 `cubic` 以结构化方式约束其尾部，因此没有算子依赖破坏梯度的 clamp。`spectral` 内的两个模块级缓存（`_PROBE_CACHE` 与 `_EYE_CACHE`）按 `(device, dtype, size)` 复用探测向量与单位掩码，使热路径不产生新分配。

-----

<a id="further-exploration"></a>
## 进一步探索

- [model](../model/README.md)：配置及核所消费的数值常量。
- [dense](../../crates/dense/README.md)：调用这些算子的前向与学习路径。
- [learning](../dense/learning/README.md)：委托到本 crate 的状态适配层。
