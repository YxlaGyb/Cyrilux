---
description: "dense、corebrain 与 attention 各 crate 共享的叶层算子库。"
kind: "package-reference"
---

# kernels

[English](README.md) | 中文

## 概述

`crates.kernels` 是叶层算子库。它承载学习与前向路径共享的每一个无梯度、与设备无关的函数：激活、归一化、稀疏竞争、神经调制、正则化、直通估计、塑性规则、谱守卫、声明式前向图绑定的逐步级联读写算子，以及把张量绑到静态形状的对齐视图。它只依赖 `torch`，不含状态、不持网络引用，因此 `crates.model`、`crates.hybrid`、`crates.corebrain` 与 `crates.attention` 都能依赖它而不产生环。

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
| `ach_noise(mu3, nov)` | `modulation` | 同输入 | 逐元素 ACh 抖动 `±0.03·(0.5 + nov)`，随机符号：图中唯一带随机性的算子。 |
| `compute_uncertainty(...)` | `modulation` | 标量 | 近期误差的不确定性估计。 |
| `combine_modulation(...)` | `modulation` | 标量 | 把各调制通道合成一个增益。 |
| `gate_psay(...)` | `modulation` | 标量 | P3-b 行为门：说与看。 |
| `foldiak_decorrelate(...)` | `regularize` | 就地 | 对槽位协方差做反希伯去相关。 |
| `variance_floor(...)` | `regularize` | 就地 | 把低于下限的离差整列重标的逐槽增益，限制在 `[1, 2]`。 |
| `participation_ratio(...)` | `regularize` | 标量 | 离差诊断量。 |
| `activity_baseline(post, ema, initialized)` | `plasticity` | `[out]` | 窗内活动²及相对慢 EMA 的超额部分。 |
| `energy_constraint(W, dW, post, ema, initialized, oja_alpha, oja_elasticity)` | `plasticity` | `[out, in]` | 真实 Oja 项加活动相关遗忘。 |
| `rho_control(dW, W_ref, adaptive)` | `plasticity` | `[out, in]` | 通道级塑性控制，返回缩放后的 `dW`、`rho` 与 `s`。 |
| `decorr_W(W, E, coef, max_delta_ratio, learn_boost, gate)` | `spectral` | 就地 | 当最大奇异值高于白噪声上界 `(√m+√n)/√(mn)·‖W‖_F` 的 `gate` 倍时，以范数信任域移除 top-1 主方向。 |
| `spectral_radius_guard(W, rescale, bound)` | `spectral` | 就地 | 在估计谱半径发散时约束递归。 |
| `align_rows(w, n)` | `align` | `[n, ...]` | 前 `n` 行，把权重绑到静态行数的视图。 |
| `align_square(m, n)` | `align` | `[n, n]` | 左上 `[n, n]` 块。 |
| `align_mem_window(m_seq, n, s, k, a)` | `align` | `[n, s, k, a]` | 记忆环读取的记忆窗口。 |
| `pack3(a, b, c)` | `align` | 元组 | 把三个张量打成一个节点输出，即 STP 资源三元组。 |

声明式前向图的逐步算子（`cascade`）：

| 导出 | 模块 | 形状 | 含义 |
|---|---|---|---|
| `zero_input(dev, n, s, d)` | `cascade` | `[n, s, d]` | 自由运行步的外部输入，恒零。 |
| `one_hot_input(byte_ids, alphabet)` | `cascade` | `[N, S, A]` | fp16 的字节 one-hot 输入通道。 |
| `history_join(padmax, z0, alphabet)` | `cascade` | `[N, S, A]` | 时序输入通道：上一帧输入，首帧用 pad 补。 |
| `join_last(x, y)` | `cascade` | 末维 | 拼接两条通道。 |
| `project(x, w, b)` | `cascade` | `[N, S, d]` | 线性投影 `x @ w.T + b`。 |
| `bias_3d(b)` | `cascade` | `[1, 1, d]` | 把偏置升维以按帧广播。 |
| `add_scaled(x, y, k)` | `cascade` | 同输入 | 加性调制 `x + k·y`。 |
| `add_unsqueezed(x, y)` | `cascade` | 同输入 | 按帧广播叠加，即中心脑修正量。 |
| `nonzero_mask(byte_ids)` | `cascade` | `[N, S]` 布尔 | 有效位置掩码，填充为 0。 |
| `osc_phase(tab, cnt, span, s)` | `cascade` | `[1, S, 1]` | 查表取相位，运行时无三角函数。 |
| `osc_advance(cnt, s, span)` | `cascade` | 就地 | 推进相位计数。 |
| `foldiak_gate(z, m)` | `cascade` | 同输入 | 侧抑制 `z - 0.2·(M @ zᵀ)ᵀ`。 |
| `center_last(x)` | `cascade` | 同输入 | 末维去中心化。 |
| `drop_last(x)` | `cascade` | `[N, S-1, ...]` | 去掉末帧，与差分对齐。 |
| `seq_diff(x)` | `cascade` | `[N, S-1, ...]` | 逐帧差分。 |
| `unit_norm(x)` | `cascade` | 同输入 | 带零保护的单位化。 |
| `prepend_pad(x, padmax, d)` | `cascade` | `[N, S+1, ...]` | 首帧前置 pad。 |
| `pred_err(dz, mu)` | `cascade` | 标量 | 差分预测残差，即自由能。 |
| `mem_reset(m_seq, mem_m, n)` | `cascade` | 就地 | 把当前记忆本体写入首帧。 |
| `mem_advance(m_seq, z4, t, mem_a)` | `cascade` | 就地 | 第 `t` 帧的泄漏积分推进。 |
| `mem_read(m_seq, z4, k)` | `cascade` | `[N, S, K·a]` | 内容寻址读出：余弦相似度 softmax 门控，和为 `K`。 |
| `mem_last(m_seq)` | `cascade` | `[K, a]` | 末帧记忆按批取均值。 |
| `nov_reset(z4)` | `cascade` | `[N, S, ...]` | 慢通道以快通道首帧起始。 |
| `nov_advance(zslow, z4, t)` | `cascade` | 就地 | 慢通道推进，`0.99` / `0.01` 泄漏。 |
| `nov_read(z4, zslow)` | `cascade` | `[N, S]` | 快慢散度作为逐帧新奇度。 |

-----

<a id="understand-the-implementation"></a>
## 理解实现

本 crate 是那些原先散落在 `crates.model`、`crates.hybrid`、`crates.corebrain` 与 `crates.attention` 中的算子的唯一归属。把它们移到这里，打断了从学习路径反向指回 `crates.model` 的依赖，因为纯数学从不需要编排层。

每个函数都是纯函数：读取参数并返回结果，就地写入器（`soft_norm_preserve`、`foldiak_decorrelate`、`variance_floor`、`decorr_W`、`spectral_radius_guard`、`osc_advance`、`mem_reset`、`mem_advance`、`nov_advance`）例外，它们直接改写传入张量。没有函数持有网络引用。当算子需要网络状态（慢 EMA 缓冲、冷启动标志、`cfg` 系数）时，`crates.hybrid.learning._common` 的调用侧读取该状态并传入，使核保持可独立测试。

声明式前向图新增两个模块。`cascade` 为激活路径的逐步算子命名，使每一步都是一个声明节点，而不是引擎里的一行内联代码；它不持状态、不做控制流决策。`align` 承载四个纯视图与打包算子，把静态行数或帧数高于动态值的张量绑上，从而在不 clamp、不复制的前提下给出形状。`ach_noise` 是唯一带随机性的算子：它从生成器取 ACh 抖动，使唯一随机源落在一个具名节点上。

稀疏与调制路径保持 fp16 契约。`rms_norm` 与 `l2_norm` 用掩码式分母而非 clamp 保护零向量情形，`saturate` 与 `cubic` 以结构化方式约束其尾部，因此没有算子依赖破坏梯度的 clamp。`spectral` 内的两个模块级缓存（`_PROBE_CACHE` 与 `_EYE_CACHE`）按 `(device, dtype, size)` 复用探测向量与单位掩码，使热路径不产生新分配。

-----

<a id="further-exploration"></a>
## 进一步探索

- [model](../model/README.md)：配置及核所消费的数值常量。
- [dense](../../crates/hybrid/README.md)：调用这些算子的前向与学习路径。
- [learning](../hybrid/learning/README.zh.md)：委托到本 crate 的状态适配层。
