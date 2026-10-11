---
description: "learning：稠密 PPA 循环的迹调制学习引擎，组织为编排器加七个逐域 mixin：action、bind、feedforward、metabolism、predict、readout、temporal。"
kind: "package-reference"
---

# learning

[English](README.md) | 中文

## 概述

`crates.hybrid.learning` 实现稠密 PPA 循环的迹调制学习：一次前馈、逐域误差、然后进行无反向传播、无教师目标的局部权重更新。`LearningEngine` 由 `EngineCore` 组合七个 mixin 构成。`engine.py` 持有 `learn()` 编排与共享的逐步上下文；`_common.py` 持有各域共用的逐网络状态适配层，把稳态算子委托给 `crates.kernels`（能量约束、去相关、谱守卫），而资格迹与其调制由 `crates.rtr` 拥有；`action.py`、`bind.py`、`feedforward.py`、`metabolism.py`、`predict.py`、`readout.py` 与 `temporal.py` 各自拥有一个更新域。所有状态都流经引擎所持有的网络。

## 目录

- [使用本包](#use-this-package)
- [理解实现](#understand-the-implementation)
- [进一步探索](#further-exploration)

-----

<a id="use-this-package"></a>
## 使用本包

| 导出 | 形状 | 含义 |
|---|---|---|
| `LearningEngine(net)` | 引擎 | 持有网络并运行一个学习步。自动构造为 `net.learning_engine`。 |
| `net.learn(byte_ids, closed_loop, free_run, force_perception)` | 学习步 | 单个迹调制步。经由 P3-b 行为门在内部路由到感知或回声；不接收目标。 |
| `LearningEngine.learn(...)` | 学习步 | 同一调用，由引擎直接暴露。 |

`learn()` 步接受以下由标志选择的输入。

| 导出 | 形状 | 含义 |
|---|---|---|
| `byte_ids` | `[N, S]` long 张量 | 目标字节。当为 None（且非自由运行）时，该步为以上一窗口为种子的自回声。 |
| `closed_loop` | 标志 | 自回归暴露训练。输入为前 `k` 个真实锚定字节加模型自身自生成的续写；目标始终是真实字节。 |
| `free_run` | 标志 | 输入置零，只更新内部动力学权重，字节域权重冻结。 |
| `force_perception` | 标志 | P3-b 旁路，即便行为门选择回声也强制感知（首步与恢复期使用）。 |

-----

<a id="understand-the-implementation"></a>
## 理解实现

`engine.py` 以固定调用序编排各域。它先运行 `_predict` 收集层状态，自底向上计算逐层预测误差，然后通过 P3-b 行为门与 C5 冻结合约把该步路由到感知、回声、闭环或自由运行。回声相冻结世界模型。随后构建差分窗与 LM 信号，在动作读出前更新代谢，应用前馈与时序域，更新读出头，最后为下一步结算行为门。它暴露遥测（`free_energy`、`future_err`、`d_polarity`）并 detach 其存储的状态。

`_common.py` 是共享状态适配层。它读取网络的 EMA 缓冲、冷启动标志与 `cfg` 系数，再调用 `crates.kernels` 中的稳态算子：`energy_constraint`、`activity_baseline`、`rho_control`、`decorr_W` 与 `spectral_radius_guard`。`energy_constraint` 应用真实 Oja 项加活动相关遗忘，`decorr_W` 以范数信任域移除 top-1 主方向（且仅当该方向高于白噪声上界时），`spectral_radius_guard` 在递归发散时加以约束。`activity_baseline` 相对慢 EMA 追踪窗口活动。每一次权重更新都改为经由 `crates.rtr`：`trace` 推进逐权重资格迹 `E`，`rtr_apply` 用生存信号携带的逐状态残差 `m` 对其缩放。

七个域分别是：`feedforward`（偏置稳态与带 explain-away 加权和稳态增益的 `W_04`/`W_42`/`W_56`/`W_23`/`W_state_pred` 更新）、`temporal`（针对 `W_diff` 的多尺度软时间窗加带谱守卫的 `W_t*` 时间核）、`readout`（LM 头 `W_lm`/`W_lm_2`/`W1`/`bias_lm`，加熵驱动的牵引与可选重建臂）、`predict`（带未来折扣目标与 RPE 门控的自组织预测引擎 `W_pred_54`/`W_pred_43`）、`bind`（Foldiak `M_l5` 去相关与带内源振荡器的 `W_bind`/`W_bind_self` 概念槽）、`action`（`W_act` 闭环自洽生成学习）与 `metabolism`（自由能账本、生存信号、新奇度账本与 P3-b 行为门）。

-----

<a id="further-exploration"></a>
## 进一步探索

- [model](../model/README.zh.md)：这些域更新的网络与配置声明。
- [rtr](../rtr/README.zh.md)：这些域调用的迹、残差与生存信号写点。
- [dense](../README.zh.md)：前馈与修剪引擎。