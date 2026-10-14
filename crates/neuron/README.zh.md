---
description: "动态神经元机制，主链拓扑修剪与记忆单元群的出生与死亡。"
kind: "package-reference"
---

# neuron

[English](README.md) | 中文

## 概述

`crates.neuron` 持有稠密 PPA 的动态神经元机制：主链的拓扑修剪，以及竞争性记忆单元群的出生与死亡。两者都在原地重塑活动网络并重新注册受影响的张量，使每个行映射权重始终与其下游列保持对齐。本 crate 是网络的强制组成部分，不是可选开关。

## 目录

- [使用本包](#use-this-package)
- [理解实现](#understand-the-implementation)
- [进一步探索](#further-exploration)

-----

<a id="use-this-package"></a>
## 使用本包

两个引擎都接受一个网络并持有其引用，由 `HybridPCNet.__init__` 构造。

| 导出 | 形状 | 含义 |
|---|---|---|
| `PruningEngine(net)` | 引擎 | 持有网络并驱动拓扑重塑。自动构造为 `net.pruner`。 |
| `MemoryPool(net)` | 引擎 | 持有网络并驱动记忆单元生命周期。自动构造为 `net.mem_pool`。 |
| `ROW_STATE` | dict | 行索引状态登记表：每项为名称、层、形式与处理者。 |

重塑。判决侧只写死亡契约；压实侧才是唯一改变形状的一步。

| 导出 | 形状 | 含义 |
|---|---|---|
| `net.maybe_prune(step)` | 接缝 | 由代谢触发的重塑入口，归 `crates.model` 所有。每 8 步轮询一次，返回触发原因或 None，并委托给下面两个方法。 |
| `pruner.decide()` | 判决 | 只写死亡标志与死缓簿：登记死行，对行范数已回升过阈值的行撤销标记，推进并裁定死缓期满，再按相对排名标出最弱的行。形状与 `active_size` 不动。 |
| `pruner.compact()` | 压实 | 唯一的物理步骤：把每层的行映射权重与其下游列一起置换，gather L4 辅助量，收缩到存活宽度，并回收内存。 |
| `pruner.has_releasable()` | 守卫 | 某层关闭工位达到该层宽度的 1% 以上时为真，是压实回合的前置条件。 |

记忆池。

| 导出 | 形状 | 含义 |
|---|---|---|
| `mem_pool.update(ctx, sh)` | 每步 | 先结算生存收入，再处理记忆单元的死亡与出生。在 LM 头更新之后调用。 |
| `mem_pool.famine_kill()` | 死亡回合 | 移除增益最低的单元（索引 0 免疫，K 保底 1）。内容与身份永久丢失。 |
| `mem_pool.birth(parent)` | 出生 | 从父单元二分出子单元，并尾插一个零 `W1` 块。 |
| `mem_pool.resize(keep)` | 重索引 | 为新的 K 重新注册单元缓冲以及 `W1`/`W1_elig` 行。 |

-----

<a id="understand-the-implementation"></a>
## 理解实现

`PruningEngine` 把一个重塑回合拆成两段。`decide()` 在代谢回合上轮询，只写死亡契约：登记死行，对行范数已回升过 `death_threshold` 的行清除标记与死缓计数，其余推进死缓，对熬过 `death_probation` 的行裁定死亡，最后按相对排名标出最弱候选。`active_size` 与一切张量形状都不动，因此活体行不会因形状折叠而丢掉工位。`compact()` 是唯一的物理步骤：它把每层的前馈权重、时间核、偏置、资格迹、死亡簿与 STP 资源连同下游列用同一置换重塑，把 L4 辅助量（`W1`、`W_diff`、`W_state_pred`、记忆单元）与 CoreBrain 投影器状态 gather 到同一置换上，收缩到存活宽度，并回收被释放的内存。淘汰先经死缓，再按行范数相对排名（L4 额外计入 W1 贡献），一直降到配置的下限尺寸。

记忆池把 K 个单元保持在一个固定布局上：它们的状态存在 `_mem_m`、`_mem_a`、`_mem_g`、`_mem_q` 中，解码器行则位于 `W1` 里自 `dim_4 + bind_slot_dim` 起的块内。死亡与出生都会重新注册该块，因此不变量 `net._lm_in == W1.shape[0]` 在每次事件后都成立：这是让 `zh` 列与 `W1` 行永不错位的唯一汇合点。

成员资格由生存账本定价，而不是由字节损失定价。单元的收入是生存残差的模长与"单元记忆同账本状态方向 `ẑ` 之间的余弦"之积，以慢速 `_mem_q` EMA 平滑。增益有界，完全没有收入的单元会把 `g` 推向零。当 `g` 低于 `mem_g_min` 持续 `mem_death_steps` 步时触发死亡；当误差快 EMA 超过慢 EMA 的 `mem_birth_thresh` 倍且冷却期满时触发出生。索引 0 免疫死亡，`mem_k_max` 限制增长。

饥荒击杀是代谢死亡回合的直接损失：它立即移除增益最低的单元，该内容与身份永久消失。之后的出生从存活父单元克隆出新单元，因此 K 可以回弹；不可逆性体现在只减不增的 `active_size` 上。

-----

<a id="further-exploration"></a>
## 进一步探索

- [hybrid](../hybrid/README.zh.md)：调用本 crate 的引擎及其驱动的网络。
- [model](../model/README.zh.md)：网络构造、配置与生命周期。
- [corebrain](../corebrain/README.zh.md)：其投影器状态与 L4 一同被置换的槽位层。
