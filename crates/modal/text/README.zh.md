---
description: "UTF-8 字节容器的几何与拓扑预设，只作纯数据声明"
kind: "package-reference"
---

# text

[English](README.md) | 中文

## 概述

`crates.modal.text` 是文本模态的器官。它声明字节流所住的容器：符号域、填充位、可打印域，以及 UTF-8 自动机（哪些字节开启字符、哪些字节延续字符、一个首字节要求几个续字节），外加网络由这些事实导出的两张运行期掩码。它只声明形状。这里没有词表与合并规则，没有逐符号向量，没有语料频率，没有语言或字符集标签，没有评分器输出，没有任何可学习参数，也没有任何张量。

## 目录

- [使用本包](#use-this-package)
- [理解实现](#understand-the-implementation)
- [进一步探索](#further-exploration)

-----

<a id="use-this-package"></a>
## 使用本包

容器本体。

| 导出 | 形状 | 含义 |
|---|---|---|
| `ALPHABET` | `int` | 符号域基数，一个字节值一个符号。 |
| `PAD_ID` | `int` | 写入定长窗口尾部的填充符号。 |
| `PRINTABLE_START` | `int` | 可打印域的最低符号；低于它的字节是控制字节。 |

UTF-8 自动机。

| 导出 | 形状 | 含义 |
|---|---|---|
| `CONTINUATION` | `(int, int)` | 续字节的闭区间。 |
| `OVERLONG_LEADS` | `(int, int)` | overlong 首字节的闭区间，作为字符起始非法。 |
| `LEAD_RANGES` | `tuple[(int, int)]` | 合法多字节首字节的三个区间：2、3、4 字节序列。 |
| `LEAD_CONTINUATIONS` | `tuple[int]` | 与 `LEAD_RANGES` 平行的续字节数。 |
| `START_RANGES` | `tuple[(int, int)]` | 全部合法起始：ASCII 加三个首字节区间。 |
| `CONTINUATION_PREFIX` | `(int, int)` | 位前缀判据 `(b & first) != second`，张量侧唯一无需 gather 就能用的判据。 |
| `is_continuation(b)` | `bool` | 续字节区间上的 Python 判据。 |
| `is_start(b)` | `bool` | `START_RANGES` 上的 Python 判据。 |
| `lead_length(b)` | `int` | 首字节要求的续字节数；ASCII 与非法首字节为 0。 |

派生表，每个符号一项。

| 导出 | 形状 | 含义 |
|---|---|---|
| `UTF8_BLOCK_CNT` | `tuple[bool]` | 该位置不能延续字符：首字节或 ASCII 字节。 |
| `UTF8_BLOCK_START` | `tuple[bool]` | 该位置不能开启字符：续字节或 overlong 首字节。 |
| `PRINT_MASK` | `tuple[bool]` | 可打印域，作为网络在构造期物化的掩码。 |

```python
from crates.modal.text.geometry import ALPHABET, LEAD_RANGES, CONTINUATION_PREFIX
```

-----

<a id="understand-the-implementation"></a>
## 理解实现

`geometry.py` 就是整个包，除 `__future__` 外什么都不 import。它只持有纯 `int`、`tuple`、`bool`，因此三类读者共享同一份事实：网络由它物化运行期缓冲，工具一边读填充位一边做自己的 torch 工作，纯 Python 把它当数据读。

器官只声明形状，从不构造张量。这条界不是"仓库里禁 torch"（工具 `crates.utils.codec` 就 import 它）：一个持有张量的预设会连带持有 dtype、device、生命周期与 `state_dict` 键位，而那些是 `crates.model` 的构造策略，也是 `HybridPCNet.load` 据此快速失败的 schema。一份事实一个所有者，所以掩码在构造缓冲的地方构造。它也划出"声明"与"处理"的界：几何若真需要计算，未来模态也只用 `math`；一旦需要 `torch`，它就在变换信号，那已经是工具。

两张生成掩码是前向路径真正消费的字节域事实：`UTF8_BLOCK_CNT` 与 `UTF8_BLOCK_START` 变成 `crates.model` 为 UTF-8 边界检查注册的两个 bool 缓冲，`PRINT_MASK` 变成 logits 携带的 fp16 符号掩码，可打印域之外的字节因此永远不会被选中。网络读这些预设，方向不会倒过来。

两把报告尺刻意留在本包之外：`scripts/world_lang.py` 与 `scripts/audit_char_coverage.py` 自带字节分类，因为一把量模型语言输出的尺，不能拿模型自己的域当尺。

-----

<a id="further-exploration"></a>
## 进一步探索

- [modal](../README.zh.md)：拥有"身体与工具"这条界的包组。
- [model](../../model/README.zh.md)：把这些预设物化为运行期掩码的网络。
- [utils](../../utils/README.zh.md)：读取这些预设的编解码与语料工具。
