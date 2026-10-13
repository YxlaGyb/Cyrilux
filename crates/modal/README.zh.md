---
description: "模态包组，只放每种信号的几何/拓扑预设，处理语料的工具在其外。"
kind: "package-group"
---

# modal

[English](README.md) | 中文

## 概述

`crates.modal` 是本项目各模态的包组。每个子包是一种信号的器官：声明该信号容器的几何与拓扑，即符号域、填充位、以及哪些字节位可以开启或延续一个字符。模态是叶层：不 import 任何网络代码，因此新增一个模态只多一个子包，不动网络拓扑。凡是处理语料的东西都不在本组：读样本、切窗口、转换与切分数据集都是工具，在 `crates.utils`，由 `crates.cli` 编排。

## 目录

- [使用本包](#use-this-package)
- [理解实现](#understand-the-implementation)
- [进一步探索](#further-exploration)

-----

<a id="use-this-package"></a>
## 使用本包

| 子包 | 信号 | 器官内容 |
|---|---|---|
| [text](text/README.zh.md) | UTF-8 字节流 | 字节容器几何：符号域、填充位、可打印域、UTF-8 字节分类，以及由它导出的生成掩码。 |

```python
from crates.modal.text.geometry import ALPHABET, PRINTABLE_START
```

把语料搬进这个容器的字节编解码与窗口切分是处理工具而非器官：它们在 [utils](../utils/README.zh.md)，并服从这里的预设。

-----

<a id="understand-the-implementation"></a>
## 理解实现

本组划的界是"信号长什么形状"与"语料被怎么处理"之间的界。几何与拓扑预设描述信号所住的容器，属于模态。读样本、切窗口、转换语料与切分语料都在改动数据，属于 `crates.utils`，由 `crates.cli` 编排。

依赖方向只有一条，且刻意允许一条边：网络 import 模态的预设（`crates.model` 由 `crates.modal.text.geometry` 构造运行期掩码，`crates.hybrid` 的正向与学习路径直接读它的字节域常数），工具也 import 它，因此 `crates.utils.codec` 的填充位取自器官而不是自己重述。模态既不 import 网络也不 import 工具，因此是与 `crates.kernels` 并列的叶层。

本组内部刻意不做共享。若一个预设同时被文本与未来的模态读取，就会把一种信号的形状强加给另一种，因此每个子包各自声明；新增模态就是加一个同级的兄弟子包，仓库里别处不动。

-----

<a id="further-exploration"></a>
## 进一步探索

- [text](text/README.zh.md)：文本模态的字节域预设。
- [model](../model/README.zh.md)：由这些预设构造运行期掩码的网络。
- [utils](../utils/README.zh.md)：刻意留居本组之外的编解码与语料工具。