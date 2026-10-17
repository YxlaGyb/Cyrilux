---
description: "把每一个逐步张量计算变成一张静态图的声明层。"
kind: "package-reference"
---

# graphs

[English](README.md) | 中文

## 概述

`crates.graphs` 是静态图的声明层。它持 `Op` 与 `Graph`：算子序列、名字接线、写回契约与变体键，四者合起来是一段计算的唯一事实源。它只依赖标准库，不持张量、不持网络引用，自己不做任何算术：算子是仓库里既有模块级函数的身份。两层：`spec` 负责声明与自检，`exec` 在名字环境上解释一份声明。同一份声明将驱动 CUDA Graph 捕获后端。

## 目录

- [使用本包](#use-this-package)
- [理解实现](#understand-the-implementation)
- [进一步探索](#further-exploration)

-----

<a id="use-this-package"></a>
## 使用本包

从包根导入声明名与引用解释器。

```python
from crates.kernels import rms_norm
from crates.graphs import Graph, Op, apply_writes, run_ref, validate_graph

graph = Graph(
    name="hidden",
    ops=(Op("norm", rms_norm, ("pre",), ("post",)),),
    inputs=("pre",),
    outputs=("post",),
    writes=(("post", "_post"),),
    variant=(("S", 8),),
)
validate_graph(graph)
run_ref(graph, env)
apply_writes(graph, env, net)
```

| 导出 | 模块 | 形状 | 含义 |
|---|---|---|---|
| `Op(name, fn, inputs, outputs)` | `spec` | 声明 | 一次具名调用：`fn` 的身份，加上它读写的环境名。 |
| `Graph(name, ops, inputs, outputs, writes, variant)` | `spec` | 声明 | 有序的算子序列、写回契约与变体键。 |
| `validate_graph(graph)` | `spec` | `None` 或抛错 | 结构自检：接线闭合、算子名唯一、只接受模块级函数。 |
| `defined_names(graph)` | `spec` | `frozenset[str]` | 图定义的全部名字：输入加各算子输出。 |
| `source_digest(fn)` | `spec` | 十六进制 | 算子源码文本的摘要，按函数缓存。 |
| `graph_fingerprint(graph, env)` | `spec` | 32 位十六进制 | 架构指纹；不含权重，环境可选但必须完整。 |
| `Graph.fingerprint(env)` | `spec` | 32 位十六进制 | 同一指纹，声明上的方法形式。 |
| `run_ref(graph, env)` | `exec` | `env` | 引用解释器：按声明顺序求值，结果就地写回。 |
| `apply_writes(graph, env, net)` | `exec` | `None` | 把写回契约落到宿主对象上。 |
| `NET` | `spec` | `str` | 宿主对象在环境中的保留名（`"net"`）。 |

-----

<a id="understand-the-implementation"></a>
## 理解实现

可复现性是这一形态的第一性收益。声明是数据：同样的算子、接线与变体键在任何机器上展开成同一张图，而随运行记录落盘的指纹说明是哪张图产生了这次运行。权重与步数是状态而非架构，因此永不进指纹。

图只持结构。张量经名字环境传入，算子是住在别的 crate 里的函数的身份，声明本身不得引入算术：没有表达成算子调用的计算不该出现在图里。`validate_graph` 只机械检查可检查的部分：每个被读的名字必须先被定义，输出名不得被定义两次，算子名唯一，算子可调用且不是 lambda 或局部函数，写回名必须已定义。`writes` 把环境名与宿主属性配对，是图接触宿主对象的唯一位置，图因此不含任何网络引用。

`run_ref` 按声明顺序求值，把每个结果写回环境并返回同一个环境，所以逐帧调用不产生新容器。多输出的算子必须返回对应元数，否则抛错。写回是刻意独立的一步：一段计算可以只求值与取指纹，不碰宿主。

指纹是对版本标签、图名、写回契约、输出名、变体键，以及逐算子的名字、源码摘要、接线、绑定后的各输入形状与 dtype 求得的 128 位 BLAKE2 摘要。绑定是全有或全无：`graph_fingerprint` 拒绝不完整的环境，因为部分绑定的图会让指纹取决于取样时机而不是架构。`source_digest` 按函数缓存，改一个声明体指纹即变，未动的声明保持稳定。

两条红线由覆盖普查机械检查，而不是靠约定：本包不 import `torch`、不持张量，任何图里都不出现非算子调用的算术。捕获后端在同一模块、同一份声明上落地：捕获录下的正是声明已经写明的东西，捕获期出现宿主同步一律抛错，不得静默回落。

-----

<a id="further-exploration"></a>
## 进一步探索

- [kernels](../kernels/README.md)：声明所调用的算子库，也是新增算术的唯一入口。
- [dense](../hybrid/README.md)：其前向路径的分段被声明为图。
- [learning](../hybrid/learning/README.md)：接受同样改造的学习路径。
