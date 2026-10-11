---
description: "cli：train、resume、配置、检视与数据准备的 click 命令行入口，面向 dense PPA 后端。"
kind: "package-reference"
---

# cli

[English](README.md) | 中文

## 概述

`crates.cli` 是项目的命令行面。它构建名为 `app` 的 click 命令组，注册六个子命令：`train`、`from-config`、`resume`，以及 `data`、`list`、`config` 三个子组。它选定 dense PPA 后端，接好数据文件与检查点，在 `out/` 下分配带版本号的输出目录，并把重活委托给 `crates.model` 与 `crates.utils`。路径一律相对项目根解析，字节窗在进入模型前被 utf-8 编码并零填充。本 crate 不含模型代码，也没有自己的训练循环。

## 目录

- [使用本包](#use-this-package)
- [理解实现](#understand-the-implementation)
- [进一步探索](#further-exploration)

-----

<a id="use-this-package"></a>
## 使用本包

运行训练路径。

| 导出 | 形状 | 含义 |
|---|---|---|
| `app train -d FILE...` | 训练 | 直接训练模式。逐行装载每个 JSONL 文件，批量组织字节窗，执行迹调制 `net.learn`。将 `final.safetensors` 保存到输出目录。 |
| `app train --resume` | 训练 | 找到 `out/v*` 下最新的 `final.safetensors` 并续训。 |
| `app from-config CFG.json` | 训练 | 从 JSON 配置加载参数并训练。`--batch-size`、`--lr`、`--epochs`、`--out-dir` 等选项覆写配置。 |
| `app resume CKPT.safetensors` | 训练 | 依据检查点自身的配置重建网络，并可选以 `-d` 提供新数据继续训练。保留检查点的最大序列长度。 |

检视工作区。

| 导出 | 形状 | 含义 |
|---|---|---|
| `app list checkpoints [DIR]` | 检视 | 列出 `.safetensors` 文件，含大小与类型。`--detail` 惰性读取元数据头部而不加载张量。 |
| `app list datasets [DIR]` | 检视 | 列出 `datasets/` 下的 JSONL 文件，含行数及尽力而为的格式分类。 |
| `app config init -o OUT -t TEMPLATE` | 配置 | 生成默认配置文件模板（`train` 或 `autonomous`）。 |
| `app config show CFG` / `app config validate CFG` | 配置 | 显示或校验 JSON 配置并打印警告。 |

准备数据。

| 导出 | 形状 | 含义 |
|---|---|---|
| `app data convert IN` | 数据 | 把 `conversations` 或 `chosen`/`rejected` 样本转换为扁平的 `{"text": ...}` 行。 |
| `app data split IN --chunk-size N` | 数据 | 将大型 JSONL 拆分为多个文件，默认会转换。`--dry-run` 仅统计不分割。 |
| `app data scan [DIR]` | 数据 | 汇总数据集目录的格式与估算行数。 |

-----

<a id="understand-the-implementation"></a>
## 理解实现

`__init__.py` 构建 `app` 命令组并注册全部子命令，先把 stdout 与 stderr 重配置为 utf-8。入口命令从 `crates.cli.utils` 导入共享辅助，而非自行推导项目布局。

`utils.py` 是共享中枢。`resolve_path` 把相对路径锚定到项目根。`run_dir` 惰性调用 `crates.utils.outver.ensure_run_dir` 分配当前进程的版本目录，`pin_run_dir` 把续跑过程锚定到既有训练线。`run_file`、`load_config`、`save_config`、`merge_config` 覆盖配置管线。它还携带 `TRAIN_CONFIG_TEMPLATE` 与 `AUTONOMOUS_CONFIG_TEMPLATE` 默认值。

`train_cmd.py` 拥有 dense 训练循环。`tiled_windows`、`window_ids` 与 `skip_zero_windows` 是纯函数：样本文本编码为 utf-8 后按 `stride = max_seq_len` **平铺**、不再截断，窗拼接即原文全字节，只丢弃会变成全零窗的样本。每文档尾窗左对齐并以 0x00 补齐，这些填充位不含学习信号——`learn` 对一切 <32 的目标字节加掩码。`_train_dense` 辅助函数先扫一次行数以驱动进度条，再把字节窗堆叠成批张量，每文件结束时落地尾批，并关闭梯度后调用 `net.learn`。若干废弃选项被接受并在提示中报告为忽略，而不是直接拒绝。

`train` 命令在缺失 `-d` 列表时快速失败。`resume` 先从检查点读取配置快照，因此维度字段一律跟随快照，并且仅当请求超过缓冲长度时才上调最大序列长度。

-----

<a id="further-exploration"></a>
## 进一步探索

- [utils](../utils/README.zh.md)：共享的路径与输出管理辅助。
- [model](../model/README.zh.md)：训练循环驱动的模型与引擎。