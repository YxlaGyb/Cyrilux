---
description: "utils：数据转换、拆分、任务数据准备、训练工具与运行目录版本的共享、与模型无关辅助。"
kind: "package-reference"
---

# utils

[English](README.md) | 中文

## 概述

`crates.utils` 是复用性、模型无关辅助的汇总 crate。它拥有数据转换（`data_converter`）、大文件拆分（`data_splitter`）、面向任务的数集准备（`prepare_tasks`）、训练相关工具如日志、学习率调度、种子设置与参数统计（`trainer_utils`），以及 cli 依赖的 `out/` 版本目录方案（`outver`）。它不含模型权重，也没有训练循环。其模块被 `crates.cli` 与独立脚本导入。

## 目录

- [使用本包](#use-this-package)
- [理解实现](#understand-the-implementation)
- [进一步探索](#further-exploration)

-----

<a id="use-this-package"></a>
## 使用本包

转换与拆分数据。

| 导出 | 形状 | 含义 |
|---|---|---|
| `convert_sample(line)` | JSONL 行 → `{"text": ...}` | 把单行原始 JSONL 转换为标准文本格式，以 JSON 行返回，为 `None` 则跳过。对已是 `text` 格式的样本原样放行。 |
| `convert_file(in_path, out_path, max_samples=None)` | 文件转换 | 转换整个 JSONL 文件。返回 `(count, skipped)`。 |
| `scan_dataset_dir(data_dir)` | 文件分类 | 采样每个 JSONL 以分类其格式并估算行数。每条记录携带 `name`、`path`、`size`、`format`、`est_lines` 与 `needs_conversion`。 |
| `split_file(in_path, out_dir, chunk_size=1000, convert=True, prefix=None, dry_run=False)` | 分块 | 把单个 JSONL 拆分为 `NAME_partNNN.jsonl` 块，可选择性转换。`dry_run` 仅统计。返回 `{n_chunks, total_lines, output_files, skipped}`。 |
| `split_directory(in_dir, out_dir, ...)` | 批量拆分 | 对目录中每个匹配的 JSONL 批量拆分，并返回逐文件结果映射。 |

准备任务数据集。

| 导出 | 形状 | 含义 |
|---|---|---|
| `prepare_4tasks(data_dir=None, output_dir=None, n_per_task=20000, seed=42)` | 任务采样 | 从 `pretrain_t2t_mini`、`lora_exam`、`agent_rl_math` 与 `sft_t2t_mini` 采样，生成 `task_{a,b,c,d}_20k.jsonl` 四个文件。 |
| `prepare_hetero(data_dir=None, output_dir=None)` | 任务统一 | 把 `agent_rl_math`/`lora_medical`/`lora_exam`/`rlaif`/`agent_rl` 五个异构数据集统一为扁平的 `task_{a..e}.jsonl` 文本文件，在可行处从真实值回填空的助手回复。 |

管理训练工具与运行目录。

| 导出 | 形状 | 含义 |
|---|---|---|
| `trainer_utils.Logger(path=None, mode="a")` | 日志器 | 把日志输出同时写到 stdout，并可选地写到 utf-8 文件。 |
| `trainer_utils.get_lr(step, total, lr, warmup_ratio=0.1)` | 学习率调度 | 带线性 warmup 的余弦学习率调度。 |
| `trainer_utils.setup_seed(seed=42)` | 种子设置 | 固定 random、numpy 与 torch 的种子，并关闭 benchmark 模式。 |
| `trainer_utils.effective_params(tensor, eps=1e-4)` | 计数统计 | 统计高于相对幅度阈值（`|w| > max(|w|)*eps`）的参数个数。 |
| `trainer_utils.effective_capacity(model, eps=1e-4)` | 计数统计 | 对所有要求梯度的参数执行 `effective_params`，以百万为单位返回总数。 |
| `outver.ensure_run_dir(out_dir)` | 运行目录分配 | 分配并创建下一个 `out/v{N}-{YYYYMMDD}-{HHMMSS}/` 版本目录，返回其路径。 |
| `outver.latest_run_dir_with(out_dir, filename)` | 运行目录查找 | 返回已包含给定产物文件的最高 N 版本目录，或 `None`。 |

-----

<a id="understand-the-implementation"></a>
## 理解实现

`data_converter.py` 以 `convert_sample` 为核心，把 conversations 与 chosen/rejected 样本映射为带 `<|role|>` 标记和末尾 `<|end|>` 的标准文本格式。存在 `gt` 字段时追加真实值，已是 `text` 格式的样本保持不动。`scan_dataset_dir` 仅采样前几行来分类文件，并把行数估算上限设为一万。

`data_splitter.py` 把 `convert_sample` 包装为分块工具。`split_file` 先把整个文件读入内存，用 `math.ceil` 计算块数，写出 `_partNNN.jsonl` 分片，并返回结构化摘要，让调用方无需重扫即可渲染进度。`split_directory` 驱动其在 glob 模式上运行。

`prepare_tasks.py` 在采样前用固定种子做随机化，使拆分可复现。`prepare_4tasks` 从四个来源采样固定的逐任务数量；`prepare_hetero` 统一 RL、医疗与考试数据集，用 `gt` 回填空选手角色，或裁剪末尾空助手轮。

`trainer_utils.py` 提供余弦加 warmup 调度、确定性种子设置、双目标 `Logger` 与 Oja 感知的 `effective_params` 阈值统计。

`outver.py` 是 `out/{v}-{date}-{time}/` 方案的单一事实源。`scan_run_nums` 收集现存版本号，`next_run_num` 取最大值加一，`ensure_run_dir` 创建盖章目录。`latest_run_dir_with` 定位已持有给定产物的最新版本目录。

-----

<a id="further-exploration"></a>
## 进一步探索

- [cli](../cli/README.zh.md)：消费这些辅助的命令行面。
- [model](../model/README.zh.md)：带版本号的输出流入其中描述的模型检查点。