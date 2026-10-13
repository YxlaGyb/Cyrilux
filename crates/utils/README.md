---
description: "数据转换、拆分、任务数据准备、训练工具与运行目录版本的共享、与模型无关辅助。"
kind: "package-reference"
---

# utils

English | [中文](README.zh.md)

## Summary

`crates.utils` is the catch-all crate for reusable, model-neutral helpers. It owns data conversion (`data_converter`), large-file splitting (`data_splitter`), task-oriented dataset preparation (`prepare_tasks`), the byte codec and window segmentation that move a corpus into the text container (`codec`), training-adjacent utilities such as logging, learning-rate scheduling, seeding, and parameter counting (`trainer_utils`), and the `out/` version-directory scheme that the cli relies on (`outver`). It holds no model weights and no training loop. Its modules are imported by `crates.cli`, by `crates.hybrid` (which takes the text encoder from `codec` instead of restating it) and by standalone scripts.

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Further exploration](#further-exploration)

-----

<a id="use-this-package"></a>
## Use this package

Convert and split data.

| 导出 | 形状 | 含义 |
|---|---|---|
| `convert_sample(line)` | JSONL line to `{"text": ...}` | Converts one raw line into the standard text format and returns it as a JSON line, or `None` to skip it. Passes already-formatted `text` samples through unchanged. |
| `convert_file(in_path, out_path, max_samples=None)` | File converter | Converts a whole JSONL file. Returns `(count, skipped)`. |
| `scan_dataset_dir(data_dir)` | File classifier | Samples each JSONL to classify its format and estimate line count. Each entry carries `name`, `path`, `size`, `format`, `est_lines`, and `needs_conversion`. |
| `split_file(in_path, out_dir, chunk_size=1000, convert=True, prefix=None, dry_run=False)` | Chunking | Splits one JSONL into `NAME_partNNN.jsonl` chunks, optionally converting. `dry_run` only counts. Returns `{n_chunks, total_lines, output_files, skipped}`. |
| `split_directory(in_dir, out_dir, ...)` | Batch splitter | Batch-splits every matching JSONL in a directory and returns a per-file result map. |

Prepare task datasets.

| 导出 | 形状 | 含义 |
|---|---|---|
| `prepare_4tasks(data_dir=None, output_dir=None, n_per_task=20000, seed=42)` | Task sampler | Samples `pretrain_t2t_mini`, `lora_exam`, `agent_rl_math`, and `sft_t2t_mini` into the four `task_{a,b,c,d}_20k.jsonl` files. |
| `prepare_hetero(data_dir=None, output_dir=None)` | Task unifier | Unifies the five heterogeneous `agent_rl_math`/`lora_medical`/`lora_exam`/`rlaif`/`agent_rl` datasets into flat `task_{a..e}.jsonl` text files, filling empty assistant replies from ground truth where available. |

Encode a corpus into the text container.

| Export | Shape | Meaning |
|---|---|---|
| `ByteDataset(data_path, max_length=128, max_samples=None, lazy=False)` | `Dataset` | JSONL samples to fixed-length byte-id tensors: each sample is truncated at the window width and zero-filled, and `lazy=True` keeps byte offsets and reads one line per item instead of holding the corpus in memory. |
| `load_datasets(paths, max_length=128, max_samples=0)` | `ConcatDataset` | Wraps one `ByteDataset` per path, skipping empty ones, and concatenates them. |
| `encode_text(text)` | `bytes` | The single entry point that turns text into the container's signal order. |
| `extract_window_text(sample)` | `str` | Picks the training text of one sample. |
| `tiled_windows(sample, max_seq_len)` | `list[bytes]` | Tiles the encoded text left-aligned at a stride equal to the window width; an empty sample yields no window. |
| `window_ids(bs, max_seq_len)` | `Tensor`, long | One byte window to a fixed-length id tensor, left-aligned and zero-filled. |
| `skip_zero_windows(samples, max_seq_len)` | `(list[bytes], int)` | Batched form: keeps every tiled window and counts the samples that produced none. |

Manage training utilities and run directories.

| 导出 | 形状 | 含义 |
|---|---|---|
| `trainer_utils.Logger(path=None, mode="a")` | Logger | Tees log output to stdout and, optionally, a utf-8 file. |
| `trainer_utils.get_lr(step, total, lr, warmup_ratio=0.1)` | LR schedule | Cosine learning-rate schedule with a linear warmup. |
| `trainer_utils.setup_seed(seed=42)` | Seeder | Fixes random, numpy, and torch seeds and disables benchmark mode. |
| `trainer_utils.effective_params(tensor, eps=1e-4)` | Count statistic | Counts parameters above a relative magnitude threshold (`|w| > max(|w|)*eps`). |
| `trainer_utils.effective_capacity(model, eps=1e-4)` | Count statistic | Bootstraps `effective_params` over all require-grad parameters and returns the total in millions. |
| `outver.ensure_run_dir(out_dir)` | Run-dir allocator | Allocates and creates the next `out/v{N}-{YYYYMMDD}-{HHMMSS}/` version directory and returns its path. |
| `outver.latest_run_dir_with(out_dir, filename)` | Run-dir finder | Returns the highest-N version dir that already contains a given artifact file, or `None`. |

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

`data_converter.py` centers on `convert_sample`, which maps conversations and chosen/rejected samples into the standard text format with `<|role|>` markers and a trailing `<|end|>`. It appends ground truth when a `gt` field is present, and it passes through samples already in `text` format untouched. `scan_dataset_dir` classifies a file by sampling only its first few lines and caps line estimation at ten thousand rows.

`data_splitter.py` wraps `convert_sample` into a chunking tool. `split_file` reads the whole file into memory first, computes the chunk count with `math.ceil`, writes `_partNNN.jsonl` shards, and returns a structured summary so callers can render progress without re-scanning. `split_directory` drives it over a glob pattern.

`prepare_tasks.py` randomizes with a fixed seed before sampling so splitting is reproducible. `prepare_4tasks` samples a fixed per-task count from four sources; `prepare_hetero` unifies RL, medical, and exam datasets, folding empty assistant roles back with `gt` or trimming trailing empty assistant turns.

`trainer_utils.py` provides the cosine-with-warmup schedule, deterministic seeding, a dual-target `Logger`, and the Oja-aware `effective_params` threshold count.

`codec.py` is the tool that carries a corpus into the text container, and it takes the container's padding position from `crates.modal.text.geometry` instead of restating it, so the organ stays the one owner of that byte. Every path that turns a sample into bytes goes through `encode_text`, the prompt included. `ByteDataset` truncates each sample at the window width and fills the tail with the padding byte, in a plain or a lazy variant. The window functions are pure and share one shape: `tiled_windows` cuts at a stride equal to the window width, so the windows concatenate back into the whole byte string rather than dropping the tail, `window_ids` turns one window into the tensor that is learned from, and `skip_zero_windows` is the batched form that counts the samples yielding no window. The padding byte also carries no learning signal: it sits below the printable start, which is exactly what the network's learn-position mask tests, so the zero-filled tail of a short window is never a learning target.

`outver.py` is the single source of truth for the `out/{v}-{date}-{time}/` scheme. `scan_run_nums` collects the version numbers present, `next_run_num` picks the maximum plus one, and `ensure_run_dir` creates the stamped directory. `latest_run_dir_with` locates the newest version dir that already holds a given artifact.

-----

<a id="further-exploration"></a>
## Further exploration

- [cli](../cli/README.md)：消费这些辅助的命令行面。
- [model](../model/README.md)：带版本号的输出流入其中描述的模型检查点。