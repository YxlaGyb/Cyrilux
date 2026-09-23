---
description: "cli：train、resume、配置、检视与数据准备的 click 命令行入口，面向 dense PPA 后端。"
kind: "package-reference"
---

# cli

English | [中文](README.zh.md)

## Summary

`crates.cli` is the command surface of the project. It builds a single click group named `app` that registers six subcommands: `train`, `from-config`, `resume`, plus `data`, `list`, and `config` groups. It selects the dense PPA backend, wires data files and checkpoints, allocates versioned output directories under `out/`, and delegates heavy work to `crates.model` and `crates.utils`. Paths are resolved relative to the project root, and byte windows are utf-8 encoded and zero-padded before they reach the model. It holds no model code and no training loop of its own.

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Further exploration](#further-exploration)

-----

<a id="use-this-package"></a>
## Use this package

Run the training paths.

| 导出 | 形状 | 含义 |
|---|---|---|
| `app train -d FILE...` | Training | Direct mode. Loads each JSONL file line by line, batches byte windows, and runs pure Hebbian `net.learn`. Saves `final.safetensors` into the output directory. Rejects the frozen `sparse` backend. |
| `app train --resume` | Training | Finds the newest `final.safetensors` under `out/v*` and continues from it. |
| `app from-config CFG.json` | Training | Loads a JSON config and trains. CLI options such as `--batch-size`, `--lr`, `--epochs`, `--out-dir` override the config. |
| `app resume CKPT.safetensors` | Training | Rebuilds the network from a checkpoint with its own config and optionally keeps training on new data with `-d`. Preserves the checkpoint max sequence length. |

Inspect the workspace.

| 导出 | 形状 | 含义 |
|---|---|---|
| `app list checkpoints [DIR]` | Inspect | Lists `.safetensors` files with size and kind. `--detail` reads metadata headers lazily without loading tensors. |
| `app list datasets [DIR]` | Inspect | Lists JSONL files under `datasets/`, with line counts and a best-effort format classification. |
| `app config init -o OUT -t TEMPLATE` | Config | Writes a default config template (`train` or `autonomous`). |
| `app config show CFG` / `app config validate CFG` | Config | Displays or sanity-checks a JSON config and prints warnings. |

Prepare data.

| 导出 | 形状 | 含义 |
|---|---|---|
| `app data convert IN` | Data | Converts `conversations` or `chosen`/`rejected` samples to flat `{"text": ...}` lines. |
| `app data split IN --chunk-size N` | Data | Splits a large JSONL into multiple files, converting by default. `--dry-run` only counts. |
| `app data scan [DIR]` | Data | Summarizes format and estimated line counts for a dataset directory. |

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

`__init__.py` builds the `app` group and registers every subcommand, reconfiguring stdout and stderr to utf-8 first. Entry commands import the shared helpers from `crates.cli.utils` rather than re-deriving project layout.

`utils.py` is the shared spine. `resolve_path` anchors relative paths at the project root. `run_dir` lazily allocates this process version directory by calling `crates.utils.outver.ensure_run_dir`, and `pin_run_dir` anchors a resumed run to the existing training line. `run_file`, `load_config`, `save_config`, and `merge_config` cover the config plumbing. It also carries the `TRAIN_CONFIG_TEMPLATE` and `AUTONOMOUS_CONFIG_TEMPLATE` defaults.

`train_cmd.py` owns the dense training loop. `skip_zero_windows` and `window_or_skip` are pure functions that encode sample text to utf-8, truncate to `max_seq_len`, and drop windows that would be all zeros. The `_train_dense` helper iterates JSONL with a line-scan first pass for progress reporting, stacks byte windows into batch tensors, and calls `net.learn` with gradients disabled. Several legacy options that only mattered for the archived sparse backend are accepted and reported as ignored.

The `train` command fails fast on the archived `sparse` backend and on a missing `-d` list. `resume` reads the config snapshot from the checkpoint first so dimension fields always follow the snapshot, and bumps the buffer max sequence length only if the request exceeds it.

-----

<a id="further-exploration"></a>
## Further exploration

- [utils](../utils/README.md)：共享的路径与输出管理辅助。
- [model](../model/README.md)：训练循环驱动的模型与引擎。