"""
CLI 共享工具: 路径处理 & 配置加载.
"""

import json
import os

from pkg.outver import ensure_run_dir

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_path(p: str) -> str:
    """将相对路径解析为绝对路径 (相对于项目根)."""
    if os.path.isabs(p):
        return p
    return os.path.join(PROJECT_ROOT, p)


_RUN_DIR: str | None = None


def run_dir() -> str:
    """本次训练的输出版本目录 out/v{N}-{YYYYMMDD}-{HHMMSS}/ — 每进程一个, 首次调用时创建.

    N = out/ 下已有版本目录最大值 + 1, 全自动自增 (pkg/outver 单一事实源).
    续跑延续既有训练线时不调用本函数, 用 pin_run_dir 锚定原目录.
    """
    global _RUN_DIR
    if _RUN_DIR is None:
        _RUN_DIR = ensure_run_dir(resolve_path("out"))
    return _RUN_DIR


def pin_run_dir(path: str) -> str:
    """续跑: 把本次进程的产物目录锚定到既有训练线的版本目录."""
    global _RUN_DIR
    _RUN_DIR = path
    return _RUN_DIR


def run_file(name: str) -> str:
    """run_dir() 下的文件路径 (目录已创建)."""
    return os.path.join(run_dir(), name)


def load_config(path: str) -> dict:
    """加载 JSON 配置文件."""
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict, path: str):
    """保存 JSON 配置文件."""
    path = resolve_path(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✓ 配置已保存: {path}")


def merge_config(config: dict, cli_overrides: dict) -> dict:
    """CLI 参数覆写配置项."""
    merged = dict(config)
    for k, v in cli_overrides.items():
        if v is not None:
            merged[k] = v
    return merged


TRAIN_CONFIG_TEMPLATE = {
    "model": {
        "hidden_size": 256,
        "num_hidden_layers": 4,
    },
    "data": {
        "data_files": ["datasets/task_a_daily_20k.jsonl"],
        "combined_training": True,
        "subset": 0,
    },
    "training": {
        "batch_size": 48,
        "max_seq_len": 128,
        "lr": 3e-4,
        "epochs": 1,
        "warmup_steps": 0,
        "grad_clip": 1.0,
        "weight_decay": 0.01,
    },
    "pc": {
        "T_infer": 1,
        "gamma": 0.1,
    },
    "dopamine": {
        "enabled": True,
        "eta": 1.0,
        "beta": 0.5,
        "gamma": 0.3,
    },
    "quantize": {
        "enabled": False,
    },
    "output": {
        "out_dir": "",
        "save_interval": 10000,
    },
}

AUTONOMOUS_CONFIG_TEMPLATE = {
    "wake_steps": 20,
    "play_steps": 100,
    "sleep_interval": 500,
    "gen_max_new": 64,
    "gen_temperature": 0.8,
    "gen_top_k": 40,
    "gen_prompt_len": 32,
    "batch_size": 16,
    "max_seq_len": 128,
    "lr": 1e-4,
    "gamma": 0.05,
    "T_infer": 1,
    "grad_clip": 1.0,
    "dopamine_eta": 1.0,
    "dopamine_beta": 0.3,
    "dopamine_gamma": 0.2,
    "dopamine_threshold": 0.05,
    "max_replay_buffer": 2000,
    "replay_batch_size": 16,
    "replay_ratio": 3,
    "save_interval": 10000,
    "out_dir": "",
    "data_dir": "dataset",
    "data_rotate_interval": 500,
}
