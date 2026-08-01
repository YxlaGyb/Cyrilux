"""
CLI 共享工具: 路径处理 & 配置加载.
"""

import json
import os

# 项目根目录 — 相对于 pkg/cli/utils.py 向上 3 层
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_path(p: str) -> str:
    """将相对路径解析为绝对路径 (相对于项目根)."""
    if os.path.isabs(p):
        return p
    return os.path.join(PROJECT_ROOT, p)


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


# ═══════════════════════════════════════════════════════════════════
# 通用配置模板
# ═══════════════════════════════════════════════════════════════════

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
        "out_dir": "out_pc_unified",
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
    "out_dir": "out_autonomous",
    "data_dir": "dataset",
    "data_rotate_interval": 500,
}
