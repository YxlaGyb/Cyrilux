"""
config
"""

import json
import os

import click

from pkg.cli.utils import load_config, resolve_path, save_config

app = click.Group(name="config", help="配置管理")


TRAIN_TEMPLATE = {
    "batch_size": 48,
    "max_seq_len": 128,
    "lr": 3e-4,
    "epochs": 1,
    "subset": 0,
    "seed": 42,
    "hidden_size": 256,
    "num_hidden_layers": 4,
    "T_infer": 2,
    "gamma": 0.1,
    "max_beta": 2.0,
    "max_beta_conv": 1.0,
    "grad_clip": 1.0,
    "dopamine_eta": 1.0,
    "dopamine_beta": 0.5,
    "dopamine_gamma": 0.3,
    "replay_ratio": 5,
    "bank_size": 2000,
    "sniff_interval": 200,
    "repair_threshold": 1.2,
    "repair_steps": 10,
    "eval_samples": 100,
    "n_prototypes": 8,
    "abstraction_replay_interval": 200,
    "save_interval": 10000,
    "out_dir": "out_pc_unified",
    "task_order": ["a", "b", "c", "d"],
    "data_paths": ["dataset/agent_rl_math.jsonl"],
}

AUTO_TEMPLATE = {
    "wake_steps": 20,
    "play_steps": 100,
    "sleep_interval": 500,
    "batch_size": 16,
    "lr": 1e-4,
    "gamma": 0.05,
    "T_infer": 1,
    "data_dir": "dataset",
    "out_dir": "out_autonomous",
}


@app.command()
@click.option("--output", "-o", default="cyrilux_config.json", help="输出路径")
@click.option("--template", "-t", default="train", help="模板类型: train|autonomous")
def init(output, template):
    """生成默认配置文件模板."""
    if template == "train":
        config = TRAIN_TEMPLATE
    elif template == "autonomous":
        config = AUTO_TEMPLATE
    else:
        raise click.ClickException(f"未知模板类型: {template} (可选: train, autonomous)")

    save_config(config, output)
    print(f"✓ {template} 配置模板已生成: {output}")


@app.command()
@click.argument("config")
def show(config):
    """显示配置文件内容."""
    path = resolve_path(config)
    if not os.path.exists(path):
        raise click.ClickException(f"文件不存在: {path}")

    data = load_config(config)
    print(json.dumps(data, indent=2, ensure_ascii=False))


@app.command()
@click.argument("config")
def validate(config):
    """验证配置文件格式."""
    path = resolve_path(config)
    if not os.path.exists(path):
        raise click.ClickException(f"文件不存在: {path}")

    try:
        data = load_config(config)
    except json.JSONDecodeError as e:
        raise click.ClickException(f"JSON 格式错误: {e}")

    warnings = []
    if "training" in data:
        t = data["training"]
        if t.get("batch_size", 1) > 160:
            warnings.append("batch_size > 160 可能超出 4GB VRAM")
        if t.get("lr", 0) > 0.01:
            warnings.append("lr > 0.01 可能过高")
    else:
        warnings.append("缺少 'training' 字段")

    if "model" in data:
        m = data["model"]
        if m.get("hidden_size", 0) not in (128, 256, 512):
            warnings.append(f"hidden_size={m.get('hidden_size')} 非常规值")

    if warnings:
        print("配置检查:")
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print("✓ 配置文件通过验证")
