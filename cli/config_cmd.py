"""
virtuoso config — 配置管理子命令.
"""

import os, json
from typing import Optional

import typer
from cli.utils import resolve_path, save_config, load_config

app = typer.Typer(name="config", help="配置管理", no_args_is_help=True)


TRAIN_TEMPLATE = {
    "batch_size": 48, "max_seq_len": 128, "lr": 3e-4, "epochs": 1,
    "subset": 0, "seed": 42, "hidden_size": 256, "num_hidden_layers": 4,
    "T_infer": 2, "gamma": 0.1, "max_beta": 2.0, "max_beta_conv": 1.0, "grad_clip": 1.0,
    "dopamine_eta": 1.0, "dopamine_beta": 0.5, "dopamine_gamma": 0.3,
    "replay_ratio": 5, "bank_size": 2000, "sniff_interval": 200,
    "repair_threshold": 1.2, "repair_steps": 10, "eval_samples": 100,
    "n_prototypes": 8, "abstraction_replay_interval": 200,
    "save_interval": 500, "out_dir": "out_pc_unified",
    "task_order": ["a", "b", "c", "d"],
    "data_paths": ["dataset/agent_rl_math.jsonl"],
}

AUTO_TEMPLATE = {
    "wake_steps": 20, "play_steps": 100, "sleep_interval": 500,
    "batch_size": 16, "lr": 1e-4, "gamma": 0.05, "T_infer": 1,
    "data_dir": "dataset", "out_dir": "out_autonomous",
}


@app.command()
def init(
    ctx: typer.Context,
    output: str = typer.Option("virtuoso_config.json", "--output", "-o", help="输出路径"),
    template: str = typer.Option("train", "--template", "-t", help="模板类型: train|autonomous"),
):
    """生成默认配置文件模板."""
    if template == "train":
        config = TRAIN_TEMPLATE
    elif template == "autonomous":
        config = AUTO_TEMPLATE
    else:
        print(f"✗ 未知模板类型: {template} (可选: train, autonomous)")
        raise typer.Exit(1)

    save_config(config, output)
    print(f"✓ {template} 配置模板已生成: {output}")


@app.command()
def show(
    ctx: typer.Context,
    config: str = typer.Argument(..., help="配置文件路径"),
):
    """显示配置文件内容."""
    path = resolve_path(config)
    if not os.path.exists(path):
        print(f"✗ 文件不存在: {path}")
        raise typer.Exit(1)

    data = load_config(config)
    print(json.dumps(data, indent=2, ensure_ascii=False))


@app.command()
def validate(
    ctx: typer.Context,
    config: str = typer.Argument(..., help="配置文件路径"),
):
    """验证配置文件格式."""
    path = resolve_path(config)
    if not os.path.exists(path):
        print(f"✗ 文件不存在: {path}")
        raise typer.Exit(1)

    try:
        data = load_config(config)
    except json.JSONDecodeError as e:
        print(f"✗ JSON 格式错误: {e}")
        raise typer.Exit(1)

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
