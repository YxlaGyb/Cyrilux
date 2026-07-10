"""
virtuoso config — 配置管理子命令.

子命令:
  init      生成默认配置文件模板
  show      显示配置文件内容
  validate  验证配置文件
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional

CLI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CLI_DIR)
sys.path.insert(0, PROJECT_ROOT)

import typer
from rich import print as rprint
from rich.syntax import Syntax
from rich.console import Console

from cli.utils import (
    PROJECT_ROOT, resolve_path, save_config, load_config,
    TRAIN_CONFIG_TEMPLATE, AUTONOMOUS_CONFIG_TEMPLATE,
)

app = typer.Typer(name="config", help="配置管理", no_args_is_help=True)
console = Console()


@app.command()
def init(
    ctx: typer.Context,
    output: str = typer.Option("virtuoso_config.json", "--output", "-o", help="输出路径"),
    template: str = typer.Option("train", "--template", "-t", help="模板类型: train|autonomous"),
):
    """生成默认配置文件模板."""
    if template == "train":
        config = TRAIN_CONFIG_TEMPLATE
    elif template == "autonomous":
        config = AUTONOMOUS_CONFIG_TEMPLATE
    else:
        rprint(f"[red]✗[/] 未知模板类型: {template} (可选: train, autonomous)")
        raise typer.Exit(1)

    save_config(config, output)
    rprint(f"[green]✓[/] {template} 配置模板已生成: {output}")
    rprint("  [dim]编辑此文件后使用: virtuoso train --config <path>[/]")


@app.command()
def show(
    ctx: typer.Context,
    config: str = typer.Argument(..., help="配置文件路径"),
):
    """显示配置文件内容."""
    path = resolve_path(config)
    if not os.path.exists(path):
        rprint(f"[red]✗[/] 文件不存在: {path}")
        raise typer.Exit(1)

    data = load_config(config)
    syntax = Syntax(json.dumps(data, indent=2, ensure_ascii=False),
                    "json", theme="monokai", line_numbers=True)
    console.print(syntax)


@app.command()
def validate(
    ctx: typer.Context,
    config: str = typer.Argument(..., help="配置文件路径"),
):
    """验证配置文件格式和必填项."""
    path = resolve_path(config)
    if not os.path.exists(path):
        rprint(f"[red]✗[/] 文件不存在: {path}")
        raise typer.Exit(1)

    try:
        data = load_config(config)
    except json.JSONDecodeError as e:
        rprint(f"[red]✗[/] JSON 格式错误: {e}")
        raise typer.Exit(1)

    warnings = []

    # 校验训练配置
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
            warnings.append(f"hidden_size={m.get('hidden_size')} 非常规值 (128/256/512)")

    if warnings:
        rprint("[yellow]配置检查:[/]")
        for w in warnings:
            rprint(f"  [yellow]⚠ {w}[/]")
    else:
        rprint("[green]✓[/] 配置文件通过验证")
