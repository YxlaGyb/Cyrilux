"""
virtuoso eval — 模型评估子命令.

子命令:
  all        全面评估 (自监督 + PPL + 生成)
  language   语言能力评估 (Perplexity + 文本生成)
"""

import os
import sys
from pathlib import Path
from typing import Optional

CLI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CLI_DIR)
sys.path.insert(0, PROJECT_ROOT)

import typer
from rich import print as rprint

from cli.utils import PROJECT_ROOT, resolve_path

app = typer.Typer(name="eval", help="模型评估命令", no_args_is_help=True)


@app.command()
def all(
    ctx: typer.Context,
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="Unified 检查点路径"),
    device: str = typer.Option("cuda:0", "--device", "-d", help="计算设备"),
):
    """全面评估: 自监督指标 + Perplexity + 文本生成."""
    from eval_all import main as eval_all_main

    rprint("[bold]全面评估[/]")
    rprint(f"  [dim]检查点: {checkpoint or '默认路径 (遍历所有)'}[/]")
    rprint(f"  [dim]设备: {device}[/]")

    # eval_all 使用 argparse, 我们直接修改 sys.argv
    sys.argv = ["eval_all.py"]
    if checkpoint:
        sys.argv.extend(["--unified", resolve_path(checkpoint)])

    eval_all_main()


@app.command()
def language(
    ctx: typer.Context,
    checkpoint: str = typer.Argument(..., help="检查点路径"),
    local: bool = typer.Option(False, "--local", help="使用 Conv1 骨干网络"),
    device: str = typer.Option("cuda:0", "--device", "-d", help="计算设备"),
):
    """语言能力评估: Perplexity + 文本生成."""
    from eval_pc_language import main as eval_lang_main

    rprint("[bold]语言能力评估[/]")
    rprint(f"  [dim]检查点: {resolve_path(checkpoint)}[/]")
    rprint(f"  [dim]Local: {local}[/]")

    sys.argv = ["eval_pc_language.py", "--ckpt", resolve_path(checkpoint)]
    if local:
        sys.argv.append("--local")

    eval_lang_main()
