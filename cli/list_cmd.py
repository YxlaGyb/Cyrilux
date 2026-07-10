"""
virtuoso list — 信息查询子命令.

子命令:
  checkpoints  列出检查点
  datasets     列出数据集
"""

import os
import sys
from pathlib import Path
from typing import Optional

CLI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CLI_DIR)
sys.path.insert(0, PROJECT_ROOT)

import typer
import torch
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from cli.utils import PROJECT_ROOT, resolve_path

app = typer.Typer(name="list", help="信息查询", no_args_is_help=True)
console = Console()


@app.command()
def checkpoints(
    ctx: typer.Context,
    directory: Optional[str] = typer.Argument(None, help="检查点目录 (默认: out_pc_unified)"),
    detail: bool = typer.Option(False, "--detail", help="显示详细信息 (加载检查点)"),
):
    """列出检查点文件."""
    ckpt_dir = resolve_path(directory or "out_pc_unified")
    if not os.path.exists(ckpt_dir):
        rprint(f"[red]✗[/] 目录不存在: {ckpt_dir}")
        raise typer.Exit(1)

    pt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(('.pt', '.pth'))])

    if not pt_files:
        rprint(f"[yellow]⚠[/] 没有检查点文件: {ckpt_dir}")
        return

    table = Table(title=f"检查点: {ckpt_dir}", title_style="bold")
    table.add_column("文件", style="cyan")
    table.add_column("大小", justify="right", style="yellow")
    table.add_column("类型", style="green")
    if detail:
        table.add_column("Steps", justify="right")
        table.add_column("CE", justify="right")
        table.add_column("F", justify="right")

    for fname in pt_files:
        fpath = os.path.join(ckpt_dir, fname)
        size = os.path.getsize(fpath)
        size_str = f"{size / 1024 / 1024:.1f} MB" if size > 1024 * 1024 else f"{size / 1024:.1f} KB"

        # 判断类型
        ftype = "检查点"
        if "final" in fname:
            ftype = "最终模型"
        elif "int4" in fname:
            ftype = "Int4 量化"
        elif "best" in fname:
            ftype = "最佳"

        if detail:
            try:
                ckpt = torch.load(fpath, map_location="cpu", weights_only=True)
                steps = ckpt.get("global_step", ckpt.get("step", "?"))
                ce = f"{ckpt.get('CE_local', ckpt.get('ce_loss', '?')):.4f}" if isinstance(ckpt.get('CE_local'), (int, float)) else "?"
                F = f"{ckpt.get('F', '?'):.2f}" if isinstance(ckpt.get('F'), (int, float)) else "?"
                table.add_row(fname, size_str, ftype, str(steps), str(ce), str(F))
            except Exception:
                table.add_row(fname, size_str, ftype, "?", "?", "?")
        else:
            table.add_row(fname, size_str, ftype)

    console.print(table)


@app.command()
def datasets(
    ctx: typer.Context,
    directory: Optional[str] = typer.Option(None, "--directory", "-d", help="数据集目录 (默认: datasets/)"),
    pattern: str = typer.Option("*.jsonl", "--pattern", "-p", help="文件匹配模式"),
):
    """列出数据集文件."""
    data_dir = resolve_path(directory or "datasets")
    if not os.path.exists(data_dir):
        rprint(f"[red]✗[/] 目录不存在: {data_dir}")
        raise typer.Exit(1)

    import glob
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))

    if not files:
        rprint(f"[yellow]⚠[/] 没有匹配的文件: {data_dir}/{pattern}")
        return

    table = Table(title=f"数据集: {data_dir}", title_style="bold")
    table.add_column("文件", style="cyan")
    table.add_column("大小", justify="right", style="yellow")
    table.add_column("行数", justify="right", style="green")
    table.add_column("类型", style="white")

    total_size = 0
    total_lines = 0
    for fpath in files:
        fname = os.path.basename(fpath)
        size = os.path.getsize(fpath)
        size_str = f"{size / 1024 / 1024:.1f} MB" if size > 1024 * 1024 else f"{size / 1024:.1f} KB"
        total_size += size

        # 统计行数 (前 10000 行)
        lines = 0
        ftype = "未知"
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i == 0:
                        sample = line[:200]
                        if '"text"' in sample:
                            ftype = "text"
                        elif '"conversations"' in sample:
                            ftype = "conversations"
                        elif '"chosen"' in sample:
                            ftype = "chosen/rejected"
                        else:
                            ftype = "raw"
                    if i >= 10000:
                        lines = f"{i}+"
                        break
                    lines = i + 1
            total_lines += int(str(lines).rstrip('+'))
        except Exception:
            lines = "?"
            ftype = "?"

        table.add_row(fname, size_str, str(lines), ftype)

    console.print(table)
    rprint(f"[bold]总计:[/] {len(files)} 个文件, {total_size / 1024 / 1024:.1f} MB")
