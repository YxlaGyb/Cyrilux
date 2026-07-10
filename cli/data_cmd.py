"""
virtuoso data — 数据管理子命令.

子命令:
  convert   转换数据格式 (conversations→text)
  split     分割大 JSONL 文件
  scan      扫描数据集目录
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
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from cli.utils import PROJECT_ROOT, resolve_path
from data_converter import convert_file, scan_dataset_dir
from data_splitter import split_file

app = typer.Typer(name="data", help="数据管理命令", no_args_is_help=True)
console = Console()


@app.command()
def convert(
    ctx: typer.Context,
    input: str = typer.Argument(..., help="输入 JSONL 文件路径"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="输出文件路径 (默认: input_converted.jsonl)"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", "-n", help="最大样本数"),
):
    """转换数据格式: 将 conversations/chosen-rejected 格式转为标准 text 格式."""
    input_path = resolve_path(input)
    if not os.path.exists(input_path):
        rprint(f"[red]✗[/] 文件不存在: {input_path}")
        raise typer.Exit(1)

    output_path = output if output else input_path.replace('.jsonl', '_converted.jsonl')
    output_path = resolve_path(output_path)

    rprint(f"[dim]输入: {input_path}[/]")
    rprint(f"[dim]输出: {output_path}[/]")
    if max_samples:
        rprint(f"[dim]最大样本数: {max_samples}[/]")

    convert_file(input_path, output_path, max_samples)
    rprint(f"[green]✓[/] 转换完成: {output_path}")


@app.command()
def split(
    ctx: typer.Context,
    input: str = typer.Argument(..., help="输入 JSONL 文件"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="输出目录 (默认: {input_name}_split/)"),
    chunk_size: int = typer.Option(1000, "--chunk-size", "-c", help="每个输出文件的样本数"),
    no_convert: bool = typer.Option(False, "--no-convert", help="不自动转换格式"),
    dry_run: bool = typer.Option(False, "--dry-run", help="仅统计不分割"),
):
    """将大 JSONL 文件分割为多个小文件."""
    input_path = resolve_path(input)
    if not os.path.exists(input_path):
        rprint(f"[red]✗[/] 文件不存在: {input_path}")
        raise typer.Exit(1)

    if output_dir is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.join(os.path.dirname(input_path), f"{base}_split")
    output_dir = resolve_path(output_dir)

    result = split_file(
        input_path,
        output_dir,
        chunk_size=chunk_size,
        convert=not no_convert,
        dry_run=dry_run,
    )

    if dry_run:
        rprint(f"[yellow]├[/] 总计: {result['total_lines']} 行")
        rprint(f"[yellow]├[/] 分割后: {result['n_chunks']} 个文件")
        rprint(f"[yellow]└[/] 每个: {chunk_size} 行")
    else:
        rprint(f"[green]✓[/] 分割完成: {result['n_chunks']} 个文件 → {output_dir}")
        for f in result.get('output_files', []):
            rprint(f"  [dim]{f}[/]")


@app.command()
def scan(
    ctx: typer.Context,
    directory: Optional[str] = typer.Argument(None, help="数据集目录 (默认: datasets/)"),
):
    """扫描数据集目录, 以 Rich 表格展示统计信息."""
    scan_dir = resolve_path(directory or "datasets")
    if not os.path.exists(scan_dir):
        rprint(f"[red]✗[/] 目录不存在: {scan_dir}")
        raise typer.Exit(1)

    results = scan_dataset_dir(scan_dir)

    if not results:
        rprint(f"[yellow]⚠[/] 目录中没有 JSONL 文件: {scan_dir}")
        return

    table = Table(title=f"数据集目录: {scan_dir}", title_style="bold")
    table.add_column("文件", style="cyan")
    table.add_column("行数", justify="right", style="green")
    table.add_column("大小", justify="right", style="yellow")
    table.add_column("格式", style="white")

    total_lines = 0
    for r in results:
        table.add_row(r['name'], str(r['est_lines']), r['size_str'], r['format'])
        total_lines += r['est_lines']

    console.print(table)
    rprint(f"[bold]总计:[/] {len(results)} 个文件, {total_lines} 行 (采样)")
