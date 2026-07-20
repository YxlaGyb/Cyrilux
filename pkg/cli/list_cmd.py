"""
virtuoso list — 信息查询子命令.
"""

import os
from typing import Optional

import typer
from pkg.cli.utils import resolve_path

app = typer.Typer(name="list", help="信息查询", no_args_is_help=True)


@app.command()
def checkpoints(
    ctx: typer.Context,
    directory: Optional[str] = typer.Argument(None, help="检查点目录 (默认: out_pc_unified)"),
    detail: bool = typer.Option(False, "--detail", help="显示详细信息 (加载检查点)"),
):
    """列出检查点文件."""
    import torch

    ckpt_dir = resolve_path(directory or "out_pc_unified")
    if not os.path.exists(ckpt_dir):
        print(f"✗ 目录不存在: {ckpt_dir}")
        raise typer.Exit(1)

    pt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(('.pt', '.pth'))])
    if not pt_files:
        print(f"没有检查点文件: {ckpt_dir}")
        return

    print(f"\n检查点: {ckpt_dir}")
    print(f"{'文件':40s} {'大小':>10s}  {'类型'}")
    print("-" * 70)
    for fname in pt_files:
        fpath = os.path.join(ckpt_dir, fname)
        size = os.path.getsize(fpath)
        size_str = f"{size / 1024 / 1024:.1f} MB" if size > 1024 * 1024 else f"{size / 1024:.1f} KB"
        kind = "unified" if "unified" in fname else ("task" if "task" in fname else "ckpt")
        print(f"{fname:40s} {size_str:>10s}  {kind}")

    if detail:
        for fname in pt_files:
            fpath = os.path.join(ckpt_dir, fname)
            try:
                state = torch.load(fpath, map_location="cpu", weights_only=True)
                if isinstance(state, dict) and "step" in state:
                    print(f"  {fname}: step={state.get('step', '?')}  ce={state.get('ce_loss', '?')}")
            except Exception:
                pass

        pass


@app.command()
def datasets(
    ctx: typer.Context,
    directory: Optional[str] = typer.Option(None, "--directory", "-d", help="数据集目录 (默认: datasets/)"),
    pattern: str = typer.Option("*.jsonl", "--pattern", "-p", help="文件匹配模式"),
):
    """列出数据集文件."""
    import glob

    data_dir = resolve_path(directory or "datasets")
    if not os.path.exists(data_dir):
        print(f"✗ 目录不存在: {data_dir}")
        raise typer.Exit(1)

    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        print(f"没有匹配的文件: {data_dir}/{pattern}")
        return

    print(f"\n数据集: {data_dir}")
    print(f"{'文件':40s} {'大小':>10s} {'行数':>6s}  类型")
    print("-" * 75)
    total_size = 0
    for fpath in files:
        fname = os.path.basename(fpath)
        size = os.path.getsize(fpath)
        size_str = f"{size / 1024 / 1024:.1f} MB" if size > 1024 * 1024 else f"{size / 1024:.1f} KB"
        total_size += size
        lines = 0
        ftype = "未知"
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i == 0:
                        if '"text"' in line[:200]:
                            ftype = "text"
                        elif '"conversations"' in line[:200]:
                            ftype = "conversations"
                        elif '"chosen"' in line[:200]:
                            ftype = "chosen/rejected"
                        else:
                            ftype = "raw"
                    if i >= 10000:
                        lines = i + 1
                        break
                    lines = i + 1
        except Exception:
            lines = -1
        print(f"{fname:40s} {size_str:>10s} {str(lines) if lines >= 0 else '?':>6s}  {ftype}")
    print(f"总计: {len(files)} 个文件, {total_size / 1024 / 1024:.1f} MB")
