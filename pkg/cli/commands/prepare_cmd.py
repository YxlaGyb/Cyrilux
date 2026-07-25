"""
prepare — 数据准备子命令 (委托至 Cyrilux.core.prepare_tasks).
"""

from typing import Optional

import typer

app = typer.Typer(name="prepare", help="数据准备命令", no_args_is_help=True)


@app.command()
def four_task(
    ctx: typer.Context,
    subset: Optional[int] = typer.Option(
        None, "--subset", "-n", help="每任务采样数 (默认 20000)"
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="随机种子"),
):
    """按领域切分 4 个任务数据集 (A/B/C/D)."""
    from pkg.utils.prepare_tasks import prepare_4tasks

    print("准备 4 任务数据")
    prepare_4tasks(subset=subset, seed=seed)  # type: ignore[call-arg]
    print("✓ 4 任务数据准备完成")


@app.command()
def hetero(
    ctx: typer.Context,
):
    """异构数据集统一格式转换."""
    from pkg.utils.prepare_tasks import prepare_hetero

    print("异构数据格式转换")
    prepare_hetero()
    print("✓ 异构数据转换完成")
