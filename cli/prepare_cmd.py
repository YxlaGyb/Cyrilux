"""
virtuoso prepare — 数据准备子命令.

子命令:
  4task      按领域切分 4 个任务数据集
  hetero     异构数据集统一格式转换
"""

import os
import sys
from typing import Optional

CLI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CLI_DIR)
sys.path.insert(0, PROJECT_ROOT)

import typer
from rich import print as rprint

from cli.utils import PROJECT_ROOT

app = typer.Typer(name="prepare", help="数据准备命令", no_args_is_help=True)


@app.command()
def four_task(
    ctx: typer.Context,
    subset: Optional[int] = typer.Option(None, "--subset", "-n", help="每任务采样数 (默认 20000)"),
    seed: int = typer.Option(42, "--seed", "-s", help="随机种子"),
):
    """按领域切分 4 个任务数据集 (A:通用 / B:科技 / C:医疗 / D:指令)."""
    rprint("[bold]准备 4 任务数据[/]")

    # 修改 prepare_4task 中的全局变量
    import prepare_4task
    if subset is not None:
        prepare_4task.N_PER_TASK = subset
    prepare_4task.SEED = seed

    prepare_4task.main()
    rprint(f"[green]✓[/] 4 任务数据准备完成 → {prepare_4task.OUT_DIR}")


@app.command()
def hetero(
    ctx: typer.Context,
):
    """异构数据集统一格式 (agent_rl_math/medical/exam/rlaif/agent_rl → text)."""
    from prepare_hetero_tasks import process as hetero_process

    rprint("[bold]异构数据格式转换[/]")
    rprint("  源: agent_rl_math, lora_medical, lora_exam, rlaif, agent_rl")
    rprint("  目标: datasets/task_{a,b,c,d,e}.jsonl")

    hetero_process()
    rprint("[green]✓[/] 异构数据转换完成")
