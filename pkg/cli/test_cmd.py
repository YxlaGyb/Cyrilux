"""virtuoso test — 测试子命令.

子命令:
  forgetting  多任务灾难性遗忘压力测试
"""

import os
from typing import List, Optional

import typer

from pkg.cli.utils import resolve_path

app = typer.Typer(name="test", help="测试命令", no_args_is_help=True)


@app.command()
def forgetting(
    ctx: typer.Context,
    tasks: List[str] = typer.Argument(..., help="任务数据文件路径 (多个)"),
    task_names: Optional[List[str]] = typer.Option(
        None, "--task-names", help="任务名称 (对应 tasks)",
    ),
    max_samples: Optional[int] = typer.Option(
        None, "--max-samples", help="每任务最大样本数",
    ),
    epochs: int = typer.Option(1, "--epochs", "-e", help="每任务训练轮数"),
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="批大小"),
    lr: float = typer.Option(3e-4, "--lr", help="学习率"),
    max_seq_len: int = typer.Option(256, "--max-seq-len", help="最大序列长度"),
    seed: int = typer.Option(42, "--seed", "-s", help="随机种子"),
    bank_size: int = typer.Option(2000, "--bank-size", help="MemoryBank 大小"),
    exemplars: int = typer.Option(500, "--exemplars", help="每任务保存的 exemplar 数"),
    replay_ratio: int = typer.Option(5, "--replay-ratio", help="回放比例"),
    threshold: float = typer.Option(1.2, "--threshold", help="遗忘嗅探阈值"),
    repair_steps: int = typer.Option(10, "--repair-steps", help="修复步数"),
    check_interval: int = typer.Option(200, "--check-interval", help="检查间隔步数"),
):
    """多任务灾难性遗忘压力测试."""
    import sys

    from model.core.forgetting_pressure_test import main as forgetting_main

    print(f"遗忘压力测试 — 任务数: {len(tasks)}")

    resolved_tasks = [resolve_path(t) for t in tasks]
    for t in resolved_tasks:
        if not os.path.exists(t):
            print(f"✗ 任务文件不存在: {t}")
            raise typer.Exit(1)

    sys.argv = ["forgetting_pressure_test.py",
                "--tasks", *resolved_tasks,
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--lr", str(lr),
                "--max-seq-len", str(max_seq_len),
                "--bank-size", str(bank_size),
                "--exemplars", str(exemplars),
                "--replay-ratio", str(replay_ratio),
                "--threshold", str(threshold),
                "--repair-steps", str(repair_steps),
                "--check-interval", str(check_interval)]
    if task_names:
        sys.argv.extend(["--task-names", *task_names])
    if max_samples:
        sys.argv.extend(["--max-samples", str(max_samples)])

    forgetting_main()



