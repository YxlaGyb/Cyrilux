"""test

测试子命令.

子命令:
  forgetting  多任务灾难性遗忘压力测试
"""

import os

import click

from pkg.cli.utils import resolve_path

app = click.Group(name="test", help="测试命令")


@app.command()
@click.argument("tasks", nargs=-1, required=True)
@click.option("--task-names", multiple=True, help="任务名称 (对应 tasks)")
@click.option("--max-samples", type=int, default=None, help="每任务最大样本数")
@click.option("--epochs", "-e", default=1, type=int, help="每任务训练轮数")
@click.option("--batch-size", "-b", default=16, type=int, help="批大小")
@click.option("--lr", default=3e-4, type=float, help="学习率")
@click.option("--max-seq-len", default=256, type=int, help="最大序列长度")
@click.option("--seed", "-s", default=42, type=int, help="随机种子")
@click.option("--bank-size", default=2000, type=int, help="MemoryBank 大小")
@click.option("--exemplars", default=500, type=int, help="每任务保存的 exemplar 数")
@click.option("--replay-ratio", default=5, type=int, help="回放比例")
@click.option("--threshold", default=1.2, type=float, help="遗忘嗅探阈值")
@click.option("--repair-steps", default=10, type=int, help="修复步数")
@click.option("--check-interval", default=200, type=int, help="检查间隔步数")
def forgetting(
    tasks,
    task_names,
    max_samples,
    epochs,
    batch_size,
    lr,
    max_seq_len,
    seed,
    bank_size,
    exemplars,
    replay_ratio,
    threshold,
    repair_steps,
    check_interval,
):
    """多任务灾难性遗忘压力测试."""
    import sys

    from importlib.util import module_from_spec, spec_from_file_location

    _spec = spec_from_file_location(
        "forgetting_test", os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "forgetting_test.py")
    )
    _mod = module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    forgetting_main = _mod.main

    print(f"遗忘压力测试 — 任务数: {len(tasks)}")

    resolved_tasks = [resolve_path(t) for t in tasks]
    for t in resolved_tasks:
        if not os.path.exists(t):
            raise click.ClickException(f"任务文件不存在: {t}")

    sys.argv = [
        "forgetting_pressure_test.py",
        "--tasks",
        *resolved_tasks,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--max-seq-len",
        str(max_seq_len),
        "--bank-size",
        str(bank_size),
        "--exemplars",
        str(exemplars),
        "--replay-ratio",
        str(replay_ratio),
        "--threshold",
        str(threshold),
        "--repair-steps",
        str(repair_steps),
        "--check-interval",
        str(check_interval),
    ]
    if task_names:
        sys.argv.extend(["--task-names", *task_names])
    if max_samples:
        sys.argv.extend(["--max-samples", str(max_samples)])

    forgetting_main()
