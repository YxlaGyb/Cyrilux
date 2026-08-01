"""
prepare

数据准备子命令
"""

import click

app = click.Group(name="prepare", help="数据准备命令")


@app.command(name="four-task")
@click.option("--subset", "-n", type=int, default=None, help="每任务采样数 (默认 20000)")
@click.option("--seed", "-s", default=42, type=int, help="随机种子")
def four_task(subset, seed):
    """按领域切分 4 个任务数据集 (A/B/C/D)."""
    from pkg.utils.prepare_tasks import prepare_4tasks

    print("准备 4 任务数据")
    prepare_4tasks(n_per_task=subset, seed=seed)
    print("✓ 4 任务数据准备完成")


@app.command()
def hetero():
    """异构数据集统一格式转换."""
    from pkg.utils.prepare_tasks import prepare_hetero

    print("异构数据格式转换")
    prepare_hetero()
    print("✓ 异构数据转换完成")
