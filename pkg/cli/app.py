"""
virtuosov2 CLI — 主入口 (Typer).

使用方式:
  virtuoso [OPTIONS] COMMAND [ARGS]...
  virtuoso --help
"""

import os

import typer

from pkg.cli import (
    data_cmd, eval_cmd, test_cmd, prepare_cmd,
    config_cmd, list_cmd, train_cmd, autonomous_cmd,
)

app = typer.Typer(
    name="virtuoso",
    help="virtuosov2 CLI — Predictive Coding 局部动态小语言模型",
    no_args_is_help=True,
)


@app.callback()
def main(ctx: typer.Context):
    ctx.obj = {"project_root": PROJECT_ROOT}


app.add_typer(data_cmd.app, name="data", help="数据管理 (转换/分割/扫描)")
app.add_typer(train_cmd.app, name="train", help="Phase 1: 模型训练")
app.add_typer(eval_cmd.app, name="eval", help="模型评估")
app.add_typer(test_cmd.app, name="test", help="遗忘压力测试")
app.add_typer(prepare_cmd.app, name="prepare", help="数据准备")
app.add_typer(config_cmd.app, name="config", help="配置文件管理")
app.add_typer(list_cmd.app, name="list", help="信息查询")
app.add_typer(autonomous_cmd.app, name="autonomous", help="Phase 2: 持续自主运行")

if __name__ == "__main__":
    app()
