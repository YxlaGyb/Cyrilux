"""
virtuosov2 CLI — 主入口.

使用方式:
  virtuoso [OPTIONS] COMMAND [ARGS]...
  virtuoso --help
"""

import os
import sys

# 将项目根加入 sys.path (必须优先执行)
CLI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CLI_DIR)
sys.path.insert(0, PROJECT_ROOT)

import typer
from typing import Optional

from cli import data_cmd, eval_cmd, test_cmd, prepare_cmd
from cli import config_cmd, list_cmd, train_cmd, autonomous_cmd

# ═══════════════════════════════════════════════════════════════════
# 主应用
# ═══════════════════════════════════════════════════════════════════

app = typer.Typer(
    name="virtuoso",
    help="virtuosov2 CLI — Predictive Coding 局部动态小语言模型",
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.callback()
def main(
    ctx: typer.Context,
    device: str = typer.Option("cuda:0", "--device", "-d", help="计算设备"),
    seed: int = typer.Option(42, "--seed", "-s", help="随机种子"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细输出"),
):
    """
    virtuosov2 — 基于预测编码的局部动态小语言模型训练与评估工具.

    \b
    子命令分组:
      data        数据管理 (转换/分割/扫描)
      train       Phase 1: 模型训练
      autonomous  Phase 2: 持续自主运行 (WAKE→PLAY→SLEEP)
      eval        模型评估 (PPL/生成)
      test        遗忘压力测试
      prepare     数据准备
      config      配置文件管理
      list        信息查询
    """
    ctx.obj = {
        "device": device,
        "seed": seed,
        "verbose": verbose,
        "project_root": PROJECT_ROOT,
    }


# ── 注册子命令组 ──
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
