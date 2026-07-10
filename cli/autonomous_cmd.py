"""
virtuoso autonomous — Phase 2 持续自主运行子命令.

实现 WAKE → PLAY → SLEEP 循环, 在后台环境持续采集数据-训练-休眠.

用法:
  virtuoso autonomous [OPTIONS]
  virtuoso autonomous --checkpoint out_pc_unified/step_1000.pt
  virtuoso autonomous --data-dir dataset --out-dir out_autonomous
"""

import os
import sys
from pathlib import Path
from typing import Optional

CLI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CLI_DIR)
sys.path.insert(0, PROJECT_ROOT)

import typer
from rich import print as rprint
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from cli.utils import (
    PROJECT_ROOT, resolve_path, AUTONOMOUS_CONFIG_TEMPLATE,
    RichTrainingPanel, console,
)

app = typer.Typer(name="autonomous", help="Phase 2: 持续自主运行", no_args_is_help=True)


@app.callback(invoke_without_command=True)
def autonomous(
    ctx: typer.Context,
    # ── 核心参数 ──
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="初始检查点"),
    out_dir: str = typer.Option("out_autonomous", "--out-dir", "-o", help="输出目录"),
    data_dir: str = typer.Option("dataset", "--data-dir", "-d", help="数据目录"),

    # ── 周期控制 ──
    wake_steps: int = typer.Option(20, "--wake-steps", help="WAKE 阶段步数"),
    play_steps: int = typer.Option(100, "--play-steps", help="PLAY 阶段步数"),
    sleep_interval: int = typer.Option(500, "--sleep-interval", help="SLEEP 间隔步数"),

    # ── 生成参数 ──
    gen_max_new: int = typer.Option(64, "--gen-max-new", help="生成最大 token 数"),
    gen_temperature: float = typer.Option(0.8, "--gen-temp", help="生成温度"),
    gen_top_k: int = typer.Option(40, "--gen-top-k", help="Top-K 采样"),
    gen_prompt_len: int = typer.Option(32, "--gen-prompt-len", help="提示长度"),

    # ── 训练参数 ──
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="批大小"),
    max_seq_len: int = typer.Option(128, "--max-seq-len", help="最大序列长度"),
    lr: float = typer.Option(1e-4, "--lr", help="学习率"),
    gamma: float = typer.Option(0.05, "--gamma", help="多巴胺折扣因子"),
    T_infer: int = typer.Option(1, "--T-infer", help="Inference time steps"),
    grad_clip: float = typer.Option(1.0, "--grad-clip", help="梯度裁剪"),

    # ── 多巴胺 ──
    dopamine_eta: float = typer.Option(1.0, "--dopamine-eta", help="多巴胺学习率"),
    dopamine_beta: float = typer.Option(0.3, "--dopamine-beta", help="多巴胺灵敏度"),
    dopamine_gamma: float = typer.Option(0.2, "--dopamine-gamma", help="多巴胺衰减"),
    dopamine_threshold: float = typer.Option(0.05, "--dopamine-threshold", help="多巴胺阈值"),

    # ── 回放 ──
    max_replay_buffer: int = typer.Option(2000, "--max-replay-buffer", help="最大回放缓冲"),
    replay_batch_size: int = typer.Option(16, "--replay-batch-size", help="回放批大小"),
    replay_ratio: int = typer.Option(3, "--replay-ratio", help="回放比例"),

    # ── 保存 ──
    save_interval: int = typer.Option(200, "--save-interval", help="保存间隔"),
    data_rotate_interval: int = typer.Option(500, "--data-rotate-interval", help="数据轮转间隔"),

    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """Phase 2: 持续自主运行 (WAKE→PLAY→SLEEP 循环).

    模型在生成-训练-休眠循环中持续自主运行,
    定期保存检查点和轮转数据.
    """
    from autonomous_mind import AutonomousMind

    device = ctx.obj.get("device", "cuda:0")
    resolved_checkpoint = resolve_path(checkpoint) if checkpoint else None
    resolved_data_dir = resolve_path(data_dir)

    mind = AutonomousMind(
        # 核心
        checkpoint_path=resolved_checkpoint,
        out_dir=out_dir,
        data_dir=resolved_data_dir,

        # 周期
        wake_steps=wake_steps,
        play_steps=play_steps,
        sleep_interval=sleep_interval,

        # 生成
        gen_max_new=gen_max_new,
        gen_temperature=gen_temperature,
        gen_top_k=gen_top_k,
        gen_prompt_len=gen_prompt_len,

        # 训练
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        lr=lr,
        gamma=gamma,
        T_infer=T_infer,
        grad_clip=grad_clip,

        # 多巴胺
        dopamine_eta=dopamine_eta,
        dopamine_beta=dopamine_beta,
        dopamine_gamma=dopamine_gamma,
        dopamine_threshold=dopamine_threshold,

        # 回放
        max_replay_buffer=max_replay_buffer,
        replay_batch_size=replay_batch_size,
        replay_ratio=replay_ratio,

        # 保存
        save_interval=save_interval,
        data_rotate_interval=data_rotate_interval,

        verbose=verbose,
    )

    # 参数概览
    rprint("[bold]Phase 2: 持续自主运行[/]")
    rprint(f"  [dim]检查点:[/] {checkpoint or '无 (随机初始化)'}")
    rprint(f"  [dim]数据目录:[/] {resolved_data_dir}")
    rprint(f"  [dim]输出目录:[/] {out_dir}")
    rprint(f"  [dim]周期:[/] WAKE={wake_steps}  PLAY={play_steps}  SLEEP interval={sleep_interval}")
    rprint(f"  [dim]生成:[/] max_new={gen_max_new}  temp={gen_temperature}  top_k={gen_top_k}")
    rprint(f"  [dim]训练:[/] batch={batch_size}  seq_len={max_seq_len}  lr={lr:.2e}")
    rprint(f"  [dim]设备:[/] {device}")

    # 启动
    mind.run()
    rprint("[green]✓[/] 自主运行结束")


@app.command()
def config(
    ctx: typer.Context,
    output: str = typer.Option("autonomous_config.json", "--output", "-o", help="输出路径"),
):
    """生成 Phase 2 默认配置模板."""
    from cli.utils import save_config
    save_config(AUTONOMOUS_CONFIG_TEMPLATE, output)
    rprint(f"[green]✓[/] 自主运行配置模板: {output}")
