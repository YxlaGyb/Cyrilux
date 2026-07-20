"""
virtuoso autonomous — Phase 2 持续自主运行子命令 (委托至 virtuosov2.core.autonomous_mind).

实现 WAKE → PLAY → SLEEP 循环.
"""

import os
from typing import Optional

import typer
from pkg.cli.utils import resolve_path
from model.core.globals import DEVICE_STR

app = typer.Typer(name="autonomous", help="Phase 2: 持续自主运行", no_args_is_help=True)


@app.callback(invoke_without_command=True)
def autonomous(
    ctx: typer.Context,
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="初始检查点"),
    out_dir: str = typer.Option("out_autonomous", "--out-dir", "-o", help="输出目录"),
    data_dir: str = typer.Option("dataset", "--data-dir", "-d", help="数据目录"),

    wake_steps: int = typer.Option(20, "--wake-steps", help="WAKE 阶段步数"),
    play_steps: int = typer.Option(100, "--play-steps", help="PLAY 阶段步数"),
    sleep_interval: int = typer.Option(500, "--sleep-interval", help="SLEEP 间隔步数"),

    batch_size: int = typer.Option(16, "--batch-size", "-b", help="批大小"),
    max_seq_len: int = typer.Option(128, "--max-seq-len", help="最大序列长度"),
    gamma: float = typer.Option(0.05, "--gamma", help="多巴胺折扣因子"),
    T_infer: int = typer.Option(1, "--T-infer", help="Inference time steps"),

    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """Phase 2: 持续自主运行 (WAKE→PLAY→SLEEP 循环)."""
    from model.core.autonomous_mind import AutonomousMind

    device = ctx.obj.get("device", DEVICE_STR)
    resolved_checkpoint = resolve_path(checkpoint) if checkpoint else None
    resolved_data_dir = resolve_path(data_dir)

    cfg = {
        'out_dir': out_dir,
        'data_dir': resolved_data_dir,
        'wake_steps': wake_steps,
        'play_steps': play_steps,
        'sleep_interval': sleep_interval,
        'batch_size': batch_size,
        'max_seq_len': max_seq_len,
        'gamma': gamma,
        'T_infer': T_infer,
    }
    if resolved_checkpoint:
        cfg['checkpoint'] = resolved_checkpoint

    mind = AutonomousMind(cfg=cfg)

    print(f"Phase 2: 持续自主运行")
    print(f"  检查点: {checkpoint or '随机初始化'}")
    print(f"  数据: {resolved_data_dir}  输出: {out_dir}")
    print(f"  周期: WAKE={wake_steps}  PLAY={play_steps}  SLEEP interval={sleep_interval}")
    print(f"  训练: batch={batch_size}  seq_len={max_seq_len}")
    print(f"  设备: {device}")

    mind.run()
    print("✓ 自主运行结束")
