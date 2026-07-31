"""
autonomous

持续自主运行子命令.
"""

import torch
import click
from pkg.cli.utils import resolve_path


@click.command(name="autonomous", help="Phase 2: 持续自主运行")
@click.option("--checkpoint", "-c", default=None, help="初始检查点")
@click.option("--out-dir", "-o", default="out_autonomous", help="输出目录")
@click.option("--data-dir", "-d", default="dataset", help="数据目录")
@click.option("--wake-steps", default=20, type=int, help="WAKE 阶段步数")
@click.option("--play-steps", default=100, type=int, help="PLAY 阶段步数")
@click.option("--sleep-interval", default=500, type=int, help="SLEEP 间隔步数")
@click.option("--batch-size", "-b", default=16, type=int, help="批大小")
@click.option("--max-seq-len", default=128, type=int, help="最大序列长度")
@click.option("--gamma", default=0.05, type=float, help="多巴胺折扣因子")
@click.option("--T-infer", default=1, type=int, help="Inference time steps")
@click.option("--verbose", "-v", is_flag=True, default=False, help="详细日志")
def autonomous(
    checkpoint,
    out_dir,
    data_dir,
    wake_steps,
    play_steps,
    sleep_interval,
    batch_size,
    max_seq_len,
    gamma,
    t_infer,
    verbose,
):
    from model.core.autonomous_mind import AutonomousMind

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_checkpoint = resolve_path(checkpoint) if checkpoint else None
    resolved_data_dir = resolve_path(data_dir)

    cfg = {
        "out_dir": out_dir,
        "data_dir": resolved_data_dir,
        "wake_steps": wake_steps,
        "play_steps": play_steps,
        "sleep_interval": sleep_interval,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "gamma": gamma,
        "T_infer": t_infer,
    }
    if resolved_checkpoint:
        cfg["checkpoint"] = resolved_checkpoint

    mind = AutonomousMind(cfg=cfg)

    print("持续自主运行")
    print(f"  检查点: {checkpoint or '随机初始化'}")
    print(f"  数据: {resolved_data_dir}  输出: {out_dir}")
    print(f"  周期: WAKE={wake_steps}  PLAY={play_steps}  SLEEP interval={sleep_interval}")
    print(f"  训练: batch={batch_size}  seq_len={max_seq_len}")
    print(f"  设备: {device}")

    mind.run()  # type: ignore[attr-defined]
    print("✓ 自主运行结束")
