"""
virtuoso train — Phase 1 训练子命令.

用法:
  virtuoso train [OPTIONS]                   直接训练
  virtuoso train from-config CONFIG [OVERRIDES]  从配置文件
  virtuoso train resume CHECKPOINT           从检查点恢复

训练时显示 Rich 实时监控面板:
  ┌─ Status ──────────────────────────┐
  │ Epoch 1/1  Step 1200/5000  cuda:0 │
  ├─ Progress ────────────────────────┤
  │ ████████████░░░░░░░░░░░ 33%       │
  │ CE Loss: 6.8030                    │
  │ F (Pred): 0.28                     │
  │ Dopamine D: 0.0034                 │
  │ LR: 2.45e-04                       │
  ├─ Log ─────────────────────────────┤
  │ ✓ 检查点: out_pc_unified/step_500 │
  │ ◆ 进入 Phase 3 merge_loss         │
  └───────────────────────────────────┘
"""

import os
import sys
import json
import queue
import threading
from pathlib import Path
from typing import Optional, List

CLI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CLI_DIR)
sys.path.insert(0, PROJECT_ROOT)

import typer
from rich import print as rprint
from rich.live import Live
from rich.panel import Panel

from cli.utils import (
    PROJECT_ROOT, resolve_path, load_config, merge_config,
    RichTrainingPanel, console,
)

app = typer.Typer(name="train", help="Phase 1: 模型训练", no_args_is_help=True)


# ═══════════════════════════════════════════════════════════════════
# TrainingConfig 对应 CLI 参数 (全量)
# ═══════════════════════════════════════════════════════════════════

def _common_params(func):
    """装饰器: 注入 TrainingConfig 相关参数 (避免重复定义)."""
    # 通过 inspect 获取参数列表, 但由于 typer 需要静态参数, 
    # 这里定义共用函数以便复用
    return func


def training_options(
    model_type: str = typer.Option("pc_unified", "--model-type", help="模型类型"),
    checkpoint_path: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="初始检查点"),
    hidden_size: int = typer.Option(256, "--hidden-size", help="隐藏层维度"),
    num_hidden_layers: int = typer.Option(4, "--num-layers", help="Transformer/Pyra 层数"),
    data_files: List[str] = typer.Option([], "--data-files", "-d", help="数据文件 (可多个)"),
    combined_training: bool = typer.Option(True, "--combined/--no-combined", help="是否联合训练"),
    batch_size: int = typer.Option(48, "--batch-size", "-b", help="批大小"),
    max_seq_len: int = typer.Option(128, "--max-seq-len", help="最大序列长度"),
    lr: float = typer.Option(3e-4, "--lr", help="学习率"),
    epochs: int = typer.Option(1, "--epochs", "-e", help="训练轮数"),
    subset: int = typer.Option(0, "--subset", "-n", help="子集大小 (0=全部)"),
    T_infer: int = typer.Option(1, "--T-infer", help="Inference time steps"),
    gamma: float = typer.Option(0.1, "--gamma", help="Dopamine discount factor"),
    enable_dopamine: bool = typer.Option(True, "--dopamine/--no-dopamine", help="启用多巴胺调制"),
    dopamine_eta: float = typer.Option(1.0, "--dopamine-eta", help="Dopamine learning rate"),
    dopamine_beta: float = typer.Option(0.5, "--dopamine-beta", help="Dopamine 灵敏度"),
    dopamine_gamma: float = typer.Option(0.3, "--dopamine-gamma", help="Dopamine 衰减"),
    enable_quantize: bool = typer.Option(False, "--quantize/--no-quantize", help="启用 Int4 量化"),
    out_dir: str = typer.Option("out_pc_unified", "--out-dir", "-o", help="输出目录"),
    save_interval: int = typer.Option(500, "--save-interval", help="保存间隔步数"),
    grad_clip: float = typer.Option(1.0, "--grad-clip", help="梯度裁剪阈值"),
    use_abstraction_bank: bool = typer.Option(False, "--abstraction-bank/--no-abstraction-bank", help="启用抽象记忆库"),
    auto_start_phase2: bool = typer.Option(False, "--auto-phase2", help="训练结束后自动进入 Phase 2"),
    resume: bool = typer.Option(False, "--resume", help="从断点恢复 (自动检测 out_dir 中最新检查点)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """整理为 TrainingConfig 兼容 dict."""
    from train_manager import TrainingConfig
    return TrainingConfig(
        model_type=model_type,
        checkpoint_path=resolve_path(checkpoint_path) if checkpoint_path else None,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        data_files=[resolve_path(f) for f in data_files],
        combined_training=combined_training,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        lr=lr,
        epochs=epochs,
        subset=subset,
        T_infer=T_infer,
        gamma=gamma,
        enable_dopamine=enable_dopamine,
        dopamine_eta=dopamine_eta,
        dopamine_beta=dopamine_beta,
        dopamine_gamma=dopamine_gamma,
        enable_quantize=enable_quantize,
        out_dir=out_dir,
        save_interval=save_interval,
        use_abstraction_bank=use_abstraction_bank,
        auto_start_phase2=auto_start_phase2,
    ), grad_clip, resume, verbose


# ═══════════════════════════════════════════════════════════════════
# Rich 训练循环
# ═══════════════════════════════════════════════════════════════════

def _run_training_with_rich(
    config,
    grad_clip: float = 1.0,
    device: str = "cuda:0",
    verbose: bool = False,
):
    """在后台线程运行训练, 主线程显示 Rich 实时面板."""
    from train_manager import TrainManager

    update_queue = queue.Queue()
    error_container = []
    final_state_container = []

    # ── 回调: 线程安全地将更新送入队列 ──
    def callback(data: dict):
        update_queue.put(data)

    # ── 创建训练管理器 ──
    mgr = TrainManager(config, progress_callback=callback, verbose=verbose)

    # ── 启动训练 (后台线程) ──
    mgr.start()

    # ── Rich 面板 ──
    panel = RichTrainingPanel(device=device)

    with Live(panel.layout, refresh_per_second=4, console=console) as live:
        while mgr.is_running() or not update_queue.empty():
            try:
                data = update_queue.get(timeout=0.25)
                panel.update(data)
                live.update(panel.layout)
            except queue.Empty:
                continue

        # 清空剩余消息
        while not update_queue.empty():
            try:
                data = update_queue.get_nowait()
                panel.update(data)
            except queue.Empty:
                break

    rprint("\n[bold green]✓ 训练完成![/]")


# ═══════════════════════════════════════════════════════════════════
# 命令: virtuoso train (直接训练)
# ═══════════════════════════════════════════════════════════════════

@app.callback(invoke_without_command=True)
def train_main(
    ctx: typer.Context,
    model_type: str = typer.Option("pc_unified", "--model-type", help="模型类型"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="初始检查点"),
    hidden_size: int = typer.Option(256, "--hidden-size", help="隐藏层维度"),
    num_hidden_layers: int = typer.Option(4, "--num-layers", help="层数"),
    data_files: str = typer.Option("", "--data-files", "-d", help="数据文件 (逗号分隔)"),
    combined_training: bool = typer.Option(True, "--combined/--no-combined", help="联合训练"),
    batch_size: int = typer.Option(48, "--batch-size", "-b", help="批大小"),
    max_seq_len: int = typer.Option(128, "--max-seq-len", help="最大序列长度"),
    lr: float = typer.Option(3e-4, "--lr", help="学习率"),
    epochs: int = typer.Option(1, "--epochs", "-e", help="训练轮数"),
    subset: int = typer.Option(0, "--subset", "-n", help="子集大小 (0=全部)"),
    T_infer: int = typer.Option(1, "--T-infer", help="Inference time steps"),
    gamma: float = typer.Option(0.1, "--gamma", help="多巴胺折扣因子"),
    enable_dopamine: bool = typer.Option(True, "--dopamine/--no-dopamine", help="启用多巴胺"),
    dopamine_eta: float = typer.Option(1.0, "--dopamine-eta", help="多巴胺学习率"),
    dopamine_beta: float = typer.Option(0.5, "--dopamine-beta", help="多巴胺灵敏度"),
    dopamine_gamma: float = typer.Option(0.3, "--dopamine-gamma", help="多巴胺衰减"),
    enable_quantize: bool = typer.Option(False, "--quantize/--no-quantize", help="Int4 量化"),
    out_dir: str = typer.Option("out_pc_unified", "--out-dir", "-o", help="输出目录"),
    save_interval: int = typer.Option(500, "--save-interval", help="保存间隔"),
    grad_clip: float = typer.Option(1.0, "--grad-clip", help="梯度裁剪"),
    use_abstraction_bank: bool = typer.Option(False, "--abstraction-bank/--no-abstraction-bank", help="抽象记忆库"),
    auto_phase2: bool = typer.Option(False, "--auto-phase2", help="训练后自动进入 Phase 2"),
    resume: bool = typer.Option(False, "--resume", help="从断点恢复"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """直接训练模式: 所有参数通过 CLI 指定.

    Rich 实时面板显示: 状态栏 / 进度条 / 指标 / 日志.
    """
    if ctx.invoked_subcommand is not None:
        return  # 交给子命令

    from train_manager import TrainingConfig

    device = ctx.obj.get("device", "cuda:0")

    # 解析逗号分隔的数据文件
    data_file_list = [resolve_path(f.strip()) for f in data_files.split(",") if f.strip()]

    config = TrainingConfig(
        model_type=model_type,
        checkpoint_path=resolve_path(checkpoint) if checkpoint else None,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        data_files=data_file_list,
        combined_training=combined_training,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        lr=lr,
        epochs=epochs,
        subset=subset,
        T_infer=T_infer,
        gamma=gamma,
        enable_dopamine=enable_dopamine,
        dopamine_eta=dopamine_eta,
        dopamine_beta=dopamine_beta,
        dopamine_gamma=dopamine_gamma,
        enable_quantize=enable_quantize,
        out_dir=out_dir,
        save_interval=save_interval,
        use_abstraction_bank=use_abstraction_bank,
        auto_start_phase2=auto_phase2,
    )

    # 参数概览
    rprint("[bold]Phase 1 训练[/]")
    rprint(f"  [dim]模型:[/] PCUnified [{config.model_type}]  "
           f"hidden={config.hidden_size}  layers={config.num_hidden_layers}")
    rprint(f"  [dim]数据:[/] {len(config.data_files)} 个文件  "
           f"batch={config.batch_size}  seq_len={config.max_seq_len}")
    rprint(f"  [dim]优化:[/] lr={config.lr:.2e}  epochs={config.epochs}  "
           f"grad_clip={grad_clip}")
    rprint(f"  [dim]多巴胺:[/] {'开启' if config.enable_dopamine else '关闭'}  "
           f"η={config.dopamine_eta}  β={config.dopamine_beta}  γ={config.dopamine_gamma}")

    _run_training_with_rich(config, grad_clip=grad_clip, device=device, verbose=verbose)


# ═══════════════════════════════════════════════════════════════════
# 命令: virtuoso train from-config
# ═══════════════════════════════════════════════════════════════════

@app.command()
def from_config(
    ctx: typer.Context,
    config: str = typer.Argument(..., help="配置文件路径"),
    # 可选覆写
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="覆盖批大小"),
    lr: Optional[float] = typer.Option(None, "--lr", help="覆盖学习率"),
    epochs: Optional[int] = typer.Option(None, "--epochs", "-e", help="覆盖训练轮数"),
    out_dir: Optional[str] = typer.Option(None, "--out-dir", "-o", help="覆盖输出目录"),
    grad_clip: Optional[float] = typer.Option(None, "--grad-clip", help="覆盖梯度裁剪"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """从 JSON 配置文件加载参数并训练.

    \b
    配置文件结构:
    {
      "model": {"hidden_size": 256, "num_hidden_layers": 4},
      "data": {"data_files": ["datasets/..."], "combined_training": true},
      "training": {"batch_size": 48, "max_seq_len": 128, "lr": 3e-4, "epochs": 1},
      "pc": {"T_infer": 1, "gamma": 0.1},
      "dopamine": {"enabled": true, "eta": 1.0, "beta": 0.5, "gamma": 0.3},
      "quantize": {"enabled": false},
      "output": {"out_dir": "out_pc_unified", "save_interval": 500}
    }
    CLI 参数可以覆盖配置文件中的对应项.
    """
    from train_manager import TrainingConfig

    device = ctx.obj.get("device", "cuda:0")

    # 加载配置
    cfg = load_config(config)

    # 扁平化 → TrainingConfig
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    pc_cfg = cfg.get("pc", {})
    dop_cfg = cfg.get("dopamine", {})
    quant_cfg = cfg.get("quantize", {})
    out_cfg = cfg.get("output", {})

    config_obj = TrainingConfig(
        model_type=model_cfg.get("model_type", "pc_unified"),
        checkpoint_path=model_cfg.get("checkpoint_path"),
        hidden_size=model_cfg.get("hidden_size", 256),
        num_hidden_layers=model_cfg.get("num_hidden_layers", 4),
        data_files=[resolve_path(f) for f in data_cfg.get("data_files", [])],
        combined_training=data_cfg.get("combined_training", True),
        batch_size=train_cfg.get("batch_size", 48),
        max_seq_len=train_cfg.get("max_seq_len", 128),
        lr=train_cfg.get("lr", 3e-4),
        epochs=train_cfg.get("epochs", 1),
        subset=train_cfg.get("subset", 0),
        T_infer=pc_cfg.get("T_infer", 1),
        gamma=pc_cfg.get("gamma", 0.1),
        enable_dopamine=dop_cfg.get("enabled", True),
        dopamine_eta=dop_cfg.get("eta", 1.0),
        dopamine_beta=dop_cfg.get("beta", 0.5),
        dopamine_gamma=dop_cfg.get("gamma", 0.3),
        enable_quantize=quant_cfg.get("enabled", False),
        out_dir=out_cfg.get("out_dir", "out_pc_unified"),
        save_interval=out_cfg.get("save_interval", 500),
        use_abstraction_bank=model_cfg.get("use_abstraction_bank", False),
        auto_start_phase2=model_cfg.get("auto_start_phase2", False),
    )

    # CLI 覆写
    if batch_size is not None:
        config_obj.batch_size = batch_size
    if lr is not None:
        config_obj.lr = lr
    if epochs is not None:
        config_obj.epochs = epochs
    if out_dir is not None:
        config_obj.out_dir = out_dir

    _run_training_with_rich(
        config_obj,
        grad_clip=grad_clip or train_cfg.get("grad_clip", 1.0),
        device=device,
        verbose=verbose,
    )


# ═══════════════════════════════════════════════════════════════════
# 命令: virtuoso train resume
# ═══════════════════════════════════════════════════════════════════

@app.command()
def resume(
    ctx: typer.Context,
    checkpoint: str = typer.Argument(..., help="检查点文件 (.pt)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """从检查点文件恢复训练.

    检查点文件将自动探测对应的输出目录和配置.
    """
    from train_manager import TrainingConfig

    device = ctx.obj.get("device", "cuda:0")
    ckpt_path = resolve_path(checkpoint)

    if not os.path.exists(ckpt_path):
        rprint(f"[red]✗[/] 检查点不存在: {ckpt_path}")
        raise typer.Exit(1)

    rprint(f"[bold]恢复训练[/]")
    rprint(f"  [dim]检查点: {ckpt_path}[/]")

    # 从检查点推断输出目录
    out_dir = os.path.dirname(ckpt_path)
    out_name = os.path.basename(out_dir)

    config = TrainingConfig(
        checkpoint_path=ckpt_path,
        out_dir=out_name,
        model_type="pc_unified",
    )

    _run_training_with_rich(config, device=device, verbose=verbose)
