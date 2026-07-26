"""
train
训练子命令
"""

import os
from typing import Optional, List

import torch

import typer
from pkg.cli.utils import resolve_path, load_config

app = typer.Typer(name="train", help="Phase 1: 模型训练", no_args_is_help=True)


def training_options(
    model_type: str = typer.Option("pc_unified", "--model-type", help="模型类型"),
    hidden_size: int = typer.Option(256, "--hidden-size", help="隐藏层维度"),
    num_hidden_layers: int = typer.Option(
        4, "--num-layers", help="Transformer/Pyra 层数"
    ),
    data_files: List[str] = typer.Option(
        [], "--data-files", "-d", help="数据文件 (可多个)"
    ),
    combined_training: bool = typer.Option(
        True, "--combined/--no-combined", help="是否联合训练"
    ),
    batch_size: int = typer.Option(48, "--batch-size", "-b", help="批大小"),
    max_seq_len: int = typer.Option(128, "--max-seq-len", help="最大序列长度"),
    lr: float = typer.Option(3e-4, "--lr", help="学习率"),
    epochs: int = typer.Option(1, "--epochs", "-e", help="训练轮数"),
    subset: int = typer.Option(0, "--subset", "-n", help="子集大小 (0=全部)"),
    T_infer: int = typer.Option(1, "--T-infer", help="Inference time steps"),
    gamma: float = typer.Option(0.1, "--gamma", help="Dopamine discount factor"),
    enable_dopamine: bool = typer.Option(
        True, "--dopamine/--no-dopamine", help="启用多巴胺调制"
    ),
    dopamine_eta: float = typer.Option(
        1.0, "--dopamine-eta", help="Dopamine learning rate"
    ),
    dopamine_beta: float = typer.Option(0.5, "--dopamine-beta", help="Dopamine 灵敏度"),
    dopamine_gamma: float = typer.Option(0.3, "--dopamine-gamma", help="Dopamine 衰减"),
    out_dir: str = typer.Option("out_pc_unified", "--out-dir", "-o", help="输出目录"),
    save_interval: int = typer.Option(500, "--save-interval", help="保存间隔步数"),
    use_abstraction_bank: bool = typer.Option(
        False, "--abstraction-bank/--no-abstraction-bank", help="启用抽象记忆库"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """整理参数字典."""
    return dict(
        model_type=model_type,
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
        out_dir=out_dir,
        save_interval=save_interval,
        use_abstraction_bank=use_abstraction_bank,
    ), verbose


@app.callback(invoke_without_command=True)
def train_main(
    ctx: typer.Context,
    model_type: str = typer.Option("pc_unified", "--model-type", help="模型类型"),
    checkpoint: Optional[str] = typer.Option(
        None, "--checkpoint", "-c", help="初始检查点"
    ),
    hidden_size: int = typer.Option(256, "--hidden-size", help="隐藏层维度"),
    num_hidden_layers: int = typer.Option(4, "--num-layers", help="层数"),
    data_files: str = typer.Option(
        "", "--data-files", "-d", help="数据文件 (逗号分隔)"
    ),
    combined_training: bool = typer.Option(
        True, "--combined/--no-combined", help="联合训练"
    ),
    batch_size: int = typer.Option(48, "--batch-size", "-b", help="批大小"),
    max_seq_len: int = typer.Option(128, "--max-seq-len", help="最大序列长度"),
    lr: float = typer.Option(3e-4, "--lr", help="学习率"),
    epochs: int = typer.Option(1, "--epochs", "-e", help="训练轮数"),
    subset: int = typer.Option(0, "--subset", "-n", help="子集大小 (0=全部)"),
    T_infer: int = typer.Option(1, "--T-infer", help="Inference time steps"),
    gamma: float = typer.Option(0.1, "--gamma", help="多巴胺折扣因子"),
    enable_dopamine: bool = typer.Option(
        True, "--dopamine/--no-dopamine", help="启用多巴胺"
    ),
    dopamine_eta: float = typer.Option(1.0, "--dopamine-eta", help="多巴胺学习率"),
    dopamine_beta: float = typer.Option(0.5, "--dopamine-beta", help="多巴胺灵敏度"),
    dopamine_gamma: float = typer.Option(0.3, "--dopamine-gamma", help="多巴胺衰减"),
    out_dir: str = typer.Option("out_pc_unified", "--out-dir", "-o", help="输出目录"),
    save_interval: int = typer.Option(500, "--save-interval", help="保存间隔"),
    use_abstraction_bank: bool = typer.Option(
        False, "--abstraction-bank/--no-abstraction-bank", help="抽象记忆库"
    ),
    auto_phase2: bool = typer.Option(
        False, "--auto-phase2", help="训练后自动进入 Phase 2"
    ),
    resume: bool = typer.Option(False, "--resume", help="从断点恢复"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """直接训练模式 (纯 Hebbian, 零反向传播)."""
    if ctx.invoked_subcommand is not None:
        return

    from model.core.train import TrainingLoop, TrainingConfig
    from model.core.dataset import DualChannelDataset

    device = ctx.obj.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    data_file_list = [
        resolve_path(f.strip()) for f in data_files.split(",") if f.strip()
    ]

    config = TrainingConfig(  # type: ignore[call-arg]
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
        out_dir=out_dir,
        save_interval=save_interval,
        use_abstraction_bank=use_abstraction_bank,
    )

    print(
        f"Phase 1 训练 — {config.model_type}  hidden={config.hidden_size}  "
        f"layers={config.num_hidden_layers}  device={device}"
    )

    # 构建任务管道
    task_pipelines = []
    for i, fp in enumerate(data_file_list):
        tid = f"task_{i}"
        ds = DualChannelDataset(
            fp, max_length=max_seq_len, max_samples=subset if subset else None  # type: ignore[arg-type]
        )
        task_pipelines.append((tid, ds))

    loop = TrainingLoop(config)
    loop.train(task_pipelines)


@app.command()
def from_config(
    ctx: typer.Context,
    config: str = typer.Argument(..., help="配置文件路径"),
    # 可选覆写
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", "-b", help="覆盖批大小"
    ),
    lr: Optional[float] = typer.Option(None, "--lr", help="覆盖学习率"),
    epochs: Optional[int] = typer.Option(None, "--epochs", "-e", help="覆盖训练轮数"),
    out_dir: Optional[str] = typer.Option(None, "--out-dir", "-o", help="覆盖输出目录"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """从 JSON 配置文件加载参数并训练 (纯 Hebbian, 零反向传播)."""
    from model.core.train import TrainingLoop, TrainingConfig
    from model.core.dataset import DualChannelDataset

    cfg = load_config(config)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    pc_cfg = cfg.get("pc", {})
    dop_cfg = cfg.get("dopamine", {})
    out_cfg = cfg.get("output", {})

    data_file_list = [resolve_path(f) for f in data_cfg.get("data_files", [])]

    config_obj = TrainingConfig(  # type: ignore[call-arg]
        model_type=model_cfg.get("model_type", "pc_unified"),
        checkpoint_path=model_cfg.get("checkpoint_path"),
        hidden_size=model_cfg.get("hidden_size", 256),
        num_hidden_layers=model_cfg.get("num_hidden_layers", 4),
        data_files=data_file_list,
        combined_training=data_cfg.get("combined_training", True),
        batch_size=batch_size or train_cfg.get("batch_size", 48),
        max_seq_len=train_cfg.get("max_seq_len", 128),
        lr=lr or train_cfg.get("lr", 3e-4),
        epochs=epochs or train_cfg.get("epochs", 1),
        subset=train_cfg.get("subset", 0),
        T_infer=pc_cfg.get("T_infer", 1),
        gamma=pc_cfg.get("gamma", 0.1),
        enable_dopamine=dop_cfg.get("enabled", True),
        dopamine_eta=dop_cfg.get("eta", 1.0),
        dopamine_beta=dop_cfg.get("beta", 0.5),
        dopamine_gamma=dop_cfg.get("gamma", 0.3),
        out_dir=out_cfg.get("out_dir", "out_pc_unified"),
        save_interval=out_cfg.get("save_interval", 500),
        use_abstraction_bank=model_cfg.get("use_abstraction_bank", False),
    )

    print(f"从配置启动训练: {config}")

    # 构建任务管道
    task_pipelines = []
    for i, fp in enumerate(data_file_list):
        tid = f"task_{i}"
        ds = DualChannelDataset(
            fp,
            max_length=config_obj.max_seq_len,
            max_samples=config_obj.subset if config_obj.subset else None,  # type: ignore[arg-type]
        )
        task_pipelines.append((tid, ds))

    loop = TrainingLoop(config_obj)
    loop.train(task_pipelines)


@app.command()
def resume(
    ctx: typer.Context,
    checkpoint: str = typer.Argument(..., help="检查点文件 (.pt)"),
    batch_size: int = typer.Option(48, "--batch-size", "-b", help="批大小"),
    max_seq_len: int = typer.Option(128, "--max-seq-len", help="最大序列长度"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """从检查点文件恢复训练 (纯 Hebbian, 零反向传播)."""
    from model.core.train import TrainingLoop, TrainingConfig
    from model.core.dataset import DualChannelDataset

    ckpt_path = resolve_path(checkpoint)

    if not os.path.exists(ckpt_path):
        print(f"✗ 检查点不存在: {ckpt_path}")
        raise typer.Exit(1)

    print(f"恢复训练 — 检查点: {ckpt_path}")
    out_dir = os.path.dirname(ckpt_path)
    out_name = os.path.basename(out_dir)

    config = TrainingConfig(  # type: ignore[call-arg]
        checkpoint_path=ckpt_path,
        out_dir=out_name,
        model_type="pc_unified",
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )

    # 从检查点元数据推断数据文件
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    data_files = []
    if "config" in ckpt_data and "data_files" in ckpt_data["config"]:
        data_files = ckpt_data["config"]["data_files"]
    if not data_files:
        print("⚠ 检查点中无 data_files 信息, 请使用 `train` 命令指定数据")
        # 创建空管道 — TrainingLoop 仍能加载模型权重
        task_pipelines = []
    else:
        task_pipelines = []
        for i, fp in enumerate(data_files):
            tid = f"task_{i}"
            resolved = resolve_path(fp)
            ds = DualChannelDataset(resolved, max_length=max_seq_len, max_samples=None)
            task_pipelines.append((tid, ds))

    loop = TrainingLoop(config)
    if task_pipelines:
        loop.train(task_pipelines)
    else:
        print("Model loaded. 请提供数据文件继续训练.")
