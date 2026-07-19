#!/usr/bin/env python3
"""
virtuosov2 v0.2.0 — 统一入口

用法:
    uv run python main.py gui          # Tkinter 桌面训练 GUI
    uv run python main.py train ...    # CLI 训练
    uv run python main.py eval ...     # 评估
    uv run python main.py auto ...     # 持续自主运行

子命令参考:
    uv run python main.py train --help
    uv run python main.py eval --help
    uv run python main.py auto --help
"""

import os, sys, json, argparse

# ── 包导入 ────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)  # 仅用于 main.py 自身能被执行

from core.trainer_utils import setup_seed, Logger
from core.training import TrainingLoop, TrainingConfig
from core.threaded_trainer import ThreadedTrainer, run_training_standalone
from core.evaluation import (
    compute_perplexity,
    generate_text,
    eval_self_supervised,
    load_with_remap,
    create_eval_loader,
    run_full_evaluation,
)
from core.autonomous_mind import AutonomousMind, DEFAULT_CFG
from core.data_splitter import split_file, split_directory
from core.data_converter import scan_dataset_dir, convert_file
from core.prepare_tasks import prepare_4tasks, prepare_hetero
from model.pc_layers import PCLocalDynamicMiniMind, load_pc_checkpoint
from model.model_minimind import MiniMindConfig


# ═══════════════════════════════════════════════════════════════════
# 子命令: 训练
# ═══════════════════════════════════════════════════════════════════


def _build_common_parser(subparser):
    """子命令共用参数。"""
    subparser.add_argument("--batch-size", type=int, default=48)
    subparser.add_argument("--max-seq-len", type=int, default=128)
    subparser.add_argument("--lr", type=float, default=3e-4)
    subparser.add_argument("--epochs", type=int, default=1)
    subparser.add_argument("--subset", type=int, default=0)
    subparser.add_argument("--seed", type=int, default=42)
    subparser.add_argument("--hidden-size", type=int, default=256)
    subparser.add_argument("--num-hidden-layers", type=int, default=4)
    subparser.add_argument(
        "--checkpoint", type=str, default=None, help="恢复训练的 checkpoint 路径"
    )
    subparser.add_argument("--out-dir", type=str, default="out_pc_unified")
    return subparser


def _build_pc_parser(subparser):
    """PC / 多巴胺参数。"""
    subparser.add_argument("--T-infer", type=int, default=2)
    subparser.add_argument("--gamma", type=float, default=0.1)
    subparser.add_argument("--dopamine-eta", type=float, default=1.0)
    subparser.add_argument("--dopamine-beta", type=float, default=0.5)
    subparser.add_argument("--dopamine-gamma", type=float, default=0.3)
    return subparser


def _build_cl_parser(subparser):
    """持续学习参数。"""
    subparser.add_argument("--replay-ratio", type=int, default=5)
    subparser.add_argument("--bank-size", type=int, default=2000)
    subparser.add_argument("--sniff-interval", type=int, default=200)
    subparser.add_argument("--repair-threshold", type=float, default=1.2)
    subparser.add_argument("--repair-steps", type=int, default=10)
    subparser.add_argument("--eval-samples", type=int, default=100)
    subparser.add_argument("--n-prototypes", type=int, default=8)
    subparser.add_argument("--abstraction-replay-interval", type=int, default=200)
    return subparser


def register_train_parser(subparsers):
    p = subparsers.add_parser("train", help="训练模型")
    _build_common_parser(p)
    _build_pc_parser(p)
    _build_cl_parser(p)

    p.add_argument("data", nargs="+", help="数据文件路径")
    p.add_argument(
        "--task-order",
        type=str,
        default="a",
        help="任务顺序: 逗号分隔, 每项对应一个数据文件 (默认单任务 'a')",
    )
    p.add_argument("--save-interval", type=int, default=500, help="中间检查点间隔步数")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--split-size", type=int, default=0)


def cmd_train(args):
    """执行训练 (CLI 模式)。"""
    setup_seed(args.seed)

    # 构建配置
    cfg = TrainingConfig(
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        lr=args.lr,
        epochs=args.epochs,
        subset=args.subset,
        seed=args.seed,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        T_infer=args.T_infer,
        gamma=args.gamma,
        dopamine_eta=args.dopamine_eta,
        dopamine_beta=args.dopamine_beta,
        dopamine_gamma=args.dopamine_gamma,
        replay_ratio=args.replay_ratio,
        bank_size=args.bank_size,
        sniff_interval=args.sniff_interval,
        repair_threshold=args.repair_threshold,
        repair_steps=args.repair_steps,
        eval_samples=args.eval_samples,
        n_prototypes=args.n_prototypes,
        abstraction_replay_interval=args.abstraction_replay_interval,
        save_interval=args.save_interval,
    )

    # 任务流水线: 每个 task_id 对应一个数据文件
    task_order = args.task_order.split(",")
    task_ids = ["a", "b", "c", "d", "e"]
    pipelines = []
    for i, tid in enumerate(task_order):
        if tid in task_ids:
            data_files = [args.data[i]] if i < len(args.data) else args.data
            pipelines.append((tid, data_files, args.subset if args.subset > 0 else None))

    # 启动训练
    run_training_standalone(cfg, pipelines)


# ═══════════════════════════════════════════════════════════════════
# 子命令: 评估
# ═══════════════════════════════════════════════════════════════════


def register_eval_parser(subparsers):
    p = subparsers.add_parser("eval", help="评估模型")
    p.add_argument("checkpoint", type=str, help="检查点路径")
    p.add_argument("--data", type=str, default="dataset/sft_t2t.jsonl", help="评估数据")
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--max-batches", type=int, default=20)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--T", type=int, default=2)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["人工智能的未来在于", "小明今天去了公园，他看到", "深度学习是一种"],
    )
    # QAT related args removed


def cmd_eval(args):
    """执行评估。"""
    setup_seed(42)

    # 加载模型
    lm_cfg = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        use_moe=False,
    )
    model = PCLocalDynamicMiniMind(lm_cfg)
    model = load_with_remap(model, args.checkpoint)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    Logger(
        f"Model loaded from {args.checkpoint} ({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M)"
    )

    # 数据
    loader = create_eval_loader(
        args.data,
        max_length=args.max_seq_len,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )

    # 位置编码
    pos = model.get_position_embeddings(args.max_seq_len, device)

    # 运行评估
    models_dict = {"model": (model, pos)}
    run_full_evaluation(
        models_dict,
        loader,
        gamma=args.gamma,
        T=args.T,
        max_batches=args.max_batches,
        prompts=args.prompts,
    )


# ═══════════════════════════════════════════════════════════════════
# 子命令: 自主运行
# ═══════════════════════════════════════════════════════════════════


def register_auto_parser(subparsers):
    p = subparsers.add_parser("auto", help="持续自主运行 (WAKE→PLAY→SLEEP)")
    p.add_argument("--checkpoint", type=str, default=None, help="初始检查点")
    p.add_argument("--out-dir", type=str, default="out_autonomous")
    p.add_argument("--wake-steps", type=int, default=20)
    p.add_argument("--play-steps", type=int, default=100)
    p.add_argument("--sleep-interval", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--T-infer", type=int, default=1)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--data-dir", type=str, default="dataset")


def cmd_auto(args):
    """启动持续自主运行。"""
    cfg = {
        **DEFAULT_CFG,
        "checkpoint": args.checkpoint,
        "out_dir": args.out_dir,
        "wake_steps": args.wake_steps,
        "play_steps": args.play_steps,
        "sleep_interval": args.sleep_interval,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "T_infer": args.T_infer,
        "data_dir": args.data_dir,
    }
    lm_cfg = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        use_moe=False,
    )
    mind = AutonomousMind(lm_config=lm_cfg, cfg=cfg)
    mind.run_forever()


# ═══════════════════════════════════════════════════════════════════
# 子命令: GUI
# ═══════════════════════════════════════════════════════════════════


def register_gui_parser(subparsers):
    p = subparsers.add_parser("gui", help="启动 Tkinter 桌面训练 GUI")
    p.add_argument(
        "--dpi-scale", type=float, default=1.25, help="DPI 缩放系数 (默认 1.25=125%%)"
    )


def cmd_gui(args):
    """启动 Tkinter 桌面 GUI。"""
    launch_gui(dpi_scale=args.dpi_scale)


def launch_gui(dpi_scale: float = 1.25):
    """启动 PyQt6 骇客像素风 GUI."""
    from gui import launch_gui as _launch

    _launch()


# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="virtuosov2 v0.2.0 — Predictive Coding 本地动态小语言模型",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    register_train_parser(subparsers)
    register_eval_parser(subparsers)
    register_auto_parser(subparsers)
    register_gui_parser(subparsers)

    # data / prepare / split / convert 等工具子命令
    p_data = subparsers.add_parser("prepare", help="准备任务数据")
    p_data.add_argument(
        "--4task", action="store_true", default=False, help="准备 4 任务 (a,b,c,d)"
    )
    p_data.add_argument(
        "--hetero", action="store_true", default=False, help="准备异构任务"
    )
    p_data.add_argument("--out-dir", type=str, default=None)

    p_split = subparsers.add_parser("split", help="分割数据集")
    p_split.add_argument("input", type=str, help="输入文件或目录")
    p_split.add_argument("--train-ratio", type=float, default=0.9)
    p_split.add_argument("--val-ratio", type=float, default=0.05)

    p_convert = subparsers.add_parser("convert", help="转换数据集格式")
    p_convert.add_argument("input", type=str, help="输入文件或目录")
    p_convert.add_argument("--max-samples", type=int, default=0)

    p_list = subparsers.add_parser("list", help="列出数据集")
    p_list.add_argument("--dir", type=str, default="dataset")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "auto":
        cmd_auto(args)
    elif args.command == "gui":
        cmd_gui(args)
    elif args.command == "prepare":
        if args._4task:
            prepare_4tasks(out_dir=args.out_dir)
        if args.hetero:
            prepare_hetero(out_dir=args.out_dir)
    elif args.command == "split":
        if os.path.isdir(args.input):
            split_directory(
                args.input, train_ratio=args.train_ratio, val_ratio=args.val_ratio
            )
        else:
            split_file(
                args.input, train_ratio=args.train_ratio, val_ratio=args.val_ratio
            )
    elif args.command == "convert":
        if os.path.isdir(args.input):
            scan_dataset_dir(args.input, max_samples=args.max_samples or None)
        else:
            convert_file(args.input, max_samples=args.max_samples or None)
    elif args.command == "list":
        dataset_dir = args.dir
        if os.path.isdir(dataset_dir):
            files = [f for f in os.listdir(dataset_dir) if f.endswith(".jsonl")]
            print(f"\n数据集目录: {dataset_dir}  ({len(files)} 个文件)\n")
            for f in sorted(files):
                fpath = os.path.join(dataset_dir, f)
                size_kb = os.path.getsize(fpath) / 1024
                print(f"  {f:40s} {size_kb:>8.1f} KB")
        else:
            print(f"目录不存在: {dataset_dir}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
