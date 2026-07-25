"""eval — 模型评估子命令 (委托至 Cyrilux.core.evaluation).
"""

from typing import Optional

import typer

from pkg.cli.utils import resolve_path

app = typer.Typer(name="eval", help="模型评估命令", no_args_is_help=True)


@app.command()
def all(
    ctx: typer.Context,
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="Unified 检查点路径"),
    device: str = typer.Option("cuda:0", "--device", "-d", help="计算设备"),
):
    """全面评估: 自监督指标 + Perplexity + 文本生成."""

    from model.core.evaluation import run_full_evaluation
    from pkg.utils.trainer_utils import setup_seed
    from model.model_cyrene import CyreneConfig, CyreneModel

    setup_seed(42)
    print(f"全面评估 — 检查点: {checkpoint or '默认'}  设备: {device}")

    runner = CyreneModel(CyreneConfig(hidden_size=64, warmup_steps=50))

    # 如果指定了检查点, 尝试加载 (TODO: checkpoint save/load for sparse)
    if checkpoint:
        ckpt_path = resolve_path(checkpoint)
        print(f"检查点加载未实现 (稀疏结构): {ckpt_path}")

    from model.core.evaluation import create_eval_runner_loader
    loader = create_eval_runner_loader("dataset/sft_t2t.jsonl", max_length=128, max_samples=200)
    run_full_evaluation(
        runner, loader,
        max_batches=20,
        prompts=["人工智能的未来在于", "小明今天去了公园", "深度学习是一种"],
    )


@app.command()
def language(
    ctx: typer.Context,
    checkpoint: str = typer.Argument(..., help="检查点路径"),
    local: bool = typer.Option(False, "--local", help="使用 Conv1 骨干网络"),
    device: str = typer.Option("cuda:0", "--device", "-d", help="计算设备"),
):
    """语言能力评估: Perplexity + 文本生成."""

    from model.core.evaluation import create_eval_runner_loader, run_full_evaluation
    from pkg.utils.trainer_utils import setup_seed
    from model.model_cyrene import CyreneConfig, CyreneModel

    setup_seed(42)
    ckpt_path = resolve_path(checkpoint)
    print(f"语言评估 — 检查点: {ckpt_path}  设备: {device}")

    runner = CyreneModel(CyreneConfig(hidden_size=64, warmup_steps=50))

    loader = create_eval_runner_loader("dataset/sft_t2t.jsonl", max_length=128, max_samples=500)
    run_full_evaluation(
        runner, loader,
        max_batches=20,
        prompts=["人工智能的未来在于", "小明今天去了公园", "深度学习是一种"],
    )
