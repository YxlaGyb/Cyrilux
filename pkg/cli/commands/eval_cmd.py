"""eval — 模型评估子命令 (委托至 model.core.evaluation).
"""

from typing import Optional

import typer

from pkg.cli.utils import resolve_path

app = typer.Typer(name="eval", help="模型评估命令", no_args_is_help=True)


@app.command()
def all(
    ctx: typer.Context,
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="检查点路径"),
    device: str = typer.Option("cuda:0", "--device", "-d", help="计算设备"),
):
    """全面评估: Perplexity + 文本生成."""

    from model.core.evaluation import create_eval_runner_loader, run_full_evaluation
    from model.model_cyrene import CyreneConfig, CyreneModel

    print(f"全面评估 — 检查点: {checkpoint or '默认'}  设备: {device}")

    if checkpoint:
        runner = CyreneModel.load(resolve_path(checkpoint))
    else:
        runner = CyreneModel(CyreneConfig(hidden_size=64, warmup_steps=50))
        runner.add_hidden_layer(n_neurons=256, from_layer=0, to_layer=7, connection_density=0.2)
        runner.warmup(20)

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
    from model.model_cyrene import CyreneModel

    ckpt_path = resolve_path(checkpoint)
    print(f"语言评估 — 检查点: {ckpt_path}  设备: {device}")

    runner = CyreneModel.load(ckpt_path)

    loader = create_eval_runner_loader("dataset/sft_t2t.jsonl", max_length=128, max_samples=500)
    run_full_evaluation(
        runner, loader,
        max_batches=20,
        prompts=["人工智能的未来在于", "小明今天去了公园", "深度学习是一种"],
    )
