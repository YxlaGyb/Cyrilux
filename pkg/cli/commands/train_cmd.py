"""
train
"""

import os

import click
import torch

from pkg.cli.utils import load_config, resolve_path


def _dense_cfg(hidden_size: int, lr: float):
    from model.pc.dense import DensePCConfig, DensePCNet

    ratio = hidden_size / 1024.0
    d_cfg = DensePCConfig(
        d_l4=hidden_size,
        d_l2=max(64, int(384 * ratio)),
        d_l3=max(64, int(384 * ratio)),
        d_l5=max(64, int(256 * ratio)),
        d_l6=max(64, int(128 * ratio)),
        lr_hebbian=lr,
    )
    return DensePCNet(d_cfg), d_cfg


def _train_dense(data_files, out_dir, batch_size, max_seq_len, lr, epochs, hidden_size, device, subset):
    """dense"""
    import json as _json

    from tqdm import tqdm

    torch.set_grad_enabled(False)
    net, d_cfg = _dense_cfg(hidden_size, lr)
    net = net.to(device)
    print(f"PPA dense training  L4={d_cfg.d_l4}  params={d_cfg.param_count():,}  device={device}")

    step = 0
    fe_sum = 0.0
    batch_ids: list = []
    for epoch in range(epochs):
        for fp in data_files:
            fname = os.path.basename(fp)
            # 先扫总行数 (一次, 秒级), 供 tqdm 显示百分比/剩余时间
            with open(fp, "r", encoding="utf-8") as f:
                total = sum(1 for _ in f)
            n_lines = 0
            with open(fp, "r", encoding="utf-8") as f:
                pbar = tqdm(total=total, desc=f"    {fname}", unit="it")
                for line in f:
                    try:
                        s = _json.loads(line)
                        t = s.get("text", "")
                        if "conversations" in s:
                            t = "".join(m.get("content", m.get("value", "")) for m in s["conversations"])
                        bs = t.encode("utf-8")[:max_seq_len]
                        ids = torch.zeros(max_seq_len, dtype=torch.long)
                        ids[: len(bs)] = torch.tensor(list(bs), dtype=torch.long)
                        batch_ids.append(ids)
                    except Exception:
                        continue
                    n_lines += 1
                    pbar.update(1)
                    if len(batch_ids) < batch_size:
                        continue
                    b = torch.stack(batch_ids).to(device)
                    stats = net.learn(b)
                    step += 1
                    fe_sum += float(stats["future_err"])
                    batch_ids = []
                    pbar.set_postfix({"step": step, "fe": f"{fe_sum / step:.4f}"})
                pbar.close()
            print(f"    {fname} 全文件跑完: {n_lines} 行")
    os.makedirs(out_dir, exist_ok=True)
    net.save(os.path.join(out_dir, "final.pt"))
    print(f"PPA model saved to {out_dir}/final.pt  final_free_energy_avg={fe_sum / max(step, 1):.4f}")


def _train_sparse(
    data_files,
    out_dir,
    batch_size,
    max_seq_len,
    lr,
    epochs,
    subset,
    hidden_size,
    save_interval,
    device,
    checkpoint=None,
    num_hidden_layers=4,
    T_infer=1,
    gamma=0.1,
    dopamine_eta=1.0,
    dopamine_beta=0.5,
    dopamine_gamma=0.3,
):
    """sparse"""
    from model.core.dataset import DualChannelDataset
    from model.core.train import TrainingConfig, TrainingLoop

    config = TrainingConfig(
        checkpoint_path=resolve_path(checkpoint) if checkpoint else None,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        lr=lr,
        epochs=epochs,
        subset=subset,
        T_infer=T_infer,
        gamma=gamma,
        dopamine_eta=dopamine_eta,
        dopamine_beta=dopamine_beta,
        dopamine_gamma=dopamine_gamma,
        out_dir=out_dir,
        save_interval=save_interval,
    )
    print(f"Phase 1 Training  hidden={config.hidden_size}  device={device}")
    task_pipelines = []
    for i, fp in enumerate(data_files):
        ds = DualChannelDataset(fp, max_length=max_seq_len, max_samples=subset or None)
        task_pipelines.append((f"task_{i}", ds))
    loop = TrainingLoop(config)
    loop.train(task_pipelines)


@click.command(name="train", help="直接训练模式 (纯 Hebbian, 零反向传播)")
@click.option("--model-type", default="pc_unified", help="模型类型")
@click.option("--checkpoint", "-c", default=None, help="初始检查点")
@click.option("--hidden-size", default=256, type=int, help="隐藏层维度")
@click.option("--num-layers", default=4, type=int, help="层数")
@click.option("--data", "-d", multiple=True, help="数据文件 (可多个)")
@click.option("--combined/--no-combined", default=True, help="是否联合训练")
@click.option("--batch-size", "-b", default=48, type=int, help="批大小")
@click.option("--max-seq-len", default=128, type=int, help="最大序列长度")
@click.option("--lr", default=3e-4, type=float, help="学习率")
@click.option("--epochs", "-e", default=1, type=int, help="训练轮数")
@click.option("--subset", "-n", default=0, type=int, help="子集大小 (0=全部)")
@click.option("--T-infer", default=1, type=int, help="Inference time steps")
@click.option("--gamma", default=0.1, type=float, help="多巴胺折扣因子")
@click.option("--dopamine/--no-dopamine", default=True, help="启用多巴胺")
@click.option("--dopamine-eta", default=1.0, type=float, help="多巴胺学习率")
@click.option("--dopamine-beta", default=0.5, type=float, help="多巴胺灵敏度")
@click.option("--dopamine-gamma", default=0.3, type=float, help="多巴胺衰减")
@click.option("--out-dir", "-o", default="out_pc_unified", help="输出目录")
@click.option("--save-interval", default=10000, type=int, help="保存间隔")
@click.option("--abstraction-bank/--no-abstraction-bank", default=False, help="抽象记忆库")
@click.option("--auto-phase2", is_flag=True, default=False, help="训练后自动进入 Phase 2")
@click.option("--resume", is_flag=True, default=False, help="从断点恢复")
@click.option("--backend", default="sparse", help="sparse=event-driven(default) / dense=full-GPU matmul")
@click.option("--verbose", "-v", is_flag=True, default=False, help="verbose logging")
def train(
    model_type,
    checkpoint,
    hidden_size,
    num_layers,
    data,
    combined,
    batch_size,
    max_seq_len,
    lr,
    epochs,
    subset,
    t_infer,
    gamma,
    dopamine,
    dopamine_eta,
    dopamine_beta,
    dopamine_gamma,
    out_dir,
    save_interval,
    abstraction_bank,
    auto_phase2,
    resume,
    backend,
    verbose,
):

    if not data:
        raise click.ClickException("请用 -d 指定数据文件 (可多个)")
    data_files = [resolve_path(f.strip()) for f in data]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if backend == "dense":
        _train_dense(data_files, out_dir, batch_size, max_seq_len, lr, epochs, hidden_size, device, subset)
        return

    _train_sparse(
        data_files,
        out_dir,
        batch_size,
        max_seq_len,
        lr,
        epochs,
        subset,
        hidden_size,
        save_interval,
        device,
        checkpoint=checkpoint,
        num_hidden_layers=num_layers,
        T_infer=t_infer,
        gamma=gamma,
        dopamine_eta=dopamine_eta,
        dopamine_beta=dopamine_beta,
        dopamine_gamma=dopamine_gamma,
    )


@click.command(name="from-config", help="从 JSON 配置文件加载参数并训练 (纯 Hebbian, 零反向传播)")
@click.argument("config", type=click.Path(exists=True))
@click.option("--batch-size", "-b", type=int, default=None, help="覆盖批大小")
@click.option("--lr", type=float, default=None, help="覆盖学习率")
@click.option("--epochs", "-e", type=int, default=None, help="覆盖训练轮数")
@click.option("--out-dir", "-o", default=None, help="覆盖输出目录")
@click.option("--verbose", "-v", is_flag=True, default=False, help="详细日志")
def from_config(config, batch_size, lr, epochs, out_dir, verbose):
    """从 JSON 配置文件加载参数并训练 (纯 Hebbian, 零反向传播)."""
    from model.core.dataset import DualChannelDataset
    from model.core.train import TrainingConfig, TrainingLoop

    cfg = load_config(config)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    pc_cfg = cfg.get("pc", {})
    dop_cfg = cfg.get("dopamine", {})
    out_cfg = cfg.get("output", {})

    data_file_list = [resolve_path(f) for f in data_cfg.get("data_files", [])]
    config_obj = TrainingConfig(
        checkpoint_path=model_cfg.get("checkpoint_path"),
        hidden_size=model_cfg.get("hidden_size", 256),
        num_hidden_layers=model_cfg.get("num_hidden_layers", 4),
        batch_size=batch_size or train_cfg.get("batch_size", 48),
        max_seq_len=train_cfg.get("max_seq_len", 128),
        lr=lr or train_cfg.get("lr", 3e-4),
        epochs=epochs or train_cfg.get("epochs", 1),
        subset=train_cfg.get("subset", 0),
        T_infer=pc_cfg.get("T_infer", 1),
        gamma=pc_cfg.get("gamma", 0.1),
        dopamine_eta=dop_cfg.get("eta", 1.0),
        dopamine_beta=dop_cfg.get("beta", 0.5),
        dopamine_gamma=dop_cfg.get("gamma", 0.3),
        out_dir=out_cfg.get("out_dir", "out_pc_unified"),
        save_interval=out_cfg.get("save_interval", 10000),
    )
    print(f"从配置启动训练: {config}")
    task_pipelines = []
    for i, fp in enumerate(data_file_list):
        ds = DualChannelDataset(
            fp, max_length=config_obj.max_seq_len, max_samples=config_obj.subset or None
        )
        task_pipelines.append((f"task_{i}", ds))
    loop = TrainingLoop(config_obj)
    loop.train(task_pipelines)


@click.command(name="resume", help="从检查点文件恢复训练 (纯 Hebbian, 零反向传播)")
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option("--batch-size", "-b", default=48, type=int, help="批大小")
@click.option("--max-seq-len", default=128, type=int, help="最大序列长度")
@click.option("--verbose", "-v", is_flag=True, default=False, help="详细日志")
def resume(checkpoint, batch_size, max_seq_len, verbose):
    """从检查点文件恢复训练 (纯 Hebbian, 零反向传播)."""
    from model.core.dataset import DualChannelDataset
    from model.core.train import TrainingConfig, TrainingLoop

    ckpt_path = resolve_path(checkpoint)
    if not os.path.exists(ckpt_path):
        print(f"✗ 检查点不存在: {ckpt_path}")
        raise click.ClickException(f"检查点不存在: {ckpt_path}")

    print(f"恢复训练 — 检查点: {ckpt_path}")
    out_dir = os.path.dirname(ckpt_path)
    out_name = os.path.basename(out_dir)

    config = TrainingConfig(
        checkpoint_path=ckpt_path,
        out_dir=out_name,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    data_files = []
    if "config" in ckpt_data and "data_files" in ckpt_data["config"]:
        data_files = ckpt_data["config"]["data_files"]
    if not data_files:
        print("检查点中无 data_files 信息, 请使用 `train` 命令指定数据")
        task_pipelines = []
    else:
        task_pipelines = []
        for i, fp in enumerate(data_files):
            resolved = resolve_path(fp)
            ds = DualChannelDataset(resolved, max_length=max_seq_len, max_samples=None)
            task_pipelines.append((f"task_{i}", ds))

    loop = TrainingLoop(config)
    if task_pipelines:
        loop.train(task_pipelines)
    else:
        print("Model loaded. 请提供数据文件继续训练.")


if __name__ == "__main__":
    train()
