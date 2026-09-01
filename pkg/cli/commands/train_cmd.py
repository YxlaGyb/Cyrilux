"""
train
"""

import os

import click
import torch

from pkg.cli.utils import load_config, resolve_path


def _build_net(hidden_size: int, lr: float, max_seq_len: int, checkpoint: str | None = None):
    """dense 网络构建: 按 hidden_size 比例缩放各层; 有 checkpoint 则载入加权."""
    from model import CyreneModel, DensePCNet

    ratio = hidden_size / 1024.0
    d_cfg = CyreneModel(
        d_l4=hidden_size,
        d_l2=max(64, int(384 * ratio)),
        d_l3=max(64, int(384 * ratio)),
        d_l5=max(64, int(256 * ratio)),
        d_l6=max(64, int(128 * ratio)),
        lr_hebbian=lr,
        max_seq_len=max_seq_len,
    )
    if checkpoint is not None:
        return DensePCNet.load(resolve_path(checkpoint), d_cfg), d_cfg
    return DensePCNet(d_cfg), d_cfg


def _train_dense(
    data_files,
    out_dir,
    batch_size,
    max_seq_len,
    lr,
    epochs,
    hidden_size,
    device,
    subset,
    checkpoint=None,
    net=None,
):
    """dense — PPA 闭环批次训练: JSONL 逐行装载 → 批量 learn → 保存 state_dict."""
    import json as _json

    from tqdm import tqdm

    torch.set_grad_enabled(False)
    if net is None:
        net, d_cfg = _build_net(hidden_size, lr, max_seq_len, checkpoint=checkpoint)
    else:
        d_cfg = net.cfg
    net = net.to(device)
    print(f"PPA dense training  L4={d_cfg.d_l4}  params={d_cfg.param_count():,}  device={device}")
    if checkpoint is not None:
        print(f"  从检查点续训: {resolve_path(checkpoint)}")

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


@click.command(name="train", help="直接训练模式 (dense PPA 闭环, 纯 Hebbian, 零反向传播)")
@click.option("--model-type", default="pc_unified", help="模型类型 (dense 忽略)")
@click.option("--checkpoint", "-c", default=None, help="初始检查点 (续训时传入)")
@click.option("--hidden-size", default=256, type=int, help="隐藏层维度 (L4 基准)")
@click.option("--num-layers", default=4, type=int, help="层数 (dense 忽略)")
@click.option("--data", "-d", multiple=True, help="数据文件 (可多个)")
@click.option("--combined/--no-combined", default=True, help="是否联合训练 (dense 忽略)")
@click.option("--batch-size", "-b", default=48, type=int, help="批大小")
@click.option("--max-seq-len", default=128, type=int, help="最大序列长度")
@click.option("--lr", default=3e-4, type=float, help="学习率")
@click.option("--epochs", "-e", default=1, type=int, help="训练轮数")
@click.option("--subset", "-n", default=0, type=int, help="子集大小 (0=全部)")
@click.option("--T-infer", default=1, type=int, help="Inference time steps (dense 忽略)")
@click.option("--gamma", default=0.1, type=float, help="多巴胺折扣因子 (dense 忽略)")
@click.option("--dopamine/--no-dopamine", default=True, help="启用多巴胺 (dense 忽略)")
@click.option("--dopamine-eta", default=1.0, type=float, help="多巴胺学习率 (dense 忽略)")
@click.option("--dopamine-beta", default=0.5, type=float, help="多巴胺灵敏度 (dense 忽略)")
@click.option("--dopamine-gamma", default=0.3, type=float, help="多巴胺衰减 (dense 忽略)")
@click.option("--out-dir", "-o", default="out_pc_unified", help="输出目录")
@click.option("--save-interval", default=10000, type=int, help="保存间隔 (dense 忽略, 结束时保存 final.pt)")
@click.option("--abstraction-bank/--no-abstraction-bank", default=False, help="抽象记忆库 (dense 忽略)")
@click.option("--auto-phase2", is_flag=True, default=False, help="训练后自动进入 Phase 2 (dense 忽略)")
@click.option("--resume", is_flag=True, default=False, help="从断点恢复 (out_dir/final.pt 或 -c 指定)")
@click.option("--backend", default="dense", help="dense=PPA 闭环 (默认) / sparse=已归档冻结")
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

    if backend == "sparse":
        raise click.ClickException("sparse 管线已随 117 轮归档冻结 (model/_archived_sparse), 请用 dense")
    if not data:
        raise click.ClickException("请用 -d 指定数据文件 (可多个)")
    ignored = []
    if t_infer != 1:
        ignored.append("--T-infer")
    if gamma != 0.1:
        ignored.append("--gamma")
    if not dopamine or dopamine_eta != 1.0 or dopamine_beta != 0.5 or dopamine_gamma != 0.3:
        ignored.append("--dopamine*")
    if num_layers != 4:
        ignored.append("--num-layers")
    if not combined:
        ignored.append("--combined")
    if abstraction_bank:
        ignored.append("--abstraction-bank")
    if auto_phase2:
        ignored.append("--auto-phase2")
    if save_interval != 10000:
        ignored.append("--save-interval")
    if ignored:
        print(f"dense 后端: 忽略稀疏参数 {', '.join(ignored)}")
    if resume and checkpoint is None:
        checkpoint = os.path.join(out_dir, "final.pt")
        if not os.path.exists(checkpoint):
            raise click.ClickException(f"--resume 但找不到 {checkpoint}, 请用 -c 指定检查点")
    data_files = [resolve_path(f.strip()) for f in data]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _train_dense(
        data_files,
        out_dir,
        batch_size,
        max_seq_len,
        lr,
        epochs,
        hidden_size,
        device,
        subset,
        checkpoint=checkpoint,
    )


@click.command(name="from-config", help="从 JSON 配置文件加载参数并训练 (dense PPA 闭环)")
@click.argument("config", type=click.Path(exists=True))
@click.option("--batch-size", "-b", type=int, default=None, help="覆盖批大小")
@click.option("--lr", type=float, default=None, help="覆盖学习率")
@click.option("--epochs", "-e", type=int, default=None, help="覆盖训练轮数")
@click.option("--out-dir", "-o", default=None, help="覆盖输出目录")
@click.option("--verbose", "-v", is_flag=True, default=False, help="详细日志")
def from_config(config, batch_size, lr, epochs, out_dir, verbose):
    """从 JSON 配置文件加载参数并训练 (dense PPA 闭环)."""
    cfg = load_config(config)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    data_file_list = [resolve_path(f) for f in data_cfg.get("data_files", [])]
    if not data_file_list:
        raise click.ClickException("配置 data.data_files 为空, 请填写数据文件")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"从配置启动训练: {config}")
    _train_dense(
        data_file_list,
        out_dir or (cfg.get("output", {}).get("out_dir") or "out_pc_unified"),
        batch_size or train_cfg.get("batch_size", 48),
        train_cfg.get("max_seq_len", 128),
        lr or train_cfg.get("lr", 3e-4),
        epochs or train_cfg.get("epochs", 1),
        model_cfg.get("hidden_size", 256),
        device,
        data_cfg.get("subset", train_cfg.get("subset", 0)),
        checkpoint=model_cfg.get("checkpoint_path"),
    )


@click.command(name="resume", help="从检查点文件恢复训练 (dense PPA 闭环)")
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option("--data", "-d", multiple=True, help="数据文件 (可多个); 不传则仅加载检查点")
@click.option("--batch-size", "-b", default=48, type=int, help="批大小")
@click.option("--max-seq-len", default=128, type=int, help="最大序列长度")
@click.option("--epochs", "-e", default=1, type=int, help="训练轮数")
@click.option("--lr", default=None, type=float, help="覆盖学习率 (默认用检查点配置)")
@click.option("--out-dir", "-o", default=None, help="覆盖输出目录 (默认=检查点所在目录)")
@click.option("--verbose", "-v", is_flag=True, default=False, help="详细日志")
def resume(checkpoint, data, batch_size, max_seq_len, epochs, lr, out_dir, verbose):
    """从检查点文件恢复训练 (dense PPA 闭环)."""
    from model import DensePCNet

    ckpt_path = resolve_path(checkpoint)
    if not os.path.exists(ckpt_path):
        print(f"✗ 检查点不存在: {ckpt_path}")
        raise click.ClickException(f"检查点不存在: {ckpt_path}")

    print(f"恢复训练 — 检查点: {ckpt_path}")
    net = DensePCNet.load(ckpt_path)
    if lr is not None:
        net.cfg.lr_hebbian = lr
    if not data:
        print(f"Model loaded (L4={net.active_size['l4']}). 请用 -d 指定数据文件继续训练.")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_files = [resolve_path(f.strip()) for f in data]
    _train_dense(
        data_files,
        out_dir or os.path.dirname(ckpt_path) or ".",
        batch_size,
        max_seq_len,
        net.cfg.lr_hebbian,
        epochs,
        net.active_size["l4"],
        device,
        0,
        net=net,
    )


if __name__ == "__main__":
    train()
