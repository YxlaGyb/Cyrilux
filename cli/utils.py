"""
CLI 共享工具: 路径处理 & 配置加载.
"""

import os, json


def resolve_path(p: str) -> str:
    """将相对路径解析为绝对路径 (相对于项目根)."""
    if os.path.isabs(p):
        return p
    return os.path.join(PROJECT_ROOT, p)


def load_config(path: str) -> dict:
    """加载 JSON 配置文件."""
    with open(resolve_path(path), 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config(config: dict, path: str):
    """保存 JSON 配置文件."""
    path = resolve_path(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✓ 配置已保存: {path}")


def merge_config(config: dict, cli_overrides: dict) -> dict:
    """CLI 参数覆写配置项."""
    merged = dict(config)
    for k, v in cli_overrides.items():
        if v is not None:
            merged[k] = v
    return merged


# ═══════════════════════════════════════════════════════════════════
# Rich 训练进度面板
# ═══════════════════════════════════════════════════════════════════

class RichTrainingPanel:
    """4 区域 Rich Layout 实时训练监控面板."""

    def __init__(self, device: str = "cuda:0"):
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="status", size=3),
            Layout(name="progress", size=5),
            Layout(name="log"),
        )
        self.log_lines: list[str] = []
        self._init_panels(device)

    def _init_panels(self, device: str):
        self.layout["status"].update(
            Panel("[yellow]等待启动...[/]", title="[bold]Status")
        )
        self.layout["progress"].update(
            Panel("", title="[bold]Progress")
        )
        self.layout["log"].update(
            Panel("", title="[bold]Log")
        )

    def update(self, data: dict):
        """从 TrainManager 回调更新面板."""
        t = data.get("type", "")

        if t == "log":
            msg = data.get("message", "")
            self.log_lines.append(f"[dim]{msg}[/]")
            if len(self.log_lines) > 200:
                self.log_lines = self.log_lines[-100:]
            self.layout["log"].update(
                Panel("\n".join(self.log_lines[-25:]), title="[bold]Log")
            )

        elif t == "progress":
            step = data.get("step", 0)
            total = data.get("total_steps", 1)
            epoch = data.get("epoch", 1)
            total_epochs = data.get("total_epochs", 1)
            ce = data.get("ce_loss", 0.0)
            F = data.get("F", 0.0)
            D = data.get("D", 0.0)
            lr = data.get("lr", 0.0)

            pct = step / max(total, 1)
            bar_w = 40
            filled = int(bar_w * pct)
            bar = "█" * filled + "░" * (bar_w - filled)
            pct_str = f"{pct * 100:.0f}%"

            # 状态面板
            status = Panel(
                Text.from_markup(
                    f"Epoch [bold]{epoch}[/]/[bold]{total_epochs}[/]  "
                    f"Step [bold]{step}[/]/[bold]{total}[/]  "
                    f"Device: cuda:0"
                ),
                title="[bold]Status",
            )

            # 指标表格
            table = Table.grid(padding=(0, 2))
            table.add_row("CE Loss", f"[cyan]{ce:.4f}[/]")
            table.add_row("F (Pred)", f"[magenta]{F:.2f}[/]")
            table.add_row("Dopamine D", f"[green]{D:.4f}[/]")
            table.add_row("Learning Rate", f"[yellow]{lr:.2e}[/]")

            progress = Panel(
                Text.from_markup(
                    f"{bar} [bold]{pct_str}[/]\n"
                ) + table,
                title="[bold]Progress",
            )

            self.layout["status"].update(status)
            self.layout["progress"].update(progress)

        elif t == "checkpoint":
            ckpt = data.get("checkpoint_path", "")
            self.log_lines.append(f"[green]✓[/] 检查点: {ckpt}")
            self.layout["log"].update(
                Panel("\n".join(self.log_lines[-25:]), title="[bold]Log")
            )

        elif t == "phase":
            msg = data.get("message", "")
            self.log_lines.append(f"[blue]◆[/] {msg}")
            self.layout["log"].update(
                Panel("\n".join(self.log_lines[-25:]), title="[bold]Log")
            )

        elif t == "done":
            self.log_lines.append("[bold green]══════════════════════════════[/]")
            self.log_lines.append("[bold green]✓ 训练完成![/]")
            self.layout["log"].update(
                Panel("\n".join(self.log_lines[-25:]), title="[bold]Log")
            )

        elif t == "error":
            self.log_lines.append(f"[bold red]✗ 错误: {data.get('message', '')}[/]")
            self.layout["log"].update(
                Panel("\n".join(self.log_lines[-25:]), title="[bold]Log")
            )


# ═══════════════════════════════════════════════════════════════════
# 通用配置模板
# ═══════════════════════════════════════════════════════════════════

TRAIN_CONFIG_TEMPLATE = {
    "model": {
        "hidden_size": 256,
        "num_hidden_layers": 4,
    },
    "data": {
        "data_files": ["datasets/task_a_daily_20k.jsonl"],
        "combined_training": True,
        "subset": 0,
    },
    "training": {
        "batch_size": 48,
        "max_seq_len": 128,
        "lr": 3e-4,
        "epochs": 1,
        "warmup_steps": 0,
        "grad_clip": 1.0,
        "weight_decay": 0.01,
    },
    "pc": {
        "T_infer": 1,
        "gamma": 0.1,
    },
    "dopamine": {
        "enabled": True,
        "eta": 1.0,
        "beta": 0.5,
        "gamma": 0.3,
    },
    "quantize": {
        "enabled": False,
    },
    "output": {
        "out_dir": "out_pc_unified",
        "save_interval": 500,
    },
}

AUTONOMOUS_CONFIG_TEMPLATE = {
    "wake_steps": 20,
    "play_steps": 100,
    "sleep_interval": 500,
    "gen_max_new": 64,
    "gen_temperature": 0.8,
    "gen_top_k": 40,
    "gen_prompt_len": 32,
    "batch_size": 16,
    "max_seq_len": 128,
    "lr": 1e-4,
    "gamma": 0.05,
    "T_infer": 1,
    "grad_clip": 1.0,
    "dopamine_eta": 1.0,
    "dopamine_beta": 0.3,
    "dopamine_gamma": 0.2,
    "dopamine_threshold": 0.05,
    "max_replay_buffer": 2000,
    "replay_batch_size": 16,
    "replay_ratio": 3,
    "save_interval": 200,
    "out_dir": "out_autonomous",
    "data_dir": "dataset",
    "data_rotate_interval": 500,
}
