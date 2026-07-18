"""
全局实验状态管理 — 配置 / 任务管线 / 检查点注册 / 主题.

ExperimentState 单例, 通过订阅通知视图更新.
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass, field
from typing import Any, Callable

from gui.theme import Theme, ThemeManager


@dataclass
class ExperimentState:
    """全局实验状态."""

    # 主题
    theme_mgr: ThemeManager = field(default_factory=ThemeManager)

    # 当前训练配置 (键值对, 对应 TrainingConfig 字段)
    config: dict = field(default_factory=lambda: {
        "batch_size": "48",
        "max_seq_len": "128",
        "lr": "3e-4",
        "epochs": "1",
        "subset": "0",
        "seed": "42",
        "split_size": "0",
        "hidden_size": "256",
        "num_hidden_layers": "4",
        "T_infer": "2",
        "gamma": "0.1",
        "max_beta": "2.0",
        "max_beta_conv": "1.0",
        "grad_clip": "1.0",
        "dopamine_eta": "1.0",
        "dopamine_beta": "0.5",
        "dopamine_gamma": "0.3",
        "replay_ratio": "5",
        "bank_size": "2000",
        "sniff_interval": "200",
        "repair_threshold": "1.2",
        "repair_steps": "10",
        "eval_samples": "100",
        "n_prototypes": "8",
        "abstraction_replay_interval": "200",
        "enable_qat": "1",
        "qat_groupsize": "64",
        "no_quantize_embed": "0",
    })

    # 任务管线
    task_pipelines: list = field(default_factory=list)  # [(task_id, path), ...]

    # 配置模板
    config_templates: dict = field(default_factory=dict)  # {name: config_dict}

    # 检查点注册表
    checkpoint_registry: list = field(default_factory=list)  # [{path, dir, size, step, mtime}, ...]

    # 当前模型路径
    current_model_path: str | None = None

    # 输出目录
    out_dir: str = "ola_out"

    # ── 订阅者 ──
    _listeners: dict[str, list[Callable]] = field(default_factory=dict, repr=False)

    def subscribe(self, key: str, callback: Callable):
        """订阅状态变更: key 变更时调用 callback(new_value)."""
        self._listeners.setdefault(key, []).append(callback)

    def update(self, key: str, value: Any):
        """更新状态并通知订阅者."""
        setattr(self, key, value)
        self._notify(key, value)

    def _notify(self, key: str, value: Any):
        for cb in self._listeners.get(key, []):
            try:
                cb(value)
            except Exception:
                pass

    def config_get(self, key: str, default: str = "") -> str:
        return self.config.get(key, default)

    def config_set(self, key: str, value: str):
        self.config[key] = value

    def config_to_kwargs(self) -> dict:
        """将 config 转为 TrainingConfig 兼容的 kwargs (自动类型转换)."""
        kw = {}
        int_keys = {
            "batch_size", "max_seq_len", "epochs", "subset", "seed",
            "T_infer", "replay_ratio", "bank_size", "sniff_interval",
            "repair_steps", "eval_samples", "n_prototypes",
            "abstraction_replay_interval", "split_size", "hidden_size",
            "num_hidden_layers", "qat_groupsize",
        }
        float_keys = {
            "lr", "gamma", "max_beta", "max_beta_conv", "grad_clip",
            "dopamine_eta", "dopamine_beta", "dopamine_gamma",
            "repair_threshold",
        }
        bool_keys = {"enable_qat", "no_quantize_embed"}
        for k, v in self.config.items():
            if k in int_keys:
                kw[k] = int(v)
            elif k in float_keys:
                kw[k] = float(v)
            elif k in bool_keys:
                kw[k] = bool(int(v)) if v.isdigit() else bool(v)
        return kw

    # ── 检查点扫描 ──

    def scan_checkpoints(self, base_dir: str | None = None) -> list[dict]:
        """递归扫描 .pt 文件."""
        base = base_dir or self.out_dir
        if not os.path.isdir(base):
            self.checkpoint_registry = []
            return []
        entries = []
        for fpath in glob.glob(os.path.join(base, "**", "*.pt"), recursive=True):
            try:
                stat = os.stat(fpath)
                rel = os.path.relpath(fpath, base)
                parts = rel.replace("\\", "/").split("/")
                step = 0
                # 从文件名提取步数: unified_ckpt_s1499.pt → 1499
                fname = os.path.splitext(os.path.basename(fpath))[0]
                if "s" in fname:
                    try:
                        step = int(fname.split("s")[-1])
                    except ValueError:
                        pass
                entries.append({
                    "path": fpath,
                    "dir": parts[0] if len(parts) > 1 else "",
                    "filename": os.path.basename(fpath),
                    "size_kb": stat.st_size / 1024,
                    "step": step,
                    "mtime": stat.st_mtime,
                })
            except OSError:
                pass
        entries.sort(key=lambda e: e["mtime"], reverse=True)
        self.checkpoint_registry = entries
        return entries

    # ── 配置模板 ──

    def load_templates(self, template_dir: str = "ola_out/configs"):
        """从目录加载配置模板 JSON."""
        import json
        self.config_templates = {}
        if not os.path.isdir(template_dir):
            return self.config_templates
        for fname in sorted(os.listdir(template_dir)):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(template_dir, fname), "r") as f:
                        name = os.path.splitext(fname)[0]
                        self.config_templates[name] = json.load(f)
                except Exception:
                    pass
        return self.config_templates

    def save_template(self, name: str, template_dir: str = "ola_out/configs"):
        """将当前配置保存为模板."""
        import json
        os.makedirs(template_dir, exist_ok=True)
        path = os.path.join(template_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        self.config_templates[name] = dict(self.config)
