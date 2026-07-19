"""
控制台页面 — MetricCard 概览 + 快速入口 + 最近检查点.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QTreeWidget, QTreeWidgetItem,
                             QFrame, QSizePolicy)

from gui.theme import HackerTheme, FONT_FAMILY


class DashboardPage(QWidget):
    """控制台 — 项目总览首页."""

    def __init__(self, bridge, theme: HackerTheme):
        super().__init__()
        self._bridge = bridge
        self._theme = theme
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Heading
        heading = QLabel(">>> 控制台")
        heading.setFont(heading.font())
        heading.setObjectName("accent")
        layout.addWidget(heading)

        # ── Metric 行 ──
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(16)
        self._version_card, self._version_val = self._make_metric("版本", "v0.2.0")
        self._dataset_card, self._dataset_val = self._make_metric("数据集", "—")
        self._gpu_card, self._gpu_val = self._make_metric("GPU", "检测中...")
        metrics_row.addWidget(self._version_card)
        metrics_row.addWidget(self._dataset_card)
        metrics_row.addWidget(self._gpu_card)
        layout.addLayout(metrics_row)

        # ── 快速入口 ──
        quick_label = QLabel("快速入口")
        quick_label.setObjectName("secondary")
        layout.addWidget(quick_label)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        for text, tip in [("▶ 训练", "转到训练页"), ("▶ 评估", "转到评估页"), ("▶ 自主运行", "转到自主运行页")]:
            btn = QPushButton(text)
            btn.setToolTip(tip)
            btn_row.addWidget(btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ── 最近检查点 ──
        ckpt_label = QLabel("最近检查点")
        ckpt_label.setObjectName("secondary")
        layout.addWidget(ckpt_label)

        self._ckpt_tree = QTreeWidget()
        self._ckpt_tree.setHeaderLabels(["目录", "文件名", "步数", "大小"])
        self._ckpt_tree.setAlternatingRowColors(True)
        self._ckpt_tree.setRootIsDecorated(False)
        layout.addWidget(self._ckpt_tree, 1)

        # ── 版本信息 ──
        info = QLabel("virtuosov2 v0.2.0 — PC 局部动态小语言模型 + 多巴胺 + 持续学习")
        info.setObjectName("secondary")
        layout.addWidget(info)

    def _make_metric(self, label: str, value: str) -> tuple[QFrame, QLabel]:
        """创建 Metric 卡片, 返回 (card, value_label) 以便后续更新."""
        card = QFrame()
        card.setFrameShape(QFrame.Shape.Box)
        card.setMinimumHeight(80)
        layout = QVBoxLayout(card)
        lbl = QLabel(label)
        lbl.setObjectName("secondary")
        val = QLabel(value)
        val.setObjectName("accent")
        layout.addWidget(lbl)
        layout.addWidget(val)
        return card, val

    def refresh(self) -> None:
        """加载数据 (内部 try/except 防止闪退)."""
        try:
            # 扫描数据集
            import os
            ds_dir = "dataset"
            if os.path.isdir(ds_dir):
                count = len([f for f in os.listdir(ds_dir) if f.endswith(".jsonl")])
                self._dataset_val.setText(str(count))

            # 扫描检查点
            items = self._bridge.scan_checkpoints()
            self._ckpt_tree.clear()
            for entry in items[:10]:
                item = QTreeWidgetItem([
                    entry.get("dir", ""),
                    entry.get("filename", ""),
                    str(entry.get("step", "")),
                    f'{entry.get("size_kb", 0):.0f} KB',
                ])
                self._ckpt_tree.addTopLevelItem(item)

            # GPU 信息
            try:
                import torch
                if torch.cuda.is_available():
                    name = torch.cuda.get_device_name(0)
                    self._gpu_val.setText(name)
            except Exception:
                pass
        except Exception as exc:
            import traceback
            traceback.print_exc()
            # 在树中显示错误
            self._ckpt_tree.clear()
            err_item = QTreeWidgetItem([f"⚠ 加载失败: {exc}", "", "", ""])
            self._ckpt_tree.addTopLevelItem(err_item)

    def on_theme_changed(self, theme) -> None:
        self._theme = theme
