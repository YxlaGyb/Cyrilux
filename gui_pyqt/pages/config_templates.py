"""配置模板页面 — 分组参数树 + 参数表 + 保存/加载/导入/导出."""

from __future__ import annotations

import json

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QSplitter, QTreeWidget,
                             QTreeWidgetItem, QTableWidget, QTableWidgetItem,
                             QHeaderView, QFileDialog, QMessageBox,
                             QInputDialog, QAbstractItemView)

from gui_pyqt.theme import HackerTheme

TEMPLATE_GROUPS = {
    "训练 (Training)": {
        "lr": ("0.001", "学习率"),
        "batch_size": ("4", "批次大小"),
        "epochs": ("3", "训练轮数"),
        "max_seq_len": ("128", "最大序列长度"),
        "seed": ("42", "随机种子"),
        "subset": ("-1", "子集大小 (-1=全部)"),
        "grad_clip": ("1.0", "梯度裁剪阈值"),
    },
    "PC (预测编码)": {
        "T_infer": ("16", "推理时间步"),
        "gamma": ("0.1", "稀疏正则权重"),
        "max_beta": ("2.0", "β 上限"),
        "max_beta_conv": ("4.0", "卷积 β 上限"),
        "eval_samples": ("500", "评估样本数"),
    },
    "持续学习 (CL)": {
        "replay_ratio": ("0.3", "回放比例"),
        "bank_size": ("1000", "记忆库容量"),
        "sniff_interval": ("50", "嗅探间隔"),
        "repair_steps": ("10", "修复步数"),
        "repair_threshold": ("0.5", "修复阈值"),
        "n_prototypes": ("20", "原型数量"),
        "abstraction_replay_interval": ("50", "抽象回放间隔"),
    },
    "多巴胺 (Dopamine)": {
        "dopamine_eta": ("0.1", "η — 更新率"),
        "dopamine_beta": ("0.8", "β — 探索偏好"),
        "dopamine_gamma": ("0.9", "γ — 折扣因子"),
    },
}


class ConfigTemplatesPage(QWidget):
    """配置模板管理."""

    def __init__(self, bridge, theme: HackerTheme):
        super().__init__()
        self._bridge = bridge
        self._theme = theme
        self._current_group: str = ""
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(8)

        heading = QLabel(">>> 配置模板")
        heading.setObjectName("accent")
        outer.addWidget(heading)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左: 分组树
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("参数分组"))

        self._group_tree = QTreeWidget()
        self._group_tree.setHeaderHidden(True)
        self._group_tree.setMinimumWidth(160)
        for group_name in TEMPLATE_GROUPS:
            item = QTreeWidgetItem([group_name])
            item.setData(0, Qt.ItemDataRole.UserRole, group_name)
            self._group_tree.addTopLevelItem(item)
        self._group_tree.itemClicked.connect(self._on_group_selected)
        left_layout.addWidget(self._group_tree, 1)
        splitter.addWidget(left_panel)

        # 右: 参数表 + 按钮
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(6)

        self._param_table = QTableWidget()
        self._param_table.setColumnCount(3)
        self._param_table.setHorizontalHeaderLabels(["参数名", "值", "描述"])
        self._param_table.setAlternatingRowColors(True)
        self._param_table.horizontalHeader().setStretchLastSection(True)
        self._param_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self._param_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents)
        self._param_table.verticalHeader().setVisible(False)
        self._param_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        right_layout.addWidget(self._param_table, 1)

        btn_row = QHBoxLayout()
        self._save_template_btn = QPushButton("💾 保存模板")
        self._save_template_btn.clicked.connect(self._save_template)
        btn_row.addWidget(self._save_template_btn)
        self._load_template_btn = QPushButton("📂 加载模板")
        self._load_template_btn.clicked.connect(self._load_template)
        btn_row.addWidget(self._load_template_btn)
        self._import_btn = QPushButton("📥 导入 JSON")
        self._import_btn.clicked.connect(self._import_json)
        btn_row.addWidget(self._import_btn)
        self._export_btn = QPushButton("📤 导出 JSON")
        self._export_btn.clicked.connect(self._export_json)
        btn_row.addWidget(self._export_btn)
        self._reset_btn = QPushButton("↺ 重置默认")
        self._reset_btn.clicked.connect(self._reset_defaults)
        btn_row.addWidget(self._reset_btn)
        right_layout.addLayout(btn_row)

        splitter.addWidget(right_panel)
        splitter.setSizes([200, 600])
        outer.addWidget(splitter, 1)

        # 默认选中第一组
        if self._group_tree.topLevelItemCount():
            first = self._group_tree.topLevelItem(0)
            self._group_tree.setCurrentItem(first)
            self._on_group_selected(first, 0)

    def _on_group_selected(self, item, column) -> None:
        group = item.data(0, Qt.ItemDataRole.UserRole)
        if group and group in TEMPLATE_GROUPS:
            self._current_group = group
            self._populate_params(group)

    def _populate_params(self, group: str) -> None:
        params = TEMPLATE_GROUPS.get(group, {})
        self._param_table.setRowCount(len(params))
        for i, (key, (val, desc)) in enumerate(params.items()):
            self._param_table.setItem(i, 0, QTableWidgetItem(key))
            self._param_table.setItem(i, 1, QTableWidgetItem(val))
            self._param_table.setItem(i, 2, QTableWidgetItem(desc))

    def _collect_params(self) -> dict[str, str]:
        params = {}
        for i in range(self._param_table.rowCount()):
            key_item = self._param_table.item(i, 0)
            val_item = self._param_table.item(i, 1)
            if key_item and val_item:
                params[key_item.text()] = val_item.text()
        return params

    def _save_template(self) -> None:
        params = self._collect_params()
        for k, v in params.items():
            self._bridge.config_set(k, v)
        name = self._current_group or "unnamed"
        try:
            self._bridge.save_template(name, params)
            QMessageBox.information(self, "已保存", f"模板 '{name}' 已保存")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def _load_template(self) -> None:
        try:
            templates = self._bridge.get_templates()
        except Exception:
            templates = {}
        if not templates:
            QMessageBox.information(self, "提示", "没有已保存的模板")
            return
        names = list(templates.keys())
        name, ok = QInputDialog.getItem(self, "加载模板", "选择模板:", names, 0, False)
        if ok and name:
            params = templates[name]
            self._param_table.setRowCount(len(params))
            for i, (k, v) in enumerate(params.items()):
                self._param_table.setItem(i, 0, QTableWidgetItem(k))
                self._param_table.setItem(i, 1, QTableWidgetItem(str(v)))
                self._param_table.setItem(i, 2, QTableWidgetItem(""))
            for k, v in params.items():
                self._bridge.config_set(k, str(v))

    def _import_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "导入 JSON", "", "JSON (*.json)")
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    params = json.load(f)
                self._param_table.setRowCount(len(params))
                for i, (k, v) in enumerate(params.items()):
                    self._param_table.setItem(i, 0, QTableWidgetItem(k))
                    self._param_table.setItem(i, 1, QTableWidgetItem(str(v)))
                    self._param_table.setItem(i, 2, QTableWidgetItem(""))
            except Exception as e:
                QMessageBox.critical(self, "导入失败", str(e))

    def _export_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 JSON", "config.json", "JSON (*.json)")
        if path:
            try:
                params = self._collect_params()
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(params, f, ensure_ascii=False, indent=2)
                QMessageBox.information(self, "已导出", f"导出至 {path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", str(e))

    def _reset_defaults(self) -> None:
        if self._current_group and self._current_group in TEMPLATE_GROUPS:
            self._populate_params(self._current_group)

    def on_theme_changed(self, theme) -> None:
        self._theme = theme
