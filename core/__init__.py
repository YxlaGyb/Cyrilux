"""
virtuosov2.core — 核心逻辑层

提供以下模块供 main.py / cli/ 等消费端调用:
  - dataset:      双通道 UTF-8 字节数据集 (DualChannelDataset)
  - training:     统一训练循环 (PC + 多巴胺 + QAT + 持续学习)
  - threaded_trainer: GUI 后台线程训练管理器
  - evaluation:   Perplexity / 文本生成 / 自监督 / 遗忘压力
  - prepare_tasks: 按领域切分 / 异构数据集转换
  - data_converter: 样本格式转换 (conversations→text)
  - data_splitter:  大文件分割
  - trainer_utils:  工具函数 (Logger / get_lr / setup_seed)
  - autonomous_mind: 持续自主运行 (WAKE→PLAY→SLEEP)
"""

from . import dataset
from . import training
from . import evaluation
from . import prepare_tasks
from . import data_converter
from . import data_splitter
from . import trainer_utils
