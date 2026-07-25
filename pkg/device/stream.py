"""StreamRunner — 内部引擎别名。

模型定义已迁至 model.model_cyrene.CyreneModel。
外部代码请从 model.model_cyrene 导入，勿直接引用此模块。
"""

from model.model_cyrene import CyreneModel as StreamRunner  # noqa: F401

