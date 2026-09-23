"""
持续学习
DEPRECATED sparse 管线
记忆回放 + 遗忘嗅探 + 抽象记忆银行.
"""

from .abstraction_bank import (
    AbstractionBank,
    AbstractionSniffer,
    VariationalReplayer,
    compute_layer_importance,
)
from .forgetting_sniffer import ForgettingSniffer
from .memory_bank import Exemplar, MemoryBank
