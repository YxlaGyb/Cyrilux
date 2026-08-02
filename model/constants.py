"""共享常量 — 状态字段索引、层架构、连接类型、页参数."""

# ── 神经元状态列索引 ─────────────────────────────────────────────────
F_Z = 0
F_MU = 1
F_EPS = 2
F_THRESHOLD = 3
F_FIRING_RATE = 4
F_PI = 5
F_Z_PREV = 6
F_BCM_SLOPE = 7  # 每个神经元自己的 BCM 斜率 (2.8~5.2)
F_BCM_ZERO = 8   # 每个神经元自己的 BCM 零点 (0.15~0.45)
N_STATE_FIELDS = 9

# 6 层皮层架构层常量 (10-14, 避开感官前端 0-6)
LAYER_SENSORY = 0
LAYER_L4 = 10  # 颗粒层: 感觉输入接收 (快)
LAYER_L2 = 11  # 外颗粒层: 横向连接, 局部处理
LAYER_L3 = 12  # 外锥体层: 前馈集成
LAYER_L5 = 13  # 内锥体层: 深度输出, LM Head 读取
LAYER_L6 = 14  # 多形层: 反馈调节, 最慢

HIDDEN_LAYERS = [LAYER_L4, LAYER_L2, LAYER_L3, LAYER_L5, LAYER_L6]
TOP_LAYER = LAYER_L4  # LM Head 输出层 — L4 有输入区分度, L5 经过3层随机投影后坍缩

# 每层配置 (神经元数, 惯性α, k-WTA比例)
LAYER_CONFIG = {
    LAYER_L4: (1024, 0.2, 0.10),
    LAYER_L2: (192, 0.4, 0.10),
    LAYER_L3: (192, 0.6, 0.15),
    LAYER_L5: (128, 0.8, 0.20),
    LAYER_L6: (64, 0.95, 0.10),
}

# 连接类型
CONN_FEEDFORWARD = 0
CONN_FEEDBACK = 1
CONN_LATERAL = 2

# 页大小常量 (与 PageStorage 共享)
PAGE_NEURONS = 4096
PAGE_SYNAPSES = 16384
PAGE_TD = 16384
K_FAN = 128
