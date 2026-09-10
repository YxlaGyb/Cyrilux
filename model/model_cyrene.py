"""Cyrene 模型定义

密集 PPA 感知-预测-行动闭环网络.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass, fields, replace

import torch
from torch import nn


@dataclass
class CyreneModel:
    """PPA 网络配置."""

    # 维度
    d_input: int = 256
    d_act: int = 256  # W_act 列数 (字节域 256, 具身模式 2)
    d_l4: int = 1024
    d_l2: int = 384
    d_l3: int = 384
    d_l5: int = 1024
    d_l6: int = 128
    max_seq_len: int = 256
    input_history: bool = True  # W_04 输入拼接 [z0[t], z0[t-1]], 词序进入表示层

    # Hebbian 学习
    lr_hebbian: float = 0.003
    temporal_lr_ratio: float = 5.0
    oja_alpha: float = 0.05
    column_dropout: float = 0.25
    inertia_alpha: float = 0.3

    # 修剪
    prune_warmup: int = 5000
    prune_fraction: float = 0.05
    death_probation: int = 200
    death_threshold: float = 1e-4
    active_size_lower_bound: int = 128
    l4_lower_bound: int = 512  # L4 承载表示+预测双任务, 维度坍缩到 NaN
    adaptive_traction: bool = False
    lm_lr_boost: float = 1.0
    adaptive_rho: bool = False

    # 自由运行振荡器
    free_run_window: int = 64
    osc_amp_f: float = 0.10
    osc_amp_m: float = 0.03
    osc_amp_s: float = 0.01

    # 短期突触可塑性 (STP)
    stp_tau_min: float = 8.0
    stp_tau_max: float = 256.0
    stp_u_init: float = 0.05
    stp_u_adapt: bool = True
    stp_u_adapt_rate: float = 0.01
    stp_u_min: float = 0.01
    stp_u_max: float = 0.5

    # 突触缩放 / 谱守卫
    wt_syn_scaling: bool = True
    wt_syn_scaling_rate: float = 0.1
    spectral_guard_bound: float = 1.5
    bias_leak_rate: float = 1e-4
    oja_elasticity: float = 0.05
    probation_decay: float = 0.5

    # 生成 / 绑定
    gen_precision: float = 1.0
    bind_dim: int = 4096
    bind_k: int = 10
    bind_mode: str = "hard"
    bind_orth: bool = False

    # 读出端诊断开关
    lm_freeze_w1: bool = False
    lm_no_contrast: bool = False
    lm_no_bcm: bool = False

    # 竞争性记忆单元群
    mem_k0: int = 6
    mem_k_max: int = 16
    mem_alpha_min: float = 1.0 / 1024.0
    mem_alpha_max: float = 0.5
    mem_g_max: float = 2.0
    mem_g_min: float = 0.05
    mem_eta_g: float = 0.01
    mem_g_decay: float = 0.001
    mem_birth_thresh: float = 1.2
    mem_birth_cooldown: int = 2000
    mem_death_steps: int = 2000

    # 体内代谢 (P1): E ← E·(1−metab_d) + metab_c·ΔF; MAD EMA 新息权 = 1−metab_mad_alpha.
    # metab_mad_alpha=0.90 + metab_tanh_div=1.5 为 R1 契约标定值 (2026-09-01 P1 段标定:
    # 0.95/2.0 下 median(leg)=0.26 低于 [0.3,1] 带; 标定后 0.38 复现 R1 历史值 0.39).
    metab_d: float = 0.05
    metab_c: float = 0.1
    metab_mad_alpha: float = 0.90
    metab_tanh_div: float = 2.5
    # 死亡契约 (P2 终裁, 下探缺失): f < base·(1−metab_dip_margin) 每次清零计数器,
    # 无下探连续 metab_death_steps 步 → 死亡回合 (修剪 5% 最弱 + 记忆单元饥荒击杀).
    # 基线 = F 慢 EMA (metab_base_rate); 平台期 F 规律性下探不触发, 饥荒 (F 钉高位
    # 零下探) 触发. 术前探针标定: δ=0.10, N=250 (平台 max_run 93 → 2.7× 裕度;
    # famine_rand 415 → 1.66×).
    metab_base_rate: float = 0.002
    metab_dip_margin: float = 0.10
    metab_death_steps: int = 250
    # 连续感知步 (无发声) 上限, 与 metab_death_steps 镜像 (留校准钮)
    metab_silence_steps: int = 250
    # P3-c 全行为计价 (判词 §2.4): 看/学/记/说全耗能, 成本只进 E 账本 (资本/应激/门控货币),
    # R 吃纯 ΔF (条件二逐点: F 下降 → R > 0). κ 为 CPU 重放标定值 (docs/es/p3c_handoff.md).
    # cost_看 = κ_p·f_now (感知步输入处理 F 水平); cost_学 = κ_l·sqrt(f_now) (ΔW 幅度结构代理);
    # cost_记 = κ_m·W1 行数 (生存质量费 — P2 死亡回合直接降此费); cost_说 = κ_s (每次发声 ATP).
    metab_cost_perc: float = 1.0e-3
    metab_cost_learn: float = 1.0e-3
    metab_cost_mem: float = 3.0e-7
    metab_cost_say: float = 2.0e-3
    # E_ref = E 慢 EMA; 应激 = 饥荒契约进度 (starve_cnt/metab_death_steps), 零设计者增益
    metab_eref_rate: float = 0.002
    # P3-a τ 有界状态映射 (拆硬夹, 判词 §2.5 案一): τ = τ_lo + (τ_hi−τ_lo)·(0.5+0.5·tanh(g)).
    # 无乘性累加无 clamp — 范围由映射余域给出: 噪声区 (τ≈9.5 展平骗裁判) 与贪心区 (τ→0
    # 单字符自锁) 均物理不可达; 旧健康带 [1.0,1.54] 只是范围内一点. β_d 标定: 健康 ε
    # 差分 (+0.1~0.3) → τ≈1.8-2.2 (历史"粘 2.00"操作点回归, 目的已兑现).
    metab_tau_lo: float = 0.9
    metab_tau_hi: float = 2.5
    metab_tau_beta_d: float = 4.0
    metab_tau_beta_s: float = 1.5
    metab_tau_smooth: float = 0.3
    # P3-b 行为门 (节律内生, 判词 §2.2): p_say = σ(κ_g·(g_eco−g_perc) − κ_s·stress
    # + κ_A·(A−0.5) − κ_n·(nov_n−0.5)); 行为账本 (g_* = |R| EMA, metab_gain_rate);
    # 新奇度自归一化 (metab_nov_rate)。κ/α_g/div 为标定定稿值 (probe_p3b_calib:
    # out/v2-20260910-193044/probe_p3b_calib.json, 204/750 存活; div 取 leg 最接近 0.6)
    metab_gate_kappa_g: float = 4.0
    metab_gate_kappa_s: float = 4.0
    metab_gate_kappa_a: float = 1.0
    metab_gate_kappa_n: float = 0.5
    metab_gain_rate: float = 0.02
    metab_nov_rate: float = 0.05
    echo_seed_n: int = 16  # 回声种子字节数 (脚本 SEED_N 迁移)

    def dims(self) -> dict[str, int]:
        return {
            "l4": self.d_l4,
            "l2": self.d_l2,
            "l3": self.d_l3,
            "l5": self.d_l5,
            "l6": self.d_l6,
        }

    def param_count(self) -> int:
        """预估参数量 (含生成连接与时间核)."""
        d = self.dims()
        n = 0
        n += d["l4"] * self.d_input  # W_04
        n += d["l2"] * d["l4"]  # W_42
        n += d["l3"] * d["l2"]  # W_23
        n += d["l5"] * d["l3"]  # W_35
        n += d["l6"] * d["l5"]  # W_56
        n += d["l4"] * d["l5"]  # W_diff (L5_t → ΔL4_t)
        for k in ("l4", "l2", "l3", "l5", "l6"):
            n += d[k] * d[k]
        n += sum(d[k] for k in ("l4", "l2", "l3", "l5", "l6"))
        return n


def read_checkpoint_metadata(path: str) -> dict[str, str]:
    """只读 safetensors header, 不加载张量 (供 list --detail 等诊断用)."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        return f.metadata() or {}


# 检查点契约: state_ver 存在 = 完整检查点. 游离状态命名空间 attr./py.
_STATE_VER = "1"
_ATTR_PREFIX = "attr."
_PY_PREFIX = "py."
_CFG_FIELDS = frozenset(f.name for f in fields(CyreneModel))

# 活动 EMA 家族: 简称 → 所属层. 行索引 == 该层神经元, 故修剪必须 perm 重排 + 收缩
# (pruning._shrink_columns 读 net._active_ema_layer; 键序同时决定 _active_ema_init_mask 位序)
_ACTIVE_EMA_LAYER = {
    "w56": "l6", "w23": "l3", "w35": "l5",
    "wt4": "l4", "wt2": "l2", "wt3": "l3", "wt5": "l5", "wt6": "l6",
    "wsp": "l4", "wp54": "l5", "wp43": "l4", "w42": "l2",
    "b4": "l4", "b2": "l2", "b3": "l3", "b5": "l5", "b6": "l6",
}


class DensePCNet(nn.Module):
    """PPA 闭环网络 (门面: 权重声明 + 引擎委托)."""

    def __init__(self, config: CyreneModel | None = None):
        super().__init__()
        self.cfg = config or CyreneModel()
        d = self.cfg.dims()

        # 前馈权重，自下而上感知
        self._in_dim = self.cfg.d_input * (2 if self.cfg.input_history else 1)
        self.W_04 = nn.Parameter(torch.empty(d["l4"], self._in_dim, dtype=torch.float16))
        self.W_42 = nn.Parameter(torch.empty(d["l2"], d["l4"], dtype=torch.float16))
        self.W_23 = nn.Parameter(torch.empty(d["l3"], d["l2"], dtype=torch.float16))
        self.W_35 = nn.Parameter(torch.empty(d["l5"], d["l3"], dtype=torch.float16))
        self.W_56 = nn.Parameter(torch.empty(d["l6"], d["l5"], dtype=torch.float16))

        # 世界模型: W_diff 在 L4 空间预测 Δz4; W_state_pred 独立供表示层预测误差
        self.W_diff = nn.Parameter(torch.empty(d["l4"], d["l4"], dtype=torch.float16))
        self.b_diff = nn.Parameter(torch.zeros(d["l4"], dtype=torch.float16))
        self.W_state_pred = nn.Parameter(torch.empty(d["l4"], d["l4"], dtype=torch.float16))

        # 自组织预测引擎: 层间局部预测矩阵
        self.W_pred_54 = nn.Parameter(torch.empty(d["l5"], d["l4"], dtype=torch.float16))
        self.W_pred_43 = nn.Parameter(torch.empty(d["l4"], d["l3"], dtype=torch.float16))

        # 竞争性概念绑定层
        self.bind_slot_dim = 32
        self._lm_in = d["l4"] * (1 + self.cfg.mem_k0) + self.bind_slot_dim
        self.W_bind = nn.Parameter(torch.empty(d["l4"], self.bind_slot_dim, dtype=torch.float16))
        self.register_buffer("_theta_bind", torch.zeros(self.bind_slot_dim, dtype=torch.float16))
        self.E_bind_col = nn.Parameter(torch.zeros(self.bind_slot_dim, self.bind_slot_dim, dtype=torch.float16))

        # 槽自循环 (CA3 循环侧支): z_bind 携带自身历史
        self.W_bind_self = nn.Parameter(torch.empty(self.bind_slot_dim, self.bind_slot_dim, dtype=torch.float16))
        self.E_bind_self = nn.Parameter(torch.zeros(self.bind_slot_dim, self.bind_slot_dim, dtype=torch.float16))

        # 自发活动发生器
        self.register_buffer("_intr_cnt", torch.zeros(1, dtype=torch.float16))
        self.register_buffer(
            "_intr_sin",
            torch.tensor(
                [0.5 + 0.5 * math.sin(2.0 * math.pi * i / 20.0) for i in range(20)],
                dtype=torch.float16,
            ),
        )
        self.register_buffer("_intr_omega", torch.tensor(0.3, dtype=torch.float16))

        # 三尺度内源节律振荡器 (阶梯方波, fp16 可精确表示)
        for osc_name, osc_n in (("f", 64), ("m", 256), ("s", 1024)):
            self.register_buffer(f"_osc_{osc_name}_cnt", torch.zeros(1, dtype=torch.float16))
            self.register_buffer(
                f"_osc_{osc_name}_tab",
                (1.0 - 2.0 * torch.arange(osc_n, dtype=torch.float16) / osc_n),
            )

        # 内建能量约束活动基线 (每可塑性矩阵一个 EMA)
        for aen, lyr in _ACTIVE_EMA_LAYER.items():
            self.register_buffer(f"_active_ema_{aen}", torch.zeros(d[lyr], dtype=torch.float16))
        # 冷启动登记: 集合给热路径 (免费成员判定), mask 给持久化. 缺 mask 键的旧检查点
        # 解码为 0 → 集合空 → 首次用即快照, 与旧行为一致. 名称表与 buffer 同源, 不会漂移
        self._active_ema_names = tuple(f"_active_ema_{aen}" for aen in _ACTIVE_EMA_LAYER)
        self.register_buffer("_active_ema_init_mask", torch.zeros(1, dtype=torch.int32))
        self._active_ema_init: set[str] = set()

        # STP 资源慢变量 (每递归层 + z_bind)
        stp_layers = [("l4", d["l4"]), ("l2", d["l2"]), ("l3", d["l3"]), ("l5", d["l5"]), ("l6", d["l6"]), ("bind", self.bind_slot_dim)]
        for sln, sdim in stp_layers:
            self.register_buffer(f"_stp_r_{sln}", torch.ones(sdim, dtype=torch.float16))
            self.register_buffer(f"_stp_active_ema_{sln}", torch.full((sdim,), 0.01, dtype=torch.float16))
            tau_log = (
                torch.log(torch.tensor(self.cfg.stp_tau_min, dtype=torch.float32))
                + (torch.log(torch.tensor(self.cfg.stp_tau_max, dtype=torch.float32))
                   - torch.log(torch.tensor(self.cfg.stp_tau_min, dtype=torch.float32)))
                * torch.rand(sdim)
            )
            self.register_buffer(f"_stp_tau_{sln}", torch.exp(tau_log).to(torch.float16))
            self.register_buffer(f"_stp_u_{sln}", torch.full((sdim,), self.cfg.stp_u_init, dtype=torch.float16))
        self._stp_r_end: dict[str, torch.Tensor] = {}

        # 动作读出矩阵 W_act
        self.W_act = nn.Parameter(torch.empty(self.bind_slot_dim, self.cfg.d_act, dtype=torch.float16))
        self.register_buffer("_freq_act", torch.full((self.cfg.d_input,), 1.0 / 256.0, dtype=torch.float16))
        self.register_buffer("_s_ema_n", torch.ones(8, dtype=torch.float16))

        # 竞争性记忆单元群
        self.register_buffer("_mem_m", torch.zeros(self.cfg.mem_k0, d["l4"], dtype=torch.float16))
        self.register_buffer(
            "_mem_a",
            torch.tensor(
                [0.5, 0.125, 0.03125, 0.015625, 0.0078125, 0.00390625][: self.cfg.mem_k0],
                dtype=torch.float16,
            ),
        )
        self.register_buffer("_mem_g", torch.full((self.cfg.mem_k0,), self.cfg.mem_g_max / 2.0, dtype=torch.float16))
        self.register_buffer("_mem_q", torch.zeros(self.cfg.mem_k0, dtype=torch.float16))
        # 出生冷却: int32 buffer 为持久化载体, _mem_birth_py 为热路径 CPU 镜像
        # (对 GPU 张量做步级阈值比较是隐式 .item() 同步排空). 两者同步递增, load 回填镜像
        self.register_buffer("_mem_birth_cd", torch.zeros(1, dtype=torch.int32))
        self._mem_birth_py = 0
        self.register_buffer("_mem_death_cnt", torch.zeros(self.cfg.mem_k0, dtype=torch.int32))
        self._mem_alt = 0
        self.register_buffer("_mem_err_ema", torch.zeros(1, dtype=torch.float16))
        self.register_buffer("_mem_err_long", torch.zeros(1, dtype=torch.float16))
        if self.cfg.mem_k0 > 6:
            raise ValueError("mem_k0 > 6 需要补充初始 α 谱")

        # 体内代谢账本: 随 state_dict 持久化 (P2 死亡契约消费);
        # _metab_F_prev_* < 0 = 行为首跑哨兵; _metab_df_mad_* == 0 = 该行为冷启动哨兵
        # P3-b: ΔF 按行为分账 (感知/回声各一轨 F_prev) — 行为内差分同构可比
        self.register_buffer("_metab_E", torch.zeros(1, dtype=torch.float16))
        self.register_buffer("_metab_E_ref", torch.zeros(1, dtype=torch.float16))
        self.register_buffer("_metab_df_mad_perc", torch.zeros(1, dtype=torch.float16))
        self.register_buffer("_metab_df_mad_eco", torch.zeros(1, dtype=torch.float16))
        # 饥荒契约进度 fp16 镜像 (应激出处: starve_cnt/metab_death_steps)
        self.register_buffer("_metab_famine_prog", torch.zeros(1, dtype=torch.float16))
        self.register_buffer("_metab_F_prev_perc", torch.full((1,), -1.0, dtype=torch.float16))
        self.register_buffer("_metab_F_prev_eco", torch.full((1,), -1.0, dtype=torch.float16))
        # P3-b 行为账本 (g_* = |R| EMA, 仅运行行为记账) + 行为门状态
        self.register_buffer("_metab_gain_perc", torch.zeros(1, dtype=torch.float16))
        self.register_buffer("_metab_gain_eco", torch.zeros(1, dtype=torch.float16))
        self.register_buffer("_metab_nov_ema", torch.full((1,), 1e-3, dtype=torch.float16))
        self.register_buffer("_metab_nov", torch.full((1,), 0.5, dtype=torch.float16))
        self.register_buffer("_metab_psay", torch.full((1,), 0.5, dtype=torch.float16))
        self.register_buffer("_metab_gate_hit", torch.zeros(1, dtype=torch.float16))
        self.register_buffer("_gate_rand", torch.full((1,), 0.5, dtype=torch.float16))
        # 死亡契约计数器: 连续饥饿 / 连续沉默 / 死亡回合计数, 随 state_dict 持久化
        self.register_buffer("_metab_starve_cnt", torch.zeros(1, dtype=torch.int32))
        self.register_buffer("_metab_silence_cnt", torch.zeros(1, dtype=torch.int32))
        self.register_buffer("_metab_death_round_cnt", torch.zeros(1, dtype=torch.int32))
        # F 慢基线 (P2 基线锚): high = f_now > base·(1+κ); base==0 = 冷启动哨兵
        self.register_buffer("_metab_F_base", torch.zeros(1, dtype=torch.float16))
        # P3-c 成本遥测 (每步覆写, 报告/验证用): 看/学/记/说分量 + 总支出
        self.register_buffer("_metab_cost_perc", torch.zeros(1, dtype=torch.float16))
        self.register_buffer("_metab_cost_learn", torch.zeros(1, dtype=torch.float16))
        self.register_buffer("_metab_cost_mem", torch.zeros(1, dtype=torch.float16))
        self.register_buffer("_metab_cost_say", torch.zeros(1, dtype=torch.float16))
        self.register_buffer("_metab_cost_tot", torch.zeros(1, dtype=torch.float16))
        # P3-a: τ 低通信号滤波器 (ε 差分 EMA, 语义同 de_mad 家族; 持久化保续跑一致)
        self.register_buffer("_metab_tau_d", torch.zeros(1, dtype=torch.float16))
        # 语言带自校准锚 (感知相位 ε 中心/弥散 EMA, 恒温器输入): 跨 load 持久化,
        # 否则续跑后锚归零需 ~200 步重建. NaN = 冷启动哨兵 (ε ∈ [0,1) 永不 NaN);
        # _lang_eps_cold 为其 CPU 镜像 — 热路径判 NaN 是一次隐式 .item() 同步, 故镜像在 load 一次性恢复
        self.register_buffer("_lang_eps_ema", torch.full((), float("nan"), dtype=torch.float16))
        self.register_buffer("_lang_eps_mad", torch.zeros((), dtype=torch.float16))
        self._lang_eps_cold = True
        # 生命计数 (P2): 跨 load 持久化的机体年龄, warmup 判据用它 (旧检查点缺键 → 0 = P2 机制幼年期)
        self.register_buffer("_life_cnt", torch.zeros(1, dtype=torch.int32))

        # 内部 EMA 频率 (纯模型内部累计 target 分布)
        self.register_buffer("_freq", torch.full((self.cfg.d_input,), 1.0 / 256.0, dtype=torch.float16))

        # 张量规范共享缓冲 (热路径零新建): 零/壹标量、pad 段 (cat 前后缀按 active 切视图)、
        # 打印掩码、序列掩码 — 全部随 state_dict 持久化, 每步复用
        self.register_buffer("_zero1", torch.zeros(1, dtype=torch.float16))
        self.register_buffer("_one1", torch.ones(1, dtype=torch.float16))
        self.register_buffer("_intr_d0", torch.full((1,), 0.5, dtype=torch.float16))
        self.register_buffer("_one_s", torch.ones(1, self.cfg.max_seq_len - 1, 1, dtype=torch.float16))
        self.register_buffer(
            "_padmax", torch.zeros(1, 1, max(d["l4"], self.cfg.d_input, 256), dtype=torch.float16)
        )
        _mp = torch.zeros(256, dtype=torch.float16)
        _mp[32:] = 1.0
        self.register_buffer("_mask_print", _mp)
        self.register_buffer("_learn_mask_all", torch.ones(self.cfg.max_seq_len - 1, dtype=torch.bool))
        # continuation/_predict 工作缓冲: 续写流 (定长预分配+长度游标)、UTF-8 状态、L0 one-hot、
        # 记忆序列、ACh 噪声 — 全部就地覆写, 零热路径新建
        self.register_buffer(
            "_cont_cur",
            torch.zeros(1, self.cfg.max_seq_len + self.cfg.free_run_window + 64, dtype=torch.long),
        )
        self.register_buffer("_cont_expect", torch.zeros(1, dtype=torch.long))
        self.register_buffer(
            "_mem_m_seq",
            torch.zeros(1, self.cfg.max_seq_len, max(self.cfg.mem_k0, self.cfg.mem_k_max), d["l4"], dtype=torch.float16),
        )
        # 未来折扣目标/序列掩码 (predict 域): 预分配就地清零/置位, 零热路径新建
        self.register_buffer("_z4_fut_buf", torch.zeros(1, self.cfg.max_seq_len, d["l4"], dtype=torch.float16))
        self.register_buffer("_z5_fut_buf", torch.zeros(1, self.cfg.max_seq_len, d["l5"], dtype=torch.float16))
        self.register_buffer("_seq_arange", torch.arange(self.cfg.max_seq_len, dtype=torch.int64))
        self.register_buffer("_arange20", torch.arange(20, dtype=torch.int64))
        self.register_buffer("_maskS_a", torch.zeros(self.cfg.max_seq_len + 1, dtype=torch.bool))
        self.register_buffer("_maskS_b", torch.zeros(self.cfg.max_seq_len + 1, dtype=torch.bool))

        # 非线性混合层: h = zh @ W1; logits = h @ W_lm
        self.d_h = 256
        self.W1 = nn.Parameter(torch.empty(self._lm_in, self.d_h, dtype=torch.float16))
        self.W_lm = nn.Parameter(torch.empty(self.d_h, self.cfg.d_input, dtype=torch.float16))
        self.W_lm_2 = nn.Parameter(torch.empty(self.d_h, self.cfg.d_input, dtype=torch.float16))
        self.bias_lm = nn.Parameter(torch.zeros(self.cfg.d_input, dtype=torch.float16))

        # 多尺度软加权时间窗
        self.register_buffer("_w_soft", torch.tensor([0.1, 0.8, 0.1], dtype=torch.float16))
        # UTF-8 语法阻断掩码 (生成路径逐字节向量化: 原逐字节 .item() 布尔分支 =
        # 每回声步 ~120 次同步排空, GPU 利用率主嫌; 禁止集/值与逐位分支逐位等价).
        # block_cnt = 字符中途禁字节集 {0x00-0x7F}∪{0xC0-0xFF}; block_start = 边界禁
        # 字节集 {0x80-0xC1} (0xC0-0xC1 overlong) — 与 forward.py 历史分支一字不差.
        _b_cnt = torch.zeros(256, dtype=torch.bool)
        _b_cnt[0x00:0x80] = True
        _b_cnt[0xC0:0x100] = True
        self.register_buffer("_utf8_block_cnt", _b_cnt)
        _b_start = torch.zeros(256, dtype=torch.bool)
        _b_start[0x80:0xC2] = True
        self.register_buffer("_utf8_block_start", _b_start)
        self.register_buffer("_e_ema_2", torch.tensor(0.05, dtype=torch.float16))
        self.register_buffer("_e_ema_4", torch.tensor(0.05, dtype=torch.float16))
        self.register_buffer("_e_ema_8", torch.tensor(0.05, dtype=torch.float16))
        # 多尺度差分窗环形缓冲 (GPU 索引 [4,L,L] + 槽位 [1,L,L] — 图捕获兼容, 零热路径新建;
        # 旧四缓冲+python 索引迁移见 load)
        self.register_buffer("_dw_buf", torch.zeros(4, d["l4"], d["l4"], dtype=torch.float16))
        self.register_buffer("_buf_i", torch.zeros(1, dtype=torch.int64))
        self.register_buffer("_dw_slot", torch.zeros(1, d["l4"], d["l4"], dtype=torch.float16))
        self.register_buffer("_theta_w", torch.full((d["l4"],), 0.01, dtype=torch.float16))
        self.register_buffer("_theta_w04", torch.full((d["l4"],), 0.01, dtype=torch.float16))
        self.register_buffer("_theta_wt4", torch.zeros(d["l4"], dtype=torch.float16))

        # 时序权重 (每层时间核)
        self.W_t4 = nn.Parameter(torch.empty(d["l4"], d["l4"], dtype=torch.float16))
        self.W_t2 = nn.Parameter(torch.empty(d["l2"], d["l2"], dtype=torch.float16))
        self.W_t3 = nn.Parameter(torch.empty(d["l3"], d["l3"], dtype=torch.float16))
        self.W_t5 = nn.Parameter(torch.empty(d["l5"], d["l5"], dtype=torch.float16))
        self.W_t6 = nn.Parameter(torch.empty(d["l6"], d["l6"], dtype=torch.float16))

        # 层偏置
        self.bias_l4 = nn.Parameter(torch.zeros(d["l4"], dtype=torch.float16))
        self.bias_l2 = nn.Parameter(torch.zeros(d["l2"], dtype=torch.float16))
        self.bias_l3 = nn.Parameter(torch.zeros(d["l3"], dtype=torch.float16))
        self.bias_l5 = nn.Parameter(torch.zeros(d["l5"], dtype=torch.float16))
        self.bias_l6 = nn.Parameter(torch.zeros(d["l6"], dtype=torch.float16))

        # 动态生长状态
        self.active_size = {"l4": d["l4"], "l2": d["l2"], "l3": d["l3"], "l5": d["l5"], "l6": d["l6"]}
        # 全局步计数: int32 buffer 为持久化载体, _step_py 为热路径 CPU 镜像 (同 _behavior_py)
        self.register_buffer("_step_counter", torch.zeros(1, dtype=torch.int32))
        self._step_py = 0
        # 代谢哨兵 python 标志 (代替对 GPU 张量的 `if _metab_F_prev < 0:` — 那是每步
        # 一次隐式 .item() 同步排空; 标志由 load 从 F_prev 一次性恢复)
        # P3-b: 每行为独立哨兵 (_settled_perc/_settled_eco), _metab_settled = 全局首步
        self._metab_settled = False
        self._metab_mad_cold_perc = True
        self._metab_mad_cold_eco = True
        self._settled_perc = False
        self._settled_eco = False
        # P3-b: 行为门 CPU 路由镜像 (load 后未调 sync_gate 前默认 False = 感知)
        self._behavior_py = False
        # 死缓簿: 每层行状态 (int8 是否死缓 / int16 死缓龄), 随 state_dict 持久化.
        # 零值 = 从未进入死缓, 与旧检查点缺键同义 (修剪引擎原以 None 表达此态)
        for _sln in ("l4", "l2", "l3", "l5", "l6"):
            self.register_buffer(f"_death_row_{_sln}", torch.zeros(self.active_size[_sln], dtype=torch.int8))
            self.register_buffer(
                f"_probation_counter_{_sln}", torch.zeros(self.active_size[_sln], dtype=torch.int16)
            )

        # 神经调制与竞争机制
        self.register_buffer("_ent_ema", torch.tensor(5.5, dtype=torch.float16))
        self.register_buffer("_ent_buf", torch.zeros(20, dtype=torch.float16))
        self.register_buffer("_t_center", torch.arange(20, dtype=torch.float16) - 9.5)
        self.register_buffer("_t_denom", (torch.arange(20, dtype=torch.float16) - 9.5).square().sum())
        self._ent_i = 0
        self.register_buffer("_traction_scale", torch.tensor(1.0, dtype=torch.float16))
        self.register_buffer("_theta_novelty", torch.full((1,), 0.001, dtype=torch.float16))
        for ln, dim in (("l4", d["l4"]), ("l2", d["l2"]), ("l3", d["l3"]), ("l5", d["l5"]), ("l6", d["l6"])):
            self.register_buffer(f"_theta_{ln}", torch.full((dim,), 0.01, dtype=torch.float16))
        self.register_buffer("_theta_wlm", torch.full((self.cfg.d_input,), 0.01, dtype=torch.float16))
        self.register_buffer("_theta_wlm2", torch.full((self.cfg.d_input,), 0.01, dtype=torch.float16))

        # Foldiak 反赫布去同质化矩阵 (零起步 = 无抑制)
        self.E_l5 = nn.Parameter(torch.zeros(d["l5"], d["l5"], dtype=torch.float16))
        self.E_42 = nn.Parameter(torch.zeros(d["l2"], d["l2"], dtype=torch.float16))
        self.E_23 = nn.Parameter(torch.zeros(d["l3"], d["l3"], dtype=torch.float16))
        self.E_t2 = nn.Parameter(torch.zeros(d["l2"], d["l2"], dtype=torch.float16))
        self.E_t3 = nn.Parameter(torch.zeros(d["l3"], d["l3"], dtype=torch.float16))
        self.E_t4 = nn.Parameter(torch.zeros(d["l4"], d["l4"], dtype=torch.float16))
        self.E_t5 = nn.Parameter(torch.zeros(d["l5"], d["l5"], dtype=torch.float16))
        self.E_t6 = nn.Parameter(torch.zeros(d["l6"], d["l6"], dtype=torch.float16))
        self.E_bind = nn.Parameter(torch.zeros(d["l4"], d["l4"], dtype=torch.float16))
        self.E_04 = nn.Parameter(torch.zeros(d["l4"], d["l4"], dtype=torch.float16))
        # 侧抑制矩阵 (L5 激活去相关)
        self.M_l5 = nn.Parameter(torch.zeros(d["l5"], d["l5"], dtype=torch.float16))
        self.register_buffer("_gain_mask", (0.5 + torch.rand(d["l5"], d["l3"])).to(torch.float16))
        self.register_buffer("_gain_l3", (0.5 + torch.rand(d["l3"], d["l2"])).to(torch.float16))

        # 突触资格迹 (每 Hebbian 外积矩阵同形, 零初始化)
        _hebb_para_names = (
            "W_04", "W_42", "W_23", "W_35", "W_56",
            "W_t4", "W_t2", "W_t3", "W_t5", "W_t6",
            "W_diff", "W_state_pred", "W_pred_54", "W_pred_43",
            "W_bind", "W_bind_self", "W_act",
            "W_lm", "W_lm_2", "W1",
        )
        for _wn in _hebb_para_names:
            self.register_buffer(f"{_wn}_elig", torch.zeros_like(getattr(self, _wn).data))

        self._init_weights()

        # 引擎委托
        from model.dense.forward import ForwardEngine
        from model.dense.learning import LearningEngine
        from model.dense.pruning import PruningEngine

        self.forward_engine = ForwardEngine(self)
        self.learning_engine = LearningEngine(self)
        self.pruner = PruningEngine(self)

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "bias" in name:
                continue
            if name in ("E_l5", "E_42", "E_23", "M_l5", "E_t2", "E_t3", "E_t4", "E_t5", "E_t6", "E_bind", "E_bind_col", "E_04", "E_bind_self"):
                continue
            if name == "W_act":
                nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(self.bind_slot_dim))
            if name == "W_bind_self":
                nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(self.bind_slot_dim))
            if name in ("W_lm", "W_lm_2"):
                nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(self.d_h))
            elif name == "W1":
                nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(p.shape[0]))
            else:
                nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(p.shape[-1]))

    # 门面: 一行委托

    def forward(self, byte_ids: torch.Tensor) -> dict:
        """推理前馈: 返回未来预测偏差."""
        return self.forward_engine.forward(byte_ids)

    def generate(
        self, prompt: str, n_tokens: int = 40, temperature: float = 0.7, dev: torch.device | None = None
    ) -> bytes:
        """行动: L4 状态 + 预测差分 → W_lm 解码生成字节."""
        return self.forward_engine.generate(prompt, n_tokens=n_tokens, temperature=temperature, dev=dev)

    def _predict(self, byte_ids: torch.Tensor, store_state: bool = True, is_inference: bool = False) -> dict:
        """核心前馈: 感知 (L0→L6) + 去相关 + 增量预测."""
        return self.forward_engine._predict(byte_ids, store_state=store_state, is_inference=is_inference)

    def learn(self, byte_ids: torch.Tensor | None = None, closed_loop: bool = False, free_run: bool = False,
              force_perception: bool = False) -> dict:
        """Hebbian 学习: 前馈 → 逐域误差 → 局部权重更新, 无反向传播.

        P3-b 节律内生: byte_ids 非 None 时由行为门 (模型内部状态) 决定本次走感知还是
        回声 (learn() 内部路由); force_perception=True 强制感知 (首步/恢复期旁路).
        """
        return self.learning_engine.learn(byte_ids, closed_loop=closed_loop, free_run=free_run,
                                          force_perception=force_perception)

    def sync_gate(self) -> None:
        """P3-b 每步一次的路由交接: 读门决策 CPU 镜像 (供下一次 learn 路由).

        门公式与采样全 GPU; learn 全异步发射,
        .item() 是每步唯一同步点 (在此排空).
        _behavior_py 是 CPU 镜像 (非持久化), load 后默认 False=感知.
        """
        self._behavior_py = bool(self._metab_gate_hit.item())

    def maybe_prune(self, step: int) -> str | None:
        """代谢触发修剪接缝 (P2 下探缺失 + P3-b 对称牙): 连续饥饿或连续沉默达阈值 → 死亡回合.

        墙钟已退役. 轮询周期 8 步 (触发延迟 ≤8 步, 相对 N=250 的滞回可忽略);
        warmup 期计数器继续积累但回合被挡 (发育期免回合, 不免登记).
        死亡回合 = 记忆单元饥荒击杀 (不可逆) + 拓扑修剪 (最弱 5%, 只减不增).
        慢性饥荒/长期沉默 → 计数重新积累 → 反复死亡直到兜底 (l4=512/其余=128/K=1) 后自动 no-op.
        返回触发原因 (starve/silence) 供事件遥测; 未触发返回 None.
        """
        if step % 8 != 0:
            return None
        # 发育期免疫: 回合计 _life_cnt (跨 load 持久化年龄) 判定, 计数器照常积累
        if float(self._life_cnt.item()) <= self.cfg.prune_warmup:
            return None
        starve = float(self._metab_starve_cnt.item()) >= self.cfg.metab_death_steps
        silence = float(self._metab_silence_cnt.item()) >= self.cfg.metab_silence_steps
        if not (starve or silence):
            return None
        cause = "starve" if starve else "silence"
        self._metab_starve_cnt.zero_()
        self._metab_silence_cnt.zero_()
        self._metab_death_round_cnt.add_(1)
        self.learning_engine._mem_famine_kill()  # 先杀: reshape 在当前布局上自洽
        self.pruner._prune()  # 后剪: _sync_l4_aux 按新 K 现场取数
        return cause

    def _prune(self):
        self.pruner._prune()

    def _extra_tensors(self) -> dict[str, torch.Tensor]:
        """游离运行时状态: 既非参数也非 buffer 的张量 (惰性建立, 跨调用存活).

        别名 (如 _metab_stress → _metab_famine_prog) 靠存储指针排除 — 另存一份会让
        load 后变成两个不同张量, 破坏别名语义. 形状随调用序列长度变, 故不可在
        __init__ 预声明为 buffer.
        """
        owned = {v.untyped_storage().data_ptr()
                 for v in itertools.chain(self.parameters(), self.buffers())}
        out: dict[str, torch.Tensor] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                if v.untyped_storage().data_ptr() not in owned:
                    out[k] = v
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, torch.Tensor) and v2.untyped_storage().data_ptr() not in owned:
                        out[f"{k}.{k2}"] = v2
        return out

    def _py_counters(self) -> dict[str, int]:
        """热路径 CPU 计数器 (对张量做步级比较是隐式 .item() 同步, 故留 Python int).

        必须显式列举: 派生值 (d_h/_in_dim/_lm_in) 与 buffer 镜像 (_step_py/_mem_birth_py)
        在通用扫描下与真状态无法区分, 误存会与 buffer 冲突.
        """
        stats = getattr(self, "_block_stats", None) or {}
        return {
            "_mem_alt": self._mem_alt,
            "_ent_i": self._ent_i,
            "_block_rep": int(stats.get("rep", 0)),
            "_block_gen": int(stats.get("gen", 0)),
        }

    def _apply(self, fn, recurse: bool = True):
        """游离张量随模型迁移设备 (nn.Module._apply 只搬参数与 buffer)."""
        owned = {v.untyped_storage().data_ptr()
                 for v in itertools.chain(self.parameters(), self.buffers())}
        out = super()._apply(fn, recurse)
        for k, v in list(self.__dict__.items()):
            if k in self._parameters or k in self._buffers:
                continue
            if isinstance(v, torch.Tensor):
                if v.untyped_storage().data_ptr() not in owned:
                    out.__dict__[k] = fn(v)
            elif isinstance(v, dict):
                for k2, v2 in list(v.items()):
                    if isinstance(v2, torch.Tensor) and v2.untyped_storage().data_ptr() not in owned:
                        v[k2] = fn(v2)
        return out

    def save(self, path: str):
        """保存模型完整状态: 参数 + buffer + 游离运行时张量 + CPU 计数器.

        metadata 的 state_ver 是契约锚点 — 无它即 weights-only 快照, load 拒收.
        """
        import os

        from safetensors.torch import save_file

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        blob = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        blob.update({f"{_ATTR_PREFIX}{k}": v.detach().cpu().contiguous()
                     for k, v in self._extra_tensors().items()})
        meta = {"state_ver": _STATE_VER, "config": json.dumps(asdict(self.cfg))}
        meta.update({f"{_PY_PREFIX}{k}": str(v) for k, v in self._py_counters().items()})
        save_file(blob, path, metadata=meta)

    @classmethod
    def config_from_checkpoint(cls, path: str) -> CyreneModel:
        """读检查点 config 快照 (不加载张量, 不覆盖维度)."""
        meta = read_checkpoint_metadata(path)
        if "config" not in meta:
            raise ValueError(f"{path} 缺 config 快照 (有 state_ver 就必须有)")
        kv = {k: v for k, v in json.loads(meta["config"]).items() if k in _CFG_FIELDS}
        return CyreneModel(**kv)

    @classmethod
    def load(cls, path: str, config: CyreneModel | None = None) -> DensePCNet:
        """加载完整检查点 (含修剪后形状对齐, 重设 active_size)."""
        from safetensors.torch import load_file

        meta = read_checkpoint_metadata(path)
        if "state_ver" not in meta:
            raise ValueError(
                f"{path} 不是完整检查点 (缺 state_ver): weights-only 快照不含游离运行时状态 "
                f"(_survival_signal / _stp_r_end / _last_z* 等), 续跑必然偏离. "
                f"完整检查点由当前 DensePCNet.save 写出."
            )
        sd = load_file(path, device="cpu")
        # 维度一律以检查点形状为准 (config 只补非维度字段): 修剪是物理切片, 出生 dims
        # 与现存活尺寸故意不等, 若让 config 覆盖会得到 active_size 与 _mem_m 不一致的模型.
        dims = {
            "d_l4": sd["W_04"].shape[0],
            "d_l2": sd["W_42"].shape[0],
            "d_l3": sd["W_23"].shape[0],
            "d_l5": sd["W_56"].shape[1],
            "d_l6": sd["W_56"].shape[0],
            "input_history": sd["W_04"].shape[1] != 256,
        }
        if config is None:
            snap = meta.get("config")
            if snap is None:
                raise ValueError(f"{path} 缺 config 快照 (有 state_ver 就必须有)")
            kv = {k: v for k, v in json.loads(snap).items() if k in _CFG_FIELDS}
            config = replace(CyreneModel(**kv), **dims)
        else:
            config = replace(config, **dims)
        net = cls(config)
        # 记忆单元群是运行时状态, 按检查点重建 K 形缓冲与 W1 (修剪/生长后形状不符)
        for _bn in ("_mem_m", "_mem_g", "_mem_q", "_mem_a", "_mem_death_cnt"):
            if _bn in sd and sd[_bn].shape != getattr(net, _bn).shape:
                net.register_buffer(_bn, torch.zeros_like(sd[_bn]))
        if "W1" in sd and sd["W1"].shape != net.W1.shape:
            net.W1 = nn.Parameter(torch.zeros_like(sd["W1"], dtype=torch.float16))
            net.register_buffer("W1_elig", torch.zeros_like(sd["W1"], dtype=torch.float16))
            net._lm_in = sd["W1"].shape[0]
        nsd = net.state_dict()
        for k, v in sd.items():
            if k not in nsd:
                continue
            if nsd[k].shape == v.shape:
                nsd[k] = v
            else:
                idx = tuple(slice(0, min(a, b)) for a, b in zip(nsd[k].shape, v.shape))
                nsd[k][idx] = v[idx]
        net.load_state_dict(nsd)
        net.active_size = {
            "l4": net.W_04.shape[0],
            "l2": net.W_42.shape[0],
            "l3": net.W_23.shape[0],
            "l5": net.W_56.shape[1],
            "l6": net.W_56.shape[0],
        }
        # 代谢哨兵标志一次性恢复 (CPU 侧): F_prev<0 → 未结算 (首步哨兵); MAD≤0 → 冷启动
        net._metab_settled = float(net._metab_F_prev_perc.item()) >= 0.0
        net._settled_perc = net._metab_settled
        net._settled_eco = float(net._metab_F_prev_eco.item()) >= 0.0
        net._metab_mad_cold_perc = float(net._metab_df_mad_perc.item()) <= 0.0
        net._metab_mad_cold_eco = float(net._metab_df_mad_eco.item()) <= 0.0
        net._lang_eps_cold = bool(torch.isnan(net._lang_eps_ema).item())
        # 热路径 CPU 镜像一次性回填 (镜像每步与 buffer 同步递增, 不参与 state_dict)
        net._step_py = int(net._step_counter.item())
        net._mem_birth_py = int(net._mem_birth_cd.item())
        from model.dense.learning._common import restore_active_ema_init

        net._active_ema_init = restore_active_ema_init(net)
        net._mem_alt = int(meta[f"{_PY_PREFIX}_mem_alt"])
        net._ent_i = int(meta[f"{_PY_PREFIX}_ent_i"])
        net._block_stats = {
            "rep": int(meta[f"{_PY_PREFIX}_block_rep"]),
            "gen": int(meta[f"{_PY_PREFIX}_block_gen"]),
        }
        # 游离运行时张量: state_ver 门已保证存在, 故直接索引字典形态
        for k, v in sd.items():
            if not k.startswith(_ATTR_PREFIX):
                continue
            name = k[len(_ATTR_PREFIX):]
            if "." in name:
                dk, dkey = name.split(".", 1)
                getattr(net, dk)[dkey] = v
            else:
                setattr(net, name, v)
        # _mem_m 对齐活性 L4 (修剪后检查点)
        if net._mem_m.shape[1] != net.active_size["l4"]:
            old_m = net._mem_m.data
            net.register_buffer("_mem_m", old_m[:, : net.active_size["l4"]].contiguous())
            del old_m
        return net
