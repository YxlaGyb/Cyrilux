"""Cyrene 模型定义

密集 PPA 感知-预测-行动闭环网络.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DensePCConfig:
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
    prune_interval: int = 1000
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


class DensePCNet(nn.Module):
    """PPA 闭环网络 (门面: 权重声明 + 引擎委托)."""

    def __init__(self, config: DensePCConfig | None = None):
        super().__init__()
        self.cfg = config or DensePCConfig()
        d = self.cfg.dims()

        # 前馈权重 (自下而上感知)
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
        act_ema_dims = {
            "w56": d["l6"], "w23": d["l3"], "w35": d["l5"],
            "wt4": d["l4"], "wt2": d["l2"], "wt3": d["l3"], "wt5": d["l5"], "wt6": d["l6"],
            "wsp": d["l4"], "wp54": d["l5"], "wp43": d["l4"], "w42": d["l2"],
            "b4": d["l4"], "b2": d["l2"], "b3": d["l3"], "b5": d["l5"], "b6": d["l6"],
        }
        for aen, adim in act_ema_dims.items():
            self.register_buffer(f"_active_ema_{aen}", torch.zeros(adim, dtype=torch.float16))
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
        self._fr_state: dict[str, torch.Tensor] = {}
        self._stp_r_end: dict[str, torch.Tensor] = {}

        # 动作读出矩阵 W_act
        self.W_act = nn.Parameter(torch.empty(self.bind_slot_dim, self.cfg.d_act, dtype=torch.float16))
        self.register_buffer("_theta_act", torch.full((self.cfg.d_input,), 0.01, dtype=torch.float16))
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
        self._mem_birth_cd = 0
        self.register_buffer("_mem_death_cnt", torch.zeros(self.cfg.mem_k0, dtype=torch.int32))
        self._mem_alt = 0
        self.register_buffer("_mem_err_ema", torch.zeros(1, dtype=torch.float16))
        self.register_buffer("_mem_err_long", torch.zeros(1, dtype=torch.float16))
        if self.cfg.mem_k0 > 6:
            raise ValueError("mem_k0 > 6 需要补充初始 α 谱")

        # 内部 EMA 频率 (纯模型内部累计 target 分布)
        self.register_buffer("_freq", torch.full((self.cfg.d_input,), 1.0 / 256.0, dtype=torch.float16))

        # 非线性混合层: h = zh @ W1; logits = h @ W_lm
        self.d_h = 256
        self.W1 = nn.Parameter(torch.empty(self._lm_in, self.d_h, dtype=torch.float16))
        self.W_lm = nn.Parameter(torch.empty(self.d_h, self.cfg.d_input, dtype=torch.float16))
        self.W_lm_2 = nn.Parameter(torch.empty(self.d_h, self.cfg.d_input, dtype=torch.float16))
        self.bias_lm = nn.Parameter(torch.zeros(self.cfg.d_input, dtype=torch.float16))

        # 多尺度软加权时间窗
        self.register_buffer("_w_soft", torch.tensor([0.1, 0.8, 0.1], dtype=torch.float16))
        self.register_buffer("_e_ema_2", torch.tensor(0.05, dtype=torch.float16))
        self.register_buffer("_e_ema_4", torch.tensor(0.05, dtype=torch.float16))
        self.register_buffer("_e_ema_8", torch.tensor(0.05, dtype=torch.float16))
        for i in range(4):
            self.register_buffer(f"_dw_buf_{i}", torch.zeros(d["l4"], d["l4"], dtype=torch.float16))
        self._buf_i = 0
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
        self._step_counter = 0
        self._death_row: dict[str, torch.Tensor | None] = {
            "l4": None, "l2": None, "l3": None, "l5": None, "l6": None,
        }
        self._probation_counter: dict[str, torch.Tensor | None] = {
            "l4": None, "l2": None, "l3": None, "l5": None, "l6": None,
        }

        # 神经调制与竞争机制
        self.register_buffer("_ent_ema", torch.tensor(5.5, dtype=torch.float16))
        self.register_buffer("_ent_buf", torch.zeros(20, dtype=torch.float16))
        self.register_buffer("_t_center", torch.arange(20, dtype=torch.float16) - 9.5)
        self.register_buffer("_t_denom", (torch.arange(20, dtype=torch.float16) - 9.5).square().sum())
        self._ent_i = 0
        self.register_buffer("_traction_scale", torch.tensor(1.0, dtype=torch.float16))
        self.register_buffer("_z_slow", torch.zeros(d["l4"], dtype=torch.float16))
        self.register_buffer("_theta_novelty", torch.full((1,), 0.001, dtype=torch.float16))
        for ln, dim in (("l4", d["l4"]), ("l2", d["l2"]), ("l3", d["l3"]), ("l5", d["l5"]), ("l6", d["l6"])):
            self.register_buffer(f"_theta_{ln}", torch.full((dim,), 0.01, dtype=torch.float16))
        self.register_buffer("_theta_diff", torch.full((d["l4"],), 0.01, dtype=torch.float16))
        self.register_buffer("_theta_wlm", torch.full((self.cfg.d_input,), 0.01, dtype=torch.float16))
        self.register_buffer("_theta_wlm2", torch.full((self.cfg.d_input,), 0.01, dtype=torch.float16))
        self.register_buffer("_theta_pool", torch.full((4,), 0.01, dtype=torch.float16))

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

    def learn(self, byte_ids: torch.Tensor | None = None, closed_loop: bool = False, free_run: bool = False) -> dict:
        """Hebbian 学习: 前馈 → 逐域误差 → 局部权重更新, 无反向传播."""
        return self.learning_engine.learn(byte_ids, closed_loop=closed_loop, free_run=free_run)

    def maybe_prune(self, step: int) -> None:
        """墙钟修剪接缝: 修剪触发权由编排层显式交出."""
        if step > self.cfg.prune_warmup and step % self.cfg.prune_interval == 0:
            self.pruner._prune()

    def inject_world(self, E, E_ref=None, R=None, eps_ema=None, eps_mad=None) -> None:
        """生命-世界口: 显式接口位, 传标量/None 写入 net._world_*."""
        d = next(self.parameters()).device
        if E is not None:
            self._world_E = torch.tensor(E, dtype=torch.float16, device=d)
        if E_ref is not None:
            self._world_E_ref = torch.tensor(E_ref, dtype=torch.float16, device=d)
        if R is not None:
            self._world_R = torch.tensor(R, dtype=torch.float16, device=d)
        if eps_ema is not None:
            self._world_eps_ema = torch.tensor(eps_ema, dtype=torch.float16, device=d)
        if eps_mad is not None:
            self._world_eps_mad = torch.tensor(eps_mad, dtype=torch.float16, device=d)

    def _prune(self):
        self.pruner._prune()

    def save(self, path: str):
        """保存模型权重."""
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, config: DensePCConfig | None = None) -> DensePCNet:
        """加载模型权重 (含修剪后检查点: 按检查点形状对齐, 重设 active_size)."""
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if config is None:
            config = DensePCConfig(
                d_l4=sd["W_04"].shape[0],
                d_l2=sd["W_42"].shape[0],
                d_l3=sd["W_23"].shape[0],
                d_l5=sd["W_56"].shape[1],
                d_l6=sd["W_56"].shape[0],
                input_history=sd["W_04"].shape[1] != 256,
            )
        net = cls(config)
        # 旧键 → 新键 (活动统计缓冲更名), 旧检查点写回新键才不丢值
        for old, new in (("_act_ema_", "_active_ema_"), ("_stp_act_ema_", "_stp_active_ema_")):
            for k in list(sd.keys()):
                if old in k:
                    sd[new + k.split(old, 1)[1]] = sd.pop(k)
        # 旧 _m_pool 检查点一次性迁移到记忆单元群
        if "_m_pool" in sd and "_mem_m" not in sd:
            net._migrate_mem(sd)
        # 记忆单元群是运行时状态, 按检查点重建 K 形缓冲与 W1
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
        # 旧检查点 _stp_active_ema 可能为 0 → 统一垫到小正数
        for sln in ("l4", "l2", "l3", "l5", "l6", "bind"):
            buf = getattr(net, f"_stp_active_ema_{sln}")
            if buf.numel() > 0 and buf.abs().sum() == 0:
                buf.fill_(0.01)
        # _mem_m 对齐活性 L4 (修剪后检查点)
        if net._mem_m.shape[1] != net.active_size["l4"]:
            old_m = net._mem_m.data
            net.register_buffer("_mem_m", old_m[:, : net.active_size["l4"]].contiguous())
            del old_m
        return net

    def _migrate_mem(self, sd: dict) -> None:
        """旧 _m_pool 检查点 → 记忆单元群迁移 (一次性).

        旧 W1 行布局 [z4 | m2 | m8 | m32 | bind | 长池] → 新 [z4 | bind | 单元×k0].
        bind 恒在短池段后 (offset 4·dim_4); 不足 k0 的单元零起步.
        """
        dim_4 = sd["W_04"].shape[0]
        p = sd["_m_pool"].shape[0] // dim_4
        k0 = self.cfg.mem_k0
        bind_sz = self.bind_slot_dim
        old_w1 = sd["W1"]
        d_h = old_w1.shape[1]
        n_short = 3
        bind_off = (n_short + 1) * dim_4
        segs = [old_w1[(1 + i) * dim_4 : (2 + i) * dim_4] for i in range(n_short)]
        if p > n_short:
            segs += [
                old_w1[(bind_off + bind_sz + (i - n_short) * dim_4) : (bind_off + bind_sz + (i - n_short + 1) * dim_4)]
                for i in range(n_short, p)
            ]
        sd["W1"] = torch.cat(
            [old_w1[:dim_4], old_w1[bind_off : bind_off + bind_sz], *segs],
            dim=0,
        ).contiguous()
        if p < k0:
            sd["W1"] = torch.cat(
                [sd["W1"], torch.zeros((k0 - p) * dim_4, d_h, dtype=torch.float16)],
                dim=0,
            ).contiguous()
        m_segs = sd["_m_pool"].reshape(p, dim_4)
        cells = [m_segs[i] for i in range(p)]
        if p < k0:
            cells += [torch.zeros(dim_4, dtype=torch.float16)] * (k0 - p)
        sd["_mem_m"] = torch.stack(cells).contiguous()
        del sd["_m_pool"]
