"""PageStorage — 页式 GPU 张量分配器.

管理 TensorNeuronPool 中所有张量的页式存储:
- 神经元页 (4096 个/页): state, layer, position, channel, in_ptrs, out_ptrs, ...
- 突触页 (16384 个/页): weight, pre_id, post_id, trace, ...
- Topdown 页 (16384 个/页): td_weight, td_pre, td_post, ...
- LM Head: 随神经元页的列增长, [256, N] 稀疏视图

生长策略: freelist 耗尽时翻倍扩容 (类似 std::vector), O(log n) 次重分配.
对外暴露合并视图 (property), 与旧 torch.zeros(N) API 完全兼容.
"""

from __future__ import annotations

import torch

from .constants import K_FAN, N_STATE_FIELDS, PAGE_NEURONS, PAGE_SYNAPSES, PAGE_TD

# 默认内存预算 (2GB)
DEFAULT_MAX_MEMORY = 2_147_483_648


class MemoryBudgetError(RuntimeError):
    """池达到 max_memory_bytes 上限."""


def _next_pow2(n: int) -> int:
    """返回 ≥ n 的最小 2 的幂."""
    p = 1
    while p < n:
        p <<= 1
    return p


class PageStorage:
    """页式 GPU 张量存储, 对外暴露合并视图.

    Args:
        device: CUDA 设备或 "cpu"
        max_memory_bytes: 内存预算上限 (默认 2GB)
        initial_neurons: 初始神经元容量 (会被圆整到页大小)
        initial_synapses: 初始突触容量
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        max_memory_bytes: int = DEFAULT_MAX_MEMORY,
        initial_neurons: int = PAGE_NEURONS,
        initial_synapses: int = PAGE_SYNAPSES,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_memory_bytes = max_memory_bytes

        # 容量 (逻辑上限, 合并视图大小)
        self._N = _next_pow2(max(initial_neurons, PAGE_NEURONS))
        self._S = _next_pow2(max(initial_synapses, PAGE_SYNAPSES))
        self._S_td = _next_pow2(PAGE_TD)

        # ── 神经元张量 (合并视图) ──
        self.state = torch.zeros(self._N, N_STATE_FIELDS, dtype=torch.float16, device=self.device)
        self.layer = torch.full((self._N,), -1, dtype=torch.int16, device=self.device)
        self.position = torch.full((self._N,), -1, dtype=torch.int16, device=self.device)
        self.channel = torch.full((self._N,), -1, dtype=torch.int16, device=self.device)
        self.created_at = torch.zeros(self._N, dtype=torch.int32, device=self.device)
        self.last_active = torch.zeros(self._N, dtype=torch.int32, device=self.device)
        self.alive = torch.zeros(self._N, dtype=torch.bool, device=self.device)
        self.in_ptrs = torch.full((self._N, K_FAN), -1, dtype=torch.int32, device=self.device)
        self.out_ptrs = torch.full((self._N, K_FAN), -1, dtype=torch.int32, device=self.device)
        self._in_counts = torch.zeros(self._N, dtype=torch.int32)
        self._out_counts = torch.zeros(self._N, dtype=torch.int32)
        self.t_weight = torch.zeros(self._N, dtype=torch.float16, device=self.device)
        self.t_connected = torch.zeros(self._N, dtype=torch.bool, device=self.device)

        # ── 突触张量 (合并视图) ──
        self.pre_id = torch.zeros(self._S, dtype=torch.int32, device=self.device)
        self.post_id = torch.zeros(self._S, dtype=torch.int32, device=self.device)
        self.weight = torch.zeros(self._S, dtype=torch.float16, device=self.device)
        self.trace = torch.zeros(self._S, dtype=torch.float16, device=self.device)
        self.syn_age = torch.zeros(self._S, dtype=torch.int32, device=self.device)
        self.syn_alive = torch.zeros(self._S, dtype=torch.bool, device=self.device)
        self.conn_type = torch.zeros(
            self._S, dtype=torch.int8, device=self.device
        )  # 0=feedforward, 1=feedback, 2=lateral

        # ── Topdown 张量 (合并视图) ──
        self.td_pre = torch.zeros(self._S_td, dtype=torch.int32, device=self.device)
        self.td_post = torch.zeros(self._S_td, dtype=torch.int32, device=self.device)
        self.td_weight = torch.zeros(self._S_td, dtype=torch.float16, device=self.device)
        self.td_alive = torch.zeros(self._S_td, dtype=torch.bool, device=self.device)

        # ── LM Head (合并视图, 列方向 = N) ──
        self.lm_weight = torch.zeros(256, self._N, dtype=torch.float16, device=self.device)

        # ── Freelists ──
        self._free_neurons: list[int] = list(range(self._N))
        self._free_synapses: list[int] = list(range(self._S))

        # ── 统计 ──
        self._occupied_neurons: int = 0
        self._occupied_synapses: int = 0

    # ═══════════════════════════════════════════════════════════════
    # 容量查询
    # ═══════════════════════════════════════════════════════════════

    @property
    def N(self) -> int:
        return self._N

    @property
    def S(self) -> int:
        return self._S

    @property
    def S_td(self) -> int:
        return self._S_td

    def total_allocated_bytes(self) -> int:
        """当前合并视图总字节数."""
        total = 0
        for t in [
            self.state,
            self.layer,
            self.position,
            self.channel,
            self.created_at,
            self.last_active,
            self.alive,
            self.in_ptrs,
            self.out_ptrs,
            self._in_counts,
            self._out_counts,
            self.t_weight,
            self.t_connected,
            self.pre_id,
            self.post_id,
            self.weight,
            self.trace,
            self.syn_age,
            self.syn_alive,
            self.td_pre,
            self.td_post,
            self.td_weight,
            self.td_alive,
            self.lm_weight,
        ]:
            total += t.numel() * t.element_size()
        return total

    def usage_ratio(self) -> float:
        """已分配内存 / 预算上限."""

        return self.total_allocated_bytes() / self.max_memory_bytes

    # ═══════════════════════════════════════════════════════════════
    # 分配 / 释放
    # ═══════════════════════════════════════════════════════════════

    def alloc_neuron_slot(self) -> int:
        """分配一个神经元槽位, 返回全局 nid. 必要时扩容."""
        if not self._free_neurons:
            self._expand_neurons()
        nid = self._free_neurons.pop()
        self._occupied_neurons += 1
        return nid

    def alloc_synapse_slot(self) -> int:
        """分配一个突触槽位, 返回全局 sid. 必要时扩容."""
        if not self._free_synapses:
            self._expand_synapses()
        sid = self._free_synapses.pop()
        self._occupied_synapses += 1
        return sid

    def free_neuron_slot(self, nid: int):
        """释放神经元槽位, 回到 freelist."""
        self._free_neurons.append(nid)
        self._occupied_neurons -= 1

    def free_synapse_slot(self, sid: int):
        """释放突触槽位, 回到 freelist."""
        self._free_synapses.append(sid)
        self._occupied_synapses -= 1

    # ═══════════════════════════════════════════════════════════════
    # 扩容 (翻倍)
    # ═══════════════════════════════════════════════════════════════

    def _expand_neurons(self):
        """翻倍神经元容量, 重建所有 N-维合并视图."""
        new_n = self._N * 2
        if self._check_budget_after_expand_neurons(new_n):
            raise MemoryBudgetError(
                f"神经元扩容至 {new_n} 将超出内存预算 "
                f"({self.total_allocated_bytes() / 1e9:.2f} GB > "
                f"{self.max_memory_bytes / 1e9:.2f} GB)"
            )

        # 为旧 freelist 条目扩展新 ID
        new_slots = list(range(self._N, new_n))
        self._free_neurons.extend(new_slots)

        # 重建 N-维张量
        self._resize_neurons_view(new_n)

    def _expand_synapses(self):
        """翻倍突触容量, 重建所有 S-维合并视图."""
        new_s = self._S * 2
        if self._check_budget_after_expand_synapses(new_s):
            raise MemoryBudgetError("突触扩容超出内存预算")

        new_slots = list(range(self._S, new_s))
        self._free_synapses.extend(new_slots)
        self._resize_synapses_view(new_s)

    def _check_budget_after_expand_neurons(self, new_n: int) -> bool:
        """检查扩容后是否超出内存预算 (预估)."""
        # 增量: 所有 N-维张量从 _N 扩到 new_n
        add_n = new_n - self._N
        add_bytes = add_n * (
            N_STATE_FIELDS * 2  # state fp16
            + 3 * 2  # layer, position, channel int16
            + 2 * 4  # created_at, last_active int32
            + 1  # alive bool
            + 2 * K_FAN * 4  # in_ptrs, out_ptrs int32
            + 2  # t_weight fp16
            + 1  # t_connected bool
            + 8  # _in_counts, _out_counts int32 (CPU, 但保守计入)
            + 256 * 2  # lm_weight 列 fp16
        )
        return (self.total_allocated_bytes() + add_bytes) > self.max_memory_bytes

    def _check_budget_after_expand_synapses(self, new_s: int) -> bool:
        add_s = new_s - self._S
        add_bytes = add_s * (
            4
            + 4
            + 2
            + 2
            + 4
            + 1
            + 1  # pre_id, post_id, weight, trace, syn_age, syn_alive, conn_type
        )
        return (self.total_allocated_bytes() + add_bytes) > self.max_memory_bytes

    # ═══════════════════════════════════════════════════════════════
    # 合并视图重建
    # ═══════════════════════════════════════════════════════════════

    def _resize_neurons_view(self, new_n: int):
        """将所有 N-维张量扩容到 new_n, 保留旧数据在 [:_N] 位置."""
        old_n = self._N

        def _expand(old: torch.Tensor, new_size: int, fill: int | float = 0) -> torch.Tensor:
            shape = list(old.shape)
            shape[0] = new_size
            if isinstance(fill, int) and fill == -1:
                new = torch.full(shape, fill, dtype=old.dtype, device=old.device)
            else:
                new = torch.zeros(shape, dtype=old.dtype, device=old.device)
            new[:old_n] = old
            return new

        self.state = _expand(self.state, new_n)
        self.layer = _expand(self.layer, new_n, fill=-1)
        self.position = _expand(self.position, new_n, fill=-1)
        self.channel = _expand(self.channel, new_n, fill=-1)
        self.created_at = _expand(self.created_at, new_n)
        self.last_active = _expand(self.last_active, new_n)
        self.alive = _expand(self.alive, new_n)
        self.in_ptrs = _expand(self.in_ptrs, new_n, fill=-1)
        self.out_ptrs = _expand(self.out_ptrs, new_n, fill=-1)
        self._in_counts = _expand(self._in_counts, new_n)
        self._out_counts = _expand(self._out_counts, new_n)
        self.t_weight = _expand(self.t_weight, new_n)
        self.t_connected = _expand(self.t_connected, new_n)

        # lm_weight: [256, old_n] → [256, new_n]
        new_lm = torch.zeros(256, new_n, dtype=torch.float16, device=self.device)
        new_lm[:, :old_n] = self.lm_weight
        self.lm_weight = new_lm

        self._N = new_n

    def _resize_synapses_view(self, new_s: int):
        old_s = self._S

        def _expand_1d(old: torch.Tensor, new_size: int) -> torch.Tensor:
            new = torch.zeros(new_size, dtype=old.dtype, device=old.device)
            new[:old_s] = old
            return new

        self.pre_id = _expand_1d(self.pre_id, new_s)
        self.post_id = _expand_1d(self.post_id, new_s)
        self.weight = _expand_1d(self.weight, new_s)
        self.trace = _expand_1d(self.trace, new_s)
        self.syn_age = _expand_1d(self.syn_age, new_s)
        self.syn_alive = _expand_1d(self.syn_alive, new_s)
        self.conn_type = _expand_1d(self.conn_type, new_s)
        self._S = new_s

    # ═══════════════════════════════════════════════════════════════
    # 迁移辅助: 确保容量 ≥ 指定值
    # ═══════════════════════════════════════════════════════════════

    def ensure_neuron_capacity(self, min_n: int):
        """确保神经元容量 ≥ min_n (迁移脚本使用)."""
        while self._N < min_n:
            self._expand_neurons()

    def ensure_synapse_capacity(self, min_s: int):
        while self._S < min_s:
            self._expand_synapses()

    # ═══════════════════════════════════════════════════════════════
    # 序列化辅助: 获取 alive 索引
    # ═══════════════════════════════════════════════════════════════

    def get_alive_neuron_idx(self) -> torch.Tensor:
        return torch.where(self.alive)[0]

    def get_alive_synapse_idx(self) -> torch.Tensor:
        return torch.where(self.syn_alive)[0]

    def get_alive_td_idx(self) -> torch.Tensor:
        return torch.where(self.td_alive)[0]
