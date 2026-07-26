"""EventBridge — GPU → CPU 事件桥接 (张量化).

将 SensoryFrontend 的 GPU 张量输出转换为批量事件 tensor,
供 TensorNeuronPool 消费。不再使用逐元素 .item() 和 Python 对象。

事件以 tensor batch 传递: (pos, ch, val, layer, block_id) 各为 [E] tensor.
"""

from __future__ import annotations

import torch


class SensoryEventBatch:
    """批量感官事件, 全部为 GPU tensor (或 CPU tensor, 取决于 device).

    Fields:
        pos:      Tensor[E] int32  序列位置
        ch:       Tensor[E] int16  特征通道
        val:      Tensor[E] fp16   激活值
        layer:    Tensor[E] int8   逻辑层
        block_id: Tensor[E] int8   卷积块索引
    """

    __slots__ = ("pos", "ch", "val", "layer", "block_id")

    def __init__(
        self,
        pos: torch.Tensor,
        ch: torch.Tensor,
        val: torch.Tensor,
        layer: torch.Tensor,
        block_id: torch.Tensor,
    ):
        self.pos = pos
        self.ch = ch
        self.val = val
        self.layer = layer
        self.block_id = block_id

    def __len__(self) -> int:
        return self.pos.shape[0]

    @property
    def device(self):
        return self.pos.device


class EventBridge:
    """事件桥接 — GPU 端 top-k 过滤 + 事件 batch 管理.

    职责:
      - 从 SensoryFrontend h_list 提取事件 (GPU top-k 过滤)
      - 管理感官/网络事件队列 (tensor ring buffer)
      - 统计
    """

    def __init__(
        self,
        h_front: int = 64,
        sensory_threshold: float = 0.05,
        max_sensory_events: int = 4096,
        max_network_events: int = 2048,
        device: torch.device | str = "cpu",
    ):
        self.h_front = h_front
        self.sensory_threshold = sensory_threshold
        self.device = torch.device(device) if isinstance(device, str) else device
        self._warmup_remaining: int = 0

        # 感官事件 tensor ring buffer
        self._max_sensory = max_sensory_events
        self._sensory_pos = torch.zeros(max_sensory_events, dtype=torch.int32, device=self.device)
        self._sensory_ch = torch.zeros(max_sensory_events, dtype=torch.int16, device=self.device)
        self._sensory_val = torch.zeros(max_sensory_events, dtype=torch.float16, device=self.device)
        self._sensory_layer = torch.zeros(max_sensory_events, dtype=torch.int8, device=self.device)
        self._sensory_block = torch.zeros(max_sensory_events, dtype=torch.int8, device=self.device)
        self._sensory_write: int = 0
        self._sensory_count: int = 0

        # 网络事件 tensor ring buffer
        self._max_network = max_network_events
        self._net_nid = torch.zeros(max_network_events, dtype=torch.int32, device=self.device)
        self._net_eps = torch.zeros(max_network_events, dtype=torch.float16, device=self.device)
        self._net_val = torch.zeros(max_network_events, dtype=torch.float16, device=self.device)
        self._net_write: int = 0
        self._net_count: int = 0

        # 统计
        self._total_sensory: int = 0
        self._total_network: int = 0
        self._sensory_dropped: int = 0
        self._network_dropped: int = 0

    def set_warmup(self, warmup_steps: int):
        self._warmup_remaining = warmup_steps

    # ═══════════════════════════════════════════════════════════════
    # 感官事件提取 (GPU top-k)
    # ═══════════════════════════════════════════════════════════════

    def ingest_hlist(
        self,
        h_list: list[torch.Tensor],
        top_k: int = 0,
    ) -> SensoryEventBatch | None:
        """从 SensoryFrontend 输出提取事件 batch.

        在 GPU 上做 top-k 过滤, 不再逐元素 .item().

        Args:
            h_list: 7 个 [1, H_front, S] fp16 张量
            top_k: 每 position 保留的事件数 (0 = warmup: 全部)

        Returns:
            SensoryEventBatch, 或 None (无事件).
        """
        is_warmup = self._warmup_remaining > 0
        threshold = -1.0 if is_warmup else self.sensory_threshold

        all_pos = []
        all_ch = []
        all_val = []
        all_layer = []
        all_block = []

        for bid, h in enumerate(h_list):
            h_2d = h[0]  # [C, S]
            C, S = h_2d.shape

            if top_k > 0 and not is_warmup:
                # 每 position 取 top_k 个通道
                vals, idx = h_2d.abs().topk(min(top_k, C), dim=0)  # [top_k, S]
                min_vals = vals[-1]  # [S]
                mask = h_2d.abs() >= min_vals.unsqueeze(0)
            else:
                mask = h_2d.abs() > threshold

            # 找到满足条件的 (ch, pos) 对
            indices = torch.where(mask)  # tuple of (ch_idx, pos_idx)
            if indices[0].numel() == 0:
                continue

            ch_idx = indices[0].to(torch.int16)
            pos_idx = indices[1].to(torch.int32)
            values = h_2d[mask]

            all_pos.append(pos_idx)
            all_ch.append(ch_idx)
            all_val.append(values)
            all_layer.append(torch.full_like(ch_idx, bid, dtype=torch.int8))
            all_block.append(torch.full_like(ch_idx, bid, dtype=torch.int8))

        if not all_pos:
            return None

        batch = SensoryEventBatch(
            pos=torch.cat(all_pos),
            ch=torch.cat(all_ch),
            val=torch.cat(all_val),
            layer=torch.cat(all_layer),
            block_id=torch.cat(all_block),
        )

        # 截断到 max_sensory
        if len(batch) > self._max_sensory:
            self._sensory_dropped += len(batch) - self._max_sensory
            batch = SensoryEventBatch(
                pos=batch.pos[: self._max_sensory],
                ch=batch.ch[: self._max_sensory],
                val=batch.val[: self._max_sensory],
                layer=batch.layer[: self._max_sensory],
                block_id=batch.block_id[: self._max_sensory],
            )

        self._total_sensory += len(batch)
        if is_warmup:
            self._warmup_remaining -= 1

        return batch

    # ═══════════════════════════════════════════════════════════════
    # 网络事件 (CPU 侧)
    # ═══════════════════════════════════════════════════════════════

    def push_network_events(self, nids: torch.Tensor, eps: torch.Tensor):
        """批量推入网络事件 (活跃神经元)."""
        n = nids.shape[0]
        if n == 0:
            return

        if n > self._max_network:
            self._network_dropped += n - self._max_network
            n = self._max_network
            nids = nids[:n]
            eps = eps[:n]

        end = min(self._net_write + n, self._max_network)
        space = end - self._net_write
        self._net_nid[self._net_write : end] = nids[:space].to(torch.int32)
        self._net_eps[self._net_write : end] = eps[:space].to(torch.float16)
        self._net_count = min(self._net_count + space, self._max_network)
        self._net_write = end % self._max_network
        self._total_network += space

    def pop_network_events(self, max_events: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
        """消费网络事件.

        Returns:
            (nids, eps) tensors, 各为 [M] (M <= max_events).
        """
        available = min(max_events, self._net_count)
        if available == 0:
            return (
                torch.zeros(0, dtype=torch.int32, device=self.device),
                torch.zeros(0, dtype=torch.float16, device=self.device),
            )

        read_start = (self._net_write - self._net_count) % self._max_network
        read_end = read_start + available

        if read_end <= self._max_network:
            nids = self._net_nid[read_start:read_end].clone()
            eps = self._net_eps[read_start:read_end].clone()
        else:
            # 环形回绕
            first = self._max_network - read_start
            second = available - first
            nids = torch.cat(
                [
                    self._net_nid[read_start:],
                    self._net_nid[:second],
                ]
            )
            eps = torch.cat(
                [
                    self._net_eps[read_start:],
                    self._net_eps[:second],
                ]
            )

        self._net_count -= available
        return nids, eps

    # ═══════════════════════════════════════════════════════════════
    # 统计
    # ═══════════════════════════════════════════════════════════════

    def get_stats(self) -> dict:
        return {
            "total_sensory_events": self._total_sensory,
            "total_network_events": self._total_network,
            "sensory_dropped": self._sensory_dropped,
            "network_dropped": self._network_dropped,
            "warmup_remaining": self._warmup_remaining,
        }
