"""EventBridge — CPU ↔ GPU 事件队列桥接。

将 GPU SensoryFrontend 的 h_conv 张量转换为 SensoryEvent 对象,
送入 CPU NeuronPool 消费。CPU 内部的 NetworkEvent 也通过此桥转发。

当前实现: 同步传输 (pinned memory + CUDA event 为 Phase E 预留)。
"""

from __future__ import annotations

import heapq
from collections import deque

import torch

from model.pc.neuron_pool import (
    NeuronPool,
    SensoryEvent,
    NetworkEvent,
)


class SensoryEventQueue:
    """GPU → CPU 感官事件队列。

    从 SensoryFrontend 输出的 h_conv 张量中提取活跃特征,
    转换为 SensoryEvent 对象供 CPU NeuronPool 消费。

    使用两个 deque 实现双缓冲 (GPU 写一个, CPU 读一个),
    避免锁竞争。
    """

    def __init__(self, max_size: int = 4096):
        self._incoming: deque[SensoryEvent] = deque()
        self._outgoing: deque[SensoryEvent] = deque()
        self.max_size = max_size
        self._dropped: int = 0

    def push_from_hlist(
        self,
        h_list: list[torch.Tensor],
        block_id_offset: int = 0,
        threshold: float = 0.0,
        top_k: int = 0,
    ) -> int:
        """将 SensoryFrontend 的 7 个 h_conv 张量展平为 SensoryEvent。

        Args:
            h_list: 来自 SensoryFrontend.forward() 的 7 个张量
                    [byte_proj_out, conv0_out, ..., conv5_out]
                    每个形状 [1, H_front, S]
            block_id_offset: 起始 block_id (默认 0)
            threshold: 事件阈值。只发送 |value| > threshold 的事件。
                       0.0 = 全部发送 (warmup)。
            top_k: 每 block 每 position 最多发送的事件数。0 = 不限。

        Returns:
            本批次创建的事件数。
        """
        n_events = 0
        for bid, h in enumerate(h_list):
            # h: [1, C, S] → [C, S]
            h_2d = h[0]  # type: ignore[reportCallIssue]
            C, S = h_2d.shape  # C=H_front, S=positions

            if top_k > 0:
                # 每 position 只保留 top_k 个通道
                vals, _ = h_2d.abs().topk(top_k, dim=0)
                min_val = vals[-1]  # [S] 每 position 的阈值
                mask = h_2d.abs() >= min_val.unsqueeze(0)
            else:
                mask = h_2d.abs() > threshold

            for pos in range(S):
                for ch in range(C):
                    if not mask[ch, pos]:
                        continue
                    val = h_2d[ch, pos].item()
                    if abs(val) > threshold:
                        event = SensoryEvent(
                            time=0,  # 下游分配真实时间
                            block_id=bid + block_id_offset,
                            pos=pos,
                            channel=ch,
                            value=val,
                            layer=bid,  # 每个 block 对应一个 PC 层
                        )
                        if len(self._incoming) >= self.max_size:
                            self._dropped += 1
                            break
                        self._incoming.append(event)
                        n_events += 1

        return n_events

    def pop(self, max_events: int = -1) -> list[SensoryEvent]:
        """消费队列中的 SensoryEvent。

        使用双缓冲切换, 消费端的读操作不影响生产端写入。

        Args:
            max_events: 最多消费事件数。-1 = 全部消费。

        Returns:
            SensoryEvent 列表。
        """
        # 双缓冲切换
        if not self._outgoing:
            self._outgoing, self._incoming = self._incoming, self._outgoing

        if max_events < 0:
            events = list(self._outgoing)
            self._outgoing.clear()
        else:
            events = []
            for _ in range(min(max_events, len(self._outgoing))):
                events.append(self._outgoing.popleft())

        return events

    def __len__(self) -> int:
        return len(self._incoming) + len(self._outgoing)

    @property
    def dropped(self) -> int:
        return self._dropped


class NetworkEventQueue:
    """CPU 内部网络事件队列 (按时间戳排序)。"""

    def __init__(self, max_size: int = 2048):
        self._heap: list[NetworkEvent] = []
        self._time_counter: int = 0
        self.max_size = max_size
        self._dropped: int = 0

    def push(self, event: NetworkEvent) -> bool:
        """推入一个 NetworkEvent。

        使用 heapq 保证按时间戳排序消费。

        Args:
            event: NetworkEvent 实例 (time 字段应已设置)

        Returns:
            True=成功, False=队列满被丢弃。
        """
        if len(self._heap) >= self.max_size:
            self._dropped += 1
            return False
        heapq.heappush(self._heap, event)
        return True

    def push_many(self, events: list[NetworkEvent]) -> int:
        """批量推入事件。

        Args:
            events: NetworkEvent 列表

        Returns:
            成功推入的数量。
        """
        count = 0
        for ev in events:
            ev.time = self._time_counter
            if self.push(ev):
                count += 1
            self._time_counter += 1
        return count

    def pop(self, max_events: int = 1) -> list[NetworkEvent]:
        """消费队列中最早的时间事件。

        Args:
            max_events: 最多消费数 (默认 1)

        Returns:
            按时间排序的 NetworkEvent 列表。
        """
        n = min(max_events, len(self._heap))
        return [heapq.heappop(self._heap) for _ in range(n)]

    def __len__(self) -> int:
        return len(self._heap)

    @property
    def dropped(self) -> int:
        return self._dropped


class EventBridge:
    """事件桥接 — GPU ↔ CPU 事件队列管理。

    职责:
      - SensoryFrontend → SensoryEventQueue (GPU → CPU)
      - NetworkEventQueue 管理 (CPU 内部)
    模型逻辑 (neurons predict/update/hebbian) 已移至 CyreneModel.
    """

    def __init__(self, pool: NeuronPool, sensory_threshold: float = 0.05, h_front: int = 64):
        self.pool = pool
        self.sensory_queue = SensoryEventQueue()
        self.network_queue = NetworkEventQueue()
        self.sensory_threshold = sensory_threshold
        self.h_front = h_front
        self._warmup_remaining: int = 0  # 0 = 稳态模式

        # 统计
        self._total_sensory_events: int = 0
        self._total_network_events: int = 0

    def set_warmup(self, warmup_steps: int):
        """设置 warmup 步数。warmup 期间所有事件都发送。"""
        self._warmup_remaining = warmup_steps

    def ingest_hlist(self, h_list: list[torch.Tensor], top_k: int = 0) -> int:
        """从 SensoryFrontend 输出推入感官事件。

        Args:
            h_list: 7 个 [1, H_front, S] 张量
            top_k: warmup 时 = 0 (全部), 稳态时 > 0 限制

        Returns:
            创建的事件数。
        """
        is_warmup = self._warmup_remaining > 0
        threshold = -1.0 if is_warmup else self.sensory_threshold
        t_k = 0 if is_warmup else top_k

        n = self.sensory_queue.push_from_hlist(
            h_list,
            threshold=threshold,
            top_k=t_k,
        )
        self._total_sensory_events += n

        if is_warmup:
            self._warmup_remaining -= 1
        return n

    def get_stats(self) -> dict:
        """返回桥接统计。"""
        return {
            "total_sensory_events": self._total_sensory_events,
            "total_network_events": self._total_network_events,
            "sensory_dropped": self.sensory_queue.dropped,
            "network_dropped": self.network_queue.dropped,
            "sensory_queue_depth": len(self.sensory_queue),
            "network_queue_depth": len(self.network_queue),
            "warmup_remaining": self._warmup_remaining,
        }
