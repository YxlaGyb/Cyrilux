"""StreamRunner — 持续运行循环入口。

将 SensoryFrontend (GPU)、EventBridge (CPU↔GPU)、
NeuronPool (CPU)、sparse_forward/homeostasis (CPU)
串联为一个连续运行的事件驱动循环。

没有 train/eval 模式切换: 系统启动即持续运行。
可塑性由多巴胺信号和 warmup 计数器动态调节。
"""

from __future__ import annotations


import torch

from model.pc.homeostasis import homeostasis_step
from model.pc.neuron_pool import NeuronPool
from pkg.device.event_bridge import EventBridge
from pkg.device.sensory_frontend import SensoryFrontend


class StreamRunner:
    """持续运行入口 — 没有 train/eval, 始终运行。

    Args:
        h_front: 感官前端特征维度 (默认 64)
        max_neurons: 神经元池上限 (默认 65536)
        sensory_threshold: 感官事件发送阈值 (默认 0.05)
        warmup_steps: warmup 步数 (默认 100)
        prune_interval: 修剪间隔 (默认 100)
        grow_interval: 生长间隔 (默认 200)
        hebbian_interval: Hebbian 更新间隔 (默认 1)
        homeostasis_interval: 稳态调节间隔 (默认 50)
    """

    def __init__(
        self,
        h_front: int = 64,
        max_neurons: int = 65536,
        sensory_threshold: float = 0.05,
        warmup_steps: int = 100,
        prune_interval: int = 100,
        grow_interval: int = 200,
        hebbian_interval: int = 1,
        homeostasis_interval: int = 50,
    ):
        # GPU 感官前端
        self.frontend = SensoryFrontend(h_front=h_front)
        self.frontend.eval()  # 永恒 eval 模式

        # CPU 动态图
        self.pool = NeuronPool(max_neurons=max_neurons)
        self.bridge = EventBridge(
            self.pool, h_front=h_front,
            sensory_threshold=sensory_threshold,
        )

        # 运行参数
        self.h_front = h_front
        self.warmup_steps = warmup_steps
        self.prune_interval = prune_interval
        self.grow_interval = grow_interval
        self.hebbian_interval = hebbian_interval
        self.homeostasis_interval = homeostasis_interval
        self._step: int = 0

        # 统计与监控
        self._loss_history: list[float] = []
        self._free_energy_history: list[float] = []
        self._n_sensory_events: int = 0
        self._n_network_events: int = 0

        # warmup
        self.bridge.set_warmup(warmup_steps)
        self._hidden_layer_created: bool = False
        self._initial_connections: int = 0

    def add_hidden_layer(self, n_neurons: int,
                         from_layer: int = 0,
                         to_layer: int = 7,
                         connection_density: float = 0.2):
        """添加隐藏层并连接到 sensory 层。

        Args:
            n_neurons: 隐藏神经元数量
            from_layer: 源层 (默认 0=sensory)
            to_layer: 目标层 (默认 7=第一隐藏层)
            connection_density: 连接密度 (默认 0.2)
        """
        for _ in range(n_neurons):
            self.pool.create_neuron(layer=to_layer)
        self._initial_connections = self.pool.connect_layer(
            from_layer=from_layer,
            to_layer=to_layer,
            connection_density=connection_density,
        )
        self._hidden_layer_created = True

    @torch.inference_mode()
    def step(self, byte_seq: torch.Tensor) -> dict:
        """执行一个完整步的事件驱动处理。

        Args:
            byte_seq: [1, 2, S] fp16 双通道字节编码

        Returns:
            {step, n_sensory_events, n_network_events, free_energy,
             n_neurons, n_synapses, firing_rate, threshold} 统计字典。
        """
        self._step += 1

        # 1. GPU sensory frontend
        h_list = self.frontend(byte_seq)

        # 2. Bridge: h_conv → SensoryEventQueue
        is_warmup = self._step <= self.warmup_steps
        top_k = 0 if is_warmup else 4
        self.bridge.ingest_hlist(h_list, top_k=top_k)

        # 3. CPU: drain sensory events
        n_sensory = self.bridge.process_sensory_events(max_events=500)
        self._n_sensory_events += n_sensory

        # 4. CPU: process internal events (稀疏传播)
        n_network = self.bridge.process_network_events(max_events=10)
        self._n_network_events += n_network

        # 5. 计算自由能 (所有神经元的 ε²)
        free_energy = 0.0
        for n in self.pool.neurons.values():
            free_energy += n.ε ** 2
        self._free_energy_history.append(free_energy)

        # 6. 稳态可塑性 (冷路径)
        if self._step % self.homeostasis_interval == 0:
            hs_stats = homeostasis_step(
                self.pool, self._step,
                target_rate=0.01,
                prune_interval=self.prune_interval,
                grow_interval=self.grow_interval,
            )
        else:
            hs_stats = {}

        return {
            "step": self._step,
            "n_sensory_events": n_sensory,
            "n_network_events": n_network,
            "free_energy": free_energy,
            "n_neurons": self.pool.get_total_neurons(),
            "n_synapses": self.pool.get_total_synapses(),
            "firing_rate": self.pool.get_activity_stats()["avg_firing_rate"],
            "threshold": self.pool.get_activity_stats()["avg_threshold"],
            "warmup": is_warmup,
            **hs_stats,
        }

    def run(self, byte_seq: torch.Tensor, n_steps: int = 1) -> list[dict]:
        """运行多步事件驱动处理。

        Args:
            byte_seq: [1, 2, S] fp16, S 需要 ≥ n_steps (滑动窗口处理)
                     或提供 [1, 2, n_steps + 12] 的连续流
            n_steps: 运行步数

        Returns:
            stats 字典列表, 每步一个。
        """
        stats_list = []
        S = byte_seq.shape[-1]

        for t in range(n_steps):
            # 滑动窗口: 取当前位置为中心的 S 长度窗口
            start = min(t, max(0, S - 128))
            window = byte_seq[..., start:start + 128]
            if window.shape[-1] < 13:
                # 太短无法 conv, padding
                pad_len = 13 - window.shape[-1]
                window = torch.nn.functional.pad(window, (0, pad_len))

            stats = self.step(window)
            stats_list.append(stats)

        return stats_list

    def ingest_stream(self, byte_stream: bytes,
                      positions_per_step: int = 128,
                      batch_size: int = 1) -> int:
        """将字节流转换为事件并推入系统。

        Args:
            byte_stream: 原始字节数据
            positions_per_step: 每步处理的位置数 (默认 128)
            batch_size: 每批步数

        Returns:
            处理的总步数。
        """
        # 编码: [0..255] → fp16 归一化到 [-1, 1]
        n_steps = max(1, len(byte_stream) - 13)  # 需要至少 13 字节 causal
        processed = 0

        for t in range(0, n_steps, positions_per_step):
            chunk = byte_stream[t:t + positions_per_step + 12]
            if len(chunk) < 13:
                break

            # 编码为 [1, 2, S] fp16
            byte_vals = torch.tensor(
                [b / 128.0 - 1.0 for b in chunk], dtype=torch.half
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, S]

            # 角色掩码 (ch1=全部 1)
            mask = torch.ones_like(byte_vals)
            seq = torch.cat([byte_vals, mask], dim=1)  # [1, 2, S]

            self.step(seq)
            processed += 1

        return processed

    def get_state(self) -> dict:
        """返回当前网络状态快照。"""
        return {
            "step": self._step,
            "pool_stats": self.pool.get_activity_stats(),
            "bridge_stats": self.bridge.get_stats(),
            "free_energy": self._free_energy_history[-1] if self._free_energy_history else 0.0,
            "total_sensory_events": self._n_sensory_events,
            "total_network_events": self._n_network_events,
            "warmup_remaining": max(0, self.warmup_steps - self._step),
        }

    @property
    def step_count(self) -> int:
        return self._step
