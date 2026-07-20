"""
CUDA Graph 训练加速器 — 录制 GPU 计算图消除 Python→CUDA kernel launch 开销。

用法:
    from core.cuda_graphs import CUDAGraphTrainer

    trainer = CUDAGraphTrainer(loop, warmup_steps=10)
    trainer.capture()           # 录制
    out = trainer.replay(byte_seq, labels, precision_scales)
    F_pred_val = out['F_pred'].item()   # graph 外 .item()

架构:
    ┌─ CPU 预计算 ──┐   ┌── GPU Graph ──────────┐   ┌─ CPU 后处理 ──┐
    │ precision     │   │ forward_with_ce       │   │ dopamine      │
    │ π_ℓ (上一步)  │ → │ spatiotemporal_infer  │ → │ .update()     │
    │               │   │ loss_merge            │   │ optimizer     │
    │               │   │ backward              │   │ .step()       │
    └───────────────┘   └────────────────────────┘   │ scaler       │
                                                      │ .update()    │
                                                      └──────────────┘

约束:
    - batch_size, seq_len 固定 (TrainingConfig: 48, 128)
    - 无运行时动态控制流 (dropout=0, T_infer 固定)
    - 无 .item() / CPU sync 在 graph 内 → 通过 _graph_capture_mode 满足
"""

import torch
from typing import Optional, Any


class CUDAGraphTrainer:
    """CUDA Graph 包装器：录制 GPU 计算路径并 replay。

    单一 graph (完整 PC 路径, 永不跳过 PC)。
    optimizer.step() / scaler.update() / dopamine.update() 在 graph 外。
    """

    def __init__(self, loop, warmup_steps: int = 10):
        self.loop = loop
        self.warmup_steps = warmup_steps

        # 静态缓冲区 (固定 shape, bf16 匹配 AMP)
        B = loop.cfg.batch_size
        S = loop.cfg.max_seq_len
        self.static_byte = torch.zeros(B, 2, S, dtype=torch.bfloat16, device='cuda')
        self.static_labels = torch.zeros(B, S, dtype=torch.long, device='cuda')
        self.static_precision = torch.ones(loop.model.num_sub_layers, device='cuda')

        # 世界模型上下文静态缓冲区 (仅在启用世界模型时使用)
        C = getattr(loop.cfg, 'world_model_context_dim', 2)
        self.static_context = torch.zeros(B, C, device='cuda')

        # 输出占位 (tensor, graph 外 .item())
        self._output: dict[str, Any] = {}
        self.graph: Optional[torch.cuda.CUDAGraph] = None

    # ── 公开接口 ──

    def capture(self):
        """录制 CUDA Graph (完整 PC 路径)。"""
        loop = self.loop

        # 设 _graph_capture_mode (pc_layers 据此跳过 .item())
        loop._graph_capture_mode = True
        if hasattr(loop.model, '_graph_capture_mode'):
            loop.model._graph_capture_mode = True

        # warmup (>3 步确保 cudnn 算法和 AMP scaler 稳定)
        for _ in range(self.warmup_steps):
            loop._graph_train_step(
                self.static_byte, self.static_labels,
                self.static_precision, self.static_context,
            )

        # capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._output = loop._graph_train_step(
                self.static_byte, self.static_labels,
                self.static_precision, self.static_context,
            )

        # 恢复标志
        loop._graph_capture_mode = False
        if hasattr(loop.model, '_graph_capture_mode'):
            loop.model._graph_capture_mode = False

        self.loop._log('CUDA Graph captured')

    def replay(self, byte_seq: torch.Tensor, labels: torch.Tensor,
               precision_scales, world_model_context: torch.Tensor | None = None) -> dict:
        """用新数据 replay CUDA Graph。

        Args:
            byte_seq: [B, 2, S] bf16 输入
            labels: [B, S] 标签
            precision_scales: list[float, L] 或 None
            world_model_context: [B, C] 世界模型上下文或 None

        Returns:
            dict with tensor values: 'total_loss', 'F_pred', 'ce_loss',
                                     'world_loss', 'world_surprise'
        """
        self.static_byte.copy_(byte_seq)
        self.static_labels.copy_(labels)
        if precision_scales is not None:
            p = torch.tensor(precision_scales, device='cuda', dtype=torch.float32)
            self.static_precision.copy_(p)
        if world_model_context is not None:
            ctx = world_model_context.to(device='cuda', dtype=torch.float32)
            self.static_context.copy_(ctx)

        self.graph.replay()
        return self._output

    def is_captured(self) -> bool:
        return self.graph is not None
