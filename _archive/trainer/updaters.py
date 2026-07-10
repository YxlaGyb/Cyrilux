"""
局部学习 + 多巴胺调制更新器。
Ponytail: 最小实现，调用 pc_model.compute_pc_loss() + optimizer.step()。
"""
import torch
from model.pc_core import DopamineSignal


class PCUpdater:
    """PC 权重更新器。支持梯度累积。"""

    def __init__(self, pc_model, optimizer, base_lr=5e-4, η=1.0, β=0.5,
                 dopamine_threshold=0.0):
        self.pc_model = pc_model
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.β = β
        self.dopamine = DopamineSignal(η=η, threshold=dopamine_threshold)
        self._accum_counter = 0

    def backward(self, z, pos_emb, labels, input_ids=None, div_factor=1.0):
        """累积梯度: 计算 PC 能量 → 多巴胺更新 → backward。"""
        energy = self.pc_model.compute_pc_loss(z, pos_emb, labels, input_ids=input_ids)
        f_val = energy.item()
        self._last_dopamine = self.dopamine.update(f_val)
        (energy / div_factor).backward()
        self._accum_counter += 1
        return f_val

    def optimizer_step(self):
        """多巴胺调制学习率 → 梯度裁剪 → optimizer.step() → zero_grad()。"""
        if self._accum_counter == 0:
            return None

        D = self._last_dopamine
        effective_lr = self.dopamine.modulate_lr(D, self.base_lr, self.β)

        for pg in self.optimizer.param_groups:
            pg['lr'] = effective_lr

        torch.nn.utils.clip_grad_norm_(self.pc_model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._accum_counter = 0

        return {
            'dopamine': D,
            'lr': effective_lr,
            'gate_open': self.dopamine.gate_learning(D),
        }

    def get_dopamine(self):
        return self.dopamine
