"""
Predictive Coding 核心：节点管理、自由能、多巴胺。
Ponytail: 最小结构，够用就行。
"""
import torch


class PCNode:
    """单个 PC 变量节点。
    z:     活动值 (variable)
    μ:     自下而上预测 (prediction)
    ε:     预测误差 z - μ
    π:     精度权重 (precision), 由多巴胺调制
    """
    __slots__ = ('z', 'μ', 'ε', 'π')

    def __init__(self, z):
        self.z = z
        self.μ = torch.zeros_like(z)
        self.ε = torch.zeros_like(z)
        self.π = 1.0


class PCEnergy:
    """自由能追踪器。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.F_pred = 0.0  # 预测能量 Σ ½·π·‖ε‖²
        self.F_out = 0.0   # 输出能量 CE
        self.F_total = 0.0
        self.layer_errors = []  # 逐层 ‖ε‖² 用于可视化

    def add_prediction_energy(self, error_norm_sq, precision=1.0):
        """添加 ½·π·‖ε‖²"""
        self.F_pred += 0.5 * precision * error_norm_sq

    def set_output_energy(self, ce_loss):
        """设置输出能量"""
        self.F_out = ce_loss

    def compute_total(self):
        self.F_total = self.F_pred + self.F_out
        return self.F_total

    def log(self):
        return {
            'F_pred': self.F_pred,
            'F_out': self.F_out,
            'F_total': self.F_total,
        }


class DopamineSignal:
    """全局多巴胺信号 D = σ(-ΔF)。
    精度调制、学习率门控。
    """

    def __init__(self, η=1.0, threshold=0.01):
        self.η = η
        self.threshold = threshold
        self.F_prev = float('inf')
        self.F_history = []

    def update(self, F_current):
        """计算多巴胺 D"""
        ΔF = F_current - self.F_prev
        self.F_history.append(F_current)
        self.F_prev = F_current
        # D ∈ (0, 1), 自由能下降越多 D 越大
        D = torch.sigmoid(torch.tensor(-ΔF)).item()
        return D

    def modulate_precision(self, D, layer_error_norm):
        """π_ℓ = 1 + η·D·‖ε_ℓ‖_max"""
        return 1.0 + self.η * D * layer_error_norm

    def modulate_lr(self, D, base_lr, β=0.5):
        """α_eff = α·(1 + β·D)"""
        return base_lr * (1.0 + β * D)

    def gate_learning(self, D):
        """当 D < threshold 时冻结学习"""
        return D >= self.threshold
