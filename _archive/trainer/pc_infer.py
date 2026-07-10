"""
PC 推理引擎 — T 步自由能最小化循环。
Ponytail: 最小实现，T=1 调试, T=4 实验。
"""
import torch


def pc_infer_loop(pc_model, input_ids, labels, pos_emb, gamma=0.1, T=4):
    """T 步 PC 推理循环。

    流程:
      1. init_z: 前向传播初始化所有 z 节点
      2. T 步 infer_step: 交替更新 z 最小化自由能
      3. 返回收敛后的 z 和逐层误差

    返回:
      z_converged: 收敛后的 z 列表
      errors_hist: [(e_sq_ℓ, e_norm_ℓ), ...] 每步最后的逐层误差
      F_hist:      [F_total, ...] 每步自由能 (用于追踪收敛)
    """
    z = pc_model.init_z(input_ids)
    errors_hist = []
    F_hist = []

    for t in range(T):
        z, errors = pc_model.infer_step(z, pos_emb, gamma, labels)
        errors_hist.append(errors)

        # F = Σ ½·‖ε‖²
        F = sum(0.5 * e[0] for e in errors)  # e = (e_sq, e_norm)
        F_hist.append(F)

    return z, errors_hist, F_hist


def pc_infer_with_tracking(pc_model, input_ids, labels, pos_emb, gamma=0.1, T=4):
    """带 CE loss 追踪的 PC 推理（用于训练循环）。"""
    z = pc_model.init_z(input_ids)
    layer_errors = []  # 每步的逐层误差
    F_hist = []
    ce_hist = []

    for t in range(T):
        z, errors = pc_model.infer_step(z, pos_emb, gamma, labels)
        layer_errors.append(errors)

        F, ce = 0.0, 0.0
        for ℓ, (e_sq, _) in enumerate(errors):
            F += 0.5 * e_sq
        ce = pc_model.compute_loss(z, labels)
        F_hist.append(F)
        ce_hist.append(ce)

    return z, layer_errors, {'F': F_hist, 'ce': ce_hist}
