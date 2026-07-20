"""局部 Hebbian 学习引擎 — 大脑式自监督更新规则。

纯 PyTorch，零 autograd 依赖。所有更新直接通过 W.data.add_() 完成。
核心公式:

  F_total = Σ ½·π_ℓ·‖ε_ℓ‖² + λ(t)·‖decoder(z_L) - next_byte‖²
  π_ℓ = 1 + η_ACh·ACh·‖ε_ℓ‖_max + η_D·D·‖ε_ℓ‖_max
  ACh = sigmoid(-uncertainty + β₀)
  D = sigmoid(β·(F_prev - F_curr))
  η_eff = base_η · (0.5·D + 0.5·ACh) · (1 + γ·D)
  λ(t) = λ_min + (1-λ_min)·exp(-t/τ_λ)

权重更新: ΔW_ℓ = η_eff · ε_ℓ · activity_pre^T  (Hebbian)
"""
import math
from typing import Optional

import torch
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════
#  Phase 4: 稀疏外积 — 仅计算活跃突触前通道的 ΔW
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def _sparse_outer_product(post_error: torch.Tensor, pre_activity: torch.Tensor,
                          eta: float, k: int = 64) -> torch.Tensor:
    """稀疏 Hebbian 外积: 只计算 top-k 突触前通道, 其余为 0.

    dW[i,j] = η · Σ_{b,t} ε[b,t,i] · z[b,t,j]

    生物基础: STDP 只增强同时发放的 pre/post 对. 静默 pre → ΔW ≈ 0.
    复杂度: O(B · H_out · k · T) 而非 O(B · H_out · H_in · T).

    Args:
        post_error: [B, T, H_out] 突触后预测误差
        pre_activity: [B, T, H_in] 突触前活动 (z)
        eta: 有效学习率 (已含调制)
        k: 保留的活跃突触前通道数

    Returns:
        dW: [H_out, H_in] 稀疏权重更新 (仅 top-k 列非零)
    """
    H_in = pre_activity.size(-1)
    H_out = post_error.size(-1)
    k_use = min(k, H_in)
    device = post_error.device

    # 全局 top-k: 哪些突触前通道实际活跃
    pre_mag = pre_activity.float().abs().mean(dim=(0, 1))  # [H_in]
    _, top_k_idx = pre_mag.topk(k_use)
    top_k_idx, _ = top_k_idx.sort()

    dW = torch.zeros(H_out, H_in, device=device, dtype=torch.float32)
    err_T = post_error.float().transpose(1, 2)  # [B, H_out, T]
    pre_sel = pre_activity.float()[:, :, top_k_idx]  # [B, T, k]
    dW_sel = eta * torch.bmm(err_T, pre_sel).mean(dim=0)  # [H_out, k]
    dW[:, top_k_idx] = dW_sel
    return dW


# ═══════════════════════════════════════════════════════════════════
#  调制信号
# ═══════════════════════════════════════════════════════════════════

def compute_modulators(F_curr: float, F_prev: float,
                       uncertainty: float, cfg) -> tuple:
    """计算 D (RPE), ACh (不确定度), modulation.

    Args:
        F_curr: 当前步自由能
        F_prev: 上一步自由能
        uncertainty: 来自世界模型或 F_history 的不确定度 ∈ [0,1]
        cfg: TrainingConfig (需要 dopamine_beta, hebbian_ach_beta_0)

    Returns:
        D: RPE 多巴胺, sigmoid(β·(F_prev - F_curr)), ∈ (0,1)
        ACh: 乙酰胆碱连续调制, sigmoid(-uncertainty + β₀), ∈ (0,1)
        modulation: 组合调制 = 0.5·D + 0.5·ACh
    """
    # D: RPE — 自由能下降越多 D 越大
    δ = F_prev - F_curr
    D = float(torch.sigmoid(torch.tensor(getattr(cfg, 'dopamine_beta', 0.5) * δ)).item())

    # ACh: 连续不确定度 — 高 uncertainty → ACh 低 (不确信时降低学习)
    beta_0 = getattr(cfg, 'hebbian_ach_beta_0', 0.0)
    ACh = float(torch.sigmoid(torch.tensor(-uncertainty + beta_0)).item())

    modulation = 0.5 * D + 0.5 * ACh
    return D, ACh, modulation


def compute_precision_scales(ε_list: list, ACh: float, D: float, cfg) -> list:
    """计算 π_ℓ = 1 + η_ACh·ACh·‖ε_ℓ‖_max + η_D·D·‖ε_ℓ‖_max.

    Args:
        ε_list: list of [B, S, H] error tensors
        ACh: ACh 值
        D: 多巴胺值
        cfg: TrainingConfig (需 dopamine_eta)

    Returns:
        π_list: list of float, 长度 len(ε_list)
    """
    η = getattr(cfg, 'dopamine_eta', 1.0)
    π_list = []
    for ε in ε_list:
        max_err = ε.float().abs().reshape(-1, ε.size(-1)).norm(dim=-1).max().item() + 1e-8
        π = 1.0 + η * ACh * max_err + η * D * max_err
        π_list.append(π)
    return π_list


# ═══════════════════════════════════════════════════════════════════
#  误差计算
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_errors_by_layer(z_by_layer, model) -> tuple:
    """从收敛的 z 重新计算各层预测误差 ε_ℓ.

    Args:
        z_by_layer: list[tensor] — 收敛后的 z (含 z_0)
        model: CyrenePC 实例

    Returns:
        ε_list: list[tensor, L] — 各层 ε_ℓ, [B, S, H], fp32
        F_pred: float — Σ ½·‖ε_ℓ‖² (未加权)
    """
    L = model.num_sub_layers
    z_det = [z.detach().float() for z in z_by_layer]
    seq_len = z_det[0].size(1)
    device = z_det[0].device
    pos_emb = model.get_position_embeddings(seq_len, device)

    ε_list = []
    F_pred = 0.0

    for ℓ in range(1, L + 1):
        z_target = z_det[ℓ]
        z_prev = z_det[ℓ - 1]

        # 自下而上 (含残差)
        μ_bu = model.predict(ℓ, z_prev, pos_emb, fp32_out=True)
        μ_bu_res = μ_bu + z_prev

        # 时序
        if seq_len > 1:
            z_t_in = z_target[:, :-1, :]
            z_t_out = model.temporal_proj[ℓ - 1](z_t_in.float())
            μ_temp = torch.cat([torch.zeros_like(z_target[:, :1, :]), z_t_out], dim=1)
        else:
            μ_temp = torch.zeros_like(z_target)

        # 自上而下
        if ℓ < L and seq_len > 1:
            z_d_in = z_det[ℓ + 1][:, :-1, :]
            z_d_out = model.topdown_proj[ℓ - 1](z_d_in.float())
            μ_down = torch.cat([torch.zeros_like(z_target[:, :1, :]), z_d_out], dim=1)
        else:
            μ_down = torch.zeros_like(z_target)

        μ_total = μ_bu_res + μ_temp + μ_down
        ε = z_target - μ_total
        ε_list.append(ε)  # fp32
        F_pred += 0.5 * (ε ** 2).mean().item()

    return ε_list, F_pred


# ═══════════════════════════════════════════════════════════════════
#  逐层 Hebbian 更新
# ═══════════════════════════════════════════════════════════════════

def compute_hebbian_temporal(ε_list, z_init, modulation, base_lr,
                             gamma_rpe=0.3, oja_alpha=0.01,
                             oja_eta: float = 0.05,
                             W_curr_list=None,
                             sparse_k: int = 0) -> dict:
    """计算 temporal_proj 的 Hebbian 更新 + Oja 约束.

    ΔW_ℓ = η_eff · [ε_ℓ[:,1:]^T · z_ℓ[:,:-1] - α · diag(post²) · W_curr]

    Oja 项: -η_oja·α·Σ_t z_ℓ[t+1]² · W_curr[i,:] — 减性约束防止发散.

    Args:
        ε_list: list[tensor, L] — 各层误差 [B,S,H]
        z_init: list[tensor, L+1] — 推理开始时的 z (突触前活动)
        modulation: 组合调制值 (0.5·D+0.5·ACh)
        base_lr: 基础学习率
        gamma_rpe: RPE 增益系数
        oja_alpha: Oja 衰减系数 (0 = 禁用)
        oja_eta: Oja 独立学习率 (不绑定 Hebbian η)
        W_curr_list: list[Weight, L] — 当前 temporal 权重, 用于 Oja 项

    Returns:
        updates: dict[param_name → ΔW tensor]
    """
    L = len(ε_list)
    η = base_lr * modulation * (1.0 + gamma_rpe)
    updates = {}

    for ℓ in range(L):
        ε = ε_list[ℓ].float()                     # [B, S, H]
        z_pre = z_init[ℓ + 1].float()             # [B, S, H], 突触前 = z_ℓ
        if ε.size(1) > 1:
            # ε_ℓ[:,1:] 预测误差, z_ℓ[:,:-1] 前置活动
            ε_t1 = ε[:, 1:, :]                    # [B, S-1, H]
            z_pre_t = z_pre[:, :-1, :]             # [B, S-1, H]
            if sparse_k > 0:
                dW = _sparse_outer_product(ε_t1, z_pre_t, η, k=sparse_k)
            else:
                dW = η * torch.bmm(ε_t1.transpose(1, 2), z_pre_t)  # [B, H, H]
                dW = dW.mean(dim=0)                # [H, H]

            # Oja's rule: η_oja·α·post²·W_curr (减法: 防止权重单方向增长)
            if oja_alpha > 0 and W_curr_list is not None and W_curr_list[ℓ] is not None:
                post = z_pre[:, 1:, :]             # [B, S-1, H], 突触后活动 = z_ℓ[t+1]
                post_pow2_mean = (post.float() ** 2).mean(dim=(0, 1))  # [H]
                W_curr = W_curr_list[ℓ].float()    # [H, H]
                oja_term = oja_eta * oja_alpha * (post_pow2_mean.unsqueeze(-1) * W_curr)
                dW = dW - oja_term

            updates[f'temporal_proj.{ℓ}.weight'] = dW

    return updates


def compute_hebbian_topdown(ε_list, z_init, modulation, base_lr,
                            gamma_rpe=0.3, oja_alpha=0.01,
                            oja_eta: float = 0.05,
                            W_curr_list=None,
                            sparse_k: int = 0) -> dict:
    """计算 topdown_proj 的 Hebbian 更新 + Oja 约束.

    ΔW_ℓ = η_eff · [ε_ℓ[:,1:]^T · z_{ℓ+1}[:,:-1] - α · diag(post²) · W_curr]

    Args:
        ε_list: list[tensor, L] — 各层误差 [B,S,H]
        z_init: list[tensor, L+1] — 推理开始时的 z
        modulation: 组合调制值
        base_lr: 基础学习率
        gamma_rpe: RPE 增益系数
        oja_alpha: Oja 衰减系数 (0 = 禁用)
        oja_eta: Oja 独立学习率 (不绑定 Hebbian η)
        W_curr_list: list[Weight, L-1] — 当前 topdown 权重

    Returns:
        updates: dict[param_name → ΔW tensor]
    """
    L = len(ε_list)
    η = base_lr * modulation * (1.0 + gamma_rpe)
    updates = {}

    for ℓ in range(L):
        if ℓ >= L - 1:
            continue  # 最高层无 topdown
        ε = ε_list[ℓ].float()                     # [B, S, H]
        z_next = z_init[ℓ + 2].float()            # [B, S, H], z_{ℓ+1}
        if ε.size(1) > 1:
            ε_t1 = ε[:, 1:, :]                    # [B, S-1, H]
            z_next_t = z_next[:, :-1, :]           # [B, S-1, H]
            if sparse_k > 0:
                dW = _sparse_outer_product(ε_t1, z_next_t, η, k=sparse_k)
            else:
                dW = η * torch.bmm(ε_t1.transpose(1, 2), z_next_t)
                dW = dW.mean(dim=0)

            # Oja's rule: η·α·post²·W_curr (减法: 防止权重单方向增长)
            if oja_alpha > 0 and W_curr_list is not None and W_curr_list[ℓ] is not None:
                post = z_next[:, 1:, :]            # [B, S-1, H], 突触后 = z_{ℓ+1}[t+1]
                post_pow2_mean = (post.float() ** 2).mean(dim=(0, 1))  # [H]
                W_curr = W_curr_list[ℓ].float()    # [H, H]
                oja_term = oja_eta * oja_alpha * (post_pow2_mean.unsqueeze(-1) * W_curr)
                dW = dW - oja_term

            updates[f'topdown_proj.{ℓ}.weight'] = dW

    return updates


def compute_hebbian_conv(ε_ℓ, z_prev, conv_weight, dilation, modulation,
                         base_lr, gamma_rpe=0.3, oja_alpha=0.01,
                         oja_eta: float = 0.05,
                         sparse_k: int = 0) -> torch.Tensor:
    """计算 Conv1D 层的 Hebbian 更新 + Oja 约束.

    Conv1D: y = W ★ x, 其中 W ∈ [H, H, 3]
    ΔW = η_eff · [ε · unfold(x) - α · post² · W]

    Oja: 对每个 kernel position k, post_conv = W[:,:,k] @ z_shifted.

    Args:
        ε_ℓ: [B, S, H] 该层误差 (fp32)
        z_prev: [B, S, H] z_{ℓ-1} (fp32), 突触前活动
        conv_weight: [H, H, 3] conv 权重张量
        dilation: int, 当前层的膨胀率
        modulation: 组合调制值
        base_lr: 基础学习率
        gamma_rpe: RPE 增益系数
        oja_alpha: Oja 衰减系数 (0 = 禁用)
        oja_eta: Oja 独立学习率 (不绑定 Hebbian η)

    Returns:
        dW: [H, H, 3] 权重更新
    """
    η = base_lr * modulation * (1.0 + gamma_rpe)
    B, S, H_dim = z_prev.shape
    device = ε_ℓ.device

    # causal pad so that for output position t, we can read padded[t + k*d] for k=0,1,2
    pad = 2 * dilation
    z_padded = F.pad(z_prev.transpose(1, 2).float(), (pad, 0))  # [B, H_dim, S+2d]

    # 对每个 dilation 位置 k ∈ {0,1,2}, 收集输入活动
    # dW[:,:,k] = Σ_{b,t} ε[b,t,:]^T · padded[b,:,t+k*d]
    dW = torch.zeros(H_dim, H_dim, 3, device=device, dtype=torch.float32)
    ε_flat = ε_ℓ.float().permute(0, 2, 1).reshape(B, H_dim, S)  # [B, H, S]
    for k in range(3):
        offset = k * dilation
        z_shifted = z_padded[:, :, offset:offset + S]  # [B, H_dim, S]
        if sparse_k > 0:
            dW_k = _sparse_outer_product(
                ε_ℓ.float(),
                z_shifted.permute(0, 2, 1),
                η / S, k=sparse_k,
            )
            dW[:, :, k] = dW_k
        else:
            dW[:, :, k] = η * torch.bmm(ε_flat, z_shifted.transpose(1, 2)).mean(dim=0) / S

        # Oja's rule: η·α·post²·W_curr per kernel position (减法)
        if oja_alpha > 0:
            # post = conv output contribution at this kernel position
            # μ_k = W[:,:,k] @ z_shifted → [B, H, S]
            W_k = conv_weight[:, :, k].float()      # [H, H]
            post_k = torch.bmm(W_k.unsqueeze(0).expand(B, -1, -1),
                               z_shifted)            # [B, H, S]
            post_k = post_k.permute(0, 2, 1)         # [B, S, H]
            post_pow2_mean = (post_k.float() ** 2).mean(dim=(0, 1))  # [H]
            oja_term = oja_eta * oja_alpha * (post_pow2_mean.unsqueeze(-1) * conv_weight[:, :, k].float())
            dW[:, :, k] = dW[:, :, k] - oja_term
    return dW


def compute_hebbian_swiglu(ε_ℓ, z_conv, mlp, modulation, base_lr,
                           gamma_rpe=0.3, oja_alpha=0.01,
                           oja_eta: float = 0.05,
                           sparse_k: int = 0) -> dict:
    """计算 SwiGLU MLP 的 Hebbian 更新 + Oja 约束 (gate/up/down 三路).

    层内分解:
        ε_down = ε_ℓ
        ε_h = ε_down · W_down^T
        ε_g = ε_h ⊙ SiLU'(z_gate) · pre_up
        ε_u = ε_h ⊙ z_gate

        ΔW_down = η · ε_down^T · z_swiglu_in
        ΔW_gate = η · ε_g^T · z_swiglu_in
        ΔW_up   = η · ε_u^T · z_swiglu_in

        Oja: ΔW -= η_oja·α·post²·W_curr  (η_oja 独立于 Hebbian η)

    Args:
        ε_ℓ: [B, S, H] 该层误差 (fp32)
        z_conv: [B, S, H] conv 输出 (MLP 输入)
        mlp: FeedForward 模块 (含 gate_proj, up_proj, down_proj)
        modulation: 组合调制值
        base_lr: 基础学习率
        gamma_rpe: RPE 增益系数
        oja_alpha: Oja 衰减系数 (0 = 禁用)
        oja_eta: Oja 独立学习率 (不绑定 Hebbian η)

    Returns:
        updates: dict with 'gate_proj.weight', 'up_proj.weight', 'down_proj.weight'
            或 fused 模式下 'gate_up_proj.weight', 'down_proj.weight'
    """
    fused = hasattr(mlp, 'gate_up_proj')

    η = base_lr * modulation * (1.0 + gamma_rpe)
    device = ε_ℓ.device
    ε = ε_ℓ.float()                                # [B, S, H]
    x = z_conv.float()                              # [B, S, H]

    # 获取权重 (兼容 FusedFeedForward / 标准 MLP)
    W_down = mlp.down_proj.weight.float()           # [H, inter]
    if fused:
        W_gu = mlp.gate_up_proj.weight.float()      # [2*inter, H]
        inter = W_gu.shape[0] // 2
        W_gate = W_gu[:inter]                       # [inter, H]
        W_up   = W_gu[inter:]                       # [inter, H]
    else:
        W_gate = mlp.gate_proj.weight.float()       # [inter, H]
        W_up   = mlp.up_proj.weight.float()         # [inter, H]

    # 前向重建 (no grad)
    pre_gate = F.linear(x, W_gate)                  # [B, S, inter]
    pre_up   = F.linear(x, W_up)                    # [B, S, inter]
    gate_act = F.silu(pre_gate)
    hidden = gate_act * pre_up                       # SwiGLU [B, S, inter]

    # 误差回传通过 down_proj
    ε_down = ε                                      # [B, S, H]
    ε_hidden = torch.matmul(ε_down, W_down)          # [B, S, H] @ [H, inter] → [B, S, inter]

    # SwiGLU backward
    sig = torch.sigmoid(pre_gate)
    silu_grad = sig * (1.0 + pre_gate * sig * (1.0 - sig))
    ε_gate = (ε_hidden * pre_up) * silu_grad          # [B, S, inter]
    ε_up   = ε_hidden * gate_act                       # [B, S, inter]

    # Hebbian 更新
    B, S, H_dim = x.shape
    x_flat = x.reshape(-1, H_dim)                    # [B*S, H]
    gate_flat = ε_gate.reshape(-1, ε_gate.size(-1))
    up_flat   = ε_up.reshape(-1, ε_up.size(-1))
    down_flat = ε_down.reshape(-1, H_dim)
    hidden_flat = hidden.reshape(-1, hidden.size(-1))

    if sparse_k > 0:
        dW_gate = _sparse_outer_product(ε_gate, x, η, k=sparse_k)
        dW_up   = _sparse_outer_product(ε_up, x, η, k=sparse_k)
        dW_down = _sparse_outer_product(ε_down, hidden, η, k=sparse_k)
    else:
        dW_gate = η * (gate_flat.T @ x_flat) / (B * S)    # [inter, H], per-sample
        dW_up   = η * (up_flat.T @ x_flat) / (B * S)      # [inter, H], per-sample
        dW_down = η * (down_flat.T @ hidden_flat) / (B * S)  # [H, inter], per-sample

    # Oja's rule
    if oja_alpha > 0:
        for dW_component, post_act, W_curr in [
            (dW_gate, gate_act, W_gate),
            (dW_up,   pre_up,   W_up),
            (dW_down, hidden,   W_down),
        ]:
            post_pow2_mean = (post_act.float() ** 2).mean(dim=(0, 1))
            if W_curr.shape[0] == post_pow2_mean.shape[0]:
                oja_term = oja_eta * oja_alpha * (post_pow2_mean.unsqueeze(-1) * W_curr)
            else:
                oja_term = oja_eta * oja_alpha * (post_pow2_mean.unsqueeze(0) * W_curr)
            dW_component -= oja_term

    if fused:
        dW_gu = torch.cat([dW_gate, dW_up], dim=0)    # [2*inter, H]
        updates = {
            'gate_up_proj.weight': dW_gu.squeeze(0),
            'down_proj.weight':    dW_down.squeeze(0),
        }
    else:
        updates = {
            'gate_proj.weight': dW_gate.squeeze(0),
            'up_proj.weight':   dW_up.squeeze(0),
            'down_proj.weight': dW_down.squeeze(0),
        }
    return updates


def compute_hebbian_decoder(z_L, target_byte_embed, decoder_weight,
                            modulation, base_lr, gamma_rpe=0.3,
                            λ=0.01, oja_alpha=0.01,
                            oja_eta: float = 0.05) -> tuple:
    """计算 Decoder 的 Hebbian 更新 + Oja 约束.

    Args:
        z_L: [B, S, H] 顶层表示
        target_byte_embed: [B, S, 256] onehot 目标编码
        decoder_weight: [256, H] decoder 权重
        modulation: 组合调制值
        base_lr: 基础学习率
        gamma_rpe: RPE 增益系数
        λ: 当前衰减系数
        oja_alpha: Oja 衰减系数 (0 = 禁用)
        oja_eta: Oja 独立学习率 (不绑定 Hebbian η)

    Returns:
        dW_decoder: [256, H] 权重更新
        F_decoder: float — λ·‖decoder(z_L) - target‖²
    """
    η = base_lr * modulation * (1.0 + gamma_rpe)
    z = z_L.float()                                 # [B, S, H]
    tgt = target_byte_embed.float()                  # [B, S, 256] 或 [B, S-1, 256]

    # Decoder 预测: z[t] → next_byte[t]
    # 如果 target 长度 = S-1 (labels[:, 1:]), 用 z[:, :-1, :] 做预测
    if tgt.size(1) < z.size(1):
        z_in = z[:, :-1, :]                         # [B, S-1, H]
    else:
        z_in = z
    pred = F.linear(z_in, decoder_weight.float())    # [B, S', 256]
    # ═══════════════════════════════════════════════════════════════
    #  关键修复 v2: 使用 tgt - softmax(W·z) 作为残差
    #
    #  PC 理论: ΔW ∝ ε · z^T, 其中 ε = target - prediction
    #    → 正确字: tgt=1, pred≈0 → ε≈+0.996 → 推高正确字权重 ✓
    #    → 错误字: tgt=0, pred≈0 → ε≈±1/256 → 轻微波动
    #
    #  之前用 pred_softmax - tgt, 符号相反, 导致 CE 上升.
    # ═══════════════════════════════════════════════════════════════
    pred_softmax = F.softmax(pred, dim=-1)
    ε_decoder = tgt - pred_softmax                  # [B, S', 256] — PC 误差方向!

    # F_decoder: 用交叉熵 (而非 MSE) 衡量解码器性能
    F_decoder = λ * (-(tgt * (pred_softmax + 1e-10).log()).sum(dim=-1).mean()).item()

    # Hebbian: ΔW = η/(B·S) · Σ ε_decoder^T · z_in   (per-sample 平均)
    #   除以 B*S 保证与 temporal/topdown 一致的 per-sample 尺度,
    #   使 Oja (per-sample mean(post²)) 能有效约束.
    B, S_eff, H_dim = z_in.shape
    dW = (η / (B * S_eff)) * torch.bmm(
        ε_decoder.reshape(-1, 256).unsqueeze(0).transpose(1, 2),
        z_in.reshape(-1, H_dim).unsqueeze(0),
    ).squeeze(0)                                    # [256, H]

    # Oja's rule: η·α·post²·W_curr (减法: 防止权重发散)
    if oja_alpha > 0:
        post = pred_softmax                         # [B, S', 256], 突触后 = decoder 输出 (softmax)
        post_pow2_mean = (post.float() ** 2).mean(dim=(0, 1))  # [256]
        W_curr = decoder_weight.float()             # [256, H]
        oja_term = oja_eta * oja_alpha * (post_pow2_mean.unsqueeze(-1) * W_curr)
        dW = dW - oja_term

    return dW, F_decoder


def compute_hebbian_byte_proj(ε_1, byte_seq, byte_proj_weight, conv1_weight,
                              modulation, base_lr, gamma_rpe=0.3,
                              oja_alpha=0.01,
                              oja_eta: float = 0.05) -> dict:
    """计算 byte_proj (即 Conv1d(2, H, 13)) 的 Hebbian 更新 + Oja 约束.

    Oja: post = 卷积输出 z₀, η_oja·α·post²·W_curr 防止权重发散.

    Args:
        ε_1: [B, S, H] 第一层误差 (∂F/∂z_0)
        byte_seq: [B, 2, S] 字节输入
        byte_proj_weight: [H, 2, 13] byte_proj Conv1d weight
        conv1_weight: [H, H, 13] conv1 权重 (不使用)
        modulation: 组合调制值
        base_lr: 基础学习率
        oja_alpha: Oja 衰减系数 (0 = 禁用)
        oja_eta: Oja 独立学习率 (不绑定 Hebbian η)

    Returns:
        updates: dict with key 'model.byte_proj.weight'
    """
    η = base_lr * modulation * (1.0 + gamma_rpe)
    ε = ε_1.float()                                 # [B, S, H]
    x_byte = byte_seq.float()                       # [B, 2, S]

    B, S, H_dim = ε.shape
    pad = 12
    x_padded = F.pad(x_byte.float(), (pad, 0))       # [B, 2, S+12]
    x_unfold = x_padded.unfold(2, 13, 1)             # [B, 2, S, 13]
    x_unfold = x_unfold.permute(0, 2, 1, 3)          # [B, S, 2, 13]

    ε_t = ε.transpose(0, 1)                         # [S, B, H_dim]
    x_t = x_unfold.transpose(0, 1)                  # [S, B, 2, 13]
    dW = torch.einsum('sbi,sbjk->ijk', ε_t, x_t)    # [H_dim, 2, 13]
    dW = η * dW / (B * S)

    # Oja's rule: η·α·post²·W_curr (减法: 防止权重发散)
    if oja_alpha > 0:
        W_byte = byte_proj_weight.float()            # [H, 2, 13]
        z_0 = F.conv1d(x_byte, W_byte, padding=12)  # [B, H, S]
        z_0 = z_0.permute(0, 2, 1)                  # [B, S, H]
        post_pow2_mean = (z_0.float() ** 2).mean(dim=(0, 1))  # [H]
        oja_term = oja_eta * oja_alpha * (post_pow2_mean.unsqueeze(-1).unsqueeze(-1) * W_byte)
        dW = dW - oja_term

    return {'model.byte_proj.weight': dW}


# ═══════════════════════════════════════════════════════════════════
#  误差归一化 — 生物合理的发放率约束
# ═══════════════════════════════════════════════════════════════════

def rms_normalize(ε_list, rms_target=1.0, eps=1e-8):
    """逐层 RMS 归一化预测误差.

    生物类比: 神经元发放率有物理上限 (~0-100 Hz),
    误差信号不可能无限增长. 此函数对每层 ε 施加增益控制
    (gain control), 使其 RMS 处于可控范围, 保留方向与层内相对模式.

    不同于 LayerNorm: 不中心化, 不逐神经元归一化, 只缩放到目标 RMS.
    同一层内不同位置的相对误差幅度完全保留.

    Args:
        ε_list: list[tensor, L] — 各层误差 [B, S, H]
        rms_target: 目标 RMS (默认 1.0, 表示误差约等于一个标准差的发放)
        eps: 数值稳定常数

    Returns:
        ε_norm_list: 归一化后的 ε, 每层 RMS ≈ rms_target
    """
    ε_norm_list = []
    for ε in ε_list:
        ε_f32 = ε.float()
        rms = ε_f32.square().mean().sqrt()
        scale = rms_target / max(rms.item(), eps)
        if abs(scale - 1.0) > 1e-6:
            ε_norm_list.append(ε_f32 * scale)
        else:
            ε_norm_list.append(ε_f32)
    return ε_norm_list


# ═══════════════════════════════════════════════════════════════════
#  统一入口
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
#  Phase 2: 突触竞争 — per-weight-row WTA
# ═══════════════════════════════════════════════════════════════════

def _apply_synaptic_competition(dW: torch.Tensor, k: int = 8,
                                 use_abs: bool = False) -> torch.Tensor:
    """per-weight-row WTA: 每行仅保留 top-k 个突触更新, 其余置零.

    生物类比: 每个突触后神经元 ~10⁴ 突触, 仅 ~10² 同时增强.
    此处对权重更新 ΔW 施加行内竞争, 防止所有突触同方向漂移.

    Args:
        dW: [out_dim, in_dim] 或 [out_dim, in_dim, k_pos] (conv)
        k: 每行保留的胜者数 (≈ hidden_size × 0.01)
        use_abs: True=按绝对值竞争, False=按代数值 (保留符号方向)

    Returns:
        dW_competed: 与 dW 同形状, 非胜者位置为 0
    """
    if dW.ndim == 3:
        # Conv: [H, H, k] → [H, H*k], 行内竞争, 再 reshape 回来
        out, in_ch, k_pos = dW.shape
        dW_flat = dW.reshape(out, in_ch * k_pos)
        if use_abs:
            _, idx = dW_flat.abs().topk(k=min(k, dW_flat.size(-1)), dim=-1)
        else:
            _, idx = dW_flat.topk(k=min(k, dW_flat.size(-1)), dim=-1)
        mask = torch.zeros_like(dW_flat)
        mask.scatter_(dim=-1, index=idx, value=1.0)
        return (dW_flat * mask).reshape(out, in_ch, k_pos)
    elif dW.ndim == 2:
        out, inp = dW.shape
        k_use = min(k, inp)
        if use_abs:
            _, idx = dW.abs().topk(k=k_use, dim=-1)  # [out, k]
        else:
            _, idx = dW.topk(k=k_use, dim=-1)
        mask = torch.zeros_like(dW)
        mask.scatter_(dim=-1, index=idx, value=1.0)
        return dW * mask
    else:
        return dW  # 未知维度, 跳过


@torch.no_grad()
def compute_all_hebbian_updates(ε_list, z_init, byte_seq, model, cfg,
                                D=0.5, ACh=0.5, modulation=0.5, λ=0.01,
                                decoder=None, lm_head=None, target_byte_embed=None,
                                oja_alpha=0.05, oja_eta: float = 0.05,
                                bcm_state=None, verbose=True) -> dict:
    """统一入口: 计算所有 PC 参数的 Hebbian 更新 + Oja 约束 + BCM 调制 + 诊断日志.

    Args:
        ε_list: list[tensor, L] — 各层误差 (已由活性门控自然稀疏化)
        z_init: list[tensor, L+1] — 推理开始时的 z
        byte_seq: [B, 2, S] 字节输入
        model: CyrenePC 实例
        cfg: TrainingConfig
        D: 多巴胺值
        ACh: 乙酰胆碱值
        modulation: 组合调制 (0.5D+0.5ACh)
        λ: decoder 约束权重
        decoder: nn.Linear(H, 256) — decoder 模块
        lm_head: nn.Linear(H, 256) — 实际用于生成的 lm_head (与 decoder 结构一致)
        target_byte_embed: [B, S, 256] onehot 目标
        oja_alpha: Oja 衰减系数 (0=禁用)
        oja_eta: Oja 独立学习率 (不绑定 Hebbian η)
        bcm_state: BCMState 实例
        verbose: 是否打印诊断日志

    注意: top_k 后过滤已移除 — 自然稀疏由 Phase 1 活性门控提供.
    静默通道的 ε≈0 → ΔW≈0, 无需显式 top-k.
    """
    base_lr = getattr(cfg, 'hebbian_base_eta', 1e-4)
    gamma_rpe = getattr(cfg, 'dopamine_gamma', 0.3)
    oja_eta = getattr(cfg, 'oja_eta', oja_eta)          # cfg 可覆盖
    ε_rms_target = getattr(cfg, 'ε_rms_target', 1.0)    # 生物发放率约束

    # ε 归一化: 模拟神经元发放率上界, 防止 ΔW 数值爆炸
    ε_list = rms_normalize(ε_list, rms_target=ε_rms_target)

    # z 归一化: 防止 24 层残差连锁放大导致 ΔW 生物荒谬
    if len(z_init) > 1:
        z_init = [z_init[0].float()] + rms_normalize(z_init[1:], rms_target=1.0)

    L = len(ε_list)
    updates = {}

    # Phase 4: 稀疏外积 — 从 cfg 读取保留通道数
    sparse_k = getattr(cfg, 'sparse_outer_k', 0)
    # ε 门控跳过: 预测误差极小的层不产生 Hebbian 更新
    ε_gate_threshold = getattr(cfg, 'hebbian_eps_gate', 0.0)

    # 收集当前权重用于 Oja 规则
    temp_weights = [proj.weight for proj in model.temporal_proj]
    topdown_weights = [proj.weight for proj in model.topdown_proj]

    # 1) Temporal projections + Oja
    temp_updates = compute_hebbian_temporal(
        ε_list, z_init, modulation, base_lr, gamma_rpe,
        oja_alpha=oja_alpha, oja_eta=oja_eta,
        W_curr_list=temp_weights, sparse_k=sparse_k)
    updates.update(temp_updates)

    # 2) Topdown projections + Oja
    topdown_updates = compute_hebbian_topdown(
        ε_list, z_init, modulation, base_lr, gamma_rpe,
        oja_alpha=oja_alpha, oja_eta=oja_eta,
        W_curr_list=topdown_weights, sparse_k=sparse_k)
    updates.update(topdown_updates)

    # 3) Conv1D + SwiGLU per layer block (所有层, 活性门控已自然稀疏化)
    #    Phase 4b: ε 门控 — 跳过已收敛层的 Hebbian 更新
    for ℓ in range(L):
        block_idx = ℓ // 2
        is_conv_or_attn = ℓ % 2 == 0
        block = model.model.layers[block_idx]

        # ε 门控: ‖ε‖<阈值 → 该层已收敛, 跳过 Hebbian 更新
        if ε_gate_threshold > 0.0 and ε_list[ℓ].float().norm().item() < ε_gate_threshold:
            continue

        if is_conv_or_attn:
            dW_conv = compute_hebbian_conv(
                ε_list[ℓ], z_init[ℓ],
                block.local_conv.weight,
                block.dilation,
                modulation, base_lr, gamma_rpe,
                oja_alpha=oja_alpha, oja_eta=oja_eta,
                sparse_k=sparse_k,
            )
            updates[f'model.layers.{block_idx}.local_conv.weight'] = dW_conv
        else:
            mlp_updates = compute_hebbian_swiglu(
                ε_list[ℓ], z_init[ℓ], block.mlp,
                modulation, base_lr, gamma_rpe,
                oja_alpha=oja_alpha, oja_eta=oja_eta,
                sparse_k=sparse_k,
            )
            for k, v in mlp_updates.items():
                updates[f'model.layers.{block_idx}.mlp.{k}'] = v

    # 4) Decoder (外部约束) + Oja — 总是计算 (关键输出层)
    if decoder is not None and target_byte_embed is not None:
        dW_dec, F_dec = compute_hebbian_decoder(
            z_init[-1], target_byte_embed, decoder.weight,
            modulation, base_lr, gamma_rpe, λ,
            oja_alpha=oja_alpha, oja_eta=oja_eta,
        )
        updates['decoder.weight'] = dW_dec

    # 4b) lm_head — 实际生成使用的是 model.lm_head, 非 decoder
    #     必须与 decoder 同步训练, 否则 Hebbian 学习无法影响输出.
    #     使用与 decoder 完全相同的 Hebbian 更新 (CE 残差), 写入 model.lm_head.weight.
    if lm_head is not None and target_byte_embed is not None:
        dW_lm, _ = compute_hebbian_decoder(
            z_init[-1], target_byte_embed, lm_head.weight,
            modulation, base_lr, gamma_rpe, λ,
            oja_alpha=oja_alpha, oja_eta=oja_eta,
        )
        updates['model.lm_head.weight'] = dW_lm

    # 5) Byte projection 层 + Oja — 总是计算 (输入层)
    if hasattr(model.model, 'byte_proj'):
        byte_updates = compute_hebbian_byte_proj(
            ε_list[0], byte_seq,
            model.model.byte_proj.weight,
            None,
            modulation, base_lr, gamma_rpe,
            oja_alpha=oja_alpha, oja_eta=oja_eta,
        )
        for k, v in byte_updates.items():
            updates[f'model.{k}'] = v

    # 6) BCM 滑动阈值调制: LTP vs LTD 动态平衡
    if bcm_state is not None:
        z_for_bcm = z_init[1:]  # [z₁, ..., z_L], 共 L 层
        bcm_factors = bcm_state.compute_factors(z_for_bcm)
        for ℓ in range(L):
            factor = bcm_factors[ℓ]
            block_idx = ℓ // 2
            is_conv = (ℓ % 2 == 0)
            # temporal 和 topdown
            if f'temporal_proj.{ℓ}.weight' in updates:
                updates[f'temporal_proj.{ℓ}.weight'] *= factor
            if f'topdown_proj.{ℓ}.weight' in updates:
                updates[f'topdown_proj.{ℓ}.weight'] *= factor
            # conv 或 MLP
            if is_conv:
                key = f'model.layers.{block_idx}.local_conv.weight'
                if key in updates:
                    updates[key] *= factor
            else:
                for suffix in ['gate_proj.weight', 'up_proj.weight', 'down_proj.weight']:
                    key = f'model.layers.{block_idx}.mlp.{suffix}'
                    if key in updates:
                        updates[key] *= factor

    # 7) Phase 2: 突触竞争 — per-weight-row WTA
    #     当稀疏外积激活时跳过 (已全局 top-k, 无需 per-row 再竞争)
    competition_k = getattr(cfg, 'synaptic_competition_k', 0)
    if competition_k > 0 and sparse_k == 0:
        use_abs = getattr(cfg, 'synaptic_competition_use_abs', False)
        for name in list(updates.keys()):
            updates[name] = _apply_synaptic_competition(
                updates[name], k=competition_k, use_abs=use_abs)

    # 8) 诊断日志: 权重变化幅度 & Oja 衰减统计 (存入 dict, 供外部用 tqdm.write 输出)
    if verbose:
        total_growth = 0.0
        n_params = 0
        n_inf = 0
        for name, dW in updates.items():
            dW_finite = dW.float()
            if not torch.isfinite(dW_finite).all():
                n_inf += 1
                dW_finite = torch.where(torch.isfinite(dW_finite), dW_finite, torch.zeros_like(dW_finite))
            dW_norm = dW_finite.norm().item()
            total_growth += dW_norm
            n_params += 1
        avg_growth = total_growth / max(n_params, 1)
        updates['_diag_avg_growth'] = avg_growth
        updates['_diag_n_inf'] = n_inf
        updates['_diag_n_params'] = n_params
        updates['_diag_oja_alpha'] = float(oja_alpha)

    return updates


@torch.no_grad()
def apply_hebbian_updates(updates: dict, model, grad_clip: float = 1.0,
                          synaptic_normalize: bool = True,
                          target_norm: float = 0.0,
                          max_delta: float = 0.02,
                          weight_bound: float = 999.0):
    """将 Hebbian 更新应用到模型权重 — 3 层生物学约束保护.

    生物学类比:
      1. 逐突触增量限制 (max_delta): STDP 单次事件仅改变 ~1%
      2. 范数裁剪 (grad_clip): 整体稳定性
      3. 权重硬边界 (weight_bound): 突触传导率有物理上限
      4. 突触归一化 (synaptic_normalize): 稳态可塑性 (homeostatic scaling)

    Args:
        updates: dict[param_name → ΔW tensor]
        model: CyrenePC 实例
        grad_clip: 梯度裁剪阈值 (基于 ΔW 范数)
        synaptic_normalize: 是否应用突触归一化 (per-neuron L2)
        target_norm: 目标 L2 范数 (0=auto: sqrt(fan_in))
        max_delta: 单次更新中每个突触的最大绝对变化 (0=不限制)
        weight_bound: 权重值的硬边界 [-bound, bound]
    """
    sd = model.state_dict()
    for name, dW in updates.items():
        if name not in sd:
            continue
        # 第 1 层: 逐突触增量限制 (per-synapse delta cap)
        dW_f32 = dW.float()
        if max_delta > 0:
            dW_f32 = dW_f32.clamp(-max_delta, max_delta)

        # 第 2 层: 范数裁剪 (norm clip)
        dW_norm = dW_f32.norm().item()
        if dW_norm > grad_clip:
            dW_f32 = dW_f32 * (grad_clip / max(dW_norm, 1e-8))

        # 第 3 层: 权重硬边界 + fp16 安全
        new_w = (sd[name].float() + dW_f32).clamp(-65504, 65504)
        if weight_bound > 0:
            new_w = new_w.clamp(-weight_bound, weight_bound)

        # 突触归一化 (Synaptic Normalization): per-neuron L2 约束
        if synaptic_normalize:
            w_flat = new_w.flatten(1)               # [out, in*k]
            fan_in = w_flat.size(-1)
            target = target_norm if target_norm > 0 else math.sqrt(fan_in)
            norms = w_flat.norm(dim=-1, keepdim=True)  # [out, 1]
            w_normalized = w_flat / (norms + 1e-8) * target
            new_w = w_normalized.reshape_as(new_w)

        new_w = new_w.half()
        sd[name].data.copy_(new_w)


@torch.no_grad()
def compute_lambda(t: int, τ_λ: int = 5000, λ_min: float = 0.01) -> float:
    """计算 decoder 约束权重的退火 λ(t).

    λ(t) = λ_min + (1 - λ_min)·exp(-t / τ_λ)
    永不归零, λ_min 保证长期仍有约束.
    """
    return λ_min + (1.0 - λ_min) * math.exp(-t / τ_λ)


# ═══════════════════════════════════════════════════════════════════
#  BCM 滑动阈值 — Bienenstock-Cooper-Munro 可塑性平衡
# ═══════════════════════════════════════════════════════════════════

class BCMState:
    """BCM 滑动阈值状态 — 生物可塑性平衡.

    核心公式:
        θ_M(t+1) = (1-τ)·θ_M(t) + τ·z²
        BCM factor = (z² - θ_M) / (θ_M + ε)

    当 post-synaptic 活动 > θ_M: 因子 > 0 → LTP (长期增强)
    当 post-synaptic 活动 < θ_M: 因子 < 0 → LTD (长期抑制)

    z 经过 RMS 归一化后, z² 均值≈1, θ_M 自然收敛到 1,
    形成自平衡系统: 活跃层 → LTP, 静默层 → LTD.
    """

    def __init__(self, n_layers: int = 24, tau: float = 0.01,
                 theta_init: Optional[float] = None):
        """Args:
        n_layers: 子层数 (默认 24, 对应 12 个 LocalConvBlock)
        tau: 滑动阈值更新率 (0.01 ≈ 100 步时间常数; 0.005 用于慢速巩固)
        theta_init: 初始阈值 (默认 1.0, 与 RMS 目标对齐)
        """
        self.tau = tau
        self.n_layers = n_layers
        if theta_init is None:
            theta_init = 1.0
        self.register_buffer('theta', torch.full((n_layers,), theta_init))

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """注册类似 nn.Module 的 buffer (CPU 存储, 可 .to() 移动)."""
        self.__dict__[name] = tensor

    def to(self, device):
        """将阈值移动到目标设备."""
        self.theta = self.theta.to(device)
        return self

    @torch.no_grad()
    def compute_factors(self, z_list: list) -> list:
        """计算每层 BCM 因子, 同时更新滑动阈值.

        Args:
            z_list: list[tensor, L] — 各层 z 活动 (post-synaptic), [B, S, H]

        Returns:
            factors: list[float, L] — 每层 BCM 调制因子
        """
        factors = []
        for ℓ, z in enumerate(z_list):
            if ℓ >= self.n_layers:
                break
            z_sq = (z.float() ** 2).mean().item()
            # 更新滑动阈值: θ_M(t+1) = (1-τ)·θ_M(t) + τ·z²
            self.theta[ℓ] = (1.0 - self.tau) * self.theta[ℓ] + self.tau * z_sq
            # BCM 因子: (z² - θ_M) / (θ_M + ε)
            θ = self.theta[ℓ].item()
            factor = (z_sq - θ) / (θ + 1e-8)
            factors.append(factor)
        return factors
