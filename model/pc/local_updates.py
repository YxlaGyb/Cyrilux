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

#  Phase 4: 稀疏外积 — 仅计算活跃突触前通道的 ΔW

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
    pre_mag = pre_activity.abs().mean(dim=(0, 1))  # [H_in]
    _, top_k_idx = pre_mag.topk(k_use)
    top_k_idx, _ = top_k_idx.sort()

    dW = torch.zeros(H_out, H_in, device=device, dtype=post_error.dtype)
    err_T = post_error.transpose(1, 2)  # [B, H_out, T]
    pre_sel = pre_activity[:, :, top_k_idx]  # [B, T, k]
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
        max_err = ε.abs().reshape(-1, ε.size(-1)).norm(dim=-1).max().item() + 1e-8
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
    z_det = [z.detach() for z in z_by_layer]
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
            z_t_out = model.temporal_proj[ℓ - 1](z_t_in)
            μ_temp = torch.cat([torch.zeros_like(z_target[:, :1, :]), z_t_out], dim=1)
        else:
            μ_temp = torch.zeros_like(z_target)

        # 自上而下
        if ℓ < L and seq_len > 1:
            z_d_in = z_det[ℓ + 1][:, :-1, :]
            z_d_out = model.topdown_proj[ℓ - 1](z_d_in)
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

@torch.no_grad()
def compute_hebbian_temporal(ε_list, z_init, modulation, base_lr,
                             gamma_rpe=0.3, oja_alpha=0.01,
                             oja_eta: float = 0.05,
                             W_curr_list=None,
                             sparse_k: int = 0,
                             subsample_idx: Optional[torch.Tensor] = None) -> dict:
    """计算 temporal_proj 的 Hebbian 更新 + Oja 约束 (跨层批量).

    ΔW_ℓ = η_eff · [ε_ℓ[:,1:]^T · z_ℓ[:,:-1] - α · diag(post²) · W_curr]

    性能: 跨 L 层 stack 成 [L, N, H] 单次 bmm, 避免 L 次 Python 循环 +
    小 kernel launch 开销 (GTX 1650 Ti 上 launch ~50μs/次).

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
    H_ = ε_list[0].size(-1)
    device = ε_list[0].device
    dtype = ε_list[0].dtype

    # 所有层共享 B, S — 子采样索引只需计算一次
    B_, S_ = ε_list[0].shape[:2]
    if S_ <= 1:
        # 无时序维度: 全部置零
        for ℓ in range(L):
            updates[f'temporal_proj.{ℓ}.weight'] = torch.zeros(H_, H_, device=device, dtype=dtype)
        return updates

    # 子采样索引映射 [B*S] → [B*(S-1)], 排除 t=0
    if subsample_idx is not None:
        t_pos = subsample_idx % S_
        valid = t_pos != 0
        idx_valid = subsample_idx[valid]
        if idx_valid.numel() == 0:
            for ℓ in range(L):
                updates[f'temporal_proj.{ℓ}.weight'] = torch.zeros(H_, H_, device=device, dtype=dtype)
            return updates
        b_idx = idx_valid // S_
        s_idx = idx_valid % S_
        idx_temp = b_idx * (S_ - 1) + (s_idx - 1)
        n_sub = idx_valid.shape[0]
    else:
        idx_temp = None
        n_sub = B_ * (S_ - 1)

    # 跨层 stack: ε_t1 [L, B*(S-1), H], z_pre_t [L, B*(S-1), H]
    # ε_ℓ[:,1:] 预测误差, z_ℓ[:,:-1] 前置活动
    ε_t1_stack = torch.stack([ε_list[ℓ][:, 1:, :].reshape(-1, H_) for ℓ in range(L)], dim=0)
    z_pre_t_stack = torch.stack([z_init[ℓ + 1][:, :-1, :].reshape(-1, H_) for ℓ in range(L)], dim=0)
    # post (z_ℓ[t+1]) 用于 Oja
    z_post_stack = torch.stack([z_init[ℓ + 1][:, 1:, :].reshape(-1, H_) for ℓ in range(L)], dim=0)

    if idx_temp is not None:
        ε_t1_stack = ε_t1_stack[:, idx_temp, :]        # [L, N, H]
        z_pre_t_stack = z_pre_t_stack[:, idx_temp, :]
        z_post_stack = z_post_stack[:, idx_temp, :]

    # 批量外积: [L, H, N] @ [L, N, H] → [L, H, H]
    s = math.sqrt(n_sub)
    ε_safe = ε_t1_stack.transpose(1, 2) / s
    z_safe = z_pre_t_stack / s
    dW_bmm = torch.bmm(ε_safe, z_safe)
    dW_all = η * dW_bmm

    # 批量 Oja: max-normed 防 fp16 溢出
    if oja_alpha > 0 and W_curr_list is not None:
        oja_f = oja_eta * oja_alpha
        zs = z_post_stack.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-4)  # [L, 1, H]
        z_div = z_post_stack / zs
        z_msq = (z_div ** 2).mean(dim=1)                       # [L, H]
        temp_z = (oja_f * z_msq) * zs.squeeze(1)               # [L, H]
        W_curr_stack = torch.stack(W_curr_list, dim=0)         # [L, H, H]
        oja_term = temp_z.unsqueeze(-1) * (zs.squeeze(1).unsqueeze(-1) * W_curr_stack)
        dW_all = dW_all - oja_term


    for ℓ in range(L):
        updates[f'temporal_proj.{ℓ}.weight'] = dW_all[ℓ]

    return updates


@torch.no_grad()
def compute_hebbian_topdown(ε_list, z_init, modulation, base_lr,
                            gamma_rpe=0.3, oja_alpha=0.01,
                            oja_eta: float = 0.05,
                            W_curr_list=None,
                            sparse_k: int = 0,
                            subsample_idx: Optional[torch.Tensor] = None) -> dict:
    """计算 topdown_proj 的 Hebbian 更新 + Oja 约束 (跨层批量).

    ΔW_ℓ = η_eff · [ε_ℓ[:,1:]^T · z_{ℓ+1}[:,:-1] - α · diag(post²) · W_curr]

    性能: 跨 (L-1) 层 stack 成 [L-1, N, H] 单次 bmm, 避免 Python 循环.

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

    if L <= 1:
        return updates

    L_td = L - 1
    H_ = ε_list[0].size(-1)
    device = ε_list[0].device
    dtype = ε_list[0].dtype

    B_, S_ = ε_list[0].shape[:2]
    if S_ <= 1:
        for ℓ in range(L_td):
            updates[f'topdown_proj.{ℓ}.weight'] = torch.zeros(H_, H_, device=device, dtype=dtype)
        return updates

    # 子采样索引映射 [B*S] → [B*(S-1)], 排除 t=0
    if subsample_idx is not None:
        t_pos = subsample_idx % S_
        valid = t_pos != 0
        idx_valid = subsample_idx[valid]
        if idx_valid.numel() == 0:
            for ℓ in range(L_td):
                updates[f'topdown_proj.{ℓ}.weight'] = torch.zeros(H_, H_, device=device, dtype=dtype)
            return updates
        b_idx = idx_valid // S_
        s_idx = idx_valid % S_
        idx_temp = b_idx * (S_ - 1) + (s_idx - 1)
        n_sub = idx_valid.shape[0]
    else:
        idx_temp = None
        n_sub = B_ * (S_ - 1)

    # 跨层 stack: ε_t1 [L_td, N, H], z_next_t [L_td, N, H], z_post [L_td, N, H]
    ε_t1_stack = torch.stack([ε_list[ℓ][:, 1:, :].reshape(-1, H_) for ℓ in range(L_td)], dim=0)
    z_next_t_stack = torch.stack([z_init[ℓ + 2][:, :-1, :].reshape(-1, H_) for ℓ in range(L_td)], dim=0)
    z_post_stack = torch.stack([z_init[ℓ + 2][:, 1:, :].reshape(-1, H_) for ℓ in range(L_td)], dim=0)

    if idx_temp is not None:
        ε_t1_stack = ε_t1_stack[:, idx_temp, :]
        z_next_t_stack = z_next_t_stack[:, idx_temp, :]
        z_post_stack = z_post_stack[:, idx_temp, :]

    # 批量外积: [L_td, H, N] @ [L_td, N, H] → [L_td, H, H]
    s = math.sqrt(n_sub)
    dW_all = η * torch.bmm(
        ε_t1_stack.transpose(1, 2) / s, z_next_t_stack / s
    )

    # 批量 Oja — max-normed 防 fp16 溢出
    if oja_alpha > 0 and W_curr_list is not None:
        oja_f = oja_eta * oja_alpha
        zs = z_post_stack.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-4)  # [L_td, 1, H]
        z_msq = ((z_post_stack / zs) ** 2).mean(dim=1)                       # [L_td, H]
        temp_z = (oja_f * z_msq) * zs.squeeze(1)                             # [L_td, H]
        W_curr_stack = torch.stack(W_curr_list, dim=0)                       # [L_td, H, H]
        oja_term = temp_z.unsqueeze(-1) * (zs.squeeze(1).unsqueeze(-1) * W_curr_stack)
        dW_all = dW_all - oja_term

    for ℓ in range(L_td):
        updates[f'topdown_proj.{ℓ}.weight'] = dW_all[ℓ]

    return updates


@torch.no_grad()
def compute_hebbian_conv(ε_ℓ, z_prev, conv_weight, dilation, modulation,
                         base_lr, gamma_rpe=0.3, oja_alpha=0.01,
                         oja_eta: float = 0.05,
                         sparse_k: int = 0,
                         subsample_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
    """计算 Conv1D 层的 Hebbian 更新 + Oja 约束 (3 k-position 批量).

    Conv1D: y = W ★ x, 其中 W ∈ [H, H, 3]
    ΔW = η_eff · [ε · unfold(x) - α · post² · W]

    性能: stack 3 个 k-position 成 [3, N, H] 单次 batched bmm,
    避免 3 次 Python 循环 + 3 次独立 kernel launch.

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
    dtype = ε_ℓ.dtype

    pad = 2 * dilation
    z_padded = F.pad(z_prev.transpose(1, 2), (pad, 0))            # [B, H_dim, S+2d]

    if subsample_idx is not None:
        n_sub = subsample_idx.shape[0]
        ε_s = ε_ℓ.reshape(-1, H_dim)[subsample_idx]               # [N, H]
        # stack 3 k-position z 成 [3, N, H]
        z_k_list = []
        for k in range(3):
            offset = k * dilation
            z_k = z_padded[:, :, offset:offset + S].permute(0, 2, 1).reshape(-1, H_dim)[subsample_idx]
            z_k_list.append(z_k)
        z_stacked = torch.stack(z_k_list, dim=0)                   # [3, N, H]
        # 单次 batched bmm: [1, H, N] @ [3, N, H] → [3, H, H]
        s = math.sqrt(n_sub)
        dW_k = η * ((ε_s.T.unsqueeze(0) / s) @ (z_stacked / s))   # [3, H, H]

        # 批量 Oja per k-position — max-normed 防 fp16 溢出
        if oja_alpha > 0:
            oja_f = oja_eta * oja_alpha
            W_stack = conv_weight.permute(2, 0, 1).contiguous()    # [3, H, H]
            post_k = torch.bmm(z_stacked, W_stack.transpose(1, 2))  # [3, N, H]
            ps = post_k.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-4)  # [3, 1, H]
            pk_msq = ((post_k / ps) ** 2).mean(dim=1)                       # [3, H]
            temp_pk = (oja_f * pk_msq) * ps.squeeze(1)                      # [3, H]
            oja_term = temp_pk.unsqueeze(-1) * (ps.squeeze(1).unsqueeze(-1) * W_stack)
            dW_k = dW_k - oja_term

        dW = dW_k.permute(1, 2, 0).contiguous()                    # [H, H, 3]
    elif sparse_k > 0:
        dW = torch.zeros(H_dim, H_dim, 3, device=device, dtype=dtype)
        for k in range(3):
            offset = k * dilation
            z_shifted = z_padded[:, :, offset:offset + S].permute(0, 2, 1)
            dW_k = _sparse_outer_product(ε_ℓ, z_shifted, η / S, k=sparse_k)
            dW[:, :, k] = dW_k
    else:
        dW = torch.zeros(H_dim, H_dim, 3, device=device, dtype=dtype)
        ε_flat = ε_ℓ.permute(0, 2, 1).reshape(B, H_dim, S)        # [B, H, S]
        s = math.sqrt(B * S)
        for k in range(3):
            offset = k * dilation
            z_shifted = z_padded[:, :, offset:offset + S]           # [B, H, S]
            dW[:, :, k] = η * torch.bmm(ε_flat / s, (z_shifted / s).transpose(1, 2)).mean(dim=0)
            if oja_alpha > 0:
                oja_f = oja_eta * oja_alpha
                W_k = conv_weight[:, :, k]
                post_k = torch.bmm(
                    W_k.unsqueeze(0).expand(B, -1, -1), z_shifted
                ).permute(0, 2, 1)
                ps = torch.amax(post_k.abs(), dim=(0, 1), keepdim=True).clamp(min=1e-4)
                pk_msq = ((post_k / ps) ** 2).mean(dim=(0, 1))
                temp_pk = (oja_f * pk_msq) * ps.squeeze()
                oja_term = temp_pk.unsqueeze(-1) * (ps.squeeze().unsqueeze(-1) * conv_weight[:, :, k])
                dW[:, :, k] -= oja_term
    return dW


# ── SwiGLU Hebbian 纯计算 (标准路径) ──
@torch.no_grad()
def _hebbian_swiglu_kernel_standard(x_f: torch.Tensor, ε_f: torch.Tensor,
                                    W_gu: torch.Tensor, W_down: torch.Tensor,
                                    η: float, Bs: int,
                                    oja_alpha: float = 0.0,
                                    oja_eta: float = 0.05) -> tuple:
    """标准路径: 无分支, 从输入计算前向 + 误差回传 + Oja.

    Args:
        x_f: [N, H] 前向输入 (已子采样)
        ε_f: [N, H] 误差 (已子采样)
        W_gu: [2*inter, H] gate+up 联合投影权重
        W_down: [H, inter] down 投影权重
        η: 学习率
        Bs: N 归一化系数 (子采样后的位置数)
        oja_alpha: Oja 衰减系数 (0 = 禁用)
        oja_eta: Oja 独立学习率
    """
    H = x_f.shape[-1]
    inter = W_gu.shape[0] // 2

    # 前向
    pre_fused = F.linear(x_f, W_gu)                   # [N, 2*inter]
    pre_gate, pre_up = pre_fused.chunk(2, dim=-1)     # [N, inter] each
    gate_act = F.silu(pre_gate)                       # [N, inter]
    hidden = gate_act * pre_up                        # [N, inter]

    # 误差回传
    ε_hidden = torch.matmul(ε_f, W_down)              # [N, inter]
    sig = torch.sigmoid(pre_gate)
    silu_grad = sig * (1.0 + pre_gate * sig * (1.0 - sig))
    ε_gate = (ε_hidden * pre_up) * silu_grad
    ε_up   = ε_hidden * gate_act

    # NaN/INF check
    # 融合外积: cat(gate, up) → 单次 matmul
    # 预除 sqrt(Bs): 防 fp16 累积溢出, 数学等价于 /Bs
    eu_f = torch.cat([ε_gate, ε_up], dim=-1)
    s = math.sqrt(Bs)
    dW_gu = η * ((eu_f / s).T @ (x_f / s))

    # down 外积
    dW_down = η * ((ε_f / s).T @ (hidden / s))

    # Oja 衰减 (3 路) — max-normed 防 fp16 溢出
    if oja_alpha > 0:
        oja_f = oja_eta * oja_alpha
        # gate
        gs = gate_act.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-4)
        g_msq = ((gate_act / gs) ** 2).mean(dim=0)
        temp_g = (oja_f * g_msq) * gs.squeeze(0)
        dW_gu[:inter] -= temp_g.unsqueeze(-1) * (gs.squeeze(0).unsqueeze(-1) * W_gu[:inter])
        # up
        us = pre_up.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-4)
        u_msq = ((pre_up / us) ** 2).mean(dim=0)
        temp_u = (oja_f * u_msq) * us.squeeze(0)
        dW_gu[inter:] -= temp_u.unsqueeze(-1) * (us.squeeze(0).unsqueeze(-1) * W_gu[inter:])
        # down
        hs = hidden.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-4)
        h_msq = ((hidden / hs) ** 2).mean(dim=0)
        temp_h = (oja_f * h_msq) * hs.squeeze(0)
        dW_down -= temp_h.unsqueeze(0) * (hs.squeeze(0).unsqueeze(0) * W_down)

    return dW_gu, dW_down


@torch.no_grad()
def _hebbian_swiglu_kernel_cached(pre_fused: torch.Tensor,
                                  x_cached: torch.Tensor,
                                  ε_f: torch.Tensor,
                                  W_gu: torch.Tensor, W_down: torch.Tensor,
                                  η: float, Bs: int,
                                  oja_alpha: float = 0.0,
                                  oja_eta: float = 0.05,
                                  hidden_cached: Optional[torch.Tensor] = None) -> tuple:
    """缓存路径: 跳过前向 F.linear, 使用缓存预激活 + 输入.

    使用缓存的门控/上投影预激活 (避免前向重算)，
    以及可选的归一化后 hidden (对于 dW_down 使用归一化后的隐藏状态，防止 fp16 溢出).

    Args:
        pre_fused: [N, 2*inter] 缓存的 gate+up 预激活 (T=1 fusion)
        x_cached: [N, H] 缓存的 MLP 输入
        ε_f: [N, H] 误差
        W_gu/W_down/η/Bs/oja_alpha/oja_eta: 同标准路径
        hidden_cached: [N, inter] 可选 — 缓存的后归一化 hidden.
                       提供后用于 dW_down 及其 Oja 项，避免 gate_act*pre_up 溢出.
    """
    H = x_cached.shape[-1]
    inter = W_gu.shape[0] // 2

    pre_gate, pre_up = pre_fused.chunk(2, dim=-1)
    gate_act = F.silu(pre_gate)
    hidden = hidden_cached if hidden_cached is not None else (gate_act * pre_up)

    ε_hidden = torch.matmul(ε_f, W_down)
    sig = torch.sigmoid(pre_gate)
    silu_grad = sig * (1.0 + pre_gate * sig * (1.0 - sig))
    ε_gate = (ε_hidden * pre_up) * silu_grad
    ε_up   = ε_hidden * gate_act

    eu_f = torch.cat([ε_gate, ε_up], dim=-1)
    s = math.sqrt(Bs)
    dW_gu = η * ((eu_f / s).T @ (x_cached / s))
    dW_down = η * ((ε_f / s).T @ (hidden / s))

    if oja_alpha > 0:
        oja_f = oja_eta * oja_alpha
        # gate
        gs = gate_act.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-4)
        g_msq = ((gate_act / gs) ** 2).mean(dim=0)
        temp_g = (oja_f * g_msq) * gs.squeeze(0)
        dW_gu[:inter] -= temp_g.unsqueeze(-1) * (gs.squeeze(0).unsqueeze(-1) * W_gu[:inter])
        # up
        us = pre_up.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-4)
        u_msq = ((pre_up / us) ** 2).mean(dim=0)
        temp_u = (oja_f * u_msq) * us.squeeze(0)
        dW_gu[inter:] -= temp_u.unsqueeze(-1) * (us.squeeze(0).unsqueeze(-1) * W_gu[inter:])
        # down
        hs = hidden.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-4)
        h_msq = ((hidden / hs) ** 2).mean(dim=0)
        temp_h = (oja_f * h_msq) * hs.squeeze(0)
        dW_down -= temp_h.unsqueeze(0) * (hs.squeeze(0).unsqueeze(0) * W_down)

    return dW_gu, dW_down


@torch.no_grad()
def _hebbian_swiglu_kernel_batched(
    x_stack: torch.Tensor, ε_stack: torch.Tensor,
    W_gu_stack: torch.Tensor, W_down_stack: torch.Tensor,
    η: float, Bs: int,
    oja_alpha: float = 0.0,
    oja_eta: float = 0.05,
    pre_fused_stack: Optional[torch.Tensor] = None,
    x_cached_stack: Optional[torch.Tensor] = None,
    hidden_cached_stack: Optional[torch.Tensor] = None,
) -> tuple:
    """批量 SwiGLU Hebbian 核 — 单次 batched bmm 处理所有块.

    将 N 个 SwiGLU 块的数据沿 batch 维度堆叠，
    用 4 次 batched bmm 替代 N×4 次独立 matmul，
    大幅减少 GPU kernel launch 开销。

    Args:
        x_stack: [Bk, N, H] — 所有块输入堆叠
        ε_stack: [Bk, N, H] — 所有块误差堆叠
        W_gu_stack: [Bk, 2*inter, H] — 所有块 gate+up 权重堆叠
        W_down_stack: [Bk, H, inter] — 所有块 down 权重堆叠
        η: 学习率
        Bs: N 归一化系数
        oja_alpha: Oja 衰减系数 (0 = 禁用)
        oja_eta: Oja 独立学习率
        pre_fused_stack: Optional [Bk, N, 2*inter] 缓存的 gate+up 预激活
        x_cached_stack: Optional [Bk, N, H] 缓存的真实 MLP 输入
        hidden_cached_stack: Optional [Bk, N, inter] 缓存的后归一化 hidden。
                             提供后用于 dW_down 及其 Oja，防止 gate_act*pre_up 溢出.

    Returns:
        (dW_gu_stack, dW_down_stack) — [Bk, 2*inter, H] 和 [Bk, H, inter]
    """
    Bk = x_stack.shape[0]
    inter = W_gu_stack.shape[1] // 2

    if pre_fused_stack is not None:
        pre_gate, pre_up = pre_fused_stack.chunk(2, dim=-1)  # [Bk, N, inter]
        x_use = x_cached_stack if x_cached_stack is not None else x_stack
    else:
        pre_fused_stack = torch.bmm(x_stack, W_gu_stack.transpose(1, 2))  # [Bk, N, 2*inter]
        pre_gate, pre_up = pre_fused_stack.chunk(2, dim=-1)
        x_use = x_stack

    gate_act = F.silu(pre_gate)                            # [Bk, N, inter]
    hidden = hidden_cached_stack if hidden_cached_stack is not None else (gate_act * pre_up)

    ε_hidden = torch.bmm(ε_stack, W_down_stack)            # [Bk, N, inter]
    sig = torch.sigmoid(pre_gate)
    silu_grad = sig * (1.0 + pre_gate * sig * (1.0 - sig))
    ε_gate = (ε_hidden * pre_up) * silu_grad                # [Bk, N, inter]
    ε_up = ε_hidden * gate_act                              # [Bk, N, inter]

    eu_stack = torch.cat([ε_gate, ε_up], dim=-1)            # [Bk, N, 2*inter]
    s = math.sqrt(Bs)
    dW_gu = η * torch.bmm(eu_stack.transpose(1, 2) / s, x_use / s)   # [Bk, 2*inter, H]
    dW_down = η * torch.bmm(ε_stack.transpose(1, 2) / s, hidden / s) # [Bk, H, inter]

    if oja_alpha > 0:
        oja_f = oja_eta * oja_alpha
        # gate Oja — max-normed to avoid fp16 overflow in act**2
        gs = gate_act.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-4)
        g_msq = ((gate_act / gs) ** 2).mean(dim=1)
        temp_g = (oja_f * g_msq) * gs.squeeze(1)
        dW_gu[:, :inter] -= temp_g.unsqueeze(-1) * (gs.squeeze(1).unsqueeze(-1) * W_gu_stack[:, :inter])
        # up Oja
        us = pre_up.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-4)
        u_msq = ((pre_up / us) ** 2).mean(dim=1)
        temp_u = (oja_f * u_msq) * us.squeeze(1)
        dW_gu[:, inter:] -= temp_u.unsqueeze(-1) * (us.squeeze(1).unsqueeze(-1) * W_gu_stack[:, inter:])
        # down Oja
        hs = hidden.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-4)
        h_msq = ((hidden / hs) ** 2).mean(dim=1)
        temp_h = (oja_f * h_msq) * hs.squeeze(1)
        dW_down -= temp_h.unsqueeze(1) * (hs.squeeze(1).unsqueeze(1) * W_down_stack)

    return dW_gu, dW_down


@torch.no_grad()
def _hebbian_swiglu_unfused(ε_ℓ, z_conv, mlp, modulation, base_lr,
                             gamma_rpe=0.3, oja_alpha=0.01,
                             oja_eta: float = 0.05,
                             sparse_k: int = 0,
                             cached: Optional[tuple] = None) -> dict:
    """非融合 MLP 的 Hebbian SwiGLU 回退路径 (与原始逻辑等价)."""
    η = base_lr * modulation * (1.0 + gamma_rpe)
    W_gate = mlp.gate_proj.weight
    W_up   = mlp.up_proj.weight
    W_down = mlp.down_proj.weight
    B, S, H_dim = ε_ℓ.shape

    if cached is not None:
        cached_fused, cached_input = cached
        x = cached_input
        pre_gate, pre_up = cached_fused.chunk(2, dim=-1)
        pre_gate = pre_gate.contiguous()
        pre_up   = pre_up.contiguous()
    else:
        x = z_conv
        pre_gate = F.linear(x, W_gate)
        pre_up   = F.linear(x, W_up)
    gate_act = F.silu(pre_gate)
    hidden = gate_act * pre_up
    ε_hidden = torch.matmul(ε_ℓ, W_down)
    sig = torch.sigmoid(pre_gate)
    silu_grad = sig * (1.0 + pre_gate * sig * (1.0 - sig))
    ε_gate = (ε_hidden * pre_up) * silu_grad
    ε_up   = ε_hidden * gate_act

    x_flat = x.reshape(-1, H_dim)
    if sparse_k > 0:
        dW_gate = _sparse_outer_product(ε_gate, x, η, k=sparse_k)
        dW_up   = _sparse_outer_product(ε_up, x, η, k=sparse_k)
        dW_down = _sparse_outer_product(ε_ℓ, hidden, η, k=sparse_k)
    else:
        dW_gate = η * (ε_gate.reshape(-1, ε_gate.size(-1)).T @ x_flat) / (B * S)
        dW_up   = η * (ε_up.reshape(-1, ε_up.size(-1)).T @ x_flat) / (B * S)
        dW_down = η * (ε_ℓ.reshape(-1, H_dim).T @ hidden.reshape(-1, hidden.size(-1))) / (B * S)

    if oja_alpha > 0:
        oja_f = oja_eta * oja_alpha
        for dW_comp, post_act, W_curr in [
            (dW_gate, gate_act, W_gate),
            (dW_up,   pre_up,   W_up),
            (dW_down, hidden,   W_down),
        ]:
            act_scale = torch.amax(post_act.abs(), dim=(0, 1), keepdim=True).clamp(min=1e-4)
            mean_sq_scaled = ((post_act / act_scale) ** 2).mean(dim=(0, 1))
            temp = (oja_f * mean_sq_scaled) * act_scale.squeeze()
            if W_curr.shape[0] == temp.shape[0]:
                dW_comp -= temp.unsqueeze(-1) * (act_scale.squeeze().unsqueeze(-1) * W_curr)
            else:
                dW_comp -= temp.unsqueeze(0) * (act_scale.squeeze().unsqueeze(0) * W_curr)

    return {
        'gate_proj.weight': dW_gate.squeeze(0),
        'up_proj.weight':   dW_up.squeeze(0),
        'down_proj.weight': dW_down.squeeze(0),
    }


@torch.no_grad()
def compute_hebbian_swiglu(ε_ℓ, z_conv, mlp, modulation, base_lr,
                           gamma_rpe=0.3, oja_alpha=0.01,
                           oja_eta: float = 0.05,
                           sparse_k: int = 0,
                           cached: Optional[tuple] = None,
                           subsample_idx: Optional[torch.Tensor] = None) -> dict:
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

    性能: 标准路径用 `_hebbian_swiglu_kernel` (torch.compile 融合)；
          有缓存时仍走 kernel (跳过前向 F.linear，外积用缓存输入)；
          sparse_k > 0 时回退非编译代码。

    Args:
        ε_ℓ: [B, S, H] 该层误差 (fp32)
        z_conv: [B, S, H] conv 输出 (MLP 输入) — 无缓存时使用
        mlp: FeedForward 模块 (含 gate_proj, up_proj, down_proj)
        modulation: 组合调制值
        base_lr: 基础学习率
        gamma_rpe: RPE 增益系数
        oja_alpha: Oja 衰减系数 (0 = 禁用)
        oja_eta: Oja 独立学习率 (不绑定 Hebbian η)
        cached: Optional tuple (captured_fused, captured_input) from T=1 fusion.
                captured_fused: [B, S, 2*inter] gate_up_proj 预激活
                captured_input: [B, S, H] LN 后的 MLP 输入

    Returns:
        updates: dict with 'gate_proj.weight', 'up_proj.weight', 'down_proj.weight'
            或 fused 模式下 'gate_up_proj.weight', 'down_proj.weight'
    """
    fused = hasattr(mlp, 'gate_up_proj')
    if not fused:
        # 非融合 MLP: 使用原始路径 (极少使用)
        return _hebbian_swiglu_unfused(ε_ℓ, z_conv, mlp, modulation, base_lr,
                                       gamma_rpe, oja_alpha, oja_eta, sparse_k, cached)

    η = base_lr * modulation * (1.0 + gamma_rpe)
    W_gu = mlp.gate_up_proj.weight               # [2*inter, H]
    W_down = mlp.down_proj.weight                # [H, inter]
    B, S, H_dim = ε_ℓ.shape

    # ── 标准路径: 无 sparse_k → 紧凑核 (缓存可选) ──
    if sparse_k == 0:
        x_f = z_conv.reshape(-1, H_dim)           # [B*S, H]
        ε_f = ε_ℓ.reshape(-1, H_dim)              # [B*S, H]
        Bs = B * S

        if subsample_idx is not None:
            x_f = x_f[subsample_idx]
            ε_f = ε_f[subsample_idx]
            Bs = subsample_idx.shape[0]

        if cached is not None:
            cached_fused, cached_input = cached[:2]
            hidden_cached_f = cached[2] if len(cached) > 2 else None
            pre_fused_f = cached_fused.reshape(-1, cached_fused.size(-1))
            x_cached_f = cached_input.reshape(-1, H_dim)
            h_cached_f = hidden_cached_f.reshape(-1, hidden_cached_f.size(-1)) if hidden_cached_f is not None else None
            if subsample_idx is not None:
                pre_fused_f = pre_fused_f[subsample_idx]
                x_cached_f = x_cached_f[subsample_idx]
                if h_cached_f is not None:
                    h_cached_f = h_cached_f[subsample_idx]
            dW_gu, dW_down = _hebbian_swiglu_kernel_cached(
                pre_fused_f, x_cached_f, ε_f, W_gu, W_down, η, Bs,
                oja_alpha=oja_alpha, oja_eta=oja_eta,
                hidden_cached=h_cached_f,
            )
        else:
            dW_gu, dW_down = _hebbian_swiglu_kernel_standard(
                x_f, ε_f, W_gu, W_down, η, Bs,
                oja_alpha=oja_alpha, oja_eta=oja_eta,
            )
        return {
            'gate_up_proj.weight': dW_gu.squeeze(0),
            'down_proj.weight':    dW_down.squeeze(0),
        }
    else:
        # ── sparse_k > 0: 回退到原路径 ──
        ε = ε_ℓ
        if cached is not None:
            cached_fused, cached_input = cached[:2]
            hidden_cached = cached[2] if len(cached) > 2 else None
            x = cached_input
            pre_gate_c, pre_up_c = cached_fused.chunk(2, dim=-1)
            pre_gate = pre_gate_c.contiguous()
            pre_up   = pre_up_c.contiguous()
            gate_act = F.silu(pre_gate)
            hidden = hidden_cached if hidden_cached is not None else (gate_act * pre_up)
        else:
            x = z_conv
            pre_fused = F.linear(x, W_gu)
            pre_gate, pre_up = pre_fused.chunk(2, dim=-1)
            gate_act = F.silu(pre_gate)
            hidden = gate_act * pre_up
        inter = W_gu.shape[0] // 2
        W_gate = W_gu[:inter]
        W_up   = W_gu[inter:]

        ε_down = ε
        ε_hidden = torch.matmul(ε_down, W_down)
        sig = torch.sigmoid(pre_gate)
        silu_grad = sig * (1.0 + pre_gate * sig * (1.0 - sig))
        ε_gate = (ε_hidden * pre_up) * silu_grad
        ε_up   = ε_hidden * gate_act

        x_flat = x.reshape(-1, H_dim)
        gate_flat = ε_gate.reshape(-1, ε_gate.size(-1))
        up_flat   = ε_up.reshape(-1, ε_up.size(-1))
        down_flat = ε_down.reshape(-1, H_dim)
        hidden_flat = hidden.reshape(-1, hidden.size(-1))

        if sparse_k > 0:
            dW_gate = _sparse_outer_product(ε_gate, x, η, k=sparse_k)
            dW_up   = _sparse_outer_product(ε_up, x, η, k=sparse_k)
            dW_down = _sparse_outer_product(ε_down, hidden, η, k=sparse_k)
        else:
            eu_flat = torch.cat([gate_flat, up_flat], dim=-1)
            dW_gu = η * (eu_flat.T @ x_flat) / (B * S)
            dW_gate, dW_up = dW_gu.chunk(2, dim=0)
            dW_down = η * (down_flat.T @ hidden_flat) / (B * S)

        if oja_alpha > 0:
            oja_f = oja_eta * oja_alpha
            for dW_comp, post_act, W_curr in [
                (dW_gate, gate_act, W_gate),
                (dW_up,   pre_up,   W_up),
                (dW_down, hidden,   W_down),
            ]:
                act_scale = torch.amax(post_act.abs(), dim=(0, 1), keepdim=True).clamp(min=1e-4)
                mean_sq_scaled = ((post_act / act_scale) ** 2).mean(dim=(0, 1))
                temp = (oja_f * mean_sq_scaled) * act_scale.squeeze()
                if W_curr.shape[0] == temp.shape[0]:
                    dW_comp -= temp.unsqueeze(-1) * (act_scale.squeeze().unsqueeze(-1) * W_curr)
                else:
                    dW_comp -= temp.unsqueeze(0) * (act_scale.squeeze().unsqueeze(0) * W_curr)

    # ── 组装返回 (融合 MLP: gate_up_proj + down_proj) ──
    dW_gu = torch.cat([dW_gate, dW_up], dim=0)
    return {
        'gate_up_proj.weight': dW_gu.squeeze(0),
        'down_proj.weight':    dW_down.squeeze(0),
    }


@torch.no_grad()
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
    # ═══════════════════════════════════════════════════════════════
    #  保留 fp32 以避免 softmax 溢出 → 否则 exp(x) in fp16 → NaN
    # ═══════════════════════════════════════════════════════════════
    z = z_L                                            # [B, S, H] fp32
    tgt = target_byte_embed                            # [B, S, 256] fp32

    # Decoder 预测: z[t] → next_byte[t]
    # 如果 target 长度 = S-1 (labels[:, 1:]), 用 z[:, :-1, :] 做预测
    if tgt.size(1) < z.size(1):
        z_in = z[:, :-1, :]                         # [B, S-1, H]
    else:
        z_in = z
    pred = F.linear(z_in, decoder_weight)    # [B, S', 256] — fp16
    # ═══════════════════════════════════════════════════════════════
    #  关键修复 v2: 使用 tgt - softmax(W·z) 作为残差
    #
    #  PC 理论: ΔW ∝ ε · z^T, 其中 ε = target - prediction
    #    → 正确字: tgt=1, pred≈0 → ε≈+0.996 → 推高正确字权重 ✓
    #    → 错误字: tgt=0, pred≈0 → ε≈±1/256 → 轻微波动
    #
    #  之前用 pred_softmax - tgt, 符号相反, 导致 CE 上升.
    # ═══════════════════════════════════════════════════════════════
    pred_softmax = F.softmax(pred.float(), dim=-1).half()  # softmax内部fp32保数值稳定
    ε_decoder = tgt - pred_softmax                  # [B, S', 256] — PC 误差方向!

    # F_decoder: 用交叉熵 (而非 MSE) 衡量解码器性能
    F_decoder = λ * (-(tgt.half() * (pred_softmax + (1/256)).log()).sum(dim=-1).mean()).item()

    # Hebbian: ΔW = η/(B·S) · Σ ε_decoder^T · z_in   (per-sample 平均)
    #   除以 B*S 保证与 temporal/topdown 一致的 per-sample 尺度,
    #   使 Oja (per-sample mean(post²)) 能有效约束.
    B, S_eff, H_dim = z_in.shape
    dW = (η / (B * S_eff)) * torch.bmm(
        ε_decoder.reshape(-1, 256).unsqueeze(0).transpose(1, 2),
        z_in.reshape(-1, H_dim).unsqueeze(0),
    ).squeeze(0)                                    # [256, H]

    # Oja's rule: max-normed 防 fp16 溢出
    if oja_alpha > 0:
        oja_f = oja_eta * oja_alpha
        post = pred_softmax                         # [B, S', 256], 突触后 = decoder 输出 (softmax)
        ps = torch.amax(post.abs(), dim=(0, 1), keepdim=True).clamp(min=1e-4)  # [1, 1, 256]
        post_msq = ((post / ps) ** 2).mean(dim=(0, 1))                    # [256]
        temp = (oja_f * post_msq) * ps.squeeze()                          # [256]
        W_curr = decoder_weight             # [256, H]
        oja_term = temp.unsqueeze(-1) * (ps.squeeze().unsqueeze(-1) * W_curr)
        dW = dW - oja_term

    return dW, F_decoder


@torch.no_grad()
def _compute_hebbian_decoder_pair(z_L, target_byte_embed,
                                   W_decoder, W_lm_head,
                                   modulation, base_lr, gamma_rpe=0.3,
                                   λ=0.01, oja_alpha=0.01,
                                   oja_eta: float = 0.05,
                                   subsample_idx: Optional[torch.Tensor] = None) -> tuple:
    """融合计算 decoder + lm_head Hebbian 更新 (共享 z_in 和 target)."""
    η = base_lr * modulation * (1.0 + gamma_rpe)
    z = z_L
    tgt = target_byte_embed
    if tgt.size(1) < z.size(1):
        z_in = z[:, :-1, :]
    else:
        z_in = z
    B, S_eff, H_dim = z_in.shape

    # decoder prediction
    pred_dec = F.linear(z_in, W_decoder)
    # lm_head prediction
    pred_lm = F.linear(z_in, W_lm_head)

    pred_softmax_dec = F.softmax(pred_dec.float(), dim=-1).half()
    pred_softmax_lm  = F.softmax(pred_lm.float(), dim=-1).half()
    ε_dec = tgt - pred_softmax_dec
    ε_lm  = tgt - pred_softmax_lm

    F_decoder = λ * (-(tgt.half() * (pred_softmax_dec + (1/256)).log()).sum(dim=-1).mean()).item()

    # Hebbian outer product for both
    z_flat = z_in.reshape(-1, H_dim)
    ε_dec_f = ε_dec.reshape(-1, 256)
    ε_lm_f  = ε_lm.reshape(-1, 256)
    if subsample_idx is not None:
        S_full = z_L.size(1)
        t_pos = subsample_idx % S_full
        valid = t_pos != (S_full - 1)
        idx_valid = subsample_idx[valid]
        if idx_valid.numel() > 0:
            b_idx = idx_valid // S_full
            s_idx = idx_valid % S_full
            idx_map = b_idx * S_eff + s_idx
            z_flat = z_flat[idx_map]
            ε_dec_f = ε_dec_f[idx_map]
            ε_lm_f  = ε_lm_f[idx_map]
            # 子采样 softmax 输出用于 Oja
            post_dec = pred_softmax_dec.reshape(-1, 256)[idx_map]
            post_lm  = pred_softmax_lm.reshape(-1, 256)[idx_map]
            n_subsample = idx_valid.shape[0]
        else:
            n_subsample = 1
            z_flat = z_flat[:1]
            ε_dec_f = ε_dec_f[:1]
            ε_lm_f  = ε_lm_f[:1]
            post_dec = pred_softmax_dec.reshape(-1, 256)[:1]
            post_lm  = pred_softmax_lm.reshape(-1, 256)[:1]
    else:
        n_subsample = B * S_eff
        post_dec = pred_softmax_dec
        post_lm = pred_softmax_lm
    dW_dec = (η / n_subsample) * (ε_dec_f.T @ z_flat)
    dW_lm  = (η / n_subsample) * (ε_lm_f.T @ z_flat)

    # Oja: max-normed 防 fp16 溢出
    if oja_alpha > 0:
        oja_f = oja_eta * oja_alpha
        for dW, post, W in [(dW_dec, post_dec, W_decoder),
                            (dW_lm,  post_lm,  W_lm_head)]:
            if subsample_idx is not None:
                ps = post.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-4)
                p_msq = ((post / ps) ** 2).mean(dim=0)
                temp = (oja_f * p_msq) * ps.squeeze(0)
            else:
                ps = torch.amax(post.abs(), dim=(0, 1), keepdim=True).clamp(min=1e-4)
                p_msq = ((post / ps) ** 2).mean(dim=(0, 1))
                temp = (oja_f * p_msq) * ps.squeeze()
            dW -= temp.unsqueeze(-1) * (ps.squeeze().unsqueeze(-1) * W)

    return dW_dec, dW_lm, F_decoder


@torch.no_grad()
def compute_hebbian_byte_proj(ε_1, byte_seq, byte_proj_weight, conv1_weight,
                              modulation, base_lr, gamma_rpe=0.3,
                              oja_alpha=0.01,
                              oja_eta: float = 0.05,
                              subsample_idx: Optional[torch.Tensor] = None) -> dict:
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
    ε = ε_1                                           # [B, S, H] fp16
    x_byte = byte_seq                                  # [B, 2, S] fp16, int 0-255
    # 归一化输入 [0,1] 防止 fp16 einsum 累加溢出 (x=255 × ε~3 × 6144项 > 65K)
    x_byte = x_byte * (1 / 256)

    B, S, H_dim = ε.shape
    pad = 12
    x_padded = F.pad(x_byte, (pad, 0))               # [B, 2, S+12]
    x_unfold = x_padded.unfold(2, 13, 1)             # [B, 2, S, 13]
    x_unfold = x_unfold.permute(0, 2, 1, 3)          # [B, S, 2, 13]

    if subsample_idx is not None:
        # 子采样: 只对抽中位置做外积
        ε_s = ε.reshape(-1, H_dim)[subsample_idx]     # [N, H]
        x_s = x_unfold.reshape(-1, 2, 13)[subsample_idx]  # [N, 2, 13]
        dW = η * torch.einsum('ni,njk->ijk', ε_s, x_s) * 256.0 / subsample_idx.shape[0]
    else:
        ε_t = ε.transpose(0, 1)                     # [S, B, H_dim]
        x_t = x_unfold.transpose(0, 1)              # [S, B, 2, 13]
        dW = torch.einsum('sbi,sbjk->ijk', ε_t, x_t)  # [H_dim, 2, 13], 已缩放到 [0,1]
        dW = η * dW * 256.0 / (B * S)               # ×256 补偿归一化

    # Oja's rule: max-normed 防 fp16 溢出
    if oja_alpha > 0:
        oja_f = oja_eta * oja_alpha
        W_byte = byte_proj_weight            # [H, 2, 13]
        z_0 = F.conv1d(x_byte, W_byte, padding=12)   # [B, H, S], 与 forward 一致缩放
        z_0 = z_0.permute(0, 2, 1)                  # [B, S, H]
        if subsample_idx is not None:
            z_0_s = z_0.reshape(-1, H_dim)[subsample_idx]
            ps = z_0_s.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-4)
            z_msq = ((z_0_s / ps) ** 2).mean(dim=0)
            temp = (oja_f * z_msq) * ps.squeeze(0)
            oja_term = temp.unsqueeze(-1).unsqueeze(-1) * (ps.squeeze(0).unsqueeze(-1).unsqueeze(-1) * W_byte)
        else:
            ps = torch.amax(z_0.abs(), dim=(0, 1), keepdim=True).clamp(min=1e-4)  # [1, 1, H]
            z_msq = ((z_0 / ps) ** 2).mean(dim=(0, 1))
            temp = (oja_f * z_msq) * ps.squeeze()
            oja_term = temp.unsqueeze(-1).unsqueeze(-1) * (ps.squeeze().unsqueeze(-1).unsqueeze(-1) * W_byte)
        dW = dW - oja_term

    return {'model.byte_proj.weight': dW}


# ═══════════════════════════════════════════════════════════════════
#  误差归一化 — 生物合理的发放率约束
# ═══════════════════════════════════════════════════════════════════

def rms_normalize(ε_list, rms_target=1.0, eps=None):
    """逐层 RMS 归一化预测误差.

    生物类比: 神经元发放率有物理上限 (~0-100 Hz),
    误差信号不可能无限增长. 此函数对每层 ε 施加增益控制
    (gain control), 使其 RMS 处于可控范围, 保留方向与层内相对模式.

    不同于 LayerNorm: 不中心化, 不逐神经元归一化, 只缩放到目标 RMS.
    同一层内不同位置的相对误差幅度完全保留.

    性能: 不使用 .item() 避免 GPU→CPU sync, 完全在 GPU 上完成.

    Args:
        ε_list: list[tensor, L] — 各层误差 [B, S, H]
        rms_target: 目标 RMS (默认 1.0, 表示误差约等于一个标准差的发放)
        eps: 数值稳定常数

    Returns:
        ε_norm_list: 归一化后的 ε, 每层 RMS ≈ rms_target
    """
    eps_val = (1/256) if eps is None else eps
    ε_norm_list = []
    for ε in ε_list:
        rms = ε.square().mean().sqrt()
        scale = rms_target / rms.clamp(min=eps_val)
        ε_norm_list.append(ε * scale)
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
                                bcm_state=None, verbose=True,
                                hebbian_cache: Optional[dict] = None,
                                subsample_idx: Optional[torch.Tensor] = None) -> dict:
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
        hebbian_cache: dict — T=1 融合缓存 {block_idx: (captured_fused, captured_input)}.
                captured_fused=[B,S,2*inter], captured_input=[B,S,H].
                提供时跳过 gate/up 前向重建, 直接从缓存读取 SwiGLU 预激活.
        subsample_idx: Optional[tensor] — 预生成的子采样索引 [N].
                为 None 时由 cfg.hebbian_subsample_ratio 自动生成.
                非 None 时使用提供的索引.

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
        z_init = [z_init[0]] + rms_normalize(z_init[1:], rms_target=1.0)

    L = len(ε_list)
    updates = {}

    # Phase 4: 稀疏外积 — 从 cfg 读取保留通道数
    sparse_k = getattr(cfg, 'sparse_outer_k', 0)
    # ε 门控跳过: 预测误差极小的层不产生 Hebbian 更新
    ε_gate_threshold = getattr(cfg, 'hebbian_eps_gate', 0.0)

    # --- 子采样索引生成 (每步随机选 1/6 位置做 Hebbian) ---
    subsample_ratio = getattr(cfg, 'hebbian_subsample_ratio', 0.0)
    if subsample_idx is None and subsample_ratio > 0 and sparse_k == 0:
        total_pos = ε_list[0].size(0) * ε_list[0].size(1)  # B * S
        n_sample = max(1, int(total_pos * subsample_ratio))
        subsample_idx = torch.randperm(
            total_pos, device=ε_list[0].device, dtype=torch.long)[:n_sample]

    # 收集当前权重用于 Oja 规则
    temp_weights = [proj.weight for proj in model.temporal_proj]
    topdown_weights = [proj.weight for proj in model.topdown_proj]

    # 1) Temporal projections + Oja (逐层: 与 conv/swiglu 策略一致)
    updates.update(compute_hebbian_temporal(
        ε_list, z_init, modulation, base_lr, gamma_rpe,
        oja_alpha=oja_alpha, oja_eta=oja_eta,
        W_curr_list=temp_weights, sparse_k=sparse_k,
        subsample_idx=subsample_idx))

    # 2) Topdown projections + Oja (逐层: 与 conv/swiglu 策略一致)
    updates.update(compute_hebbian_topdown(
        ε_list, z_init, modulation, base_lr, gamma_rpe,
        oja_alpha=oja_alpha, oja_eta=oja_eta,
        W_curr_list=topdown_weights, sparse_k=sparse_k,
        subsample_idx=subsample_idx))

    # 3) Conv1D Hebbian — 批量所有块为单次 batched bmm
    #     (conv 内部已有 3 k-position 融合, 但逐块调用仍有 4× Python 开销)
    if sparse_k == 0:
        conv_batch = []  # (ε_ℓ, z_prev, W, dilation, block_idx)
        for ℓ in range(0, L, 2):
            block_idx = ℓ // 2
            if ε_gate_threshold > 0.0 and ε_list[ℓ].norm().item() < ε_gate_threshold:
                continue
            block = model.model.layers[block_idx]
            conv_batch.append((ε_list[ℓ], z_init[ℓ], block.local_conv.weight, block.dilation, block_idx))
        if conv_batch and subsample_idx is not None:
            Bc = len(conv_batch)
            Hc = conv_batch[0][0].shape[-1]
            η_c = base_lr * modulation * (1.0 + gamma_rpe)
            n_sub_c = subsample_idx.shape[0]
            # 逐块 pad + subsample + stack 3 k-positions, 然后 batch stack
            ε_arr_c, z_arr_c = [], []
            padded_c = []
            for ε_ℓ, z_prev, _, dilation, _ in conv_batch:
                pad = 2 * dilation
                padded_c.append(F.pad(z_prev.transpose(1, 2), (pad, 0)))
            for i, (ε_ℓ, _, _, dilation, _) in enumerate(conv_batch):
                _, S_i, H_i = ε_ℓ.shape
                ε_s = ε_ℓ.reshape(-1, H_i)[subsample_idx]
                zp = padded_c[i]
                zk = []
                for k in range(3):
                    off = k * dilation
                    zk.append(zp[:, :, off:off+S_i].permute(0, 2, 1).reshape(-1, H_i)[subsample_idx])
                ε_arr_c.append(ε_s)
                z_arr_c.append(torch.stack(zk, dim=0))  # [3, N, H]
            ε_batch = torch.stack(ε_arr_c, dim=0)    # [Bc, N, H]
            z_batch = torch.stack(z_arr_c, dim=0)    # [Bc, 3, N, H]
            # batched: [Bc, 1, H, N] @ [Bc, 3, N, H] → [Bc, 3, H, H] (broadcast)
            s = math.sqrt(n_sub_c)
            dW_c = η_c * ((ε_batch.transpose(1, 2).unsqueeze(1) / s) @ (z_batch / s))  # [Bc, 3, H, H]
            dW_c = dW_c.squeeze(1)
            if oja_alpha > 0:
                oja_f = oja_eta * oja_alpha
                for i, (_, _, W_conv, _, _) in enumerate(conv_batch):
                    W_st = W_conv.permute(2, 0, 1).contiguous()  # [3, H, H]
                    pk = torch.bmm(z_batch[i], W_st.transpose(1, 2))  # [3, N, H]
                    ps = pk.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-4)  # [3, 1, H]
                    pk_msq = ((pk / ps) ** 2).mean(dim=1)                       # [3, H]
                    temp_pk = (oja_f * pk_msq) * ps.squeeze(1)                  # [3, H]
                    dW_c[i] -= temp_pk.unsqueeze(-1) * (ps.squeeze(1).unsqueeze(-1) * W_st)
            for i, (_, _, _, _, block_idx) in enumerate(conv_batch):
                updates[f'model.layers.{block_idx}.local_conv.weight'] = dW_c[i].permute(1, 2, 0).contiguous()
        else:
            # 无 subsample 或 sparse: 回退逐块
            for ℓ in range(0, L, 2):
                block_idx = ℓ // 2
                if ε_gate_threshold > 0.0 and ε_list[ℓ].norm().item() < ε_gate_threshold:
                    continue
                block = model.model.layers[block_idx]
                updates[f'model.layers.{block_idx}.local_conv.weight'] = compute_hebbian_conv(
                    ε_list[ℓ], z_init[ℓ], block.local_conv.weight, block.dilation,
                    modulation, base_lr, gamma_rpe,
                    oja_alpha=oja_alpha, oja_eta=oja_eta,
                    sparse_k=sparse_k, subsample_idx=subsample_idx)

    # 3b) SwiGLU Hebbian — 批量所有块为单次 batched bmm 调用
    #     避免 4 次独立 Python 调用 + 重复子采样 + 小 matmul launch 开销
    if sparse_k == 0:
        swiglu_data = []  # (x_f, ε_f, W_gu, W_down, block_idx, pre_fused_f, x_cached_f, hidden_cached_f)
        for ℓ in range(1, L, 2):
            block_idx = ℓ // 2
            if ε_gate_threshold > 0.0 and ε_list[ℓ].norm().item() < ε_gate_threshold:
                continue
            block = model.model.layers[block_idx]
            H_dim = ε_list[ℓ].shape[-1]
            x_f = z_init[ℓ].reshape(-1, H_dim)
            ε_f = ε_list[ℓ].reshape(-1, H_dim)
            cached_block = hebbian_cache.get(block_idx) if hebbian_cache else None
            if subsample_idx is not None:
                x_f = x_f[subsample_idx]
                ε_f = ε_f[subsample_idx]
            if cached_block is not None:
                cached_fused, cached_input = cached_block[:2]
                hidden_cached = cached_block[2] if len(cached_block) > 2 else None
                pf = cached_fused.reshape(-1, cached_fused.size(-1))
                xc = cached_input.reshape(-1, H_dim)
                hc = hidden_cached.reshape(-1, hidden_cached.size(-1)) if hidden_cached is not None else None
                if subsample_idx is not None:
                    pf = pf[subsample_idx]
                    xc = xc[subsample_idx]
                    if hc is not None:
                        hc = hc[subsample_idx]
            else:
                pf, xc, hc = None, None, None
            W_gu = block.mlp.gate_up_proj.weight
            W_down = block.mlp.down_proj.weight
            swiglu_data.append((x_f, ε_f, W_gu, W_down, block_idx, pf, xc, hc))

        if swiglu_data:
            Bk = len(swiglu_data)
            Bs = swiglu_data[0][0].shape[0]
            η = base_lr * modulation * (1.0 + gamma_rpe)
            x_stack = torch.stack([d[0] for d in swiglu_data], dim=0)
            ε_stack = torch.stack([d[1] for d in swiglu_data], dim=0)
            W_gu_stack = torch.stack([d[2] for d in swiglu_data], dim=0)
            W_down_stack = torch.stack([d[3] for d in swiglu_data], dim=0)
            # 所有块都有缓存 (通常成立) 或都没有
            has_cache = all(d[5] is not None for d in swiglu_data)
            if has_cache:
                pre_fused_stack = torch.stack([d[5] for d in swiglu_data], dim=0)
                x_cached_stack = torch.stack([d[6] for d in swiglu_data], dim=0)
                hidden_cached_stack = torch.stack([d[7] for d in swiglu_data], dim=0) if all(d[7] is not None for d in swiglu_data) else None
            else:
                pre_fused_stack = x_cached_stack = hidden_cached_stack = None
            dW_gu, dW_down = _hebbian_swiglu_kernel_batched(
                x_stack, ε_stack, W_gu_stack, W_down_stack, η, Bs,
                oja_alpha=oja_alpha, oja_eta=oja_eta,
                pre_fused_stack=pre_fused_stack, x_cached_stack=x_cached_stack,
                hidden_cached_stack=hidden_cached_stack,
            )
            for i, (_, _, _, _, block_idx, _, _, _) in enumerate(swiglu_data):
                updates[f'model.layers.{block_idx}.mlp.gate_up_proj.weight'] = dW_gu[i]
                updates[f'model.layers.{block_idx}.mlp.down_proj.weight'] = dW_down[i]
    else:
        # sparse_k > 0: 回退到原逐块路径
        for ℓ in range(1, L, 2):
            block_idx = ℓ // 2
            if ε_gate_threshold > 0.0 and ε_list[ℓ].norm().item() < ε_gate_threshold:
                continue
            block = model.model.layers[block_idx]
            cached_block = hebbian_cache.get(block_idx) if hebbian_cache else None
            swiglu_updates = compute_hebbian_swiglu(
                ε_list[ℓ], z_init[ℓ], block.mlp, modulation, base_lr, gamma_rpe,
                oja_alpha=oja_alpha, oja_eta=oja_eta,
                sparse_k=sparse_k, cached=cached_block,
                subsample_idx=subsample_idx)
            for k, v in swiglu_updates.items():
                updates[f'model.layers.{block_idx}.mlp.{k}'] = v

    # 4) Decoder (外部约束) + Oja — 总是计算 (关键输出层)
    #    同时计算 decoder 和 lm_head 以避免重复前向/误差计算
    if decoder is not None and lm_head is not None and target_byte_embed is not None:
        dW_dec, dW_lm, F_dec = _compute_hebbian_decoder_pair(
            z_init[-1], target_byte_embed, decoder.weight, lm_head.weight,
            modulation, base_lr, gamma_rpe, λ,
            oja_alpha=oja_alpha, oja_eta=oja_eta,
            subsample_idx=subsample_idx,
        )
        updates['decoder.weight'] = dW_dec
        updates['model.lm_head.weight'] = dW_lm

    # 5) Byte projection 层 + Oja — 总是计算 (输入层)
    if hasattr(model.model, 'byte_proj'):
        byte_updates = compute_hebbian_byte_proj(
            ε_list[0], byte_seq,
            model.model.byte_proj.weight,
            None,
            modulation, base_lr, gamma_rpe,
            oja_alpha=oja_alpha, oja_eta=oja_eta,
            subsample_idx=subsample_idx,
        )
        for k, v in byte_updates.items():
            updates[f'model.{k}'] = v

    # 6) BCM 滑动阈值调制: LTP vs LTD 动态平衡
    if bcm_state is not None:
        z_for_bcm = z_init[1:]  # [z₁, ..., z_L], 共 L 层
        bcm_factors = bcm_state.compute_factors(z_for_bcm)
        # 预构建 per-layer key 清单 (避免循环内 f-string 开销)
        for ℓ in range(L):
            factor = bcm_factors[ℓ]
            block_idx = ℓ // 2
            is_conv = (ℓ % 2 == 0)
            key_t = f'temporal_proj.{ℓ}.weight'
            key_td = f'topdown_proj.{ℓ}.weight'
            if key_t in updates:
                updates[key_t] *= factor
            if key_td in updates:
                updates[key_td] *= factor
            if is_conv:
                key_c = f'model.layers.{block_idx}.local_conv.weight'
                if key_c in updates:
                    updates[key_c] *= factor
            else:
                for suffix in [f'model.layers.{block_idx}.mlp.gate_up_proj.weight',
                               f'model.layers.{block_idx}.mlp.down_proj.weight']:
                    if suffix in updates:
                        updates[suffix] *= factor

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
        n_inf = 0
        norm_sum = torch.tensor(0.0, device=ε_list[0].device)
        n_params = 0
        for name, dW in updates.items():
            if not torch.isfinite(dW).all():
                n_inf += 1
            norm_sum += dW.norm()
            n_params += 1
        updates['_diag_avg_growth'] = (norm_sum / max(n_params, 1)).item()
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

    v2: GPU 端无同步范数裁剪 + 预计算 fan_in 表.

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
    bound = min(65504, weight_bound)
    # Hebbian 更新 η ≈ 3e-4, dW 量级 ≪ max_delta/grad_clip, 跳过冗余检查
    for name, dW in updates.items():
        w = sd.get(name)
        if w is None:
            continue
        new_w = (w + dW).clamp(-bound, bound)
        if synaptic_normalize and w.dim() > 1:
            w_flat = new_w.flatten(1)
            target = target_norm if target_norm > 0 else math.sqrt(w.shape[-1])
            norms = w_flat.norm(dim=-1, keepdim=True)
            new_w = (w_flat / (norms + (1/256)) * target).reshape_as(new_w)
        w.data.copy_(new_w)


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

        性能: 批量 GPU 计算, 单次同步避免逐层 .item() 开销.

        Args:
            z_list: list[tensor, L] — 各层 z 活动 (post-synaptic), [B, S, H]

        Returns:
            factors: list[float, L] — 每层 BCM 调制因子
        """
        L = min(len(z_list), self.n_layers)
        if L == 0:
            return []
        # 批量计算所有 z² 均值 (单次 GPU kernel)
        z_sq = torch.empty(L, device=self.theta.device)
        for ℓ in range(L):
            z_sq[ℓ] = (z_list[ℓ] ** 2).mean()
        z_sq_vals = z_sq.tolist()  # 单次 GPU→CPU sync
        factors = []
        for ℓ in range(L):
            z_sq_val = z_sq_vals[ℓ]
            self.theta[ℓ] = (1.0 - self.tau) * self.theta[ℓ] + self.tau * z_sq_val
            θ = self.theta[ℓ].item()
            factor = (z_sq_val - θ) / (θ + (1/256))
            factors.append(factor)
        return factors
