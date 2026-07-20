"""Cyrene PC 层：每子层添加 predict()，管理所有 z 节点。

将 L 个 LocalConvBlock 展开为 2L 个子层节点：
  z_0 = byte_proj(input)           # 固定 (感觉输入)
  z_1 = Conv₁(LN(z_0)) + z_0       # Block 1 Conv 输出
  z_2 = MLP₁(LN(z_1)) + z_1        # Block 1 MLP 输出
  …
  z_{2L} = MLP_L(LN(...)) + ...    # Block L MLP 输出

子层 ℓ 的预测: μ_ℓ = sublayer_ℓ(LN(z_{ℓ-1})) + z_{ℓ-1}
误差: ε_ℓ = z_ℓ - μ_ℓ
"""
import torch
import torch.nn.functional as F
from torch import nn

from model.pc.local_blocks import LateralInhibition, SalienceGate
from model.pc.pc_backbone_local import CyreneBackbone

# ── 自适应精度噪声 (生物神经元膜电位噪声) ──────────────

class AdaptiveNeuralNoise(nn.Module):
    """每神经元自适应噪声注入 — 生物膜电位噪声的 ML 模拟。

    生物灵感：神经元的膜电位天然存在 channel noise / synaptic noise，
    高放电率神经元产生更大的膜电位波动 → 有效精度降低 → 防止精确数值累积。

    每个隐藏维度（神经元）维护 EMA 激活均值，噪声标准差正比于均值。
    高激活神经元自动获得更多噪声 → fp16 下不会累积到溢出值。
    低激活神经元几乎不受影响（噪声很小）。

    v2: 支持 step_interval，每 N 步执行一次噪声注入，减少 CUDA 随机数生成开销。
    """

    def __init__(self, hidden_dim: int, base_noise: float = 3e-5, ema_decay: float = 0.999,
                 step_interval: int = 1):
        super().__init__()
        self.base_noise = base_noise
        self.ema_decay = ema_decay
        self.hidden_dim = hidden_dim
        self.step_interval = max(1, step_interval)
        self._step_counter = 0
        self.register_buffer('_running_std', torch.zeros(hidden_dim) + base_noise)
        self.register_buffer('_running_mean', torch.zeros(hidden_dim))

    def forward(self, z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """注入自适应噪声。每 step_interval 步执行一次，其余步直接返回 z。"""
        if not self.training or not torch.is_grad_enabled():
            return z

        self._step_counter += 1
        if self._step_counter % self.step_interval != 0:
            return z

        with torch.no_grad():
            act_mean = z.abs().mean(dim=(0, 1))  # [D], 当前批次的每维度均值
            self._running_mean.mul_(self.ema_decay).add_(act_mean * (1 - self.ema_decay))
            noise_std = self.base_noise * alpha * (self._running_mean + 1e-8)
            noise_std = noise_std.clamp(max=0.5).to(z.dtype)
        noise = torch.randn_like(z) * noise_std.view(1, 1, self.hidden_dim)
        return z + noise


# ── UTF-8 结构加权 CE 权重 ──
def _utf8_ce_weight(device='cpu'):
    """UTF-8 字节级加权: lead byte 权重 2.0, continuation byte 权重 0.5."""
    w = torch.ones(256, device=device)
    w[0xE4:0xEA] = 2.0   # 中文字第一字节 (0xE4-0xE9)
    w[0x80:0xC0] = 0.5   # 续字节 (0x80-0xBF)
    return w



# BaseCyrenePC — 时空预测编码基类
# 每层节点做三维局部预测: 自下而上 + 时序 + 自上而下
# F = Σ ½·‖ε_ℓ(t)‖²  (纯预测误差, 无 CE, 无 token loss)

class BaseCyrenePC(nn.Module):
    """时空预测编码基类。

    核心转变: "预测下一个 token" → "预测下一个神经状态"
    每层 z_ℓ(t) 收三路预测后合并为 μ_total, 误差 ε = z - μ_total 驱动学习。

    子类 (CyrenePC) 负责创建具体 backbone 并重写 init_z / predict / forward_with_ce / get_position_embeddings。
    此类仅提供共享的时空推理框架方法。

    时间维度: z_by_layer[ℓ] = [bsz, seq_len, hidden_size], ℓ=0..L
    - ℓ=0: 字节映射输出 (固定, 感觉输入)
    - ℓ=1..L: 子层输出 (变量, 推理时更新)
    """

    # __init__ 由子类负责 — CyrenePC 直接调用 nn.Module.__init__(self)

    @torch.no_grad()
    def init_z_seq(self, input_ids):
        """前向初始化所有 z 节点, 返回按层组织的列表。

        Returns: z_by_layer[ℓ] = [bsz, seq_len, hidden_size], ℓ=0..L
        """
        return self.init_z(input_ids)

    # init_z, forward_with_ce, predict, get_position_embeddings
    # 由子类 (CyrenePC) 实现

    # ── 批量化 temporal/topdown (Phase E) ────────────────────

    def _compute_batched_temporal_topdown(self, z_list, seq_len):
        """批量化 temporal/topdown 预测 (fp32 安全). 使用缓存权重避免每步 torch.stack 分配。"""
        L = self.num_sub_layers
        if seq_len <= 1:
            return None, None

        # 全部提升到 fp32 防 bmm 溢出
        z_sub = torch.stack(z_list[1:], dim=0).float()  # [L, B, S, H]  — z 仍需 stack
        B, H = z_sub.size(1), z_sub.size(3)

        # 使用缓存的 stack 权重，避免每步 torch.stack([p.weight.float() ...])
        W_temp = getattr(self, '_cached_temporal_w', None)
        if W_temp is None:
            W_temp = torch.stack([p.weight.float() for p in self.temporal_proj])  # [L, H, H]
        z_flat = z_sub[:, :, :-1, :].reshape(L, -1, H)
        μ_temp_flat = torch.bmm(z_flat, W_temp)  # [L, B*(S-1), H] fp32
        μ_temp = μ_temp_flat.reshape(L, B, -1, H)  # fp32
        pad = torch.zeros(L, B, 1, H, device=z_sub.device, dtype=torch.float32)
        μ_temp_all = torch.cat([pad, μ_temp], dim=2)

        if L > 1:
            z_down = torch.stack(z_list[2:], dim=0).float()  # [L-1, B, S, H]
            W_down = getattr(self, '_cached_topdown_w', None)
            if W_down is None:
                W_down = torch.stack([p.weight.float() for p in self.topdown_proj])
            z_down_flat = z_down[:, :, :-1, :].reshape(L - 1, -1, H)
            μ_down_flat = torch.bmm(z_down_flat, W_down)
            μ_down = μ_down_flat.reshape(L - 1, B, -1, H)  # fp32
            pad_d = torch.zeros(L - 1, B, 1, H, device=z_sub.device, dtype=torch.float32)
            μ_down_all = torch.cat([pad_d, μ_down], dim=2)
        else:
            μ_down_all = None

        return μ_temp_all, μ_down_all

    # ── 时空推理单步 ──────────────────────────────────────────

    def spatiotemporal_infer_step(self, z_by_layer, pos_emb, gamma, padding_mask=None,
                                    return_errors=True, return_pred_loss=False,
                                    precision_scales=None):
        """单步时空推理: 对所有层 ell=1..L 更新 z.

        推理始终在 no_grad 下运行 (手动 ∇F_z = ε).
        F_pred 通过 compute_spatiotemporal_loss 在推理后单独计算。
        """
        return self._spatiotemporal_infer_step(
            z_by_layer, pos_emb, gamma, padding_mask,
            return_errors=return_errors,
            return_pred_loss=return_pred_loss,
            precision_scales=precision_scales,
        )

    def _spatiotemporal_infer_step(self, z_by_layer, pos_emb, gamma, padding_mask,
                                     return_errors=True, return_pred_loss=False,
                                     precision_scales=None, return_ε=False,
                                     skip_bottom_up=False):
        """单步时空推理: 手动 ∇F_z = ε, 同时可选累积 F_pred.

        核心原则:
          - z 始终不追踪梯度 (detach)
          - 如果 grad 已启用 (最后一轮), F_pred 通过 predict/temporal/topdown 累积梯度
          - 如果 no_grad, 返回 (new_z, errors_info, F_val_scalar, None)

        Returns:
            new_z: list[tensor] — 更新后的 z (均无 grad)
            errors_info: list[(e_sq, e_norm), ...] 或 [] (grad 模式下为空)
            F_val: 本轮 F 值 (纯标量, 无 grad)
            F_pred: tensor 或 None — 有 grad_fn 的 F (仅在 grad + return_pred_loss 时)
            errors_ε: list[tensor, L] (仅 return_ε=True 时)
        """
        L = self.num_sub_layers
        # ── z 保持 fp16 (原生格式), 仅在 per_pos_gain / predict 内提升到 fp32 ──
        z_det = [z.detach() for z in z_by_layer]  # no .float() — keep fp16
        seq_len = z_det[0].size(1)
        B = z_det[0].size(0)
        has_grad = torch.is_grad_enabled()

        errors_ε = []
        F_val = 0.0  # scalar, for logging
        F_pred = None

        # ── Phase 1: 依赖阈值门控 ──
        # 静默通道不参与前向预测 → 自然稀疏 (替代 hardcoded top-k)
        self._total_active_channels = 0
        self._n_active_channels = 0
        if hasattr(self, '_act_threshold') and self._act_threshold is not None:
            z_gated = [z_det[0]]  # z_0 (感觉输入) 无门控
            for ℓ in range(1, L + 1):
                z_ℓ = z_det[ℓ]
                act = z_ℓ.float().abs().mean(dim=(0, 1))  # [H] — 跨 B×S 的 per-channel 激活度
                th = self._act_threshold[ℓ - 1]
                mask = (act > th).float().to(z_ℓ.dtype)
                z_gated.append(z_ℓ * mask.view(1, 1, -1))

                # EMA 跟踪激活分布
                th.mul_(1 - self._act_ema_decay).add_(act * self._act_ema_decay)

                # 稳态可塑性: 活跃率偏离目标 → 调节阈值 (纯 tensor 操作, 无 .item())
                active_ratio = mask.mean()
                excess = active_ratio - self._act_target_ratio
                factor = 1 + self._act_homeo_rate * torch.sign(excess) * (torch.abs(excess) > 0.05).float()
                th.mul_(factor.detach())

                # 仅记录 scalar 用于日志 (冷路径), 不阻滞 GPU
                self._total_active_channels += mask.sum().item()
                self._n_active_channels += mask.numel()
        else:
            z_gated = z_det  # fallback: 无门控

        μ_temp_all, μ_down_all = self._compute_batched_temporal_topdown(z_gated, seq_len)

        for ℓ in range(1, L + 1):
            z_prev_gated = z_gated[ℓ - 1]  # 门控后的 z_{ℓ-1}
            # ── 底部上预测 μ_bu ──
            if skip_bottom_up and self._cached_mu_bu is not None:
                # 使用 init_z 中缓存的 pre-gate 预测值, 跳过冗余 predict()
                # 门控对 z_prev 的影响近似忽略 (T=1 时门控 ≈ 1.0)
                μ_bu = self._cached_mu_bu[ℓ - 1].float()
            else:
                μ_bu = self.predict(ℓ, z_prev_gated, pos_emb, fp32_out=True)
            μ_bu_res = μ_bu + z_det[ℓ - 1].float()  # 残差用 RAW z (预测值用门控输入)

            μ_temp = μ_temp_all[ℓ - 1] if μ_temp_all is not None else torch.zeros_like(z_det[ℓ]).float()
            μ_down = μ_down_all[ℓ - 1] if (μ_down_all is not None and ℓ < L) else torch.zeros_like(z_det[ℓ]).float()

            μ_total = μ_bu_res + μ_temp + μ_down
            ε = z_det[ℓ].float() - μ_total
            # ── 归一化 ε: F_pred 也基于相对误差, 消除 z 绝对尺度影响 ──
            ε_abs_mean = ε.abs().mean(dim=-1, keepdim=True) + 1.0
            ε_norm = ε / ε_abs_mean

            # F_pred with grad (基于归一化 ε)
            if has_grad and return_pred_loss:
                π = precision_scales[ℓ - 1] if precision_scales is not None else 1.0
                fe = 0.5 * π * (ε_norm ** 2).mean()
                if F_pred is None:
                    F_pred = fe
                else:
                    F_pred = F_pred + fe

            # F_val for logging (同样基于归一化 ε)
            if padding_mask is not None:
                mask = padding_mask.unsqueeze(-1).to(ε_norm.dtype)
                F_val = F_val + 0.5 * ((ε_norm * mask) ** 2).mean()
            else:
                F_val = F_val + 0.5 * (ε_norm ** 2).mean()

            errors_ε.append(ε)

        if F_pred is not None:
            F_pred = F_pred / L

        # ── 位置级增益: 底层(ℓ=1)误差大的位置 → 放大更新 ──
        with torch.no_grad():
            ε_bottom_det = errors_ε[0].detach().float()
            per_pos_mag = ε_bottom_det.norm(dim=-1, keepdim=True)
            per_pos_gain = 1.0 + 0.3 * per_pos_mag / (per_pos_mag.mean(dim=-1, keepdim=True) + 1e-8)
            per_pos_gain = per_pos_gain.clamp(max=10.0)  # fp32

        # 更新 z — 始终 detach 切断梯度流, 输出保持 fp16 (原生格式)
        new_z = [z_det[0]]
        errors_info = []

        for ℓ in range(1, L + 1):
            ε = errors_ε[ℓ - 1]
            ε_abs_mean = ε.abs().mean(dim=-1, keepdim=True) + 1.0
            ε_norm = ε / ε_abs_mean
            dz = gamma * ε_norm * per_pos_gain
            # fp16 安全: 截断到 [-1e4, 1e4] 防止 .half() 溢出到 Inf
            z_fp32 = (z_det[ℓ].float() - dz).detach()
            z_clamped = z_fp32.clamp(-1e4, 1e4)
            new_z.append(z_clamped.half())
            if return_errors and not has_grad and not self._graph_capture_mode:
                errors_info.append((
                    (ε.detach().float() ** 2).mean(),  # e_sq tensor (Phase 7a: defer .item())
                    ε.detach().float().norm(),          # e_norm tensor (Phase 7a: defer .item())
                ))

        # ── Phase 3 接入: 能量代价惩罚活跃通道 ──
        if hasattr(self, '_act_threshold') and self._act_threshold is not None and self._act_energy_cost > 0:
            active_ratio = (self._total_active_channels + 1e-8) / (self._n_active_channels + 1e-8)
            energy_penalty = self._act_energy_cost * active_ratio
            F_val = F_val + energy_penalty

        # Phase 7a: Always tensor, caller does .item() at final usage
        F_val_ret = F_val.detach()
        if return_ε and not has_grad:
            return new_z, errors_info, F_val_ret, F_pred, errors_ε
        return new_z, errors_info, F_val_ret, F_pred

    # ── 时空推理循环 ──────────────────────────────────────────

    def spatiotemporal_infer_step(self, z_by_layer, pos_emb, gamma, padding_mask=None,
                                    return_errors=True, return_pred_loss=False,
                                    precision_scales=None, return_ε=False,
                                    skip_bottom_up=False):
        """单步时空推理: 对所有层 ell=1..L 更新 z.

        推理始终在 no_grad 下运行 (手动 ∇F_z = ε).
        F_pred 通过 compute_spatiotemporal_loss 在推理后单独计算。
        """
        return self._spatiotemporal_infer_step(
            z_by_layer, pos_emb, gamma, padding_mask,
            return_errors=return_errors,
            return_pred_loss=return_pred_loss,
            precision_scales=precision_scales,
            return_ε=return_ε,
            skip_bottom_up=skip_bottom_up,
        )

    def spatiotemporal_infer(self, z_by_layer, pos_emb, gamma=0.1, T=2, padding_mask=None,
                               return_errors=True, return_pred_loss=False,
                               precision_scales=None, ach_value=0.5,
                               return_ε=False,
                               adaptive_T=False, convergence_threshold=0.01,
                               patience=2, min_T=2,
                               skip_bottom_up=False):
        """T 步时空推理循环, 返回收敛后的 z 和 F_pred.

        推理策略: 前 T-1 步纯 no_grad (快速), 最后一步启用 grad 累积 F_pred.
        消除 compute_spatiotemporal_loss 的二次前向开销.

        v3: 误差比率追踪 — error_ratio = F_t / F_{t-1}.
        error_ratio > 1 (误差涨)→ γ↑ 加速更新;
        error_ratio < 1 (误差跌)→ γ↓ 微调.
        存储 self._last_error_ratio 供外部读取 (训练循环用).

        v4 (自适应 T): 当 F_val 收敛 (|ΔF|/F < threshold 连续 patience 步)
        且 t >= min_T 时提前终止推理循环, 减少 FP 开销。

        Args:
            ach_value: ACh 调制值 (∈ (0,1)), 影响 gamma_eff = gamma · (1 + 0.3·ACh)
            return_ε: True 时额外返回 ε_by_layer (用于 bp_free Hebbian 更新)
            adaptive_T: 启用自适应 T 终止
            convergence_threshold: F_val 相对变化阈值 (默认 0.01 = 1%)
            patience: 连续满足阈值步数 (默认 2)
            min_T: 最小推理步数 (默认 2)

        Returns:
            z_by_layer: list[tensor] — 收敛后的 z (无 grad)
            errors_hist: list of errors_info
            F_hist: list of F_val scalars
            F_pred: tensor or None — 有 grad_fn 的 F (仅 return_pred_loss=True)
            如果 return_ε=True, 追加 ε_by_layer: list[tensor, L] — 各层误差
        """
        errors_hist = []
        F_hist = []
        F_pred = None
        self._last_error_ratio = 1.0
        _converged_count = 0
        _T_actual = 0

        # ── 预缓存 Conv1D fp32 权重，消除 predict() 中 24×/步的冗余 allocate ──
        self._build_conv_fp32_cache()
        for t in range(T):
            _T_actual = t + 1
            # ── 误差比率调制 γ: error_ratio>1→加速, <1→微调 ──
            if t > 0 and len(F_hist) >= 2:
                error_ratio = (F_hist[-1] + 1e-8) / (F_hist[-2] + 1e-8)
                # fp16 稳定性: error_ratio 无限增长 → gamma_eff 爆炸 → z 更新 NaN
                if getattr(self, '_graph_capture_mode', False):
                    error_ratio = torch.clamp(error_ratio, max=10.0)
                else:
                    error_ratio = min(error_ratio, 10.0)
                self._last_error_ratio = error_ratio
                gamma_eff = gamma * (0.5 + error_ratio)
                # gamma_eff 上限防 fp16 溢出
                if getattr(self, '_graph_capture_mode', False):
                    gamma_eff = torch.clamp(gamma_eff, max=10.0)
                else:
                    gamma_eff = min(gamma_eff, 10.0)
            else:
                gamma_eff = gamma

            is_last = (t == T - 1)
            # γ 受 ACh 调制: gamma_eff *= (1 + 0.3·ACh)
            gamma_eff_ach = gamma_eff * (1.0 + 0.3 * ach_value)

            # 前 T-1 步 no_grad, 最后一步 grad (累积 F_pred)
            # 最后一步 (或自适应 T 可能提前 break 的步骤) 才捕获 ε
            need_ε_this_step = return_ε and (is_last or (adaptive_T and _T_actual >= min_T))

            if is_last and return_pred_loss:
                _skip = (t == 0 and skip_bottom_up)
                step_kwargs = dict(
                    z_by_layer=z_by_layer, pos_emb=pos_emb, gamma=gamma_eff_ach,
                    padding_mask=padding_mask, precision_scales=precision_scales,
                    skip_bottom_up=_skip,
                )
                if need_ε_this_step:
                    z_by_layer, errors, F_val, F_pred, ε_list = self.spatiotemporal_infer_step(
                        **step_kwargs, return_errors=False, return_pred_loss=True, return_ε=True,
                    )
                else:
                    z_by_layer, errors, F_val, F_pred = self.spatiotemporal_infer_step(
                        **step_kwargs, return_errors=False, return_pred_loss=True,
                    )
                F_hist.append(F_val.item() if isinstance(F_val, torch.Tensor) else F_val)
            else:
                with torch.no_grad():
                    _skip = (t == 0 and skip_bottom_up)
                    step_kwargs = dict(
                        z_by_layer=z_by_layer, pos_emb=pos_emb, gamma=gamma_eff_ach,
                        padding_mask=padding_mask, precision_scales=precision_scales,
                        skip_bottom_up=_skip,
                    )
                    if need_ε_this_step:
                        z_by_layer, errors, F_val, _, ε_list = self.spatiotemporal_infer_step(
                            **step_kwargs, return_errors=return_errors, return_pred_loss=False, return_ε=True,
                        )
                    else:
                        z_by_layer, errors, F_val, _ = self.spatiotemporal_infer_step(
                            **step_kwargs, return_errors=return_errors, return_pred_loss=False,
                        )
                errors_hist.append([(e_sq.item(), e_norm.item()) for e_sq, e_norm in errors] if errors else [])
                F_hist.append(F_val.item() if isinstance(F_val, torch.Tensor) else F_val)

            # ── 自适应 T: F 收敛时提前终止 ──
            if adaptive_T and len(F_hist) >= 2 and _T_actual >= min_T:
                F_prev = F_hist[-2]
                F_curr = F_hist[-1]
                if isinstance(F_prev, torch.Tensor):
                    F_prev = F_prev.detach().item()
                if isinstance(F_curr, torch.Tensor):
                    F_curr = F_curr.detach().item()
                rel_change = abs(F_curr - F_prev) / (abs(F_curr) + 1e-8)
                if rel_change < convergence_threshold:
                    _converged_count += 1
                else:
                    _converged_count = 0
                if _converged_count >= patience:
                    break  # 提前终止推理循环

        # ── 释放 μ_bu 缓存 (Phase 5) ──
        if skip_bottom_up:
            self._cached_mu_bu = None

        # ── ε 已从最后一步推理中缓存, 直接返回 ──
        if return_ε:
            if not locals().get('ε_list'):
                # 安全兜底: 如果 ε 未被捕获 (极少情况), 用零误差
                ε_list = [torch.zeros_like(z.detach()[:, :, :1]) for z in z_by_layer[1:]]
            # ── 侧向抑制 ──
            if hasattr(self, 'lateral_inhibition'):
                ε_list = self.lateral_inhibition(ε_list)
            self._clear_conv_fp32_cache()
            return z_by_layer, errors_hist, F_hist, F_pred, ε_list

        self._clear_conv_fp32_cache()
        return z_by_layer, errors_hist, F_hist, F_pred

    # ── 权重更新损失 (纯预测误差) ─────────────────────────────

    def compute_spatiotemporal_loss(self, z_by_layer, pos_emb, padding_mask=None,
                                     precision_scales=None):
        """计算时空预测损失 F_pred (用于 backward 更新权重).

        F = Σ_{ℓ=1}^{L} π_ℓ · ½·‖z_ℓ(t) - μ_total(ℓ,t)‖²

        Args:
            z_by_layer: list[tensor] — 各层表示
            pos_emb: position embeddings
            padding_mask: optional padding mask
            precision_scales: list[float, L] — 多巴胺调制的精度权重 π_ℓ,
                              None 时退化为默认 (π_ℓ=1)

        z 被 detach (固定为收敛值), 梯度只通过子层/temporal/topdown 参数传播。
        """
        L = self.num_sub_layers
        z_det = [z.detach() for z in z_by_layer]
        seq_len = z_det[0].size(1)

        pred_loss = 0.0

        for ℓ in range(1, L + 1):
            π = precision_scales[ℓ - 1] if precision_scales is not None else 1.0
            z_target = z_det[ℓ]
            z_prev = z_det[ℓ - 1]

            # 自下而上 (含残差)
            μ_bu = self.predict(ℓ, z_prev, pos_emb)
            μ_bu_res = μ_bu + z_prev

            # 时序 (cat 而非 in-place)
            if seq_len > 1:
                z_t_in = z_target[:, :-1, :]
                z_t_out = self.temporal_proj[ℓ - 1](z_t_in)
                μ_temp = torch.cat([torch.zeros_like(z_target[:, :1, :]), z_t_out], dim=1)
            else:
                μ_temp = torch.zeros_like(z_target)

            # 自上而下 (cat 而非 in-place)
            if ℓ < L and seq_len > 1:
                z_d_in = z_det[ℓ + 1][:, :-1, :]
                z_d_out = self.topdown_proj[ℓ - 1](z_d_in)
                μ_down = torch.cat([torch.zeros_like(z_target[:, :1, :]), z_d_out], dim=1)
            else:
                μ_down = torch.zeros_like(z_target)

            μ_total = μ_bu_res + μ_temp + μ_down
            ε = z_target - μ_total

            if padding_mask is not None:
                mask = padding_mask.unsqueeze(-1).to(ε.dtype)
                pred_loss = pred_loss + 0.5 * π * ((ε.float() * mask) ** 2).sum()
            else:
                pred_loss = pred_loss + 0.5 * π * (ε.float() ** 2).sum()

        return pred_loss

    # ── 表示评估 (验证用) ─────────────────────────────────────

    @torch.no_grad()
    def compute_representation_metrics(self, z_by_layer):
        """计算表示质量指标。"""
        L = self.num_sub_layers
        metrics = {}

        # 表示稀疏度: ℓ1/ℓ2 ratio per layer
        sparsities = []
        for ℓ in range(1, L + 1):
            z_flat = z_by_layer[ℓ].reshape(-1, z_by_layer[ℓ].size(-1))
            l1 = z_flat.norm(p=1, dim=-1).mean().item()
            l2 = z_flat.norm(p=2, dim=-1).mean().item()
            sparsities.append(l1 / max(l2, 1e-8))
        metrics['sparsity'] = sparsities

        # 时序平滑度: ‖z(t+1) - z(t)‖²
        smoothness = []
        for ℓ in range(1, L + 1):
            diff = (z_by_layer[ℓ][:, 1:, :] - z_by_layer[ℓ][:, :-1, :]).norm(dim=-1).mean().item()
            smoothness.append(diff)
        metrics['temporal_smoothness'] = smoothness

        # 表示方差 (抗塌塌): 跨 batch 位置的 z 方差
        variances = []
        for ℓ in range(1, L + 1):
            var = z_by_layer[ℓ].var(dim=(0, 1)).mean().item()
            variances.append(var)
        metrics['variance'] = variances

        # 预测质量: 1 - ‖ε‖² / ‖z‖² (每层, ε ≈ z_ℓ - z_{ℓ-1} 一阶预测)
        pred_accs = []
        for ℓ in range(1, L + 1):
            z_norm_sq = (z_by_layer[ℓ] ** 2).sum(dim=-1).mean().item() + 1e-8
            ε = z_by_layer[ℓ] - z_by_layer[ℓ - 1]
            err_norm_sq = (ε ** 2).sum(dim=-1).mean().item()
            pred_accs.append(1.0 - err_norm_sq / z_norm_sq)
        metrics['prediction_accuracy'] = pred_accs

        return metrics

    # ── CE 计算 (通用, 从任意层 z 解码) ─────────────────────────

    def compute_ce_loss(self, z_by_layer, labels):
        """从指定的 z 计算 CE loss, 或返回 logits 用于生成。

        Args:
            z_by_layer: list[tensor], 同 init_z 输出格式
            labels: 传 None 时返回 logits (生成模式),
                    传 labels 时返回 scalar CE loss
        """
        h_top = self.model.norm(z_by_layer[self.num_sub_layers])
        if hasattr(self, '_cached_lm_head_w') and self._cached_lm_head_w is not None:
            logits = F.linear(h_top.float(), self._cached_lm_head_w)
        else:
            # 模型参数为 fp16 时 (生产环境 .half()), lm_head(h_top) 天然匹配
            logits = self.model.lm_head(h_top)
        if labels is None:
            return logits  # generation mode: [bsz, seq, vocab]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().long()
        return nn.functional.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            weight=_utf8_ce_weight(shift_logits.device).to(torch.float32),
        )

    # ── 生成 (统一方法, 子类通过继承共享) ──────────────────

    @torch.no_grad()
    def generate_with_pc(self, input_ids, max_new_tokens=256, T_infer=2, gamma=0.05,
                         temperature=0.85, top_k=20, top_p=0.0, eos_token_id=None):
        """每步先 PC 推理再解码的自回归生成。

        ponytail: 无 KV cache, 每步全量重算。T_infer 可小于训练时的 T。
        """
        device = input_ids.device
        for _ in range(max_new_tokens):
            seq_len = input_ids.size(1)
            pos_emb = self.get_position_embeddings(seq_len, device)
            z_init = self.init_z(input_ids)
            z_conv, _, _, _ = self.spatiotemporal_infer(
                z_init, pos_emb, gamma=gamma, T=T_infer)
            logits = self.compute_ce_loss(z_conv, None)  # -> [bsz, seq, vocab]
            next_logits = logits[0, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[-1]] = -float('Inf')

            probs = torch.nn.functional.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_id], dim=-1)

            if eos_token_id is not None and next_id.item() == eos_token_id:
                break

        return input_ids


class FP32SafeLinear(nn.Linear):
    """Linear 层包装: forward 时自动将权重提升到输入 dtype, 防 fp16 溢出."""
    def forward(self, x):
        w = self.weight.float()
        return F.linear(x.to(dtype=w.dtype), w, self.bias.float() if self.bias is not None else None)


# ═══════════════════════════════════════════════════════════════════
# CyrenePC — 纯局部 Conv 版
# ═══════════════════════════════════════════════════════════════════

class CyrenePC(BaseCyrenePC):
    """纯局部 Conv 版 CyrenePC — 字节连续波输入.

    用 Conv1D(k=13) 字节连续波输入 + 6 层 Dilated Conv1D(k=3, d=1,2,4,8,16,32) 替代全局 self-attention.
    无 RoPE, 无 tokenizer, 无 Embedding 表.
    """

    def __init__(self, config):
        # 绕过 BaseCyrenePC.__init__ (由子类自己初始化)
        nn.Module.__init__(self)
        self.model = CyreneBackbone(config)
        self.config = config
        self.num_sub_layers = 2 * len(self.model.layers)  # 6 层 dilated conv → 12 子层
        self._graph_capture_mode = False
        # 无 _init_rope! 无 freqs_cos/sin

        # ── 时序预测: z_ℓ(t-1) → z_ℓ(t) ──
        self.temporal_proj = nn.ModuleList([
            FP32SafeLinear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(self.num_sub_layers)
        ])

        # ── 自上而下预测: z_{ℓ+1}(t-1) → z_ℓ(t) ──
        self.topdown_proj = nn.ModuleList([
            FP32SafeLinear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(self.num_sub_layers - 1)
        ])

        # ── 自适应精度噪声: 每子层一个噪声注入器, step_interval=4 减少随机数生成 ──
        self.adaptive_noise = nn.ModuleList([
            AdaptiveNeuralNoise(config.hidden_size, base_noise=3e-5, step_interval=4)
            for _ in range(self.num_sub_layers)
        ])

        # ── Decoder 外部约束: z_L → next_byte 预测 ──
        # 用于 bp_free 训练模式 (λ·‖decoder(z_L) - next_byte‖²)
        # hybrid 模式下不使用
        self.decoder = nn.Linear(config.hidden_size, 256, bias=False)

        # ── Salience Gates (结构自组织) ──
        # 每子层一个 gate, 共 2L 个 (conv_out + mlp_out per block)
        self.salience_gates = nn.ModuleList([
            SalienceGate(config.hidden_size, temperature=0.1, init_open=True)
            for _ in range(self.num_sub_layers)
        ])

        # ── 侧向抑制 (Phase 3a) ──
        # 层间误差竞争, 锐化对比度, 加速收敛
        self.lateral_inhibition = LateralInhibition(
            num_layers=self.num_sub_layers, k=3, inhibition_strength=0.1,
        )

        # ── 依赖阈值发放 — Phase 1 ──
        # per-layer, per-channel 活性阈值, 模拟神经元膜电位阈值.
        # 静默通道自动不参与前向预测 → 自然产生稀疏性 (而非 top-k 后过滤).
        self.register_buffer('_act_threshold', torch.ones(self.num_sub_layers, config.hidden_size) * 0.3)
        self._act_ema_decay: float = 0.999         # EMA 跟踪激活分布
        self._act_target_ratio: float = 0.20       # 目标活跃率 ~20%
        self._act_homeo_rate: float = 0.02         # 稳态调节步长
        # 能量代价 (Phase 3 接入点)
        self._act_energy_cost: float = 0.0         # β·N_active — 暂时为0, Phase 3启用

        # ── fp32 权重缓存（消除前向中反复 .float() 转换）──
        # 这些缓存会在 _refresh_all_fp32_cache 中填充，训练步前调用一次
        self._cached_byte_proj_w = None       # fp32
        self._cached_lm_head_w = None         # fp32
        self._cached_local_conv_w: dict = {}  # block_idx -> fp32
        self._cached_temporal_w = None        # [L, H, H] fp32
        self._cached_topdown_w = None         # [L-1, H, H] fp32
        self._cache_built = False

        # ── μ_bu 缓存 (Phase 5: 推理融合) ──
        # init_z 中缓存的 pre-gate 预测值, 供 infer_step 跳过冗余 predict()
        self._cached_mu_bu = None

    def _refresh_all_fp32_cache(self):
        """一次性刷新所有 fp32 权重缓存。在训练步前调用，避免 .float() 重复分配。"""
        # byte_proj
        self._cached_byte_proj_w = self.model.byte_proj.weight.float()
        # lm_head
        self._cached_lm_head_w = self.model.lm_head.weight.float()
        # local_conv (按 block_idx)
        for block_idx, block in enumerate(self.model.layers):
            self._cached_local_conv_w[block_idx] = block.local_conv.weight.float()
        # temporal_proj stack
        self._cached_temporal_w = torch.stack([p.weight.float() for p in self.temporal_proj])
        # topdown_proj stack
        if self.num_sub_layers > 1:
            self._cached_topdown_w = torch.stack([p.weight.float() for p in self.topdown_proj])
        self._cache_built = True

    def _clear_all_fp32_cache(self):
        """清除所有 fp32 缓存，节省显存。"""
        self._cached_byte_proj_w = None
        self._cached_lm_head_w = None
        self._cached_local_conv_w.clear()
        self._cached_temporal_w = None
        self._cached_topdown_w = None
        self._cache_built = False

    def get_position_embeddings(self, seq_len, device, start_pos=0):
        """局部 Conv 不需要位置编码, 返回 None 占位."""
        return (None, None)

    def init_z(self, byte_seq):
        """前向传播初始化 z — 字节→连续波→dilated conv. 使用 fp32 缓存避免重复 .float()。"""
        self._refresh_all_fp32_cache()
        z = []
        x = nn.functional.pad(byte_seq.float(), (12, 0))
        h = F.conv1d(x, self._cached_byte_proj_w).transpose(1, 2)  # fp32
        z.append(h)

        mu_cache = []
        for block_idx, block in enumerate(self.model.layers):
            # Conv sub-layer (dilation-aware causal pad, fp32)
            res = h
            d = block.dilation
            h = nn.functional.pad(block.input_layernorm(h), (0, 0, 2 * d, 0))
            h32 = F.conv1d(
                h.transpose(1, 2).float(),
                self._cached_local_conv_w[block_idx],
                bias=None, stride=1, padding=0, dilation=d, groups=1,
            ).transpose(1, 2)
            mu_cache.append(h32.detach())  # ← 缓存 μ_bu (conv, pre-gate)
            h = self.salience_gates[2 * block_idx](h32)  # ← Conv gate
            h = h + res
            z.append(h)

            # MLP sub-layer
            res = h
            h = block.mlp(block.post_attention_layernorm(h))
            mu_cache.append(h.detach())  # ← 缓存 μ_bu (mlp, pre-gate)
            h = self.salience_gates[2 * block_idx + 1](h)  # ← MLP gate
            h = h + res
            z.append(h)

        self._cached_mu_bu = mu_cache
        return z  # len = 2L + 1 = 13

    def forward_with_ce(self, byte_seq, labels, pos_emb):
        """梯度启用的前向 (字节→连续波→dilated conv), 使用 fp32 缓存。"""
        self._refresh_all_fp32_cache()
        z = []
        x = nn.functional.pad(byte_seq.float(), (12, 0))
        h = F.conv1d(x, self._cached_byte_proj_w).transpose(1, 2)  # fp32
        z.append(h)

        for block_idx, block in enumerate(self.model.layers):
            # Conv sub-layer (dilation-aware causal pad, fp32)
            res = h
            d = block.dilation
            h = nn.functional.pad(block.input_layernorm(h), (0, 0, 2 * d, 0))
            h32 = F.conv1d(
                h.transpose(1, 2).float(),
                self._cached_local_conv_w[block_idx],
                bias=None, stride=1, padding=0, dilation=d, groups=1,
            ).transpose(1, 2)
            h = self.salience_gates[2 * block_idx](h32)  # ← Conv gate
            h = h + res
            z.append(h)

            # MLP sub-layer (FeedForward 内部 fp32)
            res = h
            h = block.mlp(block.post_attention_layernorm(h))
            h = self.salience_gates[2 * block_idx + 1](h)  # ← MLP gate
            h = h + res
            z.append(h)

        # ── 自适应精度噪声: 在 z 进入 CE head 前注入 ──
        if self.training and hasattr(self, 'adaptive_noise'):
            for ℓ in range(1, len(z)):
                z[ℓ] = self.adaptive_noise[ℓ - 1](z[ℓ], alpha=1.0)

        # CE from top layer (使用缓存的 lm_head fp32)
        h_top = self.model.norm(z[-1])
        logits = F.linear(h_top, self._cached_lm_head_w)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().long()
        ce_loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            weight=_utf8_ce_weight(shift_logits.device).to(torch.float32),
        )
        return z, ce_loss

    def _build_conv_fp32_cache(self):
        """预计算所有 Conv1D 的 fp32 权重缓存。训练步前调用一次，避免重复 allocate。"""
        if not self._cache_built:
            self._refresh_all_fp32_cache()

    def _clear_conv_fp32_cache(self):
        """清除 fp32 缓存。"""
        self._clear_all_fp32_cache()

    def predict(self, layer_idx, z_prev, pos_emb, fp32_out=False):
        """计算 μ_bu = sublayer_ℓ(z_prev) (不含残差), Conv 版.

        使用 _cached_local_conv_w 避免重复 .float() 转换。
        """
        block_idx = (layer_idx - 1) // 2
        is_conv_or_attn = (layer_idx - 1) % 2 == 0
        block = self.model.layers[block_idx]

        if is_conv_or_attn:
            # Conv1D (dilation-aware causal pad, fp32 安全)
            d = block.dilation
            h = nn.functional.pad(block.input_layernorm(z_prev), (0, 0, 2 * d, 0))
            conv_w = self._cached_local_conv_w.get(block_idx)
            if conv_w is None:
                conv_w = block.local_conv.weight.float()
            h32 = F.conv1d(
                h.transpose(1, 2).float(),
                conv_w,
                bias=None, stride=1, padding=0, dilation=d, groups=1,
            ).transpose(1, 2)
            return h32 if fp32_out else h32.half()
        else:
            return block.mlp(block.post_attention_layernorm(z_prev))
    # generate_with_pc 继承自 BaseCyrenePC

    # ── Salience Gate 工具 ──────────────────────────────────────

    def get_gate_sparsity_loss(self, β: float = 0.001) -> torch.Tensor:
        """所有 SalienceGate 的稀疏正则损失之和。

        L_gate = β · Σ_gate Σ_c (1 - σ(logit_c))²
        鼓励 gate 要么全开要么全关, 避免中间模糊态。
        """
        device = next(self.parameters()).device
        if not hasattr(self, 'salience_gates') or not self.salience_gates:
            return torch.tensor(0.0, device=device)
        total = torch.tensor(0.0, device=device)
        for gate in self.salience_gates:
            total = total + gate.get_sparsity_loss(β=β)
        return total

    @torch.no_grad()
    def get_gate_stats(self) -> dict:
        """返回所有 gate 的活性统计。"""
        if not hasattr(self, 'salience_gates') or not self.salience_gates:
            return {'active_ratio': 1.0, 'n_active': 0, 'n_total': 0}
        active_list = [g.get_active_ratio() for g in self.salience_gates]
        total = len(self.salience_gates) * self.config.hidden_size
        n_active = sum(self.config.hidden_size for g in self.salience_gates
                       if g.get_active_ratio() > 0.5)
        return {
            'active_ratio': sum(active_list) / len(active_list),
            'n_active': n_active,
            'n_total': total,
            'per_layer': active_list,
        }

# ── Checkpoint 工具 ────────────────────────────────────────────────

def load_cyrene_checkpoint(model, ckpt_path, device='cpu'):
    """加载 checkpoint 到 CyrenePC 模型.

    Args:
        model: CyrenePC 或 BaseCyrenePC 实例
        ckpt_path: checkpoint 文件路径
        device: 加载设备

    Returns:
        loaded_count: 成功加载的参数数量
        total_count: 模型参数总数
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_state = ckpt.get('model_state', ckpt)

    sd = model.state_dict()
    loaded = 0
    for k in model_state:
        if k in sd and model_state[k].shape == sd[k].shape:
            sd[k].copy_(model_state[k])
            loaded += 1

    return loaded, len(sd)
