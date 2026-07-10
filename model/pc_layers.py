"""
MiniMind PC 层扩展：每子层添加 predict()，管理所有 z 节点。
Ponytail: 不修改 model_minimind.py，只做包装。

将 L 个 MiniMindBlock 展开为 2L 个子层节点：
  z_0 = embed(input_ids)          # 固定 (感觉输入)
  z_1 = Attn₁(LN(z_0)) + z_0      # Block 1 Attention 输出
  z_2 = FFN₁(LN(z_1)) + z_1       # Block 1 FFN 输出
  …
  z_{2L} = FFN_L(LN(...)) + ...   # Block L FFN 输出

子层 ℓ 的预测: μ_ℓ = sublayer_ℓ(LN(z_{ℓ-1})) + z_{ℓ-1}
误差: ε_ℓ = z_ℓ - μ_ℓ
"""
import torch
from torch import nn
from model.model_minimind import precompute_freqs_cis
from model.pc_backbone import PCBackbone
from model.pc_backbone_local import PCLocalBackbone


# ── UTF-8 结构加权 CE 权重 ──
def _utf8_ce_weight(device='cpu'):
    """UTF-8 字节级加权: lead byte 权重 2.0, continuation byte 权重 0.5."""
    w = torch.ones(256, device=device)
    w[0xE4:0xEA] = 2.0   # 中文字第一字节 (0xE4-0xE9)
    w[0x80:0xC0] = 0.5   # 续字节 (0x80-0xBF)
    return w


class PCMiniMind(nn.Module):
    """MiniMind + 预测编码包装。"""

    def __init__(self, config):
        super().__init__()
        self.model = PCBackbone(config)
        self.config = config
        self.num_sub_layers = 2 * config.num_hidden_layers
        self._init_rope()

    def _init_rope(self):
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=self.config.head_dim, end=self.config.max_position_embeddings,
            rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def get_position_embeddings(self, seq_len, device, start_pos=0):
        return (
            self.freqs_cos[start_pos:start_pos + seq_len].to(device),
            self.freqs_sin[start_pos:start_pos + seq_len].to(device),
        )

    # ── 前向初始化 ─────────────────────────────────────────────

    @torch.no_grad()
    def init_z(self, input_ids):
        """前向传播初始化所有 z 节点。"""
        bsz, seq_len = input_ids.shape
        pos = self.get_position_embeddings(seq_len, input_ids.device)
        z = []

        h = self.model.embed_tokens(input_ids)
        z.append(h)  # z_0, 固定

        for block in self.model.layers:
            res = h
            h = block.self_attn(block.input_layernorm(h), pos)[0]
            h = h + res
            z.append(h)

            res = h
            h = block.mlp(block.post_attention_layernorm(h))
            h = h + res
            z.append(h)

        return z  # len = 2L + 1

    # ── 单子层预测 ────────────────────────────────────────────

    def predict(self, layer_idx, z_prev, pos_emb):
        """计算 μ = sublayer_ℓ(z_prev) (不含残差)。

        layer_idx ∈ [1..2L], 1=Attn₁, 2=FFN₁, 3=Attn₂, …
        """
        block_idx = (layer_idx - 1) // 2
        is_attn = (layer_idx - 1) % 2 == 0
        block = self.model.layers[block_idx]

        if is_attn:
            return block.self_attn(block.input_layernorm(z_prev), pos_emb)[0]
        else:
            return block.mlp(block.post_attention_layernorm(z_prev))

    # ── 单步推理 ──────────────────────────────────────────────

    def infer_step(self, z, pos_emb, gamma, labels=None):
        """单步 PC 推理：自由能梯度下降更新 z 节点。

        ∇F_zℓ = ε_ℓ - Jᵀ_{ℓ+1} ε_{ℓ+1}
        ∇F_zL = ε_L + ∂CE/∂z_L  (顶层含输出误差)
        z_ℓ ← z_ℓ - γ · ∇F_zℓ

        返回 (new_z, errors_info)
          errors_info: [(e_sq, e_norm), ...] ℓ=1..L
        """
        L = self.num_sub_layers
        z_det = [zi.detach().requires_grad_(True) for zi in z]

        # 计算所有 μ_ℓ (含残差)
        μ_res = [None]
        for ℓ in range(1, L + 1):
            μ = self.predict(ℓ, z_det[ℓ - 1], pos_emb)
            μ_res.append(μ + z_det[ℓ - 1])

        # ── 顶层输出梯度 ∂CE/∂z_L ──
        ce_grad = 0
        if labels is not None:
            h_top = self.model.norm(z_det[L])
            logits = self.model.lm_head(h_top)
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            ce_loss = nn.functional.cross_entropy(
                x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100,
            )
            ce_grad, = torch.autograd.grad(ce_loss, z_det[L])

        # ── 合并计算所有 Jᵀε (一次 backward 替代 7 次) ──
        # Jᵀ_{ℓ+1}ε_{ℓ+1} = ε_{ℓ+1} · ∂μ_res[ℓ+1]/∂z_ℓ
        jt_loss = 0.0
        for ℓ in range(1, L):
            ε_up = (z_det[ℓ + 1] - μ_res[ℓ + 1]).detach()
            jt_loss = jt_loss + (ε_up * μ_res[ℓ + 1]).sum()
        jt_grads = torch.autograd.grad(jt_loss, [z_det[ℓ] for ℓ in range(1, L)])
        # jt_grads[ℓ-1] = Jᵀ_{ℓ+1}ε_{ℓ+1}

        # ── 更新所有 z_ℓ ──
        new_z = [z[0]]
        errors_info = []

        for ℓ in range(1, L + 1):
            ε = z_det[ℓ] - μ_res[ℓ]
            e_sq = (ε.detach() ** 2).mean().item()
            e_norm = ε.detach().norm().item()
            errors_info.append((e_sq, e_norm))

            if ℓ < L:
                grad_F = ε - jt_grads[ℓ - 1]
            else:
                grad_F = ε + ce_grad  # 顶层: 预测误差 + 输出误差

            new_z.append(z[ℓ] - gamma * grad_F.detach())

        return new_z, errors_info

    # ── 权重更新 ──────────────────────────────────────────────

    def compute_pc_loss(self, z, pos_emb, labels, input_ids=None):
        """计算总 PC 能量 F (用于 backward)。

        F = CE(model.forward(input_ids), labels) + Σ_{ℓ=1}^{L} ½·‖z_ℓ - μ_ℓ(z_{ℓ-1})‖²

        CE 通过完整前向传播 (梯度流经所有层)，预测误差约束表示接近收敛的 z 值。
        z 被 detach (固定为收敛值)，梯度只通过子层参数传播。
        """
        L = self.num_sub_layers
        z_det = [zi.detach() for zi in z]

        # 预测误差: 约束权重使子层预测接近收敛后的 z
        pred_energy = 0.0
        for ℓ in range(1, L + 1):
            z_target = z_det[ℓ]
            z_prev = z_det[ℓ - 1]
            μ = self.predict(ℓ, z_prev, pos_emb)
            μ_res = μ + z_prev
            pred_energy = pred_energy + 0.5 * ((z_target - μ_res) ** 2).sum()

        # 输出能量: CE 通过完整前向传播 (所有层都收 CE 梯度)
        if input_ids is not None:
            _, ce_loss = self.model(input_ids=input_ids, labels=labels)
        else:
            # fallback: 只用 z_L 算 CE (旧路径)
            h = self.model.norm(z_det[L])
            logits = self.model.lm_head(h)
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            ce_loss = nn.functional.cross_entropy(
                x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100,
            )

        total_energy = pred_energy + ce_loss
        return total_energy

    # ── 输出/验证 ─────────────────────────────────────────────

    @torch.no_grad()
    def compute_loss(self, z, labels):
        """计算 CE loss (验证用)。"""
        h = self.model.norm(z[self.num_sub_layers])
        logits = self.model.lm_head(h)
        x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
        return nn.functional.cross_entropy(
            x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100,
        ).item()


# ═══════════════════════════════════════════════════════════════════════════
# PCDynamicMiniMind — 时空预测编码
# 每层节点做三维局部预测: 自下而上 + 时序 + 自上而下
# F = Σ ½·‖ε_ℓ(t)‖²  (纯预测误差, 无 CE, 无 token loss)
# ═══════════════════════════════════════════════════════════════════════════

class PCDynamicMiniMind(nn.Module):
    """时空预测编码 MiniMind。

    核心转变: "预测下一个 token" → "预测下一个神经状态"
    每层 z_ℓ(t) 收三路预测后合并为 μ_total, 误差 ε = z - μ_total 驱动学习。

    时间维度: z_by_layer[ℓ] = [bsz, seq_len, hidden_size], ℓ=0..L
    - ℓ=0: embed (固定, 感觉输入)
    - ℓ=1..L: 子层输出 (变量, 推理时更新)
    """

    def __init__(self, config):
        super().__init__()
        self.model = PCBackbone(config)
        self.config = config
        self.num_sub_layers = 2 * config.num_hidden_layers
        self._init_rope()

        # ── 时序预测: z_ℓ(t-1) → z_ℓ(t) ──
        self.temporal_proj = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(self.num_sub_layers)
        ])  # index ℓ-1 → sub-layer ℓ, ℓ=1..L

        # ── 自上而下预测: z_{ℓ+1}(t-1) → z_ℓ(t) ──
        self.topdown_proj = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(self.num_sub_layers - 1)
        ])  # index ℓ-1 → sub-layer ℓ from ℓ+1, ℓ=1..L-1

    def _init_rope(self):
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=self.config.head_dim, end=self.config.max_position_embeddings,
            rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def get_position_embeddings(self, seq_len, device, start_pos=0):
        return (
            self.freqs_cos[start_pos:start_pos + seq_len].to(device),
            self.freqs_sin[start_pos:start_pos + seq_len].to(device),
        )

    # ── 初始化 (复用 PCMiniMind.init_z) ───────────────────────

    @torch.no_grad()
    def init_z_seq(self, input_ids):
        """前向初始化所有 z 节点, 返回按层组织的列表。

        Returns: z_by_layer[ℓ] = [bsz, seq_len, hidden_size], ℓ=0..L
        """
        return self.init_z(input_ids)

    def init_z(self, input_ids):
        """前向传播初始化所有 z 节点。"""
        bsz, seq_len = input_ids.shape
        pos = self.get_position_embeddings(seq_len, input_ids.device)
        z = []

        h = self.model.embed_tokens(input_ids)
        z.append(h)  # z_0, 固定

        for block in self.model.layers:
            res = h
            h = block.self_attn(block.input_layernorm(h), pos)[0]
            h = h + res
            z.append(h)

            res = h
            h = block.mlp(block.post_attention_layernorm(h))
            h = h + res
            z.append(h)

        return z  # len = 2L + 1

    # ── 带梯度的前向 + CE (用于混合训练) ────────────────────

    def forward_with_ce(self, input_ids, labels, pos_emb):
        """梯度启用的前向: 返回 z_by_layer + CE loss.

        Returns:
            z_init: list[tensor, L+1], 每层表示 (有梯度)
            ce_loss: scalar tensor, 交叉熵损失 (梯度流遍 backbone + lm_head)
        """
        z = []
        h = self.model.embed_tokens(input_ids)
        z.append(h)

        for block in self.model.layers:
            res = h
            h = block.self_attn(block.input_layernorm(h), pos_emb)[0]
            h = h + res
            z.append(h)

            res = h
            h = block.mlp(block.post_attention_layernorm(h))
            h = h + res
            z.append(h)

        # CE from top layer
        h_top = self.model.norm(z[-1])
        logits = self.model.lm_head(h_top)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return z, ce_loss

    # ── 单子层预测 (复用 PCMiniMind.predict) ──────────────────

    def predict(self, layer_idx, z_prev, pos_emb):
        """计算 μ_bu = sublayer_ℓ(z_prev) (不含残差)."""
        block_idx = (layer_idx - 1) // 2
        is_attn = (layer_idx - 1) % 2 == 0
        block = self.model.layers[block_idx]

        if is_attn:
            return block.self_attn(block.input_layernorm(z_prev), pos_emb)[0]
        else:
            return block.mlp(block.post_attention_layernorm(z_prev))

    # ── 时空推理单步 ──────────────────────────────────────────

    def spatiotemporal_infer_step(self, z_by_layer, pos_emb, gamma, padding_mask=None,
                                    return_errors=True, return_pred_loss=False,
                                    precision_scales=None):
        """单步时空推理: 对所有层 ell=1..L 和时间 t 更新 z。

        注意: 无论调用方是否在 torch.no_grad() 下, 内部始终启用 grad。
        """
        with torch.enable_grad():
            return self._spatiotemporal_infer_step(z_by_layer, pos_emb, gamma, padding_mask,
                                                    return_errors=return_errors,
                                                    return_pred_loss=return_pred_loss,
                                                    precision_scales=precision_scales)

    def _spatiotemporal_infer_step(self, z_by_layer, pos_emb, gamma, padding_mask,
                                     return_errors=True, return_pred_loss=False,
                                     precision_scales=None):
        """grad-enabled 实现 (被包装调用)."""
        L = self.num_sub_layers
        z_det = [z.detach().requires_grad_(True) for z in z_by_layer]
        seq_len = z_det[0].size(1)


        # ── 计算所有预测和误差 ──
        F = 0.0
        # ponytail: 缓存 μ 值, 消除 error logging 中重复 predict (Phase A)
        _μ_bu_cache, _μ_temp_cache, _μ_down_cache = [], [], []

        for ℓ in range(1, L + 1):
            # 自下而上: sublayer(z_{ℓ-1}) + residual
            μ_bu = self.predict(ℓ, z_det[ℓ - 1], pos_emb)
            μ_bu_res = μ_bu + z_det[ℓ - 1]

            # 时序: W_temp(z_ℓ(t-1)) → z_ℓ(t), t=0 为零
            # ponytail: cat 而非 in-place 赋值, 保持 autograd 图完整
            if seq_len > 1:
                z_prev_t = z_det[ℓ][:, :-1, :]
                z_temp = self.temporal_proj[ℓ - 1](z_prev_t)
                μ_temp = torch.cat([torch.zeros_like(z_det[ℓ][:, :1, :]), z_temp], dim=1)
            else:
                μ_temp = torch.zeros_like(z_det[ℓ])

            # 自上而下: W_down(z_{ℓ+1}(t-1)) → z_ℓ(t), t=0 为零
            if ℓ < L and seq_len > 1:
                z_down_prev = z_det[ℓ + 1][:, :-1, :]
                z_down = self.topdown_proj[ℓ - 1](z_down_prev)
                μ_down = torch.cat([torch.zeros_like(z_det[ℓ][:, :1, :]), z_down], dim=1)
            else:
                μ_down = torch.zeros_like(z_det[ℓ])

            # ponytail: 存入缓存
            _μ_bu_cache.append(μ_bu_res)
            _μ_temp_cache.append(μ_temp)
            _μ_down_cache.append(μ_down)

            μ_total = μ_bu_res + μ_temp + μ_down
            ε = z_det[ℓ] - μ_total

            # 带 padding mask 的误差 (可选)
            if padding_mask is not None:
                mask = padding_mask.unsqueeze(-1).to(ε.dtype)
                F = F + 0.5 * ((ε * mask) ** 2).sum()
            else:
                F = F + 0.5 * (ε ** 2).sum()

        # ── 一次 backward 计算所有 ∇F_{z_ℓ} ──
        z_vars = [z_det[ℓ] for ℓ in range(1, L + 1)]
        grads = torch.autograd.grad(F, z_vars)

        # ── 更新 z_ℓ ──
        new_z = [z_by_layer[0]]  # z_0 固定
        errors_info = []

        for ℓ in range(1, L + 1):
            new_z.append(z_by_layer[ℓ] - gamma * grads[ℓ - 1].detach())
            if return_errors:
                # ponytail: 从缓存读取 μ 值 (Phase A), 消除 36 次 predict / proj 调用
                idx = ℓ - 1
                ε = z_det[ℓ] - (_μ_bu_cache[idx] + _μ_temp_cache[idx] + _μ_down_cache[idx])
                e_sq = (ε.detach() ** 2).mean().item()
                e_norm = ε.detach().norm().item()
                errors_info.append((e_sq, e_norm))


        # ── 可选: 用更新后的 z (detach) 计算 weight loss ──
        # ponytail: 消除独立 compute_spatiotemporal_loss 的冗余 forward
        weight_loss = None
        if return_pred_loss:
            weight_loss = 0.0
            z_new_det = [z.detach() for z in new_z]
            for ℓ in range(1, L + 1):
                π = precision_scales[ℓ - 1] if precision_scales is not None else 1.0
                z_target = z_new_det[ℓ]
                z_prev = z_new_det[ℓ - 1]

                μ_bu = self.predict(ℓ, z_prev, pos_emb)
                μ_bu_res = μ_bu + z_prev

                if seq_len > 1:
                    z_prev_t = z_target[:, :-1, :]
                    z_temp = self.temporal_proj[ℓ - 1](z_prev_t)
                    μ_temp = torch.cat([torch.zeros_like(z_target[:, :1, :]), z_temp], dim=1)
                else:
                    μ_temp = torch.zeros_like(z_target)

                if ℓ < L and seq_len > 1:
                    z_down_prev = z_new_det[ℓ + 1][:, :-1, :]
                    z_down = self.topdown_proj[ℓ - 1](z_down_prev)
                    μ_down = torch.cat([torch.zeros_like(z_target[:, :1, :]), z_down], dim=1)
                else:
                    μ_down = torch.zeros_like(z_target)

                μ_total = μ_bu_res + μ_temp + μ_down
                ε = z_target - μ_total

                if padding_mask is not None:
                    mask = padding_mask.unsqueeze(-1).to(ε.dtype)
                    weight_loss = weight_loss + 0.5 * π * ((ε * mask) ** 2).sum()
                else:
                    weight_loss = weight_loss + 0.5 * π * (ε ** 2).sum()

        F_val = F.item()

        return new_z, errors_info, F_val, weight_loss

    # ── 时空推理循环 ──────────────────────────────────────────

    @torch.no_grad()
    def spatiotemporal_infer(self, z_by_layer, pos_emb, gamma=0.1, T=2, padding_mask=None,
                               return_errors=True, return_pred_loss=False,
                               precision_scales=None):
        """T 步时空推理循环, 返回收敛后的 z 和历史."""
        errors_hist = []
        F_hist = []
        F_pred = None

        for t in range(T):
            do_pred = return_pred_loss and (t == T - 1)
            z_by_layer, errors, F, Fp = self.spatiotemporal_infer_step(
                z_by_layer, pos_emb, gamma, padding_mask,
                return_errors=return_errors,
                return_pred_loss=do_pred,
                precision_scales=precision_scales,
            )
            errors_hist.append(errors)
            F_hist.append(F)
            if do_pred:
                F_pred = Fp

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
                pred_loss = pred_loss + 0.5 * π * ((ε * mask) ** 2).sum()
            else:
                pred_loss = pred_loss + 0.5 * π * (ε ** 2).sum()

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
        logits = self.model.lm_head(h_top)
        if labels is None:
            return logits  # generation mode: [bsz, seq, vocab]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().long()
        return nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            weight=_utf8_ce_weight(shift_logits.device),
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


# ═══════════════════════════════════════════════════════════════════
# PCLocalDynamicMiniMind — 纯局部 Conv 版
# ═══════════════════════════════════════════════════════════════════

class PCLocalDynamicMiniMind(PCDynamicMiniMind):
    """纯局部 Conv 版 PCDynamicMiniMind — 去离散化字节输入.

    用 Conv1D(k=13) 字节连续波输入 + 6 层 Dilated Conv1D(k=3, d=1,2,4,8,16,32) 替代全局 self-attention.
    无 RoPE, 无 tokenizer, 无 Embedding 表.
    """

    def __init__(self, config):
        # 绕过 PCDynamicMiniMind.__init__ (它创建 PCBackbone + RoPE)
        nn.Module.__init__(self)
        self.model = PCLocalBackbone(config)
        self.config = config
        self.num_sub_layers = 2 * len(self.model.layers)  # 6 层 dilated conv → 12 子层
        # 无 _init_rope! 无 freqs_cos/sin

        # ── 时序预测: z_ℓ(t-1) → z_ℓ(t) ──
        self.temporal_proj = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(self.num_sub_layers)
        ])

        # ── 自上而下预测: z_{ℓ+1}(t-1) → z_ℓ(t) ──
        self.topdown_proj = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(self.num_sub_layers - 1)
        ])

    def get_position_embeddings(self, seq_len, device, start_pos=0):
        """局部 Conv 不需要位置编码, 返回 None 占位."""
        return (None, None)

    def init_z(self, byte_seq):
        """前向传播初始化 z — 字节→连续波→dilated conv."""
        z = []
        # 字节 → 连续波: [bsz, seq] → [bsz, 1, seq] → causal pad(12,0) → [bsz, hidden, seq] → [bsz, seq, hidden]
        x = byte_seq.float().unsqueeze(1)
        x = nn.functional.pad(x, (12, 0))
        h = self.model.byte_proj(x).transpose(1, 2)
        z.append(h)

        for block in self.model.layers:
            # Conv sub-layer (dilation-aware causal pad)
            res = h
            d = block.dilation
            h = nn.functional.pad(block.input_layernorm(h), (0, 0, 2 * d, 0))
            h = block.local_conv(h.transpose(1, 2)).transpose(1, 2)
            h = h + res
            z.append(h)

            # MLP sub-layer
            res = h
            h = block.mlp(block.post_attention_layernorm(h))
            h = h + res
            z.append(h)

        return z  # len = 2L + 1 = 13

    def forward_with_ce(self, byte_seq, labels, pos_emb):
        """梯度启用的前向 (字节→连续波→dilated conv), 返回 z + CE loss."""
        z = []
        x = byte_seq.float().unsqueeze(1)
        x = nn.functional.pad(x, (12, 0))
        h = self.model.byte_proj(x).transpose(1, 2)
        z.append(h)

        for block in self.model.layers:
            # Conv sub-layer (dilation-aware causal pad)
            res = h
            d = block.dilation
            h = nn.functional.pad(block.input_layernorm(h), (0, 0, 2 * d, 0))
            h = block.local_conv(h.transpose(1, 2)).transpose(1, 2)
            h = h + res
            z.append(h)

            # MLP sub-layer
            res = h
            h = block.mlp(block.post_attention_layernorm(h))
            h = h + res
            z.append(h)

        # CE from top layer
        h_top = self.model.norm(z[-1])
        logits = self.model.lm_head(h_top)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().long()
        ce_loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            weight=_utf8_ce_weight(shift_logits.device),
        )
        return z, ce_loss

    def predict(self, layer_idx, z_prev, pos_emb):
        """计算 μ_bu = sublayer_ℓ(z_prev) (不含残差), Conv 版."""
        block_idx = (layer_idx - 1) // 2
        is_conv_or_attn = (layer_idx - 1) % 2 == 0
        block = self.model.layers[block_idx]

        if is_conv_or_attn:
            # Conv1D (dilation-aware causal pad)
            d = block.dilation
            h = nn.functional.pad(block.input_layernorm(z_prev), (0, 0, 2 * d, 0))
            return block.local_conv(h.transpose(1, 2)).transpose(1, 2)
        else:
            return block.mlp(block.post_attention_layernorm(z_prev))
    # ponytail: generate_with_pc 继承自 PCDynamicMiniMind

# ── Checkpoint 工具 ────────────────────────────────────────────────

def load_pc_checkpoint(model, ckpt_path, device='cpu'):
    """加载旧 checkpoint（含 model.model. 前缀）到 PCBackbone 模型。

    旧 checkpoint 的 model_state 中 backbone 权重前缀为 model.model.xxx，
    新 PCBackbone 使用 model.xxx。本函数自动做前缀映射。

    Args:
        model: PCMiniMind 或 PCDynamicMiniMind 实例
        ckpt_path: checkpoint 文件路径
        device: 加载设备

    Returns:
        loaded_count: 成功加载的参数数量
        total_count: 模型参数总数
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_state = ckpt.get('model_state', ckpt)

    sd = model.state_dict()
    mapped = {}
    for k in model_state:
        # model.model.xxx → model.xxx
        new_k = k.replace('model.model.', 'model.', 1) if k.startswith('model.model.') else k
        mapped[new_k] = k

    loaded = 0
    for new_k, old_k in mapped.items():
        if new_k in sd and model_state[old_k].shape == sd[new_k].shape:
            sd[new_k].copy_(model_state[old_k])
            loaded += 1

    return loaded, len(sd)
