"""
纯 PyTorch 骨干网络 — 无 HuggingFace 依赖。

替代 MiniMindForCausalLM, 作为 PC 层的底层骨干。
所有 building blocks (Attention, FeedForward, MiniMindBlock, RMSNorm)
从 model_minimind.py 导入, 但 PCBackbone 本身不继承任何 HF 类。

Ponytail: 最小封装, 只暴露 PC 层需要的接口。
"""
import torch
import torch.nn.functional as F
from torch import nn
from model.model_minimind import (
    MiniMindConfig, RMSNorm, precompute_freqs_cis,
    MiniMindBlock,
)


class PCBackbone(nn.Module):
    """纯 PyTorch 骨干网络, 替代 MiniMindForCausalLM.

    内部结构:
      embed_tokens → layers (MiniMindBlock × L) → norm (RMSNorm) → lm_head

    不继承 PreTrainedModel / GenerationMixin, 无 HF 运行时依赖。

    forward_with_hidden() 返回逐子层 hidden states 供 PC 层使用。
    generate() 简化版支持自回归采样。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config

        # ── 嵌入层 ──
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # ── Transformer 层 ──
        self.layers = nn.ModuleList([
            MiniMindBlock(l, config) for l in range(config.num_hidden_layers)
        ])

        # ── 输出层 ──
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # ── 权重绑定 (同 MiniMindForCausalLM) ──
        if getattr(config, 'tie_word_embeddings', True):
            self.embed_tokens.weight = self.lm_head.weight

        # ── RoPE ──
        self._init_rope()

    # ── RoPE ──────────────────────────────────────────────────────────

    def _init_rope(self):
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=self.config.head_dim,
            end=self.config.max_position_embeddings,
            rope_base=self.config.rope_theta,
            rope_scaling=self.config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def get_position_embeddings(self, seq_len, device, start_pos=0):
        return (
            self.freqs_cos[start_pos:start_pos + seq_len].to(device),
            self.freqs_sin[start_pos:start_pos + seq_len].to(device),
        )

    # ── 前向: 全量逐子层表示 ──────────────────────────────────────────

    def forward_with_hidden(self, input_ids, pos_emb, attention_mask=None):
        """前向传播, 返回每子层 hidden states.

        Args:
            input_ids: [bsz, seq_len]
            pos_emb: (cos, sin) RoPE 位置编码
            attention_mask: [bsz, seq_len] 可选

        Returns:
            logits: [bsz, seq_len, vocab_size]
            hidden_states: list[tensor, 2L+1]
              hidden_states[0] = embed_out
              hidden_states[1] = Attn₁ output
              hidden_states[2] = FFN₁ output
              ...
              hidden_states[2L-1] = Attn_L output
              hidden_states[2L]   = FFN_L output (pre-norm)
        """
        h = self.embed_tokens(input_ids)
        hidden_states = [h]  # z_0

        for block in self.layers:
            # Attention sub-layer
            res = h
            h, _ = block.self_attn(block.input_layernorm(h), pos_emb, attention_mask=attention_mask)
            h = h + res
            hidden_states.append(h)

            # FFN sub-layer
            res = h
            h = block.mlp(block.post_attention_layernorm(h))
            h = h + res
            hidden_states.append(h)

        # LM head
        h_norm = self.norm(hidden_states[-1])
        logits = self.lm_head(h_norm)

        return logits, hidden_states

    # ── 标准前向 (兼容) ──────────────────────────────────────────────

    def forward(self, input_ids, labels=None, pos_emb=None, attention_mask=None):
        """标准前向, 返回 (logits, loss).

        Args:
            input_ids: [bsz, seq_len]
            labels: [bsz, seq_len] 可选
            pos_emb: (cos, sin) 可选; 默认用 self.get_position_embeddings
            attention_mask: [bsz, seq_len] 可选

        Returns:
            logits: [bsz, seq_len, vocab_size]
            loss: scalar tensor 或 None
        """
        if pos_emb is None:
            pos_emb = self.get_position_embeddings(input_ids.size(1), input_ids.device)

        logits, _ = self.forward_with_hidden(input_ids, pos_emb, attention_mask)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    # ── 生成 ─────────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=512, temperature=0.85,
                 top_p=0.85, top_k=50, eos_token_id=2, repetition_penalty=1.0,
                 do_sample=True):
        """自回归文本生成, 无 KV cache 简化版.

        ponytail: 从 MiniMindForCausalLM.generate 移植, 无 HF GenerationMixin.
        每步全序列重算 (T=0, 纯前向), 适合 PC 评估场景。
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            seq_len = generated.size(1)
            pos_emb = self.get_position_embeddings(seq_len, device)

            logits, _ = self.forward_with_hidden(generated, pos_emb)
            next_logits = logits[:, -1, :] / temperature

            # 重复惩罚
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    seen = torch.unique(generated[i])
                    score = next_logits[i, seen]
                    next_logits[i, seen] = torch.where(
                        score > 0, score / repetition_penalty, score * repetition_penalty
                    )

            # top-k
            if top_k > 0:
                top_vals, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < top_vals[:, -1:]] = float('-inf')

            # top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumsum > top_p
                mask[:, 1:] = mask[:, :-1].clone()
                mask[:, 0] = False
                scatter_mask = torch.zeros_like(next_logits, dtype=torch.bool)
                for i in range(batch_size):
                    scatter_mask[i, sorted_indices[i]] = mask[i]
                next_logits[scatter_mask] = float('-inf')

            # 采样
            if do_sample:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            # EOS
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )

            generated = torch.cat([generated, next_token], dim=-1)

            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break

        return generated
