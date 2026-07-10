"""
纯局部 Conv 骨干网络 — 替代 PCBackbone (无 HuggingFace/Attention/RoPE)。
Ponytail: Conv1D(k=3, causal) + SwiGLU MLP, 零位置编码。
"""
import torch
import torch.nn.functional as F
from torch import nn
from model.model_minimind import MiniMindConfig, RMSNorm
from model.local_blocks import LocalConvBlock


class PCLocalBackbone(nn.Module):
    """纯局部 Conv 骨干网络, 替代 PCBackbone.

    内部结构:
      embed_tokens → layers (LocalConvBlock × L) → norm (RMSNorm) → lm_head

    与 PCBackbone 接口兼容:
      forward_with_hidden(input_ids, pos_emb=None, ...) → (logits, hidden_states[2L+1])
      forward(input_ids, labels, ...) → (logits, loss)
      generate(input_ids, ...) → generated_ids
      get_position_embeddings(...) → (None, None)  # 无 RoPE

    关键差异:
      - 无 RoPE / freqs_cos / freqs_sin
      - 无 self-attention
      - pos_emb 参数保留但不使用 (接口兼容)
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config

        # ── 字节输入投影 (替代 Embedding) ──
        # Conv1D(1→hidden, k=13, causal), 13 字节滑动窗口 ≈ 4-13 UTF-8 字符
        self.byte_proj = nn.Conv1d(1, config.hidden_size, kernel_size=13, padding=0, bias=False)

        # ── 局部 Dilated Conv 层 (L=6, d=1,2,4,8,16,32, RF=127) ──
        self.layers = nn.ModuleList([
            LocalConvBlock(l, config, dilation=d)
            for l, d in enumerate([1, 2, 4, 8, 16, 32])
        ])

        # ── 输出层 ──
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, 256, bias=False)

    def get_position_embeddings(self, seq_len, device, start_pos=0):
        """接口兼容: 局部 Conv 不需要位置编码。"""
        return (None, None)

    def forward_with_hidden(self, byte_seq, pos_emb=None, attention_mask=None):
        """前向传播, 返回每子层 hidden states.

        Args:
            byte_seq: [bsz, seq_len] uint8
            pos_emb: 忽略 (接口兼容)
            attention_mask: 忽略 (接口兼容, 因果 conv 无 mask 需求)

        Returns:
            logits: [bsz, seq_len, 256]
            hidden_states: list[tensor, 2L+1]
              hidden_states[0] = byte_proj_out
              hidden_states[1] = Conv₁ output (pre-MLP)
              hidden_states[2] = MLP₁ output (pre-next-Conv)
              ...
        """
        # 字节 → 连续波: [bsz, seq] → [bsz, 1, seq] → causal pad(12,0) → [bsz, hidden, seq] → [bsz, seq, hidden]
        x = byte_seq.float().unsqueeze(1)
        x = F.pad(x, (12, 0))  # causal pad for byte_proj k=13
        h = self.byte_proj(x).transpose(1, 2)
        hidden_states = [h]

        for block in self.layers:
            # Conv sub-layer (causal padding, dilation-aware)
            res = h
            d = block.dilation
            h = F.pad(block.input_layernorm(h), (0, 0, 2 * d, 0))
            h = block.local_conv(h.transpose(1, 2)).transpose(1, 2)
            h = h + res
            hidden_states.append(h)

            # MLP sub-layer
            res = h
            h = block.mlp(block.post_attention_layernorm(h))
            h = h + res
            hidden_states.append(h)

        # LM head
        h_norm = self.norm(hidden_states[-1])
        logits = self.lm_head(h_norm)

        return logits, hidden_states

    def forward(self, byte_seq, labels=None, pos_emb=None, attention_mask=None):
        """标准前向, 返回 (logits, loss). logits.size(-1)=256 字节级."""
        logits, _ = self.forward_with_hidden(byte_seq, pos_emb, attention_mask)

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

    @torch.inference_mode()
    def generate(self, byte_seq, max_new_tokens=512, temperature=0.85,
                 top_p=0.85, top_k=50, eos_byte=None, repetition_penalty=1.0,
                 do_sample=True):
        """逐字节自回归生成.

        Args:
            byte_seq: [bsz, seq] uint8, 初始上下文
            eos_byte: int 或 None, 遇到该字节停止

        Returns:
            generated: [bsz, seq+generated] uint8
        """
        device = byte_seq.device
        batch_size = byte_seq.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        generated = byte_seq.clone()

        for _ in range(max_new_tokens):
            logits, _ = self.forward_with_hidden(generated)
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
                next_byte = torch.multinomial(probs, 1)
            else:
                next_byte = torch.argmax(next_logits, dim=-1, keepdim=True)

            # EOS
            if eos_byte is not None:
                next_byte = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_byte, eos_byte),
                    next_byte,
                )

            generated = torch.cat([generated, next_byte.to(torch.uint8)], dim=-1)

            if eos_byte is not None:
                finished |= next_byte.squeeze(-1).eq(eos_byte)
                if finished.all():
                    break

        return generated
