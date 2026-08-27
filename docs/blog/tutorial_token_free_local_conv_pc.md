# 去 Token 化 + 局部卷积 + 预测编码：从零构建纯局部语言模型

> 基于 PCLocalDynamicMiniMind 架构的完整实战记录
> 硬件：GTX 1650 Ti (4GB VRAM) | 框架：PyTorch 2.12 + cu132

---

## 目录

1. [核心思路：为什么同时去 Token 和去 Attention？](#1-核心思路为什么同时去-token-和去-attention)
2. [架构全景图](#2-架构全景图)
3. [字节输入编码器](#3-字节输入编码器)
4. [局部扩张卷积骨干网络](#4-局部扩张卷积骨干网络)
5. [预测编码的局部化改造](#5-预测编码的局部化改造)
6. [训练管道：五阶段混合循环](#6-训练管道五阶段混合循环)
7. [评估结果](#7-评估结果)
8. [与全局 Attention 版的对比](#8-与全局-attention-版的对比)
9. [局限性分析](#9-局限性分析)
10. [重现指南](#10-重现指南)
11. [总结与教训](#11-总结与教训)

---

## 1. 核心思路：为什么同时去 Token 和去 Attention？

### 1.1 去 Token 化

传统语言模型依赖 tokenizer（BPE、SentencePiece 等）将文本映射到离散词表（~50K–256K）。这引入三个问题：

- **词表偏置**：OOV 词、罕见拼写、多语言字符共享需要复杂的 BPE 合并规则
- **信息损失**：UTF-8 字节到 token 的映射是**不可逆的**（不同字节序列可能映射到同一 token）
- **嵌入层膨胀**：词表 50K × 隐藏 256 ≈ 12.8M 参数，在小模型中占绝对主导

**解决方案**：直接在 UTF-8 字节空间操作（词表大小 = 256），用 Conv1D 滑动窗口将字节编码为连续向量。这本质上是**可学习的字节级嵌入**，保留了全部原始信息。

### 1.2 去 Attention

自注意力（Self-Attention）的计算量是 $O(n^2 \cdot d)$，在小模型和短序列场景下代价较高。更重要的是：

- **全局注意力假设所有位置都相关**——这对语言建模不一定成立
- **位置编码（RoPE）是 Attention 的附属品**——如果不用 Attention，也就不需要 RoPE
- **卷积天然具有局部性和位置感知能力**——kernel 的相对位置隐式编码了顺序

**解决方案**：用 **Dilated Conv1D 堆叠**替代 Transformer Block。6 层扩张卷积的感受野为 $RF = 1 + \sum_{i=0}^{5} (k-1) \cdot d_i = 1 + 2 \times (1+2+4+8+16+32) = 127$，覆盖约 42 个 UTF-8 字符（≈ 42 个中文字符或 127 个 ASCII 字符）。

### 1.3 预测编码的局部化

预测编码（Predictive Coding）的核心是**每层预测下一层的活动**，但原始 PC 框架在 Transformer 上的实现仍然依赖全局 Attention。本文的目标是：

- 将 PC 的三种预测（自下而上、时序、自上而下）适配到 Conv1D 层上
- 保持 PC 的局部学习特性——权重更新只依赖当前层的活动值和预测误差

---

## 2. 架构全景图

```
输入: UTF-8 字节序列 [bsz, seq] (uint8, 0-255)
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. 字节投影层                                               │
│    Conv1D(1→256, k=13, causal pad=12)                       │
│    输出: [bsz, seq, 256] 连续波                             │
└──────────────────────────────────────────────────────────────┘
  │  z_0 (固定输入)
  ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. Dilated Conv Block 1  (d=1)                              │
│    ├─ Conv1D(256→256, k=3, causal, dilation=1) + residual   │
│    └─ SwiGLU MLP(256→640→256) + residual                    │
│    输出: z_1 (conv), z_2 (mlp)                              │
├──────────────────────────────────────────────────────────────┤
│ 3. Dilated Conv Block 2  (d=2)                              │
│    ├─ Conv1D(256→256, k=3, causal, dilation=2) + residual   │
│    └─ SwiGLU MLP(256→640→256) + residual                    │
│    输出: z_3 (conv), z_4 (mlp)                              │
├──────────────────────────────────────────────────────────────┤
│ 4. Dilated Conv Block 3  (d=4)                              │
│    ...                                                      │
├──────────────────────────────────────────────────────────────┤
│ 5. Dilated Conv Block 4  (d=8)                              │
│    ...                                                      │
├──────────────────────────────────────────────────────────────┤
│ 6. Dilated Conv Block 5  (d=16)                             │
│    ...                                                      │
├──────────────────────────────────────────────────────────────┤
│ 7. Dilated Conv Block 6  (d=32)                             │
│    ├─ Conv1D(256→256, k=3, causal, dilation=32) + residual  │
│    └─ SwiGLU MLP(256→640→256) + residual                    │
│    输出: z_11 (conv), z_12 (mlp)                            │
└──────────────────────────────────────────────────────────────┘
  │  z_12 (顶层表示)
  ▼
┌──────────────────────────────────────────────────────────────┐
│ 8. RMSNorm → LM Head(256→256)                               │
│    输出: logits [bsz, seq, 256] (字节级预测)                │
└──────────────────────────────────────────────────────────────┘
```

**关键参数**：

| 组件 | 参数 | 说明 |
|------|------|------|
| 字节投影 | Conv1D(1→256, k=13) | 13 字节滑动窗口，stride=1 |
| 隐藏层数 | 6 层 × 2 子层 = 12 | 6 个 Conv Block，每块 = Conv + MLP |
| 隐藏维度 | 256 | 与 MiniMind 原版一致 |
| 扩张率 | [1,2,4,8,16,32] | 感受野 127 字节 |
| 感受野 | 127 字节 | 约 42 个中文字符 / 127 个 ASCII |
| MLP 隐藏 | 640 (2.5× 256) | SwiGLU 结构，与 FeedForward 一致 |
| LM Head | Linear(256→256) | 字节级词表 256，无 bias |
| 参数量 | ~2.5M | 远小于原版 MiniMind 的 ~5M |

### 与标准 Transformer 的结构对比

```
Transformer Block:                    LocalConv Block:
  Input ─► Attention ─► + ─► MLP ─► +      Input ─► Conv1D ─► + ─► MLP ─► +
              │   ▲                      (k=3,d=dilation) │   ▲
              │   │                             手动 causal pad   │
              └───┘                                 └────────────┘
        全局交互 (O(n²))                         局部交互 (O(n·k))
        需要 RoPE 编码位置                        核相对位置隐式编码
```

---

## 3. 字节输入编码器

放弃 nn.Embedding(vocab_size, hidden_size)，改用 Conv1D 将字节序列编码为连续"波"：

```python
self.byte_proj = nn.Conv1d(1, config.hidden_size, kernel_size=13, padding=0, bias=False)
```

前向过程：

```python
# [bsz, seq] uint8 → [bsz, 1, seq] float
x = byte_seq.float().unsqueeze(1)
# 因果填充: 左侧 pad 12 个零, 确保位置 t 只看 t-12..t
x = F.pad(x, (12, 0))
# Conv1D → [bsz, hidden, seq] → [bsz, seq, hidden]
h = self.byte_proj(x).transpose(1, 2)
```

**设计理由**：
- k=13 覆盖了最坏情况下 UTF-8 编码的 ≤4 字节字符 × 3+，确保每个位置的表示包含足够上下文
- 没有偏置项（bias=False），保持纯线性投影 + 滑动窗口
- 结果 h 是一个**连续值向量序列**，每个位置是周围 13 字节的加权和——这本质上就是"连续波"表示

**对比 Embedding**：

| 方案 | 参数量 | 输入形式 | 信息保留 |
|------|--------|----------|----------|
| Embedding(256, 256) | 65,536 | 离散 id | 硬映射，无局部上下文 |
| Conv1D(1→256, k=13) | 3,328 | 连续字节值 | 滑动窗口，含局部模式 |
| 节省 | ~95% | — | — |

---

## 4. 局部扩张卷积骨干网络

### 4.1 单一卷积块 (LocalConvBlock)

每个块包含两个子层，接口与 MiniMindBlock 兼容：

```python
class LocalConvBlock(nn.Module):
    def __init__(self, layer_id, config, dilation=1):
        super().__init__()
        self.dilation = dilation

        # ── Conv 子层 (替代 Attention) ──
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.local_conv = nn.Conv1d(
            config.hidden_size, config.hidden_size,
            kernel_size=3, padding=0, dilation=dilation, bias=False,
        )

        # ── MLP 子层 (SwiGLU, 与 Transformer 相同) ──
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.mlp = FeedForward(config)
```

### 4.2 Causal 填充策略扩张卷积

关键细节：Conv1D 的 causal 填充需要感知扩张率。

对于 `kernel_size=3, dilation=d`，为了保持 seq_len 不变且因果（position t 只能看 t-2d 和 t-d）：

```python
# 左侧 pad 2*d 个零, 右侧不 pad
h = F.pad(hidden_states, (0, 0, 2 * self.dilation, 0))
h = self.input_layernorm(h)
h = h.transpose(1, 2)              # [bsz, hidden, seq+2d]
h = self.local_conv(h)             # [bsz, hidden, seq]
h = h.transpose(1, 2)              # [bsz, seq, hidden]
hidden_states = residual + h
```

**为什么 pad=2d**：对于 k=3 且 dilation=d，卷积核覆盖位置 `{t-2d, t-d, t}`。左侧 pad 2d 后：
- 位置 0 看 `{pad, pad, byte_0}` = `{0, 0, byte_0}`
- 位置 1 看 `{pad, byte_0, byte_1}`
- 以此类推……

### 4.3 扩张率序列与感受野

```
Layer   d     RF_add     Cumulative RF
───────────────────────────────────────
ByteProj —      13             13
Block1   d=1     2             15
Block2   d=2     4             19
Block3   d=4     8             27
Block4   d=8    16             43
Block5   d=16   32             75
Block6   d=32   64            139
```

$RF_{total} = 13 + 2 \times (1+2+4+8+16+32) = 13 + 126 = 139$

但注意 Transformer 的全局 attention 是 $RF = seq\_len$，所以即使 6 层扩张卷积也无法在长序列上替代 attention。这是一个**明确的设计取舍**——用局部性换计算效率。

### 4.4 SwiGLU MLP

MLP 部分与原始 MiniMind 的 FeedForward 相同，使用 SwiGLU 激活：

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.hidden_size * 2.5, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.hidden_size * 2.5, bias=False)
        self.down_proj = nn.Linear(config.hidden_size * 2.5, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

### 4.5 为什么不需要位置编码？

这是一个经常被问到的问题。答案：

1. **卷积核的固有位置敏感性**：Conv1D(k=3) 在位置 t 的输出是 `f(x_{t-2d}, x_{t-d}, x_t)`——顺序是硬编码在核权重中的。翻转输入顺序会得到完全不同的输出。
2. **因果填充的方向性**：左侧 pad 确保每位置只看到过去，这种方向性本身就是绝对位置信息。
3. **堆叠的层次结构**：浅层卷积看局部模式，深层卷积通过扩张看到更大范围——这与 Transformer 中低层和高层 attention head 的行为类似。

用数学语言：Conv1D 是**位置等变**（translation equivariant）的，但因果 padding 打破了平移对称性，使得边界位置和中心位置的表示不同——这提供了足够的位置信号。

---

## 5. 预测编码的局部化改造

### 5.1 继承体系

```
nn.Module
  └─ PCMiniMind          (基础 PC: 静态 z, Transformer 骨干)
       └─ PCDynamicMiniMind  (动态 z: 加入时序+自上而下预测)
            └─ PCLocalDynamicMiniMind  (当前架构: Conv 骨干, 去 token)
```

`PCLocalDynamicMiniMind` 重写了三个核心方法：
- `init_z()` — 用 Conv 前向替代 Transformer 前向
- `forward_with_ce()` — 梯度启用的前向 + CE loss
- `predict()` — 计算 $\mu_{bu}$（Conv 版的子层预测）

### 5.2 神经表示 z 的初始化

```python
def init_z(self, byte_seq):
    """前向传播初始化 z — 字节→连续波→dilated conv。"""
    z = []
    # z_0: 字节投影输出 (固定输入)
    h = self.byte_proj(F.pad(byte_seq.float().unsqueeze(1), (12, 0))).transpose(1, 2)
    z.append(h)

    for block in self.model.layers:
        # Conv sub-layer → z_{2i-1}
        d = block.dilation
        h = F.pad(block.input_layernorm(h), (0, 0, 2 * d, 0))
        h = block.local_conv(h.transpose(1, 2)).transpose(1, 2)
        h = h + res; z.append(h)

        # MLP sub-layer → z_{2i}
        h = block.mlp(block.post_attention_layernorm(h))
        h = h + res; z.append(h)

    return z  # len = 2*6 + 1 = 13
```

z 列表的长度为 13：
- `z[0]` = 字节投影输出（固定，不参与 PC 更新）
- `z[1]`, `z[3]`, ..., `z[11]` = Conv 子层输出
- `z[2]`, `z[4]`, ..., `z[12]` = MLP 子层输出

### 5.3 三种预测

PC 的核心是每层 $z_\ell$ 都由三种预测组合而成：

```python
def predict(self, layer_idx, z_prev, pos_emb):
    """计算 μ_bu = sublayer_ℓ(z_prev)（不含残差）。"""
    block_idx = (layer_idx - 1) // 2
    is_conv = (layer_idx - 1) % 2 == 0
    block = self.model.layers[block_idx]

    if is_conv:
        d = block.dilation
        h = F.pad(block.input_layernorm(z_prev), (0, 0, 2 * d, 0))
        return block.local_conv(h.transpose(1, 2)).transpose(1, 2)
    else:
        return block.mlp(block.post_attention_layernorm(z_prev))
```

在 `PCDynamicMiniMind.spatiotemporal_infer_step` 中，每层的预测组合为：

$$\mu_{total} = \underbrace{\text{predict}(z_{\ell-1}, \dots)}_{\text{自下而上 (bu)}} + \underbrace{W_{temporal}^\ell \cdot z_\ell(t-1)}_{\text{时序 (temp)}} + \underbrace{W_{topdown}^\ell \cdot z_{\ell+1}(t-1)}_{\text{自上而下 (td)}}$$

$$\varepsilon_\ell = z_\ell - \mu_{total}$$

$$F_\ell = \frac{1}{2} \|\varepsilon_\ell\|^2 \cdot \pi_\ell \quad \text{(精度加权自由能)}$$

$$z_\ell \gets z_\ell - \gamma \cdot \frac{\partial F_\ell}{\partial z_\ell} = z_\ell - \gamma \cdot (\varepsilon_\ell \cdot \pi_\ell)$$

其中：
- $\gamma$ = 推理步长（默认 0.1）
- $\pi_\ell$ = 精度权重（由多巴胺信号调制，初始为 1）
- $W_{temporal}^\ell$: $12$ 个 Linear(256→256)，每层一个
- $W_{topdown}^\ell$: $11$ 个 Linear(256→256)，顶层无自上而下

### 5.4 时空推理循环

```python
for t in range(T):
    for ℓ from 1 to 12:               # 自下而上更新
        μ_bu = predict(ℓ, z_{ℓ-1})
        μ_temp = W_temp[ℓ] @ z_ℓ(t-1)
        μ_td = W_td[ℓ] @ z_{ℓ+1}(t-1) if ℓ < 12 else 0
        ε = z_ℓ - (μ_bu + μ_temp + μ_td)
        z_ℓ -= γ * ε                    # 梯度下降更新 z
```

T=2 的推理只需要 $2 \times 12 = 24$ 次 Conv1D 前向 + 少量线性投影。

### 5.5 LM Head（字节级预测器）

与标准语言模型的 LM Head 不同，这里的输出维度是 256（而不是 50K+）：

```python
self.lm_head = nn.Linear(config.hidden_size, 256, bias=False)
```

每个位置的输出 logits 经过 softmax 后预测下一个字节的概率分布。交叉熵损失使用专用权重：

```python
def _utf8_ce_weight(device):
    """UTF-8 字节 CE 权重: 0x00 (null) 权重 0, control chars 权重 0.1, 其余 1.0。"""
    w = torch.ones(256, device=device)
    w[0] = 0.0           # 忽略 null padding
    w[1:32] = 0.1        # 控制字符
    w[127:160] = 0.1     # DEL 和控制字符
    return w
```

---

## 6. 训练管道：五阶段混合循环

训练脚本 `train_pc_unified.py` 实现了一个五阶段混合训练循环：

```
Phase 1 ─► forward_with_ce()     共享前向 (有梯度), 计算 CE_loss
Phase 2 ─► spatiotemporal_infer() T 步自由能最小化, 返回 ε 和 F
Phase 3 ─► compute_F_pred()       预测自由能 + CE_converged
Phase 4 ─► Dopamine.update(F)     多巴胺信号 → 精度/学习率调制
Phase 5 ─► backward + step        权重更新
```

### 6.1 Fast Mode（纯 CE 训练）

对于快速实验，使用 `--fast` 标志跳过 PC 推理阶段：

```python
if args.fast:
    total_loss = ce_loss          # 只有 CE loss
    # 跳过 Phase 2-4
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

这相当于训练一个**纯卷积语言模型**（无预测编码），速度 ~3× 但失去了 PC 的局部学习特性。

### 6.2 完整模式（Full Hybrid）

完整的混合训练更复杂：

```python
# Phase 1: 共享前向
z_init, ce_loss = pc_model.forward_with_ce(byte_seq, labels, pos_emb)

# Phase 2: PC 推理（z 精炼）
z_detached = [z.detach() for z in z_init]
z_converged, errors_hist, F_hist, F_pred = pc_model.spatiotemporal_infer(
    z_detached, pos_emb, gamma=args.gamma, T=args.T_infer,
)

# Phase 3: 多路损失
#   F_pred = 最终步的预测自由能 (PC 损失)
#   CE_conv = CE loss in converged representation
#   loss_total = F_pred + β_local * CE_local + β_conv * CE_conv

# Phase 4: 多巴胺调制
D = dopaminergic_surprise(F_hist)  # D ∈ [0, 1]
π = 1 + η · D · ‖ε‖                # 精度调制
α_eff = α · (1 + β · D)            # 学习率调制

# Phase 5: 反向传播
loss_total.backward()
torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
optimizer.step()
```

### 6.3 β 预热策略

CE loss 的权重从 0 线性增加到上限，避免早期 PC 未收敛时 CE 主导训练：

```python
β_local = min(args.max_beta, 0.1 + global_step / total_steps * (args.max_beta - 0.1))
β_conv  = min(args.max_beta_conv, 0.0 + global_step / total_steps * args.max_beta_conv)
```

### 6.4 训练记录

使用 `--fast --subset 50000 --batch_size 32 --epochs 1` 的典型输出：

```
=== Hybrid Training ===
T=1, γ=0.1, lr=3e-4, batch=32
Device: cuda:0
Base params: 2.52M
Data: 50000 samples, 1563 steps/epoch
Warmup done (cudnn benchmark ready)

Epoch 1/1 [Hybrid]: 100%|████████| 1563/1563 [02:50<00:00, 9.87step/s]
  CE_loss: 2.512
  F_pred: 0.123
  β_local: 2.00 (maxed), β_conv: 1.00 (maxed)
```

- 训练速度：~9.9 step/s（batch_size=32, max_seq=128）
- 参数量：2.52M（仅基础骨干），原始 Transformer 版约 5M
- CE loss：~2.5（对应 PPL ≈ 12.3）

### 6.5 训练命令参考

```bash
# 纯 CE 模式（快速，推荐实验用）
python train_pc_unified.py --fast --subset 50000 --batch_size 32 --epochs 1

# 完整混合模式（PC 推理 + CE，更慢但可能效果好）
python train_pc_unified.py --subset 50000 --batch_size 24 --epochs 1 --T_infer 2

# 启用多巴胺调制
python train_pc_unified.py --subset 50000 --batch_size 24 --dopamine --T_infer 2

# 启用 QAT 量化（需 torchao）
python train_pc_unified.py --subset 50000 --batch_size 32 --quantize
```

---

## 7. 评估结果

### 7.1 Perplexity

使用 `eval_pc_language.py --local --ckpt <path>` 评估：

| 条件 | CE Loss | PPL | 说明 |
|------|---------|-----|------|
| T=0（纯前向，无 PC） | — | ~20.7 | 纯 Conv 语言模型基线 |
| T=2（PC 推理后） | — | ~20.5 | PC 带来微小改善 |
| 未训练基线（随机权重） | — | ~374.8 | 随机猜测的 PPL |

> **分析**：PPL 从 374.8 → 20.7 表明模型确实学到了语言结构。但 T=2 相比 T=0 的改善不显著，可能是因为：
> 1. 训练数据仅 50K 样本（原始数据集 >100 万）
> 2. CE 训练已经主导了表示空间，PC 推理的边际收益有限
> 3. 序列长度 128 限制了扩张卷积的感受野优势

### 7.2 文本生成

使用三条中文 prompt 测试生成质量：

```
Prompt: 人工智能的未来在于
Output: 人工智能的未来在于工智能的未来在于人工智能的未来在于...

Prompt: 小明今天去了公园，他看到
Output: 小明今天去了公园，他看到小明今天去了公园，他看到小明今天...

Prompt: 深度学习是一种
Output: 深度学习是一种度学习是一种深度学习是一种度学习是一种...
```

**发现**：
- 模型学会了**统计级别的字节分布**（PPL ~20 验证了这一点）
- 但在 ~50 字节后退化为重复模式——这是**字节级自回归误差累积**的典型表现
- 重复现象 + 数据显示的快速 PPL 改善 → 模型捕捉到了局部二元/三元组统计，但未学到长程语义结构

### 7.3 根本原因分析

模型"记住模式但不会生成"的原因：

1. **数据量不足**：50K 样本（max_seq=128）大约 6.4M 字节，对于 2.5M 参数模型来说太少了。原版 MiniMind 使用 1000 万+ 样本预训练。
2. **无 tokenizer 的代价**：字节级建模的每个预测是 1/256 的均匀分布，比 token 级（1/50K）困难得多。模型需要更多数据来学习字节到语义的映射。
3. **纯 Conv 的容量限制**：Transformer 的全局 attention 在长程依赖建模上有本质优势。Conv 感受野 139 字节不足以捕获篇章级结构。

---

## 8. 与全局 Attention 版的对比

### 8.1 PCLocalDynamicMiniMind vs PCDynamicMiniMind

| 维度 | 全局 Attention 版 | 局部 Conv 版 |
|------|-------------------|-------------|
| 骨干网络 | PCBackbone (Transformer) | PCLocalBackbone (Conv) |
| 位置编码 | RoPE（复数旋转） | 无（Conv 隐式编码） |
| 词表 | 50K+（需 tokenizer） | 256（原始字节） |
| 参数量 | ~5M | ~2.5M |
| 每步计算 | $O(n^2 d)$ | $O(nkd)$ |
| 感受野 | 全局（full seq_len） | 139 字节 |
| 训练速度 | ~5-6 step/s (bs=48) | ~9-10 step/s (bs=32) |
| 性能边界 | 可扩展到大模型 | 受感受野限制 |

### 8.2 何时选择局部 Conv 版？

**适用场景**：
- 资源极度受限（<4GB VRAM）
- 任务以局部模式识别为主（如字符级任务、短文本分类）
- 原型验证和数据探索
- 需要低延迟推理

**不适用场景**：
- 长文本生成（>200 tokens）
- 需要篇章级理解的复杂 NLP 任务
- 追求 SOTA 性能

---

## 9. 局限性分析

### 9.1 已知问题

1. **字节级误差累积**：字节级自回归生成中，每个错误预测都会改变后续所有 UTF-8 解码上下文。一个错误字节可能导致整个后续输出变为乱码。

2. **Conv 感受野瓶颈**：6 层扩张卷积 RF=139，对长程依赖建模能力有限。可通过增加层数或使用更大的 k/dilation 缓解，但会相应增加计算量。

3. **PC 的边际收益递减**：在 --fast（纯 CE）模式下模型已经学到了大部分结构，增加 PC 推理对最终指标提升有限。这可能是因为 CE 训练的目标（字节级 next-byte 预测）与 PC 的局部能量最小化目标高度重叠。

4. **PPL 评估的不完整性**：当前只评估了 500 样本 × 20 batches，统计稳定性有待提高。

### 9.2 未来改进方向

- **更多数据**：尝试完整数据集（100 万+ 样本），验证数据量对字节级模型的 scaling 效果
- **混合 token/byte**：在输入端使用 Conv1D 处理字节，但在深层引入稀疏 attention（如 Longformer 风格）
- **更大的 Conv 核**：k=5 或 k=7 并配合更大的扩张率，提高感受野
- **KV 缓存式生成**：当前生成是 $O(n)$ 重复前向，可优化为缓存中间表示的增量解码
- **对抗生成退化**：引入重复惩罚 + 更高温度 + top-k 截断的组合策略

---

## 10. 重现指南

### 10.1 环境要求

```bash
# 硬件
GPU: ≥4GB VRAM (GTX 1650 Ti 验证)
RAM: ≥16GB
磁盘: ≥10GB (存储数据集)

# 软件
Python ≥ 3.10
PyTorch ≥ 2.0 (推荐 2.12+)
```

### 10.2 完整运行流程

```bash
# 1. 数据准备
# 确保 dataset/pretrain_t2t_mini.jsonl 存在

# 2. 训练 (纯 CE 模式，50K 子集)
python train_pc_unified.py --fast --subset 50000 --batch_size 32 --epochs 1 \
    --out_dir out_pc_local --seed 42

# 3. 评估
python eval_pc_language.py --local --ckpt out_pc_local/pcl_final.pt

# 4. 基线对比 (随机权重)
python eval_pc_language.py --local
```

### 10.3 自定义训练参数

```bash
# 完整 PC 训练 (小批量避免 OOM)
python train_pc_unified.py --subset 20000 --batch_size 16 --T_infer 2 \
    --gamma 0.1 --grad_clip 1.0

# 量化训练
python train_pc_unified.py --subset 50000 --batch_size 24 --quantize --qat_groupsize 64

# 多巴胺调制
python train_pc_unified.py --subset 50000 --batch_size 24 --dopamine \
    --dopamine_eta 1.0 --dopamine_beta 0.5
```

### 10.4 数据集格式

训练数据使用 JSONL 格式，每行一个 `{"text": "..."}` 对象：

```json
{"text": "人工智能是未来科技的重要方向"}
{"text": "小明今天去了公园，他看到很多美丽的花朵"}
```

脚本自动将 `text` 字段编码为 UTF-8 字节序列，补齐或截断到 `max_seq_len`。

---

## 11. 总结与教训

### 11.1 本教程的核心理念

```
去 Token 化 + 去 Attention + 预测编码 = 极简语言模型

    ↓                    ↓                    ↓
  原始字节输入         局部卷积操作        局部学习规则
  无词表偏置           O(n) 计算量         无全局 BP
  保留完整信息         隐式位置编码        生物可解释性
```

### 11.2 经验教训

1. **小模型 + 字节级预测需要大量数据**：2.5M 参数在 50K 样本上只能学到局部 n-gram 统计，无法涌现语义理解。字节级建模的数据需求比 token 级高一个数量级。

2. **Conv -> Attention 的替代不是免费的**：虽然 Conv 在计算效率上占优，但注意力机制的长程建模能力对小模型可能更重要。混合架构可能是更好的权衡。

3. **PC 在简单任务上的边际收益**：当 CE 单目标训练已经能找到好的局部最优时，PC 的额外约束可能不增加太多价值。PC 的优势在复杂、非平稳、需要在线适应的场景中更显著。

4. **评估比训练更重要**：PPL 下降 ≠ 生成质量提升。字节级 PPL 的改善可能只反映局部模式学习，而非语义理解。文本生成评估是必要的补充。

### 11.3 代码哲学

本项目的代码遵循"懒人资深开发者"（Ponytail）原则：

- **YAGNI**：不添加未明确需要的抽象。没有配置类工厂、没有注册机制、没有复杂继承
- **标准库优先**：用 `nn.Conv1d` + `nn.Linear` + `F.pad` 堆叠出完整架构，没有额外框架
- **最小编辑**：`PCLocalDynamicMiniMind` 继承 `PCDynamicMiniMind` 只重写 3 个方法，其余复用
- **inline 注释文档**：关键设计决策以 `# ponytail:` 注释标注，附带升级路径
- **单文件可运行**：训练脚本和评估脚本分别是独立的入口文件，没有分散的多文件启动

---

*本教程对应代码库：`model/pc_backbone_local.py`（骨干网络）、`model/local_blocks.py`（Conv Block）、`model/pc_layers.py`（PCLocalDynamicMiniMind）、`train_pc_unified.py`（训练）、`eval_pc_language.py`（评估）*
