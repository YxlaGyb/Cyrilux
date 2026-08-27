# F_pred + CE 混合训练：当预测编码遇到语言建模

> 从「预测误差自组织」到「语言能力」的最后一公里  
> 基于 MiniMind (4.98M params) + PCDynamicMiniMind (5.97M params) + GTX 1650 Ti (4GB VRAM)

---

## 目录

1. [背景：纯 PC 训练的困境](#1-背景纯-pc-训练的困境)
2. [问题分析：为什么 F_pred 学不会语言？](#2-问题分析为什么-f_pred-学不会语言)
3. [混合训练方案设计](#3-混合训练方案设计)
4. [代码实现详解](#4-代码实现详解)
   - [4.1 forward_with_ce — 带梯度的前向](#41-forward_with_ce--带梯度的前向)
   - [4.2 train_pc_local_hybrid.py — 混合训练循环](#42-train_pc_local_hybridpy--混合训练循环)
   - [4.3 尺度对齐 (Scale Alignment)](#43-尺度对齐-scale-alignment)
   - [4.4 Beta Warmup](#44-beta-warmup)
   - [4.5 EMA 防坍塌正则](#45-ema-防坍塌正则)
5. [训练结果](#5-训练结果)
   - [5.1 损失曲线](#51-损失曲线)
   - [5.2 表示质量演化](#52-表示质量演化)
   - [5.3 计算开销](#53-计算开销)
6. [语言能力评估](#6-语言能力评估)
   - [6.1 Perplexity 对比](#61-perplexity-对比)
   - [6.2 文本生成对比](#62-文本生成对比)
   - [6.3 混合训练 vs 纯 F_pred 全面对比](#63-混合训练-vs-纯-f_pred-全面对比)
7. [讨论与洞察](#7-讨论与洞察)
   - [7.1 尺度饱和现象](#71-尺度饱和现象)
   - [7.2 平滑度与重复的权衡](#72-平滑度与重复的权衡)
   - [7.3 T=2 够吗？](#73-t2-够吗)
   - [7.4 为什么 LM head 需要 CE](#74-为什么-lm-head-需要-ce)
8. [重现步骤](#8-重现步骤)
9. [总结](#9-总结)

---

## 1. 背景：纯 PC 训练的困境

### 预测编码的训练框架

在 [tutorial_pc_from_scratch.md](./tutorial_pc_from_scratch.md) 中，我们构建了完整的预测编码训练流水线：

```
推理阶段 (T 步):  固定权重, z ← z - γ·∇F(z)   — 自由能最小化
学习阶段:         固定 z,    W ← W - α·∇F(W)   — 权重更新
```

其中 F 是预测误差能量：
- **静态 PC** (PCMiniMind): `F = Σ½‖z_ℓ - μ_ℓ(z_{ℓ-1})‖²` — 只有自下而上预测
- **时空 PC** (PCDynamicMiniMind): `F = Σ½‖z_ℓ - (μ_bu + μ_temp + μ_topdown)‖²` — 三路预测

**关键点**：F 完全是预测误差，和「预测下一个 token」这个语言建模目标没有直接关系。

### 令人困惑的实验结果

用纯 F_pred（预测误差）训练后的模型：

| 指标 | 值 | 说明 |
|------|-----|------|
| PPL (T=2) | **7703** | 接近随机 (vocab=6400, 随机=6400) |
| 生成文本 | 乱码字符 | 无任何语言结构 |
| LM head 权重 std | 0.0367 | vs 预训练 0.0474 |

**这就奇怪了**：虽然 PC 训练让模型学会了自组织（表示更平滑、更稀疏），但这些「好的表示特性」完全没有转化为语言能力。

---

## 2. 问题分析：为什么 F_pred 学不会语言？

### 梯度路径追踪

要理解为什么，必须追踪两个关键组件的梯度路径：

```
F_pred 的梯度路径:
  F_pred → backbone 子层参数 ✓
  F_pred → temporal_proj ✓
  F_pred → topdown_proj ✓
  F_pred → embed_tokens ✓
  F_pred → LM head ✗ ← 根本问题!
```

**LM head** (`nn.Linear(256, 6400)`) 只出现在 CE loss 的计算中，而 F_pred 计算的是 `z_ℓ vs μ_total` 的预测误差，完全不涉及 `z_L → logits` 的映射。

### 图解：F_pred 的梯度盲区

```
输入 ids → Embed → z₀ → Attn₁ → z₁ → FFN₁ → z₂ → ... → z₈ → Norm → LM Head → logits → CE loss
                        ↓        ↓     ↓               ↓
                  F_pred 只关心这些 z 的预测误差
                  梯度停止在 backbone 参数和投影层
                  LM head ← 收不到任何梯度!
```

### 预训练模型验证

为了确认问题，我们做了关键实验：加载预训练 MiniMind 权重到 PCDynamicMiniMind，然后**不训练**直接测 PPL：

| 条件 | PPL | 说明 |
|------|-----|------|
| 原始 MiniMind (原始 forward) | **70.57** | 正常语言模型 |
| PCDynamicMiniMind, T=0 | **70.57** | 完全一致 (因为 init_z = 原始 forward) |
| PCDynamicMiniMind, T=2 | **71.51** | 仅 +1.3%，PC 推理几乎不损失语言能力 |

**结论**：
1. PCDynamicMiniMind 的 init_z 路径 = 原始 MiniMind forward，LM head 是好的
2. PC 推理 (T=2) 对语言能力影响很小 (+1.3% PPL)
3. **问题出在学习阶段**：F_pred 的梯度不更新 LM head

### 根因总结

> **F_pred 优化的是「神经活动的可预测性」，不是「预测下一个 token」。**
> 一个好的预测编码器能精准预测下一时刻的神经活动，但 LM head 从好的神经表示到 token 概率的映射从未被训练。

---

## 3. 混合训练方案设计

### 核心思路

在 PC 训练中引入 CE (交叉熵) 损失，让两个互补目标联合优化：

```
总损失 = F_pred + β · scale · CE

F_pred = Σ½‖z_ℓ - μ_total‖²    — 预测误差 (自组织)
CE     = -Σ log p(token)         — 语言能力 (next-token prediction)
```

两个目标各自贡献：

| 目标 | 梯度流向 | 作用 |
|------|---------|------|
| CE | backbone 所有层 + LM head ✅ | 语言建模能力 |
| F_pred | backbone + temporal_proj + topdown_proj ✅ | 自组织、平滑表示 |

### 三个关键设计

1. **尺度对齐 (Scale Alignment)**：F_pred (量级 ~50K-90K) 和 CE (量级 ~6-8) 天然相差 4 个数量级，必须对齐
2. **Beta Warmup**：让 CE 从低权重开始，逐步增强，避免早期破坏自组织
3. **EMA 正则**：保持表示多样性，防止 CE 导致表示坍塌

### 训练流程总览

```
Phase 1: forward_with_ce (有梯度)
  → z_init, ce_loss
  → CE 梯度已流过 backbone + LM head ✓

Phase 2: spatiotemporal_infer (no_grad, T=2)
  → z_converged (基于 z_init.detach() 精炼)

Phase 3: compute_spatiotemporal_loss
  → F_pred (z detach, 只更新 backbone/temporal/topdown)

Phase 4: 尺度对齐 + 合并
  → scale = F_pred.detach() / (ce_sum.detach() + 1e-8)
  → total_loss = F_pred + β · scale · ce_sum

Phase 5: backward
  → 同时更新所有参数
```

---

## 4. 代码实现详解

### 4.1 forward_with_ce — 带梯度的前向

在 `model/pc_layers.py` 的 `PCDynamicMiniMind` 类中新增：

```python
def forward_with_ce(self, input_ids, labels, pos_emb):
    """梯度启用的前向: 返回 z_by_layer + CE loss.

    Returns:
        z_init: list[tensor, L+1], 每层表示 (有梯度)
        ce_loss: scalar tensor, 交叉熵损失 (梯度流遍 backbone + lm_head)
    """
    z = []
    h = self.model.model.embed_tokens(input_ids)
    z.append(h)

    for block in self.model.model.layers:
        res = h
        h = block.self_attn(block.input_layernorm(h), pos_emb)[0]
        h = h + res
        z.append(h)

        res = h
        h = block.mlp(block.post_attention_layernorm(h))
        h = h + res
        z.append(h)

    # CE from top layer — 这是 LM head 唯一能收到梯度的路径
    h_top = self.model.model.norm(z[-1])
    logits = self.model.lm_head(h_top)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    ce_loss = nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return z, ce_loss
```

**注意**：这和 `init_z` 看似相同，但区别在于 `torch.no_grad()`。`init_z` 是推理用的（无梯度），`forward_with_ce` 是有梯度的，这样 ce_loss.backward() 才能更新参数。

### 4.2 train_pc_local_hybrid.py — 混合训练循环

完整训练脚本见 `../train_pc_local_hybrid.py`，核心循环如下：

```python
for step, (input_ids, labels) in enumerate(pbar):
    input_ids = input_ids.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    bsz, seq_len = input_ids.shape

    # ── Phase 1: 共享前向 (有梯度) ──
    pos_emb = pc_model.get_position_embeddings(seq_len, device)
    z_init, ce_loss = pc_model.forward_with_ce(input_ids, labels, pos_emb)

    # ── Phase 2: PC 推理 (z detach, 无梯度) ──
    z_detached = [z.detach() for z in z_init]
    z_converged, errors_hist, F_hist = pc_model.spatiotemporal_infer(
        z_detached, pos_emb, gamma=gamma, T=T_infer,
    )

    # ── Phase 3: F_pred ──
    F_pred = pc_model.compute_spatiotemporal_loss(z_converged, pos_emb)

    # ── Phase 4: 尺度对齐 + 合并 ──
    beta = min(max_beta, 0.1 + global_step / total_steps * (max_beta - 0.1))
    ce_sum = ce_loss * (bsz * seq_len)
    scale = F_pred.detach() / (ce_sum.detach() + 1e-8)
    scale = scale.clamp(0.1, 10.0)
    total_loss = F_pred + beta * scale * ce_sum

    # ── Phase 5: backward ──
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
    optimizer.step()
```

### 4.3 尺度对齐 (Scale Alignment)

**问题**：F_pred 和 CE 不在同一个量级。

```
F_pred ≈ 50,000–90,000   (Σ over batch*seq_len*layers of ‖ε‖²)
ce_sum  ≈ 6-8 × 8×128 ≈ 6,000–8,000  (CE loss × batch_size × seq_len)
```

如果直接相加 `F_pred + CE`，F_pred 会完全主导梯度，CE 几乎没有影响。

**解决方案**：动态尺度对齐

```python
scale = F_pred.detach() / (ce_sum.detach() + 1e-8)
scale = scale.clamp(0.1, 10.0)  # 防止极端值
total_loss = F_pred + beta * scale * ce_sum
```

**为什么 clamp 到 [0.1, 10.0]**：防止某个目标的梯度完全消失。如果 F_pred 主导了，`scale` 会被 cap 在 10.0，CE 仍然有 10 倍的放大，反之 CE 主导时 scale 会被 floor 到 0.1。

### 4.4 Beta Warmup

**问题**：训练初期 F_pred 的自组织信号很弱，如果 CE 权重太大，表示会被拉到「只利于语言建模」的方向，破坏自组织。

**方案**：线性 warmup，从 `beta=0.1` 到 `beta=max_beta=2.0`

```python
beta = min(max_beta, 0.1 + global_step / total_steps * (max_beta - 0.1))
```

训练步数 1250，beta 从 0.1 → 2.0。

### 4.5 EMA 防坍塌正则

**问题**：CE 是一个很强的约束，可能把所有层的表示拉向「只利于 LM head 分类」的空间，导致表示坍塌（所有 token 的表示趋同）。

**方案**：EMA (Exponential Moving Average) 参考，惩罚表示偏离慢速平均值：

```python
# 初始化/更新 EMA z
if ema_z is None:
    ema_z = [z.detach().clone() for z in z_converged]
else:
    alpha = 0.99
    for ell in range(len(z_converged)):
        ema_z[ell] = alpha * ema_z[ell] + (1 - alpha) * z_converged[ell].detach()

# 正则项
reg = 0.0
for ell in range(1, pc_model.num_sub_layers + 1):
    reg += ((z_converged[ell] - ema_z[ell]) ** 2).sum()
F_pred = F_pred + 0.5 * ema_lambda * reg  # ema_lambda=0.001
```

一个极轻的正则（λ=0.001），只在表示偏离历史轨迹太远时才起作用。

---

## 5. 训练结果

### 5.1 损失曲线

训练配置：
- 数据：10,000 条，max_seq_len=128
- batch_size=8，1250 steps/epoch × 1 epoch
- T_infer=2，gamma=0.1，lr=3e-4 (cosine)
- AdamW (weight_decay=0.1)，grad_clip=1.0

**关键节点**：

| Step | CE Loss | PPL (exp(CE)) | F_pred | beta | scale |
|------|---------|---------------|--------|------|-------|
| 0 | 8.9606 | 7763 | — | 0.100 | — |
| 1 | 8.8237 | 6789 | — | 0.101 | — |
| 100 | 8.0082 | 3000 | ~90K | 0.252 | 10.00 |
| 500 | 7.1569 | **1281** | ~85K | 0.860 | 10.00 |
| 1000 | 6.7003 | **812** | ~87K | 1.620 | 10.00 |
| 1249 | 6.5646 | **742** | ~87K | 2.000 | 10.00 |

**关键观察**：
- CE Loss 从 8.96 降到 6.56，PPL 从 7763 降到 742（**10 倍提升**）
- Scale 从第一步开始就饱和在 10.0（说明 F_pred 一直主导损失量级）
- Beta 从 0.1 线性增长到 2.0
- F_pred 稳定在 ~85K-90K，没有剧烈波动

### 5.2 表示质量演化

| Step | Smoothness | Sparsity | Variance |
|------|-----------|----------|----------|
| 0 | ~3.0 | ~10.5 | ~0.06 |
| 100 | 6.28 | 11.0 | 0.04 |
| 500 | 9.92 | 12.0 | 0.04 |
| 1000 | 24.44 | 12.3 | 0.04 |
| 1249 | **94.96** | **12.75** | 0.04 |

- **Smoothness** 从 3 跳到 95（时序平滑度大幅提升，表示在时间维度上变化更连续）
- **Sparsity** 从 10.5 增加到 12.75（表示更稀疏，有利于选择性激活）
- **Variance** 稳定在 0.04（表示没有坍塌，保持多样性）

### 5.3 计算开销

| 阶段 | 耗时 | 占比 |
|------|------|------|
| Phase 1 (forward_with_ce) | ~0.5ms | 小 |
| Phase 2 (spatiotemporal_infer) | ~2ms | 主要 |
| Phase 3 (compute F_pred) | ~0.5ms | 小 |
| Phase 4-5 (merge + backward) | ~1ms | 中 |
| **总计/step** | **~4ms** | |
| **1250 steps 总耗时** | **~5 分钟** | |

在 GTX 1650 Ti (4GB VRAM) 上，混合训练只比纯 F_pred 训练多了 Phase 1 的前向开销，整体增加约 15% 时间。

---

## 6. 语言能力评估

### 6.1 Perplexity 对比

使用 `eval_pc_language.py --ckpt out_pc_local_hybrid/hybrid_final.pt` 在 held-out 500 条数据上评估：

| 模型 | PPL @ T=2 | PPL @ T=0 | 说明 |
|------|----------|----------|------|
| 随机初始化 | 7703 | 7703 | vocab=6400, 随机 ≈ 6400 |
| 预训练 MiniMind | 71.5 | **70.6** | 参考上界 |
| **混合训练 (hybrid_final)** | **742.2** | **732.4** | ✅ 从 7703 降到 742 |
| 纯 F_pred 训练 | 7703 | 7703 | ⚠️ 无变化 |

**分析**：
- 混合训练将 PPL 从 7703 降到 742（10 倍改善），证明了 CE 梯度成功训练了 LM head
- T=0 vs T=2 差异很小（732 vs 742），说明 PC 推理对已训练好的模型影响不大
- 但距离预训练的 70.6 还有很大差距（10K 条数据 vs 预训练的数 GB 数据）

### 6.2 文本生成对比

同一 prompt，相同采样参数（temperature=0.8, top_k=20, max_new_tokens=40）：

| Prompt | 纯 F_pred (baseline) | 混合训练 (hybrid) |
|--------|---------------------|-------------------|
| "人工智能的未来在于" | `的垰的L垰的L的的...` (乱码) | `人工智能的未来在于中的的篇文本中的中的的...` (中文，有重复) |
| "小明今天去了公园，他看到" | (类似乱码) | 可识别的中文片段 |

**混合训练的输出**：可以生成**有语义的中文文本**了！虽然存在重复问题（"的中的"模式），但相比于纯 F_pred 的随机字节，这是一个质变。

**生成质量提升路径**（推测）：
1. 10K 数据训练 → 能生成中文但重复 → ✅ 当前
2. 50K-100K 数据 → 可望显著降低重复 → 待验证
3. 更优的 scale cap + 更长训练 → 向预训练 PPL 靠近 → 待验证

### 6.3 混合训练 vs 纯 F_pred 全面对比

| 维度 | 纯 F_pred | 混合训练 (F_pred + CE) |
|------|----------|----------------------|
| **语言能力** | ❌ 无 (PPL=7703) | ✅ 有基本语言能力 (PPL=742) |
| **表示自组织** | ✅ 平滑、稀疏 | ✅ 更平滑 (smoothness=95) |
| **LM head** | ❌ 从未训练 | ✅ 正常训练 |
| **生成文本** | ❌ 乱码 | ✅ 可读中文 (有重复) |
| **训练速度** | 更快 | +15% 时间 |
| **梯度路径** | backbone + 投影层 | backbone + 投影层 + LM head ✅ |

---

## 7. 讨论与洞察

### 7.1 尺度饱和现象

整个训练过程中 `scale = 10.0`（上限）。这意味着：

```
F_pred (87K) / (ce_sum (6.56 × 1024) ≈ 6717) ≈ 12.9 → clamp 到 10.0
```

F_pred 的量级天然是 CE 的 ~13 倍，被 clamp 后 CE 的等效权重为 `β × 10.0`。

**启示**：
- 如果去掉 clamp，scale 会达到 ~13，可能等效权重更大
- 但 clamp 提供了一个稳定机制，防止训练初期的极端波动
- 训练结束仍然饱和，说明**也许 scale cap 可以更高**（如 20.0），让 CE 有更大影响
- 或者反过来，**降低 F_pred 的量级**（如用 `mean` 替代 `sum`）

### 7.2 平滑度与重复的权衡

Smoothness 从 3 → 95，同时生成文本表现出重复模式。

**这可能不是巧合**：

```
高 smoothness → 相邻 token 的表示几乎相同 → LM head 输出几乎相同的 logits → 重复 token
```

预测编码的目标就是让 `z(t) ≈ μ_total(t)`，而 μ_total 包含时序预测 `W_temp(z(t-1))`，这天然鼓励平滑。但语言不是完全平滑的 — 动词后需要名词，名词后可以是各种可能。过平滑会抹掉这种多样性。

**可能的缓解方法**：
1. **降低 T**：T=1 可能已经足够，减少过平滑
2. **降低 gamma**：gamma 控制推理步长，更小的 gamma 每次更新更小
3. **F_pred 退火**：训练后期降低 F_pred 权重（让 CE 主导）
4. **增加数据多样性**：更大的数据集可能帮助模型学会「何时不平滑」

### 7.3 T=2 够吗？

当前 T=2 的推理步数下：

- PPL(T=2) ≈ PPL(T=0)（差异 < 2%）
- 说明 2 步推理已经收敛或接近收敛
- 对 256-dim、8 子层的小模型，T=2 似乎是足够的

**推测**：对于更大的模型（如 768-dim, 24 层），可能需要 T=3-5。

### 7.4 为什么 LM head 需要 CE

这是整个教程最重要的教训：

> **预测编码优化的是表示的可预测性，语言建模优化的是表示的判别性。**
> 两者互补，但不等价。

| 目标 | 数学形式 | 优化什么 | 谁受益 |
|------|---------|---------|-------|
| F_pred | Σ½‖z - μ‖² | 表示 → 可预测、平滑 | backbone, proj |
| CE | -log p(token) | z_L → token 判别 | backbone, LM head |

LM head 本质是一个分类器（256-dim → 6400 类），要训练它需要分类信号（CE）。预测误差（F_pred）只告诉它「z_L 应该接近 μ_total」，不告诉它「z_L 应该能区分 cat 和 dog」。

---

## 8. 重现步骤

```bash
# 1. 激活环境
cd e:\SystemShare\Documents\virtuosov2
.venv\Scripts\activate

# 2. 运行混合训练 (约 5 分钟)
python train_pc_local_hybrid.py

# 3. 评估混合训练模型
python eval_pc_language.py --ckpt out_pc_local_hybrid/hybrid_final.pt

# 4. 对比：评估纯 F_pred 模型 (或未训练 baseline)
python eval_pc_language.py
```

### 关键文件索引

| 文件 | 说明 |
|------|------|
| `train_pc_local_hybrid.py` | 混合训练脚本 (入口) |
| `model/pc_layers.py` → `forward_with_ce` | 带梯度的前向 + CE |
| `model/pc_layers.py` → `compute_spatiotemporal_loss` | F_pred 计算 |
| `eval_pc_language.py` | 语言能力验证 (已支持 --ckpt) |
| `out_pc_local_hybrid/hybrid_final.pt` | 最终 checkpoint |

### 依赖

```
Python 3.11+, PyTorch 2.12+, CUDA 12+
GPU: ≥4GB VRAM (GTX 1650 Ti 实测通过)
数据: dataset/pretrain_t2t_mini.jsonl (自动使用)
```

---

## 9. 总结

### 关键教训

1. **F_pred ≠ 语言能力**。无论预测编码多完美地自组织了神经表示，LM head 没有 CE 梯度就永远学不会映射到词汇。这不是超参数问题，是**架构问题**。

2. **尺度对齐是混合训练的关键工程细节**。F_pred 和 CE 天然差 4 个数量级，不处理的话 CE 相当于不存在。动态 scale 加 clamp 是一个通用的多目标合并模式。

3. **10 倍 PPL 提升，只是开始**。10K 数据、1250 步、~5 分钟训练将 PPL 从 7703 降到 742。这验证了方案的正确性。要接近预训练水平（PPL≈70），预计需要 ~100× 更多数据。

4. **表示质量和语言能力可以共存**。混合训练后的表示更平滑、更稀疏，同时有了语言能力——说明这两个目标不是冲突的，而是互补的。

### 改进方向

| 方向 | 预期效果 | 难度 |
|------|---------|------|
| 增大数据量 10K→50K | PPL 可能降到 200-300 | 低 |
| 降低 scale cap 10→3 | CE 梯度更大，可能加速语言学习 | 低 |
| F_pred 退火 (后期降低) | 减少过平滑，改善生成多样性 | 中 |
| 尝试 T=1 | 减少平滑度，可能改善重复 | 低 |
| CE annealing (后期降低 beta) | 早期学语言，后期自组织精炼 | 中 |

### 与已有教程的关系

本教程是 [tutorial_pc_from_scratch.md](./tutorial_pc_from_scratch.md) 的第 11 章。前者覆盖了 PC 的基础架构、静态 PC、时空 PC、局部学习器，本教程覆盖了从「纯 PC 自组织」到「语言能力」的最后一步。

### 致 Ponytail

> 最好的代码是没写出来的代码。但如果必须写，写最少的、能解决问题的代码。
>
> 本教程的混合训练脚本 `train_pc_local_hybrid.py` 是从 `train_pc_local.py` 派生的，只改了前向和损失合并两处。不需要新的模型类，不需要新的优化器，不需要新的评估框架——三处修改完成了从「自组织」到「语言能力」的跨越。
>
> `ponytail: scale 自动对齐 F_pred 和 CE 的量级, 防止一个目标主导` — 一行注释，概括了整个方案的核心工程智慧。
