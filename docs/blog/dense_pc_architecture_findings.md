# 密集 GPU PC 网络: 架构设计与实验记录

## 动机

稀疏事件驱动版 Cyrene 在 GPU 上利用率仅 15-38%（scatter_add/gather/index_put_ 等稀疏操作是 memory-bound）。目标是创建一个完全基于 matmul 的 "密集" PC 网络，GPU 利用率 > 90%。

## 架构

6 层全连接网络，全部用 `torch.nn.Parameter` + matmul 实现，零 scatter/gather：

| 层 | 神经元 | 连接 | 参数量 |
|---|--------|------|--------|
| L0 | 256+256 | one-hot + PE sin/cos | - |
| L4 | 1024 | L0→L4 全连接 | 524K |
| L2 | 384 | L4→L2 全连接 | 393K |
| L3 | 384 | L2→L3 全连接 | 147K |
| L5 | 256 | L3→L5 全连接 | 98K |
| L6 | 128 | L5→L6 全连接 | 33K |
| **总计** | | | **1.26M (2.4MB)** |

GPU 显存占用：**15 MB**（稀疏版 529 MB）。

## 关键机制

### 1. per-frame 归一化
每帧 `z = z / (z.norm(dim=-1) + 1e-8)` 防止激活值爆炸。替代了稀疏版的 `rsqrt(fan_in)`。

### 2. 列 Oja（LM Head 平权）
每次 Hebbian 更新后 `W_LM = W_LM / W_LM.norm(dim=0)`，所有 256 列单位 L2 范数。彻底解决中文高字节垄断问题。

### 3. 时序矩阵 `W_t`
`z[t] += 0.1 * W_t @ z[t-1]` 替代稀疏版的标量 `t_weight`。每层 [dim, dim] 的可学习矩阵。

### 4. k-WTA
L4/L2/L3=80% 保留，L5/L6=100%（LM Head 前全保留）。

## 战绩

| 指标 | 稀疏版 (best) | 密集版 (10000步) |
|------|--------------|-----------------|
| off-diag | 0.094 | 0.39 |
| LM norm | 93 | 1.0 (列Oja) |
| 显存 | 529 MB | 15 MB |
| GPU 利用率 | 30% | 90% |
| 每步速度 | 5ms/seq | 1.4ms/seq |
| 生成结果 | 英文字母+符号 | 纯英文字母+空格 |
| 代码行数 | 2000+ | 312 |

## 踩坑记录

### 坑 1: eps=0 → 无学习
`z = mu` 时 `eps = z - mu = 0`，Hebbian 外积全零。必须让 `z != mu`：通过时序干扰 `z[t] += 0.1 * W_t @ z[t-1]` 制造差异。

### 坑 2: 密集 matmul + k-WTA = 矛盾
全算完再砍掉 80% 浪费算力。只在 L4/L2/L3 保留 80% 作为适度稀疏，L5/L6 全保留。

### 坑 3: W_t 时序矩阵正反馈爆炸
`W_t @ z[t-1]` 递归放大 → nan。必须加 per-frame 归一化保护。

### 坑 4: LM Head 行 norm 失衡
高字节行 norm 是英文字母行的 1000 倍 → 推理全选中文字节。解决方案：行归一化 + 列 Oja 同时用。

### 坑 5: fp16 溢出
密集版 Hebbian 外积 `dW = eps.T @ z * eta`，eta 必须远小于稀疏版（约 5000 分之一）。

## 使用方法

```sh
# 训练
uv run cyrilux train --backend dense -d dataset/en_pure.jsonl --hidden-size 1024 -b 16

# Python API
from model.dense import DensePCNet, DensePCConfig  # 原 model.pc.dense
net = DensePCNet(DensePCConfig(d_l4=1024)).cuda()
logits = net(byte_ids)
stats = net.learn(byte_ids, targets)
```
