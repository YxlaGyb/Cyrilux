# 多巴胺门控持续学习系统 — 完整教程

> **防止灾难性遗忘**: 让 PC (预测编码) 模型在顺序学习多个任务时, 记住旧任务的知识。

---

## 目录

1. [为什么需要持续学习？](#1-为什么需要持续学习)
2. [系统架构总览](#2-系统架构总览)
3. [MemoryBank — 多巴胺效用驱动的记忆银行](#3-memorybank--多巴胺效用驱动的记忆银行)
4. [ForgettingSniffer — 遗忘嗅探器](#4-forgettingsniffer--遗忘嗅探器)
5. [OfflineReplayer — 生成式自巩固（Bonus）](#5-offlinereplayer--生成式自巩固bonus)
6. [PCLocalDynamicMiniMind — 字节级 PC 模型](#6-pclocaldynamicminimind--字节级-pc-模型)
7. [4 任务压力测试实战](#7-4-任务压力测试实战)
8. [实验结果解读](#8-实验结果解读)
9. [超参数调优指南](#9-超参数调优指南)

---

## 1. 为什么需要持续学习？

### 灾难性遗忘

神经网络在**顺序学习多个任务**时, 学新任务会覆盖旧任务的权重, 导致旧任务性能断崖式下跌。这就是 **灾难性遗忘 (Catastrophic Forgetting)**。

```
任务 A (日常对话) → 任务 B (科技知识) → 任务 C (医疗问诊)
     ↑                      ↑                      ↑
  学得好                 忘掉 A                 忘掉 A, B
```

### 解决方法: 记忆回放

最直接的方法: **存一些旧样本, 学新任务时顺带复习**。但这有几个问题:

- **存什么？** 全部存 — 存不下。随机存 — 关键样本可能被淘汰。
- **什么时候复习？** 每步都复习 — 拖慢新任务学习。不复习 — 忘了。
- **复习多少？** 回放太少不管用, 太多影响新任务。

### 我们的方案: 多巴胺门控系统

用三个模块协同解决:

| 模块 | 职责 | 一句话 |
|---|---|---|
| **MemoryBank** | 存什么 & 怎么取 | 按多巴胺效用分值加权采样 |
| **ForgettingSniffer** | 什么时候复习 | 检测到遗忘才修复, 不遗忘就不打扰 |
| **OfflineReplayer** | 能不能更主动 | 自己生成合成数据来复习 (Bonus) |

---

## 2. 系统架构总览

```
                           ┌──────────────────┐
                           │  新任务数据流     │
                           │  (byte_tensor)    │
                           └────────┬─────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                   模型训练主循环                              │
│                                                             │
│  1. 正常训练步: CE(model(x), y) → backward                  │
│  2. 每隔 replay_ratio 步:                                   │
│       └─ MemoryBank.sample() → CE 回放                      │
│  3. 每隔 check_interval 步:                                 │
│       └─ Sniffer.check() → 检测到遗忘 → repair             │
│  4. 任务学完:                                               │
│       └─ MemoryBank.add_samples() 存入 exemplars            │
└─────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
    ┌──────────────┐      ┌────────────────┐      ┌──────────────┐
    │  MemoryBank  │      │ForgettingSniffer│      │OfflineReplay │
    │  (记忆银行)   │◄────►│  (遗忘嗅探器)    │◄────►│ (生成式回放)  │
    └──────────────┘      └────────────────┘      └──────────────┘
            │                       │
            │                  ┌────┴────┐
            ▼                  ▼         ▼
    ┌──────────────┐    ┌────────┐  ┌────────┐
    │ Exemplar 池   │    │ 阈值  │  │修复步数│
    │ per-task FIFO │    │ 1.2~2 │  │ 5~20   │
    └──────────────┘    └────────┘  └────────┘
```

### 训练流程 (以 Phase 2 为例)

```
任务 A (首次, 无保护):
  for step in 0..total_steps:
      train_step(A)         # 纯 CE

  → 收集 A 的 exemplars → MemoryBank

任务 B (带回放保护):
  for step in 0..total_steps:
      train_step(B)                    # 主任务 CE
      if step % replay_ratio == 0:
          replay(MemoryBank.sample())  # 回放旧任务
      if step % check_interval == 0:
          forgotten = Sniffer.check()
          if forgotten:
              repair()                # 进入修复模式

  → 收集 B 的 exemplars → MemoryBank
```

### 文件结构

```
virtuosov2/
├── continual/                    # 持续学习核心模块
│   ├── __init__.py               # 导出 MemoryBank, Exemplar, ForgettingSniffer
│   ├── memory_bank.py            # 多巴胺效用驱动的记忆银行
│   ├── forgetting_sniffer.py     # 遗忘嗅探 + 自触发修复
│   └── offline_replay.py         # 生成式自巩固 (Bonus)
├── model/
│   └── pc_layers.py              # PCLocalDynamicMiniMind 字节级 PC 模型
├── forgetting_pressure_test.py   # N 任务压力测试入口
├── prepare_4task.py              # 4 领域数据准备
├── trainer_utils.py              # get_lr / setup_seed 工具
└── docs/
    └── continual_learning_tutorial.md   # ← 本教程
```

---

## 3. MemoryBank — 多巴胺效用驱动的记忆银行

### 核心思想

不是所有样本都平等。有些样本"更有价值"——它们的信息量大、难度高、代表性好。**多巴胺分值 (dopamine_score)** 就是样本重要性的度量。

### 数据结构

每个记忆单元是一个 `Exemplar`:

```python
@dataclasses.dataclass
class Exemplar:
    byte_tensor: torch.Tensor   # [128] uint8 — UTF-8 字节序列
    label_tensor: torch.Tensor  # [128] long — -100 表示 padding
    task_id: str                # 所属任务 (A, B, C, D...)
    dopamine_score: float = 0.5 # 多巴胺效用分值 (越高越重要)
    baseline_loss: float = 0.0  # 刚存入时的 CE loss (作为遗忘检测基线)
```

Ponytail: **存张量而非文本** — 反序列化零解析开销, 直接喂模型。

### 工作原理

```
MemoryBank
├── 任务 A: [Exemplar×2000]  ← FIFO 淘汰
├── 任务 B: [Exemplar×2000]  ← FIFO 淘汰
└── 任务 C: [Exemplar×2000]  ← FIFO 淘汰
```

#### 写入: `add_samples(task_id, samples, dopamine_score, baseline_loss)`

每学完一个任务, 从中**随机抽取** N 条样本连同当前 loss 存入 bank:

```python
memory_bank.add_samples(
    task_id='C',
    samples=[(byte_t, label_t), ...],
    dopamine_score=0.5,           # 当前版本固定值, 可扩展为动态计算
    baseline_loss=total_bl / n,   # 刚存入时的 CE loss
)
```

如果某个任务的样本超过 `max_per_task`, 按 **FIFO** 淘汰最早的。

#### 读取: `sample(batch_size, strategy='dopamine')`

两种采样策略:

| 策略 | 概率权重 | 效果 |
|---|---|---|
| `dopamine` | `max(dopamine_score, 0.1)` | 高分样本更可能被回放 |
| `uniform` | 全部 1.0 | 等概率采样 |

内部实现使用 `torch.multinomial` 做加权采样 (有放回/无放回)。

```python
replay_ex = memory_bank.sample(batch_size=16, strategy='dopamine')
# → [Exemplar, Exemplar, ...]
```

#### 评估: `evaluate(model, device, N=32)`

纯前向计算 bank 中各任务的 CE loss, 与基线对比:

```python
results = memory_bank.evaluate(model, device='cuda:0')
# → {'A': {'avg_ce': 1.23, 'baseline_ce': 0.89, 'ratio': 1.38}, ...}
```

关键指标: **ratio = avg_ce / baseline_ce**。ratio > 1 说明该任务被遗忘了。

#### 序列化: `state_dict()` / `load_state_dict()`

保存/恢复整个 bank:

```python
state = memory_bank.state_dict()
torch.save(state, 'memory_bank.pt')
# ...
memory_bank.load_state_dict(torch.load('memory_bank.pt'))
```

### 代码 (简化版)

```python
class MemoryBank:
    def __init__(self, max_per_task: int = 2000):
        self.max_per_task = max_per_task
        self._store: dict[str, List[Exemplar]] = {}

    def add_samples(self, task_id, samples, dopamine_score, baseline_loss):
        if task_id not in self._store:
            self._store[task_id] = []
        buf = self._store[task_id]
        for byte_t, label_t in samples:
            buf.append(Exemplar(byte_tensor=byte_t.clone(),
                                label_tensor=label_t.clone(),
                                task_id=task_id,
                                dopamine_score=dopamine_score,
                                baseline_loss=baseline_loss))
        # FIFO 淘汰
        while len(buf) > self.max_per_task:
            buf.pop(0)

    def sample(self, batch_size, strategy='dopamine'):
        all_ex, weights = [], []
        for buf in self._store.values():
            for ex in buf:
                all_ex.append(ex)
                w = max(ex.dopamine_score, 0.1) if strategy == 'dopamine' else 1.0
                weights.append(w)
        w = torch.tensor(weights, dtype=torch.float)
        w = w / w.sum()
        idx = torch.multinomial(w, min(batch_size, len(all_ex)), replacement=False)
        return [all_ex[i] for i in idx.tolist()]
```

---

## 4. ForgettingSniffer — 遗忘嗅探器

### 核心思想

**不猜什么时候遗忘, 而是去检测**。每隔一定步数, 算一下旧任务的 CE loss, 如果比基线高了, 说明开始忘了——立刻触发修复。

### 工作原理

```
训练进行中...
    │
    ├── 每 500 步: check()
    │      ├── 从 MemoryBank 取每个任务的 N 条 exemplars
    │      ├── 纯前向计算 CE loss (无梯度, T=0)
    │      ├── loss_ratio = current_ce / baseline_ce
    │      └── 如果 loss_ratio > threshold → "遗忘!"
    │
    └── 进入修复模式 (repair)
           ├── LR 降至 repair_lr_factor × current_lr
           ├── 运行 repair_steps 步旧任务回放
           └── 恢复原始 LR
```

### 配置参数

| 参数 | 默认值 | 作用 |
|---|---|---|
| `check_interval` | 200 | 每隔多少步检测一次 |
| `threshold` | 1.2 | CE loss 比值超过此值触发修复 |
| `repair_steps` | 10 | 每次修复跑多少步 |
| `repair_lr_factor` | 0.3 | 修复时 LR 降为当前的 0.3 倍 |
| `eval_n` | 32 | 每个任务检测时用多少条 exemplars |

### 为什么 LR 要降低？

遗忘是因为新任务的梯度覆盖了旧任务的权重。修复时如果 LR 太高, 修复的梯度把新任务覆盖了→来回震荡。**降 LR → 微调而非重写**。

### 嗅探 === 轻量级

嗅探只做 **T=0 纯前向** (无 PC 推理, 不更新 z 节点), 开销约等于 1 步训练:

```python
with torch.no_grad():
    z = model.init_z(x)           # 一次前向
    h = model.model.norm(z[-1])
    logits = model.model.lm_head(h)
    # 算 CE → 对比 baseline → 决策
```

### 代码 (关键路径)

```python
class ForgettingSniffer:
    def __init__(self, memory_bank, model, check_interval=200,
                 threshold=1.2, repair_steps=10, repair_lr_factor=0.3,
                 eval_n=32):
        ...

    def check(self, gs, device):
        """检测所有已知任务, 返回遗忘的任务 ID 列表。"""
        if self.memory_bank.total == 0:
            return None
        results = self.memory_bank.evaluate(self.model, device, N=self.eval_n)
        forgotten = []
        for task_id, info in results.items():
            if info['ratio'] > self.threshold:
                forgotten.append(task_id)
        return forgotten if forgotten else None

    def repair_begin(self, optim, current_lr, device):
        """进入修复模式: 降低 LR。"""
        self.is_repairing = True
        self._repair_gs = 0
        self._original_lr = current_lr
        repair_lr = current_lr * self.repair_lr_factor
        for pg in optim.param_groups:
            pg['lr'] = repair_lr
        return repair_lr

    def repair_end(self, optim, original_lr):
        """退出修复模式: 恢复 LR。"""
        self.is_repairing = False
        for pg in optim.param_groups:
            pg['lr'] = original_lr

    def get_replay_batch(self, batch_size, device):
        """从 MemoryBank 均匀采样一个修复批次。"""
        n_tasks = max(len(self.memory_bank.tasks), 1)
        per_task = max(batch_size // n_tasks, 1)
        batch = []
        for task_id in self.memory_bank.tasks:
            batch.extend(self.memory_bank.sample(per_task, strategy='uniform'))
        ...
        return byte_batch, label_batch
```

### 在训练循环中的位置

```python
for bt, lt in loader:
    gs += 1

    # 1. 正常训练步
    loss, lr = train_step(model, optim, bt, lt, device, base_lr, gs, total_steps)

    # 2. 定期回放
    if bank.total > 0 and gs % replay_ratio == 0 and not sniffer.is_repairing:
        replay_ex = bank.sample(batch_size, strategy='dopamine')
        # → CE backward

    # 3. 嗅探检查
    if sniffer.is_repairing or (gs % check_interval == 0 and gs > 0):
        forgotten = sniffer.check(gs, device)
        if forgotten:
            sniffer.repair_begin(optim, lr, device)
            for _ in range(repair_steps):
                rb, rl = sniffer.get_replay_batch(batch_size, device)
                # → CE backward (低 LR)
            sniffer.repair_end(optim, lr)
```

---

## 5. OfflineReplayer — 生成式自巩固（Bonus）

### 核心思想

能不能**完全不需要旧数据**, 靠模型自己生成样本来复习？

**PC 模型能生成**: 给它前几个字节的 prompt, 它能自回归地生成后续文本。生成的"合成数据"可以充当训练样本。

### 工作原理

```
MemoryBank 中的 Exemplar
    │
    ├── 取前 8 字节: [0xE4, 0xBD, 0xA0, 0xE5, 0xA5, ...]
    │
    ├── 作为 prompt 喂给 model.generate_with_pc()
    │      T_infer=0, gamma=0.1, temperature=0.7, top_k=20
    │
    ├── 生成结果: [prompt_len + 120] 字节序列
    │
    ├── 截断/填充到 128 → (byte_tensor, label_tensor)
    │
    └── CE backward 训练
```

### 代码 (关键路径)

```python
class OfflineReplayer:
    def generate_for_task(self, task_id, n_samples=100, max_new_tokens=120,
                          temperature=0.7, top_k=20, prompt_len=8, device='cuda:0'):
        """为指定任务生成合成训练数据。"""
        buf = self.memory_bank._store.get(task_id, [])
        samples = []
        idx = torch.randperm(len(buf))[:min(n_samples, len(buf))].tolist()
        for i in idx:
            ex = buf[i]
            prompt = ex.byte_tensor[:prompt_len].unsqueeze(0).to(device)
            generated = self.model.generate_with_pc(
                prompt, max_new_tokens=max_new_tokens,
                T_infer=0, gamma=0.1,
                temperature=temperature, top_k=top_k,
                eos_token_id=0x02,
            )
            # 组装成 (byte_tensor, label_tensor)
            full_seq = generated[0]
            padded = torch.cat([full_seq.cpu(), torch.zeros(128 - seq_len)])
            byte_t = padded.to(torch.uint8)
            label_t = byte_t.clone().to(torch.long)
            label_t[byte_t == 0x00] = -100
            samples.append((byte_t, label_t))
        return samples
```

**Ponytail**: 这是生成式回放的最简实现 — 每次生成只做一次前向解码。实际使用时在任务间隙调用。

---

## 6. PCLocalDynamicMiniMind — 字节级 PC 模型

### 为什么是字节级？

传统 LLM 需要 tokenizer (分词器) — 词表 32K~128K, embedding 矩阵巨大。字节级直接操作 UTF-8 字节 (0-255), 词表只有 256:

| 维度 | 传统 LLM | PC 字节级 |
|---|---|---|
| 词表大小 | 32,000+ | 256 |
| Embedding | 大矩阵 | Conv1d (1→256) |
| 参数量 (4层) | ~50M | ~6.59M |
| 显存占用 | ~4GB+ | ~0.8GB |

### 模型层级

```
PCMiniMind                  ← 基础 PC 包装 (Fast Weight)
  └── PCDynamicMiniMind     ← 添加 temporal/topdown 投影 (Slow Weight)
       └── PCLocalDynamicMiniMind  ← 字节级 Conv1d 输入, 无 tokenizer
```

### PCLocalDynamicMiniMind 结构

```ascii
输入: UTF-8 字节 [0xE4, 0xBD, 0xA0, ...]  (shape [B, 128])
  │
  ├── Conv1d(1→256, kernel=13, causal padding)
  │     聚合一字节的局部上下文
  │
  ├── 6× DilatedConv (dilation=1,2,4,8,16,32)
  │     多尺度感受野
  │
  ├── RMSNorm
  │
  ├── Linear(256→vocab=256)
  │
  └── 输出: logits [B, 128, 256]
```

`forward_with_ce` 内部:

```python
def forward_with_ce(self, byte_seq, labels, pos_emb):
    z = self.init_z(byte_seq)              # Conv1d → 6×DilatedConv
    # 后续: temporal/topdown 投影层 + PC 推理
    h = self.model.norm(z[self.num_sub_layers])
    logits = self.model.lm_head(h)
    ce = cross_entropy(logits, labels, ignore_index=-100)
    return z, ce
```

### 为什么用 PC (预测编码)?

PC 模型有**两组权重**:

| 权重 | 角色 | 类比 |
|---|---|---|
| **Fast Weights** (`z` 节点) | 当前输入的"解释" | 工作记忆 |
| **Slow Weights** (模型参数) | 长期知识 | 长期记忆 |

新任务来时:
- **Slow Weights** 偏移 → 旧知识被覆盖 → 灾难性遗忘
- **回放 = 用旧 exemplars 的梯度把 Slow Weights 拉回来**
- **PC 推理** (infer_step) 更新 Fast Weights 来适配输入, 减少了对 Slow Weights 的依赖

---

## 7. 4 任务压力测试实战

### 实验设计

| Phase | 任务顺序 | 保护 | 衡量 |
|---|---|---|---|
| Phase 1 | A→B→C→D | ❌ 无回放 | 纯遗忘基线 |
| Phase 2 | A→B→C→D | ✅ MemoryBank+Sniffer | 保护效果 |

### 数据集

| 任务 | 领域 | 来源 | 样本量 |
|---|---|---|---|
| A — 日常对话 | 通用文本 | `pretrain_t2t_mini.jsonl` | 20,000 |
| B — 科技知识 | 考试/数学 | `lora_exam.jsonl` + `agent_rl_math.jsonl` | 20,000 |
| C — 医疗问诊 | 医疗 | `lora_medical.jsonl` | 20,000 |
| D — SFT 指令 | 结构化指令 | `sft_t2t_mini.jsonl` | 20,000 |

### 数据准备

```bash
python prepare_4task.py
```

输出: `datasets/task_{a,b,c,d}_{domain}_20k.jsonl`

### 运行压力测试

```bash
python forgetting_pressure_test.py ^
    --tasks datasets/task_a_daily_20k.jsonl datasets/task_b_tech_20k.jsonl ^
            datasets/task_c_medical_20k.jsonl datasets/task_d_sft_20k.jsonl ^
    --task-names "日常对话" "科技知识" "医疗问诊" "SFT指令" ^
    --epochs 1 --batch-size 16 --max-seq-len 256 --lr 3e-4 ^
    --threshold 1.5 --repair-steps 5 --check-interval 500
```

#### 参数说明

| 参数 | 作用 |
|---|---|
| `--tasks` | 任务数据路径 (按学习顺序, 至少 2 个) |
| `--task-names` | 任务显示名 (对应 --tasks) |
| `--epochs` | 每任务训练轮数 (默认 1) |
| `--batch-size` | 批次大小 (16 for 4GB VRAM) |
| `--lr` | 学习率 (3e-4 for PC 模型) |
| `--max-seq-len` | 最大序列长度 (256) |
| `--bank-size` | MemoryBank 每任务最大容量 (默认 2000) |
| `--exemplars` | 每任务存入 bank 的 exemplars 数量 (默认 500) |
| `--replay-ratio` | 每 N 步回放一次 (默认 5) |
| `--threshold` | Sniffer 触发阈值 (默认 1.5) |
| `--repair-steps` | Sniffer 每轮修复步数 (默认 5) |
| `--check-interval` | Sniffer 检测间隔 (默认 500) |

### 输出: 4×4 CE 矩阵

```
      日常对话   科技知识   医疗问诊   SFT指令
   ───────────────────────────────────────────
   A之后  1.2345
   B之后  3.4567   1.2345
   C之后  4.5678   3.4567   1.2345
   D之后  5.6789   4.5678   3.4567   1.2345
```

**解读规则**:
- 行 = 训练完第 i 个任务
- 列 = 在第 j 个任务上的 CE loss
- 对角线 (i=j) = 刚学完时的 loss
- 对角线右边为空 (未来任务尚未学习)
- **同一列从上往下看**: loss 越来越大 → 遗忘正在发生
- **Δ = 行[n-1][j] - 行[j][j]**: 全程遗忘幅度

### 完整运行脚本

```python
# 简化版: 两阶段比较
def main():
    # Phase 1: 无回放
    model = make_model(device)
    optim = make_optimizer(model, args.lr)
    p1_matrix = run_phase1(model, optim, task_paths, ...)

    # Phase 2: 有保护
    model = make_model(device)    # 重置模型 (公平对比)
    optim = make_optimizer(model, args.lr)
    p2_matrix = run_phase2(model, optim, task_paths, ...)

    # 比较
    print_conclusion(p1_matrix, p2_matrix, task_names)
```

---

## 8. 实验结果解读

### 我们实测的结果 (4 任务, GTX 1650 Ti)

```
PHASE 1 CE 矩阵 (无回放):
           日常对话    科技知识    医疗问诊    SFT指令
   ────────────────────────────────────────────────────
   A之后    1.1019
   B之后    2.2887    1.5676
   C之后    2.2132    1.6559    1.0243
   D之后    2.0851    1.6537    1.0976    0.9690

PHASE 2 CE 矩阵 (有回放):
           日常对话    科技知识    医疗问诊    SFT指令
   ────────────────────────────────────────────────────
   A之后    1.1019
   B之后    1.4836    1.5600
   C之后    1.6092    1.5422    1.0243
   D之后    1.5248    1.5687    1.1786    0.9690
```

### 分析

| 任务 | Phase 1 遗忘 (Δ) | Phase 2 遗忘 (Δ) | 改善 |
|---|---|---|---|
| 日常对话 (A) | 2.0851 - 1.1019 = **+0.98** | 1.5248 - 1.1019 = **+0.42** | **57% ↓** |
| 科技知识 (B) | 1.6537 - 1.5676 = **+0.09** | 1.5687 - 1.5600 = **+0.01** | 稳定 |
| 医疗问诊 (C) | 1.0976 - 1.0243 = **+0.07** | 1.1786 - 1.0243 = **+0.15** | 轻微上升 |

结论: **MemoryBank+Sniffer 显著减少了灾难性遗忘, 尤其在首任务 (日常对话) 上遗忘减少了 57%。**

### 为什么不是 100% 消除？

- `threshold=1.5` 相对宽松 — 只有当 loss 超过基线 50% 才触发修复
- 每个任务的 exemplars 只有 500 条 (bank_size=2000), 分布可能不够全面
- GTX 1650 Ti (4GB) 限制了模型大小和 batch size

---

## 9. 超参数调优指南

### MemoryBank 参数

| 参数 | 推荐范围 | 调高 | 调低 |
|---|---|---|---|
| `--bank-size` | 500~5000 | 更多记忆, 更多显存 | 省显存, 但可能漏掉关键样本 |
| `--exemplars` | 200~2000 | 覆盖更全面 | 减少存储开销 |
| `--replay-ratio` | 3~20 | 回放更频繁, 保护更强 | 新任务学得更快 |

### Sniffer 参数

| 参数 | 推荐范围 | 调高 | 调低 |
|---|---|---|---|
| `--threshold` | 1.2~2.0 | 减少误触发 | 更敏感, 但可能频繁中断 |
| `--repair-steps` | 5~20 | 修复更彻底 | 修复更快, 但可能不足 |
| `--check-interval` | 200~1000 | 减少检测开销 | 更及时检测遗忘 |

### 调优策略

```
遗忘严重 (Δ > 0.5)         → 降低 threshold, 增加 repair-steps
新任务学不动               → 降低 replay-ratio, 增加 check-interval
修复时来回震荡            → 降低 repair_lr_factor (默认 0.3 → 0.1)
检测开销太大 (训练变慢)    → 增加 check-interval (1000+)
```

---

## 附录: 常见问题

### Q1: 为什么第一个任务没有保护？

设计如此。第一个任务没有旧知识需要保护, 纯 CE 训练建立基线。从第二个任务开始, MemoryBank 中有 exemplars 了, Sniffer 才能工作。

### Q2: MemoryBank 和 Sniffer 谁管理回放？

合作关系:
- **平时**: MemoryBank 每隔 `replay_ratio` 步主动提供 exemplars 做 dopamine 加权回放
- **检测到遗忘时**: Sniffer 接管, 降 LR 并用 uniform 采样强制修复

### Q3: OfflineReplayer 什么时候用？

在任务间隙或训练完成后使用。它不是实时保护, 而是**额外的巩固步骤**——让模型自己生成合成数据来复习, 进一步巩固记忆。

### Q4: 支持多少个任务？

N 个。`forgetting_pressure_test.py` 接受任意数量的 `--tasks` 参数, 输出 N×N CE 矩阵。`MemoryBank` 和 `ForgettingSniffer` 都是 task-agnostic 的。

---

> **保持学习, 不要遗忘。**
