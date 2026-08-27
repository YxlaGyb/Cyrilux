# 4bit 量化训练实战教程

> 基于 MiniMind 64M 模型 + GTX 1650 Ti (4GB VRAM) 的踩坑记录与方案总结

---

## 目录

1. [环境篇：Windows 下的深度学习环境](#1-环境篇windows-下的深度学习环境)
2. [魔鬼篇：PowerShell 静默杀进程](#2-魔鬼篇powershell-静默杀进程)
3. [4bit 训练：为什么不是 bitsandbytes](#3-4bit-训练为什么不是-bitsandbytes)
4. [QAT 方案：torchao Int4WeightOnly](#4-qat-方案torchao-int4weightonly)
5. [训练流程：从数据到模型](#5-训练流程从数据到模型)
6. [性能篇：GTX 1650 Ti 优化要点](#6-性能篇gtx-1650-ti-优化要点)
7. [完整代码解析](#7-完整代码解析)
8. [常见问题](#8-常见问题)

---

## 1. 环境篇：Windows 下的深度学习环境

### 1.1 包管理器：uv

```powershell
# 安装 uv
winget install --id=astral-sh.uv

# 创建虚拟环境
uv venv --python 3.11

# 激活
.venv\Scripts\activate

# 安装 PyTorch (CUDA 12.x)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 其余依赖
uv pip install transformers tqdm einops rich
```

**为什么要用 uv？**

- 比 pip 快 10-100 倍（并行下载 + 缓存）
- 自动解析依赖冲突
- 自带 `uv.lock` 锁文件，可复现环境

### 1.2 CUDA 版本匹配

```powershell
# 查看 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 查看显卡驱动支持的最高 CUDA
nvidia-smi
```

> ⚠️ **关键经验**：`nvidia-smi` 显示的 CUDA Version 是驱动支持的*最高版本*，不是当前实际使用的版本。实际版本由 PyTorch 决定。

### 1.3 确认 GPU 可用

```python
import torch
print(torch.cuda.is_available())      # True
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce GTX 1650 Ti
print(torch.cuda.get_device_properties(0).total_memory / 1e9)  # ~4.0 GB
```

---

## 2. 魔鬼篇：PowerShell 静默杀进程

### 2.1 症状

```powershell
python train.py
# 没有任何输出，exit code 1
```

跑了等于没跑，没有任何错误信息。

### 2.2 原因

PowerShell 在 `import datasets` 或任何加载 huggingface datasets C 扩展时，会因 segfault 静默杀死进程，**不输出任何错误信息**。

逐行定位法：

```powershell
python -c "print('step 1')"
python -c "import torch; print('step 2')"
python -c "import datasets; print('step 3')"  # ← 挂在这里
```

### 2.3 解决方案

**方案 A（推荐）：用 cmd.exe 包裹**

```powershell
cmd.exe /c ".venv\Scripts\python train.py"
```

cmd.exe 会正常打印 segfault 栈回溯或 Python traceback。

**方案 B（彻底）：零 datasets 依赖**

替换 huggingface `datasets.load_dataset` 为纯 `json.loads`：

```python
class _PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path, 'r', encoding='utf-8') as f:
            self.samples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(
            str(sample['text']),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels
```

JSONL 格式要求每行一个 `{"text": "..."}` 对象。

> **为什么这个方案更好？** 不仅绕过了 segfault，还减少了依赖。Ponytail 原则：不需要的依赖就是负债。

### 2.4 附：GBK 编码问题

Windows 中文系统默认编码 GBK，读取含中文注释的 `.py` 文件可能报 `UnicodeDecodeError`。

**解决方法**：在文件第一行或第二行加：

```python
# -*- coding: utf-8 -*-
```

---

## 3. 4bit 训练：为什么不是 bitsandbytes

### 3.1 bitsandbytes 安装

```powershell
uv pip install bitsandbytes
```

查看 CUDA 版本支持：

```
.venv\Lib\site-packages\bitsandbytes\libs\  # 只有 libbitsandbytes_cuda130.dll
```

### 3.2 环境变量

bitsandbytes 0.49.2 只内置了 cuda130 的 dll。如果你的 PyTorch 是 cuda132，需要：

```python
import os
os.environ['BNB_CUDA_VERSION'] = '130'  # 强制加载 cuda130 dll
```

### 3.3 无法用 bnb 做全量 4bit 训练

```python
from bitsandbytes.nn import Linear4bit
```

**问题**：`Linear4bit` 源码中 `Params4bit(requires_grad=False)` 写死不可训练。

```python
class Params4bit(torch.nn.Parameter):
    def __new__(cls, data=None, requires_grad=False, ...):  # 硬编码 False
```

如果把 `requires_grad` 改成 `True`，[`.cuda()`](command:_vscode.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22path%22%3A%22%2Fe%3A%5C%2FSystemShare%5C%2FDocuments%5C%2Fvirtuosov2%5C%2Ftrain_mvp.py%22%7D%5D) 时会崩溃。bnb 的 4bit 层设计目标就是**推理加速 + LoRA 微调**，不是全量预训练。

### 3.4 `Linear4bit` 权重初始化陷阱

```python
# ❌ 错误：weight.data 赋值会丢失 quant_state
new = bnb.nn.Linear4bit(256, 832, compute_dtype=torch.float16, quant_type='nf4')
new.weight.data = child.weight.data  # ← quant_state 被抹掉
new.bias = child.bias

# ✅ 正确：必须走 .cuda() 初始化路径
new = bnb.nn.Linear4bit(256, 832, ...)
new = new.to('cuda')  # ← 自动编码为真实 4bit 格式
```

`.cuda()` 会将 shape `[832, 256]` 的 fp16 权重压缩为 shape `[26624, 1]` 的 nf4 格式，同时创建 `quant_state`。

### 3.5 bitsandbytes 的正确用途

- **推理**：`Int8Params` / `Linear4bit` → 大幅减少显存
- **LoRA 微调**：冻结 4bit base，只训练 adapter
- **8bit Optimizer**：`bnb.optim.AdamW8bit`（可用且好用）

---

## 4. QAT 方案：torchao Int4WeightOnly

### 4.1 安装

```powershell
uv pip install torchao
```

验证：

```python
from torchao.quantization.qat import Int4WeightOnlyQATQuantizer
```

### 4.2 torchao 的三种 4bit 方案对比

| 方案 | 训练 | 速度 | 精度 | 适用场景 |
|------|------|------|------|----------|
| `Int4WeightOnlyQATQuantizer` | ✅ 全量训练 | ⚡ 接近 fp16 | ✅ 高 | **全量 4bit 预训练** |
| `IntXQuantizationAwareTrainingConfig` | ✅ 全量训练 | 🐢 慢 | ✅ 高 | 需要 activation 量化 |
| `Int4WeightOnlyConfig` | ❌ 仅推理 | 🚀 最快 | ⚠️ 略降 | 训完后的推理部署 |

**结论**：全量预训练选 `Int4WeightOnlyQATQuantizer`。

### 4.3 QAT 原理

QAT (Quantization Aware Training) = **在训练中模拟量化噪声**：

```
前向:
  fp16 权重 → fake quantize(weights) → int4 格式模拟 → dequantize → fp16 matmul
  
反向:
  loss → 梯度通过 fake quantize 的 straight-through estimator (STE) → 更新 fp16 权重
```

最终 `convert()` 后的模型：

```
fp16 权重 → real quantize → int4 格式 → 无 dequantize → 专门的 int4 matmul 内核
```

### 4.4 `Int4WeightOnlyQATQuantizer` 配置

```python
quantizer = Int4WeightOnlyQATQuantizer(
    groupsize=64,          # 分组大小，决定精度-速度权衡
    inner_k_tiles=4,       # 内部 tile 大小 (int4 matmul 内核参数)
    precision=torch.float16,      # 计算精度
    scales_precision=torch.bfloat16,  # scale 精度
)
```

**groupsize 为什么选 64？**

MiniMind 的 FFN 中间层维度是 `832 = 64 × 13`。如果默认 `groupsize=128`，`832 ÷ 128 = 6.5` 不能整除，部分实现会报 `RuntimeError: shape '[256, -1, 128]' is invalid for input of size 212992`。

```python
# 快速检查所有 Linear 维度能否被 groupsize 整除：
for n, m in model.named_modules():
    if isinstance(m, nn.Linear):
        print(f"{n}: {m.in_features}, {m.out_features}")
```

### 4.5 QAT 训练管线

```python
# 1. 创建模型（在 CPU 上）
model = MiniMindForCausalLM(config)

# 2. QAT prepare（必须在 CPU 上）
model = quantizer.prepare(model)

# 3. 移到 GPU
model = model.to('cuda')

# 4. 正常训练（和 fp16 完全一样的训练循环）
for step, (input_ids, labels) in enumerate(loader):
    with torch.cuda.amp.autocast(dtype=torch.float16):
        res = model(input_ids, labels=labels)
    loss = res.loss
    loss.backward()
    optimizer.step()

# 5. 训完后转纯 int4 推理格式
model = quantizer.convert(model)
torch.save(model.state_dict(), 'int4_model.pt')
```

> **`prepare()` 必须在 `.to(device)` 之前**，因为在 CPU 上替换 Linear 层更稳定。移到 GPU 后再 prepare 会触发奇怪的 CUDA kernel 错误。

---

## 5. 训练流程：从数据到模型

### 5.1 目录结构

```
virtuosov2/
├── model/
│   └── model_minimind.py    # MiniMind 架构定义
├── dataset/
│   └── pretrain_t2t_mini.jsonl  # ~1.2GB 预训练数据
├── train_mvp.py             # 4bit QAT 训练脚本
├── trainer_utils.py         # Logger, get_lr, setup_seed
└── out/                     # checkpoint 输出
```

### 5.2 模型配置

```python
lm_config = MiniMindConfig(
    hidden_size=256,         # 每层宽度
    num_hidden_layers=4,     # Transformer 层数
    use_moe=False,           # 不使用 MoE
)
```

这个配置的模型约 **4.98M 参数**（5M），非常适合 GTX 1650 Ti 快速验证。

### 5.3 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 8 | 单步样本数 |
| accum_steps | 4 | 梯度累积（有效 batch = 32） |
| max_seq_len | 128 | 序列长度 |
| learning_rate | 5e-4 | 初始学习率（余弦退火） |
| epochs | 1 | 数据集过一遍 |
| subset_size | 50000 | 仅用前 5 万条 |
| optimizer | AdamW | betas=(0.9, 0.95) |

### 5.4 梯度累积

```python
for step, (input_ids, labels) in enumerate(loader):
    with autocast_ctx:
        res = model(input_ids, labels=labels)
        loss = (res.loss + res.aux_loss) / accum_steps  # 除以累积步数

    scaler.scale(loss).backward()

    if (step + 1) % accum_steps == 0:  # 每 accum_steps 步更新一次
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

梯度累积模拟了更大的 batch size，对 4GB 显存的卡至关重要。

### 5.5 学习率调度

```python
# trainer_utils.py
def get_lr(current_step, total_steps, lr):
    """余弦退火"""
    return lr / 2 * (1.0 + math.cos(math.pi * current_step / total_steps))
```

### 5.6 Checkpoint 保存

```python
torch.save({
    'epoch': epoch,
    'step': step,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'loss': loss_item,
    'lm_config': lm_config,
}, 'checkpoint.pt')
```

**恢复训练**：

```python
ckpt = torch.load('checkpoint.pt', weights_only=False)
model.load_state_dict(ckpt['model_state'])
optimizer.load_state_dict(ckpt['optimizer_state'])
```

### 5.7 最终转换

训练完成后用 `quantizer.convert()` 转为纯 int4 推理格式：

```python
model = quantizer.convert(model)
torch.save(model.state_dict(), 'int4_model.pt')
```

转换后的模型：
- 权重存储为 int4（`torch.int32` 中打包）
- 前向使用 int4 专用 matmul 内核
- 无 fake quantize 开销 → 推理更快

---

## 6. 性能篇：GTX 1650 Ti 优化要点

### 6.1 硬件特性

| 指标 | 值 |
|------|-----|
| GPU | NVIDIA GeForce GTX 1650 Ti |
| 显存 | 4 GB GDDR6 |
| 架构 | Turing (无 Tensor Core) |
| 显存带宽 | ~128 GB/s |
| 最大短板 | **显存带宽** |

### 6.2 为什么 4bit 可能比 fp16 快

对于 GTX 1650 Ti 这样的带宽瓶颈卡：

```
fp16:   读取权重 2 bytes × 参数 + 计算 ≈ 带宽等待
int4:   读取权重 0.5 bytes × 参数 + 计算 ≈ 计算等待（不卡带宽）
```

4bit 的权重读取量降为 1/4，意味着**同样的带宽可以喂给计算单元更多数据**。这是为什么 4bit QAT 可能比 fp16 更快的原因。

### 6.3 实际观测

| 方案 | steps/sec (GTX 1650 Ti) | 显存占用 |
|------|------------------------|----------|
| float16 (基线) | ~15 | ~2.6 GB |
| torchao Int4WeightOnly QAT | ~17-20 | ~1.2 GB |
| bitsandbytes Linear4bit | ❌ 不可训练 | ~1.0 GB |
| torchao IntXQuantizationAwareTraining (含 activation) | ~8 | ~1.5 GB |

> 数据基于 5M 参数模型，batch_size=8, seq_len=128. 实际速度因系统负载而异。

### 6.4 优化技巧

**1. `non_blocking=True`**

```python
input_ids = input_ids.to(device, non_blocking=True)
labels = labels.to(device, non_blocking=True)
```

**2. `pin_memory=True`**

```python
loader = DataLoader(ds, batch_size=8, pin_memory=True)
```

**3. `set_to_none=True`**

```python
optimizer.zero_grad(set_to_none=True)  # 比 zero_grad() 快
```

**4. GradScaler**

```python
scaler = torch.cuda.amp.GradScaler(enabled=True)
```

**5. 梯度裁剪**

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

---

## 7. 完整代码解析

### 7.1 `train_mvp.py` 结构

```
train_mvp.py
├── 导入依赖
├── _PretrainDataset 类 (纯 json，零 datasets)
├── train() 函数
│   ├── 模型配置
│   ├── 模型创建 + QAT prepare
│   ├── 数据加载
│   ├── 训练循环
│   │   ├── 学习率更新
│   │   ├── forward (autocast)
│   │   ├── backward (scaler)
│   │   ├── 梯度累积优化
│   │   └── checkpoint 保存
│   └── 最终转换 + 保存
└── 入口 (setup_seed + train)
```

### 7.2 关键代码段

```python
# 1. QAT prepare (必须在 CPU)
model = MiniMindForCausalLM(lm_config)
quantizer = Int4WeightOnlyQATQuantizer(groupsize=64, precision=torch.float16)
model = quantizer.prepare(model)
model = model.to(device)

# 2. AdamW + GradScaler
optimizer = optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95))
scaler = torch.cuda.amp.GradScaler(enabled=True)

# 3. 训练步
with torch.cuda.amp.autocast(dtype=torch.float16):
    res = model(input_ids, labels=labels)
    loss = (res.loss + res.aux_loss) / accum_steps

# 4. 转换为 int4 推理
model = quantizer.convert(model)
```

---

## 8. 常见问题

### Q1: QAT 训练比 fp16 慢？

QAT 本身有 fake quantize 开销，但 weight-only QAT 开销很小（只量化权重，不量化激活）。在带宽瓶颈卡上，4bit 减少的内存读取可以抵消甚至超越 fake quantize 的额外计算。

如果还是慢→检查 groupsize 是否能整除所有 Linear 维度→尝试增大 batch_size。

### Q2: `RuntimeError: shape invalid for input of size N`

```python
RuntimeError: shape '[256, -1, 128]' is invalid for input of size 212992
```

**原因**：`groupsize=128` 不能整除中间层的某个维度（如 832）。

**解决**：设 `groupsize=64`。

### Q3: `KeyError: <class '...intx_quantization_aware_training'>`

**原因**：用错了 torchao API。`intx_quantization_aware_training` 是一个类，不是 config 对象，不能直接传给 `quantize_()`。

**解决**：用 `IntXQuantizationAwareTrainingConfig` 包裹后再传，或者直接用 `Int4WeightOnlyQATQuantizer`。

### Q4: 读文件 UnicodeDecodeError

```python
UnicodeDecodeError: 'gbk' codec can't decode byte 0x9a
```

**解决**：文件加 `# -*- coding: utf-8 -*-`，读取时显式指定编码：

```python
with open(path, 'r', encoding='utf-8') as f:
    ...
```

### Q5: checkpoint 保存后无法加载

bitsandbytes 的 `Linear4bit` 权重存储为特殊格式，用普通 `.half().cpu()` 会丢失 `quant_state`。

**解决**：直接 `torch.save(model.state_dict(), path)`，不要手动转换 dtype。

### Q6: 如何加载 QAT 后的模型？

```python
# 推理模式加载
model = MiniMindForCausalLM(config)
quantizer = Int4WeightOnlyQATQuantizer(...)
model = quantizer.prepare(model)   # 重建 QAT 层结构
state = torch.load('qat_final.pt')['model_state']
# 去掉 _orig_mod. 前缀 (如果用了 torch.compile)
state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
model.load_state_dict(state, strict=False)
model = quantizer.convert(model)   # 转 int4
```

### Q7: 为什么推荐 cmd.exe 而不是 PowerShell？

PowerShell 在加载 C 扩展（huggingface datasets、bitsandbytes 等）发生 segfault 时会**静默退出**，不打印任何错误。cmd.exe 不受此影响。

如果必须用 PowerShell，可以用：

```powershell
$LASTEXITCODE  # 检查退出码
```

但**没有错误信息**仍然是致命问题。

---

## 附录：依赖清单

```
# 核心
torch==2.12.0+cu132
transformers==4.49.0
torchao==0.17.0

# 数据
tqdm
einops
rich

# 量化 (可选)
bitsandbytes==0.49.2  # 需要 BNB_CUDA_VERSION=130
```

安装命令：

```powershell
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install transformers tqdm einops rich torchao bitsandbytes
```

---

> **Ponytail 原则回顾**：① 不需要的别写 ② 已有的别重造 ③ 标准库够用就用 ④ 一行能解决别写五条 ⑤ 最简代码 = 最快跑通。
