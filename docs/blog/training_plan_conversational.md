# virtuosov2 训练到正常交流 — 完整策划

> 基线: PCLocalDynamicMiniMind (conv), hidden=256, 4 层, fp16 原生, temp_loss 已集成
> 数据格式: 双通道 [B,2,S] (ch0=字节值, ch1=角色编码), 字节级, 无 tokenizer

---

## 1. 核心理念

**字节级 PC 模型的交流能力 = 对 UTF-8 字节序列的压缩质量 + 生成连贯性。**

"正常交流"的物理含义:
- **短期**: 生成的 UTF-8 字节流能解码为有效文本 (无乱码, 基本语法正确)
- **中期**: 能维持局部语义连贯 (5-10 个词内话题一致)
- **长期**: 能完成简单问答, 指令遵循, 角色扮演 (SFT 效果)

**关键杠杆 (按重要性排序):**
1. 数据量 × 质量 (字节级模型需要更多数据, 因为 256 维分类空间比 tokenizer 大)
2. temp_loss 权重 (控制 backbone 压缩倾向 vs 预测精度)
3. T_infer 调度 (PC 推理步数决定生成时的计算深度)
4. 学习率 + batch_size (256-dim 字节空间需要稳定梯度)

---

## 2. 数据策略

### 2.1 建议数据构成 (总量 ~500MB UTF-8 文本)

| 阶段 | 数据源 | 建议量 | 作用 |
|------|--------|--------|------|
| P1 语料预训练 | 通用中文语料 (维基/新闻/博客) | ~300MB | UTF-8 字节分布建模, 基础语法 |
| P2 对话预训练 | 多轮对话数据 (日常闲聊) | ~100MB | 学习对话格式, 角色切换 |
| P3 指令微调 | SFT 数据 (问答/指令) | ~50MB | 指令遵循能力 |
| P4 特定能力 | Agent/RL/Math/医疗 | ~50MB | 垂直领域泛化 |

### 2.2 数据准备要求

**不需要标数据**, 原样文本即可。DualChannelDataset 会自动转换 UTF-8 到双通道张量。

```bash
# 直接使用纯文本 jsonl (最简单的格式)
{"text": "今天天气真不错，适合出去散步。"}
{"text": "你好，请问有什么可以帮助你的？"}

# 或 conversations 格式 (角色自动编码)
{"conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"}
]}
```

**数据放在 `dataset/` 目录下, 不要用 .jsonl 格式的文件路径在 --subset 参数控制加载量。**
建议用 `prepare_tasks.py` 按领域自动切分。

### 2.3 预训练语料获取 (无需标注)

推荐源 (纯文本, 无需处理):
- **OSCAR** 中文子集 — 互联网爬取, 自动清洗
- **CLUECorpus2020** — 中文新闻+百科+论坛混合
- **WuDaoCorpora** — 悟道中文语料
- **Wikipedia 中文 dump** — 结构化百科文本

以上全部可用 `wget` 下载原始文本后直接转为 `{"text": "..."}` 格式的 jsonl。

---

## 3. 训练策略 — 四阶段课程

### Phase 1: 字节分布建模 (基础语法)

**目标**: CE loss < 2.0 (~PPL 7.4), 生成无乱码

| 参数 | 值 | 理由 |
|------|----|------|
| batch_size | 64 | 更大的 batch 稳定字节分布学习 |
| lr | 3e-4 → 1e-4 (cosine) | 标准 AdamW 调度 |
| max_seq_len | 128 | 固定, 与架构一致 |
| T_infer | 1 | 浅层 PC 推理足够 |
| gamma | 0.05 | 弱 PC 压力, 让 backbone 先学好 |
| temp_loss_weight | 0.05 | 弱压缩正则, 避免干扰 CE |
| learning_mode | hybrid | 梯度为主 |
| 数据 | ~200MB 纯文本 | 任意中文语料 |
| 步数估计 | ~100K steps | |

**里程碑**: `python main.py eval --checkpoint out/...` CE loss 稳定 < 2.0

### Phase 2: 对话预训练 (交流格式)

**目标**: CE loss < 1.6 (~PPL 5.0), 生成含角色切换的多轮文本

| 参数 | Phase 1 → Phase 2 变化 |
|------|------------------------|
| T_infer | 1 → 2 |
| gamma | 0.05 → 0.10 |
| temp_loss_weight | 0.05 → 0.15 |
| lr | 1e-4 → 1e-4 (新阶段重置) |
| 数据 | 文本 → 对话 (conversations 格式) |
| 步数估计 | ~50K steps |

**关键**: DualChannelDataset 的角色编码 (ch1) 让模型学到 user/assistant 交替模式。
temp_loss 权重提高 → backbone 被迫产出自预测的 z, 这是长序列连贯性的前提。

### Phase 3: 指令微调 + 持续学习 (正常交流)

**目标**: CE loss < 1.2 (~PPL 3.3), 首次生成测试能回答简单问题

| 参数 | Phase 2 → Phase 3 变化 |
|------|------------------------|
| T_infer | 2 → 3 |
| gamma | 0.10 → 0.15 |
| temp_loss_weight | 0.15 → 0.20 |
| lr | 1e-4 → 5e-5 (微调模式) |
| 数据 | 对话 → SFT (问答格式) |
| 步数估计 | ~30K steps |

**开启所有持续学习模块**:
- ICM + ConceptDiscovery + MemoryGate: 自动发现高频模式
- ConsolidationPipeline + SleepEngine: 巩固对话模板
- AttractorLandscape: 监控吸引子健康度

### Phase 4: 特定能力 (Agent/RL/医疗)

**目标**: 不遗忘已有能力的前提下, 学会新领域

**使用 `prepare_4tasks()` 方式**:
```bash
# 4 个领域分别作为独立 task
uv run python main.py train datasets/task_a_daily_20k.jsonl \
    --task-id daily --out-dir out_daily
uv run python main.py train datasets/task_b_tech_20k.jsonl \
    --checkpoint out_daily/task_a_final.pt --task-id tech --out-dir out_tech
uv run python main.py train datasets/task_c_medical_20k.jsonl \
    --checkpoint out_tech/task_b_final.pt --task-id medical --out-dir out_medical
```

**连续学习参数**:
```bash
--replay-ratio 3  --bank-size 3000  --sniff-interval 150
```

---

## 4. 超参数决策树

```
batch_size 选择:
  48-64: 稳定梯度, 推荐 for Phase 1-2
  32: 大模型变大时的后备方案
  16: 仅 autonomous mode (内存受限)

learning_mode 选择:
  hybrid: 默认, 梯度为主 + 可选 Hebbian 辅助
  bp_free: 仅 Phase 3+(微调)可使用, 纯 Hebbian 训练 temporal_proj/topdown_proj

T_infer 调度 (训练时):
  Phase 1: T=1  (浅层推理)
  Phase 2: T=2  (深层推理开始)
  Phase 3: T=3-4 (全量推理)

T_infer 调度 (生成时):
  eval: T=2   (平衡质量和速度)
  交互: T=3-4 (最佳质量)

temp_loss_weight 调度:
  0.05 → 0.15 → 0.20 → 0.30 (逐步增加压缩压力)
  若 CE loss 不降: 降低 temp_loss_weight
  若生成文本重复/死循环: 提高 temp_loss_weight
```

---

## 5. 评估标准

### 5.1 定量指标 (每 500 步计算)

| 指标 | 合格线 | 良好 | 优秀 |
|------|--------|------|------|
| CE loss | < 2.0 | < 1.5 | < 1.0 |
| Perplexity | < 7.4 | < 4.5 | < 2.7 |
| 生成文本无乱码 | > 90% | > 98% | 100% |
| 回答长度 | > 3 bytes | > 20 bytes | > 50 bytes |

### 5.2 定性指标 (人工判断)

```bash
# 生成测试
uv run python -c "
from core.evaluation import generate_text
from model.pc_layers import PCLocalDynamicMiniMind, load_pc_checkpoint
model = load_pc_checkpoint('out/final.pt', PCLocalDynamicMiniMind).half().cuda()
for prompt in ['你好', '什么是人工智能', '1+1=']:
    print(generate_text(model, prompt, max_new_tokens=50))
"
```

测试 prompt 清单:
1. `你好` — 基础响应
2. `今天天气怎么样` — 开放域
3. `1+1=` → 期望 `2` — 事实性
4. `请写一首关于春天的诗` — 创造性
5. `你能做什么` — 自我认知

### 5.3 持续学习遗忘检测

```bash
uv run python main.py eval --checkpoint out/... --forgetting-test
```
`forgetting_log.json` 应显示每个旧任务遗忘率 < 20%。

---

## 6. 预期资源消耗

### 单次训练运行

| 阶段 | 数据量 | Steps | 时间估计 (RTX 3090) | GPU 显存 |
|------|--------|-------|---------------------|----------|
| Phase 1 | 200MB | 100K | ~4-6 小时 | ~3.5GB |
| Phase 2 | 100MB | 50K | ~2-3 小时 | ~3.5GB |
| Phase 3 | 50MB | 30K | ~1-2 小时 | ~3.5GB |
| Phase 4 | 4×20K | 20K×4 | ~4 小时 | ~3.5GB |

**总计: ~12-15 小时** 从零到可交流。

### 生成速度

| 模式 | 速度 |
|------|------|
| T_infer=0 (纯前向) | ~1000 tokens/s |
| T_infer=2 | ~300 tokens/s |
| T_infer=4 | ~150 tokens/s |

---

## 7. 推荐执行路径

### 最短路径 (验证可行性)

```bash
# 1. Phase 1: 字节预训练 (任意大文本)
uv run python main.py train dataset/pretrain_t2t_mini.jsonl \
    --batch-size 64 --lr 3e-4 --T-infer 1 --gamma 0.05 \
    --temp-loss-weight 0.05 --out-dir out_p1 \
    --subset 5000  # 先 5000 条验证

# 2. 评估生成效果
uv run python -c "
from core.evaluation import generate_text
from model.pc_layers import PCLocalDynamicMiniMind, load_pc_checkpoint
model = load_pc_checkpoint('out_p1/task_a_final.pt', PCLocalDynamicMiniMind).half().cuda()
print(generate_text(model, '你好', max_new_tokens=100))
"

# 3. Phase 2: 对话预训练
uv run python main.py train dataset/sft_t2t_mini.jsonl \
    --checkpoint out_p1/task_a_final.pt \
    --batch-size 48 --lr 1e-4 --T-infer 2 --gamma 0.10 \
    --temp-loss-weight 0.15 --out-dir out_p2 --subset 2000

# 4. Phase 3: SFT
uv run python main.py train dataset/agent_rl_math.jsonl \
    --checkpoint out_p2/task_a_final.pt \
    --batch-size 32 --lr 5e-5 --T-infer 3 --gamma 0.15 \
    --temp-loss-weight 0.20 --out-dir out_p3 --subset 1000

# 5. 最终测试
uv run python -c "
from core.evaluation import generate_text
from model.pc_layers import PCLocalDynamicMiniMind, load_pc_checkpoint
model = load_pc_checkpoint('out_p3/task_a_final.pt', PCLocalDynamicMiniMind).half().cuda()
for prompt in ['你好', '1+1=', '什么是人工智能']:
    print(f'Q: {prompt}')
    print(f'A: {generate_text(model, prompt, max_new_tokens=50)}')
    print()
"
```

### 完整路径 (生产级)

```bash
# Phase 1: 全量通用语料 (~200MB)
uv run python main.py train dataset/pretrain_t2t.jsonl \
    --batch-size 64 --lr 3e-4 --T-infer 1 --gamma 0.05 \
    --temp-loss-weight 0.05 --out-dir out_full_p1

# Phase 2: 对话全量 (~100MB)
uv run python main.py train dataset/sft_t2t.jsonl \
    --checkpoint out_full_p1/task_a_final.pt \
    --batch-size 48 --lr 1e-4 --T-infer 2 --gamma 0.10 \
    --temp-loss-weight 0.15 --out-dir out_full_p2

# Phase 3: 多任务持续学习
uv run python main.py train dataset/agent_rl_math.jsonl \
    --checkpoint out_full_p2/task_a_final.pt \
    --batch-size 32 --lr 5e-5 --T-infer 3 --gamma 0.15 \
    --temp-loss-weight 0.20 --out-dir out_full_p3

# Phase 4: 持续学习 (含遗忘嗅探 + 巩固 + 睡眠)
uv run python -c "
from core.prepare_tasks import prepare_4tasks
prepare_4tasks('dataset', 'dataset', n_per_task=20000)
"
# 依次训练 4 个任务 (自动遗忘防护)
for task in daily tech medical sft; do
    ckpt_flag=""
    [ -f out_cl/task_*_final.pt ] && ckpt_flag="--checkpoint $(ls out_cl/task_*_final.pt | tail -1)"
    uv run python main.py train dataset/task_${task}_20k.jsonl \
        $ckpt_flag \
        --batch-size 32 --lr 3e-5 --T-infer 3 --gamma 0.15 \
        --temp-loss-weight 0.25 --replay-ratio 3 \
        --out-dir out_cl
done
```

---

## 8. 故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| CE loss 不降 (卡在 >3) | temp_loss 过强压制 backbone | 降低 temp_loss_weight 到 0.01 |
| 生成全是乱码 | 字节分布未学够 | 增加 Phase 1 数据/步数 |
| 生成重复死循环 | temp_loss 不足, z 无时序结构 | 提高 temp_loss_weight |
| 生成很短 1-2 bytes | 模型学到"少输少错" | 提高 T_infer, 增加 gamma |
| 遗忘率 > 30% | replay/consolidation 不足 | bank_size=5000, replay_ratio=2 |
| 多轮对话答非所问 | 角色编码未有效利用 | 确认 ch1 角色编码正确 |
| GPU OOM | batch_size 过大 | 减半 batch_size, 或降低 max_seq_len |

---

## 9. 当前代码状态与下一步

### ✅ 已就绪
- 完整训练流水线 (train_step, bp_free, DataLoader)
- temp_loss 核心实现 (Phase 0a/0b)
- 持续学习全套 (ICM/WM/Consolidation/Sleep)
- 评估模块 (PPL, generate_text, eval_self_supervised)
- 多任务 pipeline (prepare_4tasks, prepare_hetero)

### 🔧 可能需要调整
1. **temp_loss_weight 默认值**: 当前 0.1, Phase 1 可能需要 0.05
2. **T_infer 生成时调度**: 当前生成函数支持 T=0/2/3+, 需要实验最优值
3. **数据集准备脚本**: 下载外部语料 → 转 jsonl 格式的一键脚本
4. **自动评估脚本**: CE + PPL + generate_text 一次性报告

---

## 10. 一句话总结

**先用 `subset=5000` 跑最短路径验证到能出字, 确认流水线通了之后, 再用全量数据跑四阶段课程到能正常交流。全程约 12-15 小时 GPU 时间。**

是否要我现在帮你:
1. 跑最短路径验证 (subset=5000, 三阶段串联)
2. 或者先调整 temp_loss_weight 默认值
3. 或者写一个全自动训练脚本来跑四阶段
