# Cyrilux 项目初始化知识

> 详细规则请参考同级指令文件。

---

## 项目身份

- **名称**: Cyrilux
- **本质**: 通用范式
- **输入**: 纯文本 `utf-8` → `[B, 2, S]` 双通道 (ch0=字节值 fp16, ch1=角色编码 0/1/2/3)
- **无 tokenizer、无词表、无位置编码、无 attention/transformer**

---

## 核心架构

简短一句话

```
byte_proj(Conv1D 2→H, k=13, causal) → 6× LocalConvBlock (dilation 1→32, RF=127)
    → RMSNorm → lm_head(vocab=256)
```

每层 `LocalConvBlock` = `Conv1D(k=3, causal)` + `SwiGLU MLP`(融合 `gate_up_proj`) + 残差。

---

## 训练

**5 阶段 `train_step()`:**

1. `forward_with_ce()` — 共享前向 (有梯度)
2. `spatiotemporal_infer()` — T 步推理 (π 可调)
3. `compute_*` — F_pred (π 加权) + CE_conv
4. `Dopamine.update(F)` → D → 3 级调制
5. `backward + step` — lr 调制

**Hebbian 引擎** (`local_updates.py`): 零 autograd, 覆盖全部权重类型:

- conv / swiglu / temporal / topdown / decoder / byte_proj
- `sparse_outer_product` top-k 稀疏, Oja 规则约束增长
- `compute_all_hebbian_updates()` 统一入口

---

## 硬规则

CLAUDE.md

---

## 参考指令文件

- `CLAUDE.md` — 主代理指令 (fp16 规则、架构守则、沟通偏好)
- `.github/instructions/专业术语.instructions.md` — 领域术语规范
- `.github/instructions/工具.instructions.md` — 工具链规范 (uv/ruff/git)
