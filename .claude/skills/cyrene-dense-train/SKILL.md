# Cyrene Dense PC Network

Load and train the dense (full-matmul) DensePCNet.

## Quick Start

```python
from model.dense import DensePCNet, DensePCConfig

dev = torch.device("cuda")
net = DensePCNet(DensePCConfig(d_l4=1024)).to(dev)

# forward
logits = net(byte_ids)  # [N, S, 256]

# learn
stats = net.learn(byte_ids, targets)  # Hebbian updates
```

## CLI Training

```sh
# Dense mode (full GPU matmul)
uv run cyrilux train --backend dense -d dataset/en_pure.jsonl --hidden-size 1024 -b 16 -o out/dense_model

# Sparse mode (event-driven, default)
uv run cyrilux train -d dataset/agent_rl_math.jsonl --hidden-size 64 -b 1 -o out/model50
```

## Checkpoint

Best checkpoint: `out/dense_10000.pt` (10000 steps, off-diag=0.39, 15MB VRAM)

```python
net = DensePCNet.load("out/dense_10000.pt")
```

## Key Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `d_l4` | 1024 | L4 width (main factor for off-diag) |
| `d_l2` | 384 | L2 width |
| `d_l3` | 384 | L3 width |
| `d_l5` | 256 | L5 width (LM head input) |
| `d_l6` | 128 | L6 width |
| `hebbian_base_eta` | 3e-4 | Learning rate |
| `column_dropout` | 0.25 | Column dropout ratio |

## Architecture

6-layer fully-connected predictive coding network:
- L0: byte one-hot + positional encoding (sin/cos, 0.5x amplitude)
- L4→L2→L3→L5→L6: feedforward matmuls with per-frame normalization
- Temporal: `z[t] += 0.1 * W_t @ z[t-1]` per layer
- k-WTA: 80% retention on L4/L2/L3, 100% on L5/L6
- LM Head: row-norm inference + column-Oja training
