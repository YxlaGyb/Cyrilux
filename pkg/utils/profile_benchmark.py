#!/usr/bin/env python3
"""PC+Hebbian 训练剖析基准 — 不改一行现有代码，只测量。

测量 train_step() 中每个阶段的实际 GPU 时间，用 torch.cuda.Event 精确计时。
报告格式: 每个阶段 ms/step + 占比。
"""

import os, sys, math

import torch
import torch.nn.functional as F

from model.core.train import TrainingConfig
from model.pc.local_updates import (
    compute_all_hebbian_updates,
    apply_hebbian_updates,
    compute_modulators,
    compute_precision_scales,
    compute_lambda,
)
from model.pc.pc_layers import CyrenePC
from model.model_cyrene import CyreneConfig

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

# ═══════════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════════

WARMUP_STEPS = 3
PROFILE_STEPS = 20
BATCH_SIZE = 48
SEQ_LEN = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 4

cfg = TrainingConfig(
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_LAYERS,
    batch_size=BATCH_SIZE,
    max_seq_len=SEQ_LEN,
    lr=3e-4,
    hebbian_infer_T=1,
    infer_adaptive_T=True,
    infer_min_T=1,
    infer_convergence_threshold=0.05,
    infer_patience=2,
    gamma=0.1,
    hebbian_base_eta=3e-4,
    oja_alpha=0.05,
    synaptic_normalize=True,
    consolidation_pipeline_interval=5,
    enable_world_model=True,
    enable_intrinsic_motivation=True,
    enable_salience_gating=True,
    enable_neurogenesis=True,
    enable_consolidation_pipeline=True,
    enable_deep_sleep=True,
    sleep_consolidation=True,
    sparse_outer_k=0,
    synaptic_competition_k=0,
    hebbian_subsample_ratio=0.1,
    hebbian_eps_gate=0.0,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

# ═══════════════════════════════════════════════════════════════════
#  构建模型
# ═══════════════════════════════════════════════════════════════════

lm_cfg = CyreneConfig(hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_LAYERS)
model = CyrenePC(lm_cfg).half().to(device)
model.train()

# 注入依赖阈值门控参数 (同 _build_model)
model._act_ema_decay = cfg.act_ema_decay
model._act_target_ratio = cfg.act_target_ratio
model._act_homeo_rate = cfg.act_homeo_rate
model._act_energy_cost = cfg.act_energy_cost
init_th = (
    torch.ones(model.num_sub_layers, model.config.hidden_size) * cfg.act_threshold_init
)
model.register_buffer("_act_threshold", init_th.to(device))

# BCM 状态
from model.pc.local_updates import BCMState

bcm_state = BCMState(n_layers=24, tau=0.01).to(device)

# 环境设置
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True

# ═══════════════════════════════════════════════════════════════════
#  Dummy 数据 — 双通道格式: [B, 2, S] (ch0=字节值, ch1=角色编码)
# ═══════════════════════════════════════════════════════════════════

byte_seq = torch.randint(
    0, 256, (BATCH_SIZE, 2, SEQ_LEN), device=device, dtype=torch.float16
)
labels = byte_seq[:, 0, :].clone().long()

# ═══════════════════════════════════════════════════════════════════
#  计时器工具
# ═══════════════════════════════════════════════════════════════════


class CudaTimer:
    """用 torch.cuda.Event 精确测量 GPU 时间。"""

    def __init__(self):
        self.events = {}  # name -> list of (start, end)

    def synced_start(self, name):
        torch.cuda.synchronize()
        if name not in self.events:
            self.events[name] = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self.events[name].append([start, end])

    def end_last(self, name):
        end = self.events[name][-1][1]
        end.record()

    def mean_ms(self, name):
        totals = []
        for s, e in self.events.get(name, []):
            torch.cuda.synchronize()
            totals.append(s.elapsed_time(e))
        return sum(totals) / len(totals) if totals else 0.0

    def summary(self):
        total = sum(self.mean_ms(n) for n in self.events)
        parts = []
        for name in sorted(self.events.keys()):
            ms = self.mean_ms(name)
            pct = ms / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            parts.append((ms, pct, name, bar))
        parts.sort(key=lambda x: -x[0])
        bar = ""
        print(f"\n{'=' * 70}")
        print(f"  GPU Timing Breakdown (avg over {PROFILE_STEPS} steps)")
        print(f"{'=' * 70}")
        print(f"  {'Phase':<40s} {'ms':>8s} {'%':>6s}")
        print(f"  {'-' * 56}")
        for ms, pct, name, bar in parts:
            print(f"  {name:<40s} {ms:>7.2f}ms {pct:>5.1f}%")
        print(f"  {'-' * 56}")
        print(f"  {'TOTAL':<40s} {total:>7.2f}ms 100.0%")
        print(f"  {' ' * 40} {'=' * 50}")
        print(f"  {bar}")  # noqa
        print()
        # Steps/sec
        print(
            f"  Throughput: {1000 / total:.1f} steps/sec, "
            f"{(BATCH_SIZE * SEQ_LEN) / (total / 1000) / 1e6:.2f}M tokens/sec"
        )
        return parts


timer = CudaTimer()

# ═══════════════════════════════════════════════════════════════════
#  预热
# ═══════════════════════════════════════════════════════════════════

print(f"\nWarming up ({WARMUP_STEPS} steps)...")
for _ in range(WARMUP_STEPS):
    # Phase 1: init_z + 缓存 SwiGLU 前激活
    model._hebbian_cache_enable()
    with torch.no_grad():
        z_init = model.init_z(byte_seq)
    # Phase 2: 推理 (T步)
    pos_emb = model.get_position_embeddings(SEQ_LEN, device)
    with torch.no_grad():
        z_conv, errors_hist, F_hist, _, ε_list = model.spatiotemporal_infer(  # type: ignore[assignment]
            z_init,
            pos_emb,
            gamma=cfg.gamma,
            T=cfg.hebbian_infer_T,
            return_errors=True,
            return_pred_loss=False,
            ach_value=0.5,
            return_ε=True,
            adaptive_T=cfg.infer_adaptive_T,
            convergence_threshold=cfg.infer_convergence_threshold,
            patience=cfg.infer_patience,
            min_T=cfg.infer_min_T,
            skip_bottom_up=True,
        )
    hebbian_cache = model._hebbian_cache_disable_and_collect()
    # Phase 3: 调制
    uncertainty = 0.5
    D, ACh_val, modulation = compute_modulators(
        F_hist[-1] if F_hist else 0.0, float("inf"), uncertainty, cfg
    )
    λ = compute_lambda(0, cfg.hebbian_lambda_decay, cfg.hebbian_lambda_min)
    π_list = compute_precision_scales(ε_list, ACh_val, D, cfg)
    # Phase 4: target
    target_onehot = F.one_hot(
        labels[:, 1:].long().clamp(0, 255), num_classes=256
    ).half()
    # Phase 5: Hebbian (使用缓存)
    updates = compute_all_hebbian_updates(
        ε_list,
        z_init,
        byte_seq,
        model,
        cfg,
        D=D,
        ACh=ACh_val,
        modulation=modulation,
        λ=λ,
        decoder=model.decoder,
        lm_head=model.model.lm_head,
        target_byte_embed=target_onehot,
        oja_alpha=cfg.oja_alpha,
        bcm_state=bcm_state,
        verbose=False,
        hebbian_cache=hebbian_cache,
    )
    apply_hebbian_updates(updates, model, synaptic_normalize=True, target_norm=0.0)

torch.cuda.synchronize()
print("Warmup done.\n")

# ═══════════════════════════════════════════════════════════════════
#  逐阶段剖析
# ═══════════════════════════════════════════════════════════════════

print(f"Profiling {PROFILE_STEPS} steps...")

for step in range(PROFILE_STEPS):
    # ── Phase 0: 数据 (不计时) ──
    seq_len = byte_seq.size(-1)
    pos_emb = model.get_position_embeddings(seq_len, device)

    # ── Phase 1: init_z + 缓存 SwiGLU 前激活 ──
    timer.synced_start("Phase 1: init_z")
    model._hebbian_cache_enable()
    with torch.no_grad():
        z_init = model.init_z(byte_seq)
    timer.end_last("Phase 1: init_z")

    # ── temp_loss 诊断 (Phase 1 子阶段, 单独计) ──
    timer.synced_start("Phase 1b: temp_loss")
    bp_temp_loss = 0.0
    if hasattr(model, "temporal_proj") and seq_len > 1:
        n_layers = len(model.temporal_proj)
        tl_acc = torch.tensor(0.0, device=device)
        for layer_i in range(n_layers):
            z_ℓ = z_init[layer_i + 1]
            if z_ℓ.size(1) > 1:
                z_proj = model.temporal_proj[layer_i](z_ℓ[:, :-1, :])
                tl = 0.5 * (z_proj - z_ℓ[:, 1:, :]).pow(2).mean()
                tl_acc = tl_acc + tl.clamp(max=100.0)
        bp_temp_loss = (tl_acc / n_layers).item() if n_layers > 0 else 0.0
    timer.end_last("Phase 1b: temp_loss")

    # ── Phase 2: PC 推理 ──
    timer.synced_start("Phase 2: spatiotemporal_infer")
    uncertainty = 0.3 + 0.4 * math.sin(step * 0.5)
    ACh = float(
        torch.sigmoid(torch.tensor(-uncertainty + cfg.hebbian_ach_beta_0)).item()
    )
    with torch.no_grad():
        z_conv, errors_hist, F_hist, _, ε_list = model.spatiotemporal_infer(  # type: ignore[assignment]
            z_init,
            pos_emb,
            gamma=cfg.gamma,
            T=cfg.hebbian_infer_T,
            return_errors=True,
            return_pred_loss=False,
            ach_value=ACh,
            return_ε=True,
            adaptive_T=cfg.infer_adaptive_T,
            convergence_threshold=cfg.infer_convergence_threshold,
            patience=cfg.infer_patience,
            min_T=cfg.infer_min_T,
            skip_bottom_up=True,
        )
    hebbian_cache = model._hebbian_cache_disable_and_collect()
    F_curr = F_hist[-1] if F_hist else 0.0
    timer.end_last("Phase 2: spatiotemporal_infer")

    # ── Phase 3: 调制信号 ──
    timer.synced_start("Phase 3: modulators")
    D, ACh_val, modulation = compute_modulators(F_curr, float("inf"), uncertainty, cfg)
    λ = compute_lambda(step, cfg.hebbian_lambda_decay, cfg.hebbian_lambda_min)
    π_list = compute_precision_scales(ε_list, ACh_val, D, cfg)
    timer.end_last("Phase 3: modulators")

    # ── Phase 4: Decoder 目标 ──
    timer.synced_start("Phase 4: decoder_target")
    target_onehot = F.one_hot(
        labels[:, 1:].long().clamp(0, 255), num_classes=256
    ).half()
    timer.end_last("Phase 4: decoder_target")

    # ── Phase 5a: compute_hebbian (使用缓存) ──
    timer.synced_start("Phase 5a: compute_hebbian")
    updates = compute_all_hebbian_updates(
        ε_list,
        z_init,
        byte_seq,
        model,
        cfg,
        D=D,
        ACh=ACh_val,
        modulation=modulation,
        λ=λ,
        decoder=model.decoder,
        lm_head=model.model.lm_head,
        target_byte_embed=target_onehot,
        oja_alpha=cfg.oja_alpha,
        bcm_state=bcm_state,
        verbose=False,
        hebbian_cache=hebbian_cache,
    )
    timer.end_last("Phase 5a: compute_hebbian")

    timer.synced_start("Phase 5b: apply_hebbian")
    apply_hebbian_updates(updates, model, synaptic_normalize=True, target_norm=0.0)
    timer.end_last("Phase 5b: apply_hebbian")

    # ── Phase 5.6: 神经发生 ──
    timer.synced_start("Phase 5c: neurogenesis")
    # neurogenesis 不是必需的 — 这里省略 (不影响 main timing)
    timer.end_last("Phase 5c: neurogenesis")

    # ── Phase 6: CE 诊断 ──
    timer.synced_start("Phase 6: ce_diag")
    ce_diag = model.compute_ce_loss(z_conv, labels).item()
    timer.end_last("Phase 6: ce_diag")

    if (step + 1) % 5 == 0:
        print(
            f"  step {step + 1}/{PROFILE_STEPS}  "
            f"F={F_curr:.2f}  CE={ce_diag:.4f}  D={D:.3f}"
        )

torch.cuda.synchronize()

# ═══════════════════════════════════════════════════════════════════
#  报告
# ═══════════════════════════════════════════════════════════════════

timer.summary()

# 额外子阶段: 分拆 Phase 5 (手动触发子函数计时)
# 使用正确签名: compute_hebbian_temporal/topdown 接受 ε_list, z_init (全列表)
print("\n── Phase 5 内部子函数 ──")
bcm_state2 = BCMState(n_layers=24, tau=0.01).to(device)

timer2 = CudaTimer()
# 额外跑一步，单独计每个子函数
with torch.no_grad():
    z_init2 = model.init_z(byte_seq)
pos_emb2 = model.get_position_embeddings(SEQ_LEN, device)
with torch.no_grad():
    z_conv2, _, F_hist2, _, ε_list2 = model.spatiotemporal_infer(  # type: ignore[assignment]
        z_init2,
        pos_emb2,
        gamma=cfg.gamma,
        T=cfg.hebbian_infer_T,
        return_errors=True,
        return_pred_loss=False,
        ach_value=0.5,
        return_ε=True,
        adaptive_T=cfg.infer_adaptive_T,
        convergence_threshold=cfg.infer_convergence_threshold,
        patience=cfg.infer_patience,
        min_T=cfg.infer_min_T,
        skip_bottom_up=True,
    )

from model.pc.local_updates import (
    compute_hebbian_temporal,
    compute_hebbian_topdown,
    compute_hebbian_conv,
    compute_hebbian_swiglu,
    compute_hebbian_decoder,
    compute_hebbian_byte_proj,
)

L = model.num_sub_layers
mod2 = 0.5  # baseline modulation

# temporal — 全层一起测 (函数内部循环)
timer2.synced_start("temporal (all 12)")
temp_weights = [p.weight for p in model.temporal_proj]
upd_t = compute_hebbian_temporal(
    ε_list2,
    z_init2,
    mod2,
    cfg.hebbian_base_eta,
    oja_alpha=cfg.oja_alpha,
    oja_eta=cfg.oja_eta,
    W_curr_list=temp_weights,
    sparse_k=cfg.sparse_outer_k,
)
timer2.end_last("temporal (all 12)")

# topdown — 全层一起测
timer2.synced_start("topdown (all 11)")
topdown_weights = [p.weight for p in model.topdown_proj]
upd_td = compute_hebbian_topdown(
    ε_list2,
    z_init2,
    mod2,
    cfg.hebbian_base_eta,
    oja_alpha=cfg.oja_alpha,
    oja_eta=cfg.oja_eta,
    W_curr_list=topdown_weights,
    sparse_k=cfg.sparse_outer_k,
)
timer2.end_last("topdown (all 11)")

# conv — 逐层测 (compute_hebbian_conv 是单层)
for block_idx in range(6):
    label = f"conv  (block {block_idx})"
    timer2.synced_start(label)
    ℓ = 2 * block_idx
    compute_hebbian_conv(
        ε_list2[ℓ],
        z_init2[ℓ],
        model.model.layers[block_idx].local_conv.weight,  # type: ignore[attr-defined]
        model.model.layers[block_idx].dilation,
        mod2,
        cfg.hebbian_base_eta,
        oja_alpha=cfg.oja_alpha,
        oja_eta=cfg.oja_eta,
        sparse_k=cfg.sparse_outer_k,
    )
    timer2.end_last(label)

target_onehot = F.one_hot(
    labels[:, 1:].long().clamp(0, 255), num_classes=256
).half()
# swiglu — 逐层测
for block_idx in range(6):
    label = f"swiglu (block {block_idx})"
    timer2.synced_start(label)
    ℓ = 2 * block_idx + 1
    compute_hebbian_swiglu(
        ε_list2[ℓ],
        z_init2[ℓ],
        model.model.layers[block_idx].mlp,
        mod2,
        cfg.hebbian_base_eta,
        oja_alpha=cfg.oja_alpha,
        oja_eta=cfg.oja_eta,
        sparse_k=cfg.sparse_outer_k,
    )
    timer2.end_last(label)

# decoder + byte_proj
timer2.synced_start("decoder")
compute_hebbian_decoder(
    z_init2[-1],
    target_onehot,
    model.decoder.weight,
    mod2,
    cfg.hebbian_base_eta,
    λ=0.0,
    oja_alpha=cfg.oja_alpha,
    oja_eta=cfg.oja_eta,
)
timer2.end_last("decoder")

timer2.synced_start("byte_proj")
compute_hebbian_byte_proj(
    ε_list2[0],
    byte_seq,
    model.model.byte_proj.weight,
    None,
    mod2,
    cfg.hebbian_base_eta,
    oja_alpha=cfg.oja_alpha,
    oja_eta=cfg.oja_eta,
)
timer2.end_last("byte_proj")

torch.cuda.synchronize()
tot2 = sum(timer2.mean_ms(n) for n in timer2.events)
for name in sorted(timer2.events.keys()):
    ms = timer2.mean_ms(name)
    pct = ms / tot2 * 100 if tot2 > 0 else 0
    print(f"  {name:<35s} {ms:>7.2f}ms {pct:>5.1f}%")
print(f"  {'─' * 50}")
print(f"  {'Sub-total (Phase 5 only)':<35s} {tot2:>7.2f}ms 100.0%")

# ═══════════════════════════════════════════════════════════════════
#  附录: CUDA 内存 & 超参快照
# ═══════════════════════════════════════════════════════════════════

if device.type == "cuda":
    print(f"\n── CUDA Memory ──")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
    print(f"  Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")
    print(f"  Peak:      {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")

print(f"\n── Config Snapshot ──")
print(f"  hebbian_infer_T={cfg.hebbian_infer_T}, adaptive_T={cfg.infer_adaptive_T}")
print(
    f"  infer_min_T={cfg.infer_min_T}, convergence_threshold={cfg.infer_convergence_threshold}"
)
print(f"  B={BATCH_SIZE}, S={SEQ_LEN}, H={HIDDEN_SIZE}, L={NUM_LAYERS}")
print(
    f"  sparse_outer_k={cfg.sparse_outer_k}, synaptic_competition_k={cfg.synaptic_competition_k}"
)
print(f"  oja_alpha={cfg.oja_alpha}, hebbian_base_eta={cfg.hebbian_base_eta}")
print(f"  consolidation_interval={cfg.consolidation_pipeline_interval}")
print(f"  synaptic_normalize={cfg.synaptic_normalize}")
