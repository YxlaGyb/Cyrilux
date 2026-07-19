#!/usr/bin/env python3
"""Phase E: 端到端验证 — 自组织机制 (Oja + SynNorm + Salience + Neurogenesis).

验证项:
  1. 模型前向 + 多步 Hebbian 更新: 无 NaN
  2. Oja 约束: 权重 max abs < 60000 (fp16 安全边界)
  3. Salience Gate: 更新后 gate 值合理, 稀疏损失可计算
  4. Neurogenesis: 剪枝/生长接口可调用
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from model.pc_layers import PCLocalDynamicMiniMind
from model.model_minimind import MiniMindConfig
from model.local_updates import (
    compute_all_hebbian_updates, apply_hebbian_updates,
    compute_modulators, compute_precision_scales,
    compute_lambda, compute_errors_by_layer,
)
from core.globals import DEVICE
from core.training import TrainingConfig

def test_basic_forward():
    """验证模型创建 + 前向 + gate 初始化。"""
    cfg = MiniMindConfig(hidden_size=128, num_hidden_layers=2)
    model = PCLocalDynamicMiniMind(cfg).to(DEVICE)
    model.eval()
    x = torch.randint(0, 256, (2, 1, 32), device=DEVICE)  # [B, 1, S] 字节流
    x = x.repeat(1, 2, 1)  # [B, 2, S] — 2 bytes per token
    
    # init_z
    with torch.no_grad():
        z_init = model.init_z(x)
    assert len(z_init) == model.num_sub_layers + 1, f"z_init length: {len(z_init)}"
    
    # forward_with_ce
    labels = torch.randint(0, 256, (2, 32), device=DEVICE)
    pos_emb = model.get_position_embeddings(32, DEVICE)
    with torch.no_grad():
        z_conv, ce_loss = model.forward_with_ce(x, labels, pos_emb)
    assert not torch.isnan(ce_loss), f"CE loss NaN: {ce_loss}"
    print(f"  [OK] init_z + forward_with_ce: CE={ce_loss:.4f}")
    
    # 验证 gate 初始化
    if hasattr(model, 'salience_gates'):
        gs = model.get_gate_stats()
        assert gs['active_ratio'] > 0.99, f"Gate active ratio too low: {gs['active_ratio']}"
        print(f"  [OK] Gates: {gs['n_active']}/{gs['n_total']} active ({gs['active_ratio']:.1%})")
    
    return model, x, labels


def test_hebbian_step(model, byte_seq, labels):
    """验证多步 Hebbian 更新不产生 NaN 且权重有界。"""
    cfg = TrainingConfig(
        hidden_size=128, num_hidden_layers=2,
        oja_alpha=0.05, synaptic_normalize=True,
    )
    pos_emb = model.get_position_embeddings(byte_seq.size(-1), DEVICE)
    
    with torch.no_grad():
        z_init = model.init_z(byte_seq)
        # spatiotemporal_infer
        z_conv, errors_hist, F_hist, _, ε_list = model.spatiotemporal_infer(
            z_init, pos_emb, gamma=0.1, T=2,
            return_errors=True, return_pred_loss=False, return_ε=True,
        )
    
    assert not any(torch.isnan(z).any() for z in z_conv), "NaN in z_conv"
    
    # 调制信号
    F_curr = F_hist[-1] if F_hist else 0.0
    D, ACh, mod = compute_modulators(F_curr, 0.0, 0.0, cfg)
    λ = compute_lambda(1, cfg.hebbian_lambda_decay, cfg.hebbian_lambda_min)
    π_list = compute_precision_scales(ε_list, ACh, D, cfg)
    
    # Hebbian 计算 + 应用
    with torch.no_grad():
        target_onehot = nn.functional.one_hot(
            labels[:, 1:].long().clamp(0, 255), num_classes=256).float()
        updates = compute_all_hebbian_updates(
            ε_list, z_init, byte_seq, model, cfg,
            D=D, ACh=ACh, modulation=mod, λ=λ,
            decoder=model.decoder, target_byte_embed=target_onehot,
            oja_alpha=0.05, bcm_state=None, verbose=False,
        )
        apply_hebbian_updates(updates, model, synaptic_normalize=True)
    
    # 检查 NaN
    for name, p in model.named_parameters():
        if torch.isnan(p).any():
            raise RuntimeError(f"NaN in {name} after Hebbian update")
    
    # 检查权重有界
    max_abs = max(p.abs().max().item() for p in model.parameters() if p.numel() > 0)
    assert max_abs < 60000, f"Weight max abs {max_abs:.1f} exceeds fp16 safety bound!"
    print(f"  [OK] Hebbian step: max|W|={max_abs:.1f}, no NaN")
    
    # 验证 gate sparsity loss
    if hasattr(model, 'salience_gates'):
        sl = model.get_gate_sparsity_loss(0.001)
        gs = model.get_gate_stats()
        print(f"  [OK] Post-step gates: ratio={gs['active_ratio']:.1%}, sparsity_loss={sl:.6f}")


def test_neurogenesis_mock():
    """验证 NeurogenesisController 剪枝/生长逻辑。"""
    from continual.neurogenesis import NeurogenesisController
    from model.local_blocks import SalienceGate
    
    gates = nn.ModuleList([SalienceGate(256) for _ in range(1)])
    ctrl = NeurogenesisController(hidden_size=256)
    
    # 模拟: 人为设置一些死亡通道
    gates[0].logits.data[:64] = -10.0  # 前 64 通道死亡
    gates[0].activation_ema[:64] = 0.0
    
    # 模拟模型 (只需局部权重)
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList([nn.Module()])
            self.model[0].local_conv = nn.Conv1d(256, 256, 3, padding=1)
            self.model[0].mlp = type('MLP', (), {
                'gate_proj': nn.Linear(256, 256*4, bias=False),
                'up_proj': nn.Linear(256, 256*4, bias=False),
                'down_proj': nn.Linear(256*4, 256, bias=False),
            })()
    
    model = MockModel()
    model.salience_gates = gates  # 挂载到模型
    
    # 模拟 ε_list
    ε_list = [torch.randn(1, 8, 256) * 3.0]  # 高误差
    
    # 执行 neurogenesis step
    result = ctrl.step(model, ε_list, global_step=100)
    print(f"  [OK] Neurogenesis step: pruned={result['n_pruned']}, "
          f"resurrected={result['n_resurrected']}, "
          f"split={result['n_split']}, "
          f"active_ratio={result['active_ratio']:.2%}")
    assert result['n_pruned'] >= 64, f"Should have pruned at least 64 channels, got {result['n_pruned']}"


def test_training_loop_bp_free():
    """验证训练循环的 bp_free 模式可执行。"""
    from core.training import TrainingConfig, TrainingLoop
    
    cfg = TrainingConfig(
        hidden_size=128, num_hidden_layers=2,
        batch_size=2, max_seq_len=16, lr=3e-4, epochs=1, subset=0,
        T_infer=1, gamma=0.1,
        hebbian_infer_T=1,         # 单步推理，避免循环放大
        hebbian_base_eta=1e-6,     # 极小 Hebbian 学习率，防止权重漂移
        oja_alpha=0.05,            # 适中的 Oja 衰减
        synaptic_normalize=True,
        enable_salience_gating=True,
        enable_neurogenesis=True,
        neurogenesis_prune_interval=5,
        neurogenesis_grow_interval=10,
    )
    loop = TrainingLoop(cfg)
    
    # 构建模型 (不创建 dataset, 直接调 train_step)
    loop.model = PCLocalDynamicMiniMind(
        MiniMindConfig(hidden_size=128, num_hidden_layers=2)
    ).to(DEVICE).half()       # fp16，与生产环境一致
    loop.model.eval()
    
    # 初始化 neurogenesis
    from continual.neurogenesis import NeurogenesisController
    loop.neurogenesis = NeurogenesisController(
        hidden_size=128, prune_interval=5, grow_interval=10,
    )
    
    # 执行多步
    byte_seq = torch.randint(0, 256, (2, 1, 32), device=DEVICE).repeat(1, 2, 1)
    labels = torch.randint(0, 256, (2, 32), device=DEVICE)
    
    for step in range(15):
        result = loop.train_step(byte_seq, labels)
        assert not any(k in result and isinstance(result[k], float) and 
                      (result[k] != result[k]) for k in ['ce_val', 'F_val']), \
            f"NaN detected at step {step}: ce={result.get('ce_val')}, F={result.get('F_val')}"
    
    print(f"  [OK] 15 bp-free steps: ce={result['ce_val']:.4f}, F={result['F_val']:.4f}")
    if 'gate_active_ratio' in result:
        print(f"  [OK] Gate active ratio: {result['gate_active_ratio']:.2%}")
    if 'neuro_active_ratio' in result:
        print(f"  [OK] Neuro active ratio: {result['neuro_active_ratio']:.2%}")


if __name__ == '__main__':
    print("=" * 60)
    print("Phase E: 自组织机制端到端验证")
    print("=" * 60)
    
    print("\n1. 基础前向 + Gate 初始化")
    model, x, labels = test_basic_forward()
    
    print("\n2. Hebbian 更新 + 权重约束")
    test_hebbian_step(model, x, labels)
    
    print("\n3. Neurogenesis 剪枝/生长逻辑")
    test_neurogenesis_mock()
    
    print("\n4. 训练循环 15 步 (bp_free)")
    test_training_loop_bp_free()
    
    print("\n" + "=" * 60)
    print("全部验证通过 OK")
    print("=" * 60)
