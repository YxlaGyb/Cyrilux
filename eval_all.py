"""统一评估: 所有模型对比 (自监督 + PPL + 生成)"""
import os, sys, json, math, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import Dataset, DataLoader
from model.pc_layers import PCDynamicMiniMind, PCLocalDynamicMiniMind, load_pc_checkpoint
from model.pc_backbone_local import PCLocalBackbone
from model.model_minimind import MiniMindConfig

LOG = lambda msg: print(msg)


def _resolve_pos(pos_emb, seq_len, device):
    """Handle None pos_emb (Conv backbone has no position embeddings)."""
    if pos_emb is None or pos_emb[0] is None:
        return (None, None)
    return (pos_emb[0][:seq_len].to(device), pos_emb[1][:seq_len].to(device))


# ── key remap: checkpoint有 model.model.xxx, 当前代码是 model.xxx ──
def load_with_remap(model, ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt['model_state']
    remapped = {}
    for k, v in sd.items():
        if k.startswith('model.model.'):
            remapped['model.' + k[len('model.model.'):]] = v
        else:
            remapped[k] = v
    model.load_state_dict(remapped)
    return model

# ── 数据 ──
class EvalDataset(Dataset):
    def __init__(self, data_path, max_length=128, max_samples=500):
        self.max_length = max_length
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples: break
                self.samples.append(json.loads(line))
    def __len__(self): return len(self.samples)
    def __getitem__(self, index):
        s = self.samples[index]
        byte_seq = str(s['text']).encode('utf-8')[:self.max_length]
        padded = byte_seq.ljust(self.max_length, b'\x00')
        byte_tensor = torch.frombuffer(bytearray(padded), dtype=torch.uint8).clone()
        lbl = byte_tensor.clone()
        lbl[byte_tensor == 0x00] = -100
        return byte_tensor, lbl

# ===== 1. 自监督评估 (5维) =====
@torch.no_grad()
def eval_self_supervised(pc_model, loader, pos_emb, gamma=0.1, T=2, max_batches=20):
    """返回 {F, recon_acc, per_layer}"""
    pc_model.eval()
    metrics = {'F': [], 'recon_acc': [], 
               'sparsity': {f'L{i}':[] for i in range(1, pc_model.num_sub_layers+1)},
               'smoothness': {f'L{i}':[] for i in range(1, pc_model.num_sub_layers+1)},
               'variance': {f'L{i}':[] for i in range(1, pc_model.num_sub_layers+1)}}
    
    for batch_idx, (input_ids, labels) in enumerate(loader):
        if batch_idx >= max_batches: break
        input_ids = input_ids.to(next(pc_model.parameters()).device)
        bsz, seq_len = input_ids.shape
        
        pos = _resolve_pos(pos_emb, seq_len, input_ids.device)
        
        z = pc_model.init_z(input_ids)
        z, _, F_hist = pc_model.spatiotemporal_infer(z, pos, gamma=gamma, T=T)
        
        metrics['F'].append(F_hist[-1] / (bsz * seq_len))  # F_val已是float
        
        rep = pc_model.compute_representation_metrics(z)
        for ℓ in range(1, pc_model.num_sub_layers+1):
            metrics['sparsity'][f'L{ℓ}'].append(rep['sparsity'][ℓ-1])
            metrics['smoothness'][f'L{ℓ}'].append(rep['temporal_smoothness'][ℓ-1])
            metrics['variance'][f'L{ℓ}'].append(rep['variance'][ℓ-1])
        
        h_norm = pc_model.model.norm(z[pc_model.num_sub_layers])
        logits = pc_model.model.lm_head(h_norm)
        valid = labels != -100
        acc = (logits.argmax(-1)[valid] == input_ids[valid]).float().mean().item() if valid.any() else 0.0
        metrics['recon_acc'].append(acc)
        
        if (batch_idx+1) % 5 == 0: LOG(f'  自监督 batch {batch_idx+1}/{min(max_batches, len(loader))}')
    
    summary = {'F_mean': sum(metrics['F'])/len(metrics['F']),
               'recon_acc_mean': sum(metrics['recon_acc'])/len(metrics['recon_acc']),
               'per_layer': {}}
    for ℓ in range(1, pc_model.num_sub_layers+1):
        l = f'L{ℓ}'
        summary['per_layer'][l] = {
            'sparsity': sum(metrics['sparsity'][l])/len(metrics['sparsity'][l]),
            'smoothness': sum(metrics['smoothness'][l])/len(metrics['smoothness'][l]),
            'variance': sum(metrics['variance'][l])/len(metrics['variance'][l])}
    return summary

# ===== 2. PPL 评估 =====
@torch.no_grad()
def eval_ppl(pc_model, loader, pos_emb, gamma=0.1, T=2, max_batches=20):
    pc_model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch_idx, (input_ids, labels) in enumerate(loader):
        if batch_idx >= max_batches: break
        input_ids = input_ids.to(next(pc_model.parameters()).device)
        labels = labels.to(input_ids.device)
        bsz, seq_len = input_ids.shape
        pos = _resolve_pos(pos_emb, seq_len, input_ids.device)
        z = pc_model.init_z(input_ids)
        if T > 0:
            z, _, _ = pc_model.spatiotemporal_infer(z, pos, gamma=gamma, T=T)
        h_norm = pc_model.model.norm(z[pc_model.num_sub_layers])
        logits = pc_model.model.lm_head(h_norm)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                                  shift_labels.view(-1), ignore_index=-100, reduction='sum')
        total_loss += loss.item()
        total_tokens += (shift_labels != -100).sum().item()
        if (batch_idx+1) % 5 == 0: LOG(f'  PPL batch {batch_idx+1}')
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss), avg_loss

# ===== 3. 文本生成 =====
@torch.no_grad()
def generate(pc_model, prompt, pos_emb, max_new=50, gamma=0.1, T=2, temp=0.8, top_k=20):
    pc_model.eval()
    device = next(pc_model.parameters()).device
    byte_seq = list(prompt.encode('utf-8'))
    for _ in range(max_new):
        ids = torch.tensor([byte_seq], device=device, dtype=torch.uint8)
        pos = _resolve_pos(pos_emb, min(len(byte_seq), 128), device)
        z = pc_model.init_z(ids)
        if T > 0:
            z, _, _ = pc_model.spatiotemporal_infer(z, pos, gamma=gamma, T=T)
        h_norm = pc_model.model.norm(z[pc_model.num_sub_layers])
        logits = pc_model.model.lm_head(h_norm)
        next_logits = logits[0, -1, :] / temp
        if top_k > 0:
            tv, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < tv[-1]] = float('-inf')
        nid = torch.multinomial(torch.softmax(next_logits, -1), 1).item()
        byte_seq.append(nid)
        if nid == 0x02: break  # EOS
    return bytes(byte_seq).decode('utf-8', errors='replace')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='统一评估: 自监督 + PPL + 生成')
    parser.add_argument('--unified', type=str, default=None,
                        help='Unified checkpoint path (PCLocalDynamicMiniMind)')
    args = parser.parse_args()

    config = MiniMindConfig(hidden_size=256, num_hidden_layers=4, use_moe=False)
    ROOT = os.path.dirname(os.path.abspath(__file__))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    LOG(f'Device: {device}')

    ds = EvalDataset(os.path.join(ROOT, 'dataset', 'pretrain_t2t_mini.jsonl'))
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    loader_ss = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)  # 自监督用shuffle
    
    # ── 加载模型 ──
    models = {}
    
    if args.unified:
        # Unified: PCLocalDynamicMiniMind (Conv backbone + dopamine)
        model_cls = PCLocalDynamicMiniMind
        m_u = model_cls(config).to(device)
        ckpt = torch.load(args.unified, map_location=device, weights_only=False)
        m_u.load_state_dict(ckpt['model_state'], strict=False)
        models['Unified (Conv+Dopamine)'] = (m_u, (None, None))
        LOG(f'  Loaded: Unified ({args.unified}) ✓')
    else:
        # 1) Hybrid (PCDynamicMiniMind + F_pred + CE)
        m_hybrid = PCDynamicMiniMind(config).to(device)
        try:
            load_with_remap(m_hybrid, os.path.join(ROOT, 'out_pc_local_hybrid', 'hybrid_final.pt'))
            pos_hybrid = (m_hybrid.freqs_cos.to(device), m_hybrid.freqs_sin.to(device))
            models['Hybrid (F_pred+CE)'] = (m_hybrid, pos_hybrid)
            LOG('  Loaded: Hybrid (out_pc_local_hybrid/hybrid_final.pt) ✓')
        except Exception as e: LOG(f'  Hybrid load failed: {e}')
        
        # 2) PC-Local (PCDynamicMiniMind + F_pred only)
        m_pcl = PCDynamicMiniMind(config).to(device)
        try:
            load_with_remap(m_pcl, os.path.join(ROOT, 'out_pc_local', 'pcl_final.pt'))
            pos_pcl = (m_pcl.freqs_cos.to(device), m_pcl.freqs_sin.to(device))
            models['PC-Local (F_pred)'] = (m_pcl, pos_pcl)
            LOG('  Loaded: PC-Local (out_pc_local/pcl_final.pt) ✓')
        except Exception as e: LOG(f'  PC-Local load failed: {e}')
        
        # 3) PC-Original (无 temporal/topdown, 加载 backbone 部分)
        m_pc = PCDynamicMiniMind(config).to(device)
        try:
            ckpt_pc = torch.load(os.path.join(ROOT, 'out_pc', 'pc_final.pt'), map_location=device, weights_only=False)
            sd_pc = {}
            for k, v in ckpt_pc['model_state'].items():
                if k.startswith('model.model.'):
                    sd_pc['model.' + k[len('model.model.'):]] = v
                else:
                    sd_pc[k] = v
            # 只加载 backbone 部分, temporal/topdown 保持随机初始化
            m_pc.load_state_dict(sd_pc, strict=False)
            pos_pc = (m_pc.freqs_cos.to(device), m_pc.freqs_sin.to(device))
            models['PC-Original'] = (m_pc, pos_pc)
            LOG('  Loaded: PC-Original (out_pc/pc_final.pt, temporal/topdown随机初始化) ✓')
        except Exception as e: LOG(f'  PC-Original load failed: {e}')
    
    if not models:
        LOG('No models loaded!')
        return
    
    # ═══════════════════════════════════════
    # 评估 1: 自监督指标
    # ═══════════════════════════════════════
    LOG('\n' + '═'*60)
    LOG('评估 1: 自监督指标 (5维)')
    LOG('═'*60)
    
    results_ss = {}
    for name, (model, pos) in models.items():
        LOG(f'\n--- {name} ---')
        try:
            s = eval_self_supervised(model, loader_ss, pos, gamma=0.1, T=2, max_batches=20)
            results_ss[name] = s
            LOG(f'  F_total:        {s["F_mean"]:.4f}')
            LOG(f'  Recon Acc:      {s["recon_acc_mean"]:.4f}')
            for l, m in s['per_layer'].items():
                LOG(f'  {l}: sparsity={m["sparsity"]:.4f} smooth={m["smoothness"]:.4f} var={m["variance"]:.4f}')
        except Exception as e: LOG(f'  ERROR: {e}')
    
    # ═══════════════════════════════════════
    # 评估 2: Perplexity (T=0 vs T=2)
    # ═══════════════════════════════════════
    LOG('\n' + '═'*60)
    LOG('评估 2: Perplexity (PPL ↓)')
    LOG('═'*60)
    
    results_ppl = {}
    for name, (model, pos) in models.items():
        LOG(f'\n--- {name} ---')
        try:
            ppl0, loss0 = eval_ppl(model, loader, pos, T=0, max_batches=20)
            ppl2, loss2 = eval_ppl(model, loader, pos, T=2, max_batches=20)
            results_ppl[name] = {'T0_PPL': ppl0, 'T0_CE': loss0, 'T2_PPL': ppl2, 'T2_CE': loss2}
            delta = (ppl2 - ppl0) / ppl0 * 100
            LOG(f'  T=0: CE={loss0:.4f}  PPL={ppl0:.2f}')
            LOG(f'  T=2: CE={loss2:.4f}  PPL={ppl2:.2f}  ({"改善" if ppl2<ppl0 else "劣化"} {abs(delta):.1f}%)')
        except Exception as e: LOG(f'  ERROR: {e}')
    
    # ═══════════════════════════════════════
    # 评估 3: 文本生成
    # ═══════════════════════════════════════
    LOG('\n' + '═'*60)
    LOG('评估 3: 文本生成')
    LOG('═'*60)
    
    prompts = ['人工智能的未来在于', '小明今天去了公园，他看到', '深度学习是一种']
    
    for name, (model, pos) in models.items():
        LOG(f'\n--- {name} ---')
        for prompt in prompts:
            text = generate(model, tokenizer, prompt, pos, max_new=40, T=2, temp=0.8, top_k=20)
            LOG(f'  Prompt: {prompt}')
            LOG(f'  → {text}')
            LOG('')
    
    # ═══════════════════════════════════════
    # 汇总表
    # ═══════════════════════════════════════
    LOG('\n' + '═'*60)
    LOG('汇总对比')
    LOG('═'*60)
    
    LOG(f'\n{"Model":25s} {"F↓":>10s} {"Recon↑":>8s} {"PPL(T=0)↓":>12s} {"PPL(T=2)↓":>12s}')
    LOG('─'*70)
    for name in models:
        ss = results_ss.get(name, {})
        pp = results_ppl.get(name, {})
        F = f'{ss.get("F_mean", 0):.2f}' if ss else 'N/A'
        recon = f'{ss.get("recon_acc_mean", 0):.4f}' if ss else 'N/A'
        ppl0 = f'{pp.get("T0_PPL", 0):.2f}' if pp else 'N/A'
        ppl2 = f'{pp.get("T2_PPL", 0):.2f}' if pp else 'N/A'
        LOG(f'{name:25s} {F:>10s} {recon:>8s} {ppl0:>12s} {ppl2:>12s}')
    
    LOG('\nDone.')

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
