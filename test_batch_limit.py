"""快速扫描 batch_size VRAM 上限 (PC pipeline, GTX 1650 Ti 4GB)"""
import torch, sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.pc_layers import PCLocalDynamicMiniMind
from model.model_minimind import MiniMindConfig

lm_config = MiniMindConfig(hidden_size=256, num_hidden_layers=4, use_moe=False)
device = 'cuda:0'
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

results = []
for batch in [80, 96, 112, 128, 144, 160]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = PCLocalDynamicMiniMind(lm_config).to(device)
        # Warmup
        dummy = torch.randint(0, 256, (batch, 128), device=device).long()
        dummy_pos = model.get_position_embeddings(128, device)
        with torch.no_grad():
            model.forward_with_ce(dummy, dummy, dummy_pos)

        # Full pipeline
        byte_seq = torch.randint(0, 256, (batch, 128), device=device).long()
        labels = torch.randint(0, 256, (batch, 128), device=device).long()
        pos_emb = model.get_position_embeddings(128, device)

        z_init, ce_loss = model.forward_with_ce(byte_seq, labels, pos_emb)
        z_detached = [z.detach() for z in z_init]
        z_conv, _, F_hist, F_pred = model.spatiotemporal_infer(z_detached, pos_emb, gamma=0.1, T=2, return_errors=False, return_pred_loss=True)
        ce_conv = model.compute_ce_loss(z_conv, labels)
        total = F_pred + ce_loss + ce_conv
        total.backward()

        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f'[OK]   batch={batch:3d}  max_mem={mem:.2f}GB')
        results.append((batch, mem))

        del model, dummy, byte_seq, labels, pos_emb, z_init, ce_loss
        del z_detached, z_conv, errors, F_hist, F_pred, ce_conv, total
    except RuntimeError as e:
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f'[OOM]  batch={batch:3d}  max_mem={mem:.2f}GB  ({str(e)[:60]})')
        torch.cuda.empty_cache()
        break

print('\n=== Batch limit scan result ===')
for b, m in results:
    print(f'  batch={b:3d}  mem={m:.2f}GB')
print(f'Safe max batch: {results[-1][0] if results else "N/A"}')
