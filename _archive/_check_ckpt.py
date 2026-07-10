"""Quick checkpoint inspection."""
import torch, sys

paths = sys.argv[1:] if len(sys.argv) > 1 else [
    'out_pc_local_hybrid/hybrid_ckpt_s0.pt',
    'out_pc_local_hybrid/hybrid_ckpt_s499.pt',
    'out_pc_local_hybrid/hybrid_ckpt_s999.pt',
    'out_pc_local_hybrid/hybrid_final.pt',
]

for p in paths:
    try:
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
        info = {k: ckpt.get(k, 'N/A') for k in ['epoch','step','F','CE_local','CE_converged','beta_local','beta_conv']}
        print(f'{p}:')
        for k, v in info.items():
            print(f'  {k}: {v}')
    except Exception as e:
        print(f'{p}: ERROR {e}')
    print()
