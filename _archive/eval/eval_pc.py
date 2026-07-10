"""分析 PC 训练结果 & 生成文本。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from model.pc_layers import PCMiniMind
from model.model_minimind import MiniMindConfig
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(ROOT, 'out_pc')

# ── Checkpoint CE 曲线 ──
ckpts = sorted([f for f in os.listdir(out_dir) if f.startswith('pc_ckpt')])
print(f'=== CE History ({len(ckpts)} checkpoints) ===')
for ckpt in ckpts:
    d = torch.load(os.path.join(out_dir, ckpt), map_location='cpu', weights_only=False)
    step = d['step']
    print(f'  Step {step:>5}: CE={d["loss"]:.4f}, F={d["F"]:.4f}')

# ── 加载最终模型 ──
final = torch.load(os.path.join(out_dir, 'pc_final.pt'), map_location='cpu', weights_only=False)
print(f'\nFinal: CE={final["loss"]:.4f}, F={final["F"]:.4f}')

lm_config = MiniMindConfig(hidden_size=256, num_hidden_layers=4, use_moe=False)
model = PCMiniMind(lm_config)
model.load_state_dict(final['model_state'])
model.eval()

# ── 生成测试 ──
tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT, 'model'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

prompts = [
    '人工智能的未来是',
    '机器学习是一种',
    '自然语言处理',
]

print('\n=== Generation Samples ===')
for prompt in prompts:
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
    input_ids = torch.cat([torch.tensor([[tokenizer.bos_token_id]]).to(device), input_ids], dim=1)

    with torch.no_grad():
        # 自回归生成
        gen_ids = input_ids
        for _ in range(50):
            out = model.model(input_ids=gen_ids)
            logits = out.logits[:, -1, :]
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            gen_ids = torch.cat([gen_ids, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break

    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    print(f'\nPrompt: {prompt}')
    print(f'Generated: {text}')
