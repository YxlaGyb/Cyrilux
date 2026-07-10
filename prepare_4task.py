"""
按领域切分 4 个任务数据集 (每任务 20K), 用于灾难性遗忘压力测试.

Task A: 日常对话 / 通用文本   ← pretrain_t2t_mini.jsonl (text 字段)
Task B: 科技 / 考试知识       ← lora_exam.jsonl + agent_rl_math.jsonl (conversations)
Task C: 医疗问诊              ← lora_medical.jsonl (conversations)
Task D: 对抗 / 结构化指令      ← sft_t2t_mini.jsonl (conversations)

输出: datasets/task_{a,b,c,d}_20k.jsonl
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.abspath(__file__))

# --- 源数据路径 ---
SOURCES = {
    'A': os.path.join(ROOT, 'datasets', 'pretrain_t2t_mini.jsonl'),
    'B_exam': os.path.join(ROOT, 'datasets', 'lora_exam.jsonl'),
    'B_math': os.path.join(ROOT, 'datasets', 'agent_rl_math.jsonl'),
    'C': os.path.join(ROOT, 'datasets', 'lora_medical.jsonl'),
    'D': os.path.join(ROOT, 'datasets', 'sft_t2t_mini.jsonl'),
}

OUT_DIR = os.path.join(ROOT, 'datasets')
N_PER_TASK = 20_000
SEED = 42


def extract_conversations(sample):
    """统一提取 conversations 列表, 兼容 text / conversations / chosen / fallback.

    返回: [{"role": "user"|"assistant"|"system", "content": str}, ...]
    """
    if isinstance(sample, str):
        return [{"role": "assistant", "content": sample}]
    if 'conversations' in sample:
        convs = []
        for m in sample['conversations']:
            content = m.get('content') or m.get('value') or ''
            role = m.get('role', 'user')
            convs.append({"role": role, "content": content})
        return convs
    if 'text' in sample:
        text = sample['text']
        first_nl = text.find('\n')
        if 0 < first_nl < len(text) - 1:
            return [
                {"role": "user", "content": text[:first_nl].strip()},
                {"role": "assistant", "content": text[first_nl + 1:].strip()}
            ]
        return [{"role": "assistant", "content": text.strip()}]
    if 'chosen' in sample:
        return [{"role": "assistant", "content": sample['chosen']}]
    return [{"role": "assistant", "content": str(sample)}]


def sample_jsonl(src_path, n, seed=SEED):
    """从 JSONL 文件中随机采样 n 条 (先读全量再 shuffle 采样)."""
    if not os.path.exists(src_path):
        print(f'  ⚠️  文件不存在, 跳过: {src_path}')
        return []
    with open(src_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    rng = random.Random(seed)
    rng.shuffle(lines)
    sampled = []
    for line in lines[:n]:
        sample = json.loads(line)
        convs = extract_conversations(sample)
        total_len = sum(len(c.get('content', '').strip()) for c in convs)
        if total_len > 10:  # 过滤空/过短样本
            sampled.append({'conversations': convs})
    print(f'  {src_path}: {len(lines)} lines → {len(sampled)} valid samples (requested {n})')
    return sampled


def write_jsonl(samples, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f'  → {out_path} ({len(samples)} samples)')


def main():
    print('=' * 60)
    print('  准备 4 任务数据 (每任务 20K)')
    print('=' * 60)

    # Task A: 通用文本 (pretrain)
    print('\n--- Task A: 日常对话 / 通用文本 ---')
    a = sample_jsonl(SOURCES['A'], N_PER_TASK)
    write_jsonl(a, os.path.join(OUT_DIR, 'task_a_daily_20k.jsonl'))

    # Task B: 科技知识 (exam + math)
    print('\n--- Task B: 科技 / 考试知识 ---')
    b1 = sample_jsonl(SOURCES['B_exam'], N_PER_TASK // 2)
    b2 = sample_jsonl(SOURCES['B_math'], N_PER_TASK // 2)
    b = (b1 + b2)[:N_PER_TASK]
    random.Random(SEED).shuffle(b)
    write_jsonl(b, os.path.join(OUT_DIR, 'task_b_tech_20k.jsonl'))

    # Task C: 医疗问诊
    print('\n--- Task C: 医疗问诊 ---')
    c = sample_jsonl(SOURCES['C'], N_PER_TASK)
    write_jsonl(c, os.path.join(OUT_DIR, 'task_c_medical_20k.jsonl'))

    # Task D: 结构化/对抗指令
    print('\n--- Task D: 结构化指令 (SFT) ---')
    d = sample_jsonl(SOURCES['D'], N_PER_TASK)
    write_jsonl(d, os.path.join(OUT_DIR, 'task_d_sft_20k.jsonl'))

    print('\n' + '=' * 60)
    print('  完成! 4 任务数据已就绪:')
    for name in ['task_a_daily_20k.jsonl', 'task_b_tech_20k.jsonl', 'task_c_medical_20k.jsonl', 'task_d_sft_20k.jsonl']:
        fpath = os.path.join(OUT_DIR, name)
        if os.path.exists(fpath):
            n_lines = sum(1 for _ in open(fpath, 'r', encoding='utf-8'))
            print(f'    {name}: {n_lines} lines')
    print('=' * 60)


if __name__ == '__main__':
    main()
