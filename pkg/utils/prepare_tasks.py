"""
按领域切分 + 异构数据集统一转换 → core/dataset.py 可读的 {"text": "..."} 格式。

功能:
  1. prepare_4tasks: 从 pretrain_t2t_mini/lora_exam/agent_rl_math/lora_medical/sft_t2t_mini
     切分出 A:日常/B:科技/C:医疗/D:SFT 四个任务 (各 20K)
  2. prepare_hetero: 将 5 种异构 RL+Medical+Exam 数据集统一为 text 格式
     任务映射: a=agent_rl_math, b=lora_medical, c=lora_exam, d=rlaif, e=agent_rl

用法:
  from pkg.utils.prepare_tasks import prepare_4tasks, prepare_hetero
"""
import os, json, random
from pathlib import Path


# ── 基础路径 ──

def _find_project_root() -> str:
    """向上查找包含 pyproject.toml 的目录。"""
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / 'pyproject.toml').exists():
            return str(parent)
    return str(cwd)


PROJECT_ROOT = _find_project_root()
DATASETS_DIR = os.path.join(PROJECT_ROOT, 'datasets')


# ═══════════════════════════════════════════════════════════════════
# 4 任务切分 (原 prepare_4task.py)
# ═══════════════════════════════════════════════════════════════════

def extract_conversations(sample: dict) -> str:
    """从样本提取纯文本 (不保留角色标记)。"""
    if 'conversations' in sample:
        parts = []
        for m in sample['conversations']:
            content = m.get('content', '')
            if content:
                parts.append(content)
        return '\n'.join(parts)
    return sample.get('text', json.dumps(sample, ensure_ascii=False))


def sample_jsonl(input_path: str, n: int, seed: int = 42) -> list[dict]:
    """从 jsonl 文件中随机采样 n 条。"""
    random.seed(seed)
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) <= n:
        sampled = lines
    else:
        sampled = random.sample(lines, n)
    return [json.loads(s) for s in sampled]


def write_jsonl(samples: list[dict], output_path: str):
    """写入 jsonl 文件。"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for s in samples:
            text = extract_conversations(s)
            f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')


def prepare_4tasks(data_dir: str = None, output_dir: str = None,
                   n_per_task: int = 20000, seed: int = 42):
    """
    从 pretrain_t2t_mini/lora_exam/agent_rl_math/lora_medical/sft_t2t_mini
    切分 4 个任务数据集。

    任务定义:
      A (daily)  = pretrain_t2t_mini.jsonl    — 日常对话
      B (tech)   = lora_exam.jsonl            — 科技知识
      C (medical)= agent_rl_math.jsonl        — 医疗问诊 (原命名有误但保留)
      D (sft)    = lora_identity.jsonl + SFT  — 指令遵循
    """
    data_dir = data_dir or DATASETS_DIR
    output_dir = output_dir or data_dir

    task_configs = [
        ('task_a_daily', 'pretrain_t2t_mini.jsonl'),
        ('task_b_tech', 'lora_exam.jsonl'),
        ('task_c_medical', 'agent_rl_math.jsonl'),
        ('task_d_sft', 'sft_t2t_mini.jsonl'),
    ]

    results = {}
    for task_name, src_file in task_configs:
        src_path = os.path.join(data_dir, src_file)
        if not os.path.exists(src_path):
            print(f'⚠️  跳过 {task_name}: {src_file} 不存在')
            continue
        print(f'📖  读取 {src_file}...')
        samples = sample_jsonl(src_path, n_per_task, seed)
        dst_path = os.path.join(output_dir, f'{task_name}_20k.jsonl')
        write_jsonl(samples, dst_path)
        print(f'✅  {task_name}: {len(samples)} 条 → {dst_path}')
        results[task_name] = len(samples)

    return results


# ═══════════════════════════════════════════════════════════════════
# 异构数据集转换 (原 prepare_hetero_tasks.py)
# ═══════════════════════════════════════════════════════════════════

HETERO_TASKS = {
    'a': 'agent_rl_math.jsonl',
    'b': 'lora_medical.jsonl',
    'c': 'lora_exam.jsonl',
    'd': 'rlaif.jsonl',
    'e': 'agent_rl.jsonl',
}


def conv_to_text(conversations: list, gt: list = None) -> str:
    """将 conversation 转为纯文本, 空 assistant 用 gt 填充或跳过。"""
    lines = []
    for turn in conversations:
        role = turn.get('role', 'unknown')
        content = turn.get('content', '')
        if role == 'system' and not content:
            continue
        lines.append(f'{role}: {content}')

    text = '\n'.join(lines)
    stripped = text.strip()
    if stripped.endswith('assistant:') or stripped.endswith('assistant: '):
        if gt and isinstance(gt, list) and any(g for g in gt if g):
            answers = '\n'.join(str(g) for g in gt if g)
            text = text.rstrip() + '\n' + answers
        else:
            # 无 gt → 去掉最后的空 assistant 轮次
            last_break = text.rfind('\nassistant:')
            if last_break >= 0:
                text = text[:last_break]
    return text


def prepare_hetero(data_dir: str = None, output_dir: str = None):
    """
    将 5 种异构数据集统一转换为 text 格式。

    处理规则:
      - agent_rl_math / agent_rl: 用 gt 填充空的 assistant 回复
      - rlaif: 跳过尾部空回复
      - lora_medical / lora_exam: 正常转换 (含 role 标记)
    """
    data_dir = data_dir or DATASETS_DIR
    output_dir = output_dir or data_dir

    results = {}
    for task_id, fname in sorted(HETERO_TASKS.items()):
        src_path = os.path.join(data_dir, fname)
        if not os.path.exists(src_path):
            print(f'⚠️  跳过 task_{task_id}: {fname} 不存在')
            continue

        dst_path = os.path.join(output_dir, f'task_{task_id}.jsonl')
        count = 0
        with open(src_path, 'r', encoding='utf-8') as fin, \
             open(dst_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                sample = json.loads(line)
                gt = sample.get('gt', None)

                if 'conversations' in sample:
                    text = conv_to_text(sample['conversations'], gt)
                elif 'chosen' in sample:
                    text = conv_to_text(sample['chosen'])
                elif 'text' in sample:
                    text = sample['text']
                else:
                    text = json.dumps(sample, ensure_ascii=False)

                if text.strip():
                    fout.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                    count += 1

        print(f'✅  Task {task_id} ({fname}): {count} samples → {dst_path}')
        results[task_id] = count

    return results
