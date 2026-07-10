"""
将异构数据集统一转为 _LocalDataset 可读的 {"text": "..."} 格式。

问题: agent_rl_math/agent_rl/rlaif 最后一条 assistant 回复为空
  (RL 数据集, 回复由模型生成), 但 agent_rl_math/agent_rl 有 gt 答案字段。
  修复: 用 gt 填充空回复; 无 gt 时跳过尾部空轮次。

任务映射:
  a = agent_rl_math   (20K, math agent)     → gt 填充
  b = lora_medical    (25K, medical QA)     → 正常
  c = lora_exam       (53K, exam questions) → 正常
  d = rlaif           (20K, RL feedback)    → 跳过尾部空回复
  e = agent_rl        (40K, general agent)  → gt 填充/跳过空回复
"""
import os, json

ROOT = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(ROOT, 'datasets')

TASKS = {
    'a': 'agent_rl_math.jsonl',
    'b': 'lora_medical.jsonl',
    'c': 'lora_exam.jsonl',
    'd': 'rlaif.jsonl',
    'e': 'agent_rl.jsonl',
}


def conv_to_text(conversations, gt=None):
    """将 conversation 转为纯文本, 空 assistant 用 gt 填充或跳过。"""
    lines = []
    for turn in conversations:
        role = turn.get('role', 'unknown')
        content = turn.get('content', '')
        if role == 'system' and not content:
            continue
        lines.append(f'{role}: {content}')

    # 检查最后一条是否为空 assistant
    text = '\n'.join(lines)
    stripped = text.strip()
    if stripped.endswith('assistant:') or stripped.endswith('assistant: '):
        if gt and isinstance(gt, list) and any(g for g in gt if g):
            answers = '\n'.join(str(g) for g in gt if g)
            text = text.rstrip() + '\n' + answers
        else:
            # 无 gt → 去掉最后的空 assistant 轮次
            text = text[:text.rfind('\nassistant:')]
    return text


def process():
    src_dir = os.path.join(ROOT, 'datasets')
    for task_id, fname in sorted(TASKS.items()):
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(DST, f'task_{task_id}.jsonl')
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
        print(f'Task {task_id} ({fname}): {count} samples → task_{task_id}.jsonl')


if __name__ == '__main__':
    process()
