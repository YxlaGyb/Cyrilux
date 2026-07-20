"""
数据格式转换器: 将"原生样本"(conversations/chosen-rejected 格式) → 项目标准 text 格式。

支持格式:
  1. {"conversations": [{"role": "user"/"assistant"/"system", "content": "..."}, ...]}
  2. {"chosen": [...messages], "rejected": [...messages]}
  3. {"text": "..."}  ← 已是项目格式，直接透传
  4. {"conversations": [...], "gt": [...]}  ← 追加 ground truth

输出: {"text": "<|user|>...<|assistant|>...<|end|>"}
"""
import os, json, argparse
from typing import Iterator


# ── 角色标记 ──
ROLE_MARKERS = {
    'user': '<|user|>',
    'assistant': '<|assistant|>',
    'system': '<|system|>',
    'tool': '<|tool|>',
}
SEP = '\n'
EOS = '<|end|>'


def conversations_to_text(conv_list: list, gt_list: list = None) -> str:
    """将 conversations 数组转换为纯文本。"""
    parts = []
    for msg in conv_list:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        marker = ROLE_MARKERS.get(role, ROLE_MARKERS['user'])
        parts.append(f"{marker}{SEP}{content}" if content else marker)

    # 如果有 ground truth，追加
    if gt_list:
        parts.append(f"{ROLE_MARKERS['assistant']}{SEP}{'; '.join(str(g) for g in gt_list)}")

    parts.append(EOS)
    return SEP.join(parts)


def convert_sample(line: str) -> str | None:
    """转换单行 jsonl 样本。返回 text 格式的 json 行，或 None（跳过）。"""
    try:
        sample = json.loads(line)
    except json.JSONDecodeError:
        return None

    # 格式 3: 已经是 text 格式
    if 'text' in sample and isinstance(sample['text'], str):
        return line  # 原样透传

    # 格式 1/4: conversations 格式
    if 'conversations' in sample:
        gt = sample.get('gt', None)
        text = conversations_to_text(sample['conversations'], gt)
        return json.dumps({'text': text}, ensure_ascii=False)

    # 格式 2: chosen/rejected 格式
    if 'chosen' in sample:
        text = conversations_to_text(sample['chosen'])
        return json.dumps({'text': text}, ensure_ascii=False)

    # 其他未知格式: 尝试兜底
    return None


def convert_file(input_path: str, output_path: str, max_samples: int = None):
    """转换整个 jsonl 文件。"""
    count = 0
    skipped = 0
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for i, line in enumerate(fin):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            converted = convert_sample(line)
            if converted:
                fout.write(converted + '\n')
                count += 1
            else:
                skipped += 1
    return count, skipped


def scan_dataset_dir(data_dir: str) -> list[dict]:
    """扫描目录，识别需要转换的文件。返回文件信息列表。"""
    results = []
    if not os.path.isdir(data_dir):
        return results

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.jsonl'):
            continue
        fpath = os.path.join(data_dir, fname)
        fsize = os.path.getsize(fpath)

        # 采样前 3 行判断格式
        format_type = 'unknown'
        is_converted = False
        sample_rows = []
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                for _ in range(3):
                    line = f.readline().strip()
                    if line:
                        sample_rows.append(line)
                        if '"text"' in line and '"conversations"' not in line:
                            is_converted = True
        except:
            pass

        if is_converted:
            format_type = 'text (已转换)'
        elif any('"conversations"' in r for r in sample_rows):
            format_type = 'conversations (需转换)'
        elif any('"chosen"' in r for r in sample_rows):
            format_type = 'chosen/rejected (需转换)'

        # 估算行数
        est_lines = 0
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                for _ in range(10000):
                    if f.readline():
                        est_lines += 1
                    else:
                        break
        except:
            pass

        results.append({
            'name': fname,
            'path': fpath,
            'size': fsize,
            'size_str': f'{fsize / 1024 / 1024:.1f}MB' if fsize > 1024 * 1024 else f'{fsize / 1024:.1f}KB',
            'format': format_type,
            'est_lines': est_lines,
            'needs_conversion': not is_converted,
        })
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='数据格式转换: 原生样本 → text 格式')
    parser.add_argument('input', type=str, help='输入 jsonl 文件路径')
    parser.add_argument('output', type=str, nargs='?', default=None, help='输出 jsonl 文件路径（默认 input_converted.jsonl）')
    parser.add_argument('--max_samples', type=int, default=None, help='最大转换样本数')
    args = parser.parse_args()

    output = args.output or args.input.replace('.jsonl', '_converted.jsonl')
    count, skipped = convert_file(args.input, output, args.max_samples)
    print(f'转换完成: {count} 条已转换, {skipped} 条跳过')
    print(f'输出: {output}')
