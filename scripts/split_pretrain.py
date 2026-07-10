"""
切分 pretrain_t2t_mini.jsonl 为 5 个不重叠子集 (Task A~E).

Usage:
    python scripts/split_pretrain.py                          # 默认切分
    python scripts/split_pretrain.py --verify                 # 验证切分结果
    python scripts/split_pretrain.py --skip_lines 10000       # 调试: 只处理前 1 万行
    python scripts/split_pretrain.py --n_tasks 3              # 自定义任务数
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def split(args):
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(ROOT, 'dataset', 'pretrain_t2t_mini.jsonl')
    if not os.path.exists(src):
        print(f'[Error] Source not found: {src}')
        sys.exit(1)

    # 先统计总行数
    total_lines = 0
    with open(src, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    print(f'Total lines: {total_lines}')

    n_tasks = args.n_tasks
    lines_per_task = total_lines // n_tasks
    task_prefix = args.prefix

    if args.skip_lines:
        total_lines = min(total_lines, args.skip_lines)
        lines_per_task = total_lines // n_tasks
        print(f'Skip mode: using {total_lines} lines, {lines_per_task} per task')

    # 逐任务写入
    task_writers = []
    task_counts = []
    task_names = []
    for i in range(n_tasks):
        task_id = chr(ord('a') + i)
        task_name = f'{task_prefix}{task_id}'
        task_path = os.path.join(ROOT, 'dataset', f'{task_name}.jsonl')
        task_writers.append(open(task_path, 'w', encoding='utf-8'))
        task_counts.append(0)
        task_names.append(task_name)
        print(f'  Task {task_id} → {task_name}.jsonl')

    with open(src, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= total_lines:
                break
            task_idx = idx % n_tasks  # round-robin → 均匀分布
            task_writers[task_idx].write(line)
            task_counts[task_idx] += 1

    for w in task_writers:
        w.close()

    print('\nSplit complete:')
    for i, (name, count) in enumerate(zip(task_names, task_counts)):
        print(f'  {name}.jsonl: {count} lines')

    return task_names


def verify(args):
    """验证: 统计每文件行数 + 检查无重叠."""
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n_tasks = args.n_tasks
    task_prefix = args.prefix

    task_names = [f'{task_prefix}{chr(ord("a") + i)}' for i in range(n_tasks)]
    print(f'Verifying {n_tasks} task files under {ROOT}/dataset/...')

    lines_per_file = {}
    for name in task_names:
        path = os.path.join(ROOT, 'dataset', f'{name}.jsonl')
        if not os.path.exists(path):
            print(f'  [MISSING] {name}.jsonl')
            lines_per_file[name] = 0
            continue
        with open(path, 'r', encoding='utf-8') as f:
            lines_per_file[name] = sum(1 for _ in f)
        print(f'  {name}.jsonl: {lines_per_file[name]} lines')

    total = sum(lines_per_file.values())
    print(f'\n  Total: {total} lines')

    if total > 0:
        expected_total = total  # 参考值
        print(f'  Distribution:')
        for name, count in lines_per_file.items():
            pct = count / total * 100
            print(f'    {name}: {count:>8d} ({pct:.1f}%)')

    # 检查重叠 (小样本抽查, 取每文件前 10 行比较)
    print(f'\n  Overlap check (first 10 lines per file):')
    first_lines = {}
    for name in task_names:
        path = os.path.join(ROOT, 'dataset', f'{name}.jsonl')
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            first_lines[name] = [f.readline().strip() for _ in range(10)]
    n_overlap = 0
    for i, name_i in enumerate(task_names):
        for name_j in task_names[i + 1:]:
            overlap = set(first_lines.get(name_i, [])) & set(first_lines.get(name_j, []))
            if overlap:
                print(f'    [OVERLAP] {name_i} ∩ {name_j}: {len(overlap)} lines')
                n_overlap += len(overlap)
    if n_overlap == 0:
        print(f'    No overlap detected ✓')
    else:
        print(f'    {n_overlap} overlapping lines found (check split logic)')

    avg_cv = sum(count for count in lines_per_file.values()) / max(len(lines_per_file), 1)
    print(f'\n  Average: {avg_cv:.0f} lines/task')


def main():
    parser = argparse.ArgumentParser(description='Split pretrain dataset into N non-overlapping tasks')
    parser.add_argument('--verify', action='store_true', help='验证模式: 检查已切分的文件')
    parser.add_argument('--skip_lines', type=int, default=0, help='调试: 只处理前 N 行')
    parser.add_argument('--n_tasks', type=int, default=5, help='任务数 (默认 5)')
    parser.add_argument('--prefix', type=str, default='task_', help='输出文件名前缀')
    args = parser.parse_args()

    if args.verify:
        verify(args)
    else:
        split(args)


if __name__ == '__main__':
    main()
