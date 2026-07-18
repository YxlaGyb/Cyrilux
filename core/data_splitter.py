"""
数据文件分割器: 将大 jsonl 文件按指定条目数分割为多个小文件。

用途:
  - 将 100K 条目的训练文件按每 1K 分割为 100 个小文件
  - 支持按目录批量分割
  - 支持分割后自动转换格式 (conversations→text)
"""
import os, json, math, argparse
from core.data_converter import convert_sample


def split_file(input_path: str, output_dir: str, chunk_size: int = 1000,
               convert: bool = True, prefix: str = None, dry_run: bool = False):
    """
    将 jsonl 文件分割为多个小文件。

    Args:
        input_path: 输入 jsonl 文件
        output_dir: 输出目录
        chunk_size: 每个输出文件的样本数 (默认 1000)
        convert: 是否自动转换格式 (conversations→text)
        prefix: 输出文件前缀 (默认用输入文件名)
        dry_run: 仅统计不分割

    Returns:
        dict: {n_chunks, total_lines, output_files, skipped}
    """
    os.makedirs(output_dir, exist_ok=True)
    base = prefix or os.path.splitext(os.path.basename(input_path))[0]

    # 先统计总行数并收集所有样本
    all_lines = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                all_lines.append(line)

    total = len(all_lines)
    n_chunks = math.ceil(total / chunk_size) if chunk_size > 0 else 1

    if dry_run:
        return {
            'n_chunks': n_chunks,
            'total_lines': total,
            'output_files': [],
            'skipped': 0,
        }

    output_files = []
    skipped = 0

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total)
        chunk_lines = all_lines[start:end]

        # 分割文件名编号 (e.g., exam_000, exam_001, ...)
        chunk_file = os.path.join(output_dir, f'{base}_part{chunk_idx:03d}.jsonl')
        write_count = 0

        with open(chunk_file, 'w', encoding='utf-8') as fout:
            for line in chunk_lines:
                if convert:
                    converted = convert_sample(line)
                    if converted:
                        fout.write(converted + '\n')
                        write_count += 1
                    else:
                        skipped += 1
                else:
                    fout.write(line + '\n')
                    write_count += 1

        output_files.append(chunk_file)

    return {
        'n_chunks': n_chunks,
        'total_lines': total,
        'output_files': output_files,
        'skipped': skipped,
    }


def split_directory(input_dir: str, output_dir: str, chunk_size: int = 1000,
                    convert: bool = True, pattern: str = '*.jsonl', dry_run: bool = False):
    """批量分割目录下的所有 jsonl 文件。"""
    import glob
    results = {}
    for fpath in sorted(glob.glob(os.path.join(input_dir, pattern))):
        fname = os.path.basename(fpath)
        file_result = split_file(fpath, output_dir, chunk_size, convert,
                                 prefix=os.path.splitext(fname)[0], dry_run=dry_run)
        results[fname] = file_result
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='数据文件分割器')
    parser.add_argument('input', type=str, help='输入 jsonl 文件或目录')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='输出目录 (默认 input_split/)')
    parser.add_argument('-c', '--chunk_size', type=int, default=1000,
                        help='每文件样本数 (默认 1000)')
    parser.add_argument('--no-convert', action='store_true',
                        help='不自动转换格式')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅统计不分割')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.input.rstrip('/\\')),
        f'split_{os.path.splitext(os.path.basename(args.input))[0]}'
    )

    if os.path.isdir(args.input):
        results = split_directory(args.input, output_dir, args.chunk_size,
                                  not args.no_convert, dry_run=args.dry_run)
        for fname, r in results.items():
            print(f'{fname}: {r["n_chunks"]} chunks, {r["total_lines"]} lines')
    else:
        result = split_file(args.input, output_dir, args.chunk_size,
                            not args.no_convert, dry_run=args.dry_run)
        print(f'{os.path.basename(args.input)}: {result["n_chunks"]} chunks, {result["total_lines"]} lines')
        for fp in result['output_files']:
            print(f'  → {fp}')
