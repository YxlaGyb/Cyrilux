import os

import click
from pkg.cli.utils import resolve_path

app = click.Group(name="data", help="数据管理命令")


@app.command()
@click.argument("input")
@click.option("--output", "-o", default=None, help="输出文件路径 (默认: input_converted.jsonl)")
@click.option("--max-samples", "-n", type=int, default=None, help="最大样本数")
def convert(input, output, max_samples):
    """转换数据格式: conversations/chosen-rejected → text."""
    from pkg.utils.data_converter import convert_file

    input_path = resolve_path(input)
    if not os.path.exists(input_path):
        raise click.ClickException(f"文件不存在: {input_path}")

    output_path = output or input_path.replace(".jsonl", "_converted.jsonl")
    output_path = resolve_path(output_path)

    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    if max_samples:
        print(f"最大样本数: {max_samples}")

    # convert_file expects an int for max_samples; treat None as no limit
    convert_file(input_path, output_path, max_samples)
    print(f"✓ 转换完成: {output_path}")


@app.command()
@click.argument("input")
@click.option("--output-dir", "-o", default=None, help="输出目录 (默认: {input_name}_split/)")
@click.option("--chunk-size", "-c", default=1000, type=int, help="每个输出文件的样本数")
@click.option("--no-convert", is_flag=True, default=False, help="不自动转换格式")
@click.option("--dry-run", is_flag=True, default=False, help="仅统计不分割")
def split(input, output_dir, chunk_size, no_convert, dry_run):
    """将大 JSONL 文件分割为多个小文件."""
    input_path = resolve_path(input)
    if not os.path.exists(input_path):
        raise click.ClickException(f"文件不存在: {input_path}")

    if output_dir is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.join(os.path.dirname(input_path), f"{base}_split")
    output_dir = resolve_path(output_dir)

    from pkg.utils.data_splitter import split_file

    result = split_file(
        input_path,
        output_dir,
        chunk_size=chunk_size,
        convert=not no_convert,
        dry_run=dry_run,
    )

    if dry_run:
        total_lines = result["total_lines"]
        n_chunks = result["n_chunks"]
        print(f"总计: {total_lines} 行 → {n_chunks} 个文件 (每个 {chunk_size} 行)")
    else:
        print(f"✓ 分割完成: {result['n_chunks']} 个文件 → {output_dir}")
        for f in result.get("output_files", []):
            print(f"  {f}")


@app.command()
@click.argument("directory", required=False, default=None)
def scan(directory):
    """扫描数据集目录，统计 JSONL 文件."""
    from pkg.utils.data_converter import scan_dataset_dir

    scan_dir = resolve_path(directory or "datasets")
    if not os.path.exists(scan_dir):
        raise click.ClickException(f"目录不存在: {scan_dir}")

    results = scan_dataset_dir(scan_dir)
    if not results:
        print(f"目录中没有 JSONL 文件: {scan_dir}")
        return

    print(f"\n数据集目录: {scan_dir}")
    print(f"{'文件':40s} {'行数':>8s} {'大小':>10s}  格式")
    print("-" * 75)
    total_lines = 0
    for r in results:
        print(f"{r['name']:40s} {r['est_lines']:>8d} {r['size_str']:>10s}  {r['format']}")
        total_lines += r["est_lines"]
    print(f"总计: {len(results)} 个文件, {total_lines} 行")
