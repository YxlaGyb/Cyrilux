import json

from click.testing import CliRunner

from pkg.cli import app


def _write_conv_file(path, n=4):
    lines = [json.dumps({"conversations": [{"role": "user", "content": f"q{i}"}]}) for i in range(n)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestDataConvert:
    def test_convert(self, tmp_path):
        src = tmp_path / "in.jsonl"
        dst = tmp_path / "out.jsonl"
        _write_conv_file(src)
        runner = CliRunner()
        result = runner.invoke(app, ["data", "convert", str(src), "-o", str(dst)])
        assert result.exit_code == 0
        out_lines = dst.read_text(encoding="utf-8").strip().split("\n")
        assert len(out_lines) == 4
        assert json.loads(out_lines[0])["text"].startswith("<|user|>")

    def test_missing_input_fails(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(app, ["data", "convert", str(tmp_path / "nope.jsonl")])
        assert result.exit_code != 0
        assert "文件不存在" in result.output


class TestDataSplit:
    def test_split(self, tmp_path):
        src = tmp_path / "in.jsonl"
        out = tmp_path / "out"
        _write_conv_file(src, n=5)
        runner = CliRunner()
        result = runner.invoke(app, ["data", "split", str(src), "-o", str(out), "-c", "2"])
        assert result.exit_code == 0
        assert "分割完成" in result.output
        part_files = sorted(out.glob("in_part*.jsonl"))
        assert len(part_files) == 3

    def test_dry_run(self, tmp_path):
        src = tmp_path / "in.jsonl"
        _write_conv_file(src, n=5)
        runner = CliRunner()
        result = runner.invoke(app, ["data", "split", str(src), "-c", "2", "--dry-run"])
        assert result.exit_code == 0
        assert "5 行" in result.output

    def test_no_convert_keeps_conversations(self, tmp_path):
        src = tmp_path / "in.jsonl"
        out = tmp_path / "out"
        _write_conv_file(src, n=2)
        runner = CliRunner()
        result = runner.invoke(app, ["data", "split", str(src), "-o", str(out), "--no-convert"])
        assert result.exit_code == 0
        part = list(out.glob("in_part*.jsonl"))[0]
        first = json.loads(part.read_text(encoding="utf-8").strip().split("\n")[0])
        assert "conversations" in first

    def test_missing_input_fails(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(app, ["data", "split", str(tmp_path / "nope.jsonl")])
        assert result.exit_code != 0


class TestDataScan:
    def test_scan_mixed_formats(self, tmp_path):
        (tmp_path / "a.jsonl").write_text(json.dumps({"text": "x"}) + "\n", encoding="utf-8")
        (tmp_path / "b.jsonl").write_text(json.dumps({"conversations": []}) + "\n", encoding="utf-8")
        (tmp_path / "c.txt").write_text("not jsonl\n", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(app, ["data", "scan", str(tmp_path)])
        assert result.exit_code == 0
        assert "a.jsonl" in result.output
        assert "b.jsonl" in result.output
        assert "c.txt" not in result.output

    def test_scan_empty_dir(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(app, ["data", "scan", str(tmp_path)])
        assert result.exit_code == 0
        assert "没有 JSONL 文件" in result.output

    def test_scan_missing_dir_fails(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(app, ["data", "scan", str(tmp_path / "nope")])
        assert result.exit_code != 0
