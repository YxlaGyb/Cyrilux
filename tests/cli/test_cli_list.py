import json

from click.testing import CliRunner

from pkg.cli import app


class TestListCheckpoints:
    def test_empty_dir(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(app, ["list", "checkpoints", str(tmp_path)])
        assert result.exit_code == 0
        assert "没有检查点文件" in result.output

    def test_lists_files(self, tmp_path):
        (tmp_path / "model.pt").write_bytes(b"\x00\x01")
        (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(app, ["list", "checkpoints", str(tmp_path)])
        assert result.exit_code == 0
        assert "model.pt" in result.output
        assert "ignore.txt" not in result.output

    def test_detail_with_dict_ckpt(self, tmp_path):
        import torch

        torch.save({"step": 100, "ce_loss": 0.5}, tmp_path / "model.pt")
        runner = CliRunner()
        result = runner.invoke(app, ["list", "checkpoints", str(tmp_path), "--detail"])
        assert result.exit_code == 0
        assert "step=100" in result.output

    def test_detail_with_corrupt_file(self, tmp_path):
        (tmp_path / "bad.pt").write_bytes(b"not a torch file")
        runner = CliRunner()
        result = runner.invoke(app, ["list", "checkpoints", str(tmp_path), "--detail"])
        assert result.exit_code == 0  # 损坏文件被静默跳过

    def test_missing_dir_fails(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(app, ["list", "checkpoints", str(tmp_path / "nope")])
        assert result.exit_code != 0
        assert "目录不存在" in result.output


class TestListDatasets:
    def test_lists_jsonl(self, tmp_path):
        (tmp_path / "a.jsonl").write_text(json.dumps({"text": "x"}) + "\n", encoding="utf-8")
        (tmp_path / "b.jsonl").write_text(json.dumps({"conversations": []}) + "\n", encoding="utf-8")
        (tmp_path / "c.csv").write_text("x", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(app, ["list", "datasets", "-d", str(tmp_path)])
        assert result.exit_code == 0
        assert "a.jsonl" in result.output
        assert "b.jsonl" in result.output
        assert "c.csv" not in result.output

    def test_pattern_filter(self, tmp_path):
        (tmp_path / "train.jsonl").write_text(json.dumps({"text": "x"}) + "\n", encoding="utf-8")
        (tmp_path / "test.jsonl").write_text(json.dumps({"text": "x"}) + "\n", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(app, ["list", "datasets", "-d", str(tmp_path), "-p", "train*.jsonl"])
        assert "train.jsonl" in result.output
        assert "test.jsonl" not in result.output

    def test_no_match(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(app, ["list", "datasets", "-d", str(tmp_path)])
        assert result.exit_code == 0
        assert "没有匹配的文件" in result.output

    def test_missing_dir_fails(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(app, ["list", "datasets", "-d", str(tmp_path / "nope")])
        assert result.exit_code != 0
