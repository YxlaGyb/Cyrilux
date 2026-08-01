import json

from click.testing import CliRunner

from pkg.cli import app


class TestConfigInit:
    def test_train_template(self, tmp_path):
        out = tmp_path / "cfg.json"
        runner = CliRunner()
        result = runner.invoke(app, ["config", "init", "-o", str(out), "-t", "train"])
        assert result.exit_code == 0
        cfg = json.loads(out.read_text(encoding="utf-8"))
        assert cfg["batch_size"] == 48
        assert cfg["lr"] == 3e-4

    def test_autonomous_template(self, tmp_path):
        out = tmp_path / "cfg.json"
        runner = CliRunner()
        result = runner.invoke(app, ["config", "init", "-o", str(out), "-t", "autonomous"])
        assert result.exit_code == 0
        cfg = json.loads(out.read_text(encoding="utf-8"))
        assert cfg["wake_steps"] == 20
        assert cfg["play_steps"] == 100

    def test_unknown_template_fails(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(app, ["config", "init", "-o", str(tmp_path / "x.json"), "-t", "bogus"])
        assert result.exit_code != 0
        assert "未知模板类型" in result.output


class TestConfigShow:
    def test_shows_json(self, tmp_path):
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps({"a": 1}), encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(app, ["config", "show", str(cfg_path)])
        assert result.exit_code == 0
        assert json.loads(result.output) == {"a": 1}

    def test_missing_file_fails(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(app, ["config", "show", str(tmp_path / "nope.json")])
        assert result.exit_code != 0
        assert "文件不存在" in result.output


class TestConfigValidate:
    def test_valid_config_passes(self, tmp_path):
        cfg = {
            "training": {"batch_size": 48, "lr": 3e-4},
            "model": {"hidden_size": 256},
        }
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(app, ["config", "validate", str(cfg_path)])
        assert result.exit_code == 0
        assert "通过验证" in result.output

    def test_missing_training_warns(self, tmp_path):
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps({"model": {"hidden_size": 256}}), encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(app, ["config", "validate", str(cfg_path)])
        assert result.exit_code == 0
        assert "缺少 'training' 字段" in result.output

    def test_bad_values_warn(self, tmp_path):
        cfg = {"training": {"batch_size": 300, "lr": 0.1}}
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(app, ["config", "validate", str(cfg_path)])
        assert "batch_size > 160" in result.output
        assert "lr > 0.01" in result.output

    def test_odd_hidden_size_warns(self, tmp_path):
        cfg = {"training": {"batch_size": 48}, "model": {"hidden_size": 333}}
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(app, ["config", "validate", str(cfg_path)])
        assert "非常规值" in result.output

    def test_invalid_json_fails(self, tmp_path):
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text("{bad json", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(app, ["config", "validate", str(cfg_path)])
        assert result.exit_code != 0
        assert "JSON 格式错误" in result.output
