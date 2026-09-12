import os
import re

import pkg.cli.utils as u
from pkg.cli.utils import PROJECT_ROOT, load_config, merge_config, resolve_path, save_config


class TestResolvePath:
    def test_absolute_passthrough(self):
        p = "C:/some/abs/path.json"
        assert resolve_path(p) == p

    def test_relative_joined_to_project_root(self):
        out = resolve_path("out/x.json")
        assert os.path.normpath(out).startswith(os.path.normpath(PROJECT_ROOT))


class TestMergeConfig:
    def test_overrides_non_none(self):
        merged = merge_config({"a": 1, "b": 2}, {"a": 10, "c": 3})
        assert merged == {"a": 10, "b": 2, "c": 3}

    def test_none_does_not_override(self):
        merged = merge_config({"a": 1}, {"a": None, "b": None})
        assert merged == {"a": 1}

    def test_original_not_mutated(self):
        orig = {"a": 1}
        merge_config(orig, {"a": 99})
        assert orig == {"a": 1}


class TestLoadSaveConfig:
    def test_roundtrip(self, tmp_path):
        cfg = {"model": {"hidden_size": 256}, "lr": 1e-4}
        path = str(tmp_path / "cfg.json")
        save_config(cfg, path)
        assert load_config(path) == cfg

    def test_load_relative_to_project_root(self, tmp_path):
        # 相对路径解析到项目根，绝对路径直接用
        path = str(tmp_path / "cfg.json")
        save_config({"a": 1}, path)
        assert load_config(path) == {"a": 1}


class TestRunDir:
    def _pin(self, tmp_path, monkeypatch):
        monkeypatch.setattr(u, "PROJECT_ROOT", str(tmp_path))
        monkeypatch.setattr(u, "_RUN_DIR", None)

    def test_first_run_is_v1_with_timestamp(self, tmp_path, monkeypatch):
        self._pin(tmp_path, monkeypatch)
        d = u.run_dir()
        assert re.match(r"^v1-\d{8}-\d{6}$", os.path.basename(d))
        assert os.path.isdir(d)

    def test_increments_past_existing_dirs(self, tmp_path, monkeypatch):
        self._pin(tmp_path, monkeypatch)
        out = tmp_path / "out"
        out.mkdir()
        (out / "v3-20260101-000000").mkdir()
        (out / "v10-20260101-000000").mkdir()
        (out / "prof").mkdir()
        (out / "v99-not-a-dir").write_text("x", encoding="utf-8")  # 只认目录
        d = u.run_dir()
        assert os.path.basename(d).startswith("v11-")

    def test_memoized_and_run_file_inside(self, tmp_path, monkeypatch):
        self._pin(tmp_path, monkeypatch)
        assert u.run_dir() == u.run_dir()
        assert os.path.dirname(u.run_file("a.json")) == u.run_dir()

    def test_pin_run_dir_resume_reuses_line_dir(self, tmp_path, monkeypatch):
        # 续跑锚定: pin 后 run_file 落进既有训练线目录, 不新开版本
        self._pin(tmp_path, monkeypatch)
        out = tmp_path / "out"
        (out / "v3-20260911-115631").mkdir(parents=True)
        u.pin_run_dir(str(out / "v3-20260911-115631"))
        assert u.run_dir() == str(out / "v3-20260911-115631")
        assert os.path.dirname(u.run_file("w.safetensors")) == str(out / "v3-20260911-115631")
