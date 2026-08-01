import os

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
