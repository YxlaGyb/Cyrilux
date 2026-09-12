"""
pkg/outver 版本目录管理单元测试
扫描 / 自增 / 时间戳格式 / 最新产物目录定位.
"""
import os
import re

from pkg.outver import ensure_run_dir, latest_run_dir_with, next_run_num, scan_run_nums

RUN_DIR_RE = re.compile(r"^v(\d+)-\d{8}-\d{6}$")


def _mk(out, *names):
    os.makedirs(out, exist_ok=True)
    for n in names:
        os.makedirs(os.path.join(out, n), exist_ok=True)


class TestScanRunNums:
    def test_version_dirs_only(self, tmp_path):
        out = str(tmp_path)
        _mk(out, "v3-20260911-115631", "v10-20260101-000000", "prof")
        assert sorted(scan_run_nums(out)) == [3, 10]

    def test_ignores_non_matching(self, tmp_path):
        # 纯 semver (v3-0-1) / 缺时间戳段的目录不算版本目录
        out = str(tmp_path)
        _mk(out, "v3-0-1", "v3-notime")
        assert scan_run_nums(out) == []

    def test_ignores_files(self, tmp_path):
        out = str(tmp_path)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "v5-20260101-000000"), "w", encoding="utf-8") as f:
            f.write("x")  # 同名文件 (非目录) 不计
        assert scan_run_nums(out) == []


class TestNextRunNum:
    def test_increments_past_max(self, tmp_path):
        out = str(tmp_path)
        _mk(out, "v3-20260911-115631", "v10-20260101-000000")
        assert next_run_num(out) == 11

    def test_no_history_starts_at_1(self, tmp_path):
        assert next_run_num(str(tmp_path)) == 1


class TestEnsureRunDir:
    def test_creates_with_timestamp_format(self, tmp_path):
        out = str(tmp_path)
        d1 = ensure_run_dir(out)
        d2 = ensure_run_dir(out)
        assert RUN_DIR_RE.match(os.path.basename(d1))
        assert RUN_DIR_RE.match(os.path.basename(d2))
        assert d1 != d2  # 时间戳保证不同
        assert os.path.isdir(d2)

    def test_n_is_max_plus_one(self, tmp_path):
        out = str(tmp_path)
        _mk(out, "v3-20260911-115631")
        d = ensure_run_dir(out)
        assert os.path.basename(d).startswith("v4-")


class TestLatestRunWith:
    def test_highest_n_wins(self, tmp_path):
        out = str(tmp_path)
        _mk(out, "v3-20260911-115631", "v5-20260911-120000")
        with open(os.path.join(out, "v3-20260911-115631", "sc.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        # v5 无该文件 → 回落 v3
        assert os.path.basename(latest_run_dir_with(out, "sc.json")) == "v3-20260911-115631"
        with open(os.path.join(out, "v5-20260911-120000", "sc.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        assert os.path.basename(latest_run_dir_with(out, "sc.json")) == "v5-20260911-120000"

    def test_none_when_absent(self, tmp_path):
        assert latest_run_dir_with(str(tmp_path), "sc.json") is None
