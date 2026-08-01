import json

from pkg.utils.data_splitter import split_directory, split_file


class TestSplitFile:
    def _write_input(self, path, n=5):
        lines = [json.dumps({"text": f"line{i}"}) for i in range(n)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_chunking(self, tmp_path):
        src = tmp_path / "data.jsonl"
        out = tmp_path / "out"
        self._write_input(src, n=5)
        result = split_file(str(src), str(out), chunk_size=2)
        assert result["total_lines"] == 5
        assert result["n_chunks"] == 3
        assert len(result["output_files"]) == 3
        sizes = [len(open(f, encoding="utf-8").read().strip().split("\n")) for f in result["output_files"]]
        assert sizes == [2, 2, 1]

    def test_dry_run_no_files(self, tmp_path):
        src = tmp_path / "data.jsonl"
        out = tmp_path / "out"
        self._write_input(src, n=4)
        result = split_file(str(src), str(out), chunk_size=2, dry_run=True)
        assert result["n_chunks"] == 2
        assert result["output_files"] == []
        assert not list(tmp_path.glob("data_part*.jsonl"))

    def test_convert_off_keeps_raw(self, tmp_path):
        src = tmp_path / "data.jsonl"
        out = tmp_path / "out"
        lines = [json.dumps({"conversations": [{"role": "user", "content": "hi"}]}) for _ in range(3)]
        src.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = split_file(str(src), str(out), chunk_size=10, convert=False)
        first_line = open(result["output_files"][0], encoding="utf-8").read().strip().split("\n")[0]
        kept = json.loads(first_line)
        assert "conversations" in kept

    def test_bad_sample_skipped_when_converting(self, tmp_path):
        src = tmp_path / "data.jsonl"
        out = tmp_path / "out"
        lines = [
            json.dumps({"conversations": [{"role": "user", "content": "hi"}]}),
            "not json",
            json.dumps({"text": "fine"}),
        ]
        src.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = split_file(str(src), str(out), chunk_size=10)
        assert result["skipped"] == 1
        assert len(open(result["output_files"][0], encoding="utf-8").read().strip().split("\n")) == 2

    def test_empty_input(self, tmp_path):
        src = tmp_path / "empty.jsonl"
        out = tmp_path / "out"
        src.write_text("", encoding="utf-8")
        result = split_file(str(src), str(out), chunk_size=10)
        assert result["n_chunks"] == 0
        assert result["total_lines"] == 0


class TestSplitDirectory:
    def test_multi_file(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        for name in ("a.jsonl", "b.jsonl"):
            lines = [json.dumps({"text": f"{name}-{i}"}) for i in range(5)]
            (in_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

        out_dir = tmp_path / "out"
        results = split_directory(str(in_dir), str(out_dir), chunk_size=10)
        assert set(results.keys()) == {"a.jsonl", "b.jsonl"}
        for name, r in results.items():
            assert r["total_lines"] == 5
            assert r["n_chunks"] == 1
