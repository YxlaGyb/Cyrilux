import json

from pkg.utils.data_converter import (
    ROLE_MARKERS,
    conversations_to_text,
    convert_file,
    convert_sample,
)


class TestConversationsToText:
    def test_single_turn(self):
        text = conversations_to_text([{"role": "user", "content": "hi"}])
        assert text == f"{ROLE_MARKERS['user']}\nhi\n<|end|>"

    def test_multiple_turns(self):
        conv = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        text = conversations_to_text(conv)
        assert text.startswith(f"{ROLE_MARKERS['user']}\nq1")
        assert f"{ROLE_MARKERS['assistant']}\na1" in text
        assert text.endswith("<|end|>")

    def test_unknown_role_falls_back_to_user(self):
        text = conversations_to_text([{"role": "nobody", "content": "x"}])
        assert text.startswith(ROLE_MARKERS["user"])

    def test_empty_content_keeps_marker(self):
        text = conversations_to_text([{"role": "system", "content": ""}])
        assert text == f"{ROLE_MARKERS['system']}\n<|end|>"

    def test_gt_appended(self):
        text = conversations_to_text([{"role": "user", "content": "1+1?"}], gt_list=[2])
        assert f"{ROLE_MARKERS['assistant']}\n2" in text


class TestConvertSample:
    def test_text_passthrough(self):
        line = json.dumps({"text": "hello"})
        assert convert_sample(line) == line

    def test_conversations(self):
        line = json.dumps({"conversations": [{"role": "user", "content": "hi"}]})
        out = json.loads(convert_sample(line))
        assert out["text"].startswith(ROLE_MARKERS["user"])
        assert out["text"].endswith("<|end|>")

    def test_conversations_with_gt(self):
        line = json.dumps({"conversations": [{"role": "user", "content": "q"}], "gt": [42]})
        out = json.loads(convert_sample(line))
        assert ROLE_MARKERS["assistant"] in out["text"]
        assert "42" in out["text"]

    def test_chosen(self):
        line = json.dumps({"chosen": [{"role": "user", "content": "hi"}]})
        out = json.loads(convert_sample(line))
        assert out["text"].startswith(ROLE_MARKERS["user"])

    def test_invalid_json_returns_none(self):
        assert convert_sample("{not json") is None

    def test_unknown_format_returns_none(self):
        assert convert_sample(json.dumps({"weird": 1})) is None

    def test_empty_line_returns_none(self):
        assert convert_sample("") is None


class TestConvertFile:
    def _write_samples(self, path):
        lines = [
            json.dumps({"conversations": [{"role": "user", "content": "hi"}]}),
            "not json",
            json.dumps({"text": "already"}),  # 原样透传
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_convert_counts(self, tmp_path):
        src = tmp_path / "in.jsonl"
        dst = tmp_path / "out.jsonl"
        self._write_samples(src)
        count, skipped = convert_file(str(src), str(dst))
        assert count == 2  # conversations + text passthrough
        assert skipped == 1  # invalid json
        out_lines = dst.read_text(encoding="utf-8").strip().split("\n")
        assert len(out_lines) == 2

    def test_max_samples(self, tmp_path):
        src = tmp_path / "in.jsonl"
        dst = tmp_path / "out.jsonl"
        lines = [json.dumps({"text": f"t{i}"}) for i in range(10)]
        src.write_text("\n".join(lines) + "\n", encoding="utf-8")
        count, _ = convert_file(str(src), str(dst), max_samples=3)
        assert count == 3
