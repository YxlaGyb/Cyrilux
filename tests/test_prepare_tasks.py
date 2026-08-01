import json

from pkg.utils.prepare_tasks import (
    HETERO_TASKS,
    conv_to_text,
    extract_conversations,
    prepare_4tasks,
    prepare_hetero,
    sample_jsonl,
    write_jsonl,
)


class TestExtractConversations:
    def test_conversations_content_joined(self):
        sample = {
            "conversations": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ]
        }
        assert extract_conversations(sample) == "q1\na1"

    def test_empty_content_skipped(self):
        sample = {"conversations": [{"role": "user", "content": ""}, {"role": "assistant", "content": "a"}]}
        assert extract_conversations(sample) == "a"

    def test_text_fallback(self):
        assert extract_conversations({"text": "plain"}) == "plain"

    def test_other_falls_back_to_json(self):
        out = extract_conversations({"x": 1})
        assert json.loads(out) == {"x": 1}


class TestConvToText:
    def test_roles_prefixed(self):
        conv = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
        assert conv_to_text(conv) == "user: hi\nassistant: yo"

    def test_empty_system_skipped(self):
        conv = [{"role": "system", "content": ""}, {"role": "user", "content": "hi"}]
        assert conv_to_text(conv) == "user: hi"

    def test_trailing_empty_assistant_with_gt(self):
        conv = [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""}]
        assert conv_to_text(conv, gt=["answer"]) == "user: q\nassistant:\nanswer"

    def test_trailing_empty_assistant_no_gt(self):
        conv = [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""}]
        assert conv_to_text(conv) == "user: q"

    def test_empty_assistant_mid_turn_kept(self):
        conv = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "q2"},
        ]
        assert conv_to_text(conv) == "user: q\nassistant: \nuser: q2"


class TestSampleJsonl:
    def _write(self, path, n=10):
        lines = [json.dumps({"text": f"t{i}"}) for i in range(n)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_samples_n(self, tmp_path):
        src = tmp_path / "d.jsonl"
        self._write(src)
        samples = sample_jsonl(str(src), 3)
        assert len(samples) == 3

    def test_fewer_lines_returns_all(self, tmp_path):
        src = tmp_path / "d.jsonl"
        self._write(src, n=2)
        samples = sample_jsonl(str(src), 5)
        assert len(samples) == 2

    def test_seed_reproducible(self, tmp_path):
        src = tmp_path / "d.jsonl"
        self._write(src, n=50)
        a = sample_jsonl(str(src), 10, seed=7)
        b = sample_jsonl(str(src), 10, seed=7)
        assert a == b
        c = sample_jsonl(str(src), 10, seed=8)
        assert a != c


class TestWriteJsonl:
    def test_writes_text(self, tmp_path):
        dst = tmp_path / "out.jsonl"
        write_jsonl([{"conversations": [{"role": "user", "content": "hi"}]}, {"text": "plain"}], str(dst))
        lines = dst.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"text": "hi"}
        assert json.loads(lines[1]) == {"text": "plain"}


def _write_source_files(tmp_path, names, n=6):
    for name in names:
        lines = [json.dumps({
            "conversations": [{"role": "user", "content": f"{name}-{i}"}]
        }) for i in range(n)]
        (tmp_path / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestPrepare4Tasks:
    def test_creates_task_files(self, tmp_path):
        _write_source_files(
            tmp_path,
            ["pretrain_t2t_mini.jsonl", "lora_exam.jsonl", "agent_rl_math.jsonl", "sft_t2t_mini.jsonl"],
        )
        results = prepare_4tasks(data_dir=str(tmp_path), output_dir=str(tmp_path), n_per_task=4)
        assert set(results.keys()) == {"task_a_daily", "task_b_tech", "task_c_medical", "task_d_sft"}
        for task_name in results:
            out = tmp_path / f"{task_name}_20k.jsonl"
            assert out.exists()
            lines = out.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 4
            # write_jsonl 用 extract_conversations → 内容来自某个源文件, 不含角色标记前缀
            for line in lines:
                text = json.loads(line)["text"]
                assert any(
                    text.startswith(f"{name}.jsonl-")
                    for name in ("pretrain_t2t_mini", "lora_exam", "agent_rl_math", "sft_t2t_mini")
                )

    def test_skips_missing_sources(self, tmp_path):
        _write_source_files(tmp_path, ["pretrain_t2t_mini.jsonl"])  # 只有 1 个源文件
        results = prepare_4tasks(data_dir=str(tmp_path), output_dir=str(tmp_path), n_per_task=2)
        assert set(results.keys()) == {"task_a_daily"}
        assert (tmp_path / "task_b_tech_20k.jsonl").exists() is False


class TestPrepareHetero:
    def test_converts_all_tasks(self, tmp_path):
        _write_source_files(tmp_path, list(HETERO_TASKS.values()))
        results = prepare_hetero(data_dir=str(tmp_path), output_dir=str(tmp_path))
        assert set(results.keys()) == set(HETERO_TASKS.keys())
        for task_id in HETERO_TASKS:
            out = tmp_path / f"task_{task_id}.jsonl"
            assert out.exists()
            lines = out.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 6
            assert json.loads(lines[0])["text"].startswith("user: ")

    def test_gt_filled_into_empty_assistant(self, tmp_path):
        # conv_to_text: 结尾空 assistant + 有 gt → gt 填充到回复
        lines = [
            json.dumps(
                {"conversations": [
                    {"role": "user", "content": "q"}, {"role": "assistant", "content": ""}
                        ], "gt": ["answer"]
                })
            for _ in range(3)
        ]
        (tmp_path / "agent_rl_math.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        results = prepare_hetero(data_dir=str(tmp_path), output_dir=str(tmp_path))
        assert results["a"] == 3
        out = (tmp_path / "task_a.jsonl").read_text(encoding="utf-8")
        assert "answer" in out

    def test_skips_missing_sources(self, tmp_path):
        _write_source_files(tmp_path, ["agent_rl_math.jsonl"])  # 只有 1 个源文件
        results = prepare_hetero(data_dir=str(tmp_path), output_dir=str(tmp_path))
        assert set(results.keys()) == {"a"}
        assert (tmp_path / "task_b.jsonl").exists() is False
