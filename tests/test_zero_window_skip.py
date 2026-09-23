"""零窗跳过测试 (批令 3/34, v1.12–v1.13): 两条训练路径的判据 + 跨路径一致性.

判据 = 全零窗 (空文本行经 ljust 填充) 不得进入训练流 —— 代谢会把"看不见"记成
"看见全零", 伪造 F 基线 / starve_cnt; 且跳过计数必须与实际跳过数一致 (审计口径).

覆盖范围:
  - 流式路径 `_SamplePrefetch` (exp115_say.py);
  - 文件遍历路径 `window_or_skip` / `skip_zero_windows` (train_cmd.py, 批令 34 抽出的纯函数);
  - 跨路径一致性: 同一批样本上两者的零窗判定必须等价 (ByteDataset 张量口径 vs 纯函数口径).
"""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from exp115_say import _SamplePrefetch  # noqa: E402  (需先注入 scripts/ 到 sys.path)
from crates.cli.commands.train_cmd import skip_zero_windows, window_or_skip  # noqa: E402


class _StubDS:
    """桩数据集: 按索引返回预置窗口 (可含全零窗)."""

    def __init__(self, windows):
        self._windows = windows

    def __getitem__(self, index):
        return self._windows[index]


def _win(values):
    t = torch.zeros(8, dtype=torch.long)
    if values:
        t[: len(values)] = torch.tensor(values, dtype=torch.long)
    return t


def test_prefetch_skips_zero_windows_and_counts():
    """6 窗含 3 个零窗: 取 8 次 → 全部非零窗, 计数 ∈ [7,10] (下界=交付所需, 上界=预取深度 1 个额外窗)."""
    windows = [_win([65, 66]), _win([]), _win([67, 68]), _win([]), _win([]), _win([69])]
    pf = _SamplePrefetch(_StubDS(windows), list(range(len(windows))), torch.device("cpu"),
                         start=0, depth=1)
    out = [pf.get() for _ in range(8)]
    assert all(int(x.sum()) > 0 for x in out), "零窗进入了训练流"
    n_skipped = pf.zero_win_skipped
    assert 7 <= n_skipped <= 10, f"zero_win_skipped={n_skipped} 越界 (期望 7..10)"


def test_prefetch_counter_zero_when_no_zero_window():
    """无零窗语料: 计数恒 0, 采样序列与 idxs 同序."""
    windows = [_win([65]), _win([66]), _win([67])]
    pf = _SamplePrefetch(_StubDS(windows), list(range(len(windows))), torch.device("cpu"),
                         start=0, depth=1)
    out = [int(pf.get()[0, 0]) for _ in range(4)]
    assert out == [65, 66, 67, 65]
    assert pf.zero_win_skipped == 0


def test_prefetch_all_zero_windows_raises():
    """全零窗语料: 护栏触发 (RuntimeError 经队列重抛), 不静默挂死."""
    windows = [_win([]), _win([])]
    pf = _SamplePrefetch(_StubDS(windows), list(range(len(windows))), torch.device("cpu"),
                         start=0, depth=1)
    with pytest.raises(RuntimeError):
        pf.get()


# ---------------------------------------------------------------- train_cmd 纯函数 (批令 34)


def test_window_or_skip_text_field():
    """text 字段: 空串 → None; 纯空白 → 保留 (字节非零); 超长 → 截断到 max_seq_len."""
    assert window_or_skip({"text": ""}, 8) is None
    assert window_or_skip({"text": "  "}, 8) == b"  "
    assert window_or_skip({"text": "abcdefghijk"}, 8) == b"abcdefgh"
    assert window_or_skip({"text": "你好"}, 8) == "你好".encode("utf-8")


def test_window_or_skip_conversations_field():
    """conversations 字段: 全空 content → None; 有内容 → 拼接保留."""
    empty = {"conversations": [{"role": "user", "content": ""}, {"role": "assistant", "value": ""}]}
    assert window_or_skip(empty, 8) is None
    some = {"conversations": [{"role": "user", "content": "ab"}, {"role": "assistant", "value": "cd"}]}
    assert window_or_skip(some, 8) == b"abcd"


def test_skip_zero_windows_count():
    """批量口径: 计数与保留数之和等样本数, 零窗恰好被剔除."""
    samples = [{"text": "aa"}, {"text": ""}, {"text": "bb"}, {"text": ""}, {"conversations": []}]
    kept, skipped = skip_zero_windows(samples, 8)
    assert kept == [b"aa", b"bb"]
    assert skipped == 3
    assert len(kept) + skipped == len(samples)


def test_cross_path_judge_equivalence():
    """跨路径一致性: ByteDataset 张量口径 ((t==0).all()) ⟺ 纯函数口径 (None) 必须等价.

    构造 ByteDataset 同款窗口 (bytes[:L].ljust(L, b"\\x00")), 两侧判定逐条比对 ——
    流式路径与文件遍历路径共用同一零窗语义, 由本用例保证.
    """
    texts = ["", "a", "", "  ", "你好", "", "\n", "\t"]
    for t in texts:
        ids = torch.zeros(16, dtype=torch.long)
        bs = t.encode("utf-8")[:16]
        ids[: len(bs)] = torch.tensor(list(bs), dtype=torch.long)
        tensor_is_zero = bool((ids == 0).all())
        pure_says_skip = window_or_skip({"text": t}, 16) is None
        assert tensor_is_zero == pure_says_skip, f"判据分歧: text={t!r}"
