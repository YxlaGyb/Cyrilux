"""
文本模态的几何/拓扑预设

只声明信号容器的形状:
    符号域基数、填充位、可打印域、UTF-8 自动机的字节分类与合法序列约束.
    不含语义:
        无词表与合并规则、无逐符号向量、无语料频率、无语言或字符集标签、无评分器输出, 也没有任何可学习参数.
    只声明形状, 不构造张量: 运行期

缓冲由 `crates.model` 依这些表构造, 因此本模块不 import `torch`.
"""

from __future__ import annotations

ALPHABET = 256
PAD_ID = 0x00
PRINTABLE_START = 0x20

CONTINUATION = (0x80, 0xBF)
OVERLONG_LEADS = (0xC0, 0xC1)
LEAD_RANGES = ((0xC2, 0xDF), (0xE0, 0xEF), (0xF0, 0xF4))
LEAD_CONTINUATIONS = (1, 2, 3)

START_RANGES = ((0x00, 0x7F),) + LEAD_RANGES

CONTINUATION_PREFIX = (0xC0, 0x80)


def is_continuation(b: int) -> bool:
    """续字节，只能出现在字符中途, 不能开启字符"""
    return CONTINUATION[0] <= b <= CONTINUATION[1]


def is_start(b: int) -> bool:
    """合法起始，ASCII 或 2/3/4 字节起始，overlong 0xC0-0xC1 与 0xF5-0xFF 不在内"""
    return any(lo <= b <= hi for lo, hi in START_RANGES)


def lead_length(b: int) -> int:
    """起始字节需要的续字节数，ASCII 与非法起始皆为 0"""
    for n, (lo, hi) in zip(LEAD_CONTINUATIONS, LEAD_RANGES):
        if lo <= b <= hi:
            return n
    return 0


UTF8_BLOCK_CNT = tuple(not is_continuation(b) for b in range(ALPHABET))
UTF8_BLOCK_START = tuple(is_continuation(b) or b in OVERLONG_LEADS for b in range(ALPHABET))
PRINT_MASK = tuple(b >= PRINTABLE_START for b in range(ALPHABET))
