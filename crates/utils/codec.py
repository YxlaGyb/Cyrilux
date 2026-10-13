"""
字节编解码与窗口口径

服从 `crates.modal.text.geometry` 声明的字节容器, 只负责把语料样本搬进该容器.
ByteDataset: jsonl 样本 → 定长字节 id 张量 (按窗宽截断, 余位以 PAD_ID 填充);
窗口纯函数: 样本文本 → UTF-8 平铺字节窗 (stride = 窗宽, 左对齐) → 定长 id 张量.

样本格式:
  1. {"conversations": [{"role":..., "content":...}, ...]}  — 含角色
  2. {"text": "...<|user|>...<|assistant|>...<|end|>"}      — 标记格式
  3. {"text": "..."}                                        — 纯文本 (启发式角色拆分)
  4. {"chosen": [...]}                                      — RLHF chosen
  5. {"conversations": [...], "gt": [...]}                  — 含 ground truth
"""

import json
import re

import torch
from torch.utils.data import ConcatDataset, Dataset

from crates.modal.text.geometry import PAD_ID


class ByteDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = 128, max_samples: int | None = None, lazy: bool = False):
        super().__init__()
        self.data_path = data_path
        self.max_length = max_length
        self.max_samples = max_samples
        self.lazy = lazy
        self.byte_tensors: list[torch.Tensor] = []  # list of [max_length] long

        if lazy:
            # 流式模式: 只记录每行的字节偏移 (二进制读, 字节偏移), 不加载内容
            # 1.2GB 大文件不再一次性读入, __getitem__ 按需 seek 读取单行
            self._offsets: list[int] = []
            with open(data_path, "rb") as f:
                offset = 0
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    self._offsets.append(offset)
                    offset += len(line)
            return

        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                sample = json.loads(line)
                raw_text, _ = self._extract_with_roles(sample)
                byte_seq = encode_text(raw_text)[:max_length]
                padded = byte_seq.ljust(max_length, bytes([PAD_ID]))
                byte_ids = torch.frombuffer(bytearray(padded), dtype=torch.uint8).clone().long()
                self.byte_tensors.append(byte_ids)

    def __len__(self):
        if self.lazy:
            return len(self._offsets)
        return len(self.byte_tensors)

    def __getitem__(self, index):
        if self.lazy:
            # 流式: 二进制模式 seek 字节偏移, 只读一行解码 (UTF-8 seek 会错位)
            with open(self.data_path, "rb") as f:
                f.seek(self._offsets[index])
                line = f.readline().decode("utf-8", errors="replace")
            sample = json.loads(line)
            raw_text, _ = self._extract_with_roles(sample)
            byte_seq = encode_text(raw_text)[: self.max_length]
            padded = byte_seq.ljust(self.max_length, bytes([PAD_ID]))
            return torch.frombuffer(bytearray(padded), dtype=torch.uint8).clone().long()
        # 非 lazy: 张量在 DataLoader 中会被正确复制到 GPU，clone 是冗余的 CPU 开销
        return self.byte_tensors[index]

    @staticmethod
    def _extract_with_roles(sample: dict) -> tuple[str, list]:
        """从 sample 提取 (raw_text, [(start, end, role), ...]).

        支持 conversations 格式 (含 role 字段) 和 text 格式
        """
        # ── 格式 1/5: conversations 格式 ──
        if "conversations" in sample:
            raw = ""
            roles = []
            gt = sample.get("gt")
            for m in sample["conversations"]:
                role = m.get("role", "assistant")
                content = m.get("content", m.get("value", ""))
                start = len(raw)
                raw += content
                end = len(raw)
                roles.append((start, end, role))
            # ground truth 填充
            if gt and isinstance(gt, list) and any(g for g in gt if g):
                answers = "\n".join(str(g) for g in gt if g)
                start = len(raw)
                raw += "\n" + answers
                end = len(raw)
                roles.append((start, end, "assistant"))
            return raw, roles

        # ── 格式 4: chosen 格式 ──
        if "chosen" in sample:
            raw = ""
            roles = []
            for m in sample["chosen"]:
                role = m.get("role", "assistant")
                content = m.get("content", m.get("value", ""))
                start = len(raw)
                raw += content
                end = len(raw)
                roles.append((start, end, role))
            return raw, roles

        # ── 格式 2/3: text 格式 ──
        if "text" in sample:
            raw = sample["text"]
        else:
            raw = str(sample)

        # 1) 优先解析 <|user|>/<|assistant|>/<|system|> 标记
        marker_re = r"(<\|user\|>|<\|assistant\|>|<\|system\|>|<\|end\|>|<\|tool\|>)"
        parts = re.split(marker_re, raw)
        if len(parts) >= 5:  # 至少 user + content + assistant + content + end
            clean_raw = ""
            roles = []
            i = 1  # parts[0] 是标记前的空字符串
            while i < len(parts) - 1:
                tag = parts[i]
                if tag == "<|end|>":
                    break
                if tag in ("<|user|>", "<|assistant|>", "<|system|>", "<|tool|>"):
                    role = tag.strip("<>|")
                    content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                    if content:
                        start = len(clean_raw)
                        clean_raw += content
                        end = len(clean_raw)
                        roles.append((start, end, role))
                    i += 2
                else:
                    i += 1
            if roles:
                return clean_raw, roles

        # 2) 回退: 启发式拆分 — 首行 \n 前 → user, 后 → assistant
        first_nl = raw.find("\n")
        if 0 < first_nl < len(raw) * 0.6:
            roles = [(0, first_nl, "user"), (first_nl, len(raw), "assistant")]
        else:
            roles = [(0, len(raw), "assistant")]
        return raw, roles


def load_datasets(
    paths: list[str],
    max_length: int = 128,
    max_samples: int = 0,
) -> ConcatDataset:
    """加载多个 JSONL 文件并合并为 ConcatDataset。

    Args:
        paths: JSONL 文件路径列表。
        max_length: 最大序列长度。
        max_samples: 每个文件最大样本数 (0 = 全部)。

    Returns:
        ConcatDataset 包装的 ByteDataset 列表。
    """
    datasets = []
    for p in paths:
        ds = ByteDataset(p, max_length=max_length, max_samples=max_samples or None)
        if len(ds) > 0:
            datasets.append(ds)
    return ConcatDataset(datasets)


def encode_text(text: str) -> bytes:
    """文本 → 字节容器的信号序 (UTF-8): 语料文本与提示词一律走此入口."""
    return text.encode("utf-8")


def extract_window_text(sample: dict) -> str:
    """样本 → 训练文本: `text` 字段; 含 `conversations` 则拼接其 content/value."""
    t = sample.get("text", "")
    if "conversations" in sample:
        t = "".join(m.get("content", m.get("value", "")) for m in sample["conversations"])
    return t


def tiled_windows(sample: dict, max_seq_len: int) -> list[bytes]:
    """样本 → 全部字节窗，utf-8 编码后按 stride = max_seq_len 左对齐平铺.

    窗拼接恒等于原文全字节 —— 旧路径 `[:max_seq_len]` 截断丢弃的 65.81% 语料不再丢弃.
    零窗判定与 `_SamplePrefetch` (exp115_say.py) 同源: 空文本 → [] (整窗全零 → 跳过),
    跨路径一致性由 tests/test_zero_window_skip.py 保证. 尾窗短于窗宽, 其 PAD_ID 填充位
    在 `learn` 的目标位掩码 (目标字节低于可打印起点) 下无学习信号.
    """
    bs = encode_text(extract_window_text(sample))
    return [bs[i : i + max_seq_len] for i in range(0, len(bs), max_seq_len)]


def window_ids(bs: bytes, max_seq_len: int) -> torch.Tensor:
    """字节窗 → 定长 id 张量，前 len(bs) 位真实字节, 其余以 PAD_ID 填充 (左对齐)."""
    ids = torch.full((max_seq_len,), PAD_ID, dtype=torch.long)
    ids[: len(bs)] = torch.tensor(list(bs), dtype=torch.long)
    return ids


def skip_zero_windows(samples, max_seq_len: int) -> tuple[list[bytes], int]:
    """批量口径，样本序列 → 保留的全部平铺字节窗, 零窗样本跳过数"""
    kept, skipped = [], 0
    for sample in samples:
        wins = tiled_windows(sample, max_seq_len)
        if not wins:
            skipped += 1
            continue
        kept.extend(wins)
    return kept, skipped
