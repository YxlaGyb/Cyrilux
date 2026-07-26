"""
统一双通道数据集 — DualChannelDataset

ch0 = 原始 UTF-8 字节值 (float32)
ch1 = 角色编码: pad=0 / user=1 / assistant=2 / system=3

支持格式:
  1. {"conversations": [{"role":..., "content":...}, ...]}  — 含角色
  2. {"text": "...<|user|>...<|assistant|>...<|end|>"}     — 标记格式
  3. {"text": "..."}                                        — 纯文本 (启发式角色拆分)
  4. {"chosen": [...]}                                      — RLHF chosen
  5. {"conversations": [...], "gt": [...]}                  — 含 ground truth
"""

import json
import re
import torch
from torch.utils.data import Dataset


class DualChannelDataset(Dataset):
    """双通道数据集: ch0=原始字节值, ch1=角色编码 0:pad/1:user/2:assistant/3:system."""

    ROLE_MAP = {"padding": 0, "user": 1, "assistant": 2, "system": 3, "tool": 2}

    def __init__(self, data_path: str, max_length: int = 128, max_samples: int = None):
        super().__init__()
        self.dual_tensors: list[torch.Tensor] = []  # list of [2, max_length] float32
        self.label_tensors: list[torch.Tensor] = []  # list of [max_length] long

        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                sample = json.loads(line)
                raw_text, roles = self._extract_with_roles(sample)
                byte_seq = raw_text.encode("utf-8")[:max_length]
                padded = byte_seq.ljust(max_length, b"\x00")
                byte_raw = (
                    torch.frombuffer(bytearray(padded), dtype=torch.uint8)
                    .clone()
                )

                # 标签先从原始字节提取 [0,255], padding 位置设为 -100
                lbl = byte_raw.clone().long()
                lbl[byte_raw == 0x00] = -100
                self.label_tensors.append(lbl)

                # 归一化: [0, 255] → [-1, 1] (与 ingest_stream/generate_text 一致)
                byte_t = byte_raw.half().div_(128.0).sub_(1.0)

                # 角色编码: 按字节偏移映射, padding 处填 0
                role_t = torch.zeros(max_length, dtype=torch.float16)
                for start, end, role in roles:
                    b_start = len(raw_text[:start].encode("utf-8"))
                    b_end = min(len(raw_text[:end].encode("utf-8")), max_length)
                    if b_start < max_length:
                        role_t[b_start:b_end] = self.ROLE_MAP.get(role, 2.0)

                dual_t = torch.stack([byte_t, role_t], dim=0)  # [2, max_length]
                self.dual_tensors.append(dual_t)

    @staticmethod
    def _extract_with_roles(sample: dict) -> tuple[str, list]:
        """从 sample 提取 (raw_text, [(start, end, role), ...]).

        支持 conversations 格式 (含 role 字段) 和 text 格式 (启发式拆分).
        """
        # ── 格式 1/5: conversations 格式 ──
        if "conversations" in sample:
            raw = ""
            roles = []
            gt = sample.get("gt", None)
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

    def __len__(self):
        return len(self.dual_tensors)

    def __getitem__(self, index):
        # 不再 clone() — 张量在 DataLoader 中会被正确复制到 GPU，clone 是冗余的 CPU 开销
        return self.dual_tensors[index], self.label_tensors[index]


def load_datasets(
    paths: list[str],
    max_length: int = 128,
    max_samples: int = 0,
) -> torch.utils.data.ConcatDataset:
    """加载多个 JSONL 文件并合并为 ConcatDataset。

    Args:
        paths: JSONL 文件路径列表。
        max_length: 最大序列长度。
        max_samples: 每个文件最大样本数 (0 = 全部)。

    Returns:
        ConcatDataset 包装的 DualChannelDataset 列表。
    """
    datasets = []
    for p in paths:
        ds = DualChannelDataset(
            p, max_length=max_length, max_samples=max_samples or None
        )
        if len(ds) > 0:
            datasets.append(ds)
    return torch.utils.data.ConcatDataset(datasets)
