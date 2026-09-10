"""cli resume: 缓冲须按 max(检查点快照, --max-seq-len) 分配.

短序列检查点 + 本命令的长批次 → 形状撞车 (config 快照落地后暴露).
"""

import torch
from click.testing import CliRunner

from model import CyreneModel, DensePCNet
from pkg.cli import app


def test_resume_reconciles_max_seq_len(tmp_path):
    cfg = CyreneModel(d_l4=64, d_l2=32, d_l3=32, d_l5=64, d_l6=16, max_seq_len=16)
    net = DensePCNet(cfg)
    b = torch.randint(0, 255, (1, 16), dtype=torch.long)
    for _ in range(2):
        net.learn(b)
    ckpt = tmp_path / "ck.safetensors"
    net.save(str(ckpt))

    data = tmp_path / "d.jsonl"
    data.write_text("\n".join(f'{{"text": "hello world {i}"}}' for i in range(60)), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["resume", str(ckpt), "-d", str(data), "--max-seq-len", "128",
         "-b", "1", "-o", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "final.safetensors").exists()
