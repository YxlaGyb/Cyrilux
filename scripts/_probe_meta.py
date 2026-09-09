"""探针产物自解释元数据: 让每个 out/*.json 说清"我是在什么配置下测的".

动机: 旧结论腐坏的根因是产物与代码/口径脱钩.
缺了这层元数据的数字, 三个月后无法判断是否适用于当前代码 — 必须自带配置指纹.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime

import torch


def _run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return r.stdout.strip().splitlines()[0] if r.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def config_meta(**kwargs: object) -> dict[str, object]:
    """配置指纹 + 探针参数. kwargs 覆盖/追加任意字段."""
    meta: dict[str, object] = {
        "captured_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "git_commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "driver": _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "argv": sys.argv,
    }
    try:
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            meta["gpu"] = {
                "name": p.name,
                "sm_count": p.multi_processor_count,
                "total_mib": round(p.total_memory / 1024**2),
                "cap": f"{p.major}.{p.minor}",
            }
    except Exception as e:
        meta["gpu"] = f"unavailable: {type(e).__name__}"
    meta.update(kwargs)
    return meta
