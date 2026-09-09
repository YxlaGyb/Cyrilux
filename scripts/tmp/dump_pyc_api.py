"""从 .pyc 反解 API 面 (函数名/参数/属性名/全部常量) 到 UTF-8 文本, 用于比对重写是否丢功能."""

from __future__ import annotations

import marshal
import sys
import types
from pathlib import Path


def walk(code: types.CodeType, depth: int = 0, out: list[str] | None = None) -> list[str]:
    if out is None:
        out = []
    pad = "  " * depth
    out.append(f"{pad}CODE {code.co_name}(args={code.co_varnames[:code.co_argcount]})")
    if code.co_names:
        out.append(f"{pad}  names={code.co_names}")
    for k in code.co_consts:
        if isinstance(k, types.CodeType):
            walk(k, depth + 1, out)
        elif isinstance(k, str):
            text = k if len(k) < 300 else k[:300] + "..."
            out.append(f"{pad}  str={text!r}")
        else:
            out.append(f"{pad}  const={k!r}")
    return out


def main() -> None:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    code = marshal.loads(src.read_bytes()[16:])
    dst.write_text("\n".join(walk(code)), encoding="utf-8")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
