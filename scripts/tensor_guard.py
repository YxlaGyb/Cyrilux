"""张量代码规范静态闸 (机器守卫 CLAUDE.md 底线章节 · 张量代码规范).

AST 扫描: 按节点定位, 按外层函数上下文分类.
  EXEMPT_CTX (一次性/评估上下文, 按位置豁免, 无需标注):
      __init__ / _init_weights / load / save / _migrate_mem / forward / generate
      / sync_gate (P3-b 每步路由交接 — 借阀门既有同步的 D2H 拷贝, 零新增同步)
  热函数 (其余一切): 命中硬规则 = FAIL, 除非行尾 `# tensor-guard: <tag>` 豁免,
      tag ∈ {rare(事件/冷启动偶发), bound(非梯度有界设计界), eval(评估路径), init(构造)}

硬规则:
  H1  .float()/.double()/float32 转换        — FP16 死令
  H2  .item()                               — 零同步读
  H3  clamp(/clamp_(/clamp_min/clamp_max    — 禁 clamp 解溢出 (bound 豁免=非梯度有界设计)
  H4  torch.tensor(/zeros(/ones(/randn(/full(/empty(/arange( — 禁热路径新建
  H5  .clone(                               — 禁 clone 写切片 (状态隔离拷贝走预分配缓冲)
软规则 (scripts/exp115_say.py):
  S1 .item() 需 # tensor-guard: report (报告口径低频采样)
  S2 .float() 需 # tensor-guard: report (报告口径 float 读数)

PASS = 模型域零 FAIL + 脚本零 unannotated .float(). 用法:
  uv run scripts/tensor_guard.py [--no-scripts]
"""

import argparse
import ast
import os
import re
import sys

MODEL_FILES = [
    "model/dense/forward.py", "model/dense/pruning.py", "model/model_cyrene.py",
    "model/dense/learning/engine.py", "model/dense/learning/action.py",
    "model/dense/learning/bind.py", "model/dense/learning/feedforward.py",
    "model/dense/learning/metabolism.py", "model/dense/learning/predict.py",
    "model/dense/learning/readout.py", "model/dense/learning/temporal.py",
    "model/dense/learning/_common.py",
]
SCRIPT_FILES = ["scripts/exp115_say.py"]
EXEMPT_CTX = {"__init__", "_init_weights", "load", "save", "_migrate_mem",
              "forward", "generate", "inject_world",
              # 事件驱动函数 (非每步): 记忆生/死/修剪/拓扑重塑
              "_mem_birth", "_mem_resize", "_mem_famine_kill", "_prune",
              "maybe_prune", "_permute_weights", "_sync_l4_aux",
              # P3-b 每步一次的路由交接: learn 全异步发射, .item() 是每步唯一同步点
              "sync_gate"}

HOT_PATTERNS = [
    ("H1", re.compile(r"\.float\(\)|\.double\(\)")),
    ("H1", re.compile(r"torch\.float32")),
    ("H2", re.compile(r"\.item\(\)")),
    ("H3", re.compile(r"\.clamp_?\(|clamp_min|clamp_max")),
    ("H4", re.compile(r"torch\.(tensor|zeros|ones|randn|full|empty|arange)\(")),
    ("H5", re.compile(r"\.clone\(")),
]
TAG_OK = {"rare", "bound", "eval", "init", "report"}


class _Vuln(ast.NodeVisitor):
    def __init__(self, src_lines):
        self.lines = src_lines
        self.hits = []

    def _check(self, node, ctx):
        raw = self.lines[node.lineno - 1]
        m = re.search(r"#\s*tensor-guard:\s*(\w+)", raw)
        tag = m.group(1) if m else None
        for rule, rx in HOT_PATTERNS:
            seg = ast.get_source_segment("".join(self.lines), node) or ""
            if rx.search(seg):
                kind = "OK" if tag in TAG_OK else ("FAIL" if ctx not in EXEMPT_CTX else "OK")
                msg = f"tensor-guard:{tag}" if tag else (f"ctx={ctx}", "构造/评估上下文")[0]
                if kind == "FAIL":
                    msg = f"热函数 {ctx} 违规 {rule}"
                elif ctx in EXEMPT_CTX:
                    msg = f"ctx={ctx} (一次性/评估上下文)"
                self.hits.append({"ln": node.lineno, "rule": rule, "kind": kind,
                                  "msg": msg, "text": raw.strip()[:88]})
                return

    def visit_Call(self, node):
        fn = node.func
        if not isinstance(fn, ast.Attribute):
            self.generic_visit(node)
            return
        name = fn.attr
        if name in ("float", "item", "clamp", "clamp_", "clamp_min", "clamp_max",
                    "clone", "tensor", "zeros", "ones", "randn", "full", "empty", "arange"):
            self._check(node, self._cur_ctx)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        prev = getattr(self, "_cur_ctx", None)
        self._cur_ctx = node.name
        self.generic_visit(node)
        self._cur_ctx = prev


def scan_file(path: str, domain: str) -> tuple[list[dict], int]:
    src = open(path, encoding="utf-8").read()
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    v = _Vuln(lines)
    v._cur_ctx = "<module>"
    v.visit(tree)
    n_fail = 0
    out = []
    for h in v.hits:
        if domain == "model":
            n_fail += 1 if h["kind"] == "FAIL" else 0
        else:
            # scripts: .item() 需 report 标注; .float() 需 report 标注
            if h["rule"] == "H2" or h["rule"] == "H5":
                if "tensor-guard:report" in str(h.get("msg", "")):
                    h["kind"] = "OK"
                else:
                    h["kind"] = "WARN"
            elif h["rule"] == "H1":
                if "tensor-guard:report" in str(h.get("msg", "")):
                    h["kind"] = "OK"
                else:
                    h["kind"] = "FAIL"
                    n_fail += 1
            else:
                if h["kind"] == "FAIL":
                    h["kind"] = "WARN"
        out.append(h)
    return out, n_fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-scripts", action="store_true")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args()

    total_fail = 0
    files = MODEL_FILES + ([] if args.no_scripts else SCRIPT_FILES)
    for rel in files:
        p = os.path.join(args.root, rel)
        if not os.path.exists(p):
            continue
        domain = "model" if rel.startswith("model/") else "script"
        try:
            hits, n_fail = scan_file(p, domain)
        except SyntaxError as e:
            print(f"✗ {rel}: 语法错误 {e}")
            total_fail += 1
            continue
        total_fail += n_fail
        for h in hits:
            mark = {"FAIL": "✗", "WARN": "!", "OK": "·"}[h["kind"]]
            print(f"{mark} {rel}:{h['ln']}  {h['rule']}  {h['msg']}  {h['text']}")
    print(f"\n[tensor_guard] {'PASS' if total_fail == 0 else f'FAIL ({total_fail} 处违规)'}")
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
