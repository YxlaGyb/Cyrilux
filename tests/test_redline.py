"""
语言通用性红线守卫 (R0 红线固化轮).

锁三件事:
1. model/ 活动源码零红线 token (断链不可复辟, 体外账本死代码不可复生);
2. 资格迹信号源只允许体内 _metab_R (action.py 源码级断言);
3. world_lang 仪表无 step_E (与 test_world_lang.py 双保险).
"""
import importlib.util
import inspect
from pathlib import Path

_spec = importlib.util.spec_from_file_location("redline", "scripts/redline.py")
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
REDLINE_TOKENS = _mod.REDLINE_TOKENS
scan_model_sources = _mod.scan_model_sources


def test_model_sources_zero_redline_tokens():
    hits = scan_model_sources()
    assert hits == [], f"红线 token 命中 (断链复辟/死代码复生): {hits}"


def test_redline_tokens_nonempty():
    # 防扫描器失活: token 清单必须非空且含核心禁词
    assert "step_E" in REDLINE_TOKENS
    assert "inject_world" in REDLINE_TOKENS


def test_action_survival_source_is_internal():
    # 资格迹 R 来源只允许体内 _metab_R; 注入通道源码级零容忍
    from model.dense.learning.action import ActionMixin

    src = inspect.getsource(ActionMixin._update_w_act)
    assert "_metab_R" in src, "资格迹信号源未走体内 _metab_R"
    for tok in ("step_E", "inject_world", "_world_R"):
        assert tok not in src, f"资格迹源码出现红线 token: {tok}"


def test_world_lang_module_isolated_from_model():
    # 核心库 (model/) 任何文件不得 import world_lang (报告仪表不进核心库)
    model_root = Path(__file__).resolve().parent.parent / "model"
    for p in model_root.rglob("*.py"):
        if "_archived_sparse" in p.parts:
            continue
        assert "world_lang" not in p.read_text(encoding="utf-8"), f"{p} 引用 world_lang"
