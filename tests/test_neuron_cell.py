"""
单细胞前向测试

覆盖 形状/dtype 契约, alive 门控与收入死亡, 四点 XOR 在 10x 步数下收敛到干净值
(RTR 学习规则原样调用, 价值账本速率是夹具标定量) 与 cell.py 约束扫描.
"""

import ast
from pathlib import Path

import torch

from crates.kernels import rms_norm
from crates.model.constants import ELIG_GAMMA
from crates.neuron import Cell
from crates.rtr import elig_accum, rtr_apply, rtr_residual

_CELL = Path(__file__).resolve().parent.parent / "crates" / "neuron" / "cell.py"
_SELF = Path(__file__).resolve()
_BANNED = (
    ".float(",
    ".double(",
    ".item(",
    ".reshape(",
    "gather(",
    "scatter",
    "nonzero",
    "sparse",
    "float32",
    "bfloat16",
)
_XOR_X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float16)
_XOR_T = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float16)
_STEPS = 20000
_WARMUP = 2000
_ETA = 0.05
_ETA_V = 0.01


def test_contract():
    cell = Cell(2, 128, 1)
    y = cell.forward(torch.randn(4, 2, dtype=torch.float16))
    assert cell.W1.shape == (128, 2) and cell.W1.dtype == torch.float16
    assert cell.W2.shape == (1, 128) and cell.W2.dtype == torch.float16
    assert cell.alive.shape == (128,) and cell.alive.dtype == torch.float16
    assert cell._income.shape == (128,) and cell._death_cnt.shape == (128,)
    assert y.shape == (4, 1) and y.dtype == torch.float16
    assert not cell.W2.any()
    assert (cell.alive == 1).all()


def test_alive_gates_forward():
    cell = Cell(2, 128, 1)
    cell.W2.fill_(1.0)
    x = torch.ones(4, 2, dtype=torch.float16)
    assert cell.forward(x).any()
    cell.alive.zero_()
    assert not cell.forward(x).any()


def test_update_alive_kills_starved_units_permanently():
    torch.manual_seed(0)
    cell = Cell(2, 128, 1)
    cell.W2.fill_(0.0)
    cell.W2[:, :16] = 1.0
    x = torch.ones(4, 2, dtype=torch.float16)
    m = torch.ones(1, dtype=torch.float16)
    for _ in range(cell.death_steps + 1):
        cell.update_alive(x, m)
    alive = cell.alive.clone()
    assert 0.0 < alive.mean() < 1.0
    for _ in range(100):
        cell.update_alive(x, m)
    assert torch.equal(cell.alive, alive)
    print(f"[neuron.cell] starved alive.mean()={alive.mean()}")


def test_xor_converges_clean_with_survival():
    torch.manual_seed(0)
    cell = Cell(2, 128, 1)
    E1 = torch.zeros_like(cell.W1)
    E2 = torch.zeros_like(cell.W2)
    ledger = torch.zeros(cell.alive.shape[0], dtype=torch.float16)
    alive_log = []
    for step in range(_STEPS):
        a = torch.sigmoid(_XOR_X @ cell.W1.T)
        am = a * cell.alive
        y = cell.forward(_XOR_X)
        loss = (y - _XOR_T).square().mean()
        E1 = elig_accum(E1, (a.T @ rms_norm(_XOR_X)) / _XOR_X.shape[0], ELIG_GAMMA)
        E2 = elig_accum(E2, ((_XOR_T - y).T @ rms_norm(am)) / _XOR_X.shape[0], ELIG_GAMMA)
        m = rtr_residual(ledger, am.mean(dim=0), 1.0 - loss, _ETA_V)
        cell.W1 += rtr_apply(E1, m) * _ETA
        cell.W2 += rtr_apply(E2, m) * _ETA
        if step >= _WARMUP:
            cell.update_alive(_XOR_X, m)
        if step % 2000 == 0:
            alive_log.append(float(cell.alive.mean()))
    y = cell.forward(_XOR_X)
    alive_log.append(float(cell.alive.mean()))
    print(f"[neuron.cell] alive.mean()={alive_log} xor_loss={loss} y={y.flatten().tolist()}")
    assert y[0, 0] < 0.15 and y[3, 0] < 0.15
    assert y[1, 0] > 0.85 and y[2, 0] > 0.85
    assert 0.0 < alive_log[-1] < 1.0
    assert alive_log[-1] == alive_log[-2]


def test_cell_source_has_no_escape_hatches():
    src = _CELL.read_text(encoding="utf-8")
    for tok in _BANNED:
        assert tok not in src, tok
    assert not any(isinstance(node, (ast.If, ast.IfExp)) for node in ast.walk(ast.parse(src)))


def test_xor_path_has_no_autograd():
    tokens = ("optimizer", "backward", ".step(")
    src = _SELF.read_text(encoding="utf-8").split("def test_xor_path_has_no_autograd")[0]
    for tok in tokens:
        assert tok not in src, tok
    cell_src = _CELL.read_text(encoding="utf-8")
    for tok in tokens:
        assert tok not in cell_src, tok
