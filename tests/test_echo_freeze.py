"""
C5 回声冻结合约测试

echo 步冻结集逐位不变 + 表达端 W_act 推进; free_run 内生动力学保持.
冻结名单与 engine.py 权威注释块同款 (单一出处); 判据 = 逐位相等 (守卫零泄漏进权重).
"""

import torch

from model import CyreneModel, DensePCNet

_FROZEN_WEIGHTS = (
    "W_04", "W_42", "W_23", "W_35", "W_56", "W_state_pred", "W_pred_54", "W_pred_43",
    "W_diff", "b_diff", "W_t4", "W_t2", "W_t3", "W_t5", "W_t6", "M_l5",
    "W_lm", "W_lm_2", "W1", "bias_lm", "W_bind", "W_bind_self",
    "bias_l4", "bias_l2", "bias_l3", "bias_l5", "bias_l6",
)
_FROZEN_STATE = (
    "_theta_l5", "_theta_wt4", "_theta_w04", "_theta_novelty",
    "_mem_g", "_mem_err_ema", "_lang_eps_ema",
    "_active_ema_w35", "_active_ema_w56", "W_35_elig", "W_23_elig", "W_56_elig",
    "E_23", "E_l5",
)
# _mem_m 不在冻结集: forward 侧泄漏积分每步照走 (合约边界条款)


def _make_net():
    torch.manual_seed(0)
    cfg = CyreneModel(
        d_l4=64, d_l2=32, d_l3=32, d_l5=64, d_l6=16,
        max_seq_len=16, free_run_window=16, mem_k0=2,
    )
    return DensePCNet(cfg)


def _snap(net, names):
    out = {}
    for n in names:
        t = getattr(net, n)
        out[n] = (t.data if isinstance(t, torch.nn.Parameter) else t).clone()
    return out


def _assert_same(net, snap):
    for n, w in snap.items():
        t = getattr(net, n)
        cur = t.data if isinstance(t, torch.nn.Parameter) else t
        assert torch.equal(cur, w), f"冻结集被触碰: {n}"


def test_echo_step_freezes_world_model():
    """感知步定哨 → echo 步: 冻结集逐位不变, W_act 已变."""
    net = _make_net()
    net.learn(torch.randint(32, 256, (1, 8), dtype=torch.long))  # 感知步定哨
    before = _snap(net, _FROZEN_WEIGHTS + _FROZEN_STATE)
    w_act = net.W_act.data.clone()
    net.learn()  # byte_ids=None 恒回声
    _assert_same(net, before)
    assert not torch.equal(net.W_act.data, w_act)  # 表达端推进


def test_free_run_internal_dynamics_still_learn():
    """free_run 内生动力学不冻结: W_35/W_56/W_t4 仍学习 (C5 只冻回声, 不冻自由运行)."""
    net = _make_net()
    w35 = net.W_35.data.clone()
    w56 = net.W_56.data.clone()
    wt4 = net.W_t4.data.clone()
    net.learn(free_run=True)
    assert not torch.equal(net.W_35.data, w35)
    assert not torch.equal(net.W_56.data, w56)
    assert not torch.equal(net.W_t4.data, wt4)
