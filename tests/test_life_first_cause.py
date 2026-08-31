"""
生命第一因落地配置检查.
"""

from model import CyreneModel


def test_stp_u_adapt_default_on():
    cfg = CyreneModel()
    assert cfg.stp_u_adapt is True
    assert 0.0 < cfg.stp_u_adapt_rate <= 0.1
    assert cfg.stp_u_min < cfg.stp_u_max


def test_bias_leak_default_on():
    cfg = CyreneModel()
    assert cfg.bias_leak_rate > 0.0


def test_wt_syn_scaling_default_on():
    cfg = CyreneModel()
    assert cfg.wt_syn_scaling is True
    assert 0.0 < cfg.wt_syn_scaling_rate <= 0.2


def test_wt_syn_scaling_rate_is_fast_enough():
    # 第 83 轮实证: 速率必须与 Hebbian 增长同量级 (0.01 时操作点仍由守卫设定)
    cfg = CyreneModel()
    assert cfg.wt_syn_scaling_rate >= 0.05


def test_spectral_guard_bound_is_configurable():
    # 第 83 轮: 谱守卫 bound 是死亡保险的物理参数 (须高于自然工作区间)
    cfg = CyreneModel()
    assert cfg.spectral_guard_bound >= 1.5
