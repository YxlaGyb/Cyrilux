"""生命第一因落地配置检查.

第 82 轮新增的局部自组织机制必须存在且默认开启:
- STP U 自适应: 逐神经元活动依赖释放概率慢适应
- bias 泄漏: 防止 bias 成为定点支柱
第 83 轮新增:
- W_t 突触缩放: 逐神经元权重级慢稳态增益 (自由运行专用)
"""

from model import DensePCConfig


def test_stp_u_adapt_default_on():
    cfg = DensePCConfig()
    assert cfg.stp_u_adapt is True
    assert 0.0 < cfg.stp_u_adapt_rate <= 0.1
    assert cfg.stp_u_min < cfg.stp_u_max


def test_bias_leak_default_on():
    cfg = DensePCConfig()
    assert cfg.bias_leak_rate > 0.0


def test_wt_syn_scaling_default_on():
    cfg = DensePCConfig()
    assert cfg.wt_syn_scaling is True
    assert 0.0 < cfg.wt_syn_scaling_rate <= 0.2


def test_wt_syn_scaling_rate_is_fast_enough():
    # 第 83 轮实证: 速率必须与 Hebbian 增长同量级 (0.01 时操作点仍由守卫设定)
    cfg = DensePCConfig()
    assert cfg.wt_syn_scaling_rate >= 0.05


def test_spectral_guard_bound_is_configurable():
    # 第 83 轮: 谱守卫 bound 是死亡保险的物理参数 (须高于自然工作区间)
    cfg = DensePCConfig()
    assert cfg.spectral_guard_bound >= 1.5
