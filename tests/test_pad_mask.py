"""pad 泄漏掩码测试 (批 2): 目标字节 <32 的位置零学习信号进 LM 头.

判据 = 模型自身 _mask_print 输出域 (0x00-0x1F 不可打印): 学习端与自身输出域自洽,
pad 0x00 (dataset ljust 填充) 不再经 W_lm[:,0] / bias_lm[0] / W_lm_2[:,0] 注入赫布驱动.
"""

import torch

from model import CyreneModel, DensePCNet


def _make_net():
    torch.manual_seed(0)
    cfg = CyreneModel(
        d_l4=64, d_l2=32, d_l3=32, d_l5=64, d_l6=16,
        max_seq_len=16, free_run_window=16, mem_k0=2,
    )
    return DensePCNet(cfg)


def test_pad_target_zeroes_lm_col0_update():
    """窗内 0x00 目标 (含 t+1 活/t+2=0x00 行): W_lm[:,0]/bias_lm[0]/W_lm_2 col0 无泄漏."""
    net = _make_net()
    x = torch.tensor([[65, 66, 67, 0, 68, 69, 70, 71]], dtype=torch.long)
    w0 = net.W_lm.data[:, 0].clone()
    b = net.bias_lm.data.clone()
    net.learn(x)
    # dW 链路纯零 (probs[0x00]≡0, 目标列 0, 行掩码清零) → 归一化/等比帽保零, 位级相等
    assert torch.equal(net.W_lm.data[:, 0], w0)
    # bias 更新含去均值项, fp16 softmax 归一化尘埃 ~1e-8 (旧泄漏 ~7e-3/步, 阈值居中)
    assert float((net.bias_lm.data[0] - b[0]).abs()) <= 1e-4
    # elig 在归一化后入迹, 免疫 soft_norm 全矩阵缩放 → 位级判据
    assert (net.W_lm_2_elig[:, 0] == 0).all()
    # 非空转: 活目标列确有信号入迹
    assert net.W_lm_2_elig[:, 67].abs().sum() > 0
    assert (net.bias_lm.data - b).abs().sum() > 0


def test_trailing_pad_no_col0_leak():
    """dataset ljust 形态 (尾部连续 0x00): 同判据, t+1 活/t+2=0x00 行不进 W_lm_2 col0."""
    net = _make_net()
    x = torch.tensor([[65, 66, 67, 68, 69, 70, 0, 0]], dtype=torch.long)
    w0 = net.W_lm.data[:, 0].clone()
    b0 = net.bias_lm.data[0].clone()
    net.learn(x)
    assert torch.equal(net.W_lm.data[:, 0], w0)
    assert float((net.bias_lm.data[0] - b0).abs()) <= 1e-4
    assert (net.W_lm_2_elig[:, 0] == 0).all()
    assert net.W_lm_2_elig[:, 68].abs().sum() > 0


def test_closed_loop_pad_target_no_nan():
    """closed_loop 一步: 0x00 目标行掩码后锚定/生成混合步 no-NaN, col0 仍零."""
    net = _make_net()
    x = torch.tensor([[65, 66, 67, 0, 68, 69, 70, 71]], dtype=torch.long)
    w0 = net.W_lm.data[:, 0].clone()
    stats = net.learn(x, closed_loop=True)
    assert torch.isfinite(stats["free_energy"])
    assert torch.isfinite(stats["future_err"])
    assert torch.equal(net.W_lm.data[:, 0], w0)
    assert (net.W_lm_2_elig[:, 0] == 0).all()
    assert torch.isfinite(net._last_z3).all()  # NaN 前哨
