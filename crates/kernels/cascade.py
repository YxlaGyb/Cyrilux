"""
感知级联纯算子

感知逐步路径的具名算子:
    输入装配、投影、振荡器相位、侧抑制、差分残差、记忆环、新奇度环
每个算子逐条对应静态图的一个声明节点, 不持状态, 不做控制流决策
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def zero_input(dev: torch.device, n: int, s: int, d: int) -> torch.Tensor:
    """自由运行，外部输入恒零."""
    return torch.zeros(n, s, d, dtype=torch.float16, device=dev)


def one_hot_input(byte_ids: torch.Tensor, alphabet: int) -> torch.Tensor:
    """字节 → one-hot 输入通道."""
    return F.one_hot(byte_ids, num_classes=alphabet).to(torch.float16)


def history_join(padmax: torch.Tensor, z0: torch.Tensor, alphabet: int) -> torch.Tensor:
    """时序双通道，上一帧输入通道, 首帧用 padmax 补."""
    return torch.cat([padmax[:, :, :alphabet], z0[:, :-1]], dim=1)


def join_last(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """双通道末维拼接."""
    return torch.cat([x, y], dim=-1)


def project(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """线性投影 x @ Wᵀ + b."""
    return x @ w.T + b


def bias_3d(b: torch.Tensor) -> torch.Tensor:
    """偏置升到 [1,1,d] 以按帧广播."""
    return b.unsqueeze(0).unsqueeze(0)


def add_scaled(x: torch.Tensor, y: torch.Tensor, k: float) -> torch.Tensor:
    """加性调制: x + k·y."""
    return x + k * y


def add_unsqueezed(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """按帧广播叠加 (中心脑修正量)."""
    return x + y.unsqueeze(1)


def nonzero_mask(byte_ids: torch.Tensor) -> torch.Tensor:
    """有效位置掩码 (0 = 填充)."""
    return byte_ids != 0


def osc_phase(tab: torch.Tensor, cnt: torch.Tensor, span: int, s: int) -> torch.Tensor:
    """查表取相位向量 [1,S,1], 零运行时三角."""
    return tab[(cnt.long() + torch.arange(s, device=cnt.device)) % span].unsqueeze(0).unsqueeze(-1)


def osc_advance(cnt: torch.Tensor, s: int, span: int) -> None:
    """相位计数推进 (就地)."""
    cnt.add_(s).remainder_(span)


def foldiak_gate(z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """协方差白化侧抑制: z − 0.2·(M @ zᵀ)ᵀ."""
    return z - 0.2 * (m @ z.transpose(-2, -1)).transpose(-2, -1)


def center_last(x: torch.Tensor) -> torch.Tensor:
    """末维去中心化."""
    return x - x.mean(dim=-1, keepdim=True)


def drop_last(x: torch.Tensor) -> torch.Tensor:
    """去掉末帧 (差分对齐)."""
    return x[:, :-1]


def seq_diff(x: torch.Tensor) -> torch.Tensor:
    """逐帧差分 Δz."""
    return x[:, 1:] - x[:, :-1]


def unit_norm(x: torch.Tensor) -> torch.Tensor:
    """零向量保护单位化 (RMS 归一化)."""
    return x / (x.norm(dim=-1, keepdim=True) + 1e-3)


def prepend_pad(x: torch.Tensor, padmax: torch.Tensor, d: int) -> torch.Tensor:
    """首帧前置 pad."""
    return torch.cat([padmax[:, :, :d], x], dim=1)


def pred_err(dz: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """差分预测残差 (自由能)."""
    return (dz - mu[:, :-1]).square().mean()


def mem_reset(m_seq: torch.Tensor, mem_m: torch.Tensor, n: int) -> None:
    """记忆序列首帧写入当前记忆本体 (就地)."""
    k, a4 = mem_m.shape
    m_seq[:, :1] = mem_m.unsqueeze(0).unsqueeze(0).expand(n, 1, k, a4)


def mem_advance(m_seq: torch.Tensor, z4: torch.Tensor, t: int, mem_a: torch.Tensor) -> None:
    """泄漏积分单元逐帧推进 (就地)."""
    zt = z4[:, t : t + 1].unsqueeze(2)
    mt = (1 - mem_a)[None, None, :, None] * m_seq[:, t - 1 : t] + mem_a[None, None, :, None] * zt
    m_seq[:, t : t + 1] = mt


def mem_read(m_seq: torch.Tensor, z4: torch.Tensor, k: float) -> torch.Tensor:
    """内容寻址读出: 余弦相似度 softmax 门控, Σgate = K."""
    num = (z4.unsqueeze(2) * m_seq).sum(dim=-1)
    den = z4.norm(dim=-1, keepdim=True) * (m_seq.norm(dim=-1) + 1e-3)
    attn = torch.softmax((num / den) * k, dim=-1) * k
    return (m_seq * attn.unsqueeze(-1)).reshape(m_seq.shape[0], m_seq.shape[1], -1)


def mem_last(m_seq: torch.Tensor) -> torch.Tensor:
    """末帧记忆按批均值写回本体."""
    return m_seq[:, -1].mean(dim=0).detach()


def nov_reset(z4: torch.Tensor) -> torch.Tensor:
    """慢通道首帧 = 快通道首帧."""
    zslow = torch.zeros_like(z4)
    zslow[:, 0] = z4[:, 0]
    return zslow


def nov_advance(zslow: torch.Tensor, z4: torch.Tensor, t: int) -> None:
    """慢通道逐帧推进 (就地)."""
    zslow[:, t] = 0.99 * zslow[:, t - 1] + 0.01 * z4[:, t]


def nov_read(z4: torch.Tensor, zslow: torch.Tensor) -> torch.Tensor:
    """快慢散度逐帧新奇度."""
    return ((z4 - zslow).square().mean(dim=-1)).detach()
