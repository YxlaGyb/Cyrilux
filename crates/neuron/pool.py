"""
MemoryPool
竞争性记忆单元群的生死原语，全部纯局部机制

出生: 从父单元二分 (α×2 或 ÷2 交替)
死亡: 归一/饥荒击杀, 单元缓冲 + W1 行 + W1_elig 迹同步重注册
复用工位与扩桌由 resize/birth 在同一 lm_in 不变量下完成
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from crates.kernels import l2_norm

if TYPE_CHECKING:
    from crates.model.model_cyrene import HybridPCNet


class MemoryPool:
    """
    记忆单元池
    持 net 引用, 操作 net._mem_*/W1 单元块.
    """

    def __init__(self, net: HybridPCNet):
        self.net = net

    def account(self, ctx, lm):
        """
        竞争性记忆单元群的生存账目 (纯张量, 可入静态图):
        收入 → 增益 → 死亡计数 → 出生统计

        收入: |m| · cos(单元内容, 账本状态方向 ẑ), m = 生存账本残差.
        返回判死掩码, 生死判定与 K 变化交给 decide (宿主).
        """
        net = self.net
        if ctx.free_run or ctx.echo_world_frozen or lm is None:
            return net._mem_death_cnt < 0
        cfg = net.cfg
        if lm.e_h is None or getattr(net, "_w1v2", False):
            return net._mem_death_cnt < 0

        # 1) 收入: 生存货币 |m| × 单元内容与账本状态方向 ẑ 的余弦
        pooled = net._z4.mean(dim=(0, 1))
        z_hat = pooled / (pooled.norm() + 1e-6)
        q = net._survival_signal.abs() * (l2_norm(net._mem_m) @ z_hat)  # [K] ∈ [-2,2]
        net._mem_q.mul_(0.99).add_(0.01 * q)

        # 2) 增益: 生存货币入账 (clamp 为界, 非梯度路径, 非溢出修补)
        net._mem_g.mul_(1.0 - cfg.mem_g_decay).clamp_(min=0.0)  # tensor-guard: bound (记忆增益界, 非梯度)
        net._mem_g.add_(cfg.mem_eta_g * net._mem_q)
        net._mem_g.clamp_(max=cfg.mem_g_max)  # tensor-guard: bound (记忆增益界, 非梯度)

        # 3) 死亡: g 低 → 计数, 持续超限 → 判死; 索引 0 永不判死
        low = net._mem_g < cfg.mem_g_min
        net._mem_death_cnt.copy_(torch.where(low, net._mem_death_cnt + 1, torch.zeros_like(net._mem_death_cnt)))
        dead_mask = (net._mem_death_cnt >= cfg.mem_death_steps) & low
        dead_mask[0:1].fill_(False)

        # 4) 出生统计: 误差快/慢 EMA, 冷却计数
        err_rms = lm.eps_total.square().mean().sqrt()
        net._mem_err_ema.mul_(0.99).add_(0.01 * err_rms)
        net._mem_err_long.mul_(0.999).add_(0.001 * err_rms)
        net._mem_birth_cd.add_(1)
        return dead_mask

    def decide(self, dead_mask):
        """
        生死判定 (宿主): 判死 → 冷却 → 收缩, 冷却期满 → 增益最高单元二分出生

        宿主侧与图形分离: 判死/出生要 resize 重注册整张参数布局, 无法入静态图.
        时机: 整步图之后 → K 变化只影响下一步 (zh 列与 W1 行永不错位).
        """
        if dead_mask is None:
            return
        net = self.net
        K = net._mem_m.shape[0]
        if dead_mask.shape[0] == K:
            if dead_mask[1:].any():  # tensor-guard: rare (死亡事件, 每 2000 步窗口至多一次)
                net._mem_death_cnt.copy_(torch.where(~dead_mask, net._mem_death_cnt, torch.zeros_like(net._mem_death_cnt)))
                self.resize(~dead_mask)
        cfg = net.cfg
        net._mem_birth_py += 1
        if (
            net._mem_birth_py >= cfg.mem_birth_cooldown
            and cfg.mem_k_max > K
            and bool((net._mem_err_ema > net._mem_err_long * cfg.mem_birth_thresh).item())  # tensor-guard: rare (出生事件每冷却窗至多一次)
        ):
            parent = int(torch.argmax(net._mem_g))  # tensor-guard: rare (出生事件同步)
            self.birth(parent)
            net._mem_birth_cd.zero_()
            net._mem_birth_py = 0

    def resize(self, keep: torch.Tensor):
        """
        K 变化: 单元缓冲 + W1 行 + W1_elig 迹同步重注册.
        """
        net = self.net
        dim_4 = net._mem_m.shape[1]
        d_h = net.d_h
        if keep.dtype == torch.bool:
            keep = keep.nonzero(as_tuple=False).squeeze(1)
        old_m = net._mem_m.data[keep].contiguous()
        old_a = net._mem_a.data[keep].contiguous()
        old_g = net._mem_g.data[keep].contiguous()
        old_q = net._mem_q.data[keep].contiguous()
        net.register_buffer("_mem_m", old_m)
        net.register_buffer("_mem_a", old_a)
        net.register_buffer("_mem_g", old_g)
        net.register_buffer("_mem_q", old_q)
        net.register_buffer("_mem_death_cnt", net._mem_death_cnt[keep].contiguous())
        # W1 行同步 (单元块位于 [dim_4+32 : dim_4+32+K·dim_4])
        head = dim_4 + net.bind_slot_dim
        w1 = net.W1.data
        new_w1 = torch.cat(
            [w1[:head], w1[head:].reshape(-1, dim_4, d_h)[keep].reshape(keep.shape[0] * dim_4, d_h)],
            dim=0,
        ).contiguous()
        net.W1 = nn.Parameter(new_w1)
        # W1_elig 迹同形同步 (出生/死亡后迹必须与 W1 行对齐)
        elig = getattr(net, "W1_elig", None)
        if elig is not None:
            ne = torch.cat(
                [elig.data[:head], elig.data[head:].reshape(-1, dim_4, d_h)[keep].reshape(keep.shape[0] * dim_4, d_h)],
                dim=0,
            ).contiguous()
            net.register_buffer("W1_elig", ne)
        # 迁移扩通道同切片 (同一 lm_in 不变量): W1_new 行选 + B_new/B_recon 解码器列选
        w1n = getattr(net, "W1_new", None)
        if w1n is not None:
            net.W1_new = nn.Parameter(torch.cat(
                [w1n.data[:head],
                 w1n.data[head:].reshape(-1, dim_4, d_h)[keep].reshape(keep.shape[0] * dim_4, d_h)],
                dim=0,
            ).contiguous())
        for _bn in ("B_new", "B_recon"):
            bt = getattr(net, _bn, None)
            if bt is None:
                continue
            nb = torch.cat(
                [bt.data[:, :head],
                 bt.data[:, head:].reshape(bt.shape[0], -1, dim_4)[:, keep, :].reshape(bt.shape[0], keep.shape[0] * dim_4)],
                dim=1,
            ).contiguous()
            setattr(net, _bn, nn.Parameter(nb))
        # w1v2b 组合方向同 lm_in 不变量 (与 W1_new 行同款切片; 尾部补零后仍单位范数, 无需重归一)
        _wzh = getattr(net, "_w1v2_wzh", None)
        if _wzh is not None:
            net._w1v2_wzh = torch.cat(
                [_wzh[:head], _wzh[head:].reshape(-1, dim_4)[keep].reshape(keep.shape[0] * dim_4)],
                dim=0,
            ).contiguous()
        net._lm_in = head + keep.shape[0] * dim_4
        net._struct_epoch += 1

    def famine_kill(self):
        """
        饥荒击杀: 移除最低 _mem_g 的记忆单元，索引 0 免疫, K_min=1 保底

        与 _update_mem_units 的 2000 步慢死亡不同: 这是代谢死亡回合的直接不可逆
        损失，内容与身份永久消失, 后续 birth 只会从存活父克隆新单元.
        """
        net = self.net
        K = net._mem_m.shape[0]
        if K <= 1:
            return
        ix = int(torch.argmin(net._mem_g[1:])) + 1  # 与 birth 的 argmax 同款单次同步
        keep = torch.ones(K, dtype=torch.bool, device=net._mem_g.device)  # tensor-guard: rare (饥荒击杀事件)
        keep[ix] = False
        self.resize(keep)

    def birth(self, parent: int):
        """从父单元二分出生 (α×2 或 ÷2, _mem_alt 交替, 越界取另一侧)."""
        net = self.net
        cfg = net.cfg
        K = net._mem_m.shape[0]
        dim_4 = net._mem_m.shape[1]
        d_h = net.d_h
        a_par = float(net._mem_a[parent])
        alt = net._mem_alt % 2
        net._mem_alt += 1
        a_child = a_par * 2.0 if alt == 0 else a_par / 2.0
        if a_child > cfg.mem_alpha_max:
            a_child = a_par / 2.0
        elif a_child < cfg.mem_alpha_min:
            a_child = a_par * 2.0
        a_child = max(cfg.mem_alpha_min, min(cfg.mem_alpha_max, a_child))
        # 子单元: m 继承父, g 小值起步 (需证明自己), q 零, α 二分
        m_child = net._mem_m.data[parent : parent + 1].clone()  # tensor-guard: rare (出生事件)
        new_m = torch.cat([net._mem_m.data, m_child], dim=0).contiguous()
        new_a = torch.cat([net._mem_a.data, torch.tensor([a_child], dtype=torch.float16, device=net._mem_m.device)], dim=0).contiguous()
        new_g = torch.cat([net._mem_g.data, torch.full((1,), cfg.mem_g_min * 2.0, dtype=torch.float16, device=net._mem_m.device)], dim=0).contiguous()
        new_q = torch.cat([net._mem_q.data, torch.zeros(1, dtype=torch.float16, device=net._mem_m.device)], dim=0).contiguous()
        net.register_buffer("_mem_m", new_m)
        net.register_buffer("_mem_a", new_a)
        net.register_buffer("_mem_g", new_g)
        net.register_buffer("_mem_q", new_q)
        net.register_buffer(
            "_mem_death_cnt",
            torch.cat(
                [net._mem_death_cnt, torch.zeros(1, dtype=torch.int32, device=net._mem_g.device)]
            ).contiguous(),
        )
        # W1 单元块尾插零行 (零起步: 新单元先"无贡献", 由 W1 学权重自然放大)
        head = dim_4 + net.bind_slot_dim
        w1 = net.W1.data
        new_w1 = torch.cat(
            [w1[:head], w1[head:], torch.zeros(dim_4, d_h, dtype=torch.float16, device=w1.device)],
            dim=0,
        ).contiguous()
        net.W1 = nn.Parameter(new_w1)
        elig = getattr(net, "W1_elig", None)
        if elig is not None:
            ne = torch.cat(
                [elig.data[:head], elig.data[head:], torch.zeros(dim_4, d_h, dtype=torch.float16, device=elig.device)],
                dim=0,
            ).contiguous()
            net.register_buffer("W1_elig", ne)
        # 迁移扩通道尾插零块 (同一 lm_in 不变量 + 同 W1 零起步约定): W1_new 零行 + B_new/B_recon 零列
        w1n = getattr(net, "W1_new", None)
        if w1n is not None:
            net.W1_new = nn.Parameter(torch.cat(
                [w1n.data[:head], w1n.data[head:],
                 torch.zeros(dim_4, d_h, dtype=torch.float16, device=w1n.device)],
                dim=0,
            ).contiguous())
        for _bn in ("B_new", "B_recon"):
            bt = getattr(net, _bn, None)
            if bt is None:
                continue
            nb = torch.cat(
                [bt.data[:, :head], bt.data[:, head:],
                 torch.zeros(bt.shape[0], dim_4, dtype=torch.float16, device=bt.device)],
                dim=1,
            ).contiguous()
            setattr(net, _bn, nn.Parameter(nb))
        # w1v2b 组合方向同 lm_in 不变量 (尾插零: 新单元先无贡献, 同 W1_new 零行; 尾部补零后仍单位范数)
        _wzh = getattr(net, "_w1v2_wzh", None)
        if _wzh is not None and _wzh.shape[0] != head + (K + 1) * dim_4:
            net._w1v2_wzh = torch.cat(
                [_wzh, torch.zeros(dim_4, dtype=_wzh.dtype, device=_wzh.device)]
            )
        net._lm_in = head + (K + 1) * dim_4
        net._struct_epoch += 1
