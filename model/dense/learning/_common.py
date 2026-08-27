"""学习引擎共享纯函数层 (零 net 状态, 零 autograd, 全 fp16).

原 model/dense/learning.py 的模块级函数 + learn() 内提升的闭包.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..network import DensePCNet


class _MixinBase:
    """mixin 公共基类: 让 ty 知道组合类成员 (self.net 在 mixin 文件中可见)."""

    net: DensePCNet

# 第 103 轮实验: 读出端信任域 — round-102 从零防噪设 1% (chat101 从零噪声淹没
# 实证). 中期训练 (14000+ 步) 噪声已均值掉, 1% 可能钳制内容学习 (hit1 平台
# ~0.14). 提升到 2.5% 测试内容是否突破. 若不稳定回退 0.01.
LM_TRUST_REGION = 0.025

# 突触资格迹
# 让突触记住走过的路, 等待那个迟到的好结果
# E 与 W 同形, 零初始化, 单位增益归一化指数累积:
#   E <- gamma·E + (1-gamma)·dW_raw        (稳态量级 = 瞬时外积)
ELIG_GAMMA = 0.95


def _activity_baseline(
    net: DensePCNet, post: torch.Tensor, ema_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """活动² 慢速局部基线 (第 79 轮抽取, 与第 78 轮能量约束同源): 返回 (p2, excess),
    更新 EMA 缓冲 (0.99/0.01, 首窗延迟初始化: ema ← post², excess 从零起步). 纯局部.

    # ponytail: 修剪 permute/shrink 可塑参数但不重排 _act_ema_*, 0.99 速率数百步自愈;
    # 仅当修剪后异常再现才需在 pruning.py 同步 EMA 缓冲.
    """
    p2 = (post * post).mean(dim=(0, 1))  # [out] 窗内活动²
    ema = getattr(net, ema_name)[: p2.shape[0]]  # 修剪后 active 收缩: 只写头部切片
    if ema_name not in net._act_ema_init:
        ema.copy_(p2)
        net._act_ema_init.add(ema_name)
        excess = torch.zeros_like(p2)
    else:
        ema.mul_(0.99).add_(0.01 * p2)
        excess = torch.relu(p2 - ema)
    return p2, excess


def _energy_constraint(
    net: DensePCNet, W: torch.Tensor, dW: torch.Tensor, post: torch.Tensor, ema_name: str
) -> torch.Tensor:
    """内建能量约束 (第 78 轮): 真 Oja + 活动依赖遗忘项, 逐输出单元, 纯局部.

    post: [N,S,out] 与该点赫布外积同源的输出侧因子; ema: [out] 活动² 慢速基线.
    dW ← dW − (α·post² + β·relu(post²−ema)) ⊙ W;  ema ← 0.99·ema + 0.01·post².
    首窗延迟初始化: ema ← post² (excess 从零起步, 防早期过度抑制).
    α = cfg.oja_alpha, β = cfg.oja_elasticity (零全局统计, 无目标值).
    """
    p2, excess = _activity_baseline(net, post, ema_name)
    coef = net.cfg.oja_alpha * p2 + net.cfg.oja_elasticity * excess
    return dW - coef.unsqueeze(1) * W


def _elig_accum(net: DensePCNet, wname: str, dW_raw: torch.Tensor) -> torch.Tensor:
    """迹积累 + 返回当前迹 (用户公式, 零修改): E ← γ·E + pre⊗post.
    wname = 权重名 (如 "W_42"); dW_raw = 原始外积 (与 E 同形).
    自动对齐修剪后的活性切片 (前缘 min 切片, 同既有机制).

    行为等价 (执行指令 3): 迹稳态幅度 = dW/(1-γ) ≈ 20×dW (γ=0.95),
    R=1.0 会把更新放大 20 倍 — 没有任何标量 R 能同时对全部权重做
    逐尺度补偿 (W_lm 单位向量化/各层 dW 量级不同). 唯一保证"无信号
    时现状零变"的取值: R 默认 = 0.0 — 迹每步照常积累 (⌊彳亍⌉: 它
    在记录), 但 ΔW=η·R·E=0, 现有 Hebbian 逐位不变. 生存环境接入时
    设 net._survival_signal (0-1 级环境标量), 迹通道启用 — 20× 幅度
    由信号本身的小量级自然承担.
    """
    E = getattr(net, f"{wname}_elig")
    E = E[: dW_raw.shape[0], : dW_raw.shape[1]]
    E.mul_(ELIG_GAMMA).add_(dW_raw)
    return E


def _spectral_radius_guard(W: torch.Tensor, rescale: bool = True, bound: float = 1.5) -> torch.Tensor:
    """递归矩阵谱半径安全约束 (第 79 轮改判, 第 81 轮重定义): 10 次幂迭代估 ρ.

    rescale=True: ρ > bound 时 W ← W·(bound/ρ) — 死亡保险 (发散的物理硬界),
    非临界点之墙. 第 81 轮: bound 从 0.95 移至 1.5 — 正常动力学由能量耗散
    (STP 资源耗尽/分流抑制/Oja) 自然饱和, 守卫只在能量失效时兜底. 无全局统计;
    修剪置换后子矩阵 ρ 越界 → 下一步即被拉回.
    rescale=False: 仅观测 (observer 用). 返回 ρ (0 维张量, 可 .item()).
    # ponytail: 每步 6 个小 randn 分配 (~µs 级), perf 敏感再改持久缓冲
    """
    a = W.shape[0]
    v = torch.randn(a, 1, device=W.device, dtype=W.dtype)
    v = v / (v.norm() + 1e-4)
    for _ in range(10):
        v = W @ v
        v = v / (v.norm() + 1e-4)
    rho = (W @ v).norm()
    if rescale:
        W.mul_(torch.minimum(bound / rho, torch.ones_like(rho)))
    return rho


def _decorr_W(
    W: torch.Tensor,
    E: torch.Tensor,
    coef: float = 1.0,
    max_delta_ratio: float | None = None,
    learn_boost: float = 1.0,
) -> torch.Tensor:
    """权重去主成分投影 (超量抑制): W -= β·(W@v)⊗v, v = W 的 top1 奇异方向.

    Hebbian 更新把每层 W 行的输入空间分量收敛到 ±w 单一方向 (行间有符号
    cos≈0 被符号随机掩盖, 但绝对 cos 134 倍于随机 → 投影秩 1, PR_eff 焊死
    根因). β=1.0 超量抑制 — 必须压倒 Hebbian 正反馈 (β=0.5 被证被覆盖).
    主方向直接从 W 幂迭代 (3 次, 不绕 E 的 |cos| 中介 — E 幂迭代 3 次收敛
    不充分, v 含噪声, 超量抑制放大噪声 → E_l5 NaN, 334 步实测复现)

    (由 learn() 内嵌闭包提升; learn_boost 原从 net._traction_scale 导出, 现显式传参)
    """
    dev = W.device
    dim = W.shape[0]
    Wn = W / (W.norm(dim=1, keepdim=True) + 1e-3)
    dE = (Wn @ Wn.T).abs()  # 绝对相关 (诊断: 行收敛指标)
    eye_mask = 1.0 - torch.eye(dim, device=dev, dtype=W.dtype)
    E.data.mul_(0.97).add_((dE * eye_mask) * (0.05 * learn_boost))
    # top1 方向: 幂迭代 W^T W (列空间), 3 次. 第 81 轮: 迭代前把 W 缩放
    # 到单位 Frobenius 范数 — 幂迭代数学上尺度不变 (只估方向), 但 fp16
    # 下小矩阵 (范数 <~0.1, 元素进入非正规范围 <6e-5) 在乘加累积中
    # 下溢/FTZ 产生 NaN (W_t3 被 Oja 衰减到范数 0.011 后 410 步实测).
    # Wn 只用于方向; 修正量 c@v.T 仍用原 W (幅度按原尺度, W→0 时修正→0)
    Wn = W / (W.norm() + 1e-8)
    v = torch.randn(W.shape[1], 1, device=dev, dtype=W.dtype) * 0.01
    for _ in range(3):
        v = Wn.T @ (Wn @ v)
        v = v / (v.norm() + 1e-8)
    c = W @ v  # 每行在 top1 方向上的投影系数 (含 ± 符号)
    dW = c @ v.T  # 行 i 减 c_i·v^T (超量抑制, 切断秩 1)
    # 范数信任域 (闭环制动): 单步扰动上限 = max_delta_ratio·‖W‖_F,
    # 方向保持等比缩放 (信任域方法, 开环有界扰动防下游突变)
    if max_delta_ratio is not None:
        dmax = max_delta_ratio * W.norm()
        dn = dW.norm()
        if dn > dmax:
            dW = dW * (dmax / dn)
    dW = dW * coef
    W -= dW
    return dW


def _rho_ctrl(dW: torch.Tensor, W_ref: torch.Tensor, tag: str, net: DensePCNet) -> torch.Tensor:
    """通道级塑性控制 (第 75 轮): 预测连接时间尺度统一控制律.

    ρ_i = ||ΔW_i||/||W_i||, s_i = clip(0.03/ρ_i, 0.005, 1.0).
    冻结 BCM/Hebbian/Cos 竞争: 唯一变量 = 外积/注入时间尺度.
    (由 learn() 内嵌闭包提升; adaptive_rho 开关与 _rho_map 记录保持在 net 上)
    """
    if not net.cfg.adaptive_rho:
        return dW
    nW_i = W_ref.norm() + 1e-8
    rho = dW.norm() / nW_i  # fp16 范数比 (无量纲, [0,~1e2] 内可表示)
    s_i = (0.03 / (rho + 1e-8)).clamp(0.005, 1.0)
    dW_s = dW * s_i
    net._rho_map[tag] = (rho, dW_s.norm() / nW_i, s_i)
    return dW_s
