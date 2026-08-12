"""ForwardEngine

密集 PPA 前馈/推理/生成/稀疏绑定.

感知: L0(纯 one-hot, 时序双通道) → L4 → L2 → L3 → L5(微柱阵列) → L6
时序: 每层时间核 W_t 递归 z[t] 依赖 z[t-1]
生成: L4 状态 + W_diff 预测差分 → W_lm 解码字节
绑定: z4 → W_bind → K=16 概念槽, 软竞争归一化

全 fp16, 零 .float(), 零反向传播.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .network import DensePCNet


def _rms(x: torch.Tensor) -> torch.Tensor:
    """零向量保护 RMS 归一化 (fp16 下 1e-8 舍入为 0, 需掩码保护分母)."""
    rms = x.square().mean(dim=-1, keepdim=True)
    alive = (rms > 1e-8).to(x.dtype)
    denom = torch.where(alive > 0, (rms * 1.01).sqrt(), torch.ones_like(rms))
    return x * alive / denom


def _l2_norm(x: torch.Tensor) -> torch.Tensor:
    """零向量保护 L2 范数归一化 (前馈投影前的原语义, 与 _rms 量纲不同)."""
    nrm = x.norm(dim=-1, keepdim=True)
    alive = (nrm > 1e-8).to(x.dtype)
    denom = torch.where(alive > 0, nrm * 1.01, torch.ones_like(nrm))
    return x * alive / denom


class ForwardEngine:
    """前馈引擎: 持 net 引用, 状态经 net._z*/net.active_size 传递."""

    def __init__(self, net: DensePCNet):
        self.net = net

    def forward(self, byte_ids: torch.Tensor) -> dict:
        """推理前馈: 返回未来预测偏差.  ACh 关闭, 确定性."""
        return self._predict(byte_ids, store_state=True, is_inference=True)

    def generate(
        self, prompt: str, n_tokens: int = 40, temperature: float = 0.7, dev: torch.device | None = None
    ) -> bytes:
        """行动: L4 状态 + 预测差分 → W_lm 解码生成字节."""
        if dev is None:
            dev = next(self.net.parameters()).device
        ids = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=dev)
        out = self.continuation(ids, n_tokens, temperature=temperature, rep_backstop=True)
        return bytes(out[0].tolist())

    def continuation(
        self,
        byte_ids: torch.Tensor,
        n_gen: int,
        temperature: float = 0.7,
        rep_backstop: bool = False,
    ) -> torch.Tensor:
        """自回归续写: 基于 [N,S] 前缀批量并行生成 n_gen 字节, 返回 [N, S+n_gen].

        训练 (rollout) 与验证 (generate) 共用同一 logits 管线, 保证训练看到的
        自生成分布与最终生成测试一致. 纯前向, 零梯度. 全批并行 (每步一次
        前向覆盖 N 样本), 生成后段逐字节推进. 生成期 n-gram 习惯化
        (rep_backstop, 第 64 轮): 最近 2/3/4 字节模式重复 ≥2 次 → 物理屏蔽
        周期末字节的 logits (置 -1e4). 第 68 轮 TF 锚点门控已移除 (被证
        无法防坍缩); 复读对治 = 第 69 轮快慢散度多巴胺 (learning.py D 极性
        翻转, 内生判别, 无外部干预). 以及 UTF-8 续字节超长阻断: 连续 3 个
        续字节 (合法单字符上限) 后屏蔽全部续字节.
        """
        net = self.net
        N, S = byte_ids.shape
        dev = byte_ids.device
        cur = byte_ids.clone()
        last_byte = torch.full((N,), -1, dtype=torch.long, device=dev)
        utf8_run = torch.zeros(N, dtype=torch.long, device=dev)
        if not hasattr(net, "_block_stats"):
            net._block_stats = {"rep": 0, "utf8": 0, "gen": 0}
        stats = net._block_stats
        freq_safe = net._freq + 1e-2  # 频率去偏基线 (两读出分支共用)
        for _ in range(n_gen):
            bv = cur[:, -64:]  # 上下文窗口 64 (任务语义)
            _ = self._predict(bv, store_state=True, is_inference=True)
            if getattr(net, "use_w_act", False):
                # 动作回路 (第 76 轮): 概念槽 z_bind → W_act 脉冲字节, 离散事件驱动
                # 裁决 10: 思考循环已剥离 (NaN 工厂); 裁决 12: 内部状态错误配对
                z_bind = net._bind_vec[:, -1]  # [N,16] 概念槽 (连续值稀疏)
                # ── 内部状态错误配对 (裁决 12/13): 持续低强度运作 ──
                # 复读检测: 最近 10 步唯一字节 <3 → 复读深度累计 (探索清零).
                # 错配始终在线: 强度 = 0.01 + 0.15·min(rep_depth/10, 1) —
                # 复读深度 0 也有 0.01 背景探索 (蓝斑强直性活动, 防完全固化);
                # 深度 ≥1 强度线性增长至 0.16 (满幅探索).
                # 扰动: 从 z_bind@W_bind_self 转移偏好 top-3 目标槽随机选一,
                # z_bind += strength·(tgt_emb - z_bind). 定向 (非随机噪声),
                # 由内部转移模型结构决定方向 (海马尖波涟漪组合性探索).
                # 注意: 生成端独立维护 rep_depth (学习端 _rep_run 不同步到此)
                if getattr(net, "_mismatch", False) and cur.shape[1] >= 10:
                    gb_recent = cur[:, -10:]  # [N,10]
                    oh_r = F.one_hot(gb_recent, num_classes=256).to(torch.float16)
                    n_uniq = (oh_r.sum(dim=1) > 0).sum(dim=-1)  # [N] 每样本唯一字节数
                    in_rep = (n_uniq < 3).to(torch.float16)  # [N]
                    # _rep_depth 按当前 batch 对齐 (跨 batch 残留会广播错位,
                    # 第 76 轮实测: 训练 batch=8 残留 → 评估 N=1 时 z_bind 扩到 [8,16])
                    rep_depth = getattr(net, "_rep_depth", None)
                    if rep_depth is None or rep_depth.shape[0] != N:
                        rep_depth = torch.zeros(N, device=dev, dtype=torch.float16)
                    rep_depth = torch.where(in_rep > 0, rep_depth + 1.0, torch.zeros_like(rep_depth))
                    net._rep_depth = rep_depth
                    # 错配强度: 0.01 背景 + 0.15·min(深度/10, 1) (裁决 13 数值)
                    strength = 0.01 + 0.15 * (rep_depth / 10.0).clamp(0.0, 1.0)  # [N]
                    # 转移偏好 top-3 目标槽, 随机选一
                    trans = z_bind @ net.W_bind_self  # [N,16]
                    _, top_idx = trans.topk(3, dim=-1)  # [N,3]
                    pick = torch.randint(0, 3, (N,), device=dev)  # [N] 0-2
                    tgt = top_idx.gather(1, pick.unsqueeze(1)).squeeze(1)  # [N]
                    tgt_emb = net.W_bind_self[:, tgt].T  # [N,16] 目标槽转移向量
                    shift = strength.unsqueeze(1) * (tgt_emb - z_bind)
                    z_bind = z_bind + shift  # 持续在线, 全样本施加
                    z_bind = z_bind / (1.0 + z_bind.abs())  # 分流抑制
                    net._mismatch_active = in_rep.mean()  # 诊断: 复读占比
                pot = z_bind @ net.W_act  # [N,256] 动作电位
                # 可打印掩码对齐训练读出空间 (第 76 轮): 0x00-0x1F 物理屏蔽
                mask_print = torch.zeros(256, dtype=torch.float16, device=dev)
                mask_print[32:] = 1.0
                pot = pot + (1.0 - mask_print) * -1e4
                # 频率去偏对齐 W_lm 生成路径 (第 76 轮): 无去偏时 argmax 恒选
                # 最高频字节 (0xEF/续字节/空格) → 屏蔽一个选次高 → � 周期复读
                pot = pot - (6.0 * torch.log(freq_safe)).to(torch.float16).unsqueeze(0)
                last = pot / (pot.abs().max(dim=-1, keepdim=True).values + 1e-4) * 60.0  # fp16
            else:
                a4 = net.active_size["l4"]
                W_diff_a = net.W_diff[:a4, :a4]
                z4_n = net._z4 / (net._z4.norm(dim=-1, keepdim=True) + 1e-3)
                pred_delta = z4_n @ W_diff_a.T + net.b_diff[:a4].unsqueeze(0).unsqueeze(0)
                z4_next = net._z4 + pred_delta
                # 与训练头同款: 分流抑制 → RMS → 三阶 → 记忆池+绑定拼接 → W1 混合层
                z4_nl = z4_next / (1.0 + z4_next.abs())
                z4_nl = _rms(z4_nl)
                z4_nl = z4_nl * (1.0 - 0.5 * z4_nl.pow(2))
                zh_next = torch.cat([z4_nl, net._m2, net._m8, net._m32, net._bind_vec], dim=-1)
                h_next = zh_next @ net.W1
                h_next = h_next / (1.0 + h_next.abs())  # 分流抑制 (与训练同款)
                h_next = h_next / (h_next.square().mean(dim=-1, keepdim=True).sqrt() * 1.01 + 1e-8)
                h_next = h_next * (1.0 - 0.5 * h_next.pow(2))
                inv_h = 1.0 / math.sqrt(net.d_h)
                mu0_top = (h_next @ net.W_lm + net.bias_lm) * inv_h
                logits_c = (mu0_top - mu0_top.mean(dim=-1, keepdim=True)) / (
                    mu0_top.std(dim=-1, keepdim=True) + 1e-4
                )
                mu0_top = logits_c / logits_c.abs().max(dim=-1, keepdim=True).values * 60.0
                mu0_top = mu0_top - (6.0 * torch.log(freq_safe)).to(torch.float16).unsqueeze(0).unsqueeze(0)
                mask_print = torch.zeros(256, dtype=torch.float16, device=dev)
                mask_print[32:] = 1.0
                mu0_top = mu0_top + (1.0 - mask_print) * -1e4
                last = mu0_top[:, -1]  # [N,256] fp16
            if temperature > 0:
                last = last / temperature
            is_utf8_cont = (last_byte >= 0x80) & (last_byte <= 0xBF)
            utf8_run = torch.where(is_utf8_cont, utf8_run + 1, torch.zeros_like(utf8_run))
            if rep_backstop:
                stats["gen"] += N
                # n-gram 周期检测 (第 64 轮): 最近 p 字节模式 == 前 p 字节模式
                # (周期重复 ≥2 次) → 物理屏蔽周期末字节. p=3 覆盖 3 字节字符
                # 循环 (ef bf bd), p=2 覆盖双字符交替, p=4 覆盖字符与 ASCII
                # 交错. 词干 (nn/nn+functional) 无 ≥2 次周期重复, 不受影响
                for p in (2, 3, 4):
                    if cur.shape[1] >= 2 * p:
                        pat = cur[:, -p:]  # [N,p] 最近 p 字节模式
                        prev = cur[:, -2 * p : -p]  # [N,p] 再前 p 字节模式
                        period = (pat == prev).all(dim=-1)  # [N] 周期重复
                        n_block = int(period.sum().item())
                        if n_block:
                            last[period, cur[:, -1][period]] = -1e4
                            stats["rep"] += n_block
                # UTF-8 结构阻断: 当前字节是非法起始字节 (0x80-0xC1) 后紧跟续字节,
                # 或续字节超长 (≥3, 合法单字符上限) — 屏蔽全部续字节 + 非法起始,
                # 模型被迫从 ASCII / 合法起始字节 (0xC2-0xF4) 中选新字符
                bad_start = (last_byte >= 0x80) & (last_byte <= 0xC1)
                overrun = (utf8_run >= 3) | bad_start
                n_utf8 = int(overrun.sum().item())
                if n_utf8:
                    last[overrun, 0x80:0xC2] = -1e4
                    stats["utf8"] += n_utf8
            topv, _ = torch.topk(last, min(15, 256), dim=-1)
            last[last < topv[:, -1:]] = -float("inf")
            probs = torch.softmax(last, dim=-1)
            if temperature <= 0.0 or rep_backstop:
                b = probs.argmax(dim=-1)
            else:
                b = torch.multinomial(probs, 1).squeeze(-1)
            cur = torch.cat([cur, b.unsqueeze(-1)], dim=1)
            last_byte = b
        return cur

    def _recurrent(
        self, mu: torch.Tensor, dev: torch.device, W_t: torch.Tensor, sweeps: int = 1
    ) -> torch.Tensor:
        """时间核递归: z[t] 依赖 z[t-1] (经学习到的 W_t), 纯 matmul 全 fp16.

        因果移位 + 前向卷积式扫描. 只归一化递归项, 不归一化整体 z —
        保留 mu 携带的输入语义区分度 (mu4 跨上下文 cos=0.26, 旧实现递归后
        整体 RMS 归一化把方向抹到 0.985; sweeps=1 单次微扰防递归项累积污染方向).
        """
        N, S, d = mu.shape
        z = mu
        inv_d = 1.0 / math.sqrt(d)
        for _ in range(sweeps):
            shift = torch.cat([torch.zeros(N, 1, d, dtype=z.dtype, device=dev), z[:, :-1]], dim=1)
            shift_n = _rms(shift)
            rec = (shift_n @ W_t.T) * inv_d
            # 递归项单独归一化再乘 inv_d 小扰动: 保 mu 语义方向 (cos 0.69→0.98 抹平),
            # 同时幅度有界防逐层累积 (旧整体 RMS 压幅度但抹方向; 无约束则 58 步 NaN)
            rec = _rms(rec) * inv_d
            z = z + rec
        return z

    def _predict(self, byte_ids: torch.Tensor, store_state: bool = True, is_inference: bool = False) -> dict:
        """核心前馈: 感知 (L0→L6) + 微柱路由 + Foldiak 去相关 + 去中心化 + 增量预测.

        机制:
        - 无位置编码, 时序全靠学习的时间核 W_t 递归
        - 微柱阵列 (L5 拆 4 块, 时间步交错路由), 输出经 Foldiak 去相关后去中心化
          (z5 去均值喂下游保持 PR; z5_raw 原始幅度喂 W_diff 增量预测, 路由分离)
        - ACh 噪声注入 L3 投影输入, is_inference=True 时关闭 (推理确定性)
        """
        net = self.net
        N, S = byte_ids.shape
        dev = byte_ids.device
        a4, a2, a3, a5, a6 = (net.active_size[k] for k in ("l4", "l2", "l3", "l5", "l6"))

        W_04_a = net.W_04[:a4]
        W_42_a = net.W_42[:a2]
        W_23_a = net.W_23[:a3]
        W_56_a = net.W_56[:a6]
        W_diff_a = net.W_diff[:a4, :a4]

        # L0 纯 one-hot; 时序双通道: [z0[t], z0[t-1]] 拼接 (词序信息进入表示层)
        z0 = F.one_hot(byte_ids, num_classes=256).to(torch.float16)  # [N,S,256]
        if net.cfg.input_history:
            z0_prev = torch.cat([torch.zeros(N, 1, 256, dtype=z0.dtype, device=dev), z0[:, :-1]], dim=1)
            z0 = torch.cat([z0, z0_prev], dim=-1)  # [N,S,512]

        mu4 = z0 @ W_04_a.T + net.bias_l4[:a4]
        # --- 诊断插桩 (第 75 轮, 零行为影响): 采集 PR(mu4) 区分 W_04 vs W_t4 坍缩 ---
        net._mu4_diag = mu4
        # --- 插桩结束 ---
        z4 = self._recurrent(mu4, dev, net.W_t4[:a4, :a4])
        z4_n = _l2_norm(z4)
        mu2 = z4_n @ W_42_a.T + net.bias_l2[:a2]
        z2 = self._recurrent(mu2, dev, net.W_t2[:a2, :a2])
        # 预投影 RMSNorm (CLAUDE.md 铁律: 投影前加 RMSNorm 防 fp16 溢出):
        # 只归一化输入, 保方向压尖峰; 不归一化输出 mu, 不抹平差异
        z2_n = _l2_norm(z2)
        mu3 = z2_n @ W_23_a.T + net.bias_l3[:a3]
        if not is_inference:
            mu3 = mu3 + torch.sign(2.0 * (torch.rand_like(mu3) - 0.5)) * 0.03  # ACh 噪声
        z3 = self._recurrent(mu3, dev, net.W_t3[:a3, :a3])
        z3_n = _l2_norm(z3)
        # L5 统一矩阵 (撤销微柱硬切块): 全量 z3 → 单 W_35 [a5, a3]
        z5 = z3_n @ net.W_35[:a5].T + net.bias_l5[:a5]
        # 路由分离: z5_raw 原始幅度喂 W_diff, z5 去中心化喂下游.
        # Foldiak 反赫布侧抑制 (方案 D): z5 -= 0.2·M@z5, M 零起步 = 恒等;
        # M 学 z_out 协方差 → 白化去相关 → 打破行收敛 ±w 的共线激活
        # 注: z5 不做三阶 — z5 有稀疏尖峰结构 (Foldiak 侧抑制孤立放大),
        # 任何范数归一化保留相对形状, 尖峰经三阶翻转放大 → W_35 差分 NaN (实测
        # step 111, 全局 std 与逐行 RMS 均压不住). z5 保留线性, 非线性由 z3 承载
        z5_fd = z5 - 0.2 * (net.M_l5[:a5, :a5] @ z5.transpose(-2, -1)).transpose(-2, -1)
        z5_raw = z5_fd
        z5 = z5_raw - z5_raw.mean(dim=-1, keepdim=True)
        mu6 = z5 @ W_56_a.T + net.bias_l6[:a6]
        z6 = self._recurrent(mu6, dev, net.W_t6[:a6, :a6])

        # 下一状态预测 (显式预测目标): W_diff 在 L4 空间预测 Δz4 = z4[t] - z4[t-1]
        # target_delta = z4_next - z4; pred_delta = z4 @ W_diff;
        # eps_diff = pred_delta - target_delta (训练误差, 驱动 W_diff 学习动力学)
        dz4_pred = z4[:, 1:] - z4[:, :-1]
        dz4_pred = dz4_pred / (dz4_pred.norm(dim=-1, keepdim=True) + 1e-3)  # RMS 归一化
        z4_prev = z4[:, :-1] / (z4[:, :-1].norm(dim=-1, keepdim=True) + 1e-3)
        z4_prev_pad = torch.cat([torch.zeros(N, 1, a4, dtype=z4.dtype, device=dev), z4_prev], dim=1)
        bd = net.b_diff[:a4].unsqueeze(0).unsqueeze(0)
        mu_diff = z4_prev_pad @ W_diff_a.T + bd
        diff_err = (dz4_pred - mu_diff[:, :-1]).square().mean()  # eps_diff 训练误差

        if store_state:
            net._z0 = z0
            net._z4 = z4
            net._z2 = z2
            net._z3 = z3
            net._z5 = z5
            net._z5_raw = z5_raw
            net._z6 = z6
            # 多级记忆池 (3 级因果卷积核, 替代单变量 h): 即时 2 步 / 短时 8 步 /
            # 长时 32 步 — 每级 = 核窗口内 z4 的因果滑动平均 (纯机制, 无超参权重),
            # 位置信息由各级窗口保留, 跨序列经 _m_pool 延续 (零填充 = 接续上序列末态)
            # 注: 保留循环版 — 张量化 (幂次权重矩阵) 在 fp16 下与循环乘法链
            # 舍入不同 (0.5^64 下溢为 0 vs 逐步乘保留), 数学不等价 (第 76 轮实测
            # 最大差 17.9), 不可用于训练
            m_prev = net._m_pool.unsqueeze(0).unsqueeze(0).expand(N, 1, -1).clone()  # [N,1,3a4]
            m2 = m_prev[:, :, :a4]
            m8 = m_prev[:, :, a4 : 2 * a4]
            m32 = m_prev[:, :, 2 * a4 :]
            for t in range(1, S):
                zt = z4[:, t : t + 1]  # [N,1,a4]
                m2 = torch.cat([m2, 0.5 * m2[:, -1:] + 0.5 * zt], dim=1)
                m8 = torch.cat([m8, 0.125 * m8[:, -1:] + 0.875 * zt], dim=1)
                m32 = torch.cat([m32, 0.03125 * m32[:, -1:] + 0.96875 * zt], dim=1)
            net._m_pool = torch.cat([m2[:, -1], m8[:, -1], m32[:, -1]], dim=-1).mean(dim=0)  # [3a4]
            net._m2, net._m8, net._m32 = m2, m8, m32  # [N,S,a4] 每级记忆池
            # 才是"状态是否推进"的信号. tau 按原始量级自动校准 (EMA).
            # 死循环: 帧不再转移 → 慢通道收敛向快通道 → N → 0 (LTD 主动遗忘);
            # 正常推进: z4 持续领先 → N 健康正值 (LTP)
            # 注: 保留循环版 — 张量化 (0.99^t 幂次矩阵) fp16 下溢 (0.99^63≈0.53
            # 但 0.01·0.99^63 累加链舍入不同), 与循环乘法链不等价
            zslow = torch.zeros_like(z4)
            zslow[:, 0] = z4[:, 0]
            for t in range(1, S):
                zslow[:, t] = 0.99 * zslow[:, t - 1] + 0.01 * z4[:, t]
            net._novelty = ((z4 - zslow).square().mean(dim=-1)).detach()  # [N,S] 逐帧新奇度

        # 稀疏绑定 (角色分离三槽): 连续 z4 → W_bind 三块 → 槽内 top-k WTA 硬稀疏
        # bind_vec = [实体(256) | 角色(256) | 谓语(256)] 拼进 W_lm 输入 (第 5 段);
        # 推理也计算 (生成/评估与训练同构)
        if net.cfg.bind_mode != "none":
            self._bind(z4)

        return {
            "mu_diff": mu_diff,
            "diff_err": diff_err,
            "free_energy": diff_err,
        }

    def _bind(self, z4: torch.Tensor) -> None:
        """竞争性概念绑定: z4 → W_bind → K=16 槽位 → 连续值稀疏激活向量.

        第 76 轮终局升级 (裁决: z_bind 表示层从概率分布 → 超完备连续编码):
        - 旧: softmax(T_inv=4) + sum=1 归一化 → 15 维单纯形, 区分度天花板
          ~0.003 量级 → 三任信号源 (外部目标/W_lm 陌生度/W_bind_self 转移
          误差) 全部无梯度失效 (实测).
        - 新: relu(raw - θ) 阈值竞争 + 分流抑制 x/(1+|x|), 无 sum=1 约束.
          输出 = 连续值稀疏向量: 大多数槽零, 少数激活槽携带幅度信息.
          范数成为自由度 → 不同上下文的 z_bind 差异不再被压缩.
        - θ 复用已有分流抑制滑阈 (_theta_bind 跟踪槽能量, 高激活槽被压).
        - 软 WTA 竞争内核保留 (relu 阈值 = 竞争), 无温度参数.
        - W_act/W_bind_self/decorr/自适应全部保持不变 (输入仍是 [N,S,16]).
        """
        net = self.net
        a4 = net.active_size["l4"]
        z4_n = _l2_norm(z4)
        raw = z4_n @ net.W_bind[:a4]  # [N,S,K] 自下而上投影
        if getattr(net, "_bind_loop", True):
            z_bind = torch.relu(raw[:, :1] - net._theta_bind)  # t=0 无历史
            z_binds = [z_bind]
            for t in range(1, raw.shape[1]):
                raw_t = raw[:, t : t + 1] + 0.5 * (z_bind @ net.W_bind_self)
                z_bind = torch.relu(raw_t - net._theta_bind)
                z_binds.append(z_bind)
            z_bind = torch.cat(z_binds, dim=1)
        else:
            z_bind = torch.relu(raw - net._theta_bind)
        # 分流抑制: 上界 1.0, 幅度保留 (无 sum=1 归一化, 范数 = 自由度)
        th_b = net._theta_bind
        th_b.mul_(0.98).add_(0.02 * (z_bind * z_bind).mean(dim=(0, 1)))
        z_bind = z_bind / (1.0 + z_bind.abs())
        net._bind_pre = raw
        net._bind_vec = z_bind  # [N,S,K] 连续值稀疏向量
        # 动作电位 (第 76 轮): 概念槽 → W_act → 256 字节脉冲潜能, 供学习器三因子更新
        net._act_pot = z_bind @ net.W_act  # [N,S,256] (soft_norm 列归一有界)

    def _precise(self, eps: torch.Tensor) -> torch.Tensor:
        """精度加权: π_l = 1/(σ_εl + c), 归一化每层误差尺度."""
        s = eps.std() + 1e-3
        return eps / s
