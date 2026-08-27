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
        # 第 103 轮: UTF-8 语法状态机 — 当前字符还需续字节数 (0 = 字符边界).
        # 替代旧 bad_start 逻辑 (见 blocker 注释: 旧逻辑屏蔽续字节 → 多字节字符
        # 永远无法完成, 生成恒 "lead+1续" 死循环的根因)
        expect_cont = torch.zeros(N, dtype=torch.long, device=dev)
        if not hasattr(net, "_block_stats"):
            net._block_stats = {"rep": 0, "utf8": 0, "gen": 0}
        stats = net._block_stats
        # 第 102e 轮: 频率去偏已全部移除 (见两分支注释), 不再需要 freq_safe
        # 基线; _freq 缓冲保留 (诊断/学习端频率统计用)
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
                # 第 102e 轮: 移除频率去偏 (原 -6·log(freq)/freq_act) — 与 W_lm
                # 分支同因: 去偏量级与电位相当, 把表达结构压平 (chat102d 表达
                # dec 0.03 实证). 表达结构由 W_act 学习塑造 (决策态自洽 + 感知
                # 相位锚定), 去偏是旧 bias 时代的对抗手段, 前提已消除
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
                # 第 104 轮: 三阶输出分流止血 (与 learning.py 第 101 轮同款) —
                # 生成路径此前缺此步, 训练端有 → 部署分布尾巴厚于训练 (|x|>1.4
                # 三阶放大区), 属同族口径错配. 分流 x/(1+|x|), 结构化非 clamp.
                z4_nl = z4_nl / (1.0 + z4_nl.abs())
                zh_next = torch.cat([z4_nl, net._m2, net._m8, net._m32, net._bind_vec], dim=-1)
                # 第 104 轮: zh 整体 RMS 前置 — 与训练端 (learning.py) 和观测器
                # (chat103/104 _wlm_logits) 同口径. 此前生成路径直连 W1, 训练分布
                # (rms(zh)) 与部署分布 (未归一) 不同源 — 与 z4/z4_next 错配同族.
                # 结构化 pre-norm, 非 clamp, 零行为风险 (echo 模式 z4 3x 防护同款).
                zh_next = _rms(zh_next)
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
                # 第 102e 轮: 移除频率去偏 — 与学习端同因 (见 learning.py 同款
                # 注释): 去偏量级与 logits 相当, 把 W_lm 输出结构压平 → 生成
                # 恒平复读. bias_lm 已降 (target=10), 高频垄断前提消除
                mask_print = torch.zeros(256, dtype=torch.float16, device=dev)
                mask_print[32:] = 1.0
                mu0_top = mu0_top + (1.0 - mask_print) * -1e4
                last = mu0_top[:, -1]  # [N,256] fp16
            if temperature > 0:
                last = last / temperature
            if rep_backstop:
                stats["gen"] += N
                # n-gram 周期检测 (第 64 轮): 最近 p 字节模式 == 前 p 字节模式
                # (周期重复 ≥2 次) → 物理屏蔽周期末字节. p=3 覆盖 3 字节字符
                # 循环 (ef bf bd), p=2 覆盖双字符交替, p=4 覆盖字符与 ASCII
                # 交错. 词干 (nn/nn+functional) 无 ≥2 次周期重复, 不受影响.
                # 第 103 轮: 扩展 p=5,6,8,10,12 — 模型逃逸到长周期循环 (周期
                # 6-9 实测), p≤4 挡不住; 24 字节级精确重复必是循环, 非合法文本
                # 第 104 轮: p 连续 2-24 — 旧列表 (2,3,4,5,6,8,10,12) 漏 7/9/11/13/14/15
                # 字节周期 (实测 "句子的"=9B 双词循环逃逸), 长短语复读
                # ("宠物，这些天。"=17B) 需更长 p; 短语级重复必是循环
                for p in range(2, 25):
                    if cur.shape[1] >= 2 * p:
                        pat = cur[:, -p:]  # [N,p] 最近 p 字节模式
                        prev = cur[:, -2 * p : -p]  # [N,p] 再前 p 字节模式
                        period = (pat == prev).all(dim=-1)  # [N] 周期重复
                        n_block = int(period.sum().item())
                        if n_block:
                            # 第 104 轮: 屏蔽整个模式字节值 — 单字节屏蔽 (旧:
                            # 只屏蔽周期末字节) 只挡相位命中时刻, 模型逃逸到
                            # +1 相位继续同一循环 (实测 澘=e6b88c 三轮换挡).
                            # 命中时已完成 >=2 份周期, 屏蔽模式值只影响第 3 份
                            # 及以后 — "谢谢"类双字词 (2 份) 完整保留, 长循环
                            # 被切断. 解码器物理约束, 非学习 clamp.
                            cyc_vals = torch.unique(pat[period])
                            for bv_ in cyc_vals.tolist():
                                last[period, bv_] = -1e4
                            stats["rep"] += n_block
                # UTF-8 语法阻断 (第 103 轮修复): 按 expect_cont 判定合法集合.
                # 旧实现 (round 64) 把"上一字节是续字节"当 bad_start 屏蔽续字节 —
                # 三字节/四字节字符永远无法完成, 生成恒 "lead+1续" 死循环
                # (chat102 全链 dec 0.00-0.09 根因). 正确语义:
                #   expect_cont>0 (字符中途): 下一字节必须是续字节 0x80-0xBF
                #   expect_cont==0 (字符边界): 下一字节是 ASCII 或合法起始 0xC2-0xF4
                # 结构阻断是解码器语法 (与 rep_backstop 同族), 非学习 clamp
                n_utf8 = int((expect_cont > 0).sum().item())
                if n_utf8:
                    last[expect_cont > 0, 0x00:0x80] = -1e4  # 中途禁 ASCII
                    last[expect_cont > 0, 0xC0:0x100] = -1e4  # 中途禁 lead/非法
                    stats["utf8"] += n_utf8
                n_bad = int((expect_cont == 0).sum().item())
                if n_bad:
                    # 边界禁续字节 0x80-0xBF + 非法起始 0xC0-0xC1 (overlong)
                    last[expect_cont == 0, 0x80:0xC2] = -1e4
            topv, _ = torch.topk(last, min(15, 256), dim=-1)
            last[last < topv[:, -1:]] = -float("inf")
            probs = torch.softmax(last, dim=-1)
            if getattr(net, "use_w_act", False) and getattr(net, "_entropy_sample", False):
                # 生成端熵激励 (裁决 17): W_act 分支软采样 + 频率偏置 —
                # pot_bias = pot + β·(1-freq_act), 罕见字节获得生成机会
                # (argmax 确定性采样导致未见过字节永不出现在生成里, 表达库
                # 上限 11-14 的根源). 软采样替代硬 argmax, 温度 1.0.
                # 频率偏置仅作用于生成采样, 不进任何学习更新 (自组织筛选)
                beta = getattr(net, "_entropy_beta", 0.3)
                freq_bias = (beta * (1.0 - net._freq_act)).unsqueeze(0)  # [1,256]
                pot_b = last + freq_bias
                b = torch.multinomial(torch.softmax(pot_b, dim=-1), 1).squeeze(-1)
            elif temperature <= 0.0 or rep_backstop:
                b = probs.argmax(dim=-1)
            else:
                b = torch.multinomial(probs, 1).squeeze(-1)
            cur = torch.cat([cur, b.unsqueeze(-1)], dim=1)
            last_byte = b
            # 更新 UTF-8 字符状态: 边界发合法起始 → 记录续字节需求; 中途发续 → 递减
            is_l2 = (b >= 0xC2) & (b <= 0xDF)
            is_l3 = (b >= 0xE0) & (b <= 0xEF)
            is_l4 = (b >= 0xF0) & (b <= 0xF4)
            lead_len = torch.where(is_l2, 1, torch.where(is_l3, 2, torch.where(is_l4, 3, torch.zeros_like(b))))
            expect_cont = torch.where(expect_cont > 0, expect_cont - 1, lead_len)
        return cur

    def _recurrent(
        self,
        mu: torch.Tensor,
        dev: torch.device,
        W_t: torch.Tensor,
        sweeps: int = 1,
        stp: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        stp_tag: str | None = None,
    ) -> torch.Tensor:
        """时间核递归: z[t] 依赖 z[t-1] (经学习到的 W_t), 纯 matmul 全 fp16.

        因果移位 + 前向卷积式扫描. 只归一化递归项, 不归一化整体 z —
        保留 mu 携带的输入语义区分度 (mu4 跨上下文 cos=0.26, 旧实现递归后
        整体 RMS 归一化把方向抹到 0.985; sweeps=1 单次微扰防递归项累积污染方向).

        stp (第 77 轮): (r, tau, u) 资源慢变量 — 逐帧递归项 ×r[t] (资源耗尽
        → 递归减弱), r 逐帧更新 r ← r + (1−r)/τ − U·r·|z| (活动爆发消耗资源,
        静默期恢复 → 自发间歇). 仅在自由运行 (free_run) 时启用.
        stp_tag: 层名, 用于把末资源写回 net._stp_r_end (供窗末写回)
        """
        N, S, d = mu.shape
        z = mu
        inv_d = 1.0 / math.sqrt(d)
        if stp is None:
            for _ in range(sweeps):
                shift = torch.cat([torch.zeros(N, 1, d, dtype=z.dtype, device=dev), z[:, :-1]], dim=1)
                shift_n = _rms(shift)
                rec = (shift_n @ W_t.T) * inv_d
                # 递归项单独归一化再乘 inv_d 小扰动: 保 mu 语义方向 (cos 0.69→0.98 抹平),
                # 同时幅度有界防逐层累积 (旧整体 RMS 压幅度但抹方向; 无约束则 58 步 NaN)
                rec = _rms(rec) * inv_d
                z = z + rec
            return z
        # STP 路径 (第 77 轮): 逐帧 — 递归项 × 当前资源, 资源随活动消耗/恢复.
        # 第 81 轮 (生命第一因): 双重 RMS 归一化把递归幅度焊死 (~0.03 常数, 与
        # W_t 范数无关) — 递归增益成为硬编码常数, E-I 平衡自由度被架空的直接证据
        # (slope -2.06 冻结). 撤销二重归一化: rec = shift_n@W_t^T (RMS 输入只防
        # 溢出), 幅度由 W_t 行范数承载 — 行范数就是增益, 归系统自己. 防发散不再
        # 靠范数锁, 靠能量: STP 资源耗尽 + 分流抑制 (层外) + Oja (学习端).
        # 递推用 z_out (含 STP 调制的输出), 首帧 = mu 首帧 (无递归)
        r, tau, u = stp
        r = r.unsqueeze(0)  # [1,d]
        inv_tau = (1.0 / tau).unsqueeze(0)  # [1,d]
        u_b = u.unsqueeze(0)  # [1,d]
        z_out = z[:, :1].clone()  # 首帧无递归 (shift=0)
        for t in range(1, S):
            shift = z_out[:, t - 1 : t]  # [N,1,d] 用已调制的输出递推
            shift_n = _rms(shift)
            rec = (shift_n @ W_t.T) * inv_d  # 幅度 = W_t 行范数 (增益自由度)
            z_t = z[:, t : t + 1] + rec * r  # 递归项 × 可用资源
            z_out = torch.cat([z_out, z_t], dim=1)
            # 资源更新: r ← r + (1−r)/τ − U_eff·r·|z|. U_eff (第 81 轮) = U·act/(act+ema):
            # 活动与自身时间均值之比 — 高活动耗尽加速, 低活动恢复. 自参照
            # (rel_n 家族, 零外部目标), ema 首步 = act (U_eff=U/2, 无尖峰)
            act = z_t.abs().mean(dim=1)  # [1,d] 帧均值 (原形状语义)
            ema = getattr(self.net, f"_stp_act_ema_{stp_tag}")[: r.shape[1]]  # 修剪后对齐活性尺寸
            ema.mul_(0.99).add_(0.01 * act.squeeze(0))
            # 第 82 轮: 局部 U 自适应 — 活动高于自身慢基线 → 释放概率升高
            # (钙积累); 活动低于基线 → U 回落. 纯局部负反馈, 无外部目标.
            if self.net.cfg.stp_u_adapt:
                rel = (act - ema.unsqueeze(0)) / (ema.unsqueeze(0) + 1e-6)
                u_b.add_(self.net.cfg.stp_u_adapt_rate * rel * u_b)
                u_b.clamp_(self.net.cfg.stp_u_min, self.net.cfg.stp_u_max)

            u_eff = u_b * act / (act + ema + 1e-6)
            r = r + (1.0 - r) * inv_tau - u_eff * r * act
            r = r.clamp(0.01, 1.0)  # 资源界 (有界防发散)
        if stp_tag is not None:
            self.net._stp_r_end[stp_tag] = r.squeeze(0).clone()  # 窗末写回 (诊断/更新用)
        return z_out

    def _predict(self, byte_ids: torch.Tensor, store_state: bool = True, is_inference: bool = False) -> dict:
        """核心前馈: 感知 (L0→L6) + 微柱路由 + Foldiak 去相关 + 去中心化 + 增量预测.

        机制:
        - 无位置编码, 时序全靠学习的时间核 W_t 递归
        - 微柱阵列 (L5 拆 4 块, 时间步交错路由), 输出经 Foldiak 去相关后去中心化
          (z5 去均值喂下游保持 PR; z5_raw 原始幅度喂 W_diff 增量预测, 路由分离)
        - ACh 噪声注入 L3 投影输入, is_inference=True 时关闭 (推理确定性)
        """
        net = self.net
        free_run = byte_ids is None
        if free_run:
            # 自由运行 (第 77 轮, 生命第一因): 外部输入恒零, 活动由内部递归 +
            # 三尺度加性振荡器驱动. 跨窗延续: 上一窗末帧 z 前置拼接进 _recurrent
            # (shift[0]=0 → carry 帧精确复现), 时间核递归跨窗生效
            N, S = 1, net.cfg.free_run_window
            dev = next(net.parameters()).device
            z0 = torch.zeros(N, S, net._in_dim, dtype=torch.float16, device=dev)
            # 相位向量每窗一次 (查表 index_select, 零运行时三角): 值 = 1-2·phase
            vf = net._osc_f_tab[
                (net._osc_f_cnt.long() + torch.arange(S, device=dev)) % 64
            ].unsqueeze(0).unsqueeze(-1)  # [1,S,1]
            vm = net._osc_m_tab[
                (net._osc_m_cnt.long() + torch.arange(S, device=dev)) % 256
            ].unsqueeze(0).unsqueeze(-1)
            vs = net._osc_s_tab[
                (net._osc_s_cnt.long() + torch.arange(S, device=dev)) % 1024
            ].unsqueeze(0).unsqueeze(-1)
            net._osc_f_cnt.add_(S).remainder_(64)
            net._osc_m_cnt.add_(S).remainder_(256)
            net._osc_s_cnt.add_(S).remainder_(1024)
        else:
            N, S = byte_ids.shape
            dev = byte_ids.device
        a4, a2, a3, a5, a6 = (net.active_size[k] for k in ("l4", "l2", "l3", "l5", "l6"))

        W_04_a = net.W_04[:a4]
        W_42_a = net.W_42[:a2]
        W_23_a = net.W_23[:a3]
        W_56_a = net.W_56[:a6]
        W_diff_a = net.W_diff[:a4, :a4]

        if not free_run:
            # L0 纯 one-hot; 时序双通道: [z0[t], z0[t-1]] 拼接 (词序信息进入表示层)
            z0 = F.one_hot(byte_ids, num_classes=256).to(torch.float16)  # [N,S,256]
            if net.cfg.input_history:
                z0_prev = torch.cat([torch.zeros(N, 1, 256, dtype=z0.dtype, device=dev), z0[:, :-1]], dim=1)
                z0 = torch.cat([z0, z0_prev], dim=-1)  # [N,S,512]

        mu4 = z0 @ W_04_a.T + net.bias_l4[:a4]
        if free_run:
            mu4 = mu4 + net.cfg.osc_amp_m * vm  # 中节律加性电流 → L4 预激活
        # --- 诊断插桩 (第 75 轮, 零行为影响): 采集 PR(mu4) 区分 W_04 vs W_t4 坍缩 ---
        net._mu4_diag = mu4
        # --- 插桩结束 ---
        # 去窗口化 (第 77 轮裁决): 移除跨窗 carry 拼接 — 自由运行是连续时间演化,
        # 每窗首帧从零/注入起步 (无上一窗末帧注入, 消除跨窗能量累积通道)
        if free_run:
            stp4 = (net._stp_r_l4[:a4], net._stp_tau_l4[:a4], net._stp_u_l4[:a4])
        else:
            stp4 = None
        z4 = self._recurrent(mu4, dev, net.W_t4[:a4, :a4], stp=stp4, stp_tag="l4" if free_run else None)
        z4_n = _l2_norm(z4)
        mu2 = z4_n @ W_42_a.T + net.bias_l2[:a2]
        if free_run:
            mu2 = mu2 + net.cfg.osc_amp_m * vm  # 中节律 → L2
        z2 = self._recurrent(
            mu2, dev, net.W_t2[:a2, :a2],
            stp=(net._stp_r_l2[:a2], net._stp_tau_l2[:a2], net._stp_u_l2[:a2]) if free_run else None,
            stp_tag="l2" if free_run else None,
        )
        # 预投影 RMSNorm :
        # 只归一化输入, 保方向压尖峰; 不归一化输出 mu, 不抹平差异
        z2_n = _l2_norm(z2)
        mu3 = z2_n @ W_23_a.T + net.bias_l3[:a3]
        if free_run:
            mu3 = mu3 + net.cfg.osc_amp_s * vs  # 慢节律 → L3
        if not is_inference:
            mu3 = mu3 + torch.sign(2.0 * (torch.rand_like(mu3) - 0.5)) * 0.03  # ACh 噪声
        z3 = self._recurrent(
            mu3, dev, net.W_t3[:a3, :a3],
            stp=(net._stp_r_l3[:a3], net._stp_tau_l3[:a3], net._stp_u_l3[:a3]) if free_run else None,
            stp_tag="l3" if free_run else None,
        )
        # 分流抑制 (第 78 轮, 仅自由运行): z3 上界 ±1 — 断 z3→z5→eps_b→phi_b
        # 尖峰链 (第 77 轮 NaN 根因), 非线性由 z3 承载 (见下方注释)
        if free_run:
            z3 = z3 / (1.0 + z3.abs())
        z3_n = _l2_norm(z3)
        # L5 统一矩阵 (撤销微柱硬切块): 全量 z3 → 单 W_35 [a5, a3]
        z5 = z3_n @ net.W_35[:a5].T + net.bias_l5[:a5]
        if free_run:
            z5 = z5 + net.cfg.osc_amp_f * vf  # 快节律 → L5 预激活
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
        if free_run:
            mu6 = mu6 + net.cfg.osc_amp_s * vs  # 慢节律 → L6
        z6 = self._recurrent(
            mu6, dev, net.W_t6[:a6, :a6],
            stp=(net._stp_r_l6[:a6], net._stp_tau_l6[:a6], net._stp_u_l6[:a6]) if free_run else None,
            stp_tag="l6" if free_run else None,
        )

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
            if free_run:
                # 去窗口化 (第 77 轮裁决): 不再传递 z 末帧 — 连续时间演化,
                # 每窗独立起步 (消除跨窗能量累积). STP 资源状态跨窗延续
                for sln, sz in (("l4", z4), ("l2", z2), ("l3", z3), ("l6", z6)):
                    if sln in net._stp_r_end:
                        self._stp_update(sln, net._stp_r_end[sln], sz)
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
            net._fr_vf = vf if free_run else None  # 快节律向量供 _bind 注入
            self._bind(z4)
            net._fr_vf = None
        else:
            net._fr_vf = None

        return {
            "mu_diff": mu_diff,
            "diff_err": diff_err,
            "free_energy": diff_err,
        }

    def _stp_update(self, layer: str, r_end: torch.Tensor, z: torch.Tensor) -> None:
        """STP 资源写回 (窗末): r 末帧写回 buffer (跨窗延续资源状态).
        τ/U 固定初值, 不再自适应 (第 78 轮裁决: EMA 类失败机制移除)."""
        r_buf = getattr(self.net, f"_stp_r_{layer}")
        r_buf[: r_end.shape[0]].copy_(r_end)  # 修剪后 active 收缩: 只写头部切片

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
            # 自由运行快节律注入 (第 77 轮): 加性电流 → z_bind 预激活 (基准幅度)
            if getattr(net, "_fr_vf", None) is not None:
                raw = raw + net.cfg.osc_amp_f * net._fr_vf
            # STP 槽资源 (第 77 轮): 逐帧 z_bind 输出 ×r (资源耗尽 → 槽间歇切换),
            # r 随槽活动消耗/恢复 (同递归层规则)
            stp_bind = None
            if getattr(net, "_fr_vf", None) is not None:
                stp_bind = (net._stp_r_bind, net._stp_tau_bind, net._stp_u_bind)
            # 跨窗延续 (去窗口化第 77 轮): 不再传递槽末帧 — 每窗独立起步
            z_bind = self._bind_sparse(raw[:, :1] - net._theta_bind)  # t=0 无历史
            z_binds = [z_bind]
            r_bind = stp_bind[0].unsqueeze(0) if stp_bind else None
            inv_tau_b = (1.0 / stp_bind[1]).unsqueeze(0) if stp_bind else None
            u_bind_b = stp_bind[2].unsqueeze(0) if stp_bind else None
            for t in range(1, raw.shape[1]):
                raw_t = raw[:, t : t + 1] + 0.5 * (z_bind @ net.W_bind_self)
                if stp_bind:
                    raw_t = raw_t * r_bind  # 槽输入 × 可用资源
                z_bind = self._bind_sparse(raw_t - net._theta_bind)
                z_binds.append(z_bind)
                if stp_bind:
                    act = z_bind.abs().mean(dim=1)  # [1,K] (同递归层修正)
                    ema_b = getattr(net, "_stp_act_ema_bind")[: r_bind.shape[1]]
                    ema_b.mul_(0.99).add_(0.01 * act.squeeze(0))
                    # 第 82 轮: 槽位局部 U 自适应 (同递归层, 纯局部负反馈).
                    if self.net.cfg.stp_u_adapt:
                        rel_b = (act - ema_b.unsqueeze(0)) / (ema_b.unsqueeze(0) + 1e-6)
                        u_bind_b.add_(self.net.cfg.stp_u_adapt_rate * rel_b * u_bind_b)
                        u_bind_b.clamp_(self.net.cfg.stp_u_min, self.net.cfg.stp_u_max)

                    u_eff_b = u_bind_b * act / (act + ema_b + 1e-6)  # 第 81 轮自参照耗尽
                    r_bind = r_bind + (1.0 - r_bind) * inv_tau_b - u_eff_b * r_bind * act
                    r_bind = r_bind.clamp(0.01, 1.0)
            if stp_bind and r_bind is not None:
                net._stp_r_end["bind"] = r_bind.squeeze(0).clone()
            z_bind = torch.cat(z_binds, dim=1)
        else:
            z_bind = self._bind_sparse(raw - net._theta_bind)
        # 分流抑制: 上界 1.0, 幅度保留 (无 sum=1 归一化, 范数 = 自由度)
        th_b = net._theta_bind
        th_b.mul_(0.98).add_(0.02 * (z_bind * z_bind).mean(dim=(0, 1)))
        z_bind = z_bind / (1.0 + z_bind.abs())
        net._bind_pre = raw
        net._bind_vec = z_bind  # [N,S,K] 连续值稀疏向量
        if getattr(net, "_fr_vf", None) is not None:
            if "bind" in net._stp_r_end:
                self._stp_update("bind", net._stp_r_end["bind"], z_bind)  # 槽 STP 资源写回
        # 动作电位 (第 76 轮): 概念槽 → W_act → 256 字节脉冲潜能, 供学习器三因子更新
        net._act_pot = z_bind @ net.W_act  # [N,S,256] (soft_norm 列归一有界)

    def _bind_sparse(self, pre_act: torch.Tensor, k: int = 4) -> torch.Tensor:
        """槽稀疏竞争 (第 102q 轮, 用户批准架构变更): 每位置只保留 top-k 强
        激活槽, 其余清零 — 替代 relu(raw-θ) 的"全部正激活通过".

        背景 (chat102 全链实证): relu 竞争无 WTA 机制 → z_bind 每位置激活
        11-16 槽 (稠密) → 槽模式间 cos 0.7 (首/续字节) → 线性读出 pot 趋
        均匀 (中心极限) → W_act 学不出条件映射. 稀疏 top-k 恢复区分度:
        - 保留连续值幅度 (relu 后值, 非二值) — 第 76 轮"超完备连续编码"
          的语义不变, 范数仍是自由度
        - k=4/32 (12.5%) — 强竞争但保留多样 (软 WTA 时代 K=16 top-k=10
          的先例是 62%)
        - 阈值 θ 仍起"弱激活过滤"作用 (先 relu(raw-θ), 再 top-k 选择)
        - 纯张量: topk 零 GPU→CPU 同步, fp16 安全
        """
        x = torch.relu(pre_act)  # [...,K] 弱激活过滤
        if x.shape[-1] <= k:
            return x
        # 每位置 top-k 掩码: 只保留最大的 k 个值
        thr = x.topk(k, dim=-1).values[..., -1:]  # 第 k 大值
        mask = (x >= thr).to(x.dtype)
        return x * mask

    def _precise(self, eps: torch.Tensor) -> torch.Tensor:
        """精度加权: π_l = 1/(σ_εl + c), 归一化每层误差尺度."""
        s = eps.std() + 1e-3
        return eps / s
