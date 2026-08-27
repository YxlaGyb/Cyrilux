"""表达端域: W_act 闭环自洽软目标 + 生存信号 R (资格迹) + 复读诊断.

从原 learn() 单函数按块拆分, 块内语句顺序逐行保持 (数值逐位等价).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ...modulation import soft_norm_preserve
from ._common import _elig_accum, _MixinBase


class ActionMixin(_MixinBase):
    """表达端域 (方法挂载到 LearningEngine)."""

    def _update_w_act(self, ctx, sh):
        """闭环自洽生成学习 (第 76 轮最终裁决: 表达者范式), 仅 closed_loop/自回声.

        W_act 从"预测器"转"行为生成器": 切断一切外部目标字节驱动.
        (第 102h 轮: W_act 块从冻结块中提出 — 行为域, 回声相位必须学习.)
        """
        net = self.net
        dev = ctx.dev
        if (
            not (ctx.closed_loop or ctx.echo_loop)
            or not hasattr(net, "_act_pot")
            or not hasattr(net, "_gen_bytes")
            or not getattr(net, "_act_enabled", True)
        ):
            return
        N = ctx.N
        pot = net._act_pot  # [N,S,256] 动作电位
        zbind = net._bind_vec
        gb = net._gen_bytes  # [N,N_gen] 自生成字节 (表达)
        N_gen = gb.shape[1]
        # 第 102 轮: 决策态对齐偏移 — zb_g 与 gb_a/oh_g/probs_g 同偏移
        # (槽状态取"字节进入前"的决策时点, 见下方对齐注释)
        n_a = N_gen - 1  # 决策态对齐长度
        zb_g = zbind[:, -N_gen:-1]  # [N,n_a,16] 决策时点槽状态
        # ── 学习进步信号 (裁决 16): 从"最小化预测误差"到"最大化学习进步" ──
        # 复读是"最小化当前预测误差"的全局最优解 — 必须放弃该目标.
        # LP_t = ε_{t-1} - ε_t (预测误差的负时间差分):
        #   LP>0: 误差下降, 系统"学会了" → 正强化 (保留该行为)
        #   LP<0: 误差上升, 系统"变笨了" → 负强化 (抑制该行为)
        # 复读: ε_{t-1}≈ε_t → LP≈0 → 行为不被强化 → 逐渐遗忘 (打破复读)
        # 探索新字节若提升可预测性 (LP>0) → 强化 → 表达库向复杂演化.
        # 多巴胺真实角色: 奖励预测误差 (比预期更好 = 学习进步), 非奖赏本身
        # 实现: φ = 0.5·(1 - tanh(α·LP)) ∈ [0,1] — LP 大 → φ→0 (弱抑制=
        # 保留), LP 小 → φ→1 (强抑制). 符号保持, 幅度受限, 纯局部.
        # (裁决公式 ΔW=-η·tanh(LP)·... 与"LP>0 正强化"矛盾, 按物理
        # 意图实现: LP>0 保留, LP<0 抑制)
        # 第 102 轮: LP 轨迹同用决策态对齐后的槽序列 (zb_g 在下方定义,
        # 与 gb_a/oh_g 同偏移 — 转移误差 ε 的时序与表达字节时序一致)
        zb_prev = torch.cat([zb_g[:, :1], zb_g[:, :-1]], dim=1)  # [N,n_a,16]
        pred_self = torch.softmax((zb_prev @ net.W_bind_self) * 4.0, dim=-1)
        eps_t = (zb_g - pred_self).square().mean(dim=-1, keepdim=True)  # [N,n_a,1] ε_t
        eps_prev = torch.cat([eps_t[:, :1], eps_t[:, :-1]], dim=1)  # ε_{t-1}
        LP = (eps_prev - eps_t).detach()  # 学习进步 [N,n_a,1]
        alpha = 50.0  # tanh 缩放 (ε 量级 ~0.003, ×50 → LP 归一)
        phi = 0.5 * (1.0 - torch.tanh(alpha * LP))  # 增益 [0,1]
        # 第 102 轮修正 (chat101 15k 步失败根因 4): 决策态对齐 —
        # 旧实现把 gb[t] 与 probs_lm 后段逐位配对: probs_lm 的 t' 行是
        # "基于 t' 之前上下文的预测", 其中 gb[t'] 已进入上下文 → p_gen =
        # P(g_t | 上下文含 g_t) 在复读时恒 1 (生成 = 继续复读同字节),
        # 自洽惊喜恒 0 → 复读被三因子负号规则"低惊喜 → 弱负向保留"
        # 永久强化 = 复读自锁 (第 76 轮已知吸引子).
        # 修正: 决策态对齐 — gb[t] 的决策上下文止于 gb[t-1], 对应
        # probs_lm 的 t-1 行 (含该字节之前的状态). 三要素统一偏移:
        #   gb_aligned = gb[:, 1:]           (t ≥ 1, 首字节决策上下文
        #                                     是回声种子/锚定段, 不在流内)
        #   probs_g = probs_lm[:, S-1-n_a : S-1]  (t-1 行, n_a = N_gen-1)
        #   zb_g     = zbind[:, -N_gen:-1]  (决策时点槽状态 = 字节进入前)
        # 偏移后 p_gen = P(g_t | 上下文不含 g_t) — 复读时 <1 (软采样给
        # 其他字节留了概率) → 复读的自洽惊喜 >0 → 负向抑制 → 复读不
        # 被固化; 表达向"高概率字节"收缩, 与感知相位学的频率结构对齐.
        # 零新机制, 纯对齐修正.
        gb_a = gb[:, 1:]  # [N,n_a] 对齐生成字节
        oh_g = F.one_hot(gb_a, num_classes=256).to(torch.float16)  # [N,n_a,256]
        # 决策态分布: probs_lm[i] = P(第 i+1 字节 | 前 i+1 字节上下文).
        # gb[t] 在完整序列位置 gb_start+t, 其决策上下文 = 前 gb_start+t 个
        # 字节 → 决策态行 = probs_lm[gb_start+t-1]. 对 gb_a[i]=gb[i+1]:
        # 行 = probs_lm[gb_start+i], i=0..n_a-1
        gb_start = ctx.S - N_gen  # 生成段在完整序列中的起始位置
        probs_g = sh.lm.probs_lm[:, gb_start : gb_start + n_a]  # [N,n_a,256]
        # (第 102c 轮: 回声锚定种子后 inp = 纯生成流 [N,S-1], gb_start=0
        #  恒成立 — gb_a[i] 的决策态 = probs_lm[i] (上下文 = 种子+生成段
        #  前缀). 闭环时 gb_start=64, 切片含锚定段后语义不变)
        p_gen = (probs_g * oh_g).sum(dim=-1, keepdim=True)  # [N,n_a,1]
        wlm_err = (1.0 - p_gen).detach()  # [N,n_a,1] 外部弱约束
        """
        第 102b 轮修正 (chat102 4000 步实证): surprise 权重交换 —
        旧 0.9·phi + 0.1·wlm_err 是第 76 轮"从成熟检查点起步"的裁决:
        世界模型已成熟时 W_bind_self 转移误差是可靠的自洽信号. 但
        从零重建时 W_bind_self 也是随机的, phi (LP 门控转移误差) 与
        字节结构无关 → 0.9 权重把 W_act 学习引向与字节无关的方向
        (chat102: W_lm hit3 0.53 已学结构, 表达 dec 仍 0.03 实证).
        wlm_err (W_lm 决策态概率) 是唯一携带字节结构的信号 — 交换
        权重让表达向"世界模型高概率的下一字节"收缩: 非法字节概率
        低 → 高 surprise → 负向抑制; 合法 UTF-8 字节概率高 → 弱
        抑制保留. 牙牙学语机制: 婴儿发声被自己听觉模型筛选.
        """
        surprise = (0.1 * phi + 0.9 * wlm_err).detach()  # 世界模型主导 + 内部转移弱约束
        # ── 字节频率门控 (裁决 15): 罕见字节抑制衰减 (辅助调制) ──
        fa = net._freq_act
        oh_gen = F.one_hot(gb_a, num_classes=256).to(torch.float16)  # [N,n_a,256]
        fa.mul_(0.99).add_(0.01 * oh_gen.mean(dim=(0, 1)))
        beta = 0.5  # 新颖偏好强度
        freq_gate = (1.0 - beta * (1.0 - fa)).unsqueeze(0).unsqueeze(0)  # [1,1,256]
        surprise = surprise * freq_gate  # 逐字节频率调制
        net._LP = LP  # 诊断: 学习进步分布
        """
        第102g轮修正:
        软目标学习替代三因子负号
        旧三因子因 W_lm 预测差恒高无区分度使 W_act 学不到, 陷入死循环.
        修正为软目标赫布: dW_act = zb_g^T @ (probs_g - oh_g),
        用世界模型预测分布作为目标, 使 W_act 向内部模型收缩
        决策态对齐:槽状态与目标分布配对.
        第102j轮修正:
        梯度平衡 — onehot 项抑制强于 probs 项强化, 净梯度无结构. 将 probs 项乘以 K=8 使强化主导.
        """
        dW_act = (zb_g.transpose(-2, -1) @ (8.0 * probs_g - oh_g)).mean(dim=0)
        # 稳态抑制保留 (防字节垄断): 全列 -= 0.1·z_bind^T @ softmax(potential)
        pot_sm = torch.softmax(pot[:, -N_gen:-1].detach(), dim=-1)  # 决策态对齐
        dW_act = dW_act - 0.1 * (zb_g.transpose(-2, -1) @ pot_sm).mean(dim=0)
        # 单步幅度上界 (W_lm 同款幅度-方向解耦)
        dW_act = dW_act / (dW_act.norm() + 1e-8)
        """
        信任域修正: 旧版单位向量×0.2 导致随机游走, 收敛至 1% 信任域; 但软目标 dW 期望近零, 1% 太慢, 提高至 5%.
        信任域 bug 修复: 旧式用整个矩阵范数致实际步幅过小, 改为平均列范数, 系数 5.0 实现真正 5% 信任域.
        学习率: W_lm 读出端量级 × 0.2 × intrinsic 并列调制.
        """
        intr_d = getattr(net, "_intr_drive", torch.tensor(0.5, device=dev, dtype=torch.float16))
        col_norm = net.W_act.data.norm(dim=0).mean()  # 平均列范数 (~1.0)
        """
        第 108 轮: 内部裁判 R (用户公式 ΔW_act = η·R·E, 资格迹通道)
        信号 = W_lm 对本窗生成字节串 s 的预测误差 ε_lm(s) 的时间差分:
          R = 0.05·tanh((ε_prev − ε_now)/0.05)
        ε_now = wlm_err 串级均值 (现成量, 零额外前向); Δε>0 (本窗比上窗
        更可预测) → 正奖励 — "发出该字节串的突触"经迹 E 与"被世界模型
        认可"绑定. 纯内部: 零解码器, 零外部标签, 零人类反馈; W_lm 在
        回声相位冻结 (裁判不变). 未开 _survival_signal 时 R=0 既有零变.
        ε_lm 峰值 ≈0.6-0.9, 步间差 0.01-0.2; tanh 归一 0.05 按 107 校准
        量级 (R≈±0.04-0.05 与 dW 同量级 → 学习发生); 复读时 Δε≈0 → R≈0
        (不强化复读自身, 只强化"更可预测"的探索).
        """
        # 第 108b 修正: 差分基线从"上一窗 ε"改"慢速预期 EMA" — 逐窗差分正负
        # 平衡 (108 探针实证 249/246) 无净信号; EMA 基线慢漂移只跟踪预期,
        # 持续改进 → 持续正 R (净学习方向), 单窗噪声不主导.
        lm_eps_prev = getattr(net, "_lm_eps_ema", None)
        if lm_eps_prev is not None:
            lm_deps = lm_eps_prev - wlm_err.mean()
            net._survival_signal = 0.05 * torch.tanh(lm_deps / 0.05)
        else:
            net._survival_signal = torch.zeros(1, dtype=torch.float16, device=dev)
        net._lm_eps = wlm_err.mean()  # 诊断: 本窗 ε_lm (fp16 张量, 零同步)
        if lm_eps_prev is None:
            net._lm_eps_ema = wlm_err.mean().detach().clone()
        else:
            net._lm_eps_ema.mul_(0.98).add_(0.02 * wlm_err.mean())
        # 第 105 轮 (资格迹): W_act 是行为域 — 轨迹接力, R=生存信号
        # 时"发出字节的突触"与"迟到的后果"跨窗口绑定. 迹输入 = 行为外积
        # (决策态槽 × 实际发出字节 one-hot), 即用户公式 W_act 生成字节串 s
        # 的轨迹本身 — 非归一化软目标 (软目标在有 R 时会让 R·E 绑定进
        # 乱码噪声方向, 108 探针实证: R 正负 249/246 无净信号).
        zbg_ac = zb_g - zb_g.mean(dim=1, keepdim=True)
        dW_elig = (zbg_ac.transpose(-2, -1) @ oh_g).mean(dim=0)
        dW_act = dW_act + _elig_accum(net, "W_act", dW_elig) * getattr(
            net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16)
        )
        net.W_act.data += dW_act * (net.cfg.lm_lr_boost * 0.2) * (0.5 + intr_d) * col_norm * 5.0
        # ── 复读检测跟踪 (裁决 12): 随机扰动已升级为内部状态错误配对
        # (forward.py 生成路径, W_bind_self 定向扰动替代各向同性噪声).
        # 此处保留复读状态跟踪供诊断/监控
        gb_recent = gb[:, -10:]  # [N,10] 最近 10 字节
        oh_r = F.one_hot(gb_recent, num_classes=256).to(torch.float16)  # [N,10,256]
        n_uniq = (oh_r.sum(dim=1) > 0).sum(dim=-1).to(torch.float16)  # [N]
        rep_run = getattr(net, "_rep_run", torch.zeros(N, device=dev, dtype=torch.float16))
        in_rep = (n_uniq < 3).to(torch.float16)  # [N]
        rep_run = torch.where(in_rep > 0, rep_run + 1.0, torch.zeros_like(rep_run))
        net._rep_run = rep_run
        net._rep_frac = in_rep.mean().to(torch.float16)  # 诊断 (fp16 张量, 零同步)
        # 列范数保持 (0.8-1.2): 槽列有界防溢出; 转置视图 in-place 直接
        # 写回原参数 (第 76 轮修复: 旧 .contiguous() 副本装饰性失效)
        soft_norm_preserve(net.W_act.data.T)
