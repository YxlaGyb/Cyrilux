"""读出端学习域: W_lm 信号构建 + LM 头更新 (W_lm/W_lm_2/W1/bias_lm).

从原 learn() 单函数按块拆分, 块内语句顺序逐行保持 (数值逐位等价).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modulation import soft_norm_preserve
from ..forward import _l2_norm, _rms
from ._common import LM_TRUST_REGION, _elig_accum, _MixinBase


class ReadoutMixin(_MixinBase):
    """读出端域 (方法挂载到 LearningEngine)."""

    def _build_lm_signal(self, ctx):
        """LM 头前向信号 (原 L401-574): 输入调制 → 混合层 → logits → 误差 → 投影.

        注意: 原代码此块只在 free_run 跳过 (回声相位同样构建 — W_act 需要
        probs_lm 决策态), 无 echo_world_frozen 守卫.
        """
        net = self.net
        if ctx.free_run:
            return None
        dev, N = ctx.dev, ctx.N
        net = net
        a4 = ctx.a4
        z4 = net._z4

        # 预测编码闭环: W_lm 误差投影回 z4 作为 top-down 误差, 迫使表示层重组以预测下一字节 (纯赫布).
        # W_lm 输入五通道 [z4, m2, m8, m32, bind]; bind 由 z4 经 W_bind 三槽 top-k 生成, 承载角色结构; 记忆池承载跨序列环境.
        # 输入前处理: 能量调制 (分流抑制 x/(1+|x|), τ=1.0 防 NaN), RMS 前置防溢出, 三阶非线性 f(x)=x(1-0.5x²).
        # 第104轮修复: W_lm 输入改用 z4_next = z4 + W_diff 预测差, 对齐训练/生成管线 (旧版直接用 z4 导致分布不匹配).
        z4_n_ = z4 / (z4.norm(dim=-1, keepdim=True) + 1e-3)
        pred_delta_ = z4_n_ @ net.W_diff[:a4, :a4].T + net.b_diff[:a4].unsqueeze(0).unsqueeze(0)
        z4r = z4 + pred_delta_
        z4_lm = z4r / (1.0 + z4r.abs())
        z4_lm = _rms(z4_lm)
        z4_lm = z4_lm * (1.0 - 0.5 * z4_lm.pow(2))
        # 第 101 轮: 三阶输出分流止血 (与 h 路径第 57 轮同构) — 三阶激活在
        # |x|>1.4 进入放大区 (f(x) 反超 x), 从零初始化 z4 的 4.5σ 尾部穿透
        # RMS 后仍存在, 实测 z4_lm_max 20-30 → zh 尖峰 35.8 → h_raw fp16
        # 溢出 → W1 NaN (chat101 step 44). 分流 x/(1+|x|) 把尾部压回 ≤1,
        # 正常区 (|x|<1) 近似恒等 — 预训练检查点行为不变. 结构化 pre-norm,
        # 非 clamp (CLAUDE.md 合规)
        z4_lm = z4_lm / (1.0 + z4_lm.abs())
        # 输出缩放 1/√H (CLAUDE.md: 投影输出溢出 → 乘 1/√H)
        # 竞争性记忆单元群 (第 109 轮): K 个泄漏积分单元 (时间常数自持, 出生/死亡
        # 由 _update_mem_units 驱动) → zh 尾段. 单元块含 z4 段历史的频带分解,
        # 与 z4_lm 直连段互补 (直接 vs 卷积摘要)
        zh = torch.cat([z4_lm, net._bind_vec, net._mem_out], dim=-1)  # [N,S,a4+32+K·a4]
        # 第 80 轮: zh 整体 RMS 前置 (与更新侧 _rms(zh) 对称) — z4_lm 段三阶在
        # z4 幅度大时进入放大区 (echo 模式 W_04 解冻, z4 比训练态大 3 倍),
        # 段间量级差异 → zh 尖峰 → h 厚尾 → h_deriv 爆炸 → dW1 fp16 累加溢出
        zh = _rms(zh)

        # ── 非线性混合层 (第 57 轮): zh → W1 → h → 三阶激活 → W_lm → logits ──
        # h = zh @ W1 [d_h=256]; h = h·(1-0.5h²) 多项式激活 (FP16 原生安全);
        # logits = h @ W_lm. W1 横向交叉组合 z4 信息 (非纵向几何缩放),
        # 把高频 e 列打散到不同子空间. 池门控 (旧机制) 随线性读出一并移除
        # 分流抑制 (第 75 轮裁定): h = zh@W1 投影产生 9.4σ 厚尾 (W1 列极化在
        # 训练中生长: 尖峰 → Hebbian 强化 → 更大尖峰 正反馈), 前向路径必须
        # 掐断 — 与 z4_lm 同款 x/(1+|x|), τ=1.0 输出渐近界 1.0 < 1.4 安全线,
        # 处处可微, 尖峰压缩保留 (非 clamp)
        d_h = net.d_h
        W1_a = net.W1  # [lm_in, d_h]
        h = zh @ W1_a  # [N,S,d_h]
        h = h / (1.0 + h.abs())  # 分流抑制 (止血, W1 稳态机制另行讨论)
        # h 前 RMS 归一化 (同 z4 调制模式): zh 4112 维点积 → h 值域 ~±37,
        # 直接三阶激活 f(h)=h·(1-0.5h²) 对 |h|>1.4 进入放大区 → 爆炸 NaN
        # (实测 e_h max 632 → dW1 inf, step 9). RMS 压到 std≈1 进饱和区
        h = _rms(h)
        # 安全监控插桩 (第 75 轮): max|h_in| 距 1.4 放大区余量 (学习器诊断用)
        net._h_in_max = h.abs().max().detach()
        # 三阶激活 f(x)=x·(1-0.5x²) 的输入 x = RMS 后 h (std≈1); 导数 1-1.5x² 必须
        # 用激活输入算 — 用激活输出 f(x) 算导数, |x|>1.4 放大区 |f(x)|>|x| 使导数
        # 无界 (-28000 级), e_h = err@W_lm.T · h_deriv 在 fp16 溢出 inf → dW1 NaN
        h_in = h
        h = h_in * (1.0 - 0.5 * h_in.pow(2))  # 多项式激活, 零 BP
        h_deriv = 1.0 - 1.5 * h_in.pow(2)  # 激活导数 (转置误差传播用, 纯张量)
        inv_h = 1.0 / math.sqrt(d_h)
        logits_lm = (h @ net.W_lm + net.bias_lm) * inv_h  # [N,S,256]
        # ── 读出端能量调制 + 可打印掩码 (第 54/55/56 轮) ──
        # 1) 能量调制 (完整标准化): 先中心化 (减均值 — raw mean +0.03 被 ×60
        #    放大成 +3.11 系统性偏移), 再归一化 (除 std); 然后 max_abs 归一化
        #    严格落在 [-60, +60] (fp16 黄金法则: 避免极值溢出 — std 缩放把
        #    raw 尖峰 5.0 放大到 459, softmax 饱和单字节主导, 实测)
        logits_c = (logits_lm - logits_lm.mean(dim=-1, keepdim=True)) / (
            logits_lm.std(dim=-1, keepdim=True) + 1e-4
        )
        logits_lm = logits_c / logits_c.abs().max(dim=-1, keepdim=True).values * 60.0
        # 2) 可打印物理掩码: 0x20-0xFF 合法, 0x00-0x1F 强制 -1e4 (fp16 安全极弱值)
        # 第 102e 轮: 移除频率去偏 (logits -= 6·log(freq)) — 第 54 轮加去偏
        # 是为对抗 bias_lm 范数锁 100 的高频垄断; 第 102 轮已把 bias_lm 降到
        # target=10 (bias_std 0.625 与 h 同量级), 去偏前提消除. 诊断实证
        # (chat102d_step900): 任何去偏系数 (1-6) 使 hit1/hit3 归零 — 去偏
        # 量级 (~8-46) 与 logits 相当, 把 W_lm 学习的误差信号抹平: 训练和
        # 生成共用同一去偏, W_lm 学到"输出反相去偏"的权重 → 两相抵消 →
        # 输出恒平 (hit1 0.03 位置无关, chat102 4000 步实证). 频率统计
        # (_freq) 保留 (诊断/其他路径用), 只是不再注入 logits.
        if ctx.closed_loop:
            target_oh = F.one_hot(ctx.byte_ids[:, 1:], num_classes=256).to(torch.float16).mean(dim=(0, 1))
        else:
            target_oh = F.one_hot(ctx.byte_ids, num_classes=256).to(torch.float16).mean(dim=(0, 1))
        # 第 102 轮: 回声相位冻结频率统计 — 乱码输入流不配做世界结构
        if not ctx.echo_world_frozen:
            net._freq.mul_(0.99).add_(0.01 * target_oh.detach())
        mask_print = torch.zeros(256, dtype=torch.float16, device=dev)
        mask_print[32:] = 1.0
        logits_lm = logits_lm + (1.0 - mask_print) * -1e4
        # 池间侧抑制 (旧线性读出机制) 随 W1 混合层移除 — 池门控依赖逐段
        # W_lm 行映射, 混合层 h 为折叠空间 (无段映射), 不再适用

        # 多步预测 (Q3 解耦): W_lm 专责 t+1, W_lm_2 独立子预测器专责 t+2.
        # 共享混合特征 h 输入, 各自更新独立 (同一突触不拟合双目标 → 无信号冲突).
        zh2 = torch.cat([z4_lm[:, :-2], net._bind_vec[:, :-2], net._mem_out[:, :-2]], dim=-1)
        zh2 = _rms(zh2)  # 第 80 轮: 同 zh 整体 RMS 前置 (见上)
        h2 = zh2 @ W1_a
        h2 = h2 / (1.0 + h2.abs())  # 分流抑制 (与 h 同款)
        h2 = _rms(h2)  # 与 h 同款前置 RMS: 4112 维点积后值域 ±37, 直接激活进入放大区 → fp16 溢出
        h2 = h2 * (1.0 - 0.5 * h2.pow(2))
        logits_t2 = (h2 @ net.W_lm_2 + net.bias_lm) * inv_h  # [N,S-2,256] (输出缩放)
        target_lm = F.one_hot(ctx.byte_ids[:, 1:], num_classes=256).to(torch.float16)
        target_lm2 = F.one_hot(ctx.byte_ids[:, 2:], num_classes=256).to(torch.float16)
        # 赫布版 softmax 误差: eps = target - softmax(logits) (概率尺度 0-1).
        # 原始 target - logits 的负信号被 logits 幅度主导 (熵 5.5 时 logit~0 但非目标位
        # 255 项累积淹没目标位); softmax 后目标位概率 1/256, 误差信号与概率匹配.
        # 全 fp16: logits 已归一化到 [-60,60] 有界, exp 输入有界无溢出
        probs_lm = torch.softmax(logits_lm, dim=-1)  # [N,S,256] fp16
        probs_t2 = torch.softmax(logits_t2, dim=-1)
        eps_lm = (target_lm - probs_lm[:, :-1]).detach()  # [N,S-1,256]
        eps_t2 = (target_lm2 - probs_t2).detach()  # [N,S-2,256] 专供 W_lm_2 更新
        # 任务 1: 多步差分目标 — 差分误差 = (target_{t+2}-target_{t+1}) - (probs_{t+2}-probs_{t+1}),
        # 两步内字节变化的方向/幅度必须匹配 (structure 上"下一步变什么").
        # W_lm 吃 diff2 (S-1 对齐) + t+1 误差; W_lm_2 吃 diff2 + t+2 误差 (同权重,
        # 不引入 BP — 差分目标只是额外的赫布外积误差通道)
        probs_l1 = probs_lm[:, :-1]  # [N,S-1,256] = t+1 概率
        probs_l2 = probs_t2  # [N,S-2,256] = t+2 概率
        target_l1 = target_lm  # [N,S-1,256] = t+1 目标
        target_l2 = target_lm2  # [N,S-2,256] = t+2 目标
        diff2 = (target_l2 - target_l1[:, :-1]) - (probs_l2 - probs_l1[:, :-1])  # [N,S-2,256]
        diff2 = torch.cat([diff2, torch.zeros(N, 1, 256, dtype=diff2.dtype, device=dev)], dim=1)  # S-1 对齐
        if ctx.closed_loop:
            lm_mask = ctx.learn_mask.unsqueeze(0).unsqueeze(-1)
            eps_lm = eps_lm * lm_mask
            eps_t2 = eps_t2 * ctx.learn_mask[1:].unsqueeze(0).unsqueeze(-1)
            diff2 = diff2 * lm_mask
        # 0.2 权重: diff2 能量占比 ~22% (0.5 时 41%, W_lm 更新方向被差分信号主宰,
        # 单步目标被稀释 → 熵慢降、命中率冻结). 差分目标保留为辅助结构信号
        eps_total = (eps_lm + 0.2 * diff2).detach()  # W_lm: t+1 误差 + 差分误差 (S-1 对齐)
        eps_t2_total = (eps_t2 + 0.2 * diff2[:, :-1]).detach()  # W_lm_2: t+2 误差 + 差分误差 (S-2)

        # 第 110 轮 D2 (语言带自校准), 110c 带宽自校准: 感知相位记录真实语言
        # 的 ε 中心与弥散 — 纯统计零学习, 供 echo 相位带状 R 校准 (action.py).
        # 中心 = EMA; 弥散 = 窗间 MAD 的 EMA — 带宽不再是设计者常数, 是
        # "她听到的语言天然有多散". 口径与 echo 侧 wlm_err=1−p_gen 对齐.
        # echo 相位冻结 (echo_world_frozen) 不更新 — 生成流不配定义语言带.
        if not ctx.echo_world_frozen:
            eps_lang = 1.0 - (probs_lm[:, :-1] * target_lm).sum(dim=-1).mean()
            _lang_ema = getattr(net, "_lang_eps_ema", None)
            if _lang_ema is None:
                net._lang_eps_ema = eps_lang.detach().clone()
                net._lang_eps_mad = torch.zeros_like(eps_lang.detach())
            else:
                net._lang_eps_ema.mul_(0.995).add_(0.005 * eps_lang)
                devi = (eps_lang - net._lang_eps_ema).abs()
                net._lang_eps_mad.mul_(0.95).add_(0.05 * devi)

        # 动态稳态竞争: 每步记录 batch 级 W_lm 熵 (全 fp16, 零精度依赖:
        # 0·log(0)≡0 信息论定义, torch.where 屏蔽零概率项 — 不用 epsilon
        # 保护常数, fp16 下 1e-9 舍入为 0 → log(0)=-inf → 熵 NaN → 全链崩,
        # 第 76 轮 fp16 整改后 21 步实测; 大脑精度下不可能事件贡献为 0)
        # 连续负反馈: 20 步窗口最小二乘斜率 (线性拟合滤噪, 零超参),
        # scale = 2/(1+exp(-slope20/σ)): 熵降 (slope20<0) → scale→0 表示层放慢
        # 保护成果; 熵停滞/上升 → scale→2 表示层放大强迫重组. 有界无 clamp
        # slope20 = 20 步熵总变化 (nats), σ = 窗口熵波动 (nats), 比值无量纲
        if net.cfg.adaptive_traction:
            log_p = torch.where(probs_lm > 0, torch.log(probs_lm), torch.zeros_like(probs_lm))
            ent = -(probs_lm * log_p).sum(dim=-1).mean()
            net._ent_buf[net._ent_i % 20].copy_(ent.detach())
            net._ent_i += 1
            if net._ent_i >= 20:
                idx = (net._ent_i - 19 + torch.arange(20, device=dev)) % 20
                w = net._ent_buf[idx]  # 按时间正序重排
                slope20 = (net._t_center * w).sum() / net._t_denom * 20.0  # 20 步总变化
                sigma = w.std() + 1e-4
                net._traction_scale.copy_(2.0 / (1.0 + torch.exp(-slope20 / sigma)))
        # Q1 显著性反馈 + 任务 3 误差剧烈度缩放: 回传误差 × 时间突变范数
        # |z4[t]-z4[t-1]| × 预测误差能量 — 状态剧变且预测失误的时刻, 自上而下
        # 大幅改写表示层; 平稳且预测准的时刻回传弱 (保护已学结构).
        # 投影用全量 W_lm (5 段含绑定) 再取 z4 维: 预测误差经所有输入段权重汇聚到
        # z4 神经元 — 不受池门控排挤影响 (若只投影 z4 段, 池权重增长会压制 z4 段
        # → 表示层收不到预测误差 → 漂移 NaN, 9000 步崩盘根因)
        if getattr(net, "_q1_enabled", True):
            dz4_sig = (z4[:, 1:] - z4[:, :-1]).norm(dim=-1, keepdim=True)  # [N,S-1,1]
            dz4_sig = dz4_sig / (dz4_sig.max() + 1e-3)
            # soft 饱和增益: gain = x/(0.5·mean+0.5·x) ∈ (0,2), 有界无 clamp;
            # 误差 x=均值 → 1, x≫均值 → 2 (大幅改写), x≪均值 → 0 (保护)
            err_mag = eps_total.norm(dim=-1, keepdim=True)  # [N,S-1,1]
            err_ref = err_mag.mean() + 1e-3
            gain = err_mag / (0.5 * err_ref + 0.5 * err_mag)
            # 混合层转置投影: e → W_lm.T → W1.T → z4 段. 投影后每位置范数
            # ~√256×行范, RMS 归一化防 W_04 更新爆 (step 7 NaN 根因)
            eps_lm_proj = (eps_total @ net.W_lm.T @ W1_a.T)[:, :, :a4] * dz4_sig * gain  # [N,S-1,a4]
            eps_lm_proj = _rms(eps_lm_proj)
        else:
            eps_lm_proj = (eps_total @ net.W_lm.T @ W1_a.T)[:, :, :a4]  # 均匀回传
            eps_lm_proj = _rms(eps_lm_proj)
        eps_lm_pad = torch.cat(
            [eps_lm_proj, torch.zeros(N, 1, a4, dtype=eps_lm_proj.dtype, device=dev)], dim=1
        )

        from .engine import LmSignal

        return LmSignal(
            eps_total=eps_total,
            eps_t2_total=eps_t2_total,
            eps_lm=eps_lm,
            eps_lm_proj=eps_lm_proj,
            eps_lm_pad=eps_lm_pad,
            logits_lm=logits_lm,
            logits_t2=logits_t2,
            probs_lm=probs_lm,
            h=h,
            h2=h2,
            h_deriv=h_deriv,
            zh=zh,
        )

    def _update_lm_head(self, ctx, sh):
        """LM 头自监督赫布更新 (原 L1013-1155): W_lm/W_lm_2/W1/bias_lm. 返回 d_t."""
        net = self.net
        if ctx.free_run or ctx.echo_world_frozen:
            return torch.ones(ctx.N, ctx.S - 1, 1, dtype=torch.float16, device=ctx.dev)
        dev = ctx.dev
        d_h = net.d_h
        inv_h = 1.0 / math.sqrt(d_h)
        byte_ids = ctx.byte_ids
        lm = sh.lm

        # ── 突触前增益控制: error 先 RMS 缩放到单位能量再外积 — 更新幅度完全由
        # 内部误差能量自适应决定, 不依赖外部 eta 缩放 (替代 W_04 解码死锁)
        # 指数遗忘 (0.999/步, 纯乘法): 单位能量外积在目标附近振荡 (无阻尼 LMS),
        # 遗忘项提供阻尼让权重收敛而非翻转
        err_norm = lm.eps_total.norm(dim=-1, keepdim=True) * 1.01
        alive = (err_norm > 1e-8).to(lm.eps_total.dtype)
        denom = torch.where(alive > 0, err_norm, torch.ones_like(err_norm))
        err_scaled = lm.eps_total * alive / denom  # 单位能量, 方向保留

        # W_lm 专属 BCM 滑阈 (防输出过冲): theta = EMA(logits²) (快 0.99 响应),
        # phi_wlm = logits_n·(logits_n - theta) — W_lm 开始输出高频极值时 theta 快速
        # 升高 → logits_n-theta<0 → phi 变负 → dW 修正反向 → 抑制过冲.
        # 纯机制剪刀, 线性, 零 BP; 0.1 系数镜像 W_diff BCM (同模式)
        # logits 先 RMS 归一化再进 BCM: 原始 logits ~52, 平方超 fp16 上限 (W_diff 同款)
        logits_n = _rms(lm.logits_lm.detach())
        th_wlm = net._theta_wlm
        th_wlm.mul_(0.01).add_(0.99 * (logits_n * logits_n).mean(dim=(0, 1)))
        phi_wlm = logits_n * (logits_n - th_wlm)
        phi_wlm = _rms(phi_wlm)
        # ── 结构对比度惩罚 (第 58 轮, 训练"区分力"非"生成力") ──
        # 核心: 强制 z4/h 空间拉大不同字符的距离. 正确列 logits 应显著高于
        # 错误列. 实现: 对误差做几何加权 — 目标位 (one-hot) 误差权重 ×1,
        # 非目标位按"与目标列的区分度"加权: 区分度低 (logits 接近目标) 的
        # 错误列被放大惩罚 (空间排斥), 区分度高 (已被压远) 的列权重衰减.
        # 纯数学空间排斥, 不改变 W_lm 更新公式结构 (仍是 h^T @ err 外积)
        target_oh = F.one_hot(byte_ids[:, 1:], num_classes=256).to(torch.float16)  # [N,S-1,256]
        logits_d = (lm.h[:, :-1] @ net.W_lm) * inv_h  # 未去偏 logits (对比度基准)
        # 对比度: 目标列 logits vs 其他列 — 目标列 logits 高则区分好
        tgt_logits = (logits_d * target_oh).sum(dim=-1, keepdim=True)  # [N,S-1,1] 目标列值
        # 惩罚权重: 非目标列中 logits 接近目标的列 (区分差) 放大, 远离的衰减
        contrast = (logits_d - tgt_logits).abs()  # 与目标列的距离
        contrast_w = 1.0 / (1.0 + contrast * 0.1)  # 距离近 → 权重大 (排斥), 距离远 → 小
        contrast_w = contrast_w * (1.0 - target_oh) + target_oh  # 目标位权重保持 1
        if net.cfg.lm_no_contrast:
            err_contrast = err_scaled * (1.0 - target_oh) + target_oh * err_scaled
        else:
            err_contrast = err_scaled * contrast_w  # 对比度加权误差 (空间排斥)
        # ── 快慢散度学习窗口 (第 70 轮): 逐帧新奇度 N[t] = ‖Z_fast[t]-Z_slow[t]‖²
        # (前向已算 [N,S]). τ = EMA(N) 自适应 (内部参照物, 零外部统计).
        # 生成头不承担探索压力: 极性翻转 (LTD 反向重写) 会让 W_lm 因一次异常
        # 生成被重解释 (第 69 轮实测: 熵 0.078→4.57, 出口被推入错误吸引子).
        # 改为学习窗口缩放: η = sigmoid(N - τ) ∈ (0,1) — 只调"允许多少变化":
        # 低新奇度 (死循环, N → 0 < τ) → η → 0 关闭输出学习窗口 (保护已形成
        # 的输出映射); 正常推进 (N > τ) → η → 1 正常学习. 不翻转方向.
        if hasattr(net, "_novelty"):
            nov = net._novelty  # [N,S] 逐帧新奇度
            tau = net._theta_novelty
            tau.mul_(0.99).add_(0.01 * nov.mean())
            d_t = torch.sigmoid((nov - tau) * 500.0).unsqueeze(-1)  # [N,S,1] η ∈ (0,1)
            d_t = d_t[:, :-1]  # 对齐 S-1 (t+1 目标)
        else:
            d_t = torch.ones(ctx.N, ctx.S - 1, 1, dtype=torch.float16, device=dev)
        # 第 62 轮周期惩罚已按最终裁定移除 (训练期干预误伤 nn 词干, 见交接文档).
        # 第 70 轮: 学习窗口 η (sigmoid 有界, 零极性翻转 — 只调变化量, 不重写方向)
        lm_update_mask = ctx.learn_mask.to(torch.float16).unsqueeze(0).unsqueeze(-1)  # [1,S-1,1]
        bcm_term = torch.zeros_like(phi_wlm[:, :-1]) if net.cfg.lm_no_bcm else 0.1 * phi_wlm[:, :-1]
        dW_lm = (
            _rms(lm.h[:, :-1]).transpose(-2, -1) @ ((err_contrast - bcm_term) * lm_update_mask * d_t)
        ).mean(dim=0) * math.sqrt(d_h)  # [d_h,256] (补偿输出缩放)
        # 单步更新幅度上界 (W_04 同款幅度-方向解耦): 防极端 batch 单步爆
        dW_lm_n = dW_lm.norm() + 1e-8
        dW_lm = dW_lm / dW_lm_n
        # 第 102 轮修正 (chat101 15k 步失败根因 3): 信任域 — 单位向量 ×
        # eta_lm=1.0 = 每步 100% 相对扰动 (W_lm 行范数 ~1), 从零初始化时
        # 信号被噪声淹没 → W_lm 只学到频率先验 (chat101c 末态 top-5
        # 位置无关的实证). 单步相对扰动收敛到 1%: 信号可积累, 噪声被
        # 遗忘项 (×0.999) 平均掉. 与 W_04 幅度-方向解耦同族 (最大单步
        # 幅度 = eta·‖W‖·0.01, 结构化缩放非 clamp)
        dW_lm = dW_lm + _elig_accum(net, "W_lm", dW_lm) * getattr(
            net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16)
        )
        net.W_lm.data += dW_lm * (ctx.eta_lm * net.W_lm.norm() * LM_TRUST_REGION)
        # bias 硬复位: 范数锁定 + 更新降幅 (第 102 轮) — 旧 target=100 →
        # bias_std 6.25, 经 inv_h 后仍 6× 主导 h 信号 → 中心化归一化后
        # bias 列排序存活 → 输出钉死频率先验 (位置无关 top-5 实证).
        # target=10 → bias_std 0.625 (与 h 信号同量级); 更新降 20× →
        # 频率先验慢速建立, 上下文信号赢得竞争. 去均值只学相对偏置.
        # 误差用原始概率误差 (不经单位能量归一化): 混合层下去偏项 ±45 让
        # softmax 尖锐, 单位能量归一化把稀疏误差放大 10 倍 (bias_d 0.03→0.58,
        # 单步增量 5.8 → 10 步爆, 实测 step 7)
        bias_err = lm.eps_lm  # [N,S-1,256] 原始 target - probs, 无归一化
        bias_d = bias_err.mean(dim=(0, 1))
        net.bias_lm.data += (bias_d - bias_d.mean()) * (ctx.eta_lm / 20.0)
        bn = net.bias_lm.norm()
        target_norm = 10.0
        if bn > target_norm:
            net.bias_lm.data.mul_(target_norm / bn)
        # 第 104 轮: W_lm 豁免 soft_norm (与第 97 轮 W_act 豁免同构) —
        # soft_norm 把行范数钉在 0.8-1.2, 而闭式解行范数 0.47-0.70
        # (类间幅度差异 = 表达载体), 单步被抹平 85%. 更新已有信任域
        # 有界 (2.5% ||W||), 无范数失控机制. 保留整体等比帽 10 防
        # fp16 溢出 (同 W_act 100轮修复: 逐列帽会钉死列差, 整体等比
        # 保留比例).
        rn_lm = net.W_lm.data.norm(dim=1)
        mx_lm = rn_lm.max()
        if mx_lm > 10.0:
            net.W_lm.data.mul_((10.0 / (mx_lm + 1e-6)).to(torch.float16))

        # ── W1 混合层更新 (第 57 轮核心: 转置误差传播, 纯赫布零 BP) ──
        # e_h = e @ W_lm.T · (1 - 1.5·h²): 读出误差经 W_lm 转置投影回混合空间,
        # 乘激活导数 (多项式激活 f=h-0.5h³ 的导数 1-1.5h²) — 告诉 W1 如何把
        # 高频 e 的权重通过组合打散分配到 n/u 的特征上. 真实神经网络折叠,
        # 非几何缩放. lr1 = lr2 = eta_lm (自然竞争)
        e_h = (err_scaled @ net.W_lm.T) * lm.h_deriv[:, :-1]  # [N,S-1,d_h] (err_scaled 已是 S-1)
        lm.e_h = e_h
        dW1 = (_rms(lm.zh[:, :-1]).transpose(-2, -1) @ _rms(e_h)).mean(dim=0) * math.sqrt(d_h)
        dW1_n = dW1.norm() + 1e-8
        dW1 = dW1 / dW1_n
        # 第 102 轮: 同 W_lm 信任域 (见上) — 从零初始化时单位向量全量更新
        # 使 W1 每步整体重排, 学不到结构
        if not net.cfg.lm_freeze_w1:
            dW1 = dW1 + _elig_accum(net, "W1", dW1) * getattr(
                net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16)
            )
            net.W1.data += dW1 * (ctx.eta_lm * net.W1.norm() * LM_TRUST_REGION)
            soft_norm_preserve(net.W1.data)

        # W_lm_2 独立更新 (Q3 解耦): 吃 t+2 误差 + 差分误差 (eps_t2_total), 与 W_lm 完全独立
        # 同款机制: 零向量保护 + 单位能量 + BCM 防抖 + 指数遗忘
        err_norm2 = lm.eps_t2_total.norm(dim=-1, keepdim=True) * 1.01
        alive2 = (err_norm2 > 1e-8).to(lm.eps_t2_total.dtype)
        denom2 = torch.where(alive2 > 0, err_norm2, torch.ones_like(err_norm2))
        err_scaled2 = lm.eps_t2_total * alive2 / denom2
        logits_n2 = _rms(lm.logits_t2.detach())
        th_wlm2 = net._theta_wlm2
        th_wlm2.mul_(0.01).add_(0.99 * (logits_n2 * logits_n2).mean(dim=(0, 1)))
        phi_wlm2 = logits_n2 * (logits_n2 - th_wlm2)
        phi_wlm2 = _rms(phi_wlm2)
        dW_lm2 = (
            _rms(lm.h2).transpose(-2, -1)
            @ (
                (err_scaled2 - 0.1 * phi_wlm2)
                * ctx.learn_mask[1:].to(torch.float16).unsqueeze(0).unsqueeze(-1)
                * d_t[:, :-1]
            )
        ).mean(dim=0) * math.sqrt(d_h)
        dW_lm2_n = dW_lm2.norm() + 1e-8
        dW_lm2 = dW_lm2 / dW_lm2_n  # 单步更新幅度上界 (防突爆)
        # 第 102 轮: 同 W_lm 信任域 (见上)
        dW_lm2 = dW_lm2 + _elig_accum(net, "W_lm_2", dW_lm2) * getattr(
            net, "_survival_signal", torch.tensor(0.0, device=dev, dtype=torch.float16)
        )
        net.W_lm_2.data += dW_lm2 * (ctx.eta_lm * net.W_lm_2.norm() * LM_TRUST_REGION)
        soft_norm_preserve(net.W_lm_2.data)
        # 每步清除新奇度 (前向已更新, 防陈旧信号跨步复用)
        if hasattr(net, "_novelty"):
            del net._novelty
        return d_t

    def _update_mem_units(self, ctx, sh):
        """竞争性记忆单元群 (第 109 轮): 适应度 → 增益 → 出生/死亡.

        四相循环, 全部纯局部机制, 零超参搜索:
        1. 适应度 q_j ← EMA(cos(单元记忆, 读出误差在 zh 单元段的投影)) — 单元
           记忆与"该单元段对当前误差的解释方向"同向 → q 涨 (该时间常数有用)
        2. 增益 g_j ← clamp(g_j + η_g·q_j, 0, g_max) — 有界参数, 非梯度路径
        3. 出生: 读取误差 (快 EMA) 持续高于长程基线 (慢 EMA) × 阈值 → 从增益
           最高单元二分出子 (α×2 或 ÷2, 交替), m 继承父, W1 新行零起步
        4. 死亡: g < g_min 持续 mem_death_steps → 单元删除, W1 行收缩 (K_min=1)

        时机: 在 _update_lm_head (dW1) 之后 — 本步 zh/W1 行数已由前向布局
        决定, K 变化只影响下一步 (zh 列与 W1 行永不错位).
        """
        net = self.net
        if ctx.free_run or ctx.echo_world_frozen or sh.lm is None:
            return
        cfg = net.cfg
        a4 = ctx.a4
        N = ctx.N
        K = net._mem_m.shape[0]
        m_out = net._mem_out
        if m_out is None or m_out.shape[2] != K * a4:
            return
        lm = sh.lm
        if lm.e_h is None:
            return

        # 1) 适应度: 误差投影回 zh 空间, 取单元段与单元记忆的余弦.
        # e_zh 是 [N,S-1,lm_in]: 前 a4 = z4 段, 次 bind_sz = bind 段, 尾段 = 单元块
        e_zh = _rms(lm.e_h) @ net.W1.T  # [N,S-1,lm_in]
        e_cells = e_zh[:, :, a4 + net.bind_slot_dim :].reshape(N, lm.e_h.shape[1], K, a4).detach()
        mc = m_out[:, :-1].reshape(N, lm.e_h.shape[1], K, a4).detach()
        cos = (_l2_norm(mc) * _l2_norm(e_cells)).sum(dim=-1).mean(dim=(0, 1))  # [K] ∈ [-1,1]
        net._mem_q.mul_(0.99).add_(0.01 * cos)

        # 2) 增益 (有界参数 clamp, 非梯度路径): q>0 上推, 乘性衰减让零贡献单元
        # g→0 (纯加法 g 恒 1.0 不降, 死亡永不可达 — 第 109 轮实测修正)
        net._mem_g.mul_(1.0 - cfg.mem_g_decay).clamp_(min=0.0)
        net._mem_g.add_(cfg.mem_eta_g * net._mem_q)
        net._mem_g.clamp_(max=cfg.mem_g_max)

        # 3) 死亡: g 低 → 计数, 持续超限 → 删除 (K_min=1 保底: 索引 0 永不判死).
        # 避免 CPU-GPU 同步: 条件展平为掩码, 全部宽张量操作
        low = net._mem_g < cfg.mem_g_min
        net._mem_death_cnt = torch.where(low, net._mem_death_cnt + 1, torch.zeros_like(net._mem_death_cnt))
        dead_mask = (net._mem_death_cnt >= cfg.mem_death_steps) & low
        dead_mask[0] = False  # K_min=1 保底: 索引 0 是永存单元
        if dead_mask[1:].any():
            net._mem_death_cnt = net._mem_death_cnt.where(~dead_mask, torch.zeros_like(net._mem_death_cnt))
            self._mem_resize(~dead_mask)

        # 4) 出生: 误差快 EMA > 长程慢 EMA × 阈值, 冷却期满, K < 上限.
        # 出生源 = 增益最高单元 (argmax g; 附注: 记忆范数不参与竞争)
        err_rms = lm.eps_total.square().mean().sqrt()
        net._mem_err_ema.mul_(0.99).add_(0.01 * err_rms)
        net._mem_err_long.mul_(0.999).add_(0.001 * err_rms)
        net._mem_birth_cd += 1
        if (
            net._mem_birth_cd >= cfg.mem_birth_cooldown
            and cfg.mem_k_max > K
            and net._mem_err_ema > net._mem_err_long * cfg.mem_birth_thresh
        ):
            parent = int(torch.argmax(net._mem_g))  # 每步至多一次出生, 同步可接受
            self._mem_birth(parent)
            net._mem_birth_cd = 0

    def _mem_resize(self, keep: torch.Tensor):
        """K 变化: 单元缓冲 + W1 行 + W1_elig 迹同步重注册 (修剪 _shrink_columns 先例)."""
        net = self.net
        a4 = net._mem_m.shape[1]
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
        # W1 行同步 (单元块位于 [a4+32 : a4+32+K·a4])
        head = a4 + net.bind_slot_dim
        w1 = net.W1.data
        new_w1 = torch.cat(
            [w1[:head], w1[head:].reshape(-1, a4, d_h)[keep].reshape(keep.shape[0] * a4, d_h)],
            dim=0,
        ).contiguous()
        net.W1 = nn.Parameter(new_w1)
        # W1_elig 迹同形同步 (出生/死亡后迹必须与 W1 行对齐)
        elig = getattr(net, "W1_elig", None)
        if elig is not None:
            ne = torch.cat(
                [elig.data[:head], elig.data[head:].reshape(-1, a4, d_h)[keep].reshape(keep.shape[0] * a4, d_h)],
                dim=0,
            ).contiguous()
            net.register_buffer("W1_elig", ne)
        net._lm_in = head + keep.shape[0] * a4

    def _mem_birth(self, parent: int):
        """从父单元二分出生 (α×2 或 ÷2, _mem_alt 交替, 越界取另一侧)."""
        net = self.net
        cfg = net.cfg
        K = net._mem_m.shape[0]
        a4 = net._mem_m.shape[1]
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
        m_child = net._mem_m.data[parent : parent + 1].clone()
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
        head = a4 + net.bind_slot_dim
        w1 = net.W1.data
        new_w1 = torch.cat(
            [w1[:head], w1[head:], torch.zeros(a4, d_h, dtype=torch.float16, device=w1.device)],
            dim=0,
        ).contiguous()
        net.W1 = nn.Parameter(new_w1)
        elig = getattr(net, "W1_elig", None)
        if elig is not None:
            ne = torch.cat(
                [elig.data[:head], elig.data[head:], torch.zeros(a4, d_h, dtype=torch.float16, device=elig.device)],
                dim=0,
            ).contiguous()
            net.register_buffer("W1_elig", ne)
        net._lm_in = head + (K + 1) * a4
