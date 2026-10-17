"""
ForwardEngine
前馈/推理/生成/稀疏绑定.

感知 L0 → L4 → L2 → L3 → L5 → L6; 时序经时间核 W_t 递归;
生成经 W_diff 预测差分 + W_lm 解码; 绑定 z4 → W_bind → 概念槽.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from crates.model.model_cyrene import HybridPCNet

from crates.kernels import flush_counts, l2_norm, rms_norm, saturate
from crates.utils.codec import encode_text

from .forward_graph import ForwardGraphs
from .gen_graph import ContinuationGraphs


class ForwardEngine:
    """前馈引擎，持 net 引用, 状态经 net._z*/net.active_size 传递"""

    def __init__(self, net: HybridPCNet):
        self.net = net
        self._graphs = ForwardGraphs(self)
        self._gen_graphs = ContinuationGraphs(self)

    def forward(self, byte_ids: torch.Tensor) -> dict:
        """推理前馈，返回未来预测偏差.  ACh 关闭, 确定性"""
        return self._predict(byte_ids, store_state=True, is_inference=True)

    def generate(
        self, prompt: str, n_tokens: int = 40, temperature: float = 0.7, dev: torch.device | None = None
    ) -> bytes:
        """
        行动，L4 状态 + 预测差分 → W_lm 解码生成字节
        """
        if dev is None:
            dev = next(self.net.parameters()).device
        ids = torch.tensor([list(encode_text(prompt))], dtype=torch.long, device=dev)
        out = self.continuation(ids, n_tokens, temperature=temperature, rep_backstop=True)
        return bytes(out[0].tolist())

    def continuation(
        self,
        byte_ids: torch.Tensor,
        n_gen: int,
        temperature: float | torch.Tensor | None = None,
        rep_backstop: bool = False,
    ) -> torch.Tensor:
        """
        自回归续写，基于 [N,S] 前缀批量并行生成, 返回 [N, S+n_gen].

        逐步张量路径由 ContinuationGraphs 声明为静态图 (一步一张图);
        步与步之间的依赖来自取样字节回写, 属自回归语义本身, 环留宿主.
        temperature: float|Tensor|None — Tensor 时 GPU 直除 (零 .item()); None 或 float<=0 = 贪心.
        rep_backstop: 屏蔽近期 2..24 字节周期末字节的 logits.
        UTF-8 续字节超长阻断: 连续 3 个续字节后屏蔽全部续字节.
        """
        net = self.net
        n_batch, s_len = byte_ids.shape
        if not hasattr(net, "_block_stats"):
            net._block_stats = {"rep": 0, "gen": 0}
        stats = net._block_stats
        commit = net._bind_commit
        if commit:
            net._commit_changed = 0  # python int 累积 (生成路径, 非训练热路径)
            net._commit_midchar = 0

        graphs = self._gen_graphs
        env: dict | None = None
        for k in range(n_gen):
            _, env = graphs.step(
                byte_ids,
                n_gen=n_gen,
                cur_len=s_len + k,
                seed=k == 0,
                last_step=k == n_gen - 1,
                temperature=temperature,
                rep_backstop=rep_backstop,
            )
        if env is None:
            _, env = graphs.step(
                byte_ids,
                n_gen=0,
                cur_len=s_len,
                seed=True,
                last_step=True,
                temperature=temperature,
                rep_backstop=rep_backstop,
            )
        stats["gen"] += n_batch * n_gen
        if rep_backstop or commit:
            rep_hits, changed, midchar = flush_counts(net._gen_counts)
            stats["rep"] += rep_hits
            if commit:
                net._commit_changed += changed
                net._commit_midchar += midchar
        return env["out"]

    def _recurrent(
        self,
        mu: torch.Tensor,
        dev: torch.device,
        W_t: torch.Tensor,
        sweeps: int = 1,
        stp: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        stp_tag: str | None = None,
    ) -> torch.Tensor:
        """
        时间核递归: z[t] 依赖 z[t-1] (经 W_t), 纯 matmul.

        只归一化递归项, 保留 mu 携带的输入语义方向.
        stp (r, tau, u): 逐帧递归项 ×r[t] (资源耗尽 → 递归减弱, 自发间歇), 仅自由运行启用.
        stp_tag: 层名, 用于末资源写回 net._stp_r_end.
        """
        N, S, d = mu.shape
        z = mu
        inv_d = 1.0 / math.sqrt(d)

        if stp is None:
            for _ in range(sweeps):
                shift = torch.cat([self.net._padmax[:, :, :d], z[:, :-1]], dim=1)
                shift_n = rms_norm(shift)
                rec = (shift_n @ W_t.T) * inv_d
                rec = rms_norm(rec) * inv_d  # 单独归一化保方向, 幅度有界防累积
                z = z + rec
            return z

        # STP 路径: 逐帧递归项 × 当前资源; 幅度由 W_t 行范数承载 (防发散靠能量而非范数锁)
        # 递推用 z_out (含 STP 调制的输出), 首帧 = mu 首帧 (无递归)
        r, tau, u = stp
        r = r.unsqueeze(0)  # [1,d]
        inv_tau = (1.0 / tau).unsqueeze(0)  # [1,d]
        u_b = u.unsqueeze(0)  # [1,d]
        z_out = z[:, :1].clone()  # tensor-guard: rare (自由运行 STP 路径, 探针专用)
        for t in range(1, S):
            shift = z_out[:, t - 1 : t]  # [N,1,d] 用已调制的输出递推
            shift_n = rms_norm(shift)
            rec = (shift_n @ W_t.T) * inv_d  # 幅度 = W_t 行范数 (增益自由度)
            z_t = z[:, t : t + 1] + rec * r  # 递归项 × 可用资源
            z_out = torch.cat([z_out, z_t], dim=1)

            # 资源更新: r ← r + (1−r)/τ − U_eff·r·|z|; U_eff = U·act/(act+ema) 自参照, 高活动耗尽加速
            act = z_t.abs().mean(dim=1)
            ema = getattr(self.net, f"_stp_active_ema_{stp_tag}")[: r.shape[1]]  # 修剪后对齐活性尺寸
            ema.mul_(0.99).add_(0.01 * act.squeeze(0))

            # 局部 U 自适应: 活动 vs 自身慢基线 → 释放概率双向调节 (纯局部负反馈)
            if self.net.cfg.stp_u_adapt:
                rel = (act - ema.unsqueeze(0)) / (ema.unsqueeze(0) + 1e-6)
                u_b.add_(self.net.cfg.stp_u_adapt_rate * rel * u_b)
                u_b.clamp_(self.net.cfg.stp_u_min, self.net.cfg.stp_u_max)  # tensor-guard: bound (STP 动力学设计界)

            u_eff = u_b * act / (act + ema + 1e-6)
            r = r + (1.0 - r) * inv_tau - u_eff * r * act
            r = r.clamp(0.01, 1.0)  # tensor-guard: bound (STP 资源界, 有界防发散)
        if stp_tag is not None:
            self.net._stp_r_end[stp_tag] = r.squeeze(0).clone()  # tensor-guard: rare (自由运行窗末写回)
        return z_out

    def _predict(self, byte_ids: torch.Tensor, store_state: bool = True, is_inference: bool = False) -> dict:
        """
        核心前馈，感知 (L0→L6) + Foldiak 去相关 + 去中心化 + 增量预测.

        算子序列、名字接线与写回契约由 forward_graph.declare 静态声明, 本方法只取图、喂料、读回;
        无位置编码, 时序全靠 W_t 递归; z5 去均值喂下游, z5_raw 喂 W_diff; ACh 噪声注入 L3.
        """
        _, env = self._graphs.step(byte_ids, store_state=store_state, is_inference=is_inference)
        return {
            "mu_diff": env["mu_diff"],
            "diff_err": env["diff_err"],
            "free_energy": env["diff_err"],
        }

    def _stp_writeback(
        self, stp_end: dict, z4: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, z6: torch.Tensor
    ) -> None:
        """
        STP 资源写回(自由运行窗末), 逐层顺序与历史逐字一致.

        写哪几层由跨窗演化的 _stp_r_end 键集承载, 该键集不在变体键闭集内, 故整段作单个算子.
        """
        for layer, z in (("l4", z4), ("l2", z2), ("l3", z3), ("l6", z6)):
            if layer in stp_end:
                self._stp_update(layer, stp_end[layer], z)

    def _stp_update(self, layer: str, r_end: torch.Tensor, z: torch.Tensor) -> None:
        """STP 资源写回 (窗末): r 末帧写回 buffer (跨窗延续)."""
        r_buf = getattr(self.net, f"_stp_r_{layer}")
        r_buf[: r_end.shape[0]].copy_(r_end)  # 修剪后 active 收缩: 只写头部切片

    def _bind(
        self, z4: torch.Tensor, fr_vf: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        竞争性概念绑定: z4 → W_bind → 槽位 → 连续值稀疏激活向量.

        relu(raw - θ) 阈值竞争 + 分流抑制, 无 sum=1 约束, 输出为连续值稀疏向量.
        θ 复用 _theta_bind (跟踪槽能量). 返回 (raw, z_bind, intent_pot), 属性发布交由调用图写回.
        """
        net = self.net
        dim_4 = net.active_size["l4"]
        z4_n = l2_norm(z4)
        raw = z4_n @ net.W_bind[:dim_4]  # [N,S,K] 自下而上投影

        # 递归绑定 (恒开): 自由运行快节律注入 → z_bind 预激活 + STP 槽资源
        if fr_vf is not None:
            raw = raw + net.cfg.osc_amp_f * fr_vf
        # STP 槽资源: 逐帧输出 ×r (资源耗尽 → 槽间歇切换)
        stp_bind = None
        if fr_vf is not None:
            stp_bind = (net._stp_r_bind, net._stp_tau_bind, net._stp_u_bind)
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
            if r_bind is not None and u_bind_b is not None and inv_tau_b is not None:
                act = z_bind.abs().mean(dim=1)  # [1,K] (同递归层修正)
                ema_b = net._stp_active_ema_bind[: r_bind.shape[1]]
                ema_b.mul_(0.99).add_(0.01 * act.squeeze(0))

                # 槽位局部 U 自适应 (同递归层, 纯局部负反馈)
                if self.net.cfg.stp_u_adapt:
                    rel_b = (act - ema_b.unsqueeze(0)) / (ema_b.unsqueeze(0) + 1e-6)
                    u_bind_b.add_(self.net.cfg.stp_u_adapt_rate * rel_b * u_bind_b)
                    u_bind_b.clamp_(self.net.cfg.stp_u_min, self.net.cfg.stp_u_max)  # tensor-guard: bound (STP 动力学设计界)

                u_eff_b = u_bind_b * act / (act + ema_b + 1e-6)  # 自参照耗尽
                r_bind = r_bind + (1.0 - r_bind) * inv_tau_b - u_eff_b * r_bind * act
                r_bind = r_bind.clamp(0.01, 1.0)  # tensor-guard: bound (STP 资源界, 有界防发散)
        if stp_bind and r_bind is not None:
            net._stp_r_end["bind"] = r_bind.squeeze(0).clone()  # tensor-guard: rare (自由运行窗末写回)
        z_bind = torch.cat(z_binds, dim=1)
        # 分流抑制: 上界 1.0, 范数保留为自由度
        th_b = net._theta_bind
        th_b.mul_(0.98).add_(0.02 * (z_bind * z_bind).mean(dim=(0, 1)))
        z_bind = saturate(z_bind)
        if fr_vf is not None:
            if "bind" in net._stp_r_end:
                self._stp_update("bind", net._stp_r_end["bind"], z_bind)  # 槽 STP 资源写回
        # 动作电位: 概念槽 → W_act → 字节脉冲潜能
        return raw, z_bind, z_bind @ net.W_act  # [N,S,ALPHABET] (soft_norm 列归一有界)

    def _bind_sparse(self, pre_act: torch.Tensor, k: int = 4) -> torch.Tensor:
        """槽稀疏竞争: 每位置只保留 top-k 强激活槽, 其余清零.

        保留连续值幅度 (relu 后值, 非二值), 范数仍是自由度; 阈值 θ 先弱激活过滤再 top-k.
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
