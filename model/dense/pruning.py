"""PruningEngine

密集 PPA 动态神经元修剪 (慢速循环).

拓扑重塑: 发育期内不剪 → 死缓二级判决 → 相对排名淘汰.
修剪同步: L4 神经元行 perm 同步重排 W_lm/W_diff/W_state_pred/_dw_buf/_theta_w,
L3 修剪同步 W_35 列与掩码/gain 矩阵 (系统性错位是历史 NaN 根因).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from .network import DensePCNet


class PruningEngine:
    """修剪引擎: 持 net 引用, 操作 net.active_size/_death_row/权重张量."""

    def __init__(self, net: DensePCNet):
        self.net = net

    # ── 辅助函数 (只加边界, 不改逻辑) ──

    def _expire_flags(self, layers: list[str], dprob: int) -> dict[str, torch.Tensor | None]:
        """死缓二级判决: 过期标记 (死缓期满且未复活) → 强制淘汰."""
        net = self.net
        expired_flags: dict[str, torch.Tensor | None] = {}
        for layer in layers:
            active = net.active_size[layer]
            dr = net._death_row.get(layer)
            pc = net._probation_counter.get(layer)
            if dr is None or pc is None:
                expired_flags[layer] = None
                continue
            dr_a = dr[:active]
            pc_a = pc[:active]
            in_death = dr_a.bool()
            expired = in_death & (pc_a >= dprob)
            W = getattr(net, self._W_attr[layer])
            rn = W[:active].data.norm(dim=1)
            revived = in_death & (rn > net.cfg.death_threshold) & ~expired
            if revived.any():
                dr_a[revived] = False
                pc_a[revived] = 0
            expired_flags[layer] = expired if expired.any() else None
        return expired_flags

    def _permute_weights(
        self,
        layer: str,
        active: int,
        n_candidate: int,
        expired: torch.Tensor | None,
        perm_map: dict[str, torch.Tensor],
        score_fn=None,
    ) -> tuple[torch.Tensor | None, int]:
        """相对排名淘汰: 行范数 top-k 候选 + 过期强制, 生成 perm 并重排行向权重.

        感知修剪 (方案 B): L4 传 score_fn, 候选分数 = 表示层行范数 × W_lm 行范数 —
        只剪"对预测无贡献"的神经元 (表示层低活跃 且 W_lm 依赖弱); 其余层用纯行范数.

        Returns:
            (perm, n_alive): perm=None 表示无修剪 (n_alive=active).
        """
        net = self.net
        W = getattr(net, self._W_attr[layer])
        rn = W[:active].data.norm(dim=1)
        if score_fn is not None:
            rn = score_fn(rn, active)

        _, dead_ix = rn.topk(n_candidate, dim=-1, largest=False)
        candidate_mask = torch.zeros(active, dtype=torch.bool, device=rn.device)
        candidate_mask[dead_ix] = True

        if expired is not None:
            candidate_mask = candidate_mask & ~expired

        alive_mask = ~candidate_mask
        if expired is not None:
            alive_mask = alive_mask & ~expired

        n_alive = max(self._bounds[layer], (int(alive_mask.sum().detach().item()) // 8) * 8)
        n_alive = max(n_alive, active - n_candidate)
        n_alive = min(n_alive, active)

        if n_alive >= active:
            return None, active

        keep = torch.where(alive_mask)[0]
        probation = torch.where(candidate_mask)[0]
        if expired is not None:
            dead = torch.where(expired)[0]
        else:
            dead = torch.zeros(0, dtype=torch.long, device=rn.device)
        perm = torch.cat([keep, probation, dead])

        dr = net._death_row.get(layer)
        pc = net._probation_counter.get(layer)
        new_dr = (
            torch.zeros(active, dtype=torch.int8, device=rn.device) if dr is None else dr[:active].clone()
        )
        new_pc = (
            torch.zeros(active, dtype=torch.int16, device=rn.device) if pc is None else pc[:active].clone()
        )

        if expired is not None:
            new_dr[expired] = 0
            new_pc[expired] = 0
        new_dr[probation] = 1
        new_pc[probation] = 0

        W.data = W.data[perm].contiguous()
        W_t = getattr(net, self._t_attr[layer])
        W_t.data = W_t.data[perm][:, perm].contiguous()
        # W_t4 decorr 状态同 perm (E_t4 行列 = L4 神经元, 错位 → decorr 拆错方向)
        et_attr = {"l4": "E_t4"}
        if layer in et_attr:
            Et = getattr(net, et_attr[layer])
            Et.data = Et.data[perm][:, perm].contiguous()
        b = getattr(net, self._b_attr[layer])
        b.data = b.data[perm].contiguous()

        net._death_row[layer] = new_dr[perm]
        net._probation_counter[layer] = new_pc[perm]
        perm_map[layer] = perm
        return perm, n_alive

    def _sync_l4_aux(self, perm: torch.Tensor) -> None:
        """L4 神经元行重排 → 同步重排所有 L4 行映射的权重,
        否则预测误差投影与 W_04 行错位 → 6000-7000 步 NaN (修剪后首爆)."""
        net = self.net
        # W_lm/W_lm_2 行 = d_h 混合空间 (无神经元映射) → 不随 L4 perm 重排
        # W1 行 = [z4 | m2 | m8 | m32 | bind] → 前 4 段同 perm, bind 段恒等
        n_bind = net.bind_slot_dim
        net.W1.data = net.W1.data[torch.cat([perm, perm, perm, perm, torch.arange(n_bind, device=perm.device)])].contiguous()
        net.W_diff.data = net.W_diff.data[perm][:, perm].contiguous()
        net.W_state_pred.data = net.W_state_pred.data[perm][:, perm].contiguous()
        # W_pred_54 行 = L5 神经元 (随 L4 修剪? 否 — 行=L5, 列=L4): 列同 L4 perm
        # W_pred_54 [a5, a4]: 列 (L4) 同 perm
        net.W_pred_54.data = net.W_pred_54.data[:, perm].contiguous()
        # W_pred_43 [a4, a3]: 行 (L4) 同 perm
        net.W_pred_43.data = net.W_pred_43.data[perm].contiguous()
        # W_bind 行 = z4 神经元 (槽共享), 同 perm
        net.W_bind.data = net.W_bind.data[perm].contiguous()
        # _dw_buf 环形缓冲同 perm 重排 (否则 4 步缓冲与 W_diff 行错位 → 更新错乱)
        for i in range(4):
            old_buf = getattr(net, f"_dw_buf_{i}").data
            net.register_buffer(f"_dw_buf_{i}", old_buf[perm][:, perm].contiguous())
            del old_buf
        # _theta_w (W_diff BCM 滑阈) 同 perm 重排 (错位阈值 → phi_w 异常 → 漂移 NaN)
        old_thw = net._theta_w.data
        net.register_buffer("_theta_w", old_thw[perm].contiguous())
        del old_thw
        # _m_pool (多级记忆池, 3 段 × a4) 同 perm 重排 (池与 z4 神经元对齐,
        # 错位 → W_lm 输入乱)
        old_m = net._m_pool.data
        net.register_buffer("_m_pool", old_m[torch.cat([perm, perm, perm])].contiguous())
        del old_m
        # E_bind 行列同 perm 重排 (行 = L4 神经元, 对称矩阵双边同步)
        old_eb = net.E_bind.data
        net.E_bind = nn.Parameter(old_eb[perm][:, perm].contiguous())
        del old_eb
        # E_04 行列同 perm 重排 (W_04 行 = L4 神经元)
        old_e04 = net.E_04.data
        net.E_04 = nn.Parameter(old_e04[perm][:, perm].contiguous())
        del old_e04

    def _shrink_columns(
        self, layer: str, n_alive: int, src: str, src_n: int, perm_map: dict[str, torch.Tensor]
    ) -> None:
        """跨层列映射同步: 源层神经元重排 → 下游权重列同 perm, 再裁到活性维."""
        net = self.net
        W = getattr(net, self._W_attr[layer])
        old = W.data
        # 顺序关键: 先 perm 后裁 (perm 长度 = 修剪前 active, 裁后列数 < active 会越界)
        src_perm = perm_map.get(src)
        if src_perm is not None:
            old = old[:, src_perm]
        setattr(net, self._W_attr[layer], nn.Parameter(old[:n_alive, :src_n].contiguous()))
        del old
        # _gain_l3 列 = L2 神经元, 用 L2 自己的 perm (非 src perm — src 是 L4)
        l2_perm = perm_map.get("l2")
        if l2_perm is not None:
            old_g = net._gain_l3.data
            net.register_buffer("_gain_l3", old_g[:, l2_perm].contiguous())
            del old_g

        # W_35 输入维 = L3 活性维: L3 修剪后同步 W_35 列数 + 按 perm 重排列,
        # 否则 z3 (重排后的 L3) 喂给旧序列 → 系统性错位 → eps5_td 突变 → surprise NaN
        # (6000 步修剪后 6100 步 eta=nan 全链路爆的根因)
        if layer == "l3":
            l3_perm = perm_map.get("l3")
            old_35 = net.W_35.data
            if l3_perm is not None:
                old_35 = old_35[:, l3_perm]
            net.W_35 = nn.Parameter(old_35[:, :src_n].contiguous())
            del old_35
            # _gain_mask 列与 W_35 列同源 (L3 神经元), 同 perm 重排
            if l3_perm is not None:
                old_gm = net._gain_mask.data
                net.register_buffer("_gain_mask", old_gm[:, l3_perm][:, :src_n].contiguous())
                del old_gm
            # _gain_l3 行与 W_23 行同源 (L3 神经元), 同 perm 重排
            if l3_perm is not None:
                old_g = net._gain_l3.data
                net.register_buffer("_gain_l3", old_g[l3_perm][:, :src_n].contiguous())
                del old_g

        W_t = getattr(net, self._t_attr[layer])
        old_t = W_t.data
        setattr(net, self._t_attr[layer], nn.Parameter(old_t[:n_alive, :n_alive].contiguous()))
        del old_t

        b = getattr(net, self._b_attr[layer])
        old_b = b.data
        setattr(net, self._b_attr[layer], nn.Parameter(old_b[:n_alive].contiguous()))
        del old_b

        if net._death_row[layer] is not None:
            net._death_row[layer] = net._death_row[layer][:n_alive]
            net._probation_counter[layer] = net._probation_counter[layer][:n_alive]

        net.active_size[layer] = n_alive

    # ── 主入口 ──

    def _prune(self):
        """拓扑重塑: 发育期内不剪 → 死缓二级判决 → 相对排名淘汰."""
        net = self.net
        layers = ["l4", "l2", "l3", "l6"]
        self._W_attr = {"l4": "W_04", "l2": "W_42", "l3": "W_23", "l6": "W_56"}
        self._t_attr = {"l4": "W_t4", "l2": "W_t2", "l3": "W_t3", "l6": "W_t6"}
        self._b_attr = {"l4": "bias_l4", "l2": "bias_l2", "l3": "bias_l3", "l6": "bias_l6"}
        src_layer = {"l4": "l0", "l2": "l4", "l3": "l2", "l6": "l5"}

        bound = net.cfg.active_size_lower_bound
        # L4 专属屏障: 预测主空间保底 512 (L2/L3/L6 保持 128 自然竞争)
        l4_bound = net.cfg.l4_lower_bound
        frac = net.cfg.prune_fraction
        dprob = net.cfg.death_probation
        self._bounds = {"l4": l4_bound, "l2": bound, "l3": bound, "l6": bound}

        expired_flags = self._expire_flags(layers, dprob)

        # 阶段一: 行 perm 重排 (active_size 尚未更新, n_alive_map 记录新尺寸)
        n_alive_map: dict[str, int] = {}
        perm_map: dict[str, torch.Tensor] = {}
        for layer in layers:
            active = net.active_size[layer]
            # L4 专属下限: 达到屏障后停止修剪 (保留 512 神经元承载双任务)
            layer_bound = self._bounds[layer]
            if active <= layer_bound:
                n_alive_map[layer] = active
                continue
            n_candidate = max(1, int(active * frac))
            expired = expired_flags.get(layer)
            # 感知修剪 (方案 B): L4 候选分数 = 表示层行范数 × W_lm z4段行范数 —
            # 只剪对预测无贡献的神经元 (低活跃 且 W_lm 依赖弱);
            # 防止修剪破坏 W_lm 学到的映射 (6000 步断崖根因)
            score_fn = None
            if layer == "l4":
                rn_lm = net.W_lm[:active].data.norm(dim=1)
                score_fn = lambda rn, a: rn * rn_lm[:a]
            perm, n_alive = self._permute_weights(layer, active, n_candidate, expired, perm_map, score_fn)
            n_alive_map[layer] = n_alive
            if perm is None:
                continue
            # L4 神经元行重排 → 同步重排所有 L4 行映射的权重
            if layer == "l4":
                self._sync_l4_aux(perm)

        # 阶段二: 列同步 & 显存回收 (顺序关键: 先 perm 后裁; up_size 链式传递,
        # 源层裁后尺寸 → 下游 src_n, l0 输入维 = 单帧 256 或双通道 512)
        up_size = {"l0": net._in_dim}
        for layer in layers:
            n_alive = n_alive_map.get(layer, net.active_size[layer])
            src = src_layer[layer]
            src_n = up_size.get(src, net.active_size.get(src, net.cfg.d_input))
            if n_alive >= net.active_size[layer] and getattr(net, self._W_attr[layer]).shape[1] == src_n:
                up_size[layer] = net.active_size[layer]
                continue
            self._shrink_columns(layer, n_alive, src, src_n, perm_map)
            up_size[layer] = n_alive
        # W_diff 是 L4 空间方阵 (行=输出维 a4, 列=输入维 a4), 随 L4 修剪同步
        if net.active_size["l4"] < net.cfg.d_l4:
            old_fut = net.W_diff.data
            net.W_diff = nn.Parameter(old_fut[: net.active_size["l4"], : net.active_size["l4"]].contiguous())
            del old_fut
            # W_state_pred 同尺寸同步
            old_sp = net.W_state_pred.data
            net.W_state_pred = nn.Parameter(
                old_sp[: net.active_size["l4"], : net.active_size["l4"]].contiguous()
            )
            del old_sp
            # W1 行同步 (L4 活性维 ×4: z4 + 3 记忆池; bind 段 16 固定保持)
            # W_lm/W_lm_2 行 = d_h 混合空间 (无神经元映射) → 不裁剪
            n_bind = net.bind_slot_dim
            old_w1 = net.W1.data
            net.W1 = nn.Parameter(old_w1[: 4 * net.active_size["l4"] + n_bind, :].contiguous())
            del old_w1
            # W_bind 行同步 (L4 活性维, 列 = 768 槽位固定)
            old_bind = net.W_bind.data
            net.W_bind = nn.Parameter(old_bind[: net.active_size["l4"], :].contiguous())
            del old_bind
            # _dw_buf 环形缓冲同尺寸同步 (否则 copy_ 形状崩: 6000 步 L4 修剪后 1024 vs 973)
            for i in range(4):
                old_buf = getattr(net, f"_dw_buf_{i}").data
                net.register_buffer(
                    f"_dw_buf_{i}",
                    old_buf[: net.active_size["l4"], : net.active_size["l4"]].contiguous(),
                )
                del old_buf
            # _theta_w (W_diff BCM 滑阈) 对齐 W_diff 行数: perm 重排只改变顺序不缩短,
            # 列同步段必须裁到活性维, 否则与 W_diff 行错位累积 → 长期漂移 NaN
            old_thw = net._theta_w.data
            net.register_buffer("_theta_w", old_thw[: net.active_size["l4"]].contiguous())
            del old_thw
            # _m_pool (多级记忆池) 裁到活性维 (与 W_lm 输入行对齐, 3 段 × a4)
            old_m = net._m_pool.data
            net.register_buffer("_m_pool", old_m[: 3 * net.active_size["l4"]].contiguous())
            del old_m
            # W_pred_54 [a5, a4]: 列 (L4) 裁到活性维; W_pred_43 [a4, a3]: 行 (L4) 裁
            old_p54 = net.W_pred_54.data
            net.W_pred_54 = nn.Parameter(old_p54[:, : net.active_size["l4"]].contiguous())
            del old_p54
            old_p43 = net.W_pred_43.data
            net.W_pred_43 = nn.Parameter(old_p43[: net.active_size["l4"], :].contiguous())
            del old_p43
            # E_t4 裁到活性维 (W_t4 方阵随 L4 收缩, decorr 状态必须同尺寸)
            old_et4 = net.E_t4.data
            net.E_t4 = nn.Parameter(old_et4[: net.active_size["l4"], : net.active_size["l4"]].contiguous())
            del old_et4
            # _theta_wt4 裁到活性维 (W_t4 homeostatic 滑阈)
            old_tht4 = net._theta_wt4.data
            net.register_buffer("_theta_wt4", old_tht4[: net.active_size["l4"]].contiguous())
            del old_tht4

        # W_56/W_t5 列同步: L5 修剪后 W_56 输入维 = W_t5 方阵维 = 活性 L5
        if net.active_size["l5"] < net.cfg.d_l5:
            old = net.W_56.data
            net.W_56 = nn.Parameter(old[: net.active_size["l6"], : net.active_size["l5"]].contiguous())
            del old
            old_t5 = net.W_t5.data
            net.W_t5 = nn.Parameter(old_t5[: net.active_size["l5"], : net.active_size["l5"]].contiguous())
            del old_t5
