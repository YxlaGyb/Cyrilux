"""
PruningEngine
动态神经元修剪 (慢速循环).

拓扑重塑三段: 发育期内不剪 → 死缓二级判决 → 相对排名淘汰.
修剪同步: 同 perm 重排所有 L4 行映射权重与下游列 (错位是历史 NaN 根因).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from model.model_cyrene import DensePCNet


class PruningEngine:
    """修剪引擎: 持 net 引用, 操作 net.active_size/_death_row/权重张量."""

    def __init__(self, net: DensePCNet):
        self.net = net

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

        L4 传 score_fn (行范数 × W_lm 依赖), 只剪对预测无贡献的神经元; 其余层用纯行范数.
        perm=None 表示无修剪 (n_alive=active).
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
        # W_t* decorr 状态同 perm (Et 行列 = 该层神经元, 错位 → decorr 拆错方向)
        et_attr = {"l4": "E_t4", "l5": "E_t5"}
        if layer in et_attr:
            Et = getattr(net, et_attr[layer])
            Et.data = Et.data[perm][:, perm].contiguous()
        b = getattr(net, self._b_attr[layer])
        b.data = b.data[perm].contiguous()

        # L5 专属 perm: 方阵三张 + 行列向量
        if layer == "l5":
            old_ml5 = net.M_l5.data
            net.M_l5 = nn.Parameter(old_ml5[perm][:, perm].contiguous())  # [dim_5, dim_5]
            del old_ml5
            old_el5 = net.E_l5.data
            net.E_l5 = nn.Parameter(old_el5[perm][:, perm].contiguous())  # [dim_5, dim_5]
            del old_el5
            net.W_pred_54.data = net.W_pred_54.data[perm].contiguous()  # 行=L5 (列=L4 由 _sync_l4_aux)
            old_gm = net._gain_mask.data
            net.register_buffer("_gain_mask", old_gm[perm].contiguous())  # 行=L5 (列=L3 由 l3 分支)
            del old_gm

        net._death_row[layer] = new_dr[perm]
        net._probation_counter[layer] = new_pc[perm]
        perm_map[layer] = perm
        return perm, n_alive

    def _sync_l4_aux(self, perm: torch.Tensor) -> None:
        """L4 行重排 → 同步重排所有 L4 行映射权重 (错位 → 修剪后首爆 NaN)."""
        net = self.net
        # W_lm 行 = d_h 混合空间 (无神经元映射), 不随 L4 perm 重排
        # W1 行 = [z4 | bind | 单元×K]: z4/bind 段同 perm, 单元块逐段偏移 head+i*a4p+perm
        n_bind = net.bind_slot_dim
        K = net._mem_m.shape[0]
        head = perm.shape[0] + n_bind
        a4p = perm.shape[0]
        idx = torch.cat(
            [perm, torch.arange(a4p, head, device=perm.device)]
            + [head + i * a4p + perm for i in range(K)]
        )
        net.W1.data = net.W1.data[idx].contiguous()
        old_m2 = net._mem_m.data
        net.register_buffer("_mem_m", old_m2[:, perm].contiguous())  # 单元状态列 = L4 神经元
        del old_m2
        net.W_diff.data = net.W_diff.data[perm][:, perm].contiguous()
        net.W_state_pred.data = net.W_state_pred.data[perm][:, perm].contiguous()
        net.W_pred_54.data = net.W_pred_54.data[:, perm].contiguous()  # 列 = L4
        net.W_pred_43.data = net.W_pred_43.data[perm].contiguous()  # 行 = L4
        net.W_bind.data = net.W_bind.data[perm].contiguous()
        for i in range(4):  # 环形缓冲同 perm (缓冲与 W_diff 行错位 → 更新错乱)
            old_buf = getattr(net, f"_dw_buf_{i}").data
            net.register_buffer(f"_dw_buf_{i}", old_buf[perm][:, perm].contiguous())
            del old_buf
        old_thw = net._theta_w.data
        net.register_buffer("_theta_w", old_thw[perm].contiguous())  # BCM 滑阈对齐 W_diff 行
        del old_thw
        old_eb = net.E_bind.data
        net.E_bind = nn.Parameter(old_eb[perm][:, perm].contiguous())
        del old_eb
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
        # 顺序关键: 先 perm 后裁 (perm 长度 = 修剪前 active). 仅当源层尚未本次收缩
        # (列数 == src_perm 长度) 才应用 src_perm, 否则上游先裁后旧 perm 会越界.
        src_perm = perm_map.get(src)
        if src_perm is not None and old.shape[1] == src_perm.numel():
            old = old[:, src_perm]
        setattr(net, self._W_attr[layer], nn.Parameter(old[:n_alive, :src_n].contiguous()))
        del old
        # W_attr 同形 _elig 迹同步
        W_elig_name = f"{self._W_attr[layer]}_elig"
        if hasattr(net, W_elig_name):
            old_ew = getattr(net, W_elig_name).data
            if src_perm is not None and old_ew.shape[1] == src_perm.numel():
                old_ew = old_ew[:, src_perm]
            net.register_buffer(W_elig_name, old_ew[:n_alive, :src_n].contiguous())
            del old_ew
        # _gain_l3 列 = L2 神经元, 用 L2 自己的 perm (非 src perm)
        l2_perm = perm_map.get("l2")
        if l2_perm is not None:
            old_g = net._gain_l3.data
            # 仅当列尚未被上游收缩到 src_n 时应用 perm
            if old_g.shape[1] == l2_perm.numel():
                net.register_buffer("_gain_l3", old_g[:, l2_perm].contiguous())
            else:
                net.register_buffer("_gain_l3", old_g.contiguous())
            del old_g

        # L3 修剪同步 W_35 列数 + 按 perm 重排列 (z3 重排后仍喂旧序列 → eps5 突变 NaN)
        if layer == "l3":
            l3_perm = perm_map.get("l3")
            old_35 = net.W_35.data
            if l3_perm is not None and old_35.shape[1] == l3_perm.numel():
                old_35 = old_35[:, l3_perm]
            net.W_35 = nn.Parameter(old_35[:, :src_n].contiguous())
            del old_35
            # W_35_elig 列同源 L3: 同行列收缩
            W35e_name = "W_35_elig"
            if hasattr(net, W35e_name):
                old_35e = getattr(net, W35e_name).data
                # 行 = L5, 本 layer==l3 分支不裁 L5 维; 保持原行数
                cur_a5 = old_35e.shape[0]
                if l3_perm is not None and old_35e.shape[1] == l3_perm.numel():
                    old_35e = old_35e[:, l3_perm]
                net.register_buffer(W35e_name, old_35e[:cur_a5, :src_n].contiguous())
                del old_35e
            # _gain_mask 列与 W_35 列同源 (L3 神经元), 同 perm 重排
            old_gm = net._gain_mask.data
            cur_a5_gm = old_gm.shape[0]
            if l3_perm is not None and old_gm.shape[1] == l3_perm.numel():
                old_gm = old_gm[:, l3_perm]
            net.register_buffer("_gain_mask", old_gm[:cur_a5_gm, :src_n].contiguous())
            del old_gm
            # _gain_l3 行与 W_23 行同源 (L3 神经元), 同 perm 重排 + 裁列到 src_n
            if l3_perm is not None:
                old_g = net._gain_l3.data
                if old_g.shape[0] == l3_perm.numel():
                    old_g = old_g[l3_perm]
                net.register_buffer("_gain_l3", old_g[:, :src_n].contiguous())
                del old_g

        W_t = getattr(net, self._t_attr[layer])
        old_t = W_t.data
        setattr(net, self._t_attr[layer], nn.Parameter(old_t[:n_alive, :n_alive].contiguous()))
        del old_t
        # t_attr 同形 _elig 迹同步 (方阵)
        Wt_elig_name = f"{self._t_attr[layer]}_elig"
        if hasattr(net, Wt_elig_name):
            old_et = getattr(net, Wt_elig_name).data
            net.register_buffer(Wt_elig_name, old_et[:n_alive, :n_alive].contiguous())
            del old_et

        b = getattr(net, self._b_attr[layer])
        old_b = b.data
        setattr(net, self._b_attr[layer], nn.Parameter(old_b[:n_alive].contiguous()))
        del old_b

        # L5 专属: 行=L5 的张量同步收缩
        if layer == "l5":
            a4_cur = net.active_size["l4"]
            a3_cur = src_n  # src=L3, src_n=active_size["l3"] 已由 L3 分支更新
            # W_pred_54 + 资格迹 (行=L5 裁; 列=L4 保持 a4_cur)
            old_p54 = net.W_pred_54.data
            net.W_pred_54 = nn.Parameter(old_p54[:n_alive, :a4_cur].contiguous())
            del old_p54
            Wp54e_name = "W_pred_54_elig"
            if hasattr(net, Wp54e_name):
                old_p54e = getattr(net, Wp54e_name).data
                net.register_buffer(Wp54e_name, old_p54e[:n_alive, :a4_cur].contiguous())
                del old_p54e
            # _gain_mask 行=L5 裁 (列=L3 在上面 layer==l3 分支已裁到 a3_cur)
            old_gm = net._gain_mask.data
            net.register_buffer("_gain_mask", old_gm[:n_alive, :a3_cur].contiguous())
            del old_gm
            # 方阵三张: M_l5 / E_l5 / E_t5
            for attr in ("M_l5", "E_l5", "E_t5"):
                old_x = getattr(net, attr).data
                setattr(net, attr, nn.Parameter(old_x[:n_alive, :n_alive].contiguous()))
                del old_x
            # BCM 滑阈 _theta_l5
            old_thl5 = net._theta_l5.data
            net.register_buffer("_theta_l5", old_thl5[:n_alive].contiguous())
            del old_thl5
            # STP l5 四兄弟 (r / tau / u / act_ema)
            for attr in ("_stp_r_l5", "_stp_tau_l5", "_stp_u_l5", "_stp_active_ema_l5"):
                old_x = getattr(net, attr).data
                net.register_buffer(attr, old_x[:n_alive].contiguous())
                del old_x
            # act_ema L5 四兄弟: w35 / wt5 / wp54 / b5
            for attr in ("_active_ema_w35", "_active_ema_wt5", "_active_ema_wp54", "_active_ema_b5"):
                old_x = getattr(net, attr).data
                net.register_buffer(attr, old_x[:n_alive].contiguous())
                del old_x

        if net._death_row[layer] is not None:
            net._death_row[layer] = net._death_row[layer][:n_alive]
            net._probation_counter[layer] = net._probation_counter[layer][:n_alive]

        net.active_size[layer] = n_alive

    def _prune(self):
        """拓扑重塑: 发育期内不剪 → 死缓二级判决 → 相对排名淘汰."""
        net = self.net
        # 顺序严格保持依赖链: 下游 src 对应的层必须先 perm/裁 (L3→L5→L6, L4→L2→L3)
        layers = ["l4", "l2", "l3", "l5", "l6"]
        self._W_attr = {"l4": "W_04", "l2": "W_42", "l3": "W_23", "l5": "W_35", "l6": "W_56"}
        self._t_attr = {"l4": "W_t4", "l2": "W_t2", "l3": "W_t3", "l5": "W_t5", "l6": "W_t6"}
        self._b_attr = {"l4": "bias_l4", "l2": "bias_l2", "l3": "bias_l3", "l5": "bias_l5", "l6": "bias_l6"}
        src_layer = {"l4": "l0", "l2": "l4", "l3": "l2", "l5": "l3", "l6": "l5"}

        bound = net.cfg.active_size_lower_bound
        l4_bound = net.cfg.l4_lower_bound  # 预测主空间保底
        frac = net.cfg.prune_fraction
        dprob = net.cfg.death_probation
        self._bounds = {"l4": l4_bound, "l2": bound, "l3": bound, "l5": bound, "l6": bound}

        expired_flags = self._expire_flags(layers, dprob)

        # 阶段一: 行 perm 重排 (active_size 尚未更新, n_alive_map 记录新尺寸)
        n_alive_map: dict[str, int] = {}
        perm_map: dict[str, torch.Tensor] = {}
        for layer in layers:
            active = net.active_size[layer]
            layer_bound = self._bounds[layer]
            if active <= layer_bound:
                n_alive_map[layer] = active
                continue
            n_candidate = max(1, int(active * frac))
            expired = expired_flags.get(layer)
            # L4 候选分数 = 表示层行范数 × W1 z4 段行范数, 只剪对预测无贡献的神经元
            score_fn = None
            if layer == "l4":
                rn_w1 = net.W1[:active].data.norm(dim=1)

                def score_fn(rn, a):
                    return rn * rn_w1[:a]
            perm, n_alive = self._permute_weights(layer, active, n_candidate, expired, perm_map, score_fn)
            n_alive_map[layer] = n_alive
            if perm is None:
                continue
            if layer == "l4":
                self._sync_l4_aux(perm)

        # 阶段二: 列同步 & 显存回收 (先 perm 后裁; up_size 链式传递, l0 = 单帧 256 或双通道 512)
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
        # W_diff / W_state_pred L4 方阵随修剪同步
        if net.active_size["l4"] < net.cfg.d_l4:
            old_fut = net.W_diff.data
            net.W_diff = nn.Parameter(old_fut[: net.active_size["l4"], : net.active_size["l4"]].contiguous())
            del old_fut
            old_sp = net.W_state_pred.data
            net.W_state_pred = nn.Parameter(
                old_sp[: net.active_size["l4"], : net.active_size["l4"]].contiguous()
            )
            del old_sp
            # W1 行 = [z4 | bind | 单元×K]; W_lm 行 = d_h 混合空间 (不裁剪)
            n_bind = net.bind_slot_dim
            n_cell = net._mem_m.shape[0] * net.active_size["l4"]
            old_w1 = net.W1.data
            net.W1 = nn.Parameter(old_w1[: net.active_size["l4"] + n_bind + n_cell, :].contiguous())
            del old_w1
            old_mm = net._mem_m.data
            net.register_buffer("_mem_m", old_mm[:, : net.active_size["l4"]].contiguous())  # 单元状态列
            del old_mm
            old_bind = net.W_bind.data
            net.W_bind = nn.Parameter(old_bind[: net.active_size["l4"], :].contiguous())  # 列 = 槽位固定
            del old_bind
            for i in range(4):  # 环形缓冲同步 (copy_ 形状错位即崩)
                old_buf = getattr(net, f"_dw_buf_{i}").data
                net.register_buffer(
                    f"_dw_buf_{i}",
                    old_buf[: net.active_size["l4"], : net.active_size["l4"]].contiguous(),
                )
                del old_buf
            old_thw = net._theta_w.data
            net.register_buffer("_theta_w", old_thw[: net.active_size["l4"]].contiguous())  # 滑阈对齐 W_diff 行
            del old_thw
            old_p54 = net.W_pred_54.data
            net.W_pred_54 = nn.Parameter(old_p54[:, : net.active_size["l4"]].contiguous())  # 列 = L4
            del old_p54
            old_p43 = net.W_pred_43.data
            net.W_pred_43 = nn.Parameter(old_p43[: net.active_size["l4"], :].contiguous())  # 行 = L4
            del old_p43
            old_et4 = net.E_t4.data
            net.E_t4 = nn.Parameter(old_et4[: net.active_size["l4"], : net.active_size["l4"]].contiguous())  # decorr 状态同尺寸
            del old_et4
            old_tht4 = net._theta_wt4.data
            net.register_buffer("_theta_wt4", old_tht4[: net.active_size["l4"]].contiguous())  # homeostatic 滑阈
            del old_tht4
