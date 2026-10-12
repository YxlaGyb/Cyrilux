"""
PruningEngine
动态神经元生死 (判决侧) 与拓扑压实 (滞后侧).

两段分离是生死的根本口径:
  判决 decide()  —— 每轮写 S_l 与死缓簿, 只关位, 不动任何张量形状,
                     active_size 保持为物理宽度 (切片长度).
  压实 compact() —— 滞后执行, 把 S_l=0 的行与下游列真正裁掉并回收显存,
                     只有到此步 active_size 才会变小.

压实同步: 同 perm 重排所有 L4 行映射权重与下游列 (错位是历史 NaN 根因).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from .registry import ROW_STATE
from .survival import LAYERS

if TYPE_CHECKING:
    from crates.model.model_cyrene import HybridPCNet


class PruningEngine:
    """
    修剪引擎，持 net 引用, 操作 net.active_size/_S_l/_death_row_*/权重张量
    """

    def __init__(self, net: HybridPCNet):
        self.net = net

    def _attrs(self) -> None:
        """
        登记表 + 各层下限. 判决与压实共用, 避免两处字面量漂移
        """
        net = self.net
        self._W_attr = {"l4": "W_04", "l2": "W_42", "l3": "W_23", "l5": "W_35", "l6": "W_56"}
        self._t_attr = {"l4": "W_t4", "l2": "W_t2", "l3": "W_t3", "l5": "W_t5", "l6": "W_t6"}
        self._b_attr = {"l4": "bias_l4", "l2": "bias_l2", "l3": "bias_l3", "l5": "bias_l5", "l6": "bias_l6"}
        bound = net.cfg.active_size_lower_bound
        self._bounds = {
            "l4": net.cfg.l4_lower_bound, "l2": bound, "l3": bound, "l5": bound, "l6": bound,
        }

    def decide(self) -> None:
        """
        判决: 行范数相对排名 → 死缓登记; 死缓期满未复活 → S_l 关位.

        本方法不改变 active_size 与任何张量形状 — 被关位的神经元只是不再点火,
        它的行/列仍占物理宽度直到压实点. 所有关位都受下限约束, 不会把活体数
        压到 active_size_lower_bound (L4 另用 l4_lower_bound) 之下.
        """
        net = self.net
        self._attrs()
        frac = net.cfg.prune_fraction
        dprob = net.cfg.death_probation
        for layer in LAYERS:
            active = net.active_size[layer]
            S = getattr(net, f"_S_{layer}")[:active]
            dr = getattr(net, f"_death_row_{layer}")[:active]
            pc = getattr(net, f"_probation_counter_{layer}")[:active]
            bound = self._bounds[layer]
            W = getattr(net, self._W_attr[layer])
            rn = W[:active].data.norm(dim=1)
            if layer == "l4":
                rn = rn * net.W1[:active].data.norm(dim=1)

            n_live = int(S.sum())
            in_death = dr.bool() & S
            revived = in_death & (rn > net.cfg.death_threshold)
            if revived.any():
                dr[revived] = 0
                pc[revived] = 0
                in_death = dr.bool() & S

            pc[in_death] += 1
            expired = in_death & (pc >= dprob)
            n_exp = int(expired.sum())
            allow = max(0, n_live - bound)
            if n_exp > allow:
                ix = torch.where(expired)[0][:allow]
                expired = torch.zeros_like(expired)
                expired[ix] = True
                n_exp = allow
            if n_exp:
                S[expired] = False
                dr[expired] = 0
                pc[expired] = 0
                n_live -= n_exp

            eligible = S & ~dr.bool()
            n_elig = int(eligible.sum())
            n_candidate = min(max(1, int(n_live * frac)), max(0, n_elig - bound))
            if n_candidate <= 0:
                continue
            score = rn.masked_fill(~eligible, float("inf"))
            _, ix = score.topk(n_candidate, dim=-1, largest=False)
            dr[ix] = 1
            pc[ix] = 0

    def has_releasable(self) -> bool:
        """释放工位是否已达压实门槛 (任一层的空位 ≥ 该层宽度 1%)."""
        for layer in LAYERS:
            S = getattr(self.net, f"_S_{layer}")
            if (S.numel() - int(S.sum())) >= max(1, S.numel() // 100):
                return True
        return False

    def _permute_weights(
        self, layer: str, active: int, perm_map: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor | None, int]:
        """压实重排: 只按 _S_l 落位 — S=1 行在前, S=0 行在后统一裁掉.

        判决不在本处: 谁该死由 decide() 每轮写进 S_l, 本处只做滞后物理折叠.
        物理宽度按 8 对齐补齐 (不足时用已关位的行填充前段, 它们不点火).
        perm=None 表示该层无需压实 (S 全一).
        """
        net = self.net
        W = getattr(net, self._W_attr[layer])
        live = getattr(net, f"_S_{layer}")[:active].bool()
        n_live = int(live.sum())
        n_alive = max(self._bounds[layer], (n_live // 8) * 8)
        n_alive = min(n_alive, active)
        if n_alive >= active:
            return None, active

        perm = torch.cat([torch.where(live)[0], torch.where(~live)[0]])
        W.data = W.data[perm].contiguous()
        W_t = getattr(net, self._t_attr[layer])
        W_t.data = W_t.data[perm][:, perm].contiguous()
        b = getattr(net, self._b_attr[layer])
        b.data = b.data[perm].contiguous()
        # 8 对齐补齐的前段占位行仍是关位: 不可用 ones 复活它们
        new_S = torch.zeros(active, dtype=torch.bool, device=live.device)
        new_S[:n_live] = True
        net.register_buffer(f"_S_{layer}", new_S)
        net.register_buffer(
            f"_death_row_{layer}", getattr(net, f"_death_row_{layer}")[:active][perm].contiguous()
        )
        net.register_buffer(
            f"_probation_counter_{layer}",
            getattr(net, f"_probation_counter_{layer}")[:active][perm].contiguous(),
        )

        # L5 专属 perm: 行=L5 的权重 (列=L4 由 _sync_l4_aux, 列=L3 由 l3 分支)
        if layer == "l5":
            net.W_pred_54.data = net.W_pred_54.data[perm].contiguous()
            old_gm = net._gain_mask.data
            net.register_buffer("_gain_mask", old_gm[perm].contiguous())
            del old_gm

        perm_map[layer] = perm
        return perm, n_alive

    def _sync_layer_state(self, layer: str, n_alive: int, perm: torch.Tensor | None) -> None:
        """层内行索引状态同步: 按 ROW_STATE 逐项 perm 重排 + 收缩 (perm=None 时只收缩)."""
        net = self.net
        for name, (lyr, form, handler) in ROW_STATE.items():
            if handler != "aux" or lyr != layer:
                continue
            buf = getattr(net, name, None)
            if not isinstance(buf, torch.Tensor):
                continue
            new = buf
            if perm is not None and buf.shape[0] == perm.numel():
                new = buf[perm][:, perm] if form == "sq" else buf[perm]
            if new.shape[0] != n_alive:
                new = new[:n_alive, :n_alive] if form == "sq" else new[:n_alive]
            if new is not buf:
                if name in net._parameters:  # b_diff 等是 nn.Parameter
                    setattr(net, name, nn.Parameter(new.contiguous()))
                else:
                    net.register_buffer(name, new.contiguous())
        if layer in net._stp_r_end:  # STP 资源窗口写回 (dict, 非 buffer)
            old_r = net._stp_r_end[layer]
            if perm is not None and old_r.shape[0] == perm.numel():
                old_r = old_r[perm]
            net._stp_r_end[layer] = old_r[:n_alive].contiguous()

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
        """
        W1_elig 迹必须与 W1 行同 idx 重排
        否则迹留在原始行序, 与 W1 布局错位 → W1 解冻下每步把迹注入错误行
        最终收缩块的 gather 也因布局不同而错位
        """
        old_e1 = net.W1_elig.data
        net.register_buffer("W1_elig", old_e1[idx].contiguous())
        del old_e1
        old_m2 = net._mem_m.data
        net.register_buffer("_mem_m", old_m2[:, perm].contiguous())  # 单元状态列 = L4 神经元
        del old_m2
        core = net._core_brain
        core.projector.P_short.data = core.projector.P_short.data[perm].contiguous()
        core.projector.P_long.data = core.projector.P_long.data[perm].contiguous()
        core.projector.register_buffer(
            "P_long_ema", core.projector.P_long_ema[perm].contiguous()
        )
        core.register_buffer("_delta_prev", core._delta_prev[perm].contiguous())
        core._state = None
        net.W_diff.data = net.W_diff.data[perm][:, perm].contiguous()
        net.W_state_pred.data = net.W_state_pred.data[perm][:, perm].contiguous()
        net.W_pred_54.data = net.W_pred_54.data[:, perm].contiguous()  # 列 = L4
        net.W_pred_43.data = net.W_pred_43.data[perm].contiguous()  # 行 = L4
        net.W_bind.data = net.W_bind.data[perm].contiguous()
        old_buf = net._dw_buf.data  # [4,L,L] 环形缓冲同 perm (缓冲与 W_diff 行错位 → 更新错乱)
        net.register_buffer("_dw_buf", old_buf[:, perm][:, :, perm].contiguous())
        del old_buf

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
        # W_attr 同形 _elig 迹: 只同步列 (行由 _sync_layer_state 按 perm 处理)
        W_elig_name = f"{self._W_attr[layer]}_elig"
        if hasattr(net, W_elig_name):
            old_ew = getattr(net, W_elig_name).data
            if src_perm is not None and old_ew.shape[1] == src_perm.numel():
                old_ew = old_ew[:, src_perm]
            net.register_buffer(W_elig_name, old_ew[:, :src_n].contiguous())
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

        b = getattr(net, self._b_attr[layer])
        old_b = b.data
        setattr(net, self._b_attr[layer], nn.Parameter(old_b[:n_alive].contiguous()))
        del old_b

        S_name = f"_S_{layer}"
        S = getattr(net, S_name, None)
        if isinstance(S, torch.Tensor) and S.shape[0] > n_alive:
            net.register_buffer(S_name, S[:n_alive].contiguous())

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
                net.register_buffer(Wp54e_name, old_p54e[:, :a4_cur].contiguous())
                del old_p54e
            # _gain_mask 行=L5 裁 (列=L3 在上面 layer==l3 分支已裁到 a3_cur)
            old_gm = net._gain_mask.data
            net.register_buffer("_gain_mask", old_gm[:n_alive, :a3_cur].contiguous())
            del old_gm
        self._sync_layer_state(layer, n_alive, perm_map.get(layer))
        for _bn in (f"_death_row_{layer}", f"_probation_counter_{layer}"):
            old_x = getattr(net, _bn).data
            net.register_buffer(_bn, old_x[:n_alive].contiguous())

        net.active_size[layer] = n_alive

    def compact(self) -> None:
        """压实: 把 S_l=0 的工位连同其下游列真正裁掉, 回收显存.

        滞后执行 — 只有在释放工位达门槛且距上次压实满一个间隔时才会被调用.
        顺序严格保持依赖链: src 对应的层必须先 perm/裁 (L4→L2→L3, L3→L5→L6).
        """
        net = self.net
        self._attrs()
        layers = list(LAYERS)
        src_layer = {"l4": "l0", "l2": "l4", "l3": "l2", "l5": "l3", "l6": "l5"}

        # 阶段一: 行 perm 重排 (active_size 尚未更新, n_alive_map 记录新尺寸)
        n_alive_map: dict[str, int] = {}
        perm_map: dict[str, torch.Tensor] = {}
        for layer in layers:
            active = net.active_size[layer]
            perm, n_alive = self._permute_weights(layer, active, perm_map)
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
        a4_new = net.active_size["l4"]
        n_bind = net.bind_slot_dim
        k_new = net._mem_m.shape[0]
        l4_perm = perm_map.get("l4")
        w1_stale = net.W1.shape[0] != a4_new + n_bind + k_new * a4_new
        if a4_new < net.cfg.d_l4 or w1_stale:
            old_fut = net.W_diff.data
            net.W_diff = nn.Parameter(old_fut[: net.active_size["l4"], : net.active_size["l4"]].contiguous())
            del old_fut
            old_sp = net.W_state_pred.data
            net.W_state_pred = nn.Parameter(
                old_sp[: net.active_size["l4"], : net.active_size["l4"]].contiguous()
            )
            del old_sp
            """
            W1 行 = [z4 | bind | 单元×K]; W_lm 行 = d_h 混合空间 (不裁剪).
            W1 布局 = _sync_l4_aux 的 perm 顺序:
            z4 块 (a4p 行, 已按 perm 排列) | bind 段 (n_bind) | 单元块×K (各 a4p 行, 已按 perm 排列).
            收缩 = 显式 gather: z4 取前列 a4_new (keep+幸存 probation), bind 段与
            单元块按 a4_new 切。头切片会把 z4 probation/死行误作 bind 与单元行，形状自洽不产 NaN, 是静默语义错位。
            """
            l4_perm = perm_map.get("l4")
            a4p = l4_perm.numel() if l4_perm is not None else a4_new
            if l4_perm is not None:
                head = a4p + n_bind

                def _gather_w1(old: torch.Tensor) -> torch.Tensor:
                    return torch.cat(
                        [old[:a4_new], old[a4p:a4p + n_bind]]
                        + [old[head + i * a4p : head + i * a4p + a4_new] for i in range(k_new)],
                        dim=0,
                    ).contiguous()

            else:

                def _gather_w1(old: torch.Tensor) -> torch.Tensor:
                    # 无 L4 perm: 布局未被重排, 头切片即恒等 (K 变化时按新块数截断)
                    return old[: a4_new + n_bind + k_new * a4_new].contiguous()

            old_w1 = net.W1.data
            net.W1 = nn.Parameter(_gather_w1(old_w1))
            del old_w1
            # W1_elig 迹同形同步 (修剪路径曾遗漏: 迹行与 W1 行错位 → W1 解冻下注入错误行)
            old_ew1 = net.W1_elig.data
            net.register_buffer("W1_elig", _gather_w1(old_ew1))
            del old_ew1
            net._lm_in = net.W1.shape[0]  # zh 列数 = W1 行数 (虚不变式, 喂 _mem_out 守卫)
            old_mm = net._mem_m.data
            net.register_buffer("_mem_m", old_mm[:, : net.active_size["l4"]].contiguous())  # 单元状态列
            del old_mm
            core = net._core_brain
            core.projector.P_short.data = core.projector.P_short.data[:a4_new].contiguous()
            core.projector.P_long.data = core.projector.P_long.data[:a4_new].contiguous()
            core.projector.register_buffer(
                "P_long_ema", core.projector.P_long_ema[:a4_new].contiguous()
            )
            core.register_buffer("_delta_prev", core._delta_prev[:a4_new].contiguous())
            core.cfg.d_in = a4_new
            core._state = None
            old_bind = net.W_bind.data
            net.W_bind = nn.Parameter(old_bind[: net.active_size["l4"], :].contiguous())  # 列 = 槽位固定
            del old_bind
            old_buf = net._dw_buf.data  # [4,L,L] 环形缓冲同步 (GPU 索引, 图捕获兼容)
            net.register_buffer(
                "_dw_buf",
                old_buf[:, : net.active_size["l4"], : net.active_size["l4"]].contiguous(),
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
        # 缓冲随活性同步 (形状比较幂等): 既有清单曾漏这几条 — 死亡回合收缩后第一步 learn 即崩
        # (temporal.py `_dw_slot.copy_(dW_diff_t)` shape 失配), 同类漏项在此一次收口.
        a4, a5 = net.active_size["l4"], net.active_size["l5"]
        for name, want in (("_dw_slot", (None, a4, a4)),
                           ("_z4_fut_buf", (None, None, a4)),
                           ("_z5_fut_buf", (None, None, a5)),
                           ("_theta_w04", (a4,))):
            buf = getattr(net, name)
            target = tuple(s if s is not None else b for s, b in zip(want, buf.shape))
            if buf.shape != target:
                net.register_buffer(name, buf[tuple(slice(0, w) for w in target)].contiguous())
