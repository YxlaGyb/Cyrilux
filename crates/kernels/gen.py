"""
生成环算子

continuation 逐步解码路径的具名算子, 与改前 forward.continuation 的内联表达式逐位对应;
算子只吃 env 传入的张量与整型常数, 不读 net.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from crates.kernels.activation import cubic, saturate
from crates.kernels.normalize import rms_norm
from crates.modal.text.geometry import LEAD_CONTINUATIONS, LEAD_RANGES


def hold_zero(bind_vec: torch.Tensor) -> torch.Tensor:
    """承诺 hold 首次分配: 与绑定同宽的全零."""
    return torch.zeros_like(bind_vec[:, -1])


def win_buf(cont_cur: torch.Tensor, n_total: int) -> torch.Tensor:
    """工作缓冲切片: 前缀 + n_gen 步预分配."""
    return cont_cur[:, :n_total]


def buf_seed(cur_buf: torch.Tensor, byte_ids: torch.Tensor, s_len: int) -> None:
    """前缀拷入工作缓冲 (就地)."""
    cur_buf[:, :s_len].copy_(byte_ids)


def clear_expect(expect_cont: torch.Tensor) -> None:
    """UTF-8 状态清零 (就地)."""
    expect_cont.zero_()


def clear_counts(counts: torch.Tensor) -> None:
    """生成计数清零 (就地)."""
    counts.zero_()


def ctx_window(cur_buf: torch.Tensor, cur_len: int, win: int) -> torch.Tensor:
    """上下文窗口: 最近 win 字节 (win = min(cur_len, 64))."""
    return cur_buf[:, cur_len - win : cur_len]


def predict_state(engine: Any, bv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """前馈一次 (P1 图内嵌套), 交出写回后的 L4 / 记忆读出 / 绑定."""
    engine._predict(bv, store_state=True, is_inference=True)
    net = engine.net
    return net._z4, net._mem_out, net._bind_vec


def pred_delta(z4_n: torch.Tensor, W_diff: torch.Tensor, b_diff: torch.Tensor, dim_4: int) -> torch.Tensor:
    """预测差分头: 单位化 L4 状态经 W_diff 生成下一帧差分."""
    return z4_n @ W_diff[:dim_4, :dim_4].T + b_diff[:dim_4].unsqueeze(0).unsqueeze(0)


def bind_hold(bind_vec: torch.Tensor, hold: torch.Tensor, expect_cont: torch.Tensor) -> torch.Tensor:
    """字符中途用承诺 hold 覆盖末位绑定 (只覆盖 zh, 不动 W_act)."""
    mid = expect_cont.unsqueeze(-1) > 0
    merged = torch.where(mid, hold, bind_vec[:, -1])
    return torch.cat([bind_vec[:, :-1], merged.unsqueeze(1)], dim=1)


def join3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """zh 组装: L4 状态 / 绑定 / 记忆读出 三路拼接."""
    return torch.cat([a, b, c], dim=-1)


def mix(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """混合层: 单次 matmul, 无偏置."""
    return x @ w


def logit_head(h_old: torch.Tensor, W_lm: torch.Tensor, bias_lm: torch.Tensor, inv_h: float) -> torch.Tensor:
    """冻结旧通道 logits: (h_old @ W_lm + bias_lm) 的 1/sqrt(d_h) 定标."""
    return (h_old @ W_lm + bias_lm) * inv_h


def logit_cat(
    h_old: torch.Tensor,
    W_lm: torch.Tensor,
    bias_lm: torch.Tensor,
    h_new: torch.Tensor,
    W_lm_new: torch.Tensor,
    inv_h: float,
    alpha: float,
) -> torch.Tensor:
    """双通道 logits: 旧通道定标项 + 残差新通道."""
    return (h_old @ W_lm + bias_lm) * inv_h + h_new @ W_lm_new * alpha


def logits_std(mu0: torch.Tensor, eps: float) -> torch.Tensor:
    """标准化 + 峰度定标: 逐行零均值单位方差, 再线性拉满 60."""
    c = (mu0 - mu0.mean(dim=-1, keepdim=True)) / (mu0.std(dim=-1, keepdim=True) + eps)
    return c / c.abs().max(dim=-1, keepdim=True).values * 60.0


def mask_print(mu0: torch.Tensor, mask: torch.Tensor, floor: float) -> torch.Tensor:
    """打印屏蔽: 不可打印位加同一个下界."""
    return mu0 + (1.0 - mask) * floor


def last_row(x: torch.Tensor) -> torch.Tensor:
    """取窗末帧 logits."""
    return x[:, -1]


def evo_logits(
    z4_nl: torch.Tensor,
    bind_vec: torch.Tensor,
    mem_out: torch.Tensor,
    W1: torch.Tensor,
    W_lm: torch.Tensor,
    bias_lm: torch.Tensor,
    inv_h: float,
) -> torch.Tensor:
    """承诺遥测: 末帧 live-bind 走与生成头同款的 logits 管线."""
    zh = torch.cat([z4_nl[:, -1:], bind_vec[:, -1:], mem_out[:, -1:]], dim=-1)
    h = cubic(rms_norm(saturate(zh @ W1)))
    return ((h @ W_lm + bias_lm) * inv_h)[:, -1]


def gen_count(
    counts: torch.Tensor, last: torch.Tensor, lg_evo: torch.Tensor, expect_cont: torch.Tensor
) -> None:
    """承诺遥测累加 (就地): [1] 承诺与 live 的 argmax 差异行数, [2] 字符中途行数.

    字符中途全无时两项皆零, 与改前 host 分支跳过等价, 但不产生同步点.
    """
    mid = expect_cont > 0
    diff = last.argmax(-1) != lg_evo.argmax(-1)
    counts[1] += diff.sum() * mid.any().to(torch.long)
    counts[2] += mid.sum()


def act_inject(last: torch.Tensor, bind_vec: torch.Tensor, W_act: torch.Tensor) -> torch.Tensor:
    """W_act 意图电位裸注入 (用 live bind)."""
    return last + (bind_vec[:, -1] @ W_act)


def temp_scale(last: torch.Tensor, temperature: Any) -> torch.Tensor:
    """温度缩放: 张量时 GPU 直除 (零同步)."""
    return last / temperature


def rep_block(
    cur_buf: torch.Tensor, cur_len: int, last: torch.Tensor, p_max: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """n-gram 周期回退: 重复周期末字节置 -1e4, 返回屏蔽后 logits 与逐行命中数 (罕见路径)."""
    n_batch = cur_buf.shape[0]
    hits = torch.zeros(n_batch, dtype=torch.long, device=cur_buf.device)
    alphabet = torch.arange(last.shape[-1], device=cur_buf.device)[None, :]
    for p in range(2, p_max + 1):
        pat = cur_buf[:, cur_len - p : cur_len]
        prev = cur_buf[:, cur_len - 2 * p : cur_len - p]
        period = (pat == prev).all(dim=-1)
        blocked = (pat.unsqueeze(-1) == alphabet).any(dim=-2) & period.unsqueeze(-1)
        last = last.masked_fill(blocked, -1e4)
        hits = hits + period.to(torch.long)
    return last, hits


def count_add0(counts: torch.Tensor, n_block: torch.Tensor) -> None:
    """周期回退命中累加 (就地)."""
    counts[0] += n_block.sum()


def utf8_mask(
    last: torch.Tensor, expect_cont: torch.Tensor, block_cnt: torch.Tensor, block_start: torch.Tensor
) -> torch.Tensor:
    """UTF-8 语法阻断: 字符中途须续字节, 字符边界须 ASCII 或合法起始."""
    is_cont = (expect_cont > 0).unsqueeze(-1)
    last = last.masked_fill(is_cont & block_cnt[None, :], -1e4)
    return last.masked_fill(~is_cont & block_start[None, :], -1e4)


def topk_floor(last: torch.Tensor, k: int) -> torch.Tensor:
    """topk 阈值: 阈值以下写 -inf, 阈值以上原值返回."""
    topv, _ = torch.topk(last, k, dim=-1)
    return torch.where(last < topv[:, -1:], -math.inf, last)


def softmax16(last: torch.Tensor) -> torch.Tensor:
    """取样分布 (fp16 直算)."""
    return torch.softmax(last, dim=-1)


def sample_entropy(last: torch.Tensor, freq_act: torch.Tensor, beta: float) -> torch.Tensor:
    """熵探索: 频率偏置软采样 (变异通道, 不进学习更新)."""
    pot = last + (beta * (1.0 - freq_act)).unsqueeze(0)
    return torch.multinomial(torch.softmax(pot, dim=-1), 1).squeeze(-1)


def sample_temp(probs: torch.Tensor) -> torch.Tensor:
    """按分布取样."""
    return torch.multinomial(probs, 1).squeeze(-1)


def sample_greedy(probs: torch.Tensor) -> torch.Tensor:
    """贪心取最大概率字节."""
    return probs.argmax(dim=-1)


def buf_write(cur_buf: torch.Tensor, b: torch.Tensor, cur_len: int) -> int:
    """采样字节回写工作缓冲 (就地), 交出推进后的游标."""
    cur_buf[:, cur_len : cur_len + 1] = b.unsqueeze(-1)
    return cur_len + 1


def lead_len_of(b: torch.Tensor) -> torch.Tensor:
    """UTF-8 前导字节的续字节需求."""
    lead = torch.zeros_like(b)
    for (lo, hi), n_cont in zip(LEAD_RANGES, LEAD_CONTINUATIONS):
        lead = torch.where((b >= lo) & (b <= hi), n_cont, lead)
    return lead


def hold_step(
    hold: torch.Tensor, bind_vec: torch.Tensor, expect_cont: torch.Tensor, lead: torch.Tensor
) -> torch.Tensor:
    """刚采出合法起始 → 快照 hold = 该步 live bind."""
    new_unit = (expect_cont == 0) & (lead > 0)
    return torch.where(new_unit.unsqueeze(-1), bind_vec[:, -1], hold)


def expect_step(expect_cont: torch.Tensor, lead: torch.Tensor) -> None:
    """UTF-8 状态推进 (就地): 中途递减, 边界改为新需求."""
    expect_cont.copy_(torch.where(expect_cont > 0, expect_cont - 1, lead))


def out_view(cur_buf: torch.Tensor, cur_len: int) -> torch.Tensor:
    """返回视图: 前缀 + 已生成步."""
    return cur_buf[:, :cur_len]


def flush_counts(counts: torch.Tensor) -> tuple[int, int, int]:
    """计数搬运: 一次 host 读取, 交出 (周期命中, 承诺差异, 字符中途)."""
    return tuple(counts.tolist())
