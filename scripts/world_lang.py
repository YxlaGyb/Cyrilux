"""
世界语言物理

本模块是外部测量仪器: 对发声字节流做语料统计评分 (L/S/X/q), 维护新颖度
历史与认证遥测 (n_certified / world_eps_ema), 产物只进 jsonl 报告, 永不
进入训练信号 (语言通用性红线: 评估器先验禁入训练信号).
机体代谢 (E/R/ΔF) 在 model/dense/learning/metabolism.py 体内计算

评分维度 (全部预计算自语料, 训练中冻结只读):
  L 覆盖率  CJK 字符 (ord>127) ∈ top-1000 常用字集占比 — 与
            audit_char_coverage 同口径 (判据线可比)
  S 结构    字节 3-gram 命中语料 3-gram 位图占比 (辅助)
  X 新颖    与最近 K 次自身发声 3-gram 并集重叠率 (复读惩罚, 报告口径)
  q         (0.8·L + 0.2·S)·(1 - X)

认证门 certify: q ≥ q_ref → 记一次认证并更新 ε 遥测 EMA — 仅报告统计;
恒温器设定点已改走体内 _lang_eps_ema, 本模块不发任何锚.

全部统计 CPU Python (非张量热路径).
常用字统计口径与 audit_char_coverage 一致 (前 2 万行字符 Counter).
"""
from __future__ import annotations

import json
from collections import Counter, deque

import numpy as np

_BITMAP_BITS = 2 ** 21  # 256^3 3-gram 存在位图 = 2MB


def _read_texts(path, start, count, min_bytes):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if len(out) >= count:
                break
            t = json.loads(line)["text"]
            if len(t.encode("utf-8")) >= min_bytes:
                out.append(t)
    return out


def _strict_decode(b):
    """逐字节 UTF-8 状态机; 非法位置记 None (占一个字符槽, 覆盖记 0)."""
    chars = []
    i, n = 0, len(b)
    while i < n:
        c = b[i]
        if c < 0x80:
            chars.append(chr(c))
            i += 1
            continue
        ln = 2 if 0xC2 <= c <= 0xDF else 3 if 0xE0 <= c <= 0xEF else 4 if 0xF0 <= c <= 0xF4 else 0
        if ln and i + ln <= n:
            try:
                chars.append(b[i:i + ln].decode("utf-8"))
                i += ln
                continue
            except UnicodeDecodeError:
                pass  # 非法序列 (含 0xF0-0xF4 第二字节过窄约束未满足) → 记 None
        chars.append(None)
        i += 1
    return chars


def _trigram_idx(a):
    """uint8 数组 → 3-gram 整数索引数组 [(b0<<16)|(b1<<8)|b2]."""
    if len(a) < 3:
        return np.empty(0, dtype=np.int64)
    x = a.astype(np.int64)
    return (x[:-2] << 16) | (x[1:-1] << 8) | x[2:]


def _hit_mask(bitmap, idx):
    if len(idx) == 0:
        return np.empty(0, dtype=bool)
    shifts = np.left_shift(np.uint8(1), (idx & 7).astype(np.uint8))
    return (bitmap[idx >> 3] & shifts) != 0


class WorldLangPhysics:
    """世界语言物理仪表: 预计算统计 + 评分 + 新颖度历史 + 认证遥测."""

    def __init__(self, data_path, n_char_lines=20000, n_trigram_lines=None,
                 top_n=1000, k_history=32, cert_frac=0.5,
                 ema_alpha=0.995, mad_alpha=0.95):
        # 常用字集 (audit_char_coverage 同口径)
        cnt = Counter()
        with open(data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n_char_lines:
                    break
                cnt.update(json.loads(line)["text"])
        self.common_set = {c for c, _ in cnt.most_common(top_n)}

        # 3-gram 存在位图 (全量语料的字节结构)
        self.bitmap = np.zeros(_BITMAP_BITS, dtype=np.uint8)
        self.n_trigrams = 0
        with open(data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if n_trigram_lines and i >= n_trigram_lines:
                    break
                a = np.frombuffer(json.loads(line)["text"].encode("utf-8"), dtype=np.uint8)
                idx = _trigram_idx(a)
                if len(idx):
                    shifts = np.left_shift(np.uint8(1), (idx & 7).astype(np.uint8))
                    np.bitwise_or.at(self.bitmap, idx >> 3, shifts)
                    self.n_trigrams += int(len(idx))
                if (i + 1) % 200000 == 0:
                    print(f"  world_lang: 3-gram 位图 {i + 1} 行...", flush=True)

        # 新颖度历史 (record 写入, score 只读)
        self.k_history = k_history
        self._hist = deque(maxlen=k_history)

        # 基线 q (heldout 行, 非位图构建段重叠控制)
        base_texts = _read_texts(data_path, 100000, 200, 64)
        Ls, Ss = [], []
        for t in base_texts:
            sc = self.score(t.encode("utf-8"))
            Ls.append(sc["L"])
            Ss.append(sc["S"])
        self.q_base = sum(Ls + Ss) / max(1, len(Ls) + len(Ss))
        self.q_ref = cert_frac * self.q_base

        # 认证遥测 (仅报告口径)
        self.world_eps_ema = None
        self.world_eps_mad = 0.0
        self.ema_alpha = ema_alpha
        self.mad_alpha = mad_alpha
        self.n_certified = 0

    def score(self, gen_bytes):
        """发声字节流 → {L, S, X, q}. 只读 (不更新历史).

        L 口径与 audit_char_coverage 一致 (CJK 字符的 top-1000 常用占比,
        判据线可比); q 按维度实测判别力加权 0.8/0.2.
        """
        b = bytes(gen_bytes)
        chars = _strict_decode(b)
        cjk = [c for c in chars if c is not None and ord(c) > 127]
        L = (sum(1 for c in cjk if c in self.common_set) / len(cjk)) if cjk else 0.0
        a = np.frombuffer(b, dtype=np.uint8)
        idx = _trigram_idx(a)
        S = float(_hit_mask(self.bitmap, idx).mean()) if len(idx) else 0.0
        cur = frozenset(int(v) for v in idx.tolist())
        if self._hist and cur:
            seen = set().union(*self._hist)  # 最近 K 次发声的 3-gram 并集
            X = len(cur.intersection(seen)) / len(cur)
        else:
            X = 0.0
        q = (0.8 * L + 0.2 * S) * (1.0 - X)
        return {"L": L, "S": S, "X": X, "q": q}

    def record(self, gen_bytes):
        """发声进入新颖度历史 (每次 echo 发声后调用)."""
        a = np.frombuffer(bytes(gen_bytes), dtype=np.uint8)
        idx = _trigram_idx(a)
        if len(idx):
            self._hist.append(frozenset(int(v) for v in idx.tolist()))

    def certify(self, q, eps_value):
        """认证遥测: q ≥ q_ref → 记一次认证并更新 ε EMA. 返回是否认证."""
        if q < self.q_ref:
            return False
        self.n_certified += 1
        if self.world_eps_ema is None:
            self.world_eps_ema = float(eps_value)
            self.world_eps_mad = 0.0
        else:
            devi = abs(float(eps_value) - self.world_eps_ema)
            self.world_eps_mad = (self.mad_alpha * self.world_eps_mad
                                  + (1.0 - self.mad_alpha) * devi)
            self.world_eps_ema = (self.ema_alpha * self.world_eps_ema
                                  + (1.0 - self.ema_alpha) * float(eps_value))
        return True

    def save_state(self, step, gen_temp):
        """世界仪表动态状态 + 恒温器温度 → 可 JSON 侧车 (断点续跑连续性).

        net.save() 只存权重; 认证遥测/_hist 是仪表运行时状态 — 分段续跑
        必须靠此侧车保持连续. 体外 E/de_mad 已随 step_E 删除,
        旧侧车中的残留键在 load_state 被忽略.
        """
        return {
            "step": int(step),
            "world_eps_ema": self.world_eps_ema,
            "world_eps_mad": self.world_eps_mad,
            "n_certified": self.n_certified,
            "hist": [sorted(s) for s in self._hist],
            "gen_temp": float(gen_temp),
            "k_history": self.k_history,
        }

    def load_state(self, st):
        """侧车恢复; 返回 (step, gen_temp) 供实验循环接管. 未知键忽略 (旧侧车兼容)."""
        if st.get("k_history", self.k_history) != self.k_history:
            raise ValueError(f"侧车 k_history={st.get('k_history')} ≠ 当前 "
                             f"{self.k_history} — 历史语义不一致, 拒绝恢复")
        self.world_eps_ema = (None if st.get("world_eps_ema") is None
                              else float(st["world_eps_ema"]))
        self.world_eps_mad = float(st.get("world_eps_mad", 0.0))
        self.n_certified = int(st.get("n_certified", 0))
        self._hist = deque((frozenset(int(v) for v in h) for h in st.get("hist", [])),
                           maxlen=self.k_history)
        return int(st.get("step", 0)), float(st.get("gen_temp", 4.0))
