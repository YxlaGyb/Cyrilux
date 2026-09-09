"""世界语言物理 — 环境统计结构 (第 111 轮).

世界的物理 = 语料的统计结构 (dataset/pretrain_t2t_mini.jsonl). 世界不设定
目标, 只设定代谢条件: 能量只流向符合世界统计结构的发声 — 如阳光波长
决定叶绿素吸收峰, 无人告诉植物该怎么长.

评分维度 (全部预计算自语料, 训练中冻结只读):
  L 覆盖率  CJK 字符 (ord>127) ∈ top-1000 常用字集占比 — 与
            audit_char_coverage 同口径 (判据线可比); probe111 实测分离
            10.6x (语料 0.950 / 汤 0.089), 主判据
  S 结构    字节 3-gram 命中语料 3-gram 位图占比 (辅助; probe111 实测
            汤 0.639 / 语料 1.0, 分离 1.6x)
  X 新颖    与最近 K 次自身发声 3-gram 并集重叠率 (D4 单调税, 复读惩罚)
  q         (0.8·L + 0.2·S)·(1 - X)   权重 = 实测判别力之比 (10.6:1.6)

E 代谢 (环境物理常数, 类比重力 g — 环境参数非学习规则):
  E ← E·(1-d) + c·q        d=0.05 消耗 / c=0.1 进食
  R_world = tanh(ΔE/(2·MAD_de)) / ‖W_act_elig‖   (113 R1 双侧自校准增益;
           旧 0.05·tanh(ΔE/0.02) std=0.016, 距资格迹契约 0-1 级 ~19x,
           112 实测学习腿饿死; 迹范数由调用方传入, 公式单点存放于此)
  平衡点 E* = c·q/d: q=0.9 → 1.8 (饱足), q=0.13 → 0.26 (汤=濒死)

认证门: q ≥ cert_frac·q_base → 发声 ε 进认证带锚 (恒温器新目标,
111 拆解 ε 代理失效: 认证门用可分维度 q, ε 只做温度方向传感器).

全部统计 CPU Python (非张量热路径); R/E 入 torch 时 fp16.
常用字统计口径与 audit_char_coverage 一致 (前 2 万行字符 Counter).
"""
from __future__ import annotations

import json
import math
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
    """世界语言物理: 预计算统计 + 评分 + E 代谢 + 认证带锚."""

    def __init__(self, data_path, n_char_lines=20000, n_trigram_lines=None,
                 top_n=1000, k_history=32, d=0.05, c=0.1,
                 cert_frac=0.5, E0=1.0, ema_alpha=0.995, mad_alpha=0.95):
        # 常用字集 (audit_char_coverage 同口径)
        cnt = Counter()
        with open(data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n_char_lines:
                    break
                cnt.update(json.loads(line)["text"])
        self.common_set = set(c for c, _ in cnt.most_common(top_n))

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

        # 代谢状态 (先于基线评分 — score 只读依赖 _hist)
        self.E = E0
        self.d = d
        self.c = c
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

        # 认证带锚 (E 认证的发声 ε 统计)
        self.world_eps_ema = None
        self.world_eps_mad = 0.0
        self.ema_alpha = ema_alpha
        self.mad_alpha = mad_alpha
        self.n_certified = 0
        # R1 信号侧: 代谢流水 ΔE 的涨落尺度 (MAD EMA); 冷启动首笔 = |ΔE|
        self.de_mad = None

    def score(self, gen_bytes):
        """发声字节流 → {L, S, X, q}. 只读 (不更新历史).

        L 口径与 audit_char_coverage 一致 (CJK 字符的 top-1000 常用占比,
        判据线可比); q 按维度实测判别力加权: L 分离 7x, S 分离 1.6x
        (probe111: S 对 topk-15 采样汤判别力弱, 汤的 S=0.64) → 0.8/0.2.
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

    def step_E(self, q, trace_norm):
        """E 代谢: 消耗 d + 进食 c·q. 返回 R_world (113 R1 双侧自校准增益).

        R = tanh(ΔE/(2·MAD)) / ‖迹‖: 信号侧除以世界自己的涨落 (de_mad EMA,
        复用 mad_alpha=0.95, 半衰期 13.5 步 = 资格迹/E 同尺度), 消费侧除以
        调用方传入的资格迹范数 (系统自己近期行为的尺度). 写入 = (迹/‖迹‖)·tanh
        → 学习腿写入幅度结构性 ≈ |tanh| ≤ 1 (基线项单位范数口径).

        探针裁决 (out/probe113_ab.json + out/probe113_ab_r1.json):
        旧 R std=0.016 距契约 ~19x → 学习腿饿死 (112: W_act 2000 步不动);
        B2 (仅除以 MAD) std=0.46 过冲 28x → 汤崩溃 + E 濒死 0.036 (D3 败);
        R1 ratio_med=0.39 入目标区 [0.3,1], 零危机, D1'/D2'/D3' 全过.
        迹膨胀 → R 自动收缩 — 对 B2 崩溃模式的内建负反馈. 零新增定数.
        """
        e_old = self.E
        self.E = self.E * (1.0 - self.d) + self.c * q
        de = self.E - e_old
        if self.de_mad is None:
            self.de_mad = abs(de)
        else:
            self.de_mad = (self.mad_alpha * self.de_mad
                           + (1.0 - self.mad_alpha) * abs(de))
        return (math.tanh(de / (2.0 * max(self.de_mad, 1e-6)))
                / max(trace_norm, 1e-6))

    def certify(self, q, eps_value):
        """世界认证门: q ≥ q_ref → 该发声 ε 进认证带锚. 返回是否认证."""
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
        """世界动态状态 + 恒温器温度 → 可 JSON 侧车 (111 续跑三重丢失修复).

        net.save() 只存权重; E/锚/τ/_hist 是运行时状态 — 长暴露分段续跑
        必须靠此侧车保持连续 (exp112_say.py 每 500 步与 net.save 同步写).
        """
        return {
            "step": int(step),
            "E": self.E,
            "de_mad": self.de_mad,
            "world_eps_ema": self.world_eps_ema,
            "world_eps_mad": self.world_eps_mad,
            "n_certified": self.n_certified,
            "hist": [sorted(s) for s in self._hist],
            "gen_temp": float(gen_temp),
            "k_history": self.k_history,
        }

    def load_state(self, st):
        """侧车恢复; 返回 (step, gen_temp) 供实验循环接管."""
        if st.get("k_history", self.k_history) != self.k_history:
            raise ValueError(f"侧车 k_history={st.get('k_history')} ≠ 当前 "
                             f"{self.k_history} — 历史语义不一致, 拒绝恢复")
        self.E = float(st["E"])
        self.de_mad = (None if st.get("de_mad") is None
                       else float(st["de_mad"]))
        self.world_eps_ema = (None if st["world_eps_ema"] is None
                              else float(st["world_eps_ema"]))
        self.world_eps_mad = float(st["world_eps_mad"])
        self.n_certified = int(st["n_certified"])
        self._hist = deque((frozenset(int(v) for v in h) for h in st.get("hist", [])),
                           maxlen=self.k_history)
        return int(st.get("step", 0)), float(st.get("gen_temp", 4.0))
