"""RTR 判别对照实验 (spec §3 判据, 分布式表示版).

任务: 4 个 ±1 分布式模式共用一个权重矩阵 W[D, N_ACT]; 每个模式有固定目标动作,
初值对每个模式都接反. 模式之间在 W 的列空间里重叠 —— 这正是真实网络的形态,
也是"全局基线为什么不够"能显形的地方: 一个标量 m 无法区分是哪条模式导致的成败,
它的梯度在重叠的方向上互相抵消.

指标: 20 个随机种子中最终把 4 个模式全部接对 (argmax 命中目标动作) 的种子数.

五条对照:
  hebbian   m ≡ 1                     —— 只会把接错的线越加越粗
  global    m = r − V_global          —— 标量基线, 信用错配
  shuffled  m 的分布与 state 相同, 只打乱 m 与 e 的配对
  state     m = r − V[p]              —— 按模式各一个基线
  bucketed  m = r − V[hash(p)]        —— 有损状态键: K 个桶把多个模式并到一起

bucketed 是试点的形态: 试点没有"真实状态 p", 只有 z4 池化哈希出的桶号, 必然把多个
真实状态并进一个桶. 桶划分按种子随机抽取(与真实哈希同构), 不是整齐的 p mod K.
该臂证明状态键的分辨率是连续作用的量: K=1 逐位等价 global 臂, K=N_PAT 逐位等价
state 臂, 中间随机碰撞下单调有效(实测 5/20 → 6/20 → 9/20 → 20/20). 因此"K=64"不是
拍脑袋, 而是这条曲线上的一个点.

但碰撞方向可以反转结论: 若把最优目标相反的两个状态并进一个桶, 桶级基线就把桶内均值
当成了比较基准, 信用被抵消. 对抗性配对 p % 2 实测 4/20, **低于**全局标量的 5/20.
所以有损键不是无条件安全 —— 这是试点必须实测桶占用的原因.

判别量是 m 与 e 的相关, 不是 m 的幅度: shuffled 一行保持同一分布、同一幅度,
唯一变化是配对, 结果归零.

本文件是独立判据实验, 不依赖也不改动任何生产代码.
"""

import numpy as np

SEEDS = 20
N_PAT = 4
N_ACT = 4
DIM = 16
SUCCESS_P = (0.9, 0.7, 0.4, 0.2)
STEPS = 12000
LAMBDA = 0.3
ETA = 0.3
ETA_V = 0.05
M_BUF = 512

HEBBIAN, GLOBAL, SHUFFLED, STATE, BUCKETED = range(5)
ARM_NAMES = ("hebbian", "global", "shuffled", "state", "bucketed")


def _problem(seed):
    rng = np.random.default_rng(seed)
    x = np.where(rng.random((N_PAT, DIM)) < 0.5, -1.0, 1.0).astype(np.float32)
    target = np.arange(N_PAT)
    w = np.zeros((DIM, N_ACT), dtype=np.float32)
    for p in range(N_PAT):
        w[:, (target[p] + 1) % N_ACT] += x[p]
        w[:, target[p]] -= x[p]
    w *= 1.0 / np.sqrt(DIM)
    w += 0.02 * rng.standard_normal((DIM, N_ACT)).astype(np.float32)
    return x, target, w


def _bucket_map(seed, n_buckets):
    if n_buckets <= 1:
        return np.zeros(N_PAT, dtype=np.int64)
    if n_buckets >= N_PAT:
        return np.arange(N_PAT, dtype=np.int64)
    return np.random.default_rng(seed + 500_000).integers(n_buckets, size=N_PAT)


def _rollout(arm, seed, n_buckets=0, pair=None):
    rng = np.random.default_rng(seed)
    m_rng = np.random.default_rng(seed + 1_000_000)
    x, target, w = _problem(seed)
    p_hit = np.asarray(SUCCESS_P, dtype=np.float64)
    eye = np.eye(N_ACT, dtype=np.float32)
    bmap = pair if pair is not None else _bucket_map(seed, n_buckets)
    if callable(bmap):
        bmap = bmap(np.arange(N_PAT))
    bmap = np.asarray(bmap, dtype=np.int64)

    e = np.zeros((DIM, N_ACT), dtype=np.float32)
    v_key = np.zeros(N_PAT, dtype=np.float64)
    v_global = 0.0
    m_buf = np.zeros(M_BUF, dtype=np.float64)
    buf_n = 0

    for _ in range(STEPS):
        p = int(rng.integers(N_PAT))
        logits = x[p] @ w
        logits -= logits.max()
        prob = np.exp(logits)
        prob /= prob.sum()
        a = int(np.searchsorted(np.cumsum(prob), rng.random()))

        r = 1.0 if (a == target[p] and rng.random() < p_hit[p]) else -1.0

        e *= LAMBDA
        e += (1.0 - LAMBDA) * np.outer(x[p], eye[a])

        if arm == HEBBIAN:
            m = 1.0
        elif arm == GLOBAL:
            m = r - v_global
            v_global += ETA_V * m
        elif arm == BUCKETED:
            key = int(bmap[p])
            m = r - float(v_key[key])
            v_key[key] += ETA_V * m
        else:
            m = r - float(v_key[p])
            v_key[p] += ETA_V * m
            if arm == SHUFFLED:
                m = m_buf[m_rng.integers(min(buf_n, M_BUF) or 1)]

        if arm in (GLOBAL, STATE, BUCKETED, SHUFFLED):
            m_buf[buf_n % M_BUF] = m
            buf_n += 1

        w += (ETA * m) * e

    return bool((np.argmax(x @ w, axis=1) == target).all())


_CACHE = {}


def _learned(arm, n_buckets=0):
    key = (arm, n_buckets)
    if key not in _CACHE:
        _CACHE[key] = sum(_rollout(arm, s, n_buckets) for s in range(SEEDS))
    return _CACHE[key]


def test_hebbian_never_escapes_wrong_init():
    # m ≡ 1: 共现即强化, 没有"差"这个概念, 接反的线只会越加越粗.
    assert _learned(HEBBIAN) == 0


def test_global_baseline_barely_escapes_wrong_init():
    # 单一标量基线: 判得出这一轮好/坏, 判不出是哪条重叠模式导致的. 实测 3-5/20,
    # 远低于 state, 且不随步数增长.
    assert _learned(GLOBAL) <= SEEDS // 2


def test_shuffled_pairing_never_escapes_wrong_init():
    # m 的分布、幅度、更新次数与 state 完全相同, 唯一变化是 m 与 e 的配对被打乱.
    # 结果归零 —— 增益住在相关里, 不在幅度里.
    assert _learned(SHUFFLED) == 0


def test_per_state_baseline_learns_every_seed():
    assert _learned(STATE) == SEEDS


def test_unit_and_full_resolution_reduce_to_known_arms():
    # 恒等式, 非统计断言: K=1 时全部模式共用唯一桶 → 逐位等价 global 臂;
    # K=N_PAT 时每个模式独占一桶 → 逐位等价 state 臂. 分辨率是连续旋钮, 两端接住已知臂.
    assert _learned(BUCKETED, 1) == _learned(GLOBAL)
    assert _learned(BUCKETED, N_PAT) == _learned(STATE)


def test_lossy_state_key_still_beats_global():
    # 试点拿不到真实状态, 只有 z4 池化哈希出的桶号, 必然把多个真实状态并进一个桶.
    # 随机碰撞下分辨率单调有效: K=1:5/20, K=2:6/20, K=3:9/20, K=4:20/20.
    # K=2 只多一个种子, 故只断言不劣于全局标量; 真正稳的增益要碰撞基本消失才拿到.
    assert _learned(BUCKETED, 2) >= _learned(GLOBAL)
    assert _learned(BUCKETED, 3) > _learned(GLOBAL)


def test_collision_alignment_is_the_real_risk():
    # 有损键不是无条件安全. 若碰撞把最优目标相反的两个状态并进一个桶, 桶级基线就把
    # 桶内均值当成了比较基准, 信用被抵消: 对抗性配对 p % 2 实测 4/20, 低于全局的 5/20.
    # 这正是试点 K=64 桶的风险来源 —— 探针实测 120 窗只占 10 组, 有效碰撞远重于均匀哈希.
    adversarial = sum(_rollout(BUCKETED, s, 2, lambda p: p % 2) for s in range(SEEDS))
    assert adversarial < _learned(GLOBAL)


if __name__ == "__main__":
    print(", ".join(f"{n}={_learned(a)}/{SEEDS}" for a, n in enumerate(ARM_NAMES[:4])))
    print("bucketed:", ", ".join(f"K={k}:{_learned(BUCKETED, k)}/{SEEDS}" for k in (1, 2, 3, 4)))
    for m in (2, 3):
        adv = sum(_rollout(BUCKETED, s, 2, lambda p: p % m) for s in range(SEEDS))
        print(f"adversarial p%{m}: {adv}/{SEEDS}")
