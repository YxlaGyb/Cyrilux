"""跨 save/load 的模型状态持久化.

判据不是"键存在", 而是**续跑与未中断的连续运行逐位等价**, 且必须覆盖全部路径
(感知 / 回声 / 自由运行) 与全部状态 (state_dict + 游离张量 + Python 计数器).

旧测试只跑感知路径、只比 7 个键 —— 正是这个漏洞让 _active_ema_init 未持久化
和 26 个游离张量长期漏网.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from model import CyreneModel, DensePCNet
from model.dense.pruning import ROW_STATE, PruningEngine

# 首维命中层尺寸但按槽位/位置索引的 (bind 维恰好等于 d_l2, 故需显式豁免)
_AUDIT_ALLOWLIST = (
    "E_bind_col", "E_bind_self", "W_act", "W_bind_self", "_osc_f_tab", "_seq_arange",
    "_stp_active_ema_bind", "_stp_r_bind", "_stp_tau_bind", "_stp_u_bind", "_theta_bind",
)


def _cfg(**over) -> CyreneModel:
    kw = {
        "d_l4": 64, "d_l2": 32, "d_l3": 32, "d_l5": 64, "d_l6": 16,
        "max_seq_len": 32, "free_run_window": 16, "mem_k0": 2, "mem_k_max": 2,
    }
    kw.update(over)
    return CyreneModel(**kw)


BATCH = torch.randint(0, 255, (1, 16), dtype=torch.long)


def _step(net: DensePCNet, i: int) -> None:
    """三条路径轮换: 感知 / 回声 / 自由运行. 每步固定种子 → 两侧随机流一致."""
    torch.manual_seed(11 + i)
    if i % 3 == 2:
        net.learn(BATCH, free_run=True)
    else:
        net._behavior_py = (i % 3 == 1)
        net.learn(BATCH)


def _assert_same_state(a: DensePCNet, b: DensePCNet) -> None:
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb)
    assert [k for k in sa if not torch.equal(sa[k], sb[k])] == []
    ea, eb = a._extra_tensors(), b._extra_tensors()
    assert set(ea) == set(eb)
    assert [k for k in ea if not torch.equal(ea[k], eb[k])] == []
    assert a._py_counters() == b._py_counters()


def test_full_state_roundtrips(tmp_path):
    """参数 + buffer + 游离张量 + 计数器 (_stp_r_end 展开项在内) 全部逐位往返."""
    net = DensePCNet(_cfg())
    for i in range(24):
        _step(net, i)
    p = tmp_path / "ck.safetensors"
    net.save(str(p))
    net2 = DensePCNet.load(str(p), _cfg())

    _assert_same_state(net, net2)
    assert net._extra_tensors(), "训练 24 步后应有游离张量"
    assert net._stp_r_end, "free_run 后应有 STP 资源写回"
    # 别名不入游离集: 否则 load 后会变成两个不同张量, 破坏别名语义
    assert net._metab_stress is net._metab_famine_prog
    assert "_metab_stress" not in net._extra_tensors()


def test_resume_equals_uninterrupted(tmp_path):
    """全路径 + 全状态续跑等值 —— 本轮的真正判据."""
    net = DensePCNet(_cfg())
    for i in range(24):
        _step(net, i)
    p = tmp_path / "ck.safetensors"
    net.save(str(p))
    net2 = DensePCNet.load(str(p), _cfg())

    for i in range(24, 33):
        _step(net, i)
        _step(net2, i)
        _assert_same_state(net, net2)


def test_extra_tensors_follow_device(tmp_path):
    """游离张量随 .to() 迁移 (nn.Module._apply 只搬参数与 buffer)."""
    net = DensePCNet(_cfg())
    for i in range(3):
        _step(net, i)
    p = tmp_path / "ck.safetensors"
    net.save(str(p))
    net2 = DensePCNet.load(str(p), _cfg()).to("cpu")
    assert {v.device.type for v in net2._extra_tensors().values()} == {"cpu"}


def test_weights_only_snapshot_refused(tmp_path):
    """无 state_ver → 拒收 (不留「缺键即默认」的后门)."""
    net = DensePCNet(_cfg())
    p = tmp_path / "weights_only.safetensors"
    save_file({k: v.contiguous() for k, v in net.state_dict().items()}, str(p))
    with pytest.raises(ValueError, match="weights-only"):
        DensePCNet.load(str(p))


def test_pruned_checkpoint_roundtrips(tmp_path):
    """修剪后检查点按形状重建: active_size 与 _mem_m 自洽, 续跑不崩."""
    cfg = _cfg(
        d_l4=600, d_l2=160, d_l3=160, d_l5=160, d_l6=160,
        prune_warmup=0, prune_fraction=0.6, death_probation=1, mem_k0=1, mem_k_max=1,
    )
    net = DensePCNet(cfg)
    for i in range(6):
        _step(net, i)
    net._prune()
    net._prune()
    assert net.active_size["l4"] < 600, "本测试需要真实收缩"

    p = tmp_path / "pruned.safetensors"
    net.save(str(p))
    net2 = DensePCNet.load(str(p))  # 不传 cfg: 维度以检查点形状为准
    assert net2.active_size["l4"] == net.active_size["l4"]
    assert net2._mem_m.shape[1] == net2.active_size["l4"]
    for i in range(6, 12):
        _step(net2, i)


def test_load_dims_follow_checkpoint(tmp_path):
    """修剪后检查点 + 出生配置: 维度以形状为准, 非维度字段仍取 config.

    修复前 active_size 取 config 的出生 dims (600) 而 _mem_m 取检查点 (512) → free_run 崩.
    """
    kw = {"d_l4": 600, "d_l2": 160, "d_l3": 160, "d_l5": 160, "d_l6": 160,
          "prune_warmup": 0, "prune_fraction": 0.6, "death_probation": 1,
          "mem_k0": 1, "mem_k_max": 1}
    net = DensePCNet(_cfg(**kw))
    for i in range(6):
        _step(net, i)
    net._prune()
    net._prune()
    assert net.active_size["l4"] < 600, "本测试需要真实收缩"

    p = tmp_path / "pruned.safetensors"
    net.save(str(p))
    loaded = DensePCNet.load(str(p), _cfg(**kw, lm_freeze_w1=True, echo_seed_n=99))
    assert loaded.cfg.d_l4 == net.active_size["l4"]
    assert loaded._mem_m.shape[1] == loaded.active_size["l4"]
    assert loaded.cfg.lm_freeze_w1 is True  # 非维度字段未被形状覆盖
    assert loaded.cfg.echo_seed_n == 99
    _step(loaded, 8)  # 8 % 3 == 2 → free_run, 修复前此处崩


def test_prune_syncs_layer_row_state():
    """ROW_STATE 每个 aux 项都须 perm 重排 + 收缩.

    覆盖度由表保证, 不由测试手挑名字 —— 手挑正是 b_diff / E_42 / 各资格迹长期漏网的原因.
    探针: 张量值 = 行号, 修剪后比对该层真实 perm (只比首列: 行签名沿列恒定, 且 row 形态的
    列会被既有列同步改宽).
    """
    net = DensePCNet(
        _cfg(d_l4=600, d_l2=160, d_l3=160, d_l5=160, d_l6=160, prune_warmup=0,
             prune_fraction=0.6, death_probation=1, mem_k0=1, mem_k_max=1)
    )
    for i in range(6):
        _step(net, i)
    net.learn(BATCH, free_run=True)  # 填充 _stp_r_end

    aux = {n: (lyr, form) for n, (lyr, form, h) in ROW_STATE.items() if h == "aux" and hasattr(net, n)}
    assert aux, "ROW_STATE 应有 aux 项"
    sig = {}
    for name in aux:
        t = getattr(net, name)
        idx = torch.arange(t.shape[0], dtype=torch.float16)
        s = idx.clone() if t.dim() == 1 else idx.reshape([t.shape[0]] + [1] * (t.dim() - 1)).expand(t.shape).contiguous()
        sig[name] = s
        t.data.copy_(s)

    perms = _prune_capture_perms(net)

    def col0(t):
        return t if t.dim() == 1 else t[:, 0]

    for name, (lyr, _form) in aux.items():
        want = net.active_size[lyr]
        perm = perms.get(lyr)
        cur = getattr(net, name)
        assert cur.shape[0] == want, f"{name} 未收缩"
        exp = col0((sig[name][perm] if perm is not None else sig[name])[:want])
        assert torch.equal(col0(cur[:want]).float(), exp.float()), f"{name} 未随 perm 重排"
    for k, v in net._stp_r_end.items():
        if k in net.active_size:
            assert v.shape[0] == net.active_size[k], f"_stp_r_end[{k}] 未收缩"


def test_prune_row_state_audit():
    """首维命中层尺寸的张量必须归入 ROW_STATE 或明确豁免 —— 新增状态不许静默漏."""
    net = DensePCNet(_cfg())
    dims = set(net.cfg.dims().values())
    allt = dict(net.named_parameters())
    allt.update(dict(net.named_buffers()))
    unclassified = [
        n for n, t in allt.items()
        if t.dim() >= 1 and t.shape[0] in dims and n not in ROW_STATE and n not in _AUDIT_ALLOWLIST
    ]
    assert unclassified == [], f"未分类的行索引状态: {unclassified}"


def _prune_capture_perms(net: DensePCNet) -> dict[str, torch.Tensor]:
    perms: dict[str, torch.Tensor] = {}
    orig = PruningEngine._permute_weights

    def _wrap(self, layer, *a, **kw):
        perm, n_alive = orig(self, layer, *a, **kw)
        if perm is not None:
            perms[layer] = perm.clone()
        return perm, n_alive

    PruningEngine._permute_weights = _wrap
    try:
        net._prune()
    finally:
        PruningEngine._permute_weights = orig
    return perms


def test_prune_keeps_state_attached_to_its_neuron():
    """端到端: 训练得到的真实状态值必须跟着自己的神经元走.

    判据: 修剪后的 b_diff 须等于修剪前的 b_diff 经该层 perm 重排.
    修复前 b_diff 不进 perm, 会整体错配到别的神经元上.
    """
    net = DensePCNet(
        _cfg(d_l4=600, d_l2=160, d_l3=160, d_l5=160, d_l6=160, prune_warmup=0,
             prune_fraction=0.6, death_probation=1, mem_k0=1, mem_k_max=1)
    )
    for i in range(6):
        _step(net, i)
    a4 = net.active_size["l4"]
    b_before = net.b_diff.detach().clone()

    perms = _prune_capture_perms(net)

    a4_new = net.active_size["l4"]
    assert a4_new < a4, "本测试需要真实收缩"
    assert torch.equal(net.b_diff[:a4_new], b_before[perms["l4"]][:a4_new]), "b_diff 未随 perm 重排"


def test_prune_then_train_stays_consistent():
    """修剪后继续训练: 各形态须与 active_size 自洽且无 NaN."""
    net = DensePCNet(
        _cfg(d_l4=600, d_l2=160, d_l3=160, d_l5=160, d_l6=160, prune_warmup=0,
             prune_fraction=0.6, death_probation=1, mem_k0=1, mem_k_max=1)
    )
    for i in range(6):
        _step(net, i)
    net._prune()
    for i in range(6, 12):
        _step(net, i)
    a4 = net.active_size["l4"]
    assert net.b_diff.shape[0] == a4
    assert net._mem_m.shape[1] == a4
    assert net.W_diff.shape == (a4, a4)
    assert net.W_04.shape[0] == a4
    assert all(torch.isfinite(v).all() for v in net.state_dict().values() if v.is_floating_point())


def test_consecutive_prunes_stay_aligned():
    """连续多次修剪: 每轮 perm 独立, 状态须逐轮跟随且始终与 active_size 自洽."""
    net = DensePCNet(
        _cfg(d_l4=600, d_l2=160, d_l3=160, d_l5=160, d_l6=160, prune_warmup=0,
             prune_fraction=0.6, death_probation=1, mem_k0=1, mem_k_max=1)
    )
    for i in range(6):
        _step(net, i)
    for r in range(3):
        prev = net.b_diff.detach().clone()
        perms = _prune_capture_perms(net)
        a4 = net.active_size["l4"]
        if "l4" in perms:
            assert torch.equal(net.b_diff[:a4], prev[perms["l4"]][:a4]), f"第 {r + 1} 轮 b_diff 未跟随"
        assert net.b_diff.shape[0] == a4
        assert net.W_diff.shape == (a4, a4)
        assert net._mem_m.shape[1] == a4
        _step(net, 10 + r)


def test_prune_between_free_run_calls():
    """修剪落在 free_run 窗口之间: _stp_r_end 跨调用存活, 须随 perm 且形状自洽."""
    net = DensePCNet(
        _cfg(d_l4=600, d_l2=160, d_l3=160, d_l5=160, d_l6=160, prune_warmup=0,
             prune_fraction=0.6, death_probation=1, mem_k0=1, mem_k_max=1)
    )
    for i in range(4):
        _step(net, i)
    net.learn(BATCH, free_run=True)
    assert net._stp_r_end, "free_run 应写入 STP 资源"
    before = {k: v.clone() for k, v in net._stp_r_end.items()}

    perms = _prune_capture_perms(net)

    for k, v in net._stp_r_end.items():
        if k not in net.active_size:
            continue
        assert v.shape[0] == net.active_size[k], f"_stp_r_end[{k}] 未收缩"
        if k in perms:
            assert torch.equal(v, before[k][perms[k]][: net.active_size[k]]), f"_stp_r_end[{k}] 未随 perm"
    net.learn(BATCH, free_run=True)  # 修剪后立刻 free_run 不崩
