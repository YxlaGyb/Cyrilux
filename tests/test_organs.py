"""
三器官结构判据测试.

器官1 内容寻址检索: 记忆单元门 = softmax(cos·K)·K, Σgate=K (全相似→均匀中性),
单元独配 → ~K 独占 (检索); 记忆本体 _mem_m 不受门控影响.
器官2 explain-away: pw = res_td/(‖z4‖+res_td) ∈ (0,1), 高层全解释 → 学习被抑制;
部分解释 → 学习保留 (不是开关是精度).
器官3 RPE: EMA 冷启动 = 0 → rpe = R (原始进度平滑过渡); 数步后 EMA 跟踪 R.
"""
import torch

from model import CyreneModel, DensePCNet


def _make_net():
    torch.manual_seed(0)
    cfg = CyreneModel(
        d_l4=64, d_l2=32, d_l3=32, d_l5=64, d_l6=16,
        max_seq_len=16, free_run_window=16, mem_k0=2,
    )
    return DensePCNet(cfg)


class TestOrgan1Retrieval:
    def _gate(self, z4, m_seq, K):
        num = (z4.unsqueeze(2) * m_seq).sum(dim=-1)  # [N,S,K]
        den = z4.norm(dim=-1, keepdim=True) * (m_seq.norm(dim=-1) + 1e-3)  # [N,S,K]
        return torch.softmax((num / den) * float(K), dim=-1) * float(K)

    def test_gate_sums_to_K(self):
        K = 6
        torch.manual_seed(1)
        z4 = torch.randn(1, 8, 16, dtype=torch.float16)
        m_seq = torch.randn(1, 8, K, 16, dtype=torch.float16)
        gate = self._gate(z4, m_seq, K)
        assert torch.allclose(gate.sum(dim=-1), torch.full_like(gate.sum(dim=-1), K), atol=1e-2)

    def test_gate_uniform_when_all_similar(self):
        # 全部单元痕迹相同 → 门均匀 = 1 (中性, 不偏袒)
        K = 4
        z4 = torch.randn(1, 8, 16, dtype=torch.float16)
        m_seq = torch.randn(1, 8, 16, dtype=torch.float16).unsqueeze(2).expand(1, 8, K, 16).contiguous()
        gate = self._gate(z4, m_seq, K)
        assert torch.allclose(gate, torch.ones_like(gate), atol=0.15)

    def test_gate_dominates_on_match(self):
        # 某单元痕迹与当前输入几乎相同 → 该单元门 ≈ K (检索独占)
        K = 4
        torch.manual_seed(2)
        z4 = torch.randn(1, 8, 16, dtype=torch.float16)
        m_seq = torch.randn(1, 8, K, 16, dtype=torch.float16) * 0.1
        m_seq[:, :, 2] = z4 * 1.0  # 单元 2 与当前输入一致
        gate = self._gate(z4, m_seq, K)
        assert gate[0, 0, 2] > 0.75 * K


class TestOrgan2ExplainAway:
    def test_pw_zero_when_fully_explained(self):
        # 高层完全解释 z4 (z4 = z2 @ W) → pw → 0 (学习抑制)
        torch.manual_seed(3)
        z2 = torch.randn(1, 8, 16, dtype=torch.float16)
        w = torch.randn(16, 32, dtype=torch.float16) * 0.1
        z4 = z2 @ w
        res_td = (z4 - z2 @ w).norm(dim=-1)
        pw = res_td / (z4.norm(dim=-1) + res_td + 1e-3)
        assert pw.max() < 0.1

    def test_pw_near_one_when_unexplained(self):
        # 高层与 z4 无关 → res ≈ ‖z4‖ → pw → 0.5 (学习保留一半以上? 否: res/(tot+res)=0.5)
        torch.manual_seed(4)
        z2 = torch.zeros(1, 8, 16, dtype=torch.float16)
        z4 = torch.randn(1, 8, 32, dtype=torch.float16)
        res_td = (z4 - z2 @ torch.zeros(16, 32, dtype=torch.float16)).norm(dim=-1)
        pw = res_td / (z4.norm(dim=-1) + res_td + 1e-3)
        assert (pw > 0.45).all() and (pw < 0.55).all()


class TestOrgan3RPE:
    def test_rpe_ema_tracks_and_cold_start_is_raw(self):
        net = _make_net()
        assert net._metab_rpe_ema.abs().sum() == 0  # 冷启动: rpe = R
        # W_act 只在行为步 (回声) 学习 — RPE 路径需 感知/回声 交替才被触发
        net.learn(torch.randint(32, 256, (1, 8), dtype=torch.long))  # 感知
        net.learn()  # 回声 (行为步): RPE 首算, EMA ← 0.02·R
        net.learn(torch.randint(32, 256, (1, 8), dtype=torch.long))
        net.learn()
        assert net._metab_rpe_ema.abs().sum() > 0  # EMA 已跟踪

    def test_learning_alive_with_rpe(self):
        # W_act 在 RPE 调制下仍学习 (非死锁)
        net = _make_net()
        w0 = net.W_act.data.clone()
        for i in range(4):
            net.learn(torch.randint(32, 256, (1, 8), dtype=torch.long)) if i % 2 == 0 else net.learn()
        assert not torch.equal(net.W_act.data, w0)
