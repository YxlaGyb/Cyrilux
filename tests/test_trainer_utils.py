
import numpy as np
import pytest
import torch

from pkg.utils.trainer_utils import Logger, effective_capacity, effective_params, get_lr, setup_seed


class TestGetLr:
    def test_warmup_linear(self):
        # warmup_ratio=0.1, total=100 → warmup_steps=10
        lr = get_lr(0, 100, 0.01, warmup_ratio=0.1)
        assert lr == pytest.approx(0.0)
        lr = get_lr(5, 100, 0.01, warmup_ratio=0.1)
        assert lr == pytest.approx(0.01 * 0.5)
        lr = get_lr(10, 100, 0.01, warmup_ratio=0.1)
        assert lr == pytest.approx(0.01)

    def test_cosine_start(self):
        # step == warmup_steps 边界：走 cosine 分支, progress=0 → 1.0 * lr
        lr = get_lr(10, 100, 0.01, warmup_ratio=0.1)
        assert lr == pytest.approx(0.01)

    def test_cosine_midpoint(self):
        # progress=0.5 → lr * (0.1 + 0.45) = 0.55 * lr
        lr = get_lr(55, 100, 0.01, warmup_ratio=0.1)
        assert lr == pytest.approx(0.01 * 0.55)

    def test_cosine_end(self):
        # progress=1.0 → lr * 0.1
        lr = get_lr(100, 100, 0.01, warmup_ratio=0.1)
        assert lr == pytest.approx(0.01 * 0.1)

    def test_total_steps_one(self):
        # 除零保护：total=1 时 warmup_steps=0 → 走 cosine, 不崩溃
        lr = get_lr(1, 1, 0.01, warmup_ratio=0.1)
        assert lr == pytest.approx(0.01 * 0.1)

    def test_monotonic_after_warmup(self):
        # cosine 段单调不增
        prev = None
        for step in range(10, 101, 5):
            lr = get_lr(step, 100, 0.01, warmup_ratio=0.1)
            if prev is not None:
                assert lr <= prev + 1e-12
            prev = lr


class TestSetupSeed:
    def test_reproducible(self):
        setup_seed(123)
        a = np.random.rand(4)
        b = torch.rand(4)
        setup_seed(123)
        assert np.allclose(a, np.random.rand(4))
        assert torch.equal(b, torch.rand(4))


class TestEffectiveParams:
    def test_zero_tensor(self):
        assert effective_params(torch.zeros(10)) == 0

    def test_all_above_threshold(self):
        w = torch.ones(10)
        assert effective_params(w) == 10

    def test_small_values_below_threshold(self):
        w = torch.tensor([1.0, 0.5, 1e-6, 0.0])
        # max=1.0, 阈值=1e-4 → 只有 1.0 和 0.5 计入
        assert effective_params(w) == 2

    def test_eps_zero_matches_nonzero_count(self):
        w = torch.tensor([1.0, 0.0, -2.0, 0.0])
        assert effective_params(w, eps=0.0) == 2


class TestEffectiveCapacity:
    def test_simple_module(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.ones(2, 3))

        m = M()
        assert effective_capacity(m) == pytest.approx(6 / 1e6)

    def test_requires_grad_false_skipped(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.ones(2, 3), requires_grad=False)

        m = M()
        assert effective_capacity(m) == 0.0


class TestLogger:
    def test_writes_stdout_and_file(self, tmp_path):
        log_path = tmp_path / "log.txt"
        logger = Logger(str(log_path))
        try:
            logger.write("hello\n")
            logger.flush()
        finally:
            logger.close()
        assert log_path.read_text(encoding="utf-8") == "hello\n"

    def test_append_mode(self, tmp_path):
        log_path = tmp_path / "log.txt"
        log_path.write_text("first\n", encoding="utf-8")
        logger = Logger(str(log_path), mode="a")
        try:
            logger.write("second\n")
        finally:
            logger.close()
        assert log_path.read_text(encoding="utf-8") == "first\nsecond\n"

    def test_no_path_stdout_only(self, capsys):
        logger = Logger(path=None)
        try:
            logger.write("to stdout\n")
        finally:
            logger.close()
        assert capsys.readouterr().out == "to stdout\n"
