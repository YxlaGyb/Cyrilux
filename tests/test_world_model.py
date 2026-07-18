import torch

from continual.memory_bank import MemoryBank
from continual.world_model import LatentWorldModel


def test_world_model_forward_and_loss():
    model = LatentWorldModel(input_dim=8, hidden_dim=16, context_dim=2)
    state = torch.randn(2, 4, 8)
    target = torch.randn(2, 4, 8)
    ctx = torch.tensor([[0.2, 0.7], [0.8, 0.1]], dtype=torch.float32)
    pred, uncertainty = model(state, ctx)
    assert pred.shape == state.shape
    assert uncertainty.shape == (state.size(0), 1)
    loss = model.loss(state, target, ctx)
    assert torch.isfinite(loss)


def test_world_model_replay_priority_roundtrip():
    bank = MemoryBank(max_per_task=4)
    sample = (torch.arange(8, dtype=torch.uint8), torch.arange(8, dtype=torch.long))
    bank.add_samples(
        'task_a',
        [sample],
        dopamine_score=0.8,
        baseline_loss=0.1,
        transition_surprise=0.4,
    )

    state = bank.state_dict()
    restored = MemoryBank(max_per_task=4)
    restored.load_state_dict(state)
    replay = restored.sample(1, strategy='world_model')
    assert replay
    assert replay[0].replay_priority > 0.0
