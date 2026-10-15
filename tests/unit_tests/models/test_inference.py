import pytest
import torch

from models.inference import resolve_device, select_greedy_q_action


def test_greedy_selection_ignores_illegal_values_and_breaks_ties_by_index():
    output = torch.tensor([[float("nan"), -2.0, -2.0]])
    assert select_greedy_q_action(output, [2, 1]) == 1


@pytest.mark.parametrize("output,legal", [
    (torch.ones(3), [0]),
    (torch.ones(2, 3), [0]),
    (torch.ones(1, 3, dtype=torch.int64), [0]),
    (torch.ones(1, 3), []),
    (torch.ones(1, 3), [-1]),
    (torch.ones(1, 3), [3]),
    (torch.ones(1, 3), [True]),
    (torch.tensor([[float("inf"), 1.0]]), [0]),
])
def test_greedy_selection_rejects_invalid_output_or_actions(output, legal):
    with pytest.raises(ValueError):
        select_greedy_q_action(output, legal)


@pytest.mark.parametrize("cuda_available,expected", [(True, "cuda"), (False, "cpu")])
def test_auto_device_uses_available_hardware(monkeypatch, cuda_available, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    assert resolve_device("auto") == torch.device(expected)


@pytest.mark.parametrize("device", ["cpu", "cuda:1", torch.device("cpu")])
def test_explicit_device_does_not_probe_or_override_hardware(monkeypatch, device):
    def unexpected_probe():
        pytest.fail("Explicit device selection should not probe CUDA")
    monkeypatch.setattr(torch.cuda, "is_available", unexpected_probe)
    assert resolve_device(device) == torch.device(device)
