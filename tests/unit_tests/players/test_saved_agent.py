import pytest
import torch
from torch import nn

from players.saved_agent import SavedAgentPlayer, select_greedy_q_action


class Adapter:
    def encode(self, game):
        return torch.tensor([game["features"]], dtype=torch.float32)

    def legal_actions(self, game):
        return game["legal"]

    def decode_action(self, game, index):
        return game["actions"][index]


class InferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3, bias=False)
        self.dropout = nn.Dropout(0.9)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(3))

    def forward(self, inputs):
        assert not self.training
        assert not torch.is_grad_enabled()
        assert inputs.device == self.linear.weight.device
        return self.dropout(self.linear(inputs))


def test_player_disables_dropout_and_gradients_and_maps_legal_action():
    model = InferenceModel()
    player = SavedAgentPlayer(model, Adapter(), device="cpu")
    game = {"features": [100.0, 2.0, 3.0], "legal": [1, 2], "actions": ["a", "b", "c"]}
    original = model.linear.weight.detach().clone()
    # A caller changing the mode must not enable dropout during player inference.
    model.train()
    assert [player(game) for _ in range(5)] == ["c"] * 5
    assert game["features"] == [100.0, 2.0, 3.0]
    assert torch.equal(model.linear.weight, original)
    assert model.linear.weight.grad is None


def test_player_supports_custom_output_selector():
    # This selector treats the output as action costs instead of Q-values.
    def lowest_cost(output, legal):
        return min(legal, key=lambda index: output[0, index].item())

    player = SavedAgentPlayer(InferenceModel(), Adapter(), lowest_cost, "cpu")
    game = {"features": [100.0, 2.0, 3.0], "legal": [1, 2], "actions": ["a", "b", "c"]}
    assert player(game) == "b"


def test_player_rejects_terminal_state_before_inference():
    player = SavedAgentPlayer(InferenceModel(), Adapter(), device="cpu")
    with pytest.raises(ValueError, match="without legal moves"):
        player({"legal": []})


def test_player_rejects_illegal_custom_selection():
    player = SavedAgentPlayer(InferenceModel(), Adapter(), lambda output, legal: 0, "cpu")
    game = {"features": [100.0, 2.0, 3.0], "legal": [1, 2], "actions": ["a", "b", "c"]}
    with pytest.raises(ValueError, match="legal output index"):
        player(game)


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
