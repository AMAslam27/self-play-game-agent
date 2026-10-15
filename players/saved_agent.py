"""Reusable inference policies for already-loaded PyTorch models.

Checkpoint selection and model construction belong to the caller. Adapters
encode a single game as a tensor including its batch dimension, expose legal
model-output indices, and translate the selected index into a game action.
"""

from collections.abc import Callable, Sequence
from typing import Generic, Protocol, TypeVar

import torch
from torch import Tensor, nn

from models.inference import resolve_device, select_greedy_q_action

GameT = TypeVar("GameT")
AdapterGameT = TypeVar("AdapterGameT", contravariant=True)
ActionT = TypeVar("ActionT")
AdapterActionT = TypeVar("AdapterActionT", covariant=True)
ActionSelector = Callable[[Tensor, Sequence[int]], int]


class GameAdapter(Protocol[AdapterGameT, AdapterActionT]):
    """Translate between a game's state/actions and a model's representation."""

    def encode(self, game: AdapterGameT) -> Tensor:
        """Return model input with a batch dimension of one."""
        ...

    def legal_actions(self, game: AdapterGameT) -> Sequence[int]:
        """Return legal output indices; return an empty sequence at game end."""
        ...

    def decode_action(self, game: AdapterGameT, index: int) -> AdapterActionT:
        """Translate a legal model-output index into the game's action type."""
        ...


class SavedAgentPlayer(Generic[GameT, ActionT]):
    """Callable policy using a model, a game adapter, and an action selector.

    The supplied model is moved to the selected device and kept in evaluation
    mode. Use a separately loaded model if training must continue elsewhere.
    Encoders own input shape and dtype; selectors own output interpretation.
    No optimizer, exploration, replay buffer, or training RNG is restored.
    """

    def __init__(
        self,
        model: nn.Module,
        adapter: GameAdapter[GameT, ActionT],
        action_selector: ActionSelector = select_greedy_q_action,
        device: str | torch.device = "auto",
    ) -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.adapter = adapter
        self.action_selector = action_selector

    def __call__(self, game: GameT) -> ActionT:
        """Infer a legal game action without gradients or exploration."""
        legal_actions = tuple(self.adapter.legal_actions(game))
        if not legal_actions:
            raise ValueError("Cannot select an action without legal moves")
        self.model.eval()
        with torch.inference_mode():
            inputs = self.adapter.encode(game).to(self.device)
            output = self.model(inputs)
            index = self.action_selector(output, legal_actions)
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index not in legal_actions
        ):
            raise ValueError("Action selector must return a legal output index")
        return self.adapter.decode_action(game, index)
