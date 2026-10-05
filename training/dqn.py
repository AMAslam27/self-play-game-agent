"""DQN action selection and learning, independent of a particular game."""

import math
import random
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Protocol, TypeVar

import torch
from torch import nn

from training.utils import (
    Transition,
    _require_positive_integer,
    build_lr_scheduler,
    build_optimizer,
)


class DQNObservation(Protocol):
    """Flat numeric features and a mask with one entry per possible action."""

    @property
    def board(self) -> Sequence[float]: ...

    @property
    def action_mask(self) -> Sequence[bool]: ...


ObservationT = TypeVar("ObservationT", bound=DQNObservation)


class DQNAgent:
    """Train a Q-network with configurable optimization and Smooth L1 loss.

    Networks must accept float32 feature batches and return one Q-value per
    action. The online network chooses moves and receives optimizer updates;
    the target network is an independent, frozen copy used for learning targets.

    The caller owns the replay buffer, epsilon schedule, and episode loop.
    optimizer and lr_scheduler accept the matching training config mappings.
    Defaults are Adam with lr=0.001, no weight decay, and no LR scheduler.
    Pass random.Random(seed) to reproduce exploration, and seed PyTorch before
    constructing the supplied network to reproduce its initial weights.
    """

    def __init__(
        self,
        network: nn.Module,
        *,
        optimizer: Mapping[str, Any] | None = None,
        lr_scheduler: Mapping[str, Any] | None = None,
        discount_factor: float = 0.99,
        target_update_every_updates: int = 250,
        device: str | torch.device = "auto",
        rng: random.Random | None = None,
    ):
        if not 0.0 <= discount_factor <= 1.0:
            raise ValueError("discount_factor must be in [0, 1]")
        _require_positive_integer(
            target_update_every_updates, "target_update_every_updates"
        )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.online_network = network.to(self.device)
        self.target_network = deepcopy(self.online_network)
        self.target_network.requires_grad_(False)
        self.target_network.eval()
        self.optimizer = build_optimizer(self.online_network.parameters(), optimizer)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, lr_scheduler)
        self.loss_function = nn.SmoothL1Loss()
        self.discount_factor = discount_factor
        self.target_update_every_updates = target_update_every_updates
        self.learning_updates = 0
        self._rng = rng if rng is not None else random.Random()

    def select_action(self, observation: DQNObservation, epsilon: float = 0.0) -> int:
        """Explore with probability epsilon, otherwise choose the best legal Q.

        Greedy inference disables gradients and dropout, restoring the online
        network's previous training mode afterward. Ties choose the first index.
        An observation with no legal actions cannot be used for action selection.
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        legal = [i for i, allowed in enumerate(observation.action_mask) if allowed]
        if not legal:
            raise ValueError("Cannot select an action without legal moves")
        if epsilon > 0.0 and self._rng.random() < epsilon:
            return self._rng.choice(legal)

        was_training = self.online_network.training
        self.online_network.eval()
        try:
            with torch.no_grad():
                states = self._states([observation])
                values = self.online_network(states)
                self._validate_output(values, 1, len(observation.action_mask))
                mask = self._masks([observation])
                return int(values.masked_fill(~mask, -torch.inf).argmax(dim=1).item())
        finally:
            self.online_network.train(was_training)

    def learn(self, batch: Sequence[Transition[ObservationT]]) -> float:
        """Perform one optimizer update and return the mean batch loss.

        Targets are reward + gamma * best legal next-state target Q-value.
        Terminal entries use reward alone and never evaluate their next states.
        Target weights are copied after each configured number of learning
        updates, independent of episode count or action-selection calls.
        An enabled LR scheduler advances after each successful optimizer step.
        """
        if not batch:
            raise ValueError("Cannot learn from an empty batch")
        action_count = len(batch[0].observation.action_mask)
        for transition in batch:
            observation = transition.observation
            if len(observation.action_mask) != action_count:
                raise ValueError("Batch observations must have matching action counts")
            action = transition.action
            if (
                isinstance(action, bool)
                or not isinstance(action, int)
                or not 0 <= action < action_count
                or not observation.action_mask[action]
            ):
                raise ValueError("Replay actions must be legal in their observations")
            if not math.isfinite(transition.reward):
                raise ValueError("Replay rewards must be finite")
            if not transition.done:
                next_mask = transition.next_observation.action_mask
                if len(next_mask) != action_count or not any(next_mask):
                    raise ValueError(
                        "Nonterminal next observations need valid legal masks"
                    )

        observations = [entry.observation for entry in batch]
        states = self._states(observations)
        actions = torch.tensor(
            [entry.action for entry in batch], dtype=torch.long, device=self.device
        ).unsqueeze(1)
        targets = torch.tensor(
            [entry.reward for entry in batch], dtype=torch.float32, device=self.device
        )
        nonterminal_indices = [i for i, entry in enumerate(batch) if not entry.done]

        self.target_network.eval()
        with torch.no_grad():
            if nonterminal_indices:
                next_observations = [
                    batch[i].next_observation for i in nonterminal_indices
                ]
                next_values = self.target_network(self._states(next_observations))
                self._validate_output(next_values, len(next_observations), action_count)
                next_values = next_values.masked_fill(
                    ~self._masks(next_observations), -torch.inf
                )
                targets[nonterminal_indices] += (
                    self.discount_factor * next_values.max(dim=1).values
                )

        was_training = self.online_network.training
        self.online_network.train()
        try:
            self.optimizer.zero_grad(set_to_none=True)
            values = self.online_network(states)
            self._validate_output(values, len(batch), action_count)
            predictions = values.gather(1, actions).squeeze(1)
            loss = self.loss_function(predictions, targets)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite DQN loss; learning update aborted")
            loss.backward()
            self.optimizer.step()
        finally:
            self.online_network.train(was_training)

        self.learning_updates += 1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if self.learning_updates % self.target_update_every_updates == 0:
            self.update_target_network()
        return float(loss.detach().item())

    def update_target_network(self) -> None:
        """Copy online weights and buffers, keeping target inference frozen."""
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.requires_grad_(False)
        self.target_network.eval()

    def state_dict(self) -> dict[str, Any]:
        """Snapshot model, optimization, progress, and exploration state."""
        return deepcopy(
            {
                "online_network": self.online_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": (
                    self.lr_scheduler.state_dict()
                    if self.lr_scheduler is not None
                    else None
                ),
                "learning_updates": self.learning_updates,
                "rng_state": self._rng.getstate(),
                "training": self.online_network.training,
            }
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore against a model and optimizer constructed from saved config."""
        if (state["lr_scheduler"] is None) != (self.lr_scheduler is None):
            raise ValueError("Checkpoint and agent scheduler configurations differ")
        updates = state["learning_updates"]
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise ValueError("Invalid learning update count in checkpoint")
        self.online_network.load_state_dict(state["online_network"])
        self.target_network.load_state_dict(state["target_network"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        self.learning_updates = updates
        self._rng.setstate(state["rng_state"])
        self.online_network.train(state["training"])
        self.target_network.requires_grad_(False)
        self.target_network.eval()

    def _states(self, observations: Sequence[DQNObservation]) -> torch.Tensor:
        return torch.tensor(
            [list(observation.board) for observation in observations],
            dtype=torch.float32,
            device=self.device,
        )

    def _masks(self, observations: Sequence[DQNObservation]) -> torch.Tensor:
        return torch.tensor(
            [list(observation.action_mask) for observation in observations],
            dtype=torch.bool,
            device=self.device,
        )

    @staticmethod
    def _validate_output(
        values: torch.Tensor, batch_size: int, action_count: int
    ) -> None:
        if values.shape != (batch_size, action_count):
            raise ValueError(
                "Network must return a Q-value for every action in each state"
            )
