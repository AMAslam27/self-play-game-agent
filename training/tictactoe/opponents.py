"""Frozen model opponents and seeded opponent pools for Tic-Tac-Toe training."""

from __future__ import annotations

import random
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from evaluation.tictactoe_matches import Policy, make_opponent
from games.tictactoe.rules import TicTacToe
from players.load_checkpoints import load_model_checkpoint
from players.saved_agent import SavedAgentPlayer
from players.tictactoe_player import TicTacToePlayerAdapter, build_tictactoe_model
from training.checkpoints import preserve_rng_state
from training.config import PROJECT_ROOT


@dataclass(frozen=True)
class Opponent:
    identity: str
    category: str
    policy: Policy


class TrainingOpponents:
    """Select a policy once per episode; model policies never learn.

    Pool weights apply to categories, independent of their number of models.
    Imported models are retained separately from the bounded snapshot history.
    Checkpoints embed weights so resume does not reread source checkpoints.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        device: torch.device,
        rng: random.Random,
    ):
        self.config = config
        self.device = device
        self.rng = rng
        self.mode = config["environment"]["opponent"]
        self.imported: list[dict[str, Any]] = []
        self.snapshots: list[dict[str, Any]] = []
        self.policies: dict[str, SavedAgentPlayer[TicTacToe, int]] = {}

    def _install(self, entry: dict[str, Any], model: nn.Module | None = None) -> None:
        if model is None:
            with preserve_rng_state():
                model = build_tictactoe_model(entry["config"])
                model.load_state_dict(entry["weights"], strict=True)
        model.requires_grad_(False)
        model.eval()
        self.policies[entry["identity"]] = SavedAgentPlayer(
            model, TicTacToePlayerAdapter(), device=self.device
        )

    def _import_checkpoint(self, path: str) -> None:
        loaded = load_model_checkpoint(
            (PROJECT_ROOT / Path(path).expanduser()).resolve(),
            build_tictactoe_model,
            device=self.device,
        )
        entry = {
            "identity": f"imported-{len(self.imported)}",
            "source_path": str(loaded.checkpoint_path),
            "source_run_id": loaded.run_id,
            "config": deepcopy(loaded.config),
            "weights": {
                name: tensor.detach().cpu().clone()
                for name, tensor in loaded.model.state_dict().items()
            },
        }
        self.imported.append(entry)
        self._install(entry, loaded.model)

    def initialize(self, learner: nn.Module) -> None:
        if self.mode == "model":
            self._import_checkpoint(self.config["opponent_model"]["checkpoint"])
        elif self.mode == "pool":
            for path in self.config["opponent_pool"]["initial_checkpoints"]:
                self._import_checkpoint(path)
            self.add_snapshot(learner, episode=0)

    def add_snapshot(self, learner: nn.Module, episode: int) -> None:
        entry = {
            "identity": f"snapshot-{episode}",
            "episode": episode,
            "config": {"network": deepcopy(self.config["network"])},
            "weights": {
                name: tensor.detach().cpu().clone()
                for name, tensor in learner.state_dict().items()
            },
        }
        self.snapshots.append(entry)
        self._install(entry)
        limit = self.config["opponent_pool"]["max_snapshots"]
        if len(self.snapshots) > limit:
            # Retain the initial opponent as well as recent model versions.
            removed = self.snapshots.pop(1)
            del self.policies[removed["identity"]]

    def after_episode(self, learner: nn.Module, episode: int) -> None:
        if self.mode == "pool":
            interval = self.config["opponent_pool"]["snapshot_every_episodes"]
            if episode % interval == 0:
                self.add_snapshot(learner, episode)

    def select(self) -> Opponent:
        if self.mode in ("random", "minimax"):
            return Opponent(self.mode, self.mode, make_opponent(self.mode, self.rng))
        if self.mode == "model":
            entry = self.imported[0]
            return Opponent(
                entry["identity"], "model", self.policies[entry["identity"]]
            )

        candidates = {
            "random": [],
            "minimax": [],
            "latest": [self.snapshots[-1]],
            "historical": self.imported + self.snapshots[:-1],
        }
        weights = self.config["opponent_pool"]["weights"]
        categories = [
            category
            for category, entries in candidates.items()
            if weights[category] > 0 and (category in ("random", "minimax") or entries)
        ]
        if not categories:
            raise ValueError("No available opponents have a positive weight")
        category = self.rng.choices(
            categories, weights=[weights[name] for name in categories], k=1
        )[0]
        if category in ("random", "minimax"):
            return Opponent(category, category, make_opponent(category, self.rng))
        entry = self.rng.choice(candidates[category])
        return Opponent(entry["identity"], category, self.policies[entry["identity"]])

    def state_dict(self) -> dict[str, Any]:
        return deepcopy(
            {
                "mode": self.mode,
                "imported": self.imported,
                "snapshots": self.snapshots,
                "rng": self.rng.getstate(),
            }
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state["mode"] != self.mode:
            raise ValueError("Checkpoint opponent mode differs from config")
        if self.mode == "model" and (len(state["imported"]) != 1 or state["snapshots"]):
            raise ValueError("Model mode requires exactly one imported opponent")
        if self.mode == "pool" and not state["snapshots"]:
            raise ValueError("Pool checkpoint is missing its snapshots")
        self.imported = deepcopy(state["imported"])
        self.snapshots = deepcopy(state["snapshots"])
        self.policies.clear()
        for entry in self.imported + self.snapshots:
            self._install(entry)
        self.rng.setstate(state["rng"])
