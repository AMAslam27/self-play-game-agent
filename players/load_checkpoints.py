"""Select and load inference checkpoints for saved players.

Callers select a checkpoint path and supply a factory that builds an untrained
model from its saved configuration. This supports different architectures
without importing game-specific models here. Other checkpoint formats need
their own reader; this loader understands the current online-network payload.
"""

import sqlite3
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
from torch import nn

from evaluation.tictactoe_ranking import checkpoint_is_available, rank_checkpoints
from models.inference import resolve_device
from training.checkpoints import load_checkpoint, preserve_rng_state

ModelT = TypeVar("ModelT", bound=nn.Module)


def resolve_checkpoint(
    selection: str | Path,
    *,
    connection: sqlite3.Connection | None = None,
    game: str = "tictactoe",
    model_type: str = "dqn",
    protocol_id: str | None = None,
) -> Path:
    """Return a path for a selector or any explicit checkpoint file.

    Explicit paths bypass ranking eligibility; the loader validates their format.
    latest selects by run completion time and needs no evaluation eligibility.
    best uses the versioned baseline ranking. Automatic selection only returns
    final checkpoints whose bytes still match their registered identities.
    """
    if isinstance(selection, Path) or selection not in ("latest", "best"):
        path = Path(selection).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path
    if connection is None:
        raise ValueError(
            "Automatic checkpoint selection requires a registry connection"
        )
    if selection == "best":
        ranked = rank_checkpoints(
            connection, game=game, model_type=model_type, protocol_id=protocol_id
        )
        if not ranked:
            raise ValueError(
                "No eligible checkpoint has complete comparable baseline evaluations"
            )
        return ranked[0].path
    rows = connection.execute(
        "SELECT c.path, c.checkpoint_hash, r.finished_at, r.run_id FROM model_checkpoints c "
        "JOIN model_runs r USING(run_id, game, model_type) "
        "WHERE c.is_final=1 AND r.status='completed' AND c.game=? AND c.model_type=?",
        (game, model_type),
    ).fetchall()
    for row in sorted(
        rows,
        key=lambda row: (
            datetime.fromisoformat(row["finished_at"]).timestamp(),
            row["run_id"],
        ),
        reverse=True,
    ):
        path = Path(row["path"])
        if checkpoint_is_available(path, row["checkpoint_hash"]):
            return path
    raise ValueError("No available final checkpoint for a completed compatible run")


@dataclass(frozen=True)
class LoadedModel(Generic[ModelT]):
    """An inference model and the saved experiment identity used to load it."""

    model: ModelT
    config: dict[str, Any]
    run_id: str
    checkpoint_path: Path


def load_model_checkpoint(
    path: str | Path,
    model_factory: Callable[[Mapping[str, Any]], ModelT],
    device: str | torch.device = "auto",
) -> LoadedModel[ModelT]:
    """Build a model from saved config and strictly restore online weights.

    The factory receives the full configuration and must return a fresh
    PyTorch module. The result is moved to the requested device and placed in
    evaluation mode. Optimizers, replay buffers, and saved RNGs are ignored;
    model construction also preserves the caller's global RNG state.

    Missing files raise FileNotFoundError. Unsupported or incomplete payloads
    and incompatible weights raise ValueError. Factory and device errors
    propagate to the caller.
    """
    checkpoint_path = Path(path).expanduser().resolve()
    payload = load_checkpoint(checkpoint_path)
    config = payload.get("config")
    run_id = payload.get("run_id")
    agent = payload.get("agent")
    if not isinstance(config, dict) or any(not isinstance(key, str) for key in config):
        raise ValueError("Checkpoint must contain a configuration dictionary")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Checkpoint must contain a nonempty run ID")
    if not isinstance(agent, Mapping) or not isinstance(
        agent.get("online_network"), Mapping
    ):
        raise ValueError("Checkpoint must contain online-network weights")

    with preserve_rng_state():
        model = model_factory(deepcopy(config))
        if not isinstance(model, nn.Module):
            raise TypeError("Model factory must return a PyTorch module")
        try:
            model.load_state_dict(agent["online_network"], strict=True)
        except (RuntimeError, TypeError) as error:
            raise ValueError(
                f"Checkpoint weights are incompatible with the supplied model: {checkpoint_path}"
            ) from error
        model.to(resolve_device(device))
        model.eval()

    return LoadedModel(
        model=model,
        config=config,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
    )
