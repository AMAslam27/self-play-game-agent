"""Portable checkpoint payloads, atomic saves, and global RNG restoration."""

from __future__ import annotations

import random
import shutil
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch

from training.dqn import DQNAgent, DQNObservation
from training.utils import ReplayBuffer, Transition

ObservationT = TypeVar("ObservationT", bound=DQNObservation)
CHECKPOINT_VERSION = 1


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state(legacy=True)
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].tolist(),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state[0],
            np.asarray(numpy_state[1], dtype=np.uint32),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


@contextmanager
def preserve_rng_state() -> Iterator[None]:
    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


def checkpoint_payload(
    *,
    run_id: str,
    config: Mapping[str, Any],
    agent: DQNAgent,
    replay: ReplayBuffer[ObservationT],
    opponent_rng: random.Random,
    progress: Mapping[str, Any],
    metric_offsets: Mapping[str, int],
) -> dict[str, Any]:
    """Encode observations as primitive values for weights_only loading."""
    replay_state = replay.state_dict()
    replay_state["transitions"] = [
        {
            "observation": {
                "board": tuple(entry.observation.board),
                "action_mask": tuple(entry.observation.action_mask),
            },
            "action": entry.action,
            "reward": entry.reward,
            "done": entry.done,
            "next_observation": {
                "board": tuple(entry.next_observation.board),
                "action_mask": tuple(entry.next_observation.action_mask),
            },
        }
        for entry in replay_state["transitions"]
    ]
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "run_id": run_id,
        "config": dict(config),
        "agent": agent.state_dict(),
        "replay": replay_state,
        "opponent_rng": opponent_rng.getstate(),
        "rng": capture_rng_state(),
        "progress": dict(progress),
        "metric_offsets": dict(metric_offsets),
    }


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)
    latest = path.parent / "latest.pt"
    if path != latest:
        temporary_latest = latest.with_suffix(".pt.tmp")
        shutil.copyfile(path, temporary_latest)
        temporary_latest.replace(latest)


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("checkpoint_version") != CHECKPOINT_VERSION
    ):
        raise ValueError("Unsupported training checkpoint")
    return payload


def restore_checkpoint(
    payload: Mapping[str, Any],
    agent: DQNAgent,
    replay: ReplayBuffer[ObservationT],
    opponent_rng: random.Random,
    observation_factory: Callable[..., ObservationT],
) -> None:
    agent.load_state_dict(payload["agent"])
    replay_state = dict(payload["replay"])
    replay_state["transitions"] = [
        Transition(
            observation_factory(**entry["observation"]),
            entry["action"],
            entry["reward"],
            observation_factory(**entry["next_observation"]),
            entry["done"],
        )
        for entry in replay_state["transitions"]
    ]
    replay.load_state_dict(replay_state)
    opponent_rng.setstate(payload["opponent_rng"])
    # Construction and loading can consume RNGs; restore global state last.
    restore_rng_state(payload["rng"])
