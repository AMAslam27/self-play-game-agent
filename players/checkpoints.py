"""Load inference models from the project's versioned training checkpoints.

Callers select a checkpoint path and supply a factory that builds an untrained
model from its saved configuration. This supports different architectures
without importing game-specific models here. Other checkpoint formats need
their own reader; this loader understands the current online-network payload.
"""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
from torch import nn

from training.checkpoints import load_checkpoint, preserve_rng_state

ModelT = TypeVar("ModelT", bound=nn.Module)


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
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(torch.device(device))
        model.eval()

    return LoadedModel(
        model=model,
        config=config,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
    )
