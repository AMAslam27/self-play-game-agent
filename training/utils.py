"""Shared replay storage and configurable optimization utilities."""

from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from torch import Tensor
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

ObservationT = TypeVar("ObservationT")


@dataclass(frozen=True)
class Transition(Generic[ObservationT]):
    """One agent decision and its outcome, including the next action mask.

    Observations should be immutable snapshots, such as the Tic-Tac-Toe
    adapter's Observation. Freezing this record prevents field reassignment;
    it does not freeze mutable objects supplied as observations.
    """

    observation: ObservationT
    action: int
    reward: float
    next_observation: ObservationT
    done: bool


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class ReplayBuffer(Generic[ObservationT]):
    """Bounded FIFO storage with uniform sampling without replacement.

    Full buffers discard the oldest entry on add. Sampling leaves entries in
    the buffer and returns ordinary Transition objects, ready for the learner
    to convert into tensor batches on its selected device.

    Pass random.Random(seed) for reproducible sampling. The default creates an
    independent generator; sampling never advances the global game RNG.
    """

    def __init__(self, capacity: int, rng: random.Random | None = None):
        _require_positive_integer(capacity, "capacity")
        self._transitions: deque[Transition[ObservationT]] = deque(maxlen=capacity)
        self._rng = rng if rng is not None else random.Random()

    def __len__(self) -> int:
        return len(self._transitions)

    def add(self, transition: Transition[ObservationT]) -> None:
        """Store a transition, evicting the oldest when at capacity."""
        self._transitions.append(transition)

    def sample(self, batch_size: int) -> list[Transition[ObservationT]]:
        """Sample distinct entries; fail if there are too few to fill a batch."""
        _require_positive_integer(batch_size, "batch_size")
        if batch_size > len(self):
            raise ValueError("Not enough transitions to sample the requested batch")
        # random.sample expects a sequence, so take a snapshot of the deque.
        return self._rng.sample(list(self._transitions), batch_size)

    def state_dict(self) -> dict[str, Any]:
        """Export entries in FIFO order and the independent sampling RNG."""
        return {
            "capacity": self._transitions.maxlen,
            "transitions": list(self._transitions),
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a buffer with the same capacity, including future samples."""
        if state["capacity"] != self._transitions.maxlen:
            raise ValueError("Checkpoint replay capacity does not match the buffer")
        entries = state["transitions"]
        if len(entries) > state["capacity"] or not all(
            isinstance(entry, Transition) for entry in entries
        ):
            raise ValueError("Invalid replay entries in checkpoint")
        rng = random.Random()
        rng.setstate(state["rng_state"])
        self._transitions = deque(entries, maxlen=state["capacity"])
        self._rng = rng


def linear_epsilon(
    decisions: int, initial: float, final: float, decay_steps: int
) -> float:
    """Exploration before the next decision, clamped after linear decay."""
    _require_positive_integer(decay_steps, "decay_steps")
    if isinstance(decisions, bool) or not isinstance(decisions, int) or decisions < 0:
        raise ValueError("decisions must be a nonnegative integer")
    if not 0 <= final <= initial <= 1:
        raise ValueError("Epsilon values must satisfy 0 <= final <= initial <= 1")
    if decisions >= decay_steps:
        return final
    fraction = min(decisions / decay_steps, 1.0)
    return initial + fraction * (final - initial)


def _config_values(
    config: Mapping[str, Any] | None, allowed_keys: set[str]
) -> dict[str, Any]:
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise ValueError("Configuration must be a mapping")
    if unknown := set(config) - allowed_keys:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    return dict(config)


def _config_options(config: Mapping[str, Any]) -> dict[str, Any]:
    options = config.get("options", {})
    if not isinstance(options, Mapping):
        raise ValueError("options must be a mapping")
    return dict(options)


def _require_finite_number(value: float, name: str, *, allow_zero: bool) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (value == 0 and not allow_zero)
    ):
        constraint = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {constraint}")


def build_optimizer(
    parameters: Iterable[Tensor], config: Mapping[str, Any] | None = None
) -> Optimizer:
    """Build Adam, AdamW, or SGD from an optimizer config mapping.

    Keys are name, learning_rate, weight_decay, and optional options for
    optimizer-specific arguments (e.g. momentum for SGD). Defaults preserve
    Adam with lr=0.001 and weight_decay=0.0. The config is never modified.
    PyTorch is imported only when optimization helpers are used, so replay
    storage remains usable without loading PyTorch.
    """
    values = _config_values(
        config, {"name", "learning_rate", "weight_decay", "options"}
    )
    learning_rate = values.get("learning_rate", 0.001)
    weight_decay = values.get("weight_decay", 0.0)
    _require_finite_number(learning_rate, "learning_rate", allow_zero=False)
    _require_finite_number(weight_decay, "weight_decay", allow_zero=True)
    options = _config_options(values)
    if set(options) & {"params", "lr", "weight_decay"}:
        raise ValueError("Set learning_rate and weight_decay outside optimizer options")

    from torch import optim

    factories: dict[str, Callable[..., Optimizer]] = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
    }
    name = values.get("name", "adam")
    if not isinstance(name, str) or name not in factories:
        raise ValueError(f"Unsupported optimizer: {name!r}")
    try:
        return factories[name](
            parameters, lr=learning_rate, weight_decay=weight_decay, **options
        )
    except TypeError as error:
        raise ValueError(f"Invalid options for optimizer {name!r}: {error}") from error


def build_lr_scheduler(
    optimizer: Optimizer, config: Mapping[str, Any] | None = None
) -> LRScheduler | None:
    """Build a per-optimizer-update scheduler: none, step, or exponential.

    Config contains name and options. Step uses step_size (positive integer)
    and gamma (positive multiplier, default 0.1). Exponential requires gamma.
    Call scheduler.step() after optimizer.step(), never during evaluation.
    """
    values = _config_values(config, {"name", "options"})
    options = _config_options(values)
    name = values.get("name", "none")
    if name == "none":
        if options:
            raise ValueError("Scheduler 'none' does not accept options")
        return None

    from torch.optim import lr_scheduler

    factories: dict[str, Callable[..., LRScheduler]] = {
        "step": lr_scheduler.StepLR,
        "exponential": lr_scheduler.ExponentialLR,
    }
    if not isinstance(name, str) or name not in factories:
        raise ValueError(f"Unsupported LR scheduler: {name!r}")
    if "optimizer" in options:
        raise ValueError("Scheduler optimizer is supplied by the agent")
    if name == "step":
        _require_positive_integer(options.get("step_size", 0), "step_size")
        options.setdefault("gamma", 0.1)
    if "gamma" not in options:
        raise ValueError("Scheduler requires gamma")
    _require_finite_number(options["gamma"], "gamma", allow_zero=False)
    try:
        return factories[name](optimizer, **options)
    except TypeError as error:
        raise ValueError(f"Invalid options for scheduler {name!r}: {error}") from error
