"""Load, resolve, and validate settings for the game training coordinator."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from training.utils import _require_finite_number, _require_positive_integer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "experiment": {
        "name": "dqn",
        "seed": 42,
        "device": "auto",
        "deterministic": True,
    },
    "environment": {"opponent": "random", "agent_seat": "alternate"},
    "network": {
        "hidden_sizes": [64, 64],
        "block_types": ["dense", "dense"],
        "activations": ["relu", "relu"],
        "dropout_probability": 0.1,
    },
    "training": {
        "episodes": 20000,
        "optimizer": {
            "name": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "options": {},
        },
        "lr_scheduler": {"name": "none", "options": {}},
        "loss": "smooth_l1",
        "discount_factor": 0.99,
        "batch_size": 64,
        "replay_capacity": 10000,
        "learning_starts": 1000,
        "train_every_steps": 1,
        "gradient_updates_per_step": 1,
        "target_update_every_updates": 250,
    },
    "exploration": {
        "initial_epsilon": 1.0,
        "final_epsilon": 0.05,
        "schedule": "linear",
        "decay_steps": 50000,
    },
    "evaluation": {
        "every_episodes": 1000,
        "opponents": ["random", "minimax"],
        "seats": ["x", "o"],
        "games_per_opponent_per_seat": 200,
        "seed": 12345,
        "epsilon": 0.0,
    },
    "artifacts": {
        "output_root": None,
        "save_resolved_config": True,
        "save_code_version": True,
        "checkpoint_every_episodes": 1000,
        "save_final_checkpoint": True,
        "metrics_every_episodes": 100,
        "plot_every_episodes": 1000,
    },
}


def _merge(
    defaults: dict[str, Any], supplied: Mapping[str, Any], prefix: str = ""
) -> dict[str, Any]:
    if not isinstance(supplied, Mapping):
        raise ValueError(f"{prefix or 'config'} must be a mapping")
    result = deepcopy(defaults)
    for key, value in supplied.items():
        if key not in defaults:
            raise ValueError(f"Unknown configuration setting: {prefix}{key}")
        if isinstance(defaults[key], dict) and key != "options":
            result[key] = _merge(defaults[key], value, f"{prefix}{key}.")
        else:
            result[key] = deepcopy(value)
    return result


def resolve_config(
    supplied: Mapping[str, Any],
    default_output_root: Path,
    *,
    episodes: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Resolve defaults; paths are relative to the project, not config or cwd."""
    config = _merge(DEFAULT_CONFIG, supplied)
    if episodes is not None:
        config["training"]["episodes"] = episodes
    if device is not None:
        config["experiment"]["device"] = device
    if (
        isinstance(config["schema_version"], bool)
        or not isinstance(config["schema_version"], int)
        or config["schema_version"] != 1
    ):
        raise ValueError("Unsupported configuration schema_version")
    experiment = config["experiment"]
    if not isinstance(experiment["name"], str) or not experiment["name"].strip():
        raise ValueError("experiment.name must be a nonempty string")
    for seed in (experiment["seed"], config["evaluation"]["seed"]):
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("Seeds must be integers in [0, 2**32)")
    if experiment["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("Device must be auto, cpu, or cuda")
    if config["environment"]["opponent"] not in ("random", "minimax"):
        raise ValueError("Opponent must be random or minimax")
    if config["environment"]["agent_seat"] not in ("alternate", "x", "o"):
        raise ValueError("Agent seat must be alternate, x, or o")

    training = config["training"]
    for key in (
        "episodes",
        "batch_size",
        "replay_capacity",
        "learning_starts",
        "train_every_steps",
        "gradient_updates_per_step",
        "target_update_every_updates",
    ):
        _require_positive_integer(training[key], f"training.{key}")
    if training["replay_capacity"] < training["batch_size"]:
        raise ValueError("Replay capacity must accommodate a full batch")
    if training["loss"] != "smooth_l1":
        raise ValueError("Only smooth_l1 loss is currently supported")
    if (
        isinstance(training["discount_factor"], bool)
        or not isinstance(training["discount_factor"], (int, float))
        or not 0 <= training["discount_factor"] <= 1
    ):
        raise ValueError("discount_factor must be in [0, 1]")
    optimizer = training["optimizer"]
    if optimizer["name"] not in ("adam", "adamw", "sgd"):
        raise ValueError("Unsupported optimizer")
    _require_finite_number(
        optimizer["learning_rate"], "learning_rate", allow_zero=False
    )
    _require_finite_number(optimizer["weight_decay"], "weight_decay", allow_zero=True)
    scheduler = training["lr_scheduler"]
    if scheduler["name"] not in ("none", "step", "exponential"):
        raise ValueError("Unsupported LR scheduler")
    for section in (optimizer, scheduler):
        if not isinstance(section["options"], Mapping):
            raise ValueError("Optimizer and scheduler options must be mappings")
    if scheduler["name"] == "none" and scheduler["options"]:
        raise ValueError("Disabled LR scheduler must have empty options")
    if scheduler["name"] == "step":
        _require_positive_integer(scheduler["options"].get("step_size", 0), "step_size")
        scheduler["options"].setdefault("gamma", 0.1)
    if scheduler["name"] != "none":
        _require_finite_number(
            scheduler["options"].get("gamma"), "gamma", allow_zero=False
        )

    network = config["network"]
    for key in ("hidden_sizes", "block_types", "activations"):
        if not isinstance(network[key], list) or not network[key]:
            raise ValueError(f"network.{key} must be a nonempty list")
    if not (
        len(network["hidden_sizes"])
        == len(network["block_types"])
        == len(network["activations"])
    ):
        raise ValueError("Network lists must have matching lengths")
    for size in network["hidden_sizes"]:
        _require_positive_integer(size, "hidden size")
    if any(
        name not in ("dense", "dropout", "residual") for name in network["block_types"]
    ):
        raise ValueError("Unsupported network block type")
    if any(
        name not in ("relu", "tanh", "gelu", "identity")
        for name in network["activations"]
    ):
        raise ValueError("Unsupported network activation")
    probability = network["dropout_probability"]
    if not isinstance(probability, (int, float)) or not 0 <= probability < 1:
        raise ValueError("dropout_probability must be in [0, 1)")

    exploration = config["exploration"]
    if exploration["schedule"] != "linear":
        raise ValueError("Only linear epsilon decay is currently supported")
    for key in ("initial_epsilon", "final_epsilon"):
        _require_finite_number(exploration[key], key, allow_zero=True)
    if not 0 <= exploration["final_epsilon"] <= exploration["initial_epsilon"] <= 1:
        raise ValueError("Epsilon must satisfy 0 <= final <= initial <= 1")
    _require_positive_integer(exploration["decay_steps"], "decay_steps")
    evaluation = config["evaluation"]
    for key in ("every_episodes", "games_per_opponent_per_seat"):
        _require_positive_integer(evaluation[key], f"evaluation.{key}")
    for key, allowed in (("opponents", ("random", "minimax")), ("seats", ("x", "o"))):
        values = evaluation[key]
        if (
            not isinstance(values, list)
            or not values
            or any(value not in allowed for value in values)
        ):
            raise ValueError(f"Invalid evaluation.{key}")
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate evaluation.{key}")
    if evaluation["epsilon"] != 0.0:
        raise ValueError("Evaluation epsilon must be zero")
    artifacts = config["artifacts"]
    for key in (
        "checkpoint_every_episodes",
        "metrics_every_episodes",
        "plot_every_episodes",
    ):
        _require_positive_integer(artifacts[key], f"artifacts.{key}")
    for value in (
        experiment["deterministic"],
        artifacts["save_resolved_config"],
        artifacts["save_code_version"],
        artifacts["save_final_checkpoint"],
    ):
        if not isinstance(value, bool):
            raise ValueError("Boolean configuration settings must be true or false")
    output_root = artifacts["output_root"]
    if output_root is not None and (
        not isinstance(output_root, str) or not output_root.strip()
    ):
        raise ValueError("output_root must be a nonempty path")
    path = Path(output_root) if output_root is not None else default_output_root
    artifacts["output_root"] = str((PROJECT_ROOT / path).resolve())
    return config


def load_config(
    path: Path, default_output_root: Path, **overrides: Any
) -> dict[str, Any]:
    import yaml

    supplied = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(supplied, Mapping):
        raise ValueError("Config file must contain a YAML mapping")
    return resolve_config(supplied, default_output_root, **overrides)
