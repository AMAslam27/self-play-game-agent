"""Evaluate a saved Tic-Tac-Toe checkpoint without resuming training."""

import argparse
import csv
import json
import logging
import platform
import random
import signal
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from evaluation.artifacts import display_project_path
from evaluation.database import PROJECT_ROOT, connect_database
from evaluation.model_registry import (
    DEFAULT_REGISTRY_PATH,
    checkpoint_hash,
    connect_registry,
    protocol_identity,
)
from evaluation.recorder import ModelIdentity, RunRecorder
from evaluation.tictactoe_matches import (
    evaluate_matches,
    evaluation_seed,
    make_opponent,
)
from evaluation.uncertainty import CONFIDENCE_LEVEL, wilson_interval
from games.tictactoe.rules import TicTacToe
from players.load_checkpoints import load_model_checkpoint, resolve_checkpoint
from players.saved_agent import SavedAgentPlayer
from players.tictactoe_player import TicTacToePlayerAdapter, build_tictactoe_model
from training.checkpoints import preserve_rng_state
from training.metrics import write_json
from training.utils import _require_positive_integer
from utils.timestamps import utc_now

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "tictactoe" / "evaluations"
EVALUATOR_VERSION = "tictactoe-sampled-v1"
logger = logging.getLogger(__name__)
CSV_FIELDS = [
    "base_seed",
    "seed",
    "opponent",
    "seat",
    "games",
    "wins",
    "draws",
    "losses",
    "average_return",
    "loss_rate",
    "loss_rate_ci_lower",
    "loss_rate_ci_upper",
    "loss_rate_ci_method",
    "loss_rate_ci_confidence",
    "loss_rate_ci_reason",
    "duration_seconds",
    "status",
    "db_run_id",
]


def _validate_settings(
    opponents: Sequence[str],
    seats: Sequence[str],
    seeds: Sequence[int],
    games: int,
) -> None:
    _require_positive_integer(games, "games")
    for name, values, allowed in (
        ("opponents", opponents, ("random", "minimax")),
        ("seats", seats, ("x", "o")),
    ):
        if not values or any(value not in allowed for value in values):
            raise ValueError(f"Invalid evaluation {name}")
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate evaluation {name}")
    if not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32
        for seed in seeds
    ):
        raise ValueError("Evaluation seeds must be integers in [0, 2**32)")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate evaluation seeds")


def _load_player(
    selection: str | Path, registry_path: Path, protocol_id: str | None, device: str
) -> tuple[SavedAgentPlayer[TicTacToe, int], ModelIdentity, dict[str, Any]]:
    expected_hash = None
    if isinstance(selection, str) and selection in ("latest", "best"):
        with closing(connect_registry(registry_path)) as registry:
            path = resolve_checkpoint(
                selection, connection=registry, protocol_id=protocol_id
            )
            record = registry.execute(
                "SELECT checkpoint_hash, protocol_id FROM model_checkpoints WHERE path=? AND is_final=1",
                (str(path),),
            ).fetchone()
            if record is None:
                raise ValueError("Selected checkpoint is no longer registered")
            expected_hash = record["checkpoint_hash"]
            protocol_id = record["protocol_id"]
    else:
        path = resolve_checkpoint(selection)
    identity = checkpoint_hash(path)
    if expected_hash is not None and expected_hash != identity:
        raise ValueError("Selected checkpoint changed after resolution")
    loaded = load_model_checkpoint(path, build_tictactoe_model, device)
    if checkpoint_hash(path) != identity:
        raise ValueError("Checkpoint changed while loading the model")
    name = str(loaded.config.get("experiment", {}).get("name", loaded.run_id))
    model = ModelIdentity(
        loaded.run_id, name, str(path), identity, str(selection), protocol_id
    )
    policy = SavedAgentPlayer(loaded.model, TicTacToePlayerAdapter(), device=device)
    return policy, model, loaded.config


def _loss_statistics(
    opponent: str, games: int, losses: int, *, complete_batch: bool = True
) -> dict[str, Any]:
    """Describe observed losses and interval applicability explicitly."""
    reason = None
    if not complete_batch:
        reason = "incomplete_batch"
    elif not games:
        reason = "no_completed_games"
    elif opponent == "minimax":
        reason = "deterministic_opponent"
    lower, upper = wilson_interval(losses, games) if reason is None else (None, None)
    return {
        "loss_rate": losses / games if games else None,
        "loss_rate_ci_lower": lower,
        "loss_rate_ci_upper": upper,
        "loss_rate_ci_method": "wilson" if reason is None else None,
        "loss_rate_ci_confidence": CONFIDENCE_LEVEL if reason is None else None,
        "loss_rate_ci_reason": reason,
    }


def _summarise(
    batches: list[dict[str, Any]],
    opponents: Sequence[str],
    seats: Sequence[str],
    seed_count: int,
) -> list[dict[str, Any]]:
    """Pool only full batches; partial batches remain available as audit rows."""
    summary = []
    for opponent in opponents:
        for seat in seats:
            rows = [
                row
                for row in batches
                if row["opponent"] == opponent
                and row["seat"] == seat
                and row["status"] == "completed"
            ]
            counts = {
                key: sum(row[key] for row in rows)
                for key in ("games", "wins", "draws", "losses")
            }
            summary.append(
                {
                    "opponent": opponent,
                    "seat": seat,
                    **counts,
                    **_loss_statistics(opponent, counts["games"], counts["losses"]),
                    "completed_seed_batches": len(rows),
                    "requested_seed_batches": seed_count,
                    "average_return": (counts["wins"] - counts["losses"])
                    / counts["games"]
                    if counts["games"]
                    else None,
                }
            )
    return summary


def evaluate_checkpoint(
    selection: str | Path,
    *,
    opponents: Sequence[str] = ("random", "minimax"),
    seats: Sequence[str] = ("x", "o"),
    seeds: Sequence[int] = (12345,),
    games: int = 1000,
    device: str = "auto",
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    protocol_id: str | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    """Save a unique assessment directory; never alter training artifacts.

    Games are requested per opponent, seat, and base seed. Each combination
    receives an independent opponent RNG, using the training seed-offset rule.
    Completed games are committed to SQLite individually. Batch CSV/JSON results
    are flushed after each batch, including gracefully interrupted batches.
    Failures preserve metadata and recorded games, then propagate to callers.
    Offline evidence is separate from the registry's existing ranking inputs.
    """
    opponents, seats, seeds = tuple(opponents), tuple(seats), tuple(seeds)
    _validate_settings(opponents, seats, seeds, games)
    if protocol_id is not None and selection != "best":
        raise ValueError("protocol_id is only used when selecting best")
    stop = should_stop if should_stop is not None else lambda: False
    policy, model, config = _load_player(selection, registry_path, protocol_id, device)
    protocol = {
        "game": "tictactoe",
        "evaluator_id": EVALUATOR_VERSION,
        "opponents": list(opponents),
        "seats": list(seats),
        "base_seeds": list(seeds),
        "games_per_opponent_per_seat_per_seed": games,
        "epsilon": 0.0,
        "encoding": "current-player-relative-v1",
        "selection": "greedy-lowest-index-v1",
        "seed_rule": "base + opponent_index * 2 + (seat == o)",
        "minimax_tie_break": "lowest-index-v1",
    }
    assessment_protocol, _ = protocol_identity(protocol)
    evaluation_model = replace(model, protocol_id=assessment_protocol)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S_%f_UTC")
    assessment_id = f"{stamp}_{uuid4().hex[:8]}"
    directory = Path(output_root).expanduser().resolve() / assessment_id
    directory.mkdir(parents=True, exist_ok=False)
    batches: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "assessment_id": assessment_id,
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "model": asdict(model),
        "model_config": config,
        "device": str(policy.device),
        "protocol_id": assessment_protocol,
        "protocol": protocol,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
        "requested_batches": len(opponents) * len(seats) * len(seeds),
        "uncertainty": {
            "metric": "loss_rate",
            "method": "wilson",
            "confidence_level": CONFIDENCE_LEVEL,
            "units": "proportion",
            "scope": "Sampling uncertainty for a fixed model against the random opponent",
            "assumption": "Independent games with a fixed loss probability",
            "excluded_opponents": ["minimax"],
            "incomplete_batches": "Excluded from intervals and pooled summaries",
            "training_seed_variation": "Not estimated",
        },
        "batches": batches,
        "active_batch": None,
        "summary": [],
    }
    write_json(directory / "assessment.json", metadata)
    logger.info("Assessment directory: %s", display_project_path(directory))
    logger.info(
        "Evaluating %s from %s",
        model.experiment_name,
        display_project_path(model.checkpoint_path),
    )
    status = "completed"
    try:
        with (
            preserve_rng_state(),
            closing(connect_database(directory / "games.sqlite3")) as connection,
            (directory / "evaluation.csv").open(
                "x", newline="", encoding="utf-8"
            ) as stream,
        ):
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            stream.flush()
            combinations = (
                (base_seed, opponent_index, opponent, seat)
                for base_seed in seeds
                for opponent_index, opponent in enumerate(opponents)
                for seat in seats
            )
            for base_seed, opponent_index, opponent, seat in combinations:
                if stop():
                    status = "interrupted"
                    break
                seed = evaluation_seed(base_seed, opponent_index, seat)
                metadata["active_batch"] = {
                    "base_seed": base_seed,
                    "seed": seed,
                    "opponent": opponent,
                    "seat": seat,
                }
                write_json(directory / "assessment.json", metadata)
                result = evaluate_matches(
                    policy,
                    make_opponent(opponent, random.Random(seed)),
                    seat=seat,
                    games=games,
                    recorder=RunRecorder(connection),
                    agent_label=f"dqn:{model.experiment_run_id}",
                    opponent_label=opponent,
                    seed=seed,
                    model=evaluation_model,
                    should_stop=stop,
                )
                row = {
                    "base_seed": base_seed,
                    "seed": seed,
                    "opponent": opponent,
                    "seat": seat,
                    **asdict(result),
                    "average_return": result.average_return,
                    **_loss_statistics(
                        opponent,
                        result.games,
                        result.losses,
                        complete_batch=result.status == "completed",
                    ),
                }
                batches.append(row)
                metadata["active_batch"] = None
                writer.writerow(row)
                stream.flush()
                metadata["summary"] = _summarise(batches, opponents, seats, len(seeds))
                write_json(directory / "assessment.json", metadata)
                logger.info(
                    "vs %s, agent %s, base seed %s: %s/%s games (%s)",
                    opponent,
                    seat.upper(),
                    base_seed,
                    result.games,
                    games,
                    result.status,
                )
                if result.status != "completed":
                    status = "interrupted"
                    break
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as error:
        status = "failed"
        metadata["error"] = str(error)
        raise
    finally:
        metadata.update(
            {
                "status": status,
                "finished_at": utc_now(),
                "summary": _summarise(batches, opponents, seats, len(seeds)),
            }
        )
        write_json(directory / "assessment.json", metadata)
    return directory


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--agent", choices=("latest", "best"))
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--protocol", help="Registry protocol ID used to select best")
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=("random", "minimax"),
        default=["random", "minimax"],
    )
    parser.add_argument("--seats", nargs="+", choices=("x", "o"), default=["x", "o"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[12345])
    parser.add_argument(
        "--games", type=int, default=1000, help="Games per opponent, seat, and seed"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    if args.protocol is not None and args.agent != "best":
        parser.error("--protocol requires --agent best")
    try:
        _validate_settings(args.opponents, args.seats, args.seeds, args.games)
    except ValueError as error:
        parser.error(str(error))
    return args


def _format_loss_statistics(row: dict[str, Any]) -> str:
    rate = f"{row['loss_rate']:.2%}" if row["loss_rate"] is not None else "n/a"
    if row["loss_rate_ci_lower"] is not None:
        return f"loss {rate} (95% CI {row['loss_rate_ci_lower']:.2%}-{row['loss_rate_ci_upper']:.2%})"
    reason = str(row["loss_rate_ci_reason"]).replace("_", " ")
    return f"loss {rate} (95% CI N/A: {reason})"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    requested = False

    def request_stop(_signal: int, _frame: Any) -> None:
        nonlocal requested
        if requested:
            raise KeyboardInterrupt
        requested = True
        logging.info("Stop requested; finishing the current game")

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    try:
        directory = evaluate_checkpoint(
            args.checkpoint if args.checkpoint is not None else args.agent,
            opponents=args.opponents,
            seats=args.seats,
            seeds=args.seeds,
            games=args.games,
            device=args.device,
            registry_path=args.registry,
            protocol_id=args.protocol,
            output_root=args.output_root,
            should_stop=lambda: requested,
        )
    except Exception as error:
        logging.error("Offline evaluation failed: %s", error)
        return 1
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    metadata = json.loads((directory / "assessment.json").read_text(encoding="utf-8"))
    print(
        f"Agent: {metadata['model']['experiment_name']} | checkpoint {display_project_path(metadata['model']['checkpoint_path'])}"
    )
    for row in metadata["summary"]:
        average = (
            f"{row['average_return']:.3f}"
            if row["average_return"] is not None
            else "n/a"
        )
        print(
            f"vs {row['opponent']}, agent {row['seat'].upper()}: {row['wins']} wins, {row['draws']} draws, {row['losses']} losses | {_format_loss_statistics(row)} | return {average} | {row['completed_seed_batches']}/{row['requested_seed_batches']} complete seed batches"
        )
    print(f"Assessment {metadata['status']}: {display_project_path(directory)}")
    return 0 if metadata["status"] == "completed" else 130


if __name__ == "__main__":
    raise SystemExit(main())
