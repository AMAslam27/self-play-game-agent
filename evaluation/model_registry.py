"""Shared checkpoint identities and baseline evaluation records.

Import completed Tic-Tac-Toe runs with:
    python -m evaluation.model_registry <run-directory> [<run-directory> ...]
"""

import argparse
import csv
import hashlib
import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from training.checkpoints import load_checkpoint

DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "results" / "models.sqlite3"
)
SCHEMA = """
CREATE TABLE IF NOT EXISTS model_runs (
    run_id TEXT NOT NULL,
    game TEXT NOT NULL,
    model_type TEXT NOT NULL,
    name TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status = 'completed'),
    PRIMARY KEY (run_id, game, model_type)
);
CREATE TABLE IF NOT EXISTS evaluation_protocols (
    protocol_id TEXT PRIMARY KEY,
    definition TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_checkpoints (
    checkpoint_hash TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    game TEXT NOT NULL,
    model_type TEXT NOT NULL,
    path TEXT NOT NULL,
    episode INTEGER NOT NULL CHECK (episode >= 0),
    updates INTEGER NOT NULL CHECK (updates >= 0),
    is_final INTEGER NOT NULL CHECK (is_final IN (0, 1)),
    protocol_id TEXT NOT NULL REFERENCES evaluation_protocols(protocol_id),
    FOREIGN KEY (run_id, game, model_type)
        REFERENCES model_runs(run_id, game, model_type)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_final_checkpoint
ON model_checkpoints(run_id, game, model_type) WHERE is_final = 1;
CREATE TABLE IF NOT EXISTS model_evaluations (
    checkpoint_hash TEXT NOT NULL REFERENCES model_checkpoints(checkpoint_hash),
    protocol_id TEXT NOT NULL REFERENCES evaluation_protocols(protocol_id),
    opponent TEXT NOT NULL,
    seat TEXT NOT NULL CHECK (seat IN ('x', 'o')),
    games INTEGER NOT NULL CHECK (games > 0),
    wins INTEGER NOT NULL CHECK (wins >= 0),
    draws INTEGER NOT NULL CHECK (draws >= 0),
    losses INTEGER NOT NULL CHECK (losses >= 0),
    CHECK (games = wins + draws + losses),
    PRIMARY KEY (checkpoint_hash, protocol_id, opponent, seat)
);
"""


def connect_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> sqlite3.Connection:
    """Open a registry, creating its schema when necessary."""
    database = Path(path)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA)
    except Exception:
        connection.close()
        raise
    return connection


def checkpoint_hash(path: str | Path) -> str:
    """Hash the checkpoint bytes, including weights and experiment identity."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def protocol_identity(definition: Mapping[str, Any]) -> tuple[str, str]:
    """Return a stable fingerprint and canonical JSON protocol definition."""
    encoded = json.dumps(
        dict(definition), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest(), encoded


def import_tictactoe_run(
    connection: sqlite3.Connection,
    run_dir: str | Path,
    *,
    evaluator_id: str = "legacy-tictactoe-baseline-v1",
) -> str:
    """Import a completed DQN run's final checkpoint and completed evaluations.

    Historical CSVs do not fingerprint opponent source code. evaluator_id
    explicitly groups runs believed to share evaluator implementations; use a
    different ID when those implementations change. Evaluation frequency and
    network/training settings do not affect the comparison protocol.

    The final numbered checkpoint is used, never the mutable latest.pt alias.
    Its identity, progress, and evaluation settings must match run metadata.
    Incomplete final sweeps can be imported but are ineligible for ranking.
    Reimporting replaces that checkpoint's evaluation rows atomically.
    """
    directory = Path(run_dir).resolve()
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise ValueError("Only completed runs can be registered")
    finished_at = metadata["finished_at"]
    timestamp = datetime.fromisoformat(finished_at)
    if timestamp.tzinfo is None:
        raise ValueError("Run completion time must include a timezone")
    finished_at = timestamp.isoformat()
    # Resolve by filename so archived runs can be relocated between machines.
    filename = str(metadata["last_checkpoint"]).replace("\\", "/").rsplit("/", 1)[-1]
    if filename == "latest.pt":
        raise ValueError("Registry requires an immutable numbered final checkpoint")
    path = directory / "checkpoints" / filename
    identity = checkpoint_hash(path)
    payload = load_checkpoint(path)
    episode = metadata["progress"]["episodes"]
    updates = metadata["learning_updates"]
    settings = metadata["config"]["evaluation"]
    if (
        payload.get("run_id") != metadata["run_id"]
        or payload.get("progress", {}).get("episodes") != episode
        or payload.get("agent", {}).get("learning_updates") != updates
        or payload.get("config", {}).get("evaluation") != settings
        or not isinstance(payload.get("agent", {}).get("online_network"), Mapping)
    ):
        raise ValueError("Final checkpoint does not match run metadata")
    if settings["epsilon"] != 0.0:
        raise ValueError("Ranking requires greedy evaluation")
    if not evaluator_id.strip():
        raise ValueError("Evaluator identity must be nonempty")
    protocol_id, definition = protocol_identity(
        {
            "game": "tictactoe",
            "evaluator_id": evaluator_id,
            "encoding": "current-player-relative-v1",
            "selection": "greedy-lowest-index-v1",
            "opponents": settings["opponents"],
            "seats": settings["seats"],
            "games_per_opponent_per_seat": settings["games_per_opponent_per_seat"],
            "seed": settings["seed"],
            "epsilon": settings["epsilon"],
        }
    )
    evaluations = {}
    with (directory / "evaluation.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (
                int(row["episode"]) != episode
                or int(row["update"]) != updates
                or row["status"] != "completed"
            ):
                continue
            key = (row["opponent"], row["seat"])
            if key in evaluations:
                raise ValueError("Duplicate final evaluation batch")
            counts = tuple(
                int(row[field]) for field in ("games", "wins", "draws", "losses")
            )
            if (
                key[0] not in settings["opponents"]
                or key[1] not in settings["seats"]
                or counts[0] != settings["games_per_opponent_per_seat"]
                or any(value < 0 for value in counts)
                or counts[0] <= 0
                or counts[0] != sum(counts[1:])
            ):
                raise ValueError("Invalid final evaluation outcomes")
            evaluations[key] = counts
    if checkpoint_hash(path) != identity:
        raise ValueError("Checkpoint changed during import")
    run_key = (metadata["run_id"], "tictactoe", "dqn")
    with connection:
        connection.execute(
            "INSERT INTO model_runs VALUES (?, ?, ?, ?, ?, 'completed') "
            "ON CONFLICT(run_id, game, model_type) DO UPDATE SET name=excluded.name, finished_at=excluded.finished_at",
            (*run_key, metadata["name"], finished_at),
        )
        connection.execute(
            "INSERT OR IGNORE INTO evaluation_protocols VALUES (?, ?)",
            (protocol_id, definition),
        )
        connection.execute(
            "UPDATE model_checkpoints SET is_final=0 WHERE run_id=? AND game=? AND model_type=?",
            run_key,
        )
        connection.execute(
            "INSERT INTO model_checkpoints VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?) "
            "ON CONFLICT(checkpoint_hash) DO UPDATE SET path=excluded.path, is_final=1, protocol_id=excluded.protocol_id",
            (identity, *run_key, str(path), episode, updates, protocol_id),
        )
        connection.execute(
            "DELETE FROM model_evaluations WHERE checkpoint_hash=?", (identity,)
        )
        connection.executemany(
            "INSERT INTO model_evaluations VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (identity, protocol_id, *key, *counts)
                for key, counts in evaluations.items()
            ],
        )
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--evaluator-id", default="legacy-tictactoe-baseline-v1")
    args = parser.parse_args()
    connection = connect_registry(args.registry)
    try:
        for directory in args.run_dirs:
            identity = import_tictactoe_run(
                connection, directory, evaluator_id=args.evaluator_id
            )
            print(f"Registered {directory.name}: {identity}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
