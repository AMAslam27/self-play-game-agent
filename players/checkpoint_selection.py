"""Resolve explicit checkpoints, latest completed runs, or ranked best runs."""

import sqlite3
from datetime import datetime
from pathlib import Path

from evaluation.tictactoe_ranking import checkpoint_is_available, rank_checkpoints


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
