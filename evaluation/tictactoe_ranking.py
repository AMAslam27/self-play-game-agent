"""Tic-Tac-Toe-specific baseline ranking of final checkpoints."""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from evaluation.model_registry import checkpoint_hash

RANKING_VERSION = "tictactoe-baseline-v1"


@dataclass(frozen=True)
class RankedCheckpoint:
    run_id: str
    name: str
    path: Path
    checkpoint_hash: str
    protocol_id: str
    finished_at: str
    worst_minimax_loss_rate: float
    worst_random_return: float
    mean_random_return: float
    ranking_version: str = RANKING_VERSION

    @property
    def explanation(self) -> str:
        return (
            f"Worst-seat minimax loss rate: {self.worst_minimax_loss_rate:.1%}; "
            f"worst-seat random return: {self.worst_random_return:.3f}; "
            f"mean random return: {self.mean_random_return:.3f}"
        )


def checkpoint_is_available(path: Path, identity: str) -> bool:
    """Missing or changed files must never inherit recorded evaluation scores."""
    try:
        return checkpoint_hash(path) == identity
    except OSError:
        return False


def rank_checkpoints(
    connection: sqlite3.Connection,
    *,
    game: str = "tictactoe",
    model_type: str = "dqn",
    protocol_id: str | None = None,
) -> list[RankedCheckpoint]:
    """Rank eligible checkpoints, requiring an explicit protocol if ambiguous.

    This initial ranking supports Tic-Tac-Toe only. Both seats and both fixed
    opponents must have full evaluation batches under one protocol. Changed or
    missing checkpoints are excluded. With no eligible models the list is empty.
    """
    if game != "tictactoe":
        raise ValueError("No ranking policy is defined for this game")
    rows = connection.execute(
        "SELECT c.*, r.name, r.finished_at, p.definition FROM model_checkpoints c "
        "JOIN model_runs r USING(run_id, game, model_type) "
        "JOIN evaluation_protocols p USING(protocol_id) "
        "WHERE c.is_final=1 AND r.status='completed' AND c.game=? AND c.model_type=?",
        (game, model_type),
    ).fetchall()
    candidates = []
    required = {
        (opponent, seat) for opponent in ("random", "minimax") for seat in ("x", "o")
    }
    for row in rows:
        if protocol_id is not None and row["protocol_id"] != protocol_id:
            continue
        protocol = json.loads(row["definition"])
        if set(protocol["opponents"]) != {"random", "minimax"} or set(
            protocol["seats"]
        ) != {"x", "o"}:
            continue
        outcomes = connection.execute(
            "SELECT * FROM model_evaluations WHERE checkpoint_hash=? AND protocol_id=?",
            (row["checkpoint_hash"], row["protocol_id"]),
        ).fetchall()
        batches = {(batch["opponent"], batch["seat"]): batch for batch in outcomes}
        if set(batches) != required or any(
            batch["games"] != protocol["games_per_opponent_per_seat"]
            for batch in outcomes
        ):
            continue
        path = Path(row["path"])
        if not checkpoint_is_available(path, row["checkpoint_hash"]):
            continue
        minimax_losses = [
            batches["minimax", seat]["losses"] / batches["minimax", seat]["games"]
            for seat in ("x", "o")
        ]
        random_returns = [
            (batches["random", seat]["wins"] - batches["random", seat]["losses"])
            / batches["random", seat]["games"]
            for seat in ("x", "o")
        ]
        candidates.append(
            RankedCheckpoint(
                run_id=row["run_id"],
                name=row["name"],
                path=path,
                checkpoint_hash=row["checkpoint_hash"],
                protocol_id=row["protocol_id"],
                finished_at=row["finished_at"],
                worst_minimax_loss_rate=max(minimax_losses),
                worst_random_return=min(random_returns),
                mean_random_return=sum(random_returns) / 2,
            )
        )
    if protocol_id is None and len({item.protocol_id for item in candidates}) > 1:
        raise ValueError(
            "Multiple evaluation protocols are eligible; specify protocol_id"
        )
    return sorted(
        candidates,
        key=lambda item: (
            item.worst_minimax_loss_rate,
            -item.worst_random_return,
            -item.mean_random_return,
            -datetime.fromisoformat(item.finished_at).timestamp(),
            item.run_id,
        ),
    )
