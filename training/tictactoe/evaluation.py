"""Fixed opponents and independent evaluation matches for trained agents."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager
from typing import Any

from evaluation.database import connect_database
from evaluation.recorder import RunRecorder
from evaluation.tictactoe_matches import (
    evaluate_matches,
    evaluation_seed,
    make_opponent,
)
from games.tictactoe.rules import TicTacToe
from players.saved_agent import SavedAgentPlayer
from players.tictactoe_player import TicTacToePlayerAdapter
from training.checkpoints import preserve_rng_state
from training.dqn import DQNAgent
from training.metrics import MetricsRecorder, read_metrics


@contextmanager
def _greedy_player(agent: DQNAgent) -> Iterator[SavedAgentPlayer[TicTacToe, int]]:
    """Use the inference player while preserving the training model's mode."""
    was_training = agent.online_network.training
    try:
        yield SavedAgentPlayer(
            agent.online_network, TicTacToePlayerAdapter(), device=agent.device
        )
    finally:
        agent.online_network.train(was_training)


def evaluate_agent(
    agent: DQNAgent,
    settings: Mapping[str, Any],
    metrics: MetricsRecorder,
    *,
    run_id: str,
    session: int,
    episode: int,
    decisions: int,
    should_stop: Callable[[], bool],
) -> bool:
    """Record fixed-seed matches; return False if the sweep was interrupted.

    Already completed match batches at this model version are skipped when
    resuming an interrupted evaluation. Partial batches remain as audit rows.
    """
    metrics.flush()
    completed = {
        (row["opponent"], row["seat"])
        for row in read_metrics(metrics.run_dir, "evaluation")
        if int(row["episode"]) == episode
        and int(row["update"]) == agent.learning_updates
        and row["status"] == "completed"
    }
    with (
        preserve_rng_state(),
        _greedy_player(agent) as policy,
        closing(connect_database(metrics.run_dir / "games.sqlite3")) as connection,
    ):
        for opponent_index, opponent in enumerate(settings["opponents"]):
            for seat in settings["seats"]:
                if (opponent, seat) in completed:
                    continue
                if should_stop():
                    return False
                seed = evaluation_seed(settings["seed"], opponent_index, seat)
                label = f"dqn:{run_id}:updates-{agent.learning_updates}"
                result = evaluate_matches(
                    policy,
                    make_opponent(opponent, random.Random(seed)),
                    seat=seat,
                    games=settings["games_per_opponent_per_seat"],
                    recorder=RunRecorder(connection),
                    agent_label=label,
                    opponent_label=opponent,
                    seed=seed,
                    should_stop=should_stop,
                )
                metrics.write(
                    "evaluation",
                    {
                        "session": session,
                        "episode": episode,
                        "decision": decisions,
                        "update": agent.learning_updates,
                        "opponent": opponent,
                        "seat": seat,
                        "games": result.games,
                        "wins": result.wins,
                        "draws": result.draws,
                        "losses": result.losses,
                        "average_return": result.average_return if result.games else "",
                        "duration_seconds": result.duration_seconds,
                        "status": result.status,
                        "db_run_id": result.db_run_id,
                    },
                )
                metrics.flush()
                if result.status != "completed":
                    return False
    return True
