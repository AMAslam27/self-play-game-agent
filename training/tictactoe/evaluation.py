"""Fixed opponents and independent evaluation matches for trained agents."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from contextlib import closing
from functools import lru_cache
from time import perf_counter
from typing import Any

from evaluation.database import connect_database
from evaluation.recorder import RunRecorder
from games.tictactoe.rules import PLAYER_O, PLAYER_X, TicTacToe
from players.minimax import minimax_policy
from training.checkpoints import preserve_rng_state
from training.dqn import DQNAgent
from training.metrics import MetricsRecorder, read_metrics
from training.tictactoe.environment import OpponentPolicy, TicTacToeAdapter


@lru_cache(maxsize=6000)
def _minimax_action(board: tuple[int, ...], current_player: int) -> int:
    game = TicTacToe()
    game.board = list(board)
    game.current_player = current_player
    return minimax_policy(game)


def make_opponent(name: str, rng: random.Random) -> OpponentPolicy:
    if name == "random":
        return lambda game: rng.choice(game.legal_actions())
    if name == "minimax":
        return lambda game: _minimax_action(tuple(game.board), game.current_player)
    raise ValueError(f"Unsupported opponent: {name}")


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
        closing(connect_database(metrics.run_dir / "games.sqlite3")) as connection,
    ):
        for opponent_index, opponent in enumerate(settings["opponents"]):
            for seat in settings["seats"]:
                if (opponent, seat) in completed:
                    continue
                if should_stop():
                    return False
                player = PLAYER_X if seat == "x" else PLAYER_O
                seed = settings["seed"] + opponent_index * 2 + (seat == "o")
                environment = TicTacToeAdapter(
                    make_opponent(opponent, random.Random(seed)), player
                )
                label = f"dqn:{run_id}:updates-{agent.learning_updates}"
                policy_x, policy_o = (
                    (label, opponent) if player == PLAYER_X else (opponent, label)
                )
                recorder = RunRecorder(connection)
                recorder.start_run(
                    policy_x,
                    policy_o,
                    settings["games_per_opponent_per_seat"],
                    seed=seed,
                )
                started = perf_counter()
                outcomes = {1: 0, 0: 0, -1: 0}
                status = "completed"
                try:
                    for game_number in range(
                        1, settings["games_per_opponent_per_seat"] + 1
                    ):
                        if should_stop():
                            status = "interrupted"
                            break
                        observation = environment.reset()
                        done = False
                        while not done:
                            action = agent.select_action(observation, epsilon=0.0)
                            observation, reward, done = environment.step(action)
                        outcomes[int(reward)] += 1
                        recorder.record_game(game_number, int(reward) * player)
                    recorder.finish_run(status)
                except BaseException as error:
                    recorder.finish_run(
                        "interrupted"
                        if isinstance(error, KeyboardInterrupt)
                        else "failed"
                    )
                    raise
                games = sum(outcomes.values())
                metrics.write(
                    "evaluation",
                    {
                        "session": session,
                        "episode": episode,
                        "decision": decisions,
                        "update": agent.learning_updates,
                        "opponent": opponent,
                        "seat": seat,
                        "games": games,
                        "wins": outcomes[1],
                        "draws": outcomes[0],
                        "losses": outcomes[-1],
                        "average_return": (outcomes[1] - outcomes[-1]) / games
                        if games
                        else "",
                        "duration_seconds": perf_counter() - started,
                        "status": status,
                        "db_run_id": recorder.run_id,
                    },
                )
                metrics.flush()
                if status != "completed":
                    return False
    return True
