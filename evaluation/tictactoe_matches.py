"""Seeded Tic-Tac-Toe baseline matches shared by training and offline eval."""

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

from evaluation.recorder import ModelIdentity, RunRecorder
from games.tictactoe.play import play_game
from games.tictactoe.rules import TicTacToe
from players.minimax import minimax_policy
from training.utils import _require_positive_integer

Policy = Callable[[TicTacToe], int | None]
logger = logging.getLogger(__name__)


@lru_cache(maxsize=6000)
def _minimax_action(board: tuple[int, ...], current_player: int) -> int:
    game = TicTacToe()
    game.board = list(board)
    game.current_player = current_player
    return minimax_policy(game)


def make_opponent(name: str, rng: random.Random) -> Policy:
    if name == "random":
        return lambda game: rng.choice(game.legal_actions())
    if name == "minimax":
        return lambda game: _minimax_action(tuple(game.board), game.current_player)
    raise ValueError(f"Unsupported opponent: {name}")


def evaluation_seed(base_seed: int, opponent_index: int, seat: str) -> int:
    """Preserve the training evaluator's opponent/seat seed offsets."""
    return base_seed + opponent_index * 2 + (seat == "o")


@dataclass(frozen=True)
class MatchResult:
    games: int
    wins: int
    draws: int
    losses: int
    duration_seconds: float
    status: str
    db_run_id: int

    @property
    def average_return(self) -> float | None:
        return (self.wins - self.losses) / self.games if self.games else None


def evaluate_matches(
    policy: Policy,
    opponent: Policy,
    *,
    seat: str,
    games: int,
    recorder: RunRecorder,
    agent_label: str = "agent",
    opponent_label: str = "opponent",
    seed: int | None = None,
    model: ModelIdentity | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> MatchResult:
    """Record one batch, returning outcomes from the agent's perspective.

    Stop requests take effect between complete games. Exceptions preserve the
    original error and mark the recorded batch failed or interrupted. Neither
    policy may quit; evaluation requires a legal action on every turn.
    """
    _require_positive_integer(games, "games")
    if seat not in ("x", "o"):
        raise ValueError("Evaluation seat must be x or o")
    stop = should_stop if should_stop is not None else lambda: False
    plays_x = seat == "x"
    policy_x, policy_o = (policy, opponent) if plays_x else (opponent, policy)
    label_x, label_o = (
        (agent_label, opponent_label) if plays_x else (opponent_label, agent_label)
    )
    run_id = recorder.start_run(
        label_x,
        label_o,
        games,
        seed=seed,
        model_x=model if plays_x else None,
        model_o=model if not plays_x else None,
    )
    started = perf_counter()
    outcomes = {1: 0, 0: 0, -1: 0}
    status = "completed"
    try:
        for game_number in range(1, games + 1):
            if stop():
                status = "interrupted"
                break
            winner = play_game(policy_x, policy_o, verbose=False)
            if winner is None:
                raise ValueError(
                    "Evaluation policies must return a legal move, not quit"
                )
            recorder.record_game(game_number, winner)
            outcomes[winner if plays_x else -winner] += 1
        recorder.finish_run(status)
    except BaseException as error:
        try:
            if not recorder.finished:
                recorder.finish_run(
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                )
        except Exception:
            logger.exception("Could not preserve evaluation batch status")
        raise
    return MatchResult(
        games=sum(outcomes.values()),
        wins=outcomes[1],
        draws=outcomes[0],
        losses=outcomes[-1],
        duration_seconds=perf_counter() - started,
        status=status,
        db_run_id=run_id,
    )
