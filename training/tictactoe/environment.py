"""Agent-facing Tic-Tac-Toe transitions against a fixed opponent policy."""

from collections.abc import Callable
from dataclasses import dataclass

from games.tictactoe.play import random_policy
from games.tictactoe.rules import EMPTY, PLAYER_O, PLAYER_X, TicTacToe

OpponentPolicy = Callable[[TicTacToe], int | None]


@dataclass(frozen=True)
class Observation:
    """Own pieces are +1, opposing pieces -1; True marks a legal action.

    Both tuples use row-major square indices 0-8. Terminal observations have
    an entirely False mask, including wins with empty squares remaining.
    """

    board: tuple[int, ...]
    action_mask: tuple[bool, ...]


class TicTacToeAdapter:
    """Expose one transition per agent decision, including the opponent reply.

    Opponents use the existing policy(game) interface and must return a legal
    move without modifying the game. The default random policy uses Python's
    random generator; seed it externally for reproducible games.
    """

    def __init__(
        self,
        opponent: OpponentPolicy = random_policy,
        agent_player: int = PLAYER_X,
    ):
        self._validate_player(agent_player)
        self.opponent = opponent
        self.agent_player = agent_player
        self._game = TicTacToe()
        self._done = True

    def reset(self, agent_player: int | None = None) -> Observation:
        """Start an episode at the agent's first decision.

        Optionally change seats, allowing one adapter to train both X and O.
        When playing O, the opponent makes X's opening move before returning.
        """
        if agent_player is not None:
            self._validate_player(agent_player)
            self.agent_player = agent_player
        self._done = True
        self._game.reset()
        if self.agent_player == PLAYER_O:
            self._opponent_move()
        self._done = False
        return self._observation()

    def step(self, action: int) -> tuple[Observation, float, bool]:
        """Return (observation, reward, done) from the agent's perspective.

        Rewards are +1 for a win, -1 for a loss, and 0 for a draw or ongoing
        play. Terminal transitions must not bootstrap in the learning target.
        Call reset before stepping and after termination or an opponent error.
        """
        if self._done:
            raise RuntimeError("Call reset before stepping a new episode")
        if action not in self._game.legal_actions():
            raise ValueError(f"Illegal action: {action}")

        # If an opponent fails after our move, require reset rather than let
        # another agent action accidentally play on the opponent's turn.
        self._done = True
        self._game.step(action)
        if not self._game.is_terminal():
            self._opponent_move()
        self._done = self._game.is_terminal()
        winner = self._game.winner()
        reward = float(winner * self.agent_player) if winner != EMPTY else 0.0
        return self._observation(), reward, self._done

    def _opponent_move(self) -> None:
        action = self.opponent(self._game)
        if action is None:
            raise ValueError("Training opponent must return a legal move, not quit")
        self._game.step(action)

    def _observation(self) -> Observation:
        return Observation(
            board=tuple(cell * self.agent_player for cell in self._game.board),
            action_mask=tuple(
                not self._done and cell == EMPTY for cell in self._game.board
            ),
        )

    @staticmethod
    def _validate_player(player: int) -> None:
        if player not in (PLAYER_X, PLAYER_O):
            raise ValueError("Agent player must be PLAYER_X (1) or PLAYER_O (-1)")
