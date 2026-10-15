from collections.abc import Sequence

EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1

WIN_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),  # rows
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),  # columns
    (0, 4, 8),
    (2, 4, 6),  # diagonals
)


def winner(board: Sequence[int]) -> int:
    """Return the winning player, or EMPTY when no line is complete."""
    for a, b, c in WIN_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return EMPTY


class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        """Start a new game and return the initial board."""
        self.board = [EMPTY] * 9
        self.current_player = PLAYER_X  # X always moves first
        return list(self.board)

    def legal_actions(self):
        """Indices (0-8) of empty squares."""
        return [i for i, x in enumerate(self.board) if x == EMPTY]

    def step(self, action):
        """
        Current player plays at `action`.
        Returns (board, reward, done), where reward is from the perspective
        of the player who just moved: +1 win, 0 draw / game continues.
        """
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action: {action}")

        self.board[action] = self.current_player
        win = self.winner()

        if win != EMPTY:
            reward, done = 1.0, True
        elif not self.legal_actions():
            reward, done = 0.0, True  # draw
        else:
            reward, done = 0.0, False
            self.current_player = -self.current_player  # swap turns

        return list(self.board), reward, done

    def is_terminal(self):
        """Game is over if someone has won or the board is full."""
        return self.winner() != EMPTY or not self.legal_actions()

    def winner(self):
        """Return PLAYER_X, PLAYER_O, or EMPTY (0) if there is no winner."""
        return winner(self.board)

    def render(self):
        symbols = {PLAYER_X: "X", PLAYER_O: "O", EMPTY: "."}
        for r in range(3):
            print(" ".join(symbols[v] for v in self.board[r * 3 : r * 3 + 3]))
        print()
