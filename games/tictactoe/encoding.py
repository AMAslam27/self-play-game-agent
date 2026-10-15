"""Player-relative board features shared by training and saved players."""

from collections.abc import Sequence


def encode_board(board: Sequence[int], player: int) -> tuple[int, ...]:
    """Encode own pieces as +1, opposing pieces as -1, and empty cells as 0."""
    return tuple(cell * player for cell in board)
