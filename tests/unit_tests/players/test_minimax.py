import pytest

from games.tictactoe.rules import winner
from players.minimax import minimax_policy, score_position
from tests.unit_tests.common.mock_utils import FakeGame


@pytest.mark.parametrize("player", [1, -1])
@pytest.mark.parametrize("line", [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
])
def test_winner_detects_each_line(player, line):
    board = [0] * 9
    for square in line:
        board[square] = player
    assert winner(board) == player


def test_winner_returns_empty_without_win():
    assert winner([1, -1, 0, 0, 1, 0, 0, 0, -1]) == 0


@pytest.mark.parametrize("bot,expected", [(1, 1), (-1, -1)])
def test_score_position_scores_win_from_bot_perspective(bot, expected):
    assert score_position([1, 1, 1, -1, -1, 0, 0, 0, 0], -1, bot) == expected


def test_score_position_scores_draw():
    assert score_position([1, -1, 1, 1, -1, -1, -1, 1, 1], -1, 1) == 0


@pytest.mark.parametrize("bot", [1, -1])
def test_policy_takes_unique_immediate_win_without_mutating_game(bot):
    board = [bot, bot, 0, -bot, -bot, 0, 0, 0, 0]
    game = FakeGame(board, bot)
    assert minimax_policy(game) == 2
    assert game.board == board
    assert game.current_player == bot


@pytest.mark.parametrize("bot", [1, -1])
def test_policy_blocks_immediate_loss(bot):
    game = FakeGame([-bot, -bot, 0, bot, 0, 0, 0, bot, 0], bot)
    assert minimax_policy(game) == 2


@pytest.mark.parametrize("board", [
    [1, 1, 1, -1, -1, 0, 0, 0, 0],
    [1, -1, 1, 1, -1, -1, -1, 1, 1],
])
def test_policy_rejects_terminal_game(board):
    with pytest.raises(ValueError, match="game ends"):
        minimax_policy(FakeGame(board, 1))


def test_policy_chooses_only_remaining_legal_move():
    game = FakeGame([1, -1, 1, 1, -1, -1, -1, 1, 0], 1)
    assert minimax_policy(game) == 8


def test_score_position_assumes_opponent_chooses_winning_reply():
    board = [-1, -1, 0, 1, 0, 0, 0, 1, 0]
    original = board.copy()
    assert score_position(board, -1, 1) == -1
    assert board == original
