import random
from dataclasses import FrozenInstanceError

import pytest

from games.tictactoe.rules import PLAYER_O, PLAYER_X
from players.minimax import minimax_policy
from training.tictactoe.environment import TicTacToeAdapter


def scripted_policy(moves):
    actions = iter(moves)
    return lambda game: next(actions)


def test_reset_as_x_starts_with_empty_board_without_calling_opponent():
    adapter = TicTacToeAdapter(scripted_policy([]))
    observation = adapter.reset()
    assert observation.board == (0,) * 9
    assert observation.action_mask == (True,) * 9


def test_reset_as_o_plays_opponent_opening_and_normalizes_board():
    adapter = TicTacToeAdapter(scripted_policy([4]), PLAYER_O)
    observation = adapter.reset()
    assert observation.board == (0, 0, 0, 0, -1, 0, 0, 0, 0)
    assert observation.action_mask == (True, True, True, True, False, True, True, True, True)


@pytest.mark.parametrize("seat,opponent_moves", [(PLAYER_X, [4]), (PLAYER_O, [4, 8])])
def test_step_includes_opponent_reply_and_returns_agent_perspective(seat, opponent_moves):
    seen_players = []
    policy = scripted_policy(opponent_moves)

    def opponent(game):
        seen_players.append(game.current_player)
        return policy(game)

    adapter = TicTacToeAdapter(opponent, seat)
    initial = adapter.reset()
    observation, reward, done = adapter.step(0)
    assert observation.board[0] == 1
    assert observation.board[4] == -1
    assert sum(cell != 0 for cell in observation.board) == (2 if seat == PLAYER_X else 3)
    assert observation.action_mask == tuple(cell == 0 for cell in observation.board)
    assert reward == 0.0
    assert done is False
    assert seen_players == [-seat] * len(opponent_moves)
    assert initial.board[0] == 0  # earlier observations remain unchanged


@pytest.mark.parametrize(
    "seat,agent_moves,opponent_moves,expected_reward",
    [
        (PLAYER_X, [0, 1, 2], [3, 4], 1.0),
        (PLAYER_O, [0, 1, 2], [3, 4, 8], 1.0),
        (PLAYER_X, [0, 1, 8], [3, 4, 5], -1.0),
        (PLAYER_O, [3, 4], [0, 1, 2], -1.0),
        (PLAYER_X, [0, 2, 3, 7, 8], [1, 4, 5, 6], 0.0),
        (PLAYER_O, [1, 4, 5, 6], [0, 2, 3, 7, 8], 0.0),
    ],
)
def test_terminal_outcomes_in_both_seats(seat, agent_moves, opponent_moves, expected_reward):
    adapter = TicTacToeAdapter(scripted_policy(opponent_moves), seat)
    adapter.reset()
    for index, action in enumerate(agent_moves):
        observation, reward, done = adapter.step(action)
        if index < len(agent_moves) - 1:
            assert done is False
            assert reward == 0.0
    assert reward == expected_reward
    assert done is True
    assert observation.action_mask == (False,) * 9
    with pytest.raises(RuntimeError, match="reset"):
        adapter.step(next((i for i, cell in enumerate(observation.board) if cell == 0), 0))


def test_step_requires_reset():
    with pytest.raises(RuntimeError, match="reset"):
        TicTacToeAdapter().step(0)


def test_reset_after_terminal_episode_allows_new_game():
    adapter = TicTacToeAdapter(scripted_policy([3, 4, 8]))
    adapter.reset()
    for action in [0, 1, 2]:
        _, _, done = adapter.step(action)
    assert done is True
    assert adapter.reset().action_mask == (True,) * 9
    assert adapter.step(0)[2] is False


@pytest.mark.parametrize("action", [-1, 9, 4])
def test_illegal_agent_move_does_not_advance_episode(action):
    adapter = TicTacToeAdapter(scripted_policy([4, 8]), PLAYER_O)
    adapter.reset()
    with pytest.raises(ValueError, match="Illegal action"):
        adapter.step(action)
    observation, _, _ = adapter.step(0)
    assert observation.board == (1, 0, 0, 0, -1, 0, 0, 0, -1)


@pytest.mark.parametrize("seat", [0, 2, -2])
def test_invalid_seats_are_rejected(seat):
    with pytest.raises(ValueError, match="Agent player"):
        TicTacToeAdapter(agent_player=seat)
    adapter = TicTacToeAdapter(scripted_policy([4]))
    adapter.reset()
    with pytest.raises(ValueError, match="Agent player"):
        adapter.reset(agent_player=seat)
    assert adapter.step(0)[0].board[0] == 1


def test_reset_discards_previous_episode_and_can_change_seats():
    adapter = TicTacToeAdapter(scripted_policy([4, 8, 2]))
    adapter.reset()
    adapter.step(0)
    assert adapter.reset().board == (0,) * 9
    observation = adapter.reset(agent_player=PLAYER_O)
    assert observation.board == (0, 0, 0, 0, 0, 0, 0, 0, -1)
    assert adapter.step(0)[0].board[0] == 1


def test_observation_is_immutable():
    observation = TicTacToeAdapter().reset()
    with pytest.raises(FrozenInstanceError):
        observation.board = (1,) * 9
    with pytest.raises(TypeError):
        observation.action_mask[0] = False


@pytest.mark.parametrize("opponent_action", [None, 0, 9])
def test_opponent_quit_or_illegal_move_requires_reset(opponent_action):
    adapter = TicTacToeAdapter(lambda game: opponent_action)
    adapter.reset()
    with pytest.raises(ValueError):
        adapter.step(0)
    with pytest.raises(RuntimeError, match="reset"):
        adapter.step(1)
    assert adapter.reset().board == (0,) * 9


def test_failed_opponent_opening_requires_reset():
    adapter = TicTacToeAdapter(lambda game: None, PLAYER_O)
    with pytest.raises(ValueError, match="not quit"):
        adapter.reset()
    with pytest.raises(RuntimeError, match="reset"):
        adapter.step(0)


@pytest.mark.parametrize("seat", [PLAYER_X, PLAYER_O])
def test_existing_minimax_policy_works_through_complete_episode(seat):
    adapter = TicTacToeAdapter(minimax_policy, seat)
    observation = adapter.reset()
    done = False
    while not done:
        action = next(i for i, legal in enumerate(observation.action_mask) if legal)
        observation, reward, done = adapter.step(action)
    assert reward in (-1.0, 0.0)


def test_default_random_opponent_is_reproducible_with_external_seed():
    def trajectory():
        adapter = TicTacToeAdapter()
        observation = adapter.reset()
        transitions = []
        done = False
        while not done:
            action = next(i for i, legal in enumerate(observation.action_mask) if legal)
            observation, reward, done = adapter.step(action)
            transitions.append((observation, reward, done))
        return transitions

    random_state = random.getstate()
    try:
        random.seed(42)
        first = trajectory()
        random.seed(42)
        assert trajectory() == first
    finally:
        random.setstate(random_state)
