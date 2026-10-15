import pytest
import torch

from games.tictactoe.rules import TicTacToe
from players.tictactoe_player import TicTacToePlayerAdapter, build_tictactoe_model
from training.tictactoe.environment import TicTacToeAdapter


@pytest.mark.parametrize("player", [1, -1])
def test_encodes_current_players_perspective_without_mutating_game(player):
    game = TicTacToe()
    game.board = [player, -player, 0, 0, 0, 0, 0, 0, 0]
    game.current_player = player
    adapter = TicTacToePlayerAdapter()
    assert adapter.encode(game).tolist() == [[1, -1, 0, 0, 0, 0, 0, 0, 0]]
    assert adapter.legal_actions(game) == list(range(2, 9))
    assert adapter.decode_action(game, 2) == 2
    assert game.board[:2] == [player, -player]


def test_terminal_board_has_no_actions_even_with_empty_squares():
    game = TicTacToe()
    game.board = [1, 1, 1, -1, -1, 0, 0, 0, 0]
    adapter = TicTacToePlayerAdapter()
    assert adapter.legal_actions(game) == []
    with pytest.raises(ValueError, match="illegal"):
        adapter.decode_action(game, 5)


def test_builds_saved_architecture():
    model = build_tictactoe_model({"network": {"hidden_sizes": [8], "block_types": ["dense"], "activations": ["relu"]}})
    assert model(torch.zeros(1, 9)).shape == (1, 9)
    assert model.output.in_features == 8


def test_rejects_missing_network_configuration():
    with pytest.raises(ValueError, match="network configuration"):
        build_tictactoe_model({})


@pytest.mark.parametrize("player", [1, -1])
def test_training_and_saved_player_encodings_agree_after_moves(player):
    first_legal = lambda game: game.legal_actions()[0]
    environment = TicTacToeAdapter(first_legal, agent_player=player)
    observation = environment.reset()
    game = TicTacToe()
    if player == -1:
        game.step(0)
    adapter = TicTacToePlayerAdapter()
    assert adapter.encode(game).tolist() == [list(observation.board)]

    observation, _, _ = environment.step(4)
    game.step(4)
    game.step(first_legal(game))
    assert adapter.encode(game).tolist() == [list(observation.board)]
