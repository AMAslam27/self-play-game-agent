import random
from contextlib import closing

import pytest

from evaluation import tictactoe_matches as matches
from evaluation.database import connect_database
from evaluation.recorder import RunRecorder
from games.tictactoe.rules import PLAYER_X, PLAYER_O
from models.tictactoe_q_network import TicTacToeQNetwork
from players.saved_agent import SavedAgentPlayer
from players.tictactoe_player import TicTacToePlayerAdapter
from training.dqn import DQNAgent
from training.metrics import MetricsRecorder
from training.tictactoe.environment import TicTacToeAdapter
from training.tictactoe.evaluation import evaluate_agent


@pytest.mark.parametrize("seat", ["x", "o"])
def test_counts_outcomes_from_agent_perspective(tmp_path, seat):
    with closing(connect_database(tmp_path / "games.sqlite3")) as connection:
        policy = lambda game: game.legal_actions()[0]
        result = matches.evaluate_matches(policy, policy, seat=seat, games=3, recorder=RunRecorder(connection))
        assert result.games == 3
        assert result.wins == (3 if seat == "x" else 0)
        assert result.losses == (0 if seat == "x" else 3)
        assert result.average_return == (1 if seat == "x" else -1)
        assert connection.execute("SELECT winner FROM games").fetchall() == [(PLAYER_X,)] * 3
        assert connection.execute("SELECT status FROM runs").fetchone()[0] == "completed"


@pytest.mark.parametrize("seat", ["x", "o"])
@pytest.mark.parametrize("opponent", ["random", "minimax"])
def test_matches_previous_training_loop_game_for_game(tmp_path, seat, opponent):
    agent = DQNAgent(TicTacToeQNetwork(), device="cpu")
    player = PLAYER_X if seat == "x" else PLAYER_O
    seed = 12345
    environment = TicTacToeAdapter(matches.make_opponent(opponent, random.Random(seed)), player)
    expected = []
    for _ in range(12):
        observation = environment.reset()
        done = False
        while not done:
            observation, reward, done = environment.step(agent.select_action(observation, epsilon=0.0))
        expected.append((int(reward) * player,))
    policy = SavedAgentPlayer(agent.online_network, TicTacToePlayerAdapter(), device="cpu")
    with closing(connect_database(tmp_path / "games.sqlite3")) as connection:
        matches.evaluate_matches(policy, matches.make_opponent(opponent, random.Random(seed)),
                                 seat=seat, games=12, recorder=RunRecorder(connection))
        assert connection.execute("SELECT winner FROM games ORDER BY game_number").fetchall() == expected


@pytest.mark.parametrize("error,status", [(RuntimeError("policy failed"), "failed"), (KeyboardInterrupt(), "interrupted")])
def test_preserves_policy_errors_and_records_status(tmp_path, error, status):
    def broken_policy(game):
        raise error
    with closing(connect_database(tmp_path / "games.sqlite3")) as connection:
        with pytest.raises(type(error)):
            matches.evaluate_matches(broken_policy, broken_policy, seat="x", games=2, recorder=RunRecorder(connection))
        assert connection.execute("SELECT status FROM runs").fetchone()[0] == status


def test_stop_preserves_completed_games_and_partial_counts(tmp_path):
    with closing(connect_database(tmp_path / "games.sqlite3")) as connection:
        stop = lambda: connection.execute("SELECT count(*) FROM games").fetchone()[0] >= 2
        policy = lambda game: game.legal_actions()[0]
        result = matches.evaluate_matches(policy, policy, seat="o", games=5, recorder=RunRecorder(connection), should_stop=stop)
        assert (result.status, result.games, result.losses) == ("interrupted", 2, 2)
        assert connection.execute("SELECT status FROM runs").fetchone()[0] == "interrupted"


def test_stop_before_first_game_has_no_fabricated_return(tmp_path):
    with closing(connect_database(tmp_path / "games.sqlite3")) as connection:
        result = matches.evaluate_matches(lambda game: None, lambda game: None,
                                         seat="x", games=2, recorder=RunRecorder(connection), should_stop=lambda: True)
        assert result.status == "interrupted"
        assert result.games == 0 and result.average_return is None


def test_evaluation_policy_must_not_quit(tmp_path):
    with closing(connect_database(tmp_path / "games.sqlite3")) as connection:
        with pytest.raises(ValueError, match="not quit"):
            matches.evaluate_matches(lambda game: None, lambda game: None,
                                     seat="x", games=2, recorder=RunRecorder(connection))
        assert connection.execute("SELECT status FROM runs").fetchone()[0] == "failed"


@pytest.mark.parametrize("training", [True, False])
def test_training_evaluation_restores_model_mode_on_error(tmp_path, monkeypatch, training):
    agent = DQNAgent(TicTacToeQNetwork(), device="cpu")
    agent.online_network.train(training)
    def broken_game(*args, **kwargs):
        raise RuntimeError("evaluation failed")
    monkeypatch.setattr(matches, "play_game", broken_game)
    with MetricsRecorder(tmp_path) as metrics:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            evaluate_agent(agent, {"seed": 1, "opponents": ["random"], "seats": ["x"],
                                   "games_per_opponent_per_seat": 2}, metrics,
                           run_id="test", session=1, episode=0, decisions=0, should_stop=lambda: False)
    assert agent.online_network.training == training
