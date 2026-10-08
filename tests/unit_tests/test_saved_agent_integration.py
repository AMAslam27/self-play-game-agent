import json
from contextlib import closing

import pytest
import torch

import runner
from evaluation.database import connect_database
from evaluation.model_registry import checkpoint_hash, connect_registry
from games.tictactoe.rules import TicTacToe
from players.checkpoint_selection import resolve_checkpoint
from tests.unit_tests.training.test_training import short_config
from training.tictactoe import train


@pytest.mark.parametrize("arguments", [
    ["--o", "dqn"],
    ["--o-agent", "latest"],
    ["--o", "dqn", "--o-agent", "latest", "--o-checkpoint", "model.pt"],
    ["--o", "dqn", "--o-agent", "latest", "--o-protocol", "abc"],
])
def test_runner_rejects_inconsistent_saved_agent_arguments(monkeypatch, arguments):
    monkeypatch.setattr("sys.argv", ["runner.py", *arguments])
    with pytest.raises(SystemExit) as error:
        runner.parse_args()
    assert error.value.code == 2


@pytest.fixture
def completed_run(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "plot_training_metrics", lambda *args: None)
    registry_path = tmp_path / "models.sqlite3"
    config = short_config(tmp_path / "runs", episodes=4)
    config["evaluation"]["opponents"] = ["random", "minimax"]
    directory = train.run_training(config, registry_path=registry_path)
    metadata = json.loads((directory / "run.json").read_text())
    return directory, registry_path, metadata


def test_training_registers_final_evaluation_and_supports_best(completed_run):
    directory, registry_path, metadata = completed_run
    assert metadata["model_registry"]["status"] == "registered"
    with closing(connect_registry(registry_path)) as registry:
        selected = resolve_checkpoint("best", connection=registry)
        assert selected == directory / "checkpoints" / selected.name
        assert checkpoint_hash(selected) == metadata["model_registry"]["checkpoint_hash"]


@pytest.mark.parametrize("selection", ["latest", "best", "explicit"])
def test_runner_plays_two_saved_models_and_records_identity(completed_run, monkeypatch, tmp_path, capsys, selection):
    directory, registry_path, metadata = completed_run
    path = directory / "checkpoints" / metadata["last_checkpoint"].replace("\\", "/").rsplit("/", 1)[-1]
    source = ["--o-checkpoint", str(path)] if selection == "explicit" else ["--o-agent", selection]
    database_path = tmp_path / "matches.sqlite3"
    monkeypatch.setattr(runner, "connect_database", connect_database)
    monkeypatch.setattr(runner, "save_run_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.argv", ["runner.py", "--x", "dqn", "--x-checkpoint", str(path),
        "--o", "dqn", *source, "--registry", str(registry_path), "--agent-device", "cpu",
        "--db-file", str(database_path), "--games", "2", "--quiet"])
    runner.main()
    with closing(connect_database(database_path)) as connection:
        assert connection.execute("SELECT count(*) FROM games").fetchone()[0] == 2
        records = connection.execute("SELECT seat, experiment_run_id, checkpoint_hash, selection FROM run_models ORDER BY CASE seat WHEN 'x' THEN 0 ELSE 1 END").fetchall()
        assert len(records) == 2
        assert all(row[1] == metadata["run_id"] and row[2] == checkpoint_hash(path) for row in records)
        assert records[1][3] == (str(path) if selection == "explicit" else selection)
    output = capsys.readouterr().out
    assert "O agent:" in output and "SHA-256" in output
    assert "2/2 games completed" in output


def test_runner_human_against_saved_agent(completed_run, monkeypatch, tmp_path):
    _, registry_path, _ = completed_run
    monkeypatch.setattr(runner, "connect_database", connect_database)
    monkeypatch.setattr(runner, "save_run_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "human_policy", lambda game: game.legal_actions()[0])
    monkeypatch.setitem(runner.POLICIES, "human", runner.human_policy)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    monkeypatch.setattr("sys.argv", ["runner.py", "--x", "human", "--o", "dqn", "--o-agent", "latest",
        "--registry", str(registry_path), "--agent-device", "cpu", "--quiet", "--db-file", str(tmp_path / "human.sqlite3")])
    runner.main()
    with closing(connect_database(tmp_path / "human.sqlite3")) as connection:
        assert connection.execute("SELECT count(*) FROM games").fetchone()[0] == 1
        assert connection.execute("SELECT seat FROM run_models").fetchone()[0] == "o"


def test_loaded_model_does_not_learn_or_change_weights(completed_run, monkeypatch):
    _, registry_path, _ = completed_run
    monkeypatch.setattr("sys.argv", ["runner.py", "--x", "dqn", "--x-agent", "latest", "--registry", str(registry_path), "--agent-device", "cpu"])
    args = runner.parse_args()
    policy, _ = runner.prepare_player(args, "x")
    before = {key: value.clone() for key, value in policy.model.state_dict().items()}
    game = TicTacToe()
    for _ in range(3):
        assert policy(game) in game.legal_actions()
    assert all(torch.equal(value, before[key]) for key, value in policy.model.state_dict().items())
    assert all(parameter.grad is None for parameter in policy.model.parameters())


def test_resume_interruption_removes_previous_final_from_selection(completed_run, monkeypatch):
    directory, registry_path, metadata = completed_run
    checkpoint = directory / "checkpoints" / "latest.pt"
    config = train.load_checkpoint(checkpoint)["config"]
    config["training"]["episodes"] = 6
    train.run_training(config, resume=checkpoint, should_stop=lambda: True, registry_path=registry_path)
    assert json.loads((directory / "run.json").read_text())["status"] == "interrupted"
    assert json.loads((directory / "run.json").read_text())["model_registry"]["status"] == "not_registered"
    with closing(connect_registry(registry_path)) as registry:
        with pytest.raises(ValueError, match="compatible run"):
            resolve_checkpoint("latest", connection=registry)


def test_saved_agent_against_minimax(completed_run, monkeypatch, tmp_path):
    _, registry_path, _ = completed_run
    database_path = tmp_path / "baseline.sqlite3"
    monkeypatch.setattr(runner, "connect_database", connect_database)
    monkeypatch.setattr(runner, "save_run_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.argv", ["runner.py", "--x", "dqn", "--x-agent", "best",
        "--o", "minimax", "--registry", str(registry_path), "--agent-device", "cpu",
        "--quiet", "--db-file", str(database_path)])
    runner.main()
    with closing(connect_database(database_path)) as connection:
        assert connection.execute("SELECT count(*) FROM games").fetchone()[0] == 1
        assert connection.execute("SELECT seat FROM run_models").fetchone()[0] == "x"


def test_bad_checkpoint_is_reported_before_creating_match_results(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["runner.py", "--o", "dqn", "--o-checkpoint", str(tmp_path / "missing.pt")])
    with pytest.raises(SystemExit, match="Cannot prepare saved agent"):
        runner.main()


def test_registration_failure_preserves_completed_training(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "plot_training_metrics", lambda *args: None)
    def fail_registration(*args):
        raise OSError("Registry write failed")
    monkeypatch.setattr(train, "import_tictactoe_run", fail_registration)
    directory = train.run_training(short_config(tmp_path / "runs", episodes=2), registry_path=tmp_path / "models.sqlite3")
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["model_registry"]["status"] == "failed"
    assert "Registry write failed" in metadata["model_registry"]["error"]


def test_existing_result_database_keeps_its_records(tmp_path):
    database = tmp_path / "old.sqlite3"
    import sqlite3
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY, started_at TEXT, finished_at TEXT, policy_x TEXT, policy_o TEXT, requested_games INTEGER, seed INTEGER, status TEXT)")
        connection.execute("INSERT INTO runs VALUES (1, 'before', 'after', 'random', 'minimax', 1, NULL, 'completed')")
    with closing(connect_database(database)) as connection:
        assert connection.execute("SELECT policy_o FROM runs WHERE id=1").fetchone()[0] == "minimax"
        assert connection.execute("SELECT count(*) FROM run_models").fetchone()[0] == 0
