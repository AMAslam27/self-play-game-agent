import csv
import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest
import torch

from evaluation import offline_eval
from evaluation.database import PROJECT_ROOT, connect_database
from evaluation.model_registry import checkpoint_hash
from tests.unit_tests.training.test_training import short_config
from training.checkpoints import capture_rng_state
from training.tictactoe import train


@pytest.fixture
def trained_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "plot_training_metrics", lambda *args: None)
    config = short_config(tmp_path / "training-runs", episodes=2)
    config["evaluation"]["opponents"] = ["random", "minimax"]
    registry_path = tmp_path / "models.sqlite3"
    directory = train.run_training(config, registry_path=registry_path)
    metadata = json.loads((directory / "run.json").read_text())
    return Path(metadata["last_checkpoint"]), registry_path


def read_assessment(directory):
    metadata = json.loads((directory / "assessment.json").read_text())
    with (directory / "evaluation.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return metadata, rows


def test_multiple_seeds_and_seats_load_once_and_preserve_training(trained_checkpoint, tmp_path, monkeypatch):
    checkpoint, registry_path = trained_checkpoint
    source = checkpoint.parent.parent
    before = {path.relative_to(source): checkpoint_hash(path) for path in source.rglob("*") if path.is_file()}
    registry_before = registry_path.read_bytes()
    calls = []
    original_loader = offline_eval.load_model_checkpoint
    def load_once(*args, **kwargs):
        calls.append(args[0])
        return original_loader(*args, **kwargs)
    monkeypatch.setattr(offline_eval, "load_model_checkpoint", load_once)
    directory = offline_eval.evaluate_checkpoint(checkpoint, seeds=[11, 22], games=3,
        device="cpu", output_root=tmp_path / "assessments")
    metadata, rows = read_assessment(directory)
    assert metadata["status"] == "completed"
    assert metadata["schema_version"] == 2
    assert metadata["uncertainty"]["confidence_level"] == 0.95
    assert len(rows) == len(metadata["batches"]) == metadata["requested_batches"] == 8
    assert len(calls) == 1
    assert all(int(row["games"]) == 3 and sum(int(row[field]) for field in ("wins", "draws", "losses")) == 3 for row in rows)
    assert all(row["games"] == 6 and row["completed_seed_batches"] == 2 for row in metadata["summary"])
    for row in metadata["summary"]:
        assert row["loss_rate"] == row["losses"] / row["games"]
        if row["opponent"] == "random":
            assert 0 <= row["loss_rate_ci_lower"] <= row["loss_rate"] <= row["loss_rate_ci_upper"] <= 1
            assert row["loss_rate_ci_method"] == "wilson"
        else:
            assert row["loss_rate_ci_lower"] is None and row["loss_rate_ci_upper"] is None
            assert row["loss_rate_ci_reason"] == "deterministic_opponent"
    assert all(row["loss_rate_ci_lower"] != "" for row in rows if row["opponent"] == "random")
    assert all(row["loss_rate_ci_lower"] == "" and row["loss_rate_ci_reason"] == "deterministic_opponent" for row in rows if row["opponent"] == "minimax")
    with closing(connect_database(directory / "games.sqlite3")) as connection:
        assert connection.execute("SELECT count(*) FROM games").fetchone()[0] == 24
        identities = connection.execute("SELECT checkpoint_hash, protocol_id FROM run_models").fetchall()
        assert len(identities) == 8
        assert all(row[0] == checkpoint_hash(checkpoint) for row in identities)
        assert all(row[1] == metadata["protocol_id"] for row in identities)
    after = {path.relative_to(source): checkpoint_hash(path) for path in source.rglob("*") if path.is_file()}
    assert before == after
    assert registry_path.read_bytes() == registry_before


def test_repeatable_games_preserve_global_rng_and_prior_assessments(trained_checkpoint, tmp_path):
    checkpoint, _ = trained_checkpoint
    state = capture_rng_state()
    directories = [offline_eval.evaluate_checkpoint(checkpoint, opponents=["random"], seeds=[19, 29],
        games=12, device="cpu", output_root=tmp_path / "assessments") for _ in range(2)]
    assert directories[0] != directories[1]
    recorded = []
    for directory in directories:
        with closing(connect_database(directory / "games.sqlite3")) as connection:
            recorded.append(connection.execute("SELECT run_id, game_number, winner FROM games ORDER BY run_id, game_number").fetchall())
    assert recorded[0] == recorded[1]
    restored = capture_rng_state()
    assert state["python"] == restored["python"] and state["numpy"] == restored["numpy"]
    torch.testing.assert_close(state["torch"], restored["torch"], rtol=0, atol=0)


@pytest.mark.parametrize("selection", ["latest", "best"])
def test_registered_selection(trained_checkpoint, tmp_path, selection):
    checkpoint, registry_path = trained_checkpoint
    directory = offline_eval.evaluate_checkpoint(selection, registry_path=registry_path,
        opponents=["random"], seats=["o"], games=2, device="cpu", output_root=tmp_path / "assessments")
    metadata, _ = read_assessment(directory)
    assert metadata["model"]["checkpoint_hash"] == checkpoint_hash(checkpoint)
    assert metadata["model"]["selection"] == selection


def test_graceful_interruption_records_partial_batch_but_excludes_it_from_summary(trained_checkpoint, tmp_path):
    checkpoint, _ = trained_checkpoint
    root = tmp_path / "assessments"
    def stop():
        databases = list(root.glob("*/games.sqlite3"))
        if not databases:
            return False
        with sqlite3.connect(databases[0]) as connection:
            return connection.execute("SELECT count(*) FROM games").fetchone()[0] >= 2
    directory = offline_eval.evaluate_checkpoint(checkpoint, games=5, device="cpu",
        output_root=root, should_stop=stop)
    metadata, rows = read_assessment(directory)
    assert metadata["status"] == "interrupted"
    assert len(rows) == 1 and rows[0]["status"] == "interrupted" and rows[0]["games"] == "2"
    assert all(row["games"] == 0 and row["average_return"] is None for row in metadata["summary"])
    assert rows[0]["loss_rate_ci_lower"] == "" and rows[0]["loss_rate_ci_reason"] == "incomplete_batch"
    assert all(row["loss_rate_ci_lower"] is None and row["loss_rate_ci_reason"] == "no_completed_games" for row in metadata["summary"])
    with closing(connect_database(directory / "games.sqlite3")) as connection:
        assert connection.execute("SELECT status FROM runs").fetchone()[0] == "interrupted"


def test_policy_failure_preserves_assessment_and_propagates_error(trained_checkpoint, tmp_path, monkeypatch):
    checkpoint, _ = trained_checkpoint
    def broken_policy(*args):
        raise RuntimeError("policy failed")
    monkeypatch.setattr(offline_eval.SavedAgentPlayer, "__call__", broken_policy)
    root = tmp_path / "assessments"
    with pytest.raises(RuntimeError, match="policy failed"):
        offline_eval.evaluate_checkpoint(checkpoint, games=2, device="cpu", output_root=root)
    directory = next(root.iterdir())
    metadata, _ = read_assessment(directory)
    assert metadata["status"] == "failed" and metadata["error"] == "policy failed"
    assert metadata["active_batch"]["opponent"] == "random"
    with closing(connect_database(directory / "games.sqlite3")) as connection:
        assert connection.execute("SELECT status FROM runs").fetchone()[0] == "failed"


@pytest.mark.parametrize("arguments", [
    [], ["--checkpoint", "model.pt", "--agent", "best"],
    ["--checkpoint", "model.pt", "--games", "0"],
    ["--agent", "latest", "--protocol", "abc"],
    ["--checkpoint", "model.pt", "--seeds", "1", "1"],
    ["--checkpoint", "model.pt", "--seeds", "-1"],
    ["--checkpoint", "model.pt", "--seats", "x", "x"],
])
def test_cli_rejects_invalid_settings(arguments):
    with pytest.raises(SystemExit) as error:
        offline_eval.parse_args(arguments)
    assert error.value.code == 2


def test_cli_runs_and_reports_saved_assessment(trained_checkpoint, tmp_path, capsys):
    checkpoint, _ = trained_checkpoint
    assert offline_eval.main(["--checkpoint", str(checkpoint), "--opponents", "random",
        "--seeds", "11", "22", "--games", "2", "--device", "cpu", "--output-root", str(tmp_path / "assessments")]) == 0
    output = capsys.readouterr().out
    assert "agent X" in output and "agent O" in output
    assert "2/2 complete seed batches" in output and "Assessment completed:" in output
    assert "95% CI" in output


def test_missing_checkpoint_creates_no_assessment(tmp_path):
    root = tmp_path / "assessments"
    with pytest.raises(FileNotFoundError):
        offline_eval.evaluate_checkpoint(tmp_path / "missing.pt", output_root=root)
    assert not root.exists()


def test_pooled_intervals_use_all_completed_games_and_ignore_partial_batches():
    batches = [
        {"opponent": "random", "seat": "x", "status": "completed", "games": 1000,
         "wins": 968, "draws": 20, "losses": 12},
        {"opponent": "random", "seat": "x", "status": "completed", "games": 1000,
         "wins": 962, "draws": 27, "losses": 11},
        {"opponent": "random", "seat": "x", "status": "completed", "games": 1000,
         "wins": 970, "draws": 20, "losses": 10},
        {"opponent": "random", "seat": "x", "status": "interrupted", "games": 20,
         "wins": 0, "draws": 0, "losses": 20},
    ]
    row = offline_eval._summarise(batches, ["random"], ["x"], 4)[0]
    assert row["games"] == 3000 and row["losses"] == 33
    assert round(row["loss_rate_ci_lower"] * 100, 2) == 0.78
    assert round(row["loss_rate_ci_upper"] * 100, 2) == 1.54


def test_console_identifies_deterministic_opponent_intervals_as_not_applicable():
    row = offline_eval._loss_statistics("minimax", 3000, 0)
    assert offline_eval._format_loss_statistics(row) == "loss 0.00% (95% CI N/A: deterministic opponent)"


def test_module_entrypoint_executes_in_a_separate_process(trained_checkpoint, tmp_path):
    checkpoint, _ = trained_checkpoint
    output_root = tmp_path / "cli-assessments"
    process = subprocess.run([sys.executable, "-m", "evaluation.offline_eval",
        "--checkpoint", str(checkpoint), "--opponents", "random", "--seats", "o",
        "--games", "2", "--device", "cpu", "--output-root", str(output_root)],
        cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=60)
    assert process.returncode == 0, process.stderr
    assert "Assessment completed:" in process.stdout
    metadata, _ = read_assessment(next(output_root.iterdir()))
    assert metadata["status"] == "completed"
