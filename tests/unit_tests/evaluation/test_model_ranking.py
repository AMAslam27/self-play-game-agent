import csv
import json

import pytest
import torch

from evaluation.model_registry import (
    checkpoint_hash, connect_registry, import_tictactoe_run, protocol_identity,
)
from evaluation.tictactoe_ranking import rank_checkpoints
from players.checkpoint_selection import resolve_checkpoint
from training.checkpoints import CHECKPOINT_VERSION


@pytest.fixture
def registry(tmp_path):
    connection = connect_registry(tmp_path / "models.sqlite3")
    yield connection
    connection.close()


def make_run(tmp_path, run_id="first", *, finished="2026-10-05T12:00:00+00:00",
             seed=12345, random_x=(8, 2, 0), random_o=(6, 3, 1),
             minimax_x=(0, 10, 0), minimax_o=(0, 10, 0), omit=None,
             partial=None, status="completed"):
    directory = tmp_path / run_id
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(parents=True)
    path = checkpoints / "episode-00000020.pt"
    config = {"evaluation": {"opponents": ["random", "minimax"], "seats": ["x", "o"],
              "games_per_opponent_per_seat": 10, "seed": seed, "epsilon": 0.0,
              "every_episodes": 10}}
    torch.save({"checkpoint_version": CHECKPOINT_VERSION, "run_id": run_id,
                "config": config, "progress": {"episodes": 20},
                "agent": {"learning_updates": 50, "online_network": {"weight": torch.ones(1)}}}, path)
    metadata = {"run_id": run_id, "name": f"Experiment {run_id}", "config": config,
                "status": status, "finished_at": finished, "progress": {"episodes": 20},
                "learning_updates": 50, "last_checkpoint": str(path)}
    (directory / "run.json").write_text(json.dumps(metadata))
    with (directory / "evaluation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode", "update", "opponent", "seat",
                                "games", "wins", "draws", "losses", "status"])
        writer.writeheader()
        # Earlier evaluations must not contribute to the final ranking.
        writer.writerow(dict(episode=10, update=25, opponent="random", seat="x", games=10,
                             wins=10, draws=0, losses=0, status="completed"))
        for opponent, seat, counts in [("random", "x", random_x), ("random", "o", random_o),
                                       ("minimax", "x", minimax_x), ("minimax", "o", minimax_o)]:
            if (opponent, seat) == omit:
                continue
            writer.writerow(dict(episode=20, update=50, opponent=opponent, seat=seat, games=sum(counts),
                                 wins=counts[0], draws=counts[1], losses=counts[2],
                                 status="interrupted" if (opponent, seat) == partial else "completed"))
    return directory, path


def test_import_is_idempotent_and_ranking_explains_scores(registry, tmp_path):
    directory, path = make_run(tmp_path)
    identity = import_tictactoe_run(registry, directory)
    assert identity == checkpoint_hash(path)
    assert import_tictactoe_run(registry, directory) == identity
    assert registry.execute("SELECT count(*) FROM model_runs").fetchone()[0] == 1
    assert registry.execute("SELECT count(*) FROM model_evaluations").fetchone()[0] == 4
    best = rank_checkpoints(registry)[0]
    assert best.run_id == "first"
    assert best.worst_minimax_loss_rate == 0
    assert best.worst_random_return == 0.5
    assert best.mean_random_return == pytest.approx(0.65)
    assert "minimax loss rate" in best.explanation
    assert resolve_checkpoint("best", connection=registry) == path
    assert resolve_checkpoint("latest", connection=registry) == path


@pytest.mark.parametrize("better_kwargs,worse_kwargs", [
    ({"random_x": (0, 10, 0), "random_o": (0, 10, 0)},
     {"random_x": (10, 0, 0), "random_o": (10, 0, 0), "minimax_o": (0, 9, 1)}),
    ({"random_x": (7, 3, 0), "random_o": (7, 3, 0)},
     {"random_x": (10, 0, 0), "random_o": (6, 4, 0)}),
    ({"random_x": (9, 1, 0), "random_o": (6, 4, 0)},
     {"random_x": (8, 2, 0), "random_o": (6, 4, 0)}),
    ({"finished": "2026-10-07T12:00:00+00:00"}, {}),
])
def test_ranking_priorities(registry, tmp_path, better_kwargs, worse_kwargs):
    for name, kwargs in [("better", better_kwargs), ("worse", worse_kwargs)]:
        directory, _ = make_run(tmp_path, name, **kwargs)
        import_tictactoe_run(registry, directory)
    assert [item.run_id for item in rank_checkpoints(registry)] == ["better", "worse"]


@pytest.mark.parametrize("kwargs", [{"omit": ("minimax", "o")}, {"partial": ("random", "x")}])
def test_incomplete_sweep_cannot_rank_but_can_be_latest(registry, tmp_path, kwargs):
    directory, path = make_run(tmp_path, **kwargs)
    import_tictactoe_run(registry, directory)
    assert rank_checkpoints(registry) == []
    with pytest.raises(ValueError, match="complete comparable"):
        resolve_checkpoint("best", connection=registry)
    assert resolve_checkpoint("latest", connection=registry) == path


def test_protocol_mismatch_requires_explicit_selection(registry, tmp_path):
    for name, seed in [("first", 1), ("second", 2)]:
        directory, _ = make_run(tmp_path, name, seed=seed)
        import_tictactoe_run(registry, directory)
    with pytest.raises(ValueError, match="Multiple evaluation protocols"):
        resolve_checkpoint("best", connection=registry)
    protocol = registry.execute("SELECT protocol_id FROM model_checkpoints WHERE run_id='first'").fetchone()[0]
    assert rank_checkpoints(registry, protocol_id=protocol)[0].run_id == "first"


@pytest.mark.parametrize("change", ["missing", "changed"])
def test_latest_and_best_skip_unavailable_checkpoint(registry, tmp_path, change):
    older, older_path = make_run(tmp_path, "older")
    newer, newer_path = make_run(tmp_path, "newer", finished="2026-10-07T12:00:00+00:00")
    for directory in (older, newer):
        import_tictactoe_run(registry, directory)
    if change == "missing":
        newer_path.unlink()
    else:
        newer_path.write_bytes(b"replaced checkpoint")
    assert resolve_checkpoint("latest", connection=registry) == older_path
    assert resolve_checkpoint("best", connection=registry) == older_path


def test_explicit_path_needs_no_registry(tmp_path):
    _, path = make_run(tmp_path)
    assert resolve_checkpoint(path) == path
    assert resolve_checkpoint(str(path)) == path
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="registry connection"):
        resolve_checkpoint("latest")


def test_no_compatible_run(registry, tmp_path):
    directory, _ = make_run(tmp_path)
    import_tictactoe_run(registry, directory)
    with pytest.raises(ValueError, match="compatible run"):
        resolve_checkpoint("latest", connection=registry, game="connect4")
    assert rank_checkpoints(registry, model_type="other") == []


def test_registry_rejects_unfinished_run(registry, tmp_path):
    directory, _ = make_run(tmp_path, status="interrupted")
    with pytest.raises(ValueError, match="completed runs"):
        import_tictactoe_run(registry, directory)
    assert registry.execute("SELECT count(*) FROM model_runs").fetchone()[0] == 0


def test_registry_rejects_checkpoint_metadata_mismatch(registry, tmp_path):
    directory, path = make_run(tmp_path)
    payload = torch.load(path, weights_only=True)
    payload["agent"]["learning_updates"] = 49
    torch.save(payload, path)
    with pytest.raises(ValueError, match="does not match"):
        import_tictactoe_run(registry, directory)
    assert registry.execute("SELECT count(*) FROM model_runs").fetchone()[0] == 0


def test_relocated_run_import_uses_local_checkpoint(registry, tmp_path):
    directory, path = make_run(tmp_path)
    metadata_path = directory / "run.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["last_checkpoint"] = "C:\\old-machine\\checkpoints\\" + path.name
    metadata_path.write_text(json.dumps(metadata))
    import_tictactoe_run(registry, directory)
    assert resolve_checkpoint("latest", connection=registry) == path


def test_exact_tie_is_stable(registry, tmp_path):
    for name in ("z", "a"):
        directory, _ = make_run(tmp_path, name)
        import_tictactoe_run(registry, directory)
    assert [item.run_id for item in rank_checkpoints(registry)] == ["a", "z"]


def test_canonical_protocol_identity():
    assert protocol_identity({"a": 1, "b": 2}) == protocol_identity({"b": 2, "a": 1})
