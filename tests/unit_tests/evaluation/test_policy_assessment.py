import json
import sys
from copy import deepcopy

import pytest

from evaluation import comparison_report, exhaustive, historical
from evaluation.tictactoe_matches import make_opponent
from games.tictactoe.rules import TicTacToe, winner


def first_legal(game):
    return game.legal_actions()[0]


@pytest.mark.parametrize("seat", ["x", "o"])
def test_minimax_is_unbeatable_against_every_response(seat):
    result = exhaustive.assess_policy(make_opponent("minimax", None), seat)
    assert result.unbeatable
    assert result.losses == 0
    assert result.draws > 0
    assert result.losing_sequence is None


@pytest.mark.parametrize("seat,player", [("x", 1), ("o", -1)])
def test_exhaustive_counts_match_independent_board_enumeration_and_witness(seat, player):
    counts = {1: 0, 0: 0, -1: 0}

    def enumerate_boards(board, turn):
        legal = [i for i, value in enumerate(board) if value == 0]
        win = winner(board)
        if win or not legal:
            counts[win * player] += 1
            return
        for action in legal[:1] if turn == player else legal:
            child = list(board)
            child[action] = turn
            enumerate_boards(child, -turn)

    enumerate_boards([0] * 9, 1)
    result = exhaustive.assess_policy(first_legal, seat)
    assert (result.wins, result.draws, result.losses) == (counts[1], counts[0], counts[-1])
    assert not result.unbeatable
    game = TicTacToe()
    for action in result.losing_sequence:
        assert not game.is_terminal()
        if game.current_player == player:
            assert action == first_legal(game)
        game.step(action)
    assert game.winner() == -player


@pytest.mark.parametrize("action", [None, True, 9, -1, 0.5])
def test_exhaustive_rejects_invalid_policy(action):
    with pytest.raises(ValueError, match="legal integer"):
        exhaustive.assess_policy(lambda game: action, "x")


def test_exhaustive_rejects_unknown_seat():
    with pytest.raises(ValueError, match="Seat"):
        exhaustive.assess_policy(first_legal, "z")


def test_historical_records_outcomes_from_candidate_perspective():
    rows = historical.compare_policies(first_legal, make_opponent("minimax", None))
    assert [row["seat"] for row in rows] == ["x", "o"]
    assert all(row["losses"] == 1 and row["games"] == 1 for row in rows)
    reverse = historical.compare_policies(make_opponent("minimax", None), first_legal)
    assert all(row["wins"] == 1 for row in reverse)


def test_historical_rejects_quitting():
    with pytest.raises(ValueError, match="cannot quit"):
        historical.compare_policies(lambda game: None, first_legal)


def assessment(seed=42, identity="a" * 64, games=100, losses=10):
    return {
        "protocol_id": "same-protocol", "status": "completed", "requested_batches": 1,
        "batches": [{"status": "completed"}],
        "model": {"experiment_run_id": f"run-{seed}", "checkpoint_hash": identity},
        "model_config": {"environment": {"opponent": "random"}, "experiment": {"seed": seed}},
        "summary": [{"opponent": "random", "seat": "x", "games": games,
                     "wins": games - losses, "draws": 0, "losses": losses,
                     "loss_rate": losses / games, "loss_rate_ci_lower": 0.05,
                     "loss_rate_ci_upper": 0.15, "loss_rate_ci_method": "wilson"}],
    }


def test_seed_summary_weights_models_equally_instead_of_pooling_games():
    rows = comparison_report.assessment_rows([
        assessment(games=10, losses=0),
        assessment(43, "b" * 64, games=1000, losses=1000),
    ])
    summary = comparison_report.summarise_training_seeds(rows)[0]
    assert summary["mean_loss_rate"] == 0.5
    assert summary["sample_sd_loss_rate"] == pytest.approx(2 ** -0.5)
    assert summary["min_loss_rate"] == 0
    assert summary["max_loss_rate"] == 1
    assert rows[0]["loss_rate_ci_lower"] == 0.05


@pytest.mark.parametrize("change", ["protocol", "status", "partial", "duplicate", "seed"])
def test_report_rejects_incompatible_or_incomplete_evidence(change):
    first = assessment()
    second = assessment(43, "b" * 64)
    if change == "protocol":
        second["protocol_id"] = "different"
    elif change == "status":
        second["status"] = "interrupted"
    elif change == "partial":
        second["batches"][0]["status"] = "interrupted"
    elif change == "duplicate":
        second["model"]["checkpoint_hash"] = "a" * 64
    else:
        second["model_config"]["experiment"]["seed"] = 42
    with pytest.raises(ValueError):
        comparison_report.assessment_rows([first, second])


def test_single_seed_sd_is_unavailable():
    rows = comparison_report.assessment_rows([assessment()])
    assert comparison_report.summarise_training_seeds(rows)[0]["sample_sd_loss_rate"] is None


def test_additional_evidence_must_match_exact_models():
    rows = comparison_report.assessment_rows([assessment()])
    model = {key: rows[0][key] for key in ("checkpoint_hash", "training_method", "training_seed", "run_id")}
    evidence = {"status": "completed", "models": [model]}
    assert comparison_report.attach_evidence(evidence, rows, "exhaustive") == [model]
    bad = deepcopy(evidence)
    bad["models"][0]["checkpoint_hash"] = "b" * 64
    with pytest.raises(ValueError, match="do not match"):
        comparison_report.attach_evidence(bad, rows, "exhaustive")
    evidence["models"][0]["training_seed"] = 99
    with pytest.raises(ValueError, match="metadata"):
        comparison_report.attach_evidence(evidence, rows, "exhaustive")


@pytest.mark.parametrize("module,filename", [(exhaustive, "exhaustive.json"), (historical, "historical.json")])
def test_commands_preserve_failed_status_and_refuse_overwrite(module, filename, tmp_path, monkeypatch):
    output = tmp_path / "assessment"
    arguments = ["command", "--checkpoints", "missing.pt", "--output", str(output)]
    if module is historical:
        arguments.extend(["--references", "reference.pt"])
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(FileNotFoundError):
        module.main()
    saved = (output / filename).read_bytes()
    assert json.loads(saved)["status"] == "failed"
    with pytest.raises(FileExistsError):
        module.main()
    assert (output / filename).read_bytes() == saved


def test_report_cli_publishes_sources_tables_and_plot(tmp_path, monkeypatch):
    # Both seats are needed for the published comparison figure.
    item = assessment()
    item["summary"].append({**item["summary"][0], "seat": "o"})
    source = tmp_path / "assessment.json"
    source.write_text(json.dumps(item), encoding="utf-8")
    output = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["report", "--assessments", str(source), "--output", str(output)])
    comparison_report.main()
    assert (output / "models.csv").is_file()
    assert (output / "baseline-losses.png").stat().st_size > 0
    assert (output / "assessments" / ("a" * 64 + ".json")).read_bytes() == source.read_bytes()
