from argparse import Namespace
from pathlib import Path

import pytest

import runner
from evaluation import artifacts
from tests.unit_tests.common.mock_utils import FakePlotter
from games.tictactoe.rules import EMPTY, PLAYER_O, PLAYER_X


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["runner.py"])

    args = runner.parse_args()

    assert args.x == "human"
    assert args.o == "human"
    assert args.games == 1
    assert args.quiet is False
    assert args.plot_file is None


def test_parse_args_custom_arguments(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner.py",
            "--x",
            "random",
            "--o",
            "random",
            "--games",
            "1000",
            "--quiet",
        ],
    )

    args = runner.parse_args()

    assert args.x == "random"
    assert args.o == "random"
    assert args.games == 1000
    assert args.quiet is True


def test_print_summary(capsys):
    results = {
        PLAYER_X: 5,
        PLAYER_O: 3,
        EMPTY: 2,
    }

    runner.print_summary(results)

    captured = capsys.readouterr()

    assert captured.out == (
        "\nResults over 10 game(s):\n  X wins: 5\n  O wins: 3\n  Draws:  2\n"
    )


def test_main_plays_requested_number_of_games(monkeypatch):
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "x": "random",
                "o": "random",
                "games": 3,
                "quiet": True,
                "plot_file": None,
                "db_file": "unused.sqlite3",
                "seed": None,
            },
        )(),
    )

    winners = []

    def fake_play_game(policy_x, policy_o, verbose):
        winners.append((policy_x, policy_o, verbose))
        return PLAYER_X

    monkeypatch.setattr(runner, "play_game", fake_play_game)

    runner.main()

    assert len(winners) == 3
    assert all(verbose is False for _, _, verbose in winners)
    assert all(
        policy_x is runner.random_policy and policy_o is runner.random_policy
        for policy_x, policy_o, _ in winners
    )


def test_main_counts_wins_and_draws(monkeypatch, capsys):
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "x": "random",
                "o": "random",
                "games": 3,
                "quiet": True,
                "plot_file": None,
                "db_file": "unused.sqlite3",
                "seed": None,
            },
        )(),
    )

    winners = iter([PLAYER_X, PLAYER_O, EMPTY])

    monkeypatch.setattr(
        runner,
        "play_game",
        lambda policy_x, policy_o, verbose: next(winners),
    )

    runner.main()

    captured = capsys.readouterr()

    assert "Results over 3 game(s):" in captured.out
    assert "X wins: 1" in captured.out
    assert "O wins: 1" in captured.out
    assert "Draws:  1" in captured.out


def test_main_does_not_print_board_in_quiet_mode(monkeypatch, capsys):
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "x": "random",
                "o": "random",
                "games": 1,
                "quiet": True,
                "plot_file": None,
                "db_file": "unused.sqlite3",
                "seed": None,
            },
        )(),
    )

    calls = []

    def fake_play_game(policy_x, policy_o, verbose):
        calls.append(verbose)
        return PLAYER_X

    monkeypatch.setattr(runner, "play_game", fake_play_game)

    runner.main()

    captured = capsys.readouterr()

    assert calls == [False]
    assert "X: random | O: random" not in captured.out


def test_main_passes_verbose_when_not_quiet(monkeypatch, capsys):
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "x": "random",
                "o": "random",
                "games": 1,
                "quiet": False,
                "plot_file": None,
                "db_file": "unused.sqlite3",
                "seed": None,
            },
        )(),
    )

    calls = []

    def fake_play_game(policy_x, policy_o, verbose):
        calls.append(verbose)
        return PLAYER_X

    monkeypatch.setattr(runner, "play_game", fake_play_game)

    runner.main()

    captured = capsys.readouterr()

    assert calls == [True]
    assert "X: random | O: random. Squares are numbered 1-9." in captured.out


def test_main_stops_when_human_quits(monkeypatch):
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "x": "human",
                "o": "random",
                "games": 10,
                "quiet": True,
                "plot_file": None,
                "db_file": "unused.sqlite3",
                "seed": None,
            },
        )(),
    )

    calls = []

    def fake_play_game(policy_x, policy_o, verbose):
        calls.append(1)
        return None

    monkeypatch.setattr(runner, "play_game", fake_play_game)

    runner.main()

    assert len(calls) == 1


def test_main_asks_to_play_again_after_human_game(monkeypatch):
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: Namespace(
            x="human",
            o="random",
            games=1,
            quiet=True,
            plot_file=None,
            db_file="unused.sqlite3",
            seed=None,
        ),
    )

    results = iter([PLAYER_X, PLAYER_O])
    inputs = iter(["y", "n"])
    calls = []

    def fake_play_game(policy_x, policy_o, verbose):
        calls.append(1)
        return next(results)

    monkeypatch.setattr(runner, "play_game", fake_play_game)
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    runner.main()

    assert len(calls) == 2


def test_main_stops_when_user_declines_to_play_again(monkeypatch):
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "x": "human",
                "o": "random",
                "games": 1,
                "quiet": True,
                "plot_file": None,
                "db_file": "unused.sqlite3",
                "seed": None,
            },
        )(),
    )

    calls = []

    def fake_play_game(policy_x, policy_o, verbose):
        calls.append(1)
        return PLAYER_X

    monkeypatch.setattr(runner, "play_game", fake_play_game)
    monkeypatch.setattr("builtins.input", lambda _: "n")

    runner.main()

    assert len(calls) == 1


def test_parse_args_plot_file_and_minimax(monkeypatch):
    monkeypatch.setattr("sys.argv", ["runner.py", "--x", "minimax", "--plot-file", "results/chart.png"])
    args = runner.parse_args()
    assert args.x == "minimax"
    assert args.plot_file == "results/chart.png"


def test_progress_reports_hundreds_and_final_partial_batch(capsys):
    for completed in range(1, 251):
        runner.print_progress(completed, 250)
    assert capsys.readouterr().out.splitlines() == [
        "100/250 games completed", "200/250 games completed", "250/250 games completed",
    ]


def test_progress_reports_exact_hundred_once(capsys):
    runner.print_progress(100, 100)
    assert capsys.readouterr().out == "100/100 games completed\n"


def test_main_reports_progress_in_quiet_mode(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["runner.py", "--x", "random", "--o", "random", "--games", "250", "--quiet"])
    monkeypatch.setattr(runner, "play_game", lambda *args, **kwargs: EMPTY)
    runner.main()
    output = capsys.readouterr().out
    assert "100/250 games completed" in output
    assert "200/250 games completed" in output
    assert "250/250 games completed" in output
    assert "Draws:  250" in output


def test_main_does_not_count_abandoned_game(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["runner.py", "--x", "human", "--o", "random", "--games", "100", "--quiet"])
    outcomes = iter([PLAYER_X, None])
    monkeypatch.setattr(runner, "play_game", lambda *args, **kwargs: next(outcomes))
    runner.main()
    output = capsys.readouterr().out
    assert "Results over 1 game(s):" in output
    assert "100/100 games completed" not in output


def test_main_passes_results_to_plotter(monkeypatch, capsys):

    plotter = FakePlotter(result="results/chart.png")
    monkeypatch.setattr(artifacts, "plot_results", plotter)
    monkeypatch.setattr("sys.argv", ["runner.py", "--x", "random", "--o", "random", "--games", "3", "--quiet", "--plot-file", "results/chart.png"])
    outcomes = iter([PLAYER_X, PLAYER_O, EMPTY])
    monkeypatch.setattr(runner, "play_game", lambda *args, **kwargs: next(outcomes))
    runner.main()
    assert plotter.calls == [
        ({PLAYER_X: 1, PLAYER_O: 1, EMPTY: 1}, "random", "random", {"output_path": Path("results/chart.png")}),
    ]
    assert "Chart saved to: results/chart.png" in capsys.readouterr().out


def test_main_plots_larger_batches_by_default(monkeypatch, capsys):

    plotter = FakePlotter()
    monkeypatch.setattr(artifacts, "plot_results", plotter)
    monkeypatch.setattr("sys.argv", ["runner.py", "--x", "random", "--o", "random", "--games", "2", "--quiet"])
    monkeypatch.setattr(runner, "play_game", lambda *args, **kwargs: EMPTY)
    runner.main()
    assert len(plotter.calls) == 1
    path = plotter.calls[0][3]["output_path"]
    assert path.parent.name == "plots"
    assert path.parent.parent.name == "tictactoe"
    assert path.name.endswith("_run-1.png")
    assert "Chart saved" not in capsys.readouterr().out


def test_main_does_not_report_chart_for_empty_run(monkeypatch, capsys):

    plotter = FakePlotter()
    monkeypatch.setattr(artifacts, "plot_results", plotter)
    monkeypatch.setattr("sys.argv", ["runner.py", "--x", "human", "--quiet", "--plot-file", "results/chart.png"])
    monkeypatch.setattr(runner, "play_game", lambda *args, **kwargs: None)
    runner.main()
    assert plotter.calls == []
    assert "Chart saved" not in capsys.readouterr().out


def test_main_records_completed_games_and_closes_connection(monkeypatch, fake_connection):
    monkeypatch.setattr("sys.argv", ["runner.py", "--x", "random", "--o", "random", "--games", "3", "--quiet", "--seed", "42"])
    outcomes = iter([1, 0, -1])
    seeds = []
    monkeypatch.setattr(runner.random, "seed", seeds.append)
    monkeypatch.setattr(runner, "play_game", lambda *args, **kwargs: next(outcomes))
    runner.main()
    game_parameters = [params for sql, params in fake_connection.executions if "INSERT INTO games" in sql]
    assert game_parameters == [(1, 1, 1), (1, 2, 0), (1, 3, -1)]
    assert fake_connection.executions[-1][1][1] == "completed"
    assert seeds == [42]
    assert fake_connection.closed


def test_main_records_abandoned_status(monkeypatch, fake_connection):
    monkeypatch.setattr("sys.argv", ["runner.py", "--x", "human", "--games", "2", "--quiet"])
    outcomes = iter([1, None])
    monkeypatch.setattr(runner, "play_game", lambda *args, **kwargs: next(outcomes))
    runner.main()
    assert fake_connection.executions[-1][1][1] == "abandoned"
    assert len([sql for sql, _ in fake_connection.executions if "INSERT INTO games" in sql]) == 1
    assert fake_connection.closed


def test_main_records_failure_and_preserves_exception(monkeypatch, fake_connection):

    monkeypatch.setattr("sys.argv", ["runner.py", "--x", "random", "--o", "random", "--quiet"])

    def fail_game(*args, **kwargs):
        raise ValueError("Policy failed")

    monkeypatch.setattr(runner, "play_game", fail_game)
    with pytest.raises(ValueError, match="Policy failed"):
        runner.main()
    assert fake_connection.executions[-1][1][1] == "failed"
    assert fake_connection.closed


def test_main_records_keyboard_interruption(monkeypatch, fake_connection):

    monkeypatch.setattr("sys.argv", ["runner.py", "--x", "random", "--o", "random", "--quiet"])

    def interrupt_game(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "play_game", interrupt_game)
    with pytest.raises(KeyboardInterrupt):
        runner.main()
    assert fake_connection.executions[-1][1][1] == "interrupted"
    assert fake_connection.closed


def test_main_creates_separate_run_for_each_replay(monkeypatch, fake_connection):
    monkeypatch.setattr("sys.argv", ["runner.py", "--x", "human", "--quiet"])
    answers = iter(["y", "n"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    monkeypatch.setattr(runner, "play_game", lambda *args, **kwargs: 0)
    runner.main()
    game_parameters = [params for sql, params in fake_connection.executions if "INSERT INTO games" in sql]
    assert game_parameters == [(1, 1, 0), (2, 1, 0)]
    assert fake_connection.closed


def test_parse_args_rejects_nonpositive_game_count(monkeypatch):

    monkeypatch.setattr("sys.argv", ["runner.py", "--games", "0"])
    with pytest.raises(SystemExit) as error:
        runner.parse_args()
    assert error.value.code == 2


def test_parse_args_database_override(monkeypatch):

    monkeypatch.setattr("sys.argv", ["runner.py", "--db-file", "results/custom.sqlite3", "--seed", "7"])
    args = runner.parse_args()
    assert args.db_file == Path("results/custom.sqlite3")
    assert args.seed == 7


def test_replay_plots_each_batch_separately(monkeypatch, isolate_runner_plotting):
    monkeypatch.setattr("sys.argv", ["runner.py", "--x", "human", "--games", "2", "--quiet"])
    outcomes = iter([1, 1, -1, -1])
    answers = iter(["y", "n"])
    monkeypatch.setattr(runner, "play_game", lambda *args, **kwargs: next(outcomes))
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    runner.main()
    calls = isolate_runner_plotting.calls
    assert calls[0][0] == {1: 2, 0: 0, -1: 0}
    assert calls[1][0] == {1: 0, 0: 0, -1: 2}
    assert calls[0][3]["output_path"].name.endswith("_run-1.png")
    assert calls[1][3]["output_path"].name.endswith("_run-2.png")


def test_replay_does_not_overwrite_explicit_plot_path(monkeypatch, isolate_runner_plotting):

    monkeypatch.setattr("sys.argv", ["runner.py", "--x", "human", "--games", "2", "--quiet", "--plot-file", "results/chart.png"])
    answers = iter(["y", "n"])
    monkeypatch.setattr(runner, "play_game", lambda *args, **kwargs: 0)
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    runner.main()
    assert [call[3]["output_path"] for call in isolate_runner_plotting.calls] == [
        Path("results/chart.png"), Path("results/chart_run-2.png"),
    ]
