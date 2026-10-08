from types import SimpleNamespace

import pytest

from evaluation import artifacts
from evaluation.artifacts import default_plot_path
from evaluation.database import PROJECT_ROOT


def test_default_plot_path_contains_utc_timestamp_and_run_id():
    path = default_plot_path(42, "2026-10-01T13:30:25.123456+01:00")
    assert path == PROJECT_ROOT / "results/tictactoe/plots/20261001T123025123456Z_run-42.png"


def test_default_plot_path_distinguishes_run_ids():
    timestamp = "2026-10-01T12:00:00+00:00"
    assert default_plot_path(1, timestamp) != default_plot_path(2, timestamp)


def test_default_plot_path_supports_other_game_directories():
    path = default_plot_path(7, "2026-10-01T12:00:00+00:00", game="connect4")
    assert path.parent == PROJECT_ROOT / "results/connect4/plots"


def test_display_project_path_is_relative():
    path = PROJECT_ROOT / "results" / "chart.png"
    assert artifacts.display_project_path(path) == "results/chart.png"


def test_display_external_path_remains_absolute(tmp_path):
    path = tmp_path / "chart.png"
    assert artifacts.display_project_path(path) == path.resolve().as_posix()


@pytest.mark.parametrize("completed", [0, 1])
@pytest.mark.parametrize("explicit_path", [None, "results/chart.png"])
def test_small_batches_do_not_generate_charts(monkeypatch, completed, explicit_path):
    def unexpected_plot(*args, **kwargs):
        pytest.fail("A chart should not be generated")

    monkeypatch.setattr(artifacts, "plot_results", unexpected_plot)

    assert artifacts.save_run_plot(
        {1: completed, 0: 0, -1: 0}, "human", "dqn",
        SimpleNamespace(run_id=1, started_at=None), output_path=explicit_path,
    ) is None


def test_two_completed_games_generate_a_chart(monkeypatch):
    calls = []

    def capture_plot(results, policy_x, policy_o, output_path):
        calls.append((results, policy_x, policy_o))
        return output_path

    monkeypatch.setattr(artifacts, "plot_results", capture_plot)
    path = artifacts.save_run_plot(
        {1: 1, 0: 1, -1: 0}, "human", "dqn",
        SimpleNamespace(run_id=1, started_at=None), output_path="results/chart.png",
    )
    assert path == artifacts.Path("results/chart.png")
    assert calls == [({1: 1, 0: 1, -1: 0}, "human", "dqn")]
