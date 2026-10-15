import pytest

from evaluation import recorder


@pytest.fixture
def run_recorder(monkeypatch, fake_connection):
    monkeypatch.setattr(recorder, "utc_now", lambda: "2026-10-01T12:00:00+00:00")
    return recorder.RunRecorder(fake_connection)


def test_start_run_records_metadata_and_commits(run_recorder, fake_connection):
    assert run_recorder.start_run("random", "minimax", 100, seed=42) == 1
    sql, parameters = fake_connection.executions[0]
    assert "INSERT INTO runs" in sql
    assert parameters == ("2026-10-01T12:00:00+00:00", "random", "minimax", 100, 42)
    assert fake_connection.commits == 1


def test_record_game_commits_each_outcome(run_recorder, fake_connection):
    run_recorder.start_run("random", "random", 3)
    for number, outcome in enumerate((1, 0, -1), start=1):
        run_recorder.record_game(number, outcome)
    assert [params for sql, params in fake_connection.executions if "INSERT INTO games" in sql] == [
        (1, 1, 1), (1, 2, 0), (1, 3, -1),
    ]
    assert fake_connection.commits == 4


@pytest.mark.parametrize("status", ["completed", "abandoned", "interrupted", "failed"])
def test_finish_run_updates_status_and_timestamp(run_recorder, fake_connection, status):
    run_recorder.start_run("random", "random", 1)
    run_recorder.finish_run(status)
    assert fake_connection.executions[-1][1] == ("2026-10-01T12:00:00+00:00", status, 1)
    assert run_recorder.finished
    assert fake_connection.commits == 2


def test_requires_started_run(run_recorder, fake_connection):
    with pytest.raises(ValueError, match="Start a run"):
        run_recorder.record_game(1, 1)
    with pytest.raises(ValueError, match="Start a run"):
        run_recorder.finish_run()
    assert fake_connection.executions == []


def test_rejects_reusing_started_or_finished_recorder(run_recorder):
    run_recorder.start_run("random", "random", 1)
    with pytest.raises(ValueError, match="already started"):
        run_recorder.start_run("random", "random", 1)
    run_recorder.finish_run()
    with pytest.raises(ValueError, match="already finished"):
        run_recorder.record_game(1, 1)
    with pytest.raises(ValueError, match="already finished"):
        run_recorder.finish_run()


@pytest.mark.parametrize("number,winner", [(0, 1), (-1, 1), (1, 2), (1, None)])
def test_rejects_invalid_game_data(run_recorder, fake_connection, number, winner):
    run_recorder.start_run("random", "random", 1)
    with pytest.raises(ValueError):
        run_recorder.record_game(number, winner)
    assert len(fake_connection.executions) == 1


def test_rejects_invalid_final_status(run_recorder):
    run_recorder.start_run("random", "random", 1)
    with pytest.raises(ValueError, match="Invalid final status"):
        run_recorder.finish_run("running")
    assert not run_recorder.finished


def test_start_failure_rolls_back_without_assigning_run_id(run_recorder, fake_connection):
    fake_connection.error = OSError("Database unavailable")
    with pytest.raises(OSError, match="Database unavailable"):
        run_recorder.start_run("random", "random", 1)
    assert run_recorder.run_id is None
    assert fake_connection.rollbacks == 1


def test_finish_failure_keeps_run_active(run_recorder, fake_connection):
    run_recorder.start_run("random", "random", 1)
    fake_connection.error = OSError("Database unavailable")
    with pytest.raises(OSError):
        run_recorder.finish_run()
    assert not run_recorder.finished
    assert fake_connection.rollbacks == 1
