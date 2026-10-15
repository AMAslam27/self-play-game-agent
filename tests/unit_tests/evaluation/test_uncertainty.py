import pytest

from evaluation.uncertainty import wilson_interval


@pytest.mark.parametrize("losses,expected", [(33, (0.78, 1.54)), (127, (3.57, 5.01))])
def test_reference_assessment_intervals(losses, expected):
    bounds = wilson_interval(losses, 3000)
    assert tuple(round(bound * 100, 2) for bound in bounds) == expected


def test_balanced_sample_reference_interval():
    assert wilson_interval(5, 10) == pytest.approx((0.2365930905, 0.7634069095))


def test_zero_events_does_not_claim_zero_uncertainty():
    lower, upper = wilson_interval(0, 100)
    assert lower == pytest.approx(0.0, abs=1e-15)
    assert upper == pytest.approx(0.0369934982)


def test_all_events_is_symmetric_and_bounded():
    lower, upper = wilson_interval(100, 100)
    assert lower == pytest.approx(0.9630065018)
    assert upper == pytest.approx(1.0)
    assert 0 <= lower <= upper <= 1


@pytest.mark.parametrize("events,trials", [(0, 0), (0, -1), (-1, 10), (11, 10),
                                         (True, 10), (0, True), (1.5, 10), (1, 10.5)])
def test_invalid_counts_are_rejected(events, trials):
    with pytest.raises(ValueError):
        wilson_interval(events, trials)
