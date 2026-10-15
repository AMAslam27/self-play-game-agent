"""Confidence intervals for sampled evaluation outcomes."""

from math import sqrt
from statistics import NormalDist

CONFIDENCE_LEVEL = 0.95
_Z95 = NormalDist().inv_cdf((1 + CONFIDENCE_LEVEL) / 2)


def wilson_interval(events: int, trials: int) -> tuple[float, float]:
    """Return two-sided 95% Wilson bounds for an independent binomial sample.

    Counts refer to repeated samples from the same fixed event probability.
    Bounds are proportions in [0, 1], not percentages. Zero observed events
    still have a positive upper bound; zero trials cannot define an interval.
    """
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError("trials must be a positive integer")
    if (
        isinstance(events, bool)
        or not isinstance(events, int)
        or not 0 <= events <= trials
    ):
        raise ValueError("events must be an integer between zero and trials")
    proportion = events / trials
    z_squared = _Z95**2
    denominator = 1 + z_squared / trials
    centre = (proportion + z_squared / (2 * trials)) / denominator
    half_width = (
        _Z95
        * sqrt(proportion * (1 - proportion) / trials + z_squared / (4 * trials**2))
        / denominator
    )
    lower = 0.0 if events == 0 else max(0.0, centre - half_width)
    upper = 1.0 if events == trials else min(1.0, centre + half_width)
    return lower, upper
