"""
Shared math for the PropSync scoring model.

This is the one place that implements:
  1. Poisson aggregation: turn a per-PA (or per-batter-faced) rate into a
     P(over the line) for a full game, given an expected number of
     opportunities.
  2. Small-sample shrinkage: pull noisy low-PA rates back toward a league
     average baseline before they're used in any per-PA rate calc.
  3. Log5 blending: combine a batter's rate and a pitcher's allowed-rate
     into a single matchup-specific rate, anchored to a league baseline.
  4. Recency-weighted form blending: combine season-long and rolling
     (15/30-day) rates into one "current form" estimate.

No prop-specific logic lives here — every prop module (home_runs.py,
hits.py, etc.) imports from this file rather than reimplementing any of
the above. Zero third-party dependencies: Poisson CDF is implemented by
hand against the stdlib (`math`) rather than via scipy, so this works
with no extra installs. If scipy is present, nothing here uses it, but
requirements.txt lists it as optional for anyone who wants to cross-check
against scipy.stats.poisson in a notebook.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. Poisson aggregation
# ---------------------------------------------------------------------------

def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for X ~ Poisson(lam)."""
    if lam < 0:
        raise ValueError(f"Poisson lambda must be >= 0, got {lam}")
    if k < 0:
        return 0.0
    # math.exp(-lam) underflows to 0.0 for very large lam (not a realistic
    # regime here — lam is a per-game expected count, always small) but
    # guard anyway rather than raising.
    try:
        return math.exp(-lam) * (lam ** k) / math.factorial(k)
    except OverflowError:
        return 0.0


def poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k) for X ~ Poisson(lam), summed directly (no scipy needed)."""
    if lam < 0:
        raise ValueError(f"Poisson lambda must be >= 0, got {lam}")
    if k < 0:
        return 0.0
    return sum(poisson_pmf(i, lam) for i in range(0, k + 1))


def poisson_sf(k: int, lam: float) -> float:
    """P(X > k) = 1 - P(X <= k). 'Survival function' for X ~ Poisson(lam)."""
    return max(0.0, 1.0 - poisson_cdf(k, lam))


def prob_over_line(expected_count: float, line: float) -> float:
    """
    P(count > line) for a count modeled as Poisson(expected_count), where
    `line` is a betting line like 0.5, 1.5, 2.5, 5.5, etc.

    MLB props are always posted on a half-integer (X.5) so there's no push;
    P(over X.5) = P(count >= X+1) = P(count > X) = poisson_sf(X, lam).
    We floor the line to the nearest integer below it to get that X, which
    handles both the standard X.5 case and a defensive fallback if a whole
    -number line (e.g. a yes/no line modeled as "1+") is passed in.
    """
    if expected_count < 0:
        raise ValueError(f"expected_count must be >= 0, got {expected_count}")
    threshold = math.floor(line)
    return poisson_sf(threshold, expected_count)


def prob_at_least_one(expected_count: float) -> float:
    """
    P(count >= 1) for a count modeled as Poisson(expected_count). This is
    the standard form for "yes/no" props like 1+ HR: 1 - P(0 events).
    Equivalent to prob_over_line(expected_count, line=0.5) but kept as its
    own function since HR props are framed as yes/no, not over/under, in
    both the notes and the Odds API market (`batter_home_runs` posts a
    single "Yes" price keyed to 1+ HR, not a numeric line).
    """
    return 1.0 - math.exp(-expected_count)


# ---------------------------------------------------------------------------
# 2. Small-sample shrinkage
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShrinkageResult:
    raw_rate: float
    shrunk_rate: float
    sample_size: float
    league_avg: float
    shrinkage_weight: float  # weight given to the league average, 0-1


def shrink_rate(
    observed_rate: float,
    sample_size: float,
    league_avg: float,
    stabilization_point: float,
) -> ShrinkageResult:
    """
    Empirical-Bayes-style shrinkage toward a league-average baseline.

    Formula (the standard "stabilization point" approach used widely in
    sabermetrics, e.g. Russell Carleton's reliability work): treat the
    league average as a prior with weight equal to `stabilization_point`
    worth of observations, and the player's own observed rate as having
    weight equal to their actual sample size. The blended rate is the
    sample-size-weighted average of the two:

        shrunk_rate = (observed_rate * n + league_avg * k) / (n + k)

    where n = sample_size, k = stabilization_point.

    This formalizes exactly what the Scoring Framework Notes ask for —
    Kasper's gut-feel "discount small samples" — as one reusable, sound
    statistical method instead of an ad hoc per-prop fudge factor.

    Args:
        observed_rate: the player's own measured rate (e.g. HR per PA).
        sample_size: number of observations behind observed_rate (e.g.
            number of PAs in the lookback window used for this rate).
        league_avg: league-average rate for the same stat, used as the
            shrinkage target.
        stabilization_point: how many observations of "signal" it takes
            before the player's own rate should start to dominate the
            league-average prior. Larger = more skepticism toward
            small samples. Each prop module passes its own value here,
            since different rates stabilize at different sample sizes
            (e.g. HR rate per PA needs many more PAs to stabilize than
            strikeout rate per PA does — this is well documented in
            sabermetric reliability research). See each prop module's
            docstring/constant for the specific value and rationale.

    Returns a ShrinkageResult with both the raw and shrunk rate, plus the
    effective weight given to the league average, so callers/tests can
    inspect how much shrinkage was actually applied.
    """
    if sample_size < 0:
        raise ValueError(f"sample_size must be >= 0, got {sample_size}")
    if stabilization_point <= 0:
        raise ValueError(f"stabilization_point must be > 0, got {stabilization_point}")

    n = sample_size
    k = stabilization_point
    shrinkage_weight = k / (n + k)
    shrunk = (observed_rate * n + league_avg * k) / (n + k)

    return ShrinkageResult(
        raw_rate=observed_rate,
        shrunk_rate=shrunk,
        sample_size=n,
        league_avg=league_avg,
        shrinkage_weight=shrinkage_weight,
    )


# ---------------------------------------------------------------------------
# 3. Log5 matchup blending
# ---------------------------------------------------------------------------

def log5(batter_rate: float, pitcher_allowed_rate: float, league_avg: float) -> float:
    """
    Bill James' log5 method: combine a batter's rate stat and the
    opposing pitcher's allowed-rate for the same stat into a single
    matchup-specific probability, anchored to the league-average rate for
    that stat. This is the standard, well-documented sabermetric approach
    for "what's the rate when these two specific opponents face each
    other" — chosen over a naive average per the build instructions.

    Formula:
        log5(A, B, lg) = (A * B / lg) / (A * B / lg + (1-A) * (1-B) / (1-lg))

    All three inputs must be rates in [0, 1] (e.g. HR per PA, not HR per
    9 innings — convert pitcher rate stats to a per-PA or per-batter-faced
    basis before calling this).

    Degenerate cases (rate exactly 0 or 1, or league_avg at the boundary)
    are clamped slightly inward to avoid division by zero, since real
    baseball rates are never exactly 0 or 1 over a large enough sample.
    """
    eps = 1e-6
    a = min(max(batter_rate, eps), 1 - eps)
    b = min(max(pitcher_allowed_rate, eps), 1 - eps)
    lg = min(max(league_avg, eps), 1 - eps)

    numerator = (a * b) / lg
    denominator = numerator + ((1 - a) * (1 - b)) / (1 - lg)
    return numerator / denominator


# ---------------------------------------------------------------------------
# 4. Recency-weighted form blending
# ---------------------------------------------------------------------------

def blend_form(
    season_rate: float,
    season_n: float,
    recent_rate: float,
    recent_n: float,
    recent_weight_multiplier: float = 2.0,
) -> tuple[float, float]:
    """
    Combine a season-long rate and a recent-window (15/30-day) rate into
    one "current form" rate + an effective combined sample size, by
    weighting the recent sample more heavily per PA than the season
    sample (recency-weighting), while still respecting that more data is
    more data.

    Approach: treat this as a weighted average of the two rates, weighted
    by each sample's size times a recency multiplier on the recent
    window. This is a deliberately simple, explainable recency weighting
    — not an exponential decay model — because the inputs are already
    two discrete windows (season total, 15/30-day rolling), not a
    per-game time series, per how the data pipeline's rolling_form.py is
    built (it produces discrete N-day aggregates, not per-game rows).

    Args:
        season_rate: rate over the full season sample.
        season_n: number of observations (PAs/batters faced) in the
            season sample.
        recent_rate: rate over the recent rolling window (15 or 30 days).
        recent_n: number of observations in the recent window.
        recent_weight_multiplier: how many times more weight to give each
            recent-window observation vs. each season observation.
            Default 2.0 means one recent PA "counts" as much as two
            season PAs when blending. This is a judgment call (see
            scoring_model README) — there's no single textbook-correct
            multiplier; 2.0 is a moderate recency tilt, strong enough to
            matter but not so strong that 15 recent PAs swamp 400 season
            PAs.

    Returns (blended_rate, effective_n) — effective_n is the season_n +
    recency-weighted recent_n, suitable for passing into shrink_rate() as
    the sample_size, so a hot streak built on very few PAs still gets
    shrunk appropriately even after the recency tilt.
    """
    if season_n < 0 or recent_n < 0:
        raise ValueError("sample sizes must be >= 0")
    if season_n == 0 and recent_n == 0:
        raise ValueError("at least one of season_n/recent_n must be > 0")

    weighted_recent_n = recent_n * recent_weight_multiplier
    total_weight = season_n + weighted_recent_n
    if total_weight == 0:
        return (season_rate, 0.0)

    blended_rate = (
        season_rate * season_n + recent_rate * weighted_recent_n
    ) / total_weight

    # Effective sample size for downstream shrinkage: don't let the
    # recency multiplier artificially inflate the *count* of real
    # observations behind the blended rate (that would under-shrink a
    # genuinely small recent sample). Use season_n + recent_n (unweighted)
    # as the effective n fed into shrinkage, while the *rate* itself still
    # reflects the recency tilt above.
    effective_n = season_n + recent_n
    return (blended_rate, effective_n)


# ---------------------------------------------------------------------------
# Misc small helpers shared across prop modules
# ---------------------------------------------------------------------------

def clamp01(x: float) -> float:
    """Clamp a probability into [0, 1] — guards against float drift."""
    return min(1.0, max(0.0, x))


def rate_per_pa_from_per9(per9_rate: float, pa_per_inning: float = 4.3) -> float:
    """
    Convert a pitcher's per-9-innings rate stat (e.g. HR/9, K/9) into a
    per-batter-faced rate, for use in log5 blending (which needs rates on
    the same per-event basis as the batter's per-PA rate).

    pa_per_inning defaults to ~4.3, the long-run MLB average number of
    batters a pitcher faces per inning (roughly 38-39 batters per 9
    innings is the modern league-average workload). This is an
    approximation — see scoring_model README for why a fixed constant is
    used here rather than the pitcher's own actual batters-faced-per-inning,
    which the data pipeline does not currently expose.
    """
    batters_per_9 = pa_per_inning * 9
    return per9_rate / batters_per_9
