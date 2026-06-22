"""
Home run prop scoring: P(1+ HR) for a given batter/pitcher matchup.

Method (per Scoring Framework Notes + build instructions):
  1. Compute the batter's per-PA HR rate from rate stats (xSLG/barrel%-
     informed HR rate, shrunk toward league average for small samples).
  2. Compute the pitcher's per-PA-allowed HR rate from HR/9 (converted to
     a per-batter-faced basis) and barrel% allowed.
  3. Blend batter rate and pitcher-allowed rate via log5, anchored to the
     league-average HR-per-PA rate — NOT a naive average of the two, per
     the build instructions (log5 correctly handles that a great hitter
     facing a great pitcher should land close to league-average, not
     literally the arithmetic mean of two extremes).
  4. Apply park factor (HR-specific) and same-day weather (if available)
     as multipliers on the blended per-PA rate.
  5. Aggregate across the batter's expected PAs for the game using
     1 - exp(-lambda) (Poisson P(>=1 event)) to get P(1+ HR).

JUDGMENT CALL: barrel% and xSLG both speak to HR risk but aren't HR rates
themselves. This module blends them into a single "power score" via a
simple linear combination (weights below), then maps that power score
onto the batter's empirical HR/PA rate as a multiplicative adjustment —
rather than trying to regress barrel%/xSLG directly into a HR
probability, which would need real historical fitting data this project
doesn't have yet. See README "judgment calls" section for the full
rationale and what to revisit once real historical outcomes are
available to fit against.
"""
from __future__ import annotations

from dataclasses import dataclass

from expected_opportunities import expected_pa_for_lineup_spot
from league_constants import (
    LEAGUE_AVG_HR_PER_PA,
    STABILIZATION_PA_HR,
    park_factor_multiplier,
)
from probability_utils import (
    blend_form,
    clamp01,
    log5,
    prob_at_least_one,
    rate_per_pa_from_per9,
    shrink_rate,
)

# Weight given to barrel%/xSLG-implied power relative to the batter's raw
# observed HR/PA rate when forming the "power-adjusted" rate. 0.0 would
# ignore barrel%/xSLG entirely; 1.0 would ignore the batter's own raw HR
# rate entirely. 0.35 is a moderate tilt: raw HR rate still dominates
# (it's the most direct signal), but barrel%/xSLG nudge it, especially
# useful for batters whose HR rate is still a noisy small sample but who
# already show real power on contact quality metrics. JUDGMENT CALL — see
# README.
POWER_METRIC_WEIGHT = 0.35

# League-average reference points for barrel% and xSLG, used to normalize
# a batter's barrel%/xSLG into a multiplier centered at 1.0.
LEAGUE_AVG_BARREL_PCT = 0.08   # ~8% league-average barrel rate
LEAGUE_AVG_XSLG = 0.400


@dataclass(frozen=True)
class HomeRunInputs:
    # Batter inputs
    batter_hr_per_pa_season: float
    batter_pa_season: float
    batter_hr_per_pa_recent: float | None = None      # 15/30-day rolling
    batter_pa_recent: float | None = None
    batter_barrel_pct: float | None = None             # e.g. 0.12 for 12%
    batter_xslg: float | None = None
    batter_lineup_spot: int | None = None

    # Pitcher inputs
    pitcher_hr_per_9: float = 1.2                      # league-average-ish default
    pitcher_barrel_pct_allowed: float | None = None

    # Context
    park_hr_factor: float = 100.0   # FanGraphs index, 100 = neutral
    weather_hr_multiplier: float = 1.0   # e.g. 1.05 for wind blowing out; 1.0 if unknown/no data


def _power_adjustment_multiplier(
    barrel_pct: float | None, xslg: float | None
) -> float:
    """
    Turn barrel%/xSLG into a multiplier centered at 1.0 (league average ->
    1.0, above average -> >1.0). Averages the two normalized ratios when
    both are available; falls back to whichever one is available; returns
    1.0 (neutral) if neither is available.
    """
    ratios = []
    if barrel_pct is not None:
        ratios.append(barrel_pct / LEAGUE_AVG_BARREL_PCT)
    if xslg is not None:
        ratios.append(xslg / LEAGUE_AVG_XSLG)
    if not ratios:
        return 1.0
    return sum(ratios) / len(ratios)


def batter_hr_rate(inputs: HomeRunInputs) -> float:
    """
    Shrunk, form-blended, power-adjusted per-PA HR rate for the batter,
    independent of the opposing pitcher (matchup blending happens
    separately in score_home_run_prop).

    IMPORTANT: barrel%/xSLG are sample-dependent metrics too — a 10-PA
    sample's barrel% is just as noisy as its raw HR rate. If the power
    multiplier were applied at full strength regardless of sample size, a
    tiny hot streak with inflated barrel%/xSLG could outweigh the
    shrinkage already applied to the raw rate, defeating the point of
    shrinking in the first place. So the power multiplier itself gets
    pulled toward neutral (1.0) using the SAME shrinkage weight computed
    for the raw rate, before being blended in.
    """
    if inputs.batter_hr_per_pa_recent is not None and inputs.batter_pa_recent:
        blended_rate, effective_n = blend_form(
            season_rate=inputs.batter_hr_per_pa_season,
            season_n=inputs.batter_pa_season,
            recent_rate=inputs.batter_hr_per_pa_recent,
            recent_n=inputs.batter_pa_recent,
        )
    else:
        blended_rate, effective_n = inputs.batter_hr_per_pa_season, inputs.batter_pa_season

    shrinkage = shrink_rate(
        observed_rate=blended_rate,
        sample_size=effective_n,
        league_avg=LEAGUE_AVG_HR_PER_PA,
        stabilization_point=STABILIZATION_PA_HR,
    )
    shrunk = shrinkage.shrunk_rate

    raw_power_mult = _power_adjustment_multiplier(inputs.batter_barrel_pct, inputs.batter_xslg)
    # Shrink the power multiplier toward 1.0 (neutral) using the same
    # league-average shrinkage weight applied to the raw rate, so a small
    # sample's inflated barrel%/xSLG can't dominate after the raw rate has
    # already been shrunk.
    w = shrinkage.shrinkage_weight
    power_mult = (1 - w) * raw_power_mult + w * 1.0

    adjustment = (1 - POWER_METRIC_WEIGHT) * 1.0 + POWER_METRIC_WEIGHT * power_mult
    return shrunk * adjustment


def pitcher_hr_rate_allowed(inputs: HomeRunInputs) -> float:
    """
    Per-batter-faced HR rate allowed for the opposing pitcher, blending
    HR/9 (converted to a per-batter-faced basis) with barrel% allowed if
    available.
    """
    base_rate = rate_per_pa_from_per9(inputs.pitcher_hr_per_9)
    if inputs.pitcher_barrel_pct_allowed is not None:
        barrel_mult = inputs.pitcher_barrel_pct_allowed / LEAGUE_AVG_BARREL_PCT
        # Same moderate-tilt approach as the batter side, but pitcher side
        # uses a slightly lower weight since HR/9 already captures most of
        # the signal for a single pitcher (it's a direct HR count, not a
        # proxy) — barrel% allowed nudges rather than dominates.
        base_rate = base_rate * (0.75 + 0.25 * barrel_mult)
    return base_rate


def score_home_run_prop(inputs: HomeRunInputs) -> float:
    """
    Full pipeline: batter rate -> pitcher rate -> log5 matchup blend ->
    park factor -> weather -> aggregate across expected PAs -> P(1+ HR).
    """
    b_rate = clamp01(batter_hr_rate(inputs))
    p_rate = clamp01(pitcher_hr_rate_allowed(inputs))

    matchup_rate = log5(b_rate, p_rate, LEAGUE_AVG_HR_PER_PA)

    park_mult = park_factor_multiplier(inputs.park_hr_factor)
    adjusted_rate = matchup_rate * park_mult * inputs.weather_hr_multiplier
    adjusted_rate = clamp01(adjusted_rate)

    expected_pa = expected_pa_for_lineup_spot(inputs.batter_lineup_spot)
    lam = adjusted_rate * expected_pa

    return clamp01(prob_at_least_one(lam))
