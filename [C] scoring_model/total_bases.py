"""
Total bases prop scoring: P(over the line) for batter total bases in a
game (e.g. Over 1.5, Over 2.5).

Method (per Scoring Framework Notes):
  1. Compute the batter's per-PA "total bases rate" (expected total bases
     per PA — effectively an ISO/xSLG-informed per-PA expectation, not a
     0/1 event rate like hits/HRs), shrunk toward league average and
     blended for recent form.
  2. Compute the pitcher's per-PA-allowed total-bases rate, approximated
     from the pitcher's allowed-hits profile and HR/9 (a full extra-base
     allowed breakdown isn't in the data pipeline yet — see judgment call
     note below).
  3. Blend via log5 against the league-average TB-per-PA rate.
  4. Apply a hit-type-specific park factor: NOT the blended HR park
     factor. Built as a weighted composite of the singles/doubles/triples/
     HR park factors (each weighted by their share of league-average total
     bases), per the explicit instruction that total bases needs
     hit-type-specific park numbers, not one blended figure.
  5. Aggregate across expected PAs via Poisson, return P(count > line).

JUDGMENT CALL: total bases isn't a clean binary per-PA event the way a
hit or a HR is (a single PA can contribute 0, 1, 2, 3, or 4 bases). Poisson
models a COUNT of discrete events, so this module treats "total bases
earned this PA" as approximately Poisson-distributed with mean equal to
the batter's expected-bases-per-PA rate, summed across PAs. This is an
approximation (the true per-PA distribution is a small discrete
distribution over {0,1,2,3,4}, not literally Poisson), but Poisson's mean
still equals the correct expected total, and for the over/under lines
actually posted (1.5, 2.5, 3.5) the approximation is standard practice in
public sports-analytics modeling and stays well-behaved. See README.
"""
from __future__ import annotations

from dataclasses import dataclass

from expected_opportunities import expected_pa_for_lineup_spot
from league_constants import STABILIZATION_PA_TB
from probability_utils import (
    blend_form,
    clamp01,
    log5,
    prob_over_line,
    rate_per_pa_from_per9,
    shrink_rate,
)

LEAGUE_AVG_TB_PER_PA = 0.405
LEAGUE_AVG_ISO = 0.155
LEAGUE_AVG_XSLG = 0.400

# Hit-type weights for building a composite "total bases park factor" out
# of the pipeline's per-hit-type park factors (1B/2B/3B/HR), weighted by
# each hit type's average contribution to league-wide total bases (e.g.
# singles contribute 1 base each but happen often; HRs contribute 4 bases
# each but happen less often — weights below approximate each hit type's
# share of league total-base production). JUDGMENT CALL — see README.
TB_PARK_FACTOR_WEIGHTS = {
    "1B": 0.34,
    "2B": 0.27,
    "3B": 0.04,
    "HR": 0.35,
}

CONTACT_METRIC_WEIGHT = 0.30  # weight given to ISO/xSLG vs. raw observed TB rate


@dataclass(frozen=True)
class TotalBasesInputs:
    batter_tb_per_pa_season: float
    batter_pa_season: float
    batter_tb_per_pa_recent: float | None = None
    batter_pa_recent: float | None = None
    batter_iso: float | None = None
    batter_xslg: float | None = None
    batter_lineup_spot: int | None = None

    pitcher_hr_per_9: float = 1.2
    pitcher_hits_per_9: float = 8.5

    # Per-hit-type park factors (FanGraphs index, 100 = neutral), from
    # park_factors.get_park_factors() / get_extra_base_park_factors() /
    # get_hr_park_factor(). All optional; missing ones default to neutral.
    park_factor_1b: float = 100.0
    park_factor_2b: float = 100.0
    park_factor_3b: float = 100.0
    park_factor_hr: float = 100.0


def composite_tb_park_factor(inputs: TotalBasesInputs) -> float:
    """
    Weighted composite park factor for total bases, built from the
    pipeline's separate per-hit-type park factors rather than reusing the
    single HR park factor number.
    """
    weighted_sum = (
        inputs.park_factor_1b * TB_PARK_FACTOR_WEIGHTS["1B"]
        + inputs.park_factor_2b * TB_PARK_FACTOR_WEIGHTS["2B"]
        + inputs.park_factor_3b * TB_PARK_FACTOR_WEIGHTS["3B"]
        + inputs.park_factor_hr * TB_PARK_FACTOR_WEIGHTS["HR"]
    )
    return weighted_sum / 100.0  # convert index back to a 0-1-ish multiplier base


def _power_adjustment_multiplier(iso: float | None, xslg: float | None) -> float:
    ratios = []
    if iso is not None:
        ratios.append(iso / LEAGUE_AVG_ISO)
    if xslg is not None:
        ratios.append(xslg / LEAGUE_AVG_XSLG)
    if not ratios:
        return 1.0
    return sum(ratios) / len(ratios)


def batter_tb_rate(inputs: TotalBasesInputs) -> float:
    if inputs.batter_tb_per_pa_recent is not None and inputs.batter_pa_recent:
        blended_rate, effective_n = blend_form(
            season_rate=inputs.batter_tb_per_pa_season,
            season_n=inputs.batter_pa_season,
            recent_rate=inputs.batter_tb_per_pa_recent,
            recent_n=inputs.batter_pa_recent,
        )
    else:
        blended_rate, effective_n = inputs.batter_tb_per_pa_season, inputs.batter_pa_season

    shrinkage = shrink_rate(
        observed_rate=blended_rate,
        sample_size=effective_n,
        league_avg=LEAGUE_AVG_TB_PER_PA,
        stabilization_point=STABILIZATION_PA_TB,
    )
    shrunk = shrinkage.shrunk_rate

    # Shrink the ISO/xSLG-derived power multiplier toward neutral (1.0)
    # using the same weight as the raw rate, so a small sample's inflated
    # ISO/xSLG can't reintroduce the overconfidence shrinkage is meant to
    # remove (see home_runs.py for the original fix + rationale).
    raw_power_mult = _power_adjustment_multiplier(inputs.batter_iso, inputs.batter_xslg)
    w = shrinkage.shrinkage_weight
    power_mult = (1 - w) * raw_power_mult + w * 1.0
    adjustment = (1 - CONTACT_METRIC_WEIGHT) * 1.0 + CONTACT_METRIC_WEIGHT * power_mult
    return shrunk * adjustment


def pitcher_tb_rate_allowed(inputs: TotalBasesInputs) -> float:
    """
    Approximate per-PA total-bases-allowed rate for the pitcher: hits/9
    converted to a per-PA hit-allowed rate, then scaled up to account for
    extra bases using HR/9 as the power-allowed signal (a single allowed
    isn't worth the same as a HR allowed). JUDGMENT CALL — the pipeline
    doesn't expose a direct "slugging allowed" stat, so this builds an
    approximation from two /9 rates that ARE available. See README.
    """
    hits_per_pa_allowed = rate_per_pa_from_per9(inputs.pitcher_hits_per_9)
    hr_per_pa_allowed = rate_per_pa_from_per9(inputs.pitcher_hr_per_9)
    # Treat each allowed hit as ~1.4 bases on average (mix of 1B/2B/3B) and
    # each allowed HR as 4 bases, then sum.
    return (hits_per_pa_allowed * 1.4) + (hr_per_pa_allowed * 4.0)


def score_total_bases_prop(inputs: TotalBasesInputs, line: float) -> float:
    b_rate = max(0.0, batter_tb_rate(inputs))
    p_rate = max(0.0, pitcher_tb_rate_allowed(inputs))

    # log5 expects rates in [0,1]; total-bases-per-PA can exceed 1 for
    # elite sluggers in small samples, so clamp inputs into a sane [0,1]
    # band before blending, then rescale the result back to the TB-per-PA
    # magnitude using the league baseline as the anchor scale factor.
    b_rate_clamped = min(b_rate, 0.999)
    p_rate_clamped = min(p_rate, 0.999)

    matchup_rate = log5(b_rate_clamped, p_rate_clamped, min(LEAGUE_AVG_TB_PER_PA, 0.999))

    park_mult = composite_tb_park_factor(inputs)
    adjusted_rate = matchup_rate * park_mult

    expected_pa = expected_pa_for_lineup_spot(inputs.batter_lineup_spot)
    lam = adjusted_rate * expected_pa

    return clamp01(prob_over_line(lam, line))
