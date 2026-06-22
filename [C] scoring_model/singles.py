"""
Singles prop scoring: P(over the line) for batter singles in a game.

Method (per Scoring Framework Notes — singles emphasize contact and
speed over power):
  1. Compute the batter's per-PA singles rate, informed by:
       - groundball% / line-drive% (most singles come off balls that stay
         in the infield/short outfield, not fly balls)
       - sprint speed (infield hits / beating out throws)
       - xBA on non-power contact (approximated here as overall xBA
         de-weighted by the batter's power profile — see judgment call
         below; the pipeline doesn't expose a separate "xBA on
         non-extra-base contact" split)
     shrunk toward league average, blended for recent form.
  2. Compute the pitcher's per-PA-allowed singles rate from groundball
     rate allowed and overall hits-allowed rate.
  3. Blend via log5 against the league-average singles-per-PA rate.
  4. Aggregate across expected PAs via Poisson.

JUDGMENT CALL: "xBA on non-power contact" (explicitly called for in the
notes) isn't a column the pipeline currently exposes — FanGraphs/Savant's
xBA leaderboards report one blended figure. Approximated here by taking
the batter's overall xBA and stripping out the estimated extra-base
share using ISO as a proxy for how much of their hard contact turns into
XBH rather than singles. This is a reasonable stand-in, not a perfect
substitute — flagged in the README for revisit if/when a play-by-play
derivation (non-HR, non-2B/3B batted-ball xBA from raw Statcast) gets
built into the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

from expected_opportunities import expected_pa_for_lineup_spot
from league_constants import LEAGUE_AVG_1B_PER_PA, STABILIZATION_PA_1B
from probability_utils import (
    blend_form,
    clamp01,
    log5,
    prob_over_line,
    rate_per_pa_from_per9,
    shrink_rate,
)

LEAGUE_AVG_GB_PCT = 0.43
LEAGUE_AVG_LD_PCT = 0.21
LEAGUE_AVG_SPRINT_SPEED = 27.0   # ft/sec, Statcast league-average sprint speed
LEAGUE_AVG_XBA = 0.245
LEAGUE_AVG_ISO = 0.155

CONTACT_PROFILE_WEIGHT = 0.30   # weight given to GB%/LD%/sprint speed vs raw observed singles rate


@dataclass(frozen=True)
class SinglesInputs:
    batter_1b_per_pa_season: float
    batter_pa_season: float
    batter_1b_per_pa_recent: float | None = None
    batter_pa_recent: float | None = None
    batter_groundball_pct: float | None = None
    batter_line_drive_pct: float | None = None
    batter_sprint_speed: float | None = None
    batter_xba: float | None = None
    batter_iso: float | None = None
    batter_lineup_spot: int | None = None

    pitcher_groundball_pct_allowed: float | None = None
    pitcher_hits_per_9: float = 8.5


def _non_power_xba(xba: float | None, iso: float | None) -> float | None:
    """
    Approximate "xBA on non-power contact" by discounting overall xBA in
    proportion to how much above-average power (ISO) the batter has — a
    higher-ISO hitter converts more of their hard contact into XBH rather
    than singles, so their *overall* xBA overstates their *singles-specific*
    contact-driven average. JUDGMENT CALL — see module docstring.
    """
    if xba is None:
        return None
    if iso is None:
        return xba
    power_ratio = iso / LEAGUE_AVG_ISO
    # Discount xBA down as power_ratio rises above 1.0; floor the discount
    # so this never zeroes out a real signal.
    discount = max(0.6, 1.0 - 0.15 * max(0.0, power_ratio - 1.0))
    return xba * discount


def _contact_profile_multiplier(inputs: SinglesInputs) -> float:
    ratios = []
    if inputs.batter_groundball_pct is not None:
        ratios.append(inputs.batter_groundball_pct / LEAGUE_AVG_GB_PCT)
    if inputs.batter_line_drive_pct is not None:
        ratios.append(inputs.batter_line_drive_pct / LEAGUE_AVG_LD_PCT)
    if inputs.batter_sprint_speed is not None:
        ratios.append(inputs.batter_sprint_speed / LEAGUE_AVG_SPRINT_SPEED)
    non_power_xba = _non_power_xba(inputs.batter_xba, inputs.batter_iso)
    if non_power_xba is not None:
        ratios.append(non_power_xba / LEAGUE_AVG_XBA)
    if not ratios:
        return 1.0
    return sum(ratios) / len(ratios)


def batter_1b_rate(inputs: SinglesInputs) -> float:
    if inputs.batter_1b_per_pa_recent is not None and inputs.batter_pa_recent:
        blended_rate, effective_n = blend_form(
            season_rate=inputs.batter_1b_per_pa_season,
            season_n=inputs.batter_pa_season,
            recent_rate=inputs.batter_1b_per_pa_recent,
            recent_n=inputs.batter_pa_recent,
        )
    else:
        blended_rate, effective_n = inputs.batter_1b_per_pa_season, inputs.batter_pa_season

    shrinkage = shrink_rate(
        observed_rate=blended_rate,
        sample_size=effective_n,
        league_avg=LEAGUE_AVG_1B_PER_PA,
        stabilization_point=STABILIZATION_PA_1B,
    )
    shrunk = shrinkage.shrunk_rate

    # Shrink the contact-profile multiplier toward neutral using the same
    # weight as the raw rate (see home_runs.py for rationale/original fix).
    raw_profile_mult = _contact_profile_multiplier(inputs)
    w = shrinkage.shrinkage_weight
    profile_mult = (1 - w) * raw_profile_mult + w * 1.0
    adjustment = (1 - CONTACT_PROFILE_WEIGHT) * 1.0 + CONTACT_PROFILE_WEIGHT * profile_mult
    return shrunk * adjustment


def pitcher_1b_rate_allowed(inputs: SinglesInputs) -> float:
    hits_per_pa_allowed = rate_per_pa_from_per9(inputs.pitcher_hits_per_9)
    if inputs.pitcher_groundball_pct_allowed is not None:
        gb_mult = inputs.pitcher_groundball_pct_allowed / LEAGUE_AVG_GB_PCT
        hits_per_pa_allowed = hits_per_pa_allowed * (0.7 + 0.3 * gb_mult)
    # Most hits allowed by groundball-heavy pitchers are singles (extra
    # bases require getting the ball past/over the outfield); approximate
    # singles share of hits allowed at ~73%, the long-run league average
    # singles share of all hits.
    return hits_per_pa_allowed * 0.73


def score_singles_prop(inputs: SinglesInputs, line: float) -> float:
    b_rate = clamp01(batter_1b_rate(inputs))
    p_rate = clamp01(pitcher_1b_rate_allowed(inputs))

    matchup_rate = log5(b_rate, p_rate, LEAGUE_AVG_1B_PER_PA)

    expected_pa = expected_pa_for_lineup_spot(inputs.batter_lineup_spot)
    lam = matchup_rate * expected_pa

    return clamp01(prob_over_line(lam, line))
