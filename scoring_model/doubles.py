"""
Doubles prop scoring: P(over the line) for batter doubles in a game
(typically posted at 0.5).

Method (per Scoring Framework Notes):
  1. Compute the batter's per-PA doubles rate, informed by gap power
     (hard-hit line drives/fly balls that reach the gaps but don't clear
     the fence — approximated from hard-hit% + line-drive% + fly-ball%
     combined, distinct from the HR-track barrel%), sprint speed (legging
     out a double matters more here than for singles), and extra-base
     hit rate / xSLG for spray-to-the-gaps hitters, shrunk + form-blended.
  2. Compute the pitcher's per-PA-allowed doubles rate, approximated from
     hits-allowed rate scaled by the league-average doubles share of
     hits, nudged by barrel% allowed (gap power against).
  3. Blend via log5 against the league-average doubles-per-PA rate.
  4. Apply the doubles-SPECIFIC park factor from the pipeline (NOT the
     blended HR park factor) — per the explicit build instruction.
  5. Aggregate across expected PAs via Poisson.

JUDGMENT CALL: "gap power" isn't a single named Statcast column. Built
here as a composite of hard-hit% and line-drive%/fly-ball% (balls hit
hard enough to reach the gap, but with a batted-ball profile that isn't
purely fly-ball/pull-side like a HR swing) rather than reusing barrel%,
since barrel% is explicitly described in the notes as "different from
HR-track barrels." Sprint speed gets its own multiplier, separate from
gap power, since a ball needs to BOTH be hit well AND be legged out for
a double, and those are different skills.
"""
from __future__ import annotations

from dataclasses import dataclass

from expected_opportunities import expected_pa_for_lineup_spot
from league_constants import LEAGUE_AVG_2B_PER_PA, STABILIZATION_PA_2B, park_factor_multiplier
from probability_utils import (
    blend_form,
    clamp01,
    log5,
    prob_over_line,
    rate_per_pa_from_per9,
    shrink_rate,
)

LEAGUE_AVG_HARD_HIT_PCT = 0.38
LEAGUE_AVG_LD_PCT = 0.21
LEAGUE_AVG_FB_PCT = 0.35
LEAGUE_AVG_SPRINT_SPEED = 27.0
LEAGUE_AVG_XSLG = 0.400
LEAGUE_AVG_BARREL_PCT_ALLOWED = 0.08

GAP_POWER_WEIGHT = 0.30
SPEED_WEIGHT = 0.15

# League-average doubles share of all hits allowed, used to convert a
# pitcher's general hits-allowed rate into a doubles-specific estimate.
LEAGUE_AVG_2B_SHARE_OF_HITS = 0.19


@dataclass(frozen=True)
class DoublesInputs:
    batter_2b_per_pa_season: float
    batter_pa_season: float
    batter_2b_per_pa_recent: float | None = None
    batter_pa_recent: float | None = None
    batter_hard_hit_pct: float | None = None
    batter_line_drive_pct: float | None = None
    batter_fly_ball_pct: float | None = None
    batter_sprint_speed: float | None = None
    batter_xslg: float | None = None
    batter_lineup_spot: int | None = None

    pitcher_hits_per_9: float = 8.5
    pitcher_barrel_pct_allowed: float | None = None

    # Doubles-specific park factor (FanGraphs index, 100 = neutral) — from
    # park_factors.get_extra_base_park_factors()['doubles_park_factor'],
    # NOT get_hr_park_factor().
    park_2b_factor: float = 100.0


def _gap_power_multiplier(inputs: DoublesInputs) -> float:
    ratios = []
    if inputs.batter_hard_hit_pct is not None:
        ratios.append(inputs.batter_hard_hit_pct / LEAGUE_AVG_HARD_HIT_PCT)
    if inputs.batter_line_drive_pct is not None:
        ratios.append(inputs.batter_line_drive_pct / LEAGUE_AVG_LD_PCT)
    if inputs.batter_fly_ball_pct is not None:
        ratios.append(inputs.batter_fly_ball_pct / LEAGUE_AVG_FB_PCT)
    if inputs.batter_xslg is not None:
        ratios.append(inputs.batter_xslg / LEAGUE_AVG_XSLG)
    if not ratios:
        return 1.0
    return sum(ratios) / len(ratios)


def _speed_multiplier(sprint_speed: float | None) -> float:
    if sprint_speed is None:
        return 1.0
    return sprint_speed / LEAGUE_AVG_SPRINT_SPEED


def batter_2b_rate(inputs: DoublesInputs) -> float:
    if inputs.batter_2b_per_pa_recent is not None and inputs.batter_pa_recent:
        blended_rate, effective_n = blend_form(
            season_rate=inputs.batter_2b_per_pa_season,
            season_n=inputs.batter_pa_season,
            recent_rate=inputs.batter_2b_per_pa_recent,
            recent_n=inputs.batter_pa_recent,
        )
    else:
        blended_rate, effective_n = inputs.batter_2b_per_pa_season, inputs.batter_pa_season

    shrinkage = shrink_rate(
        observed_rate=blended_rate,
        sample_size=effective_n,
        league_avg=LEAGUE_AVG_2B_PER_PA,
        stabilization_point=STABILIZATION_PA_2B,
    )
    shrunk = shrinkage.shrunk_rate

    # Shrink the gap-power and speed multipliers toward neutral using the
    # same weight as the raw rate (see home_runs.py for rationale). Sprint
    # speed is a more physically stable metric than batted-ball-derived
    # gap power, but both are still measured over the same limited sample
    # window in practice, so the same shrinkage weight is applied to both
    # for consistency rather than inventing a separate, unvalidated decay
    # curve for speed specifically.
    w = shrinkage.shrinkage_weight
    raw_gap_mult = _gap_power_multiplier(inputs)
    raw_speed_mult = _speed_multiplier(inputs.batter_sprint_speed)
    gap_mult = (1 - w) * raw_gap_mult + w * 1.0
    speed_mult = (1 - w) * raw_speed_mult + w * 1.0

    remaining_weight = 1 - GAP_POWER_WEIGHT - SPEED_WEIGHT
    adjustment = (
        remaining_weight * 1.0 + GAP_POWER_WEIGHT * gap_mult + SPEED_WEIGHT * speed_mult
    )
    return shrunk * adjustment


def pitcher_2b_rate_allowed(inputs: DoublesInputs) -> float:
    hits_per_pa_allowed = rate_per_pa_from_per9(inputs.pitcher_hits_per_9)
    doubles_per_pa_allowed = hits_per_pa_allowed * LEAGUE_AVG_2B_SHARE_OF_HITS
    if inputs.pitcher_barrel_pct_allowed is not None:
        barrel_mult = inputs.pitcher_barrel_pct_allowed / LEAGUE_AVG_BARREL_PCT_ALLOWED
        doubles_per_pa_allowed = doubles_per_pa_allowed * (0.8 + 0.2 * barrel_mult)
    return doubles_per_pa_allowed


def score_doubles_prop(inputs: DoublesInputs, line: float) -> float:
    b_rate = clamp01(batter_2b_rate(inputs))
    p_rate = clamp01(pitcher_2b_rate_allowed(inputs))

    matchup_rate = log5(b_rate, p_rate, LEAGUE_AVG_2B_PER_PA)

    park_mult = park_factor_multiplier(inputs.park_2b_factor)
    adjusted_rate = clamp01(matchup_rate * park_mult)

    expected_pa = expected_pa_for_lineup_spot(inputs.batter_lineup_spot)
    lam = adjusted_rate * expected_pa

    return clamp01(prob_over_line(lam, line))
