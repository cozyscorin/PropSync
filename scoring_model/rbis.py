"""
RBI prop scoring: P(over the line) for batter RBIs in a game (e.g. Over
0.5, Over 1.5).

Method (per Scoring Framework Notes — RBIs are explicitly "less about the
batter's own skill, more about context"):
  1. Start from the batter's own season RBI-per-PA rate (shrunk + form-
     blended like every other prop) as a baseline skill signal — even a
     context-driven stat has some real per-player signal (RBI-skill
     hitters who hit the ball hard with men on base do out-RBI their
     teammates in the same lineup spot over a full season).
  2. Scale that baseline by two context multipliers the notes call out
     explicitly:
       a. On-base ability of the hitters batting ahead of this player
          (more runners on base ahead of you = more RBI chances) —
          expressed as an OBP-ahead multiplier relative to league-average
          OBP.
       b. Team implied run total for the game (the Vegas team total, not
          a player stat) — expressed as a multiplier relative to a
          league-average team run total per game (~4.5 runs in the
          modern era).
  3. Blend in the opposing pitcher's general "damage allowed" via the
     same per-PA-allowed approach used elsewhere (log5), since a pitcher
     who's easy to drive in runs against still matters even in a
     context-heavy prop.
  4. Aggregate across expected PAs via Poisson.

JUDGMENT CALL: there's no clean Statcast-native "RBI opportunity rate."
This module treats "lineup OBP ahead" and "team implied run total" as
linear multipliers on the batter's own per-PA RBI rate rather than trying
to model run-expectancy-by-base/out-state directly (a full run-expectancy
matrix would need base/out state data per PA, which the pipeline doesn't
expose and which is overkill for a props model at this stage). See
README.
"""
from __future__ import annotations

from dataclasses import dataclass

from expected_opportunities import expected_pa_for_lineup_spot
from league_constants import LEAGUE_AVG_RBI_PER_PA, STABILIZATION_PA_RBI
from probability_utils import (
    blend_form,
    clamp01,
    log5,
    prob_over_line,
    rate_per_pa_from_per9,
    shrink_rate,
)

LEAGUE_AVG_OBP = 0.315
LEAGUE_AVG_TEAM_RUNS_PER_GAME = 4.5

# Judgment-call weights: how strongly the two context multipliers move the
# baseline RBI rate away from 1.0. 1.0 = full linear effect; lower values
# dampen the swing so a single extreme context input (e.g. a very low team
# total) doesn't dominate the model. 0.6 was chosen to keep both
# multipliers meaningfully influential without letting them overwhelm the
# batter's own demonstrated RBI skill signal entirely.
ON_BASE_AHEAD_SENSITIVITY = 0.6
TEAM_RUN_TOTAL_SENSITIVITY = 0.6


@dataclass(frozen=True)
class RBIInputs:
    batter_rbi_per_pa_season: float
    batter_pa_season: float
    batter_rbi_per_pa_recent: float | None = None
    batter_pa_recent: float | None = None
    batter_lineup_spot: int | None = None

    # Context inputs (notes-specified) — both optional, default to neutral.
    obp_of_hitters_ahead: float | None = None   # average OBP of the 1-3 hitters batting ahead
    team_implied_run_total: float | None = None  # Vegas team total for tonight's game

    # Pitcher side: general "damage allowed" proxy.
    pitcher_hits_per_9: float = 8.5
    pitcher_hr_per_9: float = 1.2


def _on_base_ahead_multiplier(obp_ahead: float | None) -> float:
    if obp_ahead is None:
        return 1.0
    raw_ratio = obp_ahead / LEAGUE_AVG_OBP
    # Dampen the swing around 1.0 by the sensitivity factor.
    return 1.0 + (raw_ratio - 1.0) * ON_BASE_AHEAD_SENSITIVITY


def _team_run_total_multiplier(team_total: float | None) -> float:
    if team_total is None:
        return 1.0
    raw_ratio = team_total / LEAGUE_AVG_TEAM_RUNS_PER_GAME
    return 1.0 + (raw_ratio - 1.0) * TEAM_RUN_TOTAL_SENSITIVITY


def batter_rbi_rate(inputs: RBIInputs) -> float:
    if inputs.batter_rbi_per_pa_recent is not None and inputs.batter_pa_recent:
        blended_rate, effective_n = blend_form(
            season_rate=inputs.batter_rbi_per_pa_season,
            season_n=inputs.batter_pa_season,
            recent_rate=inputs.batter_rbi_per_pa_recent,
            recent_n=inputs.batter_pa_recent,
        )
    else:
        blended_rate, effective_n = inputs.batter_rbi_per_pa_season, inputs.batter_pa_season

    shrunk = shrink_rate(
        observed_rate=blended_rate,
        sample_size=effective_n,
        league_avg=LEAGUE_AVG_RBI_PER_PA,
        stabilization_point=STABILIZATION_PA_RBI,
    ).shrunk_rate
    return shrunk


def pitcher_damage_allowed_rate(inputs: RBIInputs) -> float:
    """Same hits+HR composite approach used in total_bases, as a generic
    'how easy is this pitcher to score runs against' proxy."""
    hits_per_pa_allowed = rate_per_pa_from_per9(inputs.pitcher_hits_per_9)
    hr_per_pa_allowed = rate_per_pa_from_per9(inputs.pitcher_hr_per_9)
    return hits_per_pa_allowed + hr_per_pa_allowed * 1.5


def score_rbi_prop(inputs: RBIInputs, line: float) -> float:
    b_rate = clamp01(batter_rbi_rate(inputs))
    p_rate = clamp01(pitcher_damage_allowed_rate(inputs))

    matchup_rate = log5(b_rate, p_rate, LEAGUE_AVG_RBI_PER_PA)

    context_mult = _on_base_ahead_multiplier(
        inputs.obp_of_hitters_ahead
    ) * _team_run_total_multiplier(inputs.team_implied_run_total)

    adjusted_rate = clamp01(matchup_rate * context_mult)

    expected_pa = expected_pa_for_lineup_spot(inputs.batter_lineup_spot)
    lam = adjusted_rate * expected_pa

    return clamp01(prob_over_line(lam, line))
