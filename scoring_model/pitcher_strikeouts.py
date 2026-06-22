"""
Pitcher strikeouts prop scoring: P(over the line) for a starting
pitcher's strikeouts in a game (e.g. Over 5.5, Over 6.5).

Method (per Scoring Framework Notes):
  1. Compute the pitcher's per-batter-faced strikeout rate from K/9
     (converted to a per-batter-faced basis) and CSW% (called strikes +
     whiffs — a better predictor than K/9 alone since it captures called
     third strikes too), shrunk toward league average and blended for
     recent form.
  2. Compute the opposing team's per-PA strikeout-susceptibility rate
     (their own team strikeout rate as batters, optionally split by
     platoon if the pitcher's primary throwing hand and the lineup's
     handedness composition are known).
  3. Blend via log5 against the league-average K-per-batter-faced rate —
     same matchup logic as the hitting props, applied symmetrically: a
     strikeout-heavy pitcher facing a strikeout-prone lineup should land
     well above either number alone, and log5 captures that correctly.
  4. Aggregate across the pitcher's EXPECTED BATTERS FACED for the game
     (not PAs — pitcher props are about everyone they face, tied to
     expected innings pitched / workload) via Poisson, return
     P(count > line).

Platoon splits (pitcher's K rate vs. lefties vs. righties) are called for
explicitly in the notes but get_platoon_splits() in the pipeline is not
yet implemented. This module accepts an optional `platoon_multiplier`
the same way hits.py does, defaulting to neutral (1.0) until that data
exists.

JUDGMENT CALL: workload (expected innings, tied to recent pitch-count /
bullpen usage trends per the notes) is the single biggest uncertainty in
this prop type, bigger than the K-rate-per-batter estimate itself — a
pitcher pulled after 4 innings instead of 6 loses a third of his strikeout
opportunities regardless of how dominant his stuff is. The pipeline does
not yet expose a workload/innings-projection feed, so expected_innings is
a required-but-overridable parameter here (see expected_opportunities.py
DEFAULT_EXPECTED_INNINGS) — flagged as the top priority data-pipeline gap
to close for this specific prop type. See README.
"""
from __future__ import annotations

from dataclasses import dataclass

from expected_opportunities import expected_batters_faced
from league_constants import LEAGUE_AVG_K_PER_BF, STABILIZATION_BF_K
from probability_utils import blend_form, clamp01, log5, prob_over_line, rate_per_pa_from_per9, shrink_rate

LEAGUE_AVG_CSW_PCT = 0.29   # ~29% league-average CSW rate
CSW_WEIGHT = 0.40            # CSW% is explicitly called a better predictor than K/9 alone,
                              # so it gets a heavier weight here than the analogous
                              # contact-quality weights in the batting prop modules.


@dataclass(frozen=True)
class PitcherStrikeoutInputs:
    pitcher_k_per_9_season: float
    pitcher_bf_season: float   # season batters faced, for shrinkage sample size
    pitcher_k_per_9_recent: float | None = None
    pitcher_bf_recent: float | None = None
    pitcher_csw_pct: float | None = None
    expected_innings: float | None = None  # None -> DEFAULT_EXPECTED_INNINGS fallback

    opponent_team_k_pct: float | None = None  # opposing lineup's overall K rate as batters
    platoon_multiplier: float = 1.0


def pitcher_k_rate(inputs: PitcherStrikeoutInputs) -> float:
    season_rate_per_bf = rate_per_pa_from_per9(inputs.pitcher_k_per_9_season)

    if inputs.pitcher_k_per_9_recent is not None and inputs.pitcher_bf_recent:
        recent_rate_per_bf = rate_per_pa_from_per9(inputs.pitcher_k_per_9_recent)
        blended_rate, effective_n = blend_form(
            season_rate=season_rate_per_bf,
            season_n=inputs.pitcher_bf_season,
            recent_rate=recent_rate_per_bf,
            recent_n=inputs.pitcher_bf_recent,
        )
    else:
        blended_rate, effective_n = season_rate_per_bf, inputs.pitcher_bf_season

    shrinkage = shrink_rate(
        observed_rate=blended_rate,
        sample_size=effective_n,
        league_avg=LEAGUE_AVG_K_PER_BF,
        stabilization_point=STABILIZATION_BF_K,
    )
    shrunk = shrinkage.shrunk_rate

    if inputs.pitcher_csw_pct is not None:
        # Shrink the CSW% multiplier toward neutral using the same weight
        # as the raw rate (see home_runs.py for rationale) — a handful of
        # batters faced can produce a fluky CSW% just as easily as a
        # fluky K rate.
        w = shrinkage.shrinkage_weight
        raw_csw_mult = inputs.pitcher_csw_pct / LEAGUE_AVG_CSW_PCT
        csw_mult = (1 - w) * raw_csw_mult + w * 1.0
        adjustment = (1 - CSW_WEIGHT) * 1.0 + CSW_WEIGHT * csw_mult
        shrunk = shrunk * adjustment

    return shrunk


def opponent_k_susceptibility_rate(inputs: PitcherStrikeoutInputs) -> float:
    """
    Opposing lineup's own strikeout tendency as batters, used as the
    'pitcher-allowed'-equivalent side of the log5 matchup blend (here it's
    the batting side's susceptibility rather than a pitcher's allowed
    rate, since this prop is scored from the pitcher's perspective — the
    matchup math is symmetric either way log5 is applied).
    """
    if inputs.opponent_team_k_pct is not None:
        return inputs.opponent_team_k_pct
    return LEAGUE_AVG_K_PER_BF  # neutral fallback if lineup-level K rate isn't available


def score_pitcher_strikeouts_prop(inputs: PitcherStrikeoutInputs, line: float) -> float:
    p_rate = clamp01(pitcher_k_rate(inputs))
    opp_rate = clamp01(opponent_k_susceptibility_rate(inputs))

    matchup_rate = log5(p_rate, opp_rate, LEAGUE_AVG_K_PER_BF)
    matchup_rate = clamp01(matchup_rate * inputs.platoon_multiplier)

    bf = expected_batters_faced(inputs.expected_innings)
    lam = matchup_rate * bf

    return clamp01(prob_over_line(lam, line))
