"""
Hits prop scoring: P(over the line) for batter hits in a game (e.g. Over
0.5, Over 1.5).

Method (per Scoring Framework Notes):
  1. Compute the batter's per-PA hit rate, informed by xBA and overall
     hard-hit% (NOT barrel% — the notes are explicit that contact rate
     matters more than contact quality for plain hits props), shrunk
     toward league average and blended for recent form.
  2. Compute the pitcher's per-PA-allowed hit rate from WHIP (converted to
     a per-PA basis) and opponent batting average allowed.
  3. Blend via log5 against the league-average hit-per-PA rate.
  4. Aggregate across expected PAs via Poisson, return P(count > line).

Platoon splits (batter vs. pitcher handedness) are an explicit input per
the notes but the data pipeline's get_platoon_splits() is not yet
implemented (raises NotImplementedError — see data_pipeline README). This
module accepts an optional `platoon_multiplier` so the caller can apply
one once that data exists; defaults to 1.0 (neutral) until then.
"""
from __future__ import annotations

from dataclasses import dataclass

from expected_opportunities import expected_pa_for_lineup_spot
from league_constants import LEAGUE_AVG_HIT_PER_PA, STABILIZATION_PA_HIT
from probability_utils import blend_form, clamp01, log5, prob_over_line, shrink_rate

LEAGUE_AVG_HARD_HIT_PCT = 0.38   # ~38% league-average hard-hit rate (95+ mph EV)
LEAGUE_AVG_XBA = 0.245
LEAGUE_AVG_WHIP = 1.30

# Weight given to xBA/hard-hit% contact-quality signal vs. the batter's
# raw observed hit rate. Lower than the HR module's power-metric weight
# (0.35) because for plain hits, the raw hit rate (driven heavily by BA,
# already a hits-per-AB stat) is a more direct, less noisy signal than it
# is for HR — barrel/power metrics are a bigger marginal upgrade over raw
# HR rate than xBA/hard-hit% are over raw hit rate. JUDGMENT CALL.
CONTACT_METRIC_WEIGHT = 0.25


@dataclass(frozen=True)
class HitsInputs:
    batter_hit_per_pa_season: float
    batter_pa_season: float
    batter_hit_per_pa_recent: float | None = None
    batter_pa_recent: float | None = None
    batter_xba: float | None = None
    batter_hard_hit_pct: float | None = None
    batter_lineup_spot: int | None = None

    pitcher_whip: float = 1.30
    pitcher_ba_allowed: float | None = None

    platoon_multiplier: float = 1.0


def _contact_adjustment_multiplier(xba: float | None, hard_hit_pct: float | None) -> float:
    ratios = []
    if xba is not None:
        ratios.append(xba / LEAGUE_AVG_XBA)
    if hard_hit_pct is not None:
        ratios.append(hard_hit_pct / LEAGUE_AVG_HARD_HIT_PCT)
    if not ratios:
        return 1.0
    return sum(ratios) / len(ratios)


def batter_hit_rate(inputs: HitsInputs) -> float:
    """
    NOTE on shrinkage: the contact-quality multiplier (xBA/hard-hit%) is
    itself shrunk toward neutral (1.0) using the same sample-size-derived
    shrinkage weight as the raw hit rate. Without this, a small sample's
    inflated xBA/hard-hit% could re-introduce the overconfidence that
    shrinking the raw rate was supposed to remove (see home_runs.py for
    the same fix, with a worked example in the test suite that caught
    this).
    """
    if inputs.batter_hit_per_pa_recent is not None and inputs.batter_pa_recent:
        blended_rate, effective_n = blend_form(
            season_rate=inputs.batter_hit_per_pa_season,
            season_n=inputs.batter_pa_season,
            recent_rate=inputs.batter_hit_per_pa_recent,
            recent_n=inputs.batter_pa_recent,
        )
    else:
        blended_rate, effective_n = inputs.batter_hit_per_pa_season, inputs.batter_pa_season

    shrinkage = shrink_rate(
        observed_rate=blended_rate,
        sample_size=effective_n,
        league_avg=LEAGUE_AVG_HIT_PER_PA,
        stabilization_point=STABILIZATION_PA_HIT,
    )
    shrunk = shrinkage.shrunk_rate

    raw_contact_mult = _contact_adjustment_multiplier(inputs.batter_xba, inputs.batter_hard_hit_pct)
    w = shrinkage.shrinkage_weight
    contact_mult = (1 - w) * raw_contact_mult + w * 1.0

    adjustment = (1 - CONTACT_METRIC_WEIGHT) * 1.0 + CONTACT_METRIC_WEIGHT * contact_mult
    return shrunk * adjustment


def pitcher_hit_rate_allowed(inputs: HitsInputs) -> float:
    """
    WHIP = (Walks + Hits) / IP. Without a separate walk rate, approximate
    hits/IP as a fraction of WHIP (roughly 70% of WHIP events are hits at
    league level, the rest walks) then convert IP-basis to per-PA basis
    using the same batters-per-inning constant used elsewhere.
    JUDGMENT CALL: this is an approximation given WHIP conflates hits and
    walks. If/when the pipeline exposes opponent BA allowed directly
    (more precise), prefer pitcher_ba_allowed instead — already wired in
    below with priority over the WHIP approximation.
    """
    if inputs.pitcher_ba_allowed is not None:
        return inputs.pitcher_ba_allowed  # already a roughly per-PA hit rate (~per-AB)

    hits_per_ip = inputs.pitcher_whip * 0.70
    from probability_utils import rate_per_pa_from_per9

    # Reuse the /9-to-per-PA conversion by scaling hits_per_ip to a /9 rate.
    hits_per_9 = hits_per_ip * 9
    return rate_per_pa_from_per9(hits_per_9)


def score_hits_prop(inputs: HitsInputs, line: float) -> float:
    b_rate = clamp01(batter_hit_rate(inputs))
    p_rate = clamp01(pitcher_hit_rate_allowed(inputs))

    matchup_rate = log5(b_rate, p_rate, LEAGUE_AVG_HIT_PER_PA)
    matchup_rate = clamp01(matchup_rate * inputs.platoon_multiplier)

    expected_pa = expected_pa_for_lineup_spot(inputs.batter_lineup_spot)
    lam = matchup_rate * expected_pa

    return clamp01(prob_over_line(lam, line))
