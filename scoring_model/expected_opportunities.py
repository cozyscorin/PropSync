"""
Expected plate appearances (batters) and expected batters-faced (pitchers)
per game — the "how many chances does this player get tonight" input that
every counting-stat prop module needs before it can turn a per-PA rate
into a per-game probability.

ASSUMPTION CALLED OUT EXPLICITLY (see scoring_model README): the data
pipeline, as built, does not yet expose a lineup-spot field or a
projected-innings/pitch-count field. Until it does, this module uses
fixed, documented league-average estimates by lineup slot / starter
role. Every prop module takes expected PA / expected batters-faced as an
explicit parameter rather than hardcoding a single number internally, so
once the pipeline exposes real lineup/workload data, callers can pass
real numbers in and nothing in the prop modules themselves needs to
change.
"""
from __future__ import annotations

# Average plate appearances per 9-inning game by batting-order spot,
# reflecting that leadoff/early-order hitters bat one more time per game
# than the bottom of the order, on average, over a full season. These are
# standard, widely-cited MLB averages (lineup spot 1 averages ~4.6 PA/G,
# spot 9 averages ~3.7 PA/G in a typical season).
EXPECTED_PA_BY_LINEUP_SPOT: dict[int, float] = {
    1: 4.6,
    2: 4.5,
    3: 4.4,
    4: 4.3,
    5: 4.2,
    6: 4.1,
    7: 4.0,
    8: 3.9,
    9: 3.7,
}

DEFAULT_EXPECTED_PA = 4.1  # fallback if lineup spot is unknown


def expected_pa_for_lineup_spot(lineup_spot: int | None) -> float:
    """
    Expected plate appearances for a batter tonight, given their lineup
    spot (1-9). Falls back to a league-average estimate if lineup_spot is
    None (e.g. lineup not yet posted at scoring time).
    """
    if lineup_spot is None:
        return DEFAULT_EXPECTED_PA
    return EXPECTED_PA_BY_LINEUP_SPOT.get(lineup_spot, DEFAULT_EXPECTED_PA)


# Expected batters faced for a starting pitcher, based on expected innings
# pitched. ~4.3 batters faced per inning is the modern league-average
# workload (matches probability_utils.rate_per_pa_from_per9's default).
BATTERS_FACED_PER_INNING = 4.3

# Reasonable default expected innings for a healthy starter with no other
# workload signal available (modern usage patterns: most starters target
# 5-6 IP). Bullpen/reliever props are out of scope here — this module
# assumes a starting pitcher, consistent with `pitcher_strikeouts` props
# being posted against starters.
DEFAULT_EXPECTED_INNINGS = 5.5


def expected_batters_faced(expected_innings: float | None = None) -> float:
    """
    Expected batters faced for a starting pitcher tonight, given expected
    innings pitched. If expected_innings isn't supplied (e.g. the
    pipeline doesn't yet expose projected innings / recent pitch-count
    trends), falls back to DEFAULT_EXPECTED_INNINGS.

    Per the Scoring Framework Notes ("expected innings/pitch count...tied
    to recent workload and bullpen usage patterns") — real workload
    projection belongs in the data pipeline once it's built; this is a
    placeholder default until that exists.
    """
    innings = expected_innings if expected_innings is not None else DEFAULT_EXPECTED_INNINGS
    return innings * BATTERS_FACED_PER_INNING
