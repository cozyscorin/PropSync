"""
League-average per-PA (or per-batter-faced) baseline rates, used as the
shrinkage target and the log5 anchor for every prop module.

These are approximate modern-era (2021-2025) MLB league averages, sourced
from publicly known, stable seasonal aggregates (FanGraphs/Baseball
Reference league totals fluctuate year to year by roughly a few tenths of
a percentage point — these constants are deliberately round, "close
enough" anchors, not pulled live from the pipeline).

IMPORTANT: once the data pipeline is live, these should ideally be
replaced by an actual league-average computed from that season's full
batting_stats()/pitching_stats() pull (sum league-wide events / sum
league-wide PA) rather than a hardcoded constant — see scoring_model
README "What's a judgment call" section. Hardcoded here for now so the
scoring layer works standalone without requiring a live league-wide
aggregate on every call. Swap-in point is exactly these module-level
constants.
"""
from __future__ import annotations

# --- Per-PA league-average rates (batting side) ---------------------------
LEAGUE_AVG_HR_PER_PA = 0.032       # ~3.2% of PAs end in a HR
LEAGUE_AVG_HIT_PER_PA = 0.236      # ~23.6% of PAs end in a hit (league BA-ish, per-PA not per-AB)
LEAGUE_AVG_1B_PER_PA = 0.145       # singles make up the bulk of hits
LEAGUE_AVG_2B_PER_PA = 0.045
LEAGUE_AVG_3B_PER_PA = 0.005
LEAGUE_AVG_TB_PER_PA = 0.405       # total bases per PA (~ISO + BA-ish composite)
LEAGUE_AVG_RBI_PER_PA = 0.115
LEAGUE_AVG_K_PER_PA = 0.225        # ~22.5% strikeout rate, batter side

# --- Per-batter-faced league-average rate (pitching side) -----------------
LEAGUE_AVG_K_PER_BF = 0.225        # symmetric with batter K rate at league level
LEAGUE_AVG_HR_PER_BF = 0.032

# --- Stabilization points (shrinkage target sample sizes) -----------------
# How many PAs/batters-faced of "signal" it takes before a player's own
# rate should meaningfully outweigh the league-average prior. Sourced from
# widely cited sabermetric reliability research (e.g. Russell Carleton's
# stabilization-point studies): HR rate and BABIP-driven outcomes (hits,
# singles) take much longer to stabilize than swing-and-miss / strikeout
# outcomes, which stabilize fastest because they depend almost entirely on
# the batter's/pitcher's own swing decisions rather than batted-ball luck.
STABILIZATION_PA_HR = 150
STABILIZATION_PA_HIT = 200
STABILIZATION_PA_1B = 200
STABILIZATION_PA_2B = 250
STABILIZATION_PA_TB = 200
STABILIZATION_PA_RBI = 250          # heavily context-driven, treat as slow-stabilizing
STABILIZATION_BF_K = 60             # K% stabilizes fast (pitcher's own data)

# --- Park factor scaling -----------------------------------------------
# FanGraphs park factors are indexed to 100 = league average. Convert to a
# multiplier by dividing by 100.
def park_factor_multiplier(park_factor_index: float) -> float:
    """E.g. a park factor of 112 (12% HR-friendly) -> 1.12 multiplier."""
    return park_factor_index / 100.0
