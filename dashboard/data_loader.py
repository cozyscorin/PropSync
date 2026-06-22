"""
THE SEAM. This is the one module `app.py` imports data from. Everything
else in the dashboard (sample_data.py, formatting.py, app.py's rendering
code) is downstream of the two functions below.

------------------------------------------------------------------------
STATUS: the live path is now wired up (see `_load_raw_stats_live()` /
`_load_ranked_edges_live()` below), gated behind
`data_pipeline.config.odds_api_key_present()`. It has NOT been run
against real data in any sandbox — this project has never had outbound
internet access during a build (see ../[C] data_pipeline/README.md). The
live path was built against the pipeline's documented function
signatures and return shapes, and tested with realistic synthetic
DataFrames (see ../[C] edge_ranking/tests/test_live_integration.py), but
the first real run on cozy's machine is the first time any of this
touches actual pybaseball/FanGraphs/Odds-API output.
------------------------------------------------------------------------

So both `load_raw_stats()` and `load_ranked_edges()` below follow the
same shape:

  1. If `ODDS_API_KEY` isn't set (odds_api_key_present() is False), skip
     straight to sample data. No key means no odds to score against
     regardless of how well the stat pulls go, so there's no point
     attempting the live stat pulls at all in that case.
  2. If a key IS present, attempt the real live path.
  3. If the live path raises ANYTHING — missing pybaseball install,
     a column-name mismatch (genuinely expected on the first live run;
     see every pipeline README's repeated "print .columns and check"
     warning), a network error, an empty slate, anything — catch it,
     print a clear diagnostic to the console, and fall back to
     `sample_data.py`. The dashboard must never crash just because the
     live pipeline hit its first real-world edge case; that's the whole
     point of keeping sample_data.py in place rather than deleting it
     once a key exists.

`IS_LIVE_DATA` reflects whether the LAST load actually succeeded against
live data (not just whether a key is present) — the UI's banner should
say "live" only when live data really was used, not just attempted.

------------------------------------------------------------------------
WHAT GOT BUILT ON TOP OF THIS SEAM (see ../[C] edge_ranking/live_integration.py)
------------------------------------------------------------------------

Every prior README in this project flagged the same missing piece: an
`inputs_registry: dict[(player, market_key), *Inputs]` builder, the thing
`edge_ranking.ranking.build_candidate_legs()` needs in order to score real
odds rows. `sample_data._build_inputs_registry()` was always just a
worked example of the shape, built from fabricated profiles. The real
version now lives in `edge_ranking/live_integration.py`:

  - `build_inputs_registry(stat_dataframes, odds_df)` — the main builder.
  - `PlayerMatcher` / `normalize_name()` — resolves the odds feed's
    display-name strings (e.g. "Mike Trout") to the pipeline's own
    player records. No MLBAM-ID lookup table is wired into the pipeline
    today, so this runs on normalized-name matching only (case folding,
    accent stripping, Jr./Sr./III suffix stripping) — see that module's
    docstring for the documented failure modes (nicknames, name
    collisions, etc. are NOT handled).
  - `build_<prop>_inputs()` — one function per *Inputs dataclass, mapping
    real pipeline DataFrame columns to the exact fields each scoring
    module needs, including resolving the SPECIFIC opposing pitcher per
    game (not a league-wide average).

`_load_ranked_edges_live()` below is the only caller of
`build_inputs_registry()` in the project — that's deliberate, per the
"THE SEAM" idea: nothing else needs to know this registry-building step
exists.

Caching: `app.py` wraps calls to this module in `st.cache_data` with a
TTL. Keep that in place — each live call here can hit the Odds API (real
credits against the 500/month free tier) and pybaseball/FanGraphs
(slower, scrape-based). Don't remove the cache just because this is now
"real."
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import pandas as pd

# Same sys.path bootstrap pattern used throughout this project
# (edge_ranking/market_map.py, edge_ranking/live_integration.py,
# dashboard/sample_data.py).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _folder in ("scoring_model", "edge_ranking", "data_pipeline"):
    _p = _PROJECT_ROOT / _folder
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sample_data  # noqa: E402

# Reflects whether the MOST RECENT load actually used live data (not just
# whether a key is present/attempted). app.py's banner should read this
# after calling the loaders, not assume a fixed value at import time.
IS_LIVE_DATA = False


def _log_live_path_failure(stage: str, exc: Exception) -> None:
    """
    Single place that prints a clear, consistent diagnostic when the live
    path fails, so a real run's console output makes it obvious whether
    the problem is "no key," "pipeline column mismatch," "network error,"
    or something else entirely — per the task's explicit instruction that
    the first real run WILL likely surface something, and it should be
    easy to triage, not a silent fallback.
    """
    print(
        f"\n[data_loader] LIVE DATA PATH FAILED at stage '{stage}': "
        f"{type(exc).__name__}: {exc}\n"
        f"[data_loader] Falling back to sample data for this load. "
        f"Full traceback below for debugging:\n"
        f"{traceback.format_exc()}\n"
    )


def _load_raw_stats_live() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull real batter/pitcher season stats + sprint speed via the data
    pipeline. Returns (batter_stats_df, pitcher_stats_df) shaped as
    closely as practical to sample_data's columns, but this is the one
    function where exact column parity with sample_data.py is NOT
    guaranteed — see module docstring. app.py's raw-stats tab should
    tolerate either shape; if it doesn't yet, that's a follow-up, not a
    reason to block this function.
    """
    from statcast.savant import get_batter_season_stats, get_pitcher_season_stats
    from park_factors.park_factors import get_park_factors

    batter_df = get_batter_season_stats()
    pitcher_df = get_pitcher_season_stats()

    park_df = get_park_factors()
    team_col = next((c for c in ("Team", "team") if c in park_df.columns), None)
    if team_col and "Team" in batter_df.columns:
        rename_map = {c: f"park_factor_{c.lower()}" for c in ("1B", "2B", "3B", "HR") if c in park_df.columns}
        park_slim = park_df.rename(columns={team_col: "Team", **rename_map})
        keep_cols = ["Team"] + list(rename_map.values())
        batter_df = batter_df.merge(park_slim[keep_cols], on="Team", how="left")

    return batter_df, pitcher_df


def _load_ranked_edges_live() -> list:
    """
    The real live path: pull today's odds, pull the stat DataFrames the
    registry builder needs, build the inputs registry, then run the
    SAME `build_candidate_legs()` + `rank_edges()` calls sample_data.py's
    fabricated path already exercises.
    """
    from odds.odds_api_client import get_all_player_props_today
    from statcast.savant import get_batter_season_stats, get_pitcher_season_stats
    from statcast.rolling_form import (
        batter_rolling_batted_ball_profile,
        pitcher_rolling_profile,
    )
    from park_factors.park_factors import get_park_factors, get_extra_base_park_factors

    from live_integration import build_inputs_registry
    from ranking import build_candidate_legs, rank_edges

    odds_df = get_all_player_props_today()
    if odds_df is None or odds_df.empty:
        raise ValueError(
            "get_all_player_props_today() returned no rows — no games/props "
            "posted right now, or the Odds API call itself returned empty. "
            "Nothing to rank."
        )

    stat_dataframes: dict = {}

    # Each stat pull is wrapped individually: a single pipeline call
    # failing (e.g. one FanGraphs endpoint having an off day) shouldn't
    # block the others from at least being attempted. build_inputs_registry
    # already tolerates missing keys in stat_dataframes gracefully.
    def _try_pull(key: str, fn, *args):
        try:
            stat_dataframes[key] = fn(*args)
        except Exception as exc:  # noqa: BLE001
            print(f"[data_loader] live stat pull '{key}' failed, continuing without it: {exc}")

    _try_pull("batter_season", get_batter_season_stats)
    _try_pull("pitcher_season", get_pitcher_season_stats)
    _try_pull("park_factors", get_park_factors)
    _try_pull("extra_base_park_factors", get_extra_base_park_factors)
    _try_pull("batter_rolling_15d", batter_rolling_batted_ball_profile, 15)
    _try_pull("pitcher_rolling_15d", pitcher_rolling_profile, 15)

    if "batter_season" not in stat_dataframes and "pitcher_season" not in stat_dataframes:
        raise RuntimeError(
            "Both batter and pitcher season stat pulls failed — no usable "
            "player data to build an inputs registry from. See the "
            "individual pull failure messages above for the real cause."
        )

    registry = build_inputs_registry(stat_dataframes, odds_df)
    if not registry:
        raise ValueError(
            "build_inputs_registry() produced an empty registry — either no "
            "odds-feed player names matched any pipeline stat rows (see "
            "live_integration.PlayerMatcher's known failure modes), or the "
            "stat pulls above all came back empty. Nothing to score."
        )

    candidates = build_candidate_legs(odds_df, registry)
    if not candidates:
        raise ValueError(
            "build_candidate_legs() produced zero candidates from a "
            "non-empty registry and a non-empty odds_df — check that "
            "outcome_name values in the odds_df match what market_map.py "
            "expects ('Over'/'Under', 'Yes'/'No')."
        )

    return rank_edges(candidates)


def load_raw_stats() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (batter_stats_df, pitcher_stats_df) for the raw-stats showcase
    view. Tries the live pipeline first if ODDS_API_KEY is present (the
    key gates the WHOLE live path, not just the odds call, since there's
    no point pulling live stats with no odds to score against — see
    module docstring); falls back to sample_data.py on any failure,
    including "no key at all."
    """
    global IS_LIVE_DATA

    from config import odds_api_key_present

    if not odds_api_key_present():
        IS_LIVE_DATA = False
        return sample_data.build_sample_raw_stats_df(), sample_data.build_sample_pitcher_stats_df()

    try:
        batter_df, pitcher_df = _load_raw_stats_live()
        IS_LIVE_DATA = True
        return batter_df, pitcher_df
    except Exception as exc:  # noqa: BLE001
        _log_live_path_failure("load_raw_stats", exc)
        IS_LIVE_DATA = False
        return sample_data.build_sample_raw_stats_df(), sample_data.build_sample_pitcher_stats_df()


def load_ranked_edges() -> list:
    """
    Returns the full edge-ranked list of `EdgeCandidate` objects (real
    dataclass from edge_ranking/ranking.py) across all 7 prop types,
    sorted by edge descending. Tries the live pipeline first if
    ODDS_API_KEY is present; falls back to sample_data.py on any failure
    in the live path (missing key, pipeline column mismatch, network
    error, empty slate, anything) — see module docstring for exactly why
    this must never let an exception reach app.py.
    """
    global IS_LIVE_DATA

    from config import odds_api_key_present

    if not odds_api_key_present():
        IS_LIVE_DATA = False
        return sample_data.build_sample_edge_candidates()

    try:
        ranked = _load_ranked_edges_live()
        IS_LIVE_DATA = True
        return ranked
    except Exception as exc:  # noqa: BLE001
        _log_live_path_failure("load_ranked_edges", exc)
        IS_LIVE_DATA = False
        return sample_data.build_sample_edge_candidates()
