"""
Statcast batter/pitcher metric pulls via pybaseball (Baseball Savant + FanGraphs).

pybaseball wraps two different underlying sources depending on the function:
  - batting_stats() / pitching_stats()  -> FanGraphs season leaderboards
  - statcast_batter_expected_stats() etc -> Baseball Savant's "expected stats" pages
  - statcast() / statcast_batter() / statcast_pitcher() -> raw Savant pitch-by-pitch
    Statcast search CSVs

No API key/auth required for any of this — it's all public data pybaseball
scrapes from public pages. Network calls happen lazily inside each function
so importing this module never hits the network on its own.

Metrics covered here (per PropSync Scoring Framework Notes):

Batters:
  - barrel %, hard-hit %                  -> FanGraphs batting_stats() (Statcast cols)
  - fly ball %, groundball %, line drive % -> FanGraphs batting_stats() (batted-ball cols)
  - pulled fly ball % / pulled-air rate    -> NOT exposed by pybaseball's wrappers.
                                              Falls back to Savant's direct CSV search
                                              endpoint with hfPull pull-side filters
                                              applied to batted-ball-event-level data.
  - swinging-strike %                      -> FanGraphs batting_stats() ('SwStr%')
  - xSLG, xBA, ISO                         -> FanGraphs batting_stats() / Savant expected stats
  - sprint speed                           -> Savant's Sprint Speed leaderboard
                                              (pybaseball.statcast_sprint_speed)

Pitchers:
  - HR/9, K/9                              -> FanGraphs pitching_stats()
  - barrel % allowed                       -> FanGraphs pitching_stats() (Statcast cols)
  - swinging-strike % / whiff rate         -> FanGraphs pitching_stats() ('SwStr%')
  - CSW%                                   -> NOT a native FanGraphs/pybaseball column.
                                              Derived here from called-strike + whiff
                                              counts in the raw Statcast pitch-by-pitch
                                              pull (statcast_pitcher), since CSW% isn't
                                              published as its own leaderboard stat.
  - fastball-in-zone rate                  -> Derived from raw Statcast pitch-by-pitch
                                              pull, filtered to fastball pitch types and
                                              the strike zone flag.
  - platoon K splits                       -> FanGraphs pitching_stats() split by
                                              opponent handedness (vs L / vs R queries).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, SEASON  # noqa: E402

# Pitch types Statcast tags as some flavor of fastball, for the
# fastball-in-zone-rate derivation.
FASTBALL_PITCH_TYPES = {"FF", "FT", "FC", "SI"}


def _lazy_import_pybaseball():
    """Import pybaseball only when a function actually needs the network.

    Keeps `import savant` cheap and lets the rest of this module be
    introspected/tested even in environments where pybaseball isn't
    installed yet.
    """
    try:
        import pybaseball
    except ImportError as exc:
        raise ImportError(
            "pybaseball is not installed. Run `pip install -r requirements.txt` "
            "inside the data_pipeline folder first."
        ) from exc
    return pybaseball


def _fg_api_fetch(stats: str, season: int, qual: int) -> pd.DataFrame:
    """
    Pull a FanGraphs major-league leaderboard directly from their JSON API.

    pybaseball's batting_stats/pitching_stats wrappers hit the legacy
    leaders-legacy.aspx page which now returns 403. This bypasses pybaseball
    and calls the live /api/leaders/major-league/data endpoint instead.

    stats: 'bat' or 'pit'
    """
    import requests

    url = (
        "https://www.fangraphs.com/api/leaders/major-league/data"
        f"?pos=all&stats={stats}&lg=all&qual={qual}"
        f"&season={season}&season1={season}"
        "&startdate=&enddate=&month=0&hand=&team=0"
        "&pageitems=5000&pagenum=1&ind=0&rost=0&players="
        "&type=8&postseason=&sortdir=default&sortstat=WAR"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        raise ValueError(f"FanGraphs API returned no rows for stats={stats} season={season}")
    df = pd.DataFrame(data)
    # API returns Name and Team as HTML anchor tags — strip to plain text
    import re
    for col in ("Name", "Team", "PlayerName", "TeamName", "TeamNameAbb"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: re.sub(r"<[^>]+>", "", str(v)).strip() if pd.notna(v) else v
            )
    if "xAVG" in df.columns and "xBA" not in df.columns:
        df = df.rename(columns={"xAVG": "xBA"})
    return df


def get_batter_season_stats(season: int = SEASON, qual: int = 1) -> pd.DataFrame:
    """
    Pull FanGraphs season batting leaderboard for all qualified batters.

    Includes barrel%, hard-hit%, fly ball%, groundball%, line drive%,
    swinging-strike%, xSLG, xBA, ISO. qual=1 keeps low-sample call-ups
    so the scoring model can apply its own shrinkage.
    """
    return _fg_api_fetch("bat", season, qual)


def get_pitcher_season_stats(season: int = SEASON, qual: int = 1) -> pd.DataFrame:
    """
    Pull FanGraphs season pitching leaderboard for all pitchers.

    Includes K/9, HR/9, barrel% allowed, swinging-strike%/whiff rate.
    """
    return _fg_api_fetch("pit", season, qual)


def get_batter_expected_stats(season: int = SEASON) -> pd.DataFrame:
    """
    Pull Baseball Savant's batter "expected stats" leaderboard:
    xBA, xSLG, xwOBA, barrel%, hard-hit% — Savant's own version of these
    (slightly different methodology/coverage than the FanGraphs leaderboard
    pull above; useful as a cross-check).
    """
    pyb = _lazy_import_pybaseball()
    return pyb.statcast_batter_expected_stats(season)


def get_sprint_speed(season: int = SEASON, min_opp: int = 5) -> pd.DataFrame:
    """
    Pull Baseball Savant's Sprint Speed leaderboard.
    Needed for singles props (infield hits) and doubles props (legging out
    extra bases) per the Scoring Framework Notes.
    """
    pyb = _lazy_import_pybaseball()
    return pyb.statcast_sprint_speed(season, min_opp=min_opp)


def get_raw_statcast(start_dt: str, end_dt: str) -> pd.DataFrame:
    """
    Pull raw pitch-by-pitch Statcast data for a date range via pybaseball's
    statcast(). This is the underlying data needed for metrics that have no
    pre-built leaderboard: pulled-air rate, CSW%, fastball-in-zone rate.

    Large date ranges are slow (this hits Savant's search endpoint under
    the hood) — callers doing rolling-window pulls should keep windows to
    15/30 days, not full seasons, unless caching results.
    """
    pyb = _lazy_import_pybaseball()
    return pyb.statcast(start_dt=start_dt, end_dt=end_dt)


def compute_pulled_air_rate(raw_statcast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive pulled fly ball % / pulled-air rate per batter from raw Statcast
    batted-ball events.

    pybaseball has no direct wrapper for this metric (Baseball Savant only
    exposes pull-side splits through its UI's hfPull filter, not as a
    leaderboard pybaseball mirrors), so it's computed here from raw
    pitch-by-pitch data:
      1. Filter to batted ball events that are fly balls or popups
         (bb_type in ['fly_ball', 'popup'])
      2. Determine pull side from spray angle / hit location relative to
         batter handedness (Statcast's `bb_type` + `hc_x`/`hc_y` columns,
         or the simpler `pull_side` logic below using hit coordinates)
      3. pulled_air_rate = pulled fly balls / all fly balls (per batter)

    FALLBACK NOTE: if pybaseball's statcast() schema ever drops hc_x/hc_y
    or bb_type, the direct CSV fallback in statcast/savant_csv_fallback.py
    pulls the same raw fields straight from
    baseballsavant.mlb.com/statcast_search/csv with explicit field
    selection, bypassing pybaseball entirely.
    """
    required_cols = {"batter", "stand", "bb_type", "hc_x", "hc_y"}
    missing = required_cols - set(raw_statcast_df.columns)
    if missing:
        raise ValueError(
            f"Raw statcast data is missing columns needed for pulled-air-rate: "
            f"{missing}. Use the CSV fallback (savant_csv_fallback.py) instead."
        )

    df = raw_statcast_df.copy()
    air_balls = df[df["bb_type"].isin(["fly_ball", "popup"])].copy()

    # Spray angle from hit coordinates (Savant's hc_x/hc_y are plotted with
    # home plate near (125, 204) on a 250x250 grid). Pull side = third-base
    # side for RHB, first-base side for LHB.
    air_balls["spray_angle"] = (
        (air_balls["hc_x"] - 125.42).clip(lower=-300, upper=300)
    )
    air_balls["is_pulled"] = (
        ((air_balls["stand"] == "R") & (air_balls["spray_angle"] < -5))
        | ((air_balls["stand"] == "L") & (air_balls["spray_angle"] > 5))
    )

    grouped = air_balls.groupby("batter").agg(
        air_balls_total=("is_pulled", "size"),
        air_balls_pulled=("is_pulled", "sum"),
    )
    grouped["pulled_air_rate"] = (
        grouped["air_balls_pulled"] / grouped["air_balls_total"]
    )
    return grouped.reset_index()


def compute_csw_rate(raw_statcast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive CSW% (Called Strikes + Whiffs, as a % of total pitches) per
    pitcher from raw Statcast pitch-by-pitch data.

    Not a native FanGraphs/pybaseball leaderboard column, so computed here
    from `description` values: called_strike + swinging_strike (+
    swinging_strike_blocked) divided by total pitches thrown.
    """
    required_cols = {"pitcher", "description"}
    missing = required_cols - set(raw_statcast_df.columns)
    if missing:
        raise ValueError(f"Raw statcast data missing columns for CSW%: {missing}")

    df = raw_statcast_df.copy()
    csw_descriptions = {"called_strike", "swinging_strike", "swinging_strike_blocked"}
    df["is_csw"] = df["description"].isin(csw_descriptions)

    grouped = df.groupby("pitcher").agg(
        pitches_total=("is_csw", "size"),
        csw_count=("is_csw", "sum"),
    )
    grouped["csw_pct"] = grouped["csw_count"] / grouped["pitches_total"]
    return grouped.reset_index()


def compute_fastball_zone_rate(raw_statcast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive fastball-in-zone rate per pitcher: of all fastballs thrown
    (FF/FT/FC/SI), what % were located in the strike zone (zone 1-9, not
    the out-of-zone 11-14 codes Statcast uses).
    """
    required_cols = {"pitcher", "pitch_type", "zone"}
    missing = required_cols - set(raw_statcast_df.columns)
    if missing:
        raise ValueError(
            f"Raw statcast data missing columns for fastball-in-zone rate: {missing}"
        )

    df = raw_statcast_df[raw_statcast_df["pitch_type"].isin(FASTBALL_PITCH_TYPES)].copy()
    df["in_zone"] = df["zone"].between(1, 9)

    grouped = df.groupby("pitcher").agg(
        fastballs_total=("in_zone", "size"),
        fastballs_in_zone=("in_zone", "sum"),
    )
    grouped["fastball_zone_rate"] = (
        grouped["fastballs_in_zone"] / grouped["fastballs_total"]
    )
    return grouped.reset_index()


def get_platoon_splits(player_bbref_or_fg_id: str, pitcher: bool = True) -> pd.DataFrame:
    """
    Pull handedness splits (vs L / vs R) for a single pitcher or batter via
    pybaseball's FanGraphs splits leaderboard wrapper.

    NOTE: pybaseball's splits support is the least standardized part of its
    API and has changed across versions. If `pyb.splits_leaderboards` /
    equivalent isn't available in the installed pybaseball version, fall
    back to pulling raw Statcast (get_raw_statcast) and grouping by the
    opposing batter's `stand` (for pitcher platoon splits) or pitcher's
    `p_throws` (for batter platoon splits) instead.
    """
    pyb = _lazy_import_pybaseball()
    if not hasattr(pyb, "split_leaderboard") and not hasattr(pyb, "splits"):
        raise AttributeError(
            "This pybaseball version doesn't expose a splits leaderboard "
            "function. Use get_raw_statcast() + group by 'stand' (pitcher "
            "splits) or 'p_throws' (batter splits) as a manual fallback."
        )
    # pybaseball's splits API surface varies by version; left intentionally
    # thin here. Wire up the exact call once a pybaseball version is pinned
    # and its splits function signature is confirmed in a real environment.
    raise NotImplementedError(
        "Platoon splits via pybaseball's leaderboard wrapper need the exact "
        "function signature confirmed against the installed pybaseball "
        "version (this varies by release). Manual fallback: pull raw "
        "Statcast with get_raw_statcast() and groupby('stand') or "
        "groupby('p_throws')."
    )


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    """Save a pulled DataFrame to data/raw/ for inspection."""
    out_path = DATA_DIR / filename
    df.to_csv(out_path, index=False)
    return out_path
