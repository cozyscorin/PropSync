"""
Recent-form / rolling-window stat pulls.

The Scoring Framework Notes call for recency-weighted stats (HR props,
hits props, pitcher K props all reference "recency-weighted form") rather
than season-long aggregates alone — a hitter's last 15-30 days matters
more than what he did in April.

pybaseball's batting_stats()/pitching_stats() only return season
aggregates, so rolling windows are built here by pulling raw Statcast data
for a trailing N-day window and re-aggregating the relevant rate stats
ourselves.

This module does NOT decide how to weight recent vs. season-long form —
that's scoring-model logic, explicitly out of scope for this data
pipeline (see task scope: data sourcing only, no scoring/ranking). It just
makes the windowed raw numbers available.
"""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, ROLLING_WINDOWS_DAYS  # noqa: E402
from statcast.savant import (  # noqa: E402
    compute_csw_rate,
    compute_fastball_zone_rate,
    get_raw_statcast,
)


def window_date_range(window_days: int, as_of: _dt.date | None = None) -> tuple[str, str]:
    """Return (start_dt, end_dt) strings for a trailing N-day window ending today."""
    as_of = as_of or _dt.date.today()
    start = as_of - _dt.timedelta(days=window_days)
    return start.isoformat(), as_of.isoformat()


def batter_rolling_batted_ball_profile(window_days: int) -> pd.DataFrame:
    """
    Trailing N-day batted-ball profile per batter, computed from raw
    Statcast pitch-by-pitch data: barrel rate, hard-hit rate, fly ball %,
    groundball %, line drive % within the window.

    This re-derives from raw events rather than using FanGraphs'
    season-leaderboard columns, since FanGraphs has no rolling-window
    query — only full-season or pre-set split buckets.
    """
    start_dt, end_dt = window_date_range(window_days)
    raw = get_raw_statcast(start_dt, end_dt)

    if raw.empty:
        return pd.DataFrame()

    batted = raw[raw["type"] == "X"].copy()  # X = ball in play
    batted["is_barrel"] = batted.get("barrel", pd.Series(dtype=float)).fillna(0) == 1
    batted["is_hard_hit"] = batted["launch_speed"].fillna(0) >= 95

    grouped = batted.groupby("batter").agg(
        batted_balls=("is_barrel", "size"),
        barrels=("is_barrel", "sum"),
        hard_hits=("is_hard_hit", "sum"),
    )
    grouped["barrel_pct"] = grouped["barrels"] / grouped["batted_balls"]
    grouped["hard_hit_pct"] = grouped["hard_hits"] / grouped["batted_balls"]

    bb_type_counts = (
        batted.groupby(["batter", "bb_type"]).size().unstack(fill_value=0)
    )
    bb_type_pct = bb_type_counts.div(bb_type_counts.sum(axis=1), axis=0)
    bb_type_pct = bb_type_pct.rename(
        columns={
            "fly_ball": "fly_ball_pct",
            "ground_ball": "groundball_pct",
            "line_drive": "line_drive_pct",
            "popup": "popup_pct",
        }
    )

    result = grouped.join(bb_type_pct, how="left").reset_index()
    result["window_days"] = window_days
    result["start_dt"] = start_dt
    result["end_dt"] = end_dt
    return result


def pitcher_rolling_profile(window_days: int) -> pd.DataFrame:
    """
    Trailing N-day pitcher profile: CSW%, swinging-strike rate, fastball
    zone rate, barrel% allowed, derived from raw Statcast data the same
    way as the batter rolling profile.
    """
    start_dt, end_dt = window_date_range(window_days)
    raw = get_raw_statcast(start_dt, end_dt)

    if raw.empty:
        return pd.DataFrame()

    csw = compute_csw_rate(raw)
    zone = compute_fastball_zone_rate(raw)

    batted_against = raw[raw["type"] == "X"].copy()
    batted_against["is_barrel"] = (
        batted_against.get("barrel", pd.Series(dtype=float)).fillna(0) == 1
    )
    barrels_allowed = batted_against.groupby("pitcher").agg(
        batted_balls_allowed=("is_barrel", "size"),
        barrels_allowed=("is_barrel", "sum"),
    )
    barrels_allowed["barrel_pct_allowed"] = (
        barrels_allowed["barrels_allowed"] / barrels_allowed["batted_balls_allowed"]
    )

    result = csw.merge(zone, on="pitcher", how="outer").merge(
        barrels_allowed, on="pitcher", how="outer"
    )
    result["window_days"] = window_days
    result["start_dt"] = start_dt
    result["end_dt"] = end_dt
    return result


def pull_all_rolling_windows(windows: list[int] | None = None) -> dict[str, dict[int, pd.DataFrame]]:
    """
    Pull both batter and pitcher rolling profiles for every configured
    window (default: 15 and 30 days, per config.ROLLING_WINDOWS_DAYS).

    Returns {"batters": {15: df, 30: df}, "pitchers": {15: df, 30: df}}.
    """
    windows = windows or ROLLING_WINDOWS_DAYS
    out: dict[str, dict[int, pd.DataFrame]] = {"batters": {}, "pitchers": {}}
    for w in windows:
        out["batters"][w] = batter_rolling_batted_ball_profile(w)
        out["pitchers"][w] = pitcher_rolling_profile(w)
    return out


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    out_path = DATA_DIR / filename
    df.to_csv(out_path, index=False)
    return out_path
