"""
Pure data-transformation helpers for the dashboard's two views. No
Streamlit imports here on purpose — everything in this module is plain
pandas/stdlib so it can be unit-tested without a Streamlit runtime (see
tests/test_formatting.py).

`app.py` calls these to turn raw objects (a list of `EdgeCandidate`, the
raw-stats DataFrames from data_loader) into the exact DataFrames/values
the UI renders.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _folder in ("[C] scoring_model", "[C] edge_ranking"):
    _p = _PROJECT_ROOT / _folder
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from market_map import get_prop_spec  # noqa: E402

# ---------------------------------------------------------------------------
# Picks / edge view
# ---------------------------------------------------------------------------

PROP_TYPE_DISPLAY_ORDER = [
    "batter_home_runs",
    "batter_hits",
    "batter_total_bases",
    "batter_rbis",
    "batter_singles",
    "batter_doubles",
    "pitcher_strikeouts",
]


def prop_display_name(market_key: str) -> str:
    """Human-readable label for a market key, e.g. 'batter_hits' -> 'Hits'."""
    try:
        return get_prop_spec(market_key).display_name
    except KeyError:
        return market_key


def _format_side(candidate) -> str:
    """'Over 1.5' / 'Under 1.5' / 'Yes (1+ HR)' style label for the side
    PropSync's model actually scores (always the modeled side — Over/Yes
    — per market_map.py; Under/No legs are never scored, see edge_ranking
    README)."""
    if candidate.line is None:
        return "Yes (1+ HR)"
    return f"Over {candidate.line:g}"


def edge_candidates_to_dataframe(candidates: list) -> pd.DataFrame:
    """
    Flatten a list of `EdgeCandidate` (edge_ranking/ranking.py) into one
    display-ready DataFrame for the picks table. Column choices match
    what the task spec calls for: player, team, prop type, line, side,
    sportsbook used, model probability, book's de-vigged market
    probability, edge, which book had the better price.

    Kept as a free function (not a method on EdgeCandidate, which is
    frozen/owned by edge_ranking) so this dashboard's display concerns
    never need to touch that module.
    """
    if not candidates:
        return pd.DataFrame(columns=[
            "player", "team", "opponent", "prop_type", "market", "line", "side",
            "sportsbook", "model_prob", "market_prob", "edge",
            "model_prob_pct", "market_prob_pct", "edge_pct",
            "other_book", "other_book_prob",
        ])

    rows = []
    for c in candidates:
        other_books = [bp for bp in c.all_book_prices if bp.bookmaker != c.chosen_book]
        other_book_label = ", ".join(
            f"{bp.bookmaker} ({bp.fair_prob:.1%})" for bp in other_books
        ) or "—"
        rows.append({
            "player": c.player,
            "team": c.home_team,  # see note below; refined by caller if needed
            "opponent": c.away_team,
            "prop_type": prop_display_name(c.market),
            "market": c.market,
            "line": "—" if c.line is None else c.line,
            "side": _format_side(c),
            "sportsbook": c.chosen_book,
            "chosen_book_price": c.chosen_book_price,
            "model_prob": c.model_prob,
            "market_prob": c.chosen_market_prob,
            "edge": c.edge,
            "model_prob_pct": f"{c.model_prob:.1%}",
            "market_prob_pct": f"{c.chosen_market_prob:.1%}",
            "edge_pct": f"{c.edge:+.1%}",
            "other_book": other_book_label,
            "game_key": c.game_key,
        })
    df = pd.DataFrame(rows)
    return df.sort_values("edge", ascending=False).reset_index(drop=True)


def filter_picks(
    df: pd.DataFrame,
    prop_types: list[str] | None = None,
    min_edge: float | None = None,
    players: list[str] | None = None,
    sportsbooks: list[str] | None = None,
) -> pd.DataFrame:
    """Apply the picks-view sidebar filters. Each filter is a no-op when
    None/empty so callers don't need to special-case "no filter selected"."""
    out = df
    if prop_types:
        out = out[out["market"].isin(prop_types)]
    if min_edge is not None:
        out = out[out["edge"] >= min_edge]
    if players:
        out = out[out["player"].isin(players)]
    if sportsbooks:
        out = out[out["sportsbook"].isin(sportsbooks)]
    return out.reset_index(drop=True)


def top_n_parlay_view(ranked_candidates: list, n: int) -> pd.DataFrame:
    """
    Build the top-N parlay table using the REAL `select_top_n()` from
    edge_ranking/parlay_selector.py (plain slice, no exclusion — same
    player/game can repeat, per the Scoring Framework Notes' deliberate
    decision). Returns a display-ready DataFrame, same columns as
    edge_candidates_to_dataframe().
    """
    from parlay_selector import select_top_n

    top = select_top_n(ranked_candidates, n)
    return edge_candidates_to_dataframe(top)


# ---------------------------------------------------------------------------
# Raw stats showcase view
# ---------------------------------------------------------------------------

BATTER_SHOWCASE_COLUMNS = {
    "name": "Player",
    "team": "Team",
    "opp": "Opp",
    "park": "Park",
    "barrel_pct": "Barrel %",
    "hard_hit_pct": "Hard-Hit %",
    "fly_ball_pct": "Fly Ball %",
    "groundball_pct": "Groundball %",
    "line_drive_pct": "Line Drive %",
    "swinging_strike_pct": "SwStr %",
    "pulled_air_rate": "Pulled-Air Rate",
    "xba": "xBA",
    "xslg": "xSLG",
    "iso": "ISO",
    "sprint_speed_ft_s": "Sprint Speed (ft/s)",
    "wOBA_vs_L": "wOBA vs LHP",
    "wOBA_vs_R": "wOBA vs RHP",
    "park_factor_1b": "PF: 1B",
    "park_factor_2b": "PF: 2B",
    "park_factor_3b": "PF: 3B",
    "park_factor_hr": "PF: HR",
    "pa_season": "PA (season)",
    "pa_recent_15d": "PA (last 15d)",
}

PITCHER_SHOWCASE_COLUMNS = {
    "name": "Pitcher",
    "team": "Team",
    "opp": "Opp",
    "park": "Park",
    "k_per_9": "K/9",
    "k_per_9_recent": "K/9 (last 15d)",
    "csw_pct": "CSW %",
    "hr_per_9": "HR/9",
    "whip": "WHIP",
    "hits_per_9": "H/9",
    "groundball_pct_allowed": "GB % Allowed",
    "barrel_pct_allowed": "Barrel % Allowed",
    "expected_innings": "Exp. Innings",
    "k_pct_vs_L": "K% vs LHB",
    "k_pct_vs_R": "K% vs RHB",
}

PCT_COLUMNS = {
    "Barrel %", "Hard-Hit %", "Fly Ball %", "Groundball %", "Line Drive %",
    "SwStr %", "Pulled-Air Rate", "xBA", "GB % Allowed", "Barrel % Allowed",
    "K% vs LHB", "K% vs RHB", "CSW %",
}


def batter_showcase_table(raw_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Rename/select columns for the batter raw-stats showcase table,
    in the order the Kasper notes called out: barrel%, hard-hit%, fly
    ball%, swinging-strike%, handedness splits, park factors."""
    cols = [c for c in BATTER_SHOWCASE_COLUMNS if c in raw_stats_df.columns]
    out = raw_stats_df[cols].rename(columns=BATTER_SHOWCASE_COLUMNS)
    return _format_pct_columns(out)


def pitcher_showcase_table(pitcher_stats_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in PITCHER_SHOWCASE_COLUMNS if c in pitcher_stats_df.columns]
    out = pitcher_stats_df[cols].rename(columns=PITCHER_SHOWCASE_COLUMNS)
    return _format_pct_columns(out)


def _format_pct_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Render the known rate-stat columns as 'NN.N%' strings for display.
    Kept separate from the raw numeric DataFrame so callers needing the
    numeric value (e.g. for sorting/charting) can use the un-formatted
    columns before calling this."""
    out = df.copy()
    for col in out.columns:
        if col in PCT_COLUMNS and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].apply(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    return out


def park_factor_table(raw_stats_df: pd.DataFrame) -> pd.DataFrame:
    """One row per park, split by hit type (1B/2B/3B/HR) — not blended —
    per the explicit Scoring Framework Notes requirement. Dedupes the
    per-player raw_stats_df down to one row per team/park."""
    cols = ["team", "park", "park_factor_1b", "park_factor_2b", "park_factor_3b", "park_factor_hr"]
    out = raw_stats_df[cols].drop_duplicates(subset=["team"]).reset_index(drop=True)
    return out.rename(columns={
        "team": "Team", "park": "Park",
        "park_factor_1b": "1B Park Factor", "park_factor_2b": "2B Park Factor",
        "park_factor_3b": "3B Park Factor", "park_factor_hr": "HR Park Factor",
    })
