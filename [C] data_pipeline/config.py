"""
Shared configuration for the PropSync data pipeline.

Loads environment variables from a .env file (if present) so secrets like
ODDS_API_KEY never need to be hardcoded or committed to git.
"""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    # Load .env from this folder regardless of where the script is run from.
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    # python-dotenv isn't installed yet — fall back to whatever is already
    # in the real environment. Doesn't block the Statcast/FanGraphs side,
    # which needs no auth at all.
    pass

# --- Odds API -----------------------------------------------------------
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "").strip()
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# The sportsbooks PropSync ranks against (see Scoring Framework Notes.md).
# Both are pulled and used as cross-references: if a line isn't posted yet
# at one book, the other can still be used. Order here is not a preference.
TARGET_BOOKMAKERS = ["fanduel", "draftkings"]

# MLB sport key used by The Odds API.
ODDS_API_SPORT_KEY = "baseball_mlb"

# Player prop market keys PropSync needs, per the Scoring Framework Notes.
PLAYER_PROP_MARKETS = [
    "batter_home_runs",
    "batter_hits",
    "batter_total_bases",
    "batter_rbis",
    "batter_singles",
    "batter_doubles",
    "pitcher_strikeouts",
]

# --- Statcast / Baseball Savant -----------------------------------------
# pybaseball needs explicit YYYY-MM-DD date ranges. Default the "current
# season" pulls to opening day onward unless the caller passes their own
# date range.
def current_mlb_season() -> int:
    """Best-effort current MLB season year (handles off-season rollover)."""
    today = _dt.date.today()
    # MLB season runs roughly late March through early November. In the
    # Dec-Feb offseason, "current season" for stat-pull purposes means the
    # most recently completed season.
    if today.month <= 2:
        return today.year - 1
    return today.year


SEASON = current_mlb_season()
SEASON_START = f"{SEASON}-03-01"
# Use today's date as the rolling end date so pulls always grab everything
# available so far; pybaseball clips automatically if the season hasn't
# reached that date yet.
SEASON_END = _dt.date.today().isoformat()

# Recency windows the scoring model will want (see Scoring Framework Notes —
# "recency-weighted form" for HR/hits/strikeouts props).
ROLLING_WINDOWS_DAYS = [15, 30]

# Where raw pulls get cached/inspected from.
DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def odds_api_key_present() -> bool:
    return bool(ODDS_API_KEY) and ODDS_API_KEY != "your_api_key_here"
