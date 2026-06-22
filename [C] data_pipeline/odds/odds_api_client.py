"""
The Odds API client — FanDuel and DraftKings player prop odds.

STATUS: STUBBED, NOT YET TESTED LIVE. This requires a free API key from
https://the-odds-api.com/ that cozy has not obtained yet. This module is
fully built and structurally ready to go — the moment a real key is
dropped into .env (see .env.example), every function here should work
without code changes. Do not sign up for a key on cozy's behalf; that's
explicitly his action to take.

Per the Scoring Framework Notes, player prop odds are NOT available
through the bulk /odds endpoint that handles markets like
moneyline/totals. Player props require:
  1. GET /v4/sports/{sport}/events  -> list today's games, get event IDs
  2. GET /v4/sports/{sport}/events/{eventId}/odds?markets=...  -> one game
     at a time, scoped to that event's player prop markets

Market keys needed (confirmed against The Odds API's MLB player props
docs, see Scoring Framework Notes for the source decision):
  batter_home_runs, batter_hits, batter_total_bases, batter_rbis,
  batter_singles, batter_doubles, pitcher_strikeouts

Credit cost: each per-event call with player prop markets costs more
credits than a standard bulk-market call (exact multiplier not confirmed
in the notes — check https://the-odds-api.com/liveapi/guides/v4/#usage-quota-costs
once a key exists, before doing high-volume pulls against the 500
credits/month free tier).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    DATA_DIR,
    ODDS_API_BASE_URL,
    ODDS_API_KEY,
    ODDS_API_SPORT_KEY,
    PLAYER_PROP_MARKETS,
    TARGET_BOOKMAKERS,
    odds_api_key_present,
)


class OddsAPIKeyMissingError(RuntimeError):
    """Raised when an Odds API call is attempted with no API key configured."""


def _require_key() -> str:
    if not odds_api_key_present():
        raise OddsAPIKeyMissingError(
            "ODDS_API_KEY is not set. Sign up for a free key at "
            "https://the-odds-api.com/ and put it in a .env file in this "
            "folder (copy .env.example -> .env and fill it in), or export "
            "it as an environment variable. This is the user's action to "
            "take, not something this code does automatically."
        )
    return ODDS_API_KEY


def get_todays_events(sport_key: str = ODDS_API_SPORT_KEY) -> list[dict]:
    """
    GET /v4/sports/{sport}/events

    Returns the list of today's (and near-term upcoming) MLB games with
    their event IDs, needed before any per-event player prop call can be
    made. Each item looks like:
        {"id": "...", "sport_key": "baseball_mlb", "commence_time": "...",
         "home_team": "...", "away_team": "..."}
    """
    key = _require_key()
    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/events"
    resp = requests.get(url, params={"apiKey": key}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_event_player_props(
    event_id: str,
    markets: list[str] | None = None,
    bookmakers: str = ",".join(TARGET_BOOKMAKERS),
    sport_key: str = ODDS_API_SPORT_KEY,
    regions: str = "us",
    odds_format: str = "american",
) -> dict:
    """
    GET /v4/sports/{sport}/events/{eventId}/odds

    Per-event player prop odds — the endpoint player props actually live
    behind (NOT the bulk /odds endpoint, which only covers standard
    markets like h2h/spreads/totals).

    Args:
        event_id: from get_todays_events()
        markets: defaults to all 7 PropSync market keys from config.py
        bookmakers: comma-separated bookmaker keys, defaults to both
            FanDuel and DraftKings (config.TARGET_BOOKMAKERS). Both are
            pulled as cross-references — if one book hasn't posted a
            given line yet, the other can still be used.
        regions: 'us' for FanDuel/DraftKings/other US books
        odds_format: 'american' (-110 style) vs 'decimal'

    Returns the raw JSON response: bookmakers -> markets -> outcomes,
    each outcome having a player name, line, and price.
    """
    key = _require_key()
    markets = markets or PLAYER_PROP_MARKETS
    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": key,
        "regions": regions,
        "markets": ",".join(markets),
        "bookmakers": bookmakers,
        "oddsFormat": odds_format,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_all_player_props_today(
    markets: list[str] | None = None,
    bookmakers: str = ",".join(TARGET_BOOKMAKERS),
) -> pd.DataFrame:
    """
    Convenience orchestrator: fetch today's events, then loop the
    per-event odds call for each one, and flatten everything into one
    tidy DataFrame: one row per (event, market, player, line, price,
    bookmaker). Both FanDuel and DraftKings rows are included side by
    side (the 'bookmaker' column distinguishes them) so downstream code
    can use whichever book has a given line posted, or compare both.

    This is the function the scoring model will eventually call — it's
    the full path from "no data" to "a clean table of player prop lines
    for today's games." Cannot be tested without a real API key; every
    piece is wired up and ready.
    """
    events = get_todays_events()
    rows = []
    for event in events:
        event_id = event["id"]
        try:
            data = get_event_player_props(event_id, markets=markets, bookmakers=bookmakers)
        except requests.HTTPError as exc:
            # Don't let one bad event (e.g. a postponed game with no
            # props posted yet) kill the whole pull.
            print(f"Skipping event {event_id} ({event.get('home_team')} vs "
                  f"{event.get('away_team')}): {exc}")
            continue

        for bm in data.get("bookmakers", []):
            for market in bm.get("markets", []):
                for outcome in market.get("outcomes", []):
                    rows.append(
                        {
                            "event_id": event_id,
                            "commence_time": event.get("commence_time"),
                            "home_team": event.get("home_team"),
                            "away_team": event.get("away_team"),
                            "bookmaker": bm.get("key"),
                            "market": market.get("key"),
                            "player": outcome.get("description") or outcome.get("name"),
                            "outcome_name": outcome.get("name"),
                            "line": outcome.get("point"),
                            "price": outcome.get("price"),
                        }
                    )

    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    out_path = DATA_DIR / filename
    df.to_csv(out_path, index=False)
    return out_path
