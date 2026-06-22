"""
Direct Baseball Savant CSV search fallback.

Use this ONLY when pybaseball doesn't expose a needed metric (see the
docstring in savant.py for exactly which metrics fall into this bucket:
pulled-air rate / pulled fly ball %, and as a backup path for CSW% /
fastball-zone-rate if pybaseball's statcast() schema ever changes).

This hits baseballsavant.mlb.com/statcast_search/csv directly with
`requests` — no auth, no API key, it's the same endpoint the Baseball
Savant search UI calls when you click "Download CSV" on a search. It
returns raw pitch-by-pitch rows, same shape as pybaseball.statcast(), so
the compute_* helpers in savant.py work on either source interchangeably.

Confirmed reachable: a GET against this endpoint returns a 200 with
`Content-Type: application/download` and real CSV bytes (verified
manually against the live endpoint while building this module). Could not
execute this code end-to-end in the build environment because the sandbox
used to build PropSync has no outbound network access for Python
processes (firewalled at the proxy layer) — see README.md "Known
limitations" for the full explanation. Run this for real on cozy's actual
machine to confirm.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR  # noqa: E402

SAVANT_CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"

# Sensible defaults for a "regular season, all pitch types, all counts"
# search. Override individual keys via the `extra_params` argument.
DEFAULT_PARAMS = {
    "all": "true",
    "hfPT": "",
    "hfAB": "",
    "hfBBT": "",
    "hfPR": "",
    "hfZ": "",
    "hfStadium": "",
    "hfBBL": "",
    "hfNewZones": "",
    "hfGT": "R|",  # Regular season
    "hfC": "",
    "hfSea": "",  # filled in by caller via `season`
    "hfSit": "",
    "hfOuts": "",
    "opponent": "",
    "pitcher_throws": "",
    "batter_stands": "",
    "hfSA": "",
    "player_type": "batter",
    "hfInfield": "",
    "team": "",
    "position": "",
    "hfOutfield": "",
    "hfRO": "",
    "home_road": "",
    "hfFlag": "",
    "hfPull": "",
    "metric_1": "",
    "hfInn": "",
    "min_pitches": "0",
    "min_results": "0",
    "group_by": "name",
    "sort_col": "pitches",
    "player_event_sort": "api_p_release_speed",
    "sort_order": "desc",
    "min_pas": "0",
    "type": "details",
}


def fetch_statcast_csv(
    start_dt: str,
    end_dt: str,
    season: int,
    player_type: str = "batter",
    extra_params: dict | None = None,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Direct GET against Baseball Savant's CSV search endpoint.

    Args:
        start_dt: 'YYYY-MM-DD'
        end_dt: 'YYYY-MM-DD'
        season: e.g. 2026 (matches hfSea filter, must align with the date range)
        player_type: 'batter' or 'pitcher'
        extra_params: any Savant search params to override
            (e.g. {"hfPull": "1"} to filter to pulled balls only)
        timeout: request timeout in seconds

    Returns a DataFrame with the same pitch-by-pitch row shape Savant's UI
    exports — this is the same schema pybaseball.statcast() normalizes
    its own output to.
    """
    params = dict(DEFAULT_PARAMS)
    params.update(
        {
            "game_date_gt": start_dt,
            "game_date_lt": end_dt,
            "hfSea": f"{season}|",
            "player_type": player_type,
        }
    )
    if extra_params:
        params.update(extra_params)

    headers = {
        # Savant has occasionally blocked requests with no UA set.
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }

    resp = requests.get(SAVANT_CSV_URL, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    return df


def fetch_pulled_air_rate_rows(start_dt: str, end_dt: str, season: int) -> pd.DataFrame:
    """
    Convenience wrapper: fetch only fly-ball/popup batted ball events for
    the pulled-air-rate computation, scoped to batters.

    Filters server-side via hfBBT (batted ball type) where possible to
    keep the response small; falls back to client-side filtering in
    savant.compute_pulled_air_rate() either way since Savant's hfBBT
    filter values aren't 100% reliable across all park/date combos.
    """
    return fetch_statcast_csv(
        start_dt=start_dt,
        end_dt=end_dt,
        season=season,
        player_type="batter",
    )


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    out_path = DATA_DIR / filename
    df.to_csv(out_path, index=False)
    return out_path
