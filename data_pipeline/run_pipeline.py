"""
Manual entry point: pull everything that's currently testable and dump
sanity-check output + CSVs to data/raw/.

Run this from inside the data_pipeline folder after `pip install -r
requirements.txt`:

    python run_pipeline.py

What it does:
  1. Pulls FanGraphs season batting + pitching stats (pybaseball)
  2. Pulls FanGraphs park factors split by hit type (pybaseball)
  3. Pulls a 15-day rolling Statcast window and derives CSW%, barrel%, etc.
  4. Prints row counts + a few sample columns from each so you can
     eyeball that the numbers look sane (e.g. barrel% in single-digit to
     20%-ish range, not 300%; HR park factors clustered ~85-115, etc.)
  5. Skips the Odds API section entirely if ODDS_API_KEY isn't set, and
     tells you so instead of crashing.

This script does NOT run any scoring/ranking logic — it's pure data
sourcing, per the project's current scope.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from config import SEASON, odds_api_key_present  # noqa: E402


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> None:
    section(f"PropSync Data Pipeline — season {SEASON}")

    # --- Statcast / FanGraphs batting + pitching -------------------------
    section("1. FanGraphs season batting stats (pybaseball.batting_stats)")
    try:
        from statcast.savant import get_batter_season_stats, save_csv

        batters = get_batter_season_stats(SEASON)
        print(f"Rows: {len(batters)}")
        sample_cols = [
            c
            for c in ["Name", "Team", "Barrel%", "HardHit%", "FB%", "SwStr%", "xSLG", "xBA", "ISO"]
            if c in batters.columns
        ]
        print(batters[sample_cols].head(10).to_string(index=False))
        path = save_csv(batters, "batters_season.csv")
        print(f"Saved -> {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}")

    section("2. FanGraphs season pitching stats (pybaseball.pitching_stats)")
    try:
        from statcast.savant import get_pitcher_season_stats

        pitchers = get_pitcher_season_stats(SEASON)
        print(f"Rows: {len(pitchers)}")
        sample_cols = [
            c
            for c in ["Name", "Team", "K/9", "HR/9", "Barrel%", "SwStr%"]
            if c in pitchers.columns
        ]
        print(pitchers[sample_cols].head(10).to_string(index=False))
        save_csv(pitchers, "pitchers_season.csv")
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}")

    # --- Park factors ------------------------------------------------------
    section("3. Park factors split by hit type (pybaseball.park_factors)")
    try:
        from park_factors.park_factors import get_park_factors
        from statcast.savant import save_csv as save_csv2

        pf = get_park_factors(SEASON)
        print(f"Rows: {len(pf)}")
        print(f"Columns: {list(pf.columns)}")
        print(pf.head(10).to_string(index=False))
        save_csv2(pf, "park_factors.csv")
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}")

    # --- Rolling 15-day form -------------------------------------------
    section("4. Rolling 15-day batter batted-ball profile (derived from raw Statcast)")
    try:
        from statcast.rolling_form import batter_rolling_batted_ball_profile

        rolling_batters = batter_rolling_batted_ball_profile(15)
        print(f"Rows: {len(rolling_batters)}")
        print(rolling_batters.head(10).to_string(index=False))
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}")

    section("5. Rolling 15-day pitcher profile incl. CSW% (derived from raw Statcast)")
    try:
        from statcast.rolling_form import pitcher_rolling_profile

        rolling_pitchers = pitcher_rolling_profile(15)
        print(f"Rows: {len(rolling_pitchers)}")
        print(rolling_pitchers.head(10).to_string(index=False))
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}")

    # --- Odds API (FanDuel + DraftKings player props) -------------------
    section("6. The Odds API — FanDuel + DraftKings player props")
    if not odds_api_key_present():
        print(
            "SKIPPED: ODDS_API_KEY not set. Copy .env.example to .env, add a "
            "free key from https://the-odds-api.com/, then re-run this "
            "script to pull live FanDuel + DraftKings player prop odds."
        )
    else:
        try:
            from odds.odds_api_client import get_all_player_props_today, save_csv as save_csv3

            props = get_all_player_props_today()
            print(f"Rows: {len(props)}")
            print(props.head(20).to_string(index=False))
            save_csv3(props, "player_props_today.csv")
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED: {exc}")

    section("Done")


if __name__ == "__main__":
    main()
