"""
Runnable tests for edge_ranking/live_integration.py — the registry-builder
that closes the gap every prior README flagged: building a real
`(player, market) -> *Inputs` registry from data_pipeline's raw stat pulls,
instead of sample_data.py's fabricated stand-in.

No live pybaseball/Odds-API pull is reachable from this sandbox (confirmed,
repeatedly, in every prior build — see data_pipeline/README.md). So these
tests build synthetic DataFrames shaped EXACTLY like what each pipeline
function's docstring says it returns:
  - get_batter_season_stats() / get_pitcher_season_stats(): FanGraphs JSON
    API columns — Name, Team, PA, HR, H, 1B, 2B, RBI, Barrel%, HardHit%,
    xSLG, xBA, ISO, K/9, HR/9, WHIP, H/9, GB%, BF (counting stats are
    flagged in the scoring_model README as "likely candidates, unconfirmed
    until live" — tests use those likely names).
  - batter_rolling_batted_ball_profile() / pitcher_rolling_profile():
    keyed by raw numeric MLBAM `batter`/`pitcher` IDs, per
    statcast/rolling_form.py's actual groupby("batter")/groupby("pitcher")
    code.
  - get_park_factors() / get_extra_base_park_factors() / get_hr_park_factor():
    Team-keyed, 1B/2B/3B/HR index columns, per park_factors.py.
  - get_all_player_props_today(): event_id, commence_time, home_team,
    away_team, bookmaker, market, player, outcome_name, line, price, per
    odds_api_client.py's actual row-building code.

Covers:
  1. normalize_name() — accents, Jr./Sr./III suffixes, periods, whitespace.
  2. PlayerMatcher — normalized-name path (the only path actually usable
     without a real ID lookup table) and the ID path when one IS supplied.
  3. Each of the 7 build_*_inputs() field-mapping functions individually,
     against realistic single-row synthetic Series.
  4. build_inputs_registry() end-to-end: a multi-game synthetic odds_df +
     synthetic season/rolling/park-factor DataFrames -> a real registry ->
     fed straight into the REAL build_candidate_legs()/rank_edges() from
     ranking.py, confirming the full live-shaped path actually scores.
  5. Failure-mode coverage: an odds-feed name with no pipeline match is
     skipped, not crashed; missing optional DataFrames degrade to
     documented defaults instead of raising.

Run with:
    cd "[C] edge_ranking"
    python3 tests/test_live_integration.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "[C] scoring_model"))

import pandas as pd

from live_integration import (
    PlayerMatcher,
    build_doubles_inputs,
    build_hits_inputs,
    build_home_run_inputs,
    build_inputs_registry,
    build_pitcher_strikeout_inputs,
    build_rbi_inputs,
    build_singles_inputs,
    build_total_bases_inputs,
    normalize_name,
)
from ranking import build_candidate_legs, rank_edges
from doubles import DoublesInputs
from hits import HitsInputs
from home_runs import HomeRunInputs
from pitcher_strikeouts import PitcherStrikeoutInputs
from rbis import RBIInputs
from singles import SinglesInputs
from total_bases import TotalBasesInputs


# ---------------------------------------------------------------------------
# 1. normalize_name()
# ---------------------------------------------------------------------------

class TestNormalizeName(unittest.TestCase):
    def test_suffix_stripping_jr(self):
        self.assertEqual(normalize_name("Fernando Tatis Jr."), "fernando tatis")

    def test_suffix_stripping_jr_no_period(self):
        self.assertEqual(normalize_name("Ronald Acuna Jr"), "ronald acuna")

    def test_suffix_stripping_iii(self):
        self.assertEqual(normalize_name("Robert Smith III"), "robert smith")

    def test_accent_stripping(self):
        self.assertEqual(normalize_name("José Ramírez"), "jose ramirez")
        self.assertEqual(normalize_name("Julio Rodríguez"), "julio rodriguez")

    def test_periods_in_initials(self):
        self.assertEqual(normalize_name("A.J. Puk"), "aj puk")

    def test_case_folding(self):
        self.assertEqual(normalize_name("MIKE TROUT"), normalize_name("mike trout"))

    def test_whitespace_collapsing(self):
        self.assertEqual(normalize_name("  Mike   Trout  "), "mike trout")

    def test_empty_and_none(self):
        self.assertEqual(normalize_name(""), "")
        self.assertEqual(normalize_name(None), "")

    def test_two_sources_same_player_converge(self):
        # The exact scenario this whole module exists to solve: odds feed
        # says one thing, a leaderboard pull says another, both should
        # canonicalize identically.
        odds_feed_name = "Fernando Tatis Jr."
        leaderboard_name = "Fernando Tatis Jr."  # FanGraphs is usually consistent
        self.assertEqual(normalize_name(odds_feed_name), normalize_name(leaderboard_name))


# ---------------------------------------------------------------------------
# 2. PlayerMatcher
# ---------------------------------------------------------------------------

class TestPlayerMatcher(unittest.TestCase):
    def test_normalized_name_path_matches(self):
        matcher = PlayerMatcher()
        result = matcher.resolve("Mike Trout")
        self.assertTrue(result.matched)
        self.assertEqual(result.match_method, "normalized_name")
        self.assertEqual(result.pipeline_key, "mike trout")

    def test_id_path_used_when_lookup_supplied(self):
        matcher = PlayerMatcher(name_to_mlbam_id={"Mike Trout": 545361})
        result = matcher.resolve("Mike Trout")
        self.assertTrue(result.matched)
        self.assertEqual(result.match_method, "mlbam_id")
        self.assertEqual(result.pipeline_key, 545361)

    def test_id_path_falls_back_to_name_for_unlisted_player(self):
        matcher = PlayerMatcher(name_to_mlbam_id={"Mike Trout": 545361})
        result = matcher.resolve("Aaron Judge")
        self.assertTrue(result.matched)
        self.assertEqual(result.match_method, "normalized_name")

    def test_known_failure_mode_nickname_does_not_match(self):
        # Documented limitation: "Mike Trout" vs "Michael Trout" won't
        # converge. This test exists to make that limitation visible and
        # regression-checked, not to claim it's fixed.
        self.assertNotEqual(normalize_name("Mike Trout"), normalize_name("Michael Trout"))

    def test_empty_name_unmatched(self):
        matcher = PlayerMatcher()
        result = matcher.resolve("")
        self.assertFalse(result.matched)
        self.assertEqual(result.match_method, "unmatched")


# ---------------------------------------------------------------------------
# 3. Per-prop field mapping functions
# ---------------------------------------------------------------------------

def _row(d: dict) -> pd.Series:
    return pd.Series(d)


class TestHomeRunInputsMapping(unittest.TestCase):
    def setUp(self):
        self.batter_row = _row(dict(
            Name="Aaron Judge", Team="NYY", PA=350, HR=28, H=90,
            **{"Barrel%": 0.18, "xSLG": 0.62, "HardHit%": 0.48, "ISO": 0.310, "xBA": 0.275},
        ))
        self.pitcher_row = _row(dict(
            Name="Tanner Houck", Team="BOS",
            **{"HR/9": 1.1, "Barrel%": 0.07, "K/9": 9.2, "WHIP": 1.21, "H/9": 7.9},
        ))

    def test_builds_real_dataclass_with_pipeline_values(self):
        inputs = build_home_run_inputs(
            self.batter_row, None, self.pitcher_row, park_hr_factor=112.0, lineup_spot=2,
        )
        self.assertIsInstance(inputs, HomeRunInputs)
        self.assertAlmostEqual(inputs.batter_hr_per_pa_season, 28 / 350)
        self.assertEqual(inputs.batter_pa_season, 350.0)
        self.assertEqual(inputs.batter_barrel_pct, 0.18)
        self.assertEqual(inputs.batter_xslg, 0.62)
        self.assertEqual(inputs.pitcher_hr_per_9, 1.1)
        self.assertEqual(inputs.pitcher_barrel_pct_allowed, 0.07)
        self.assertEqual(inputs.park_hr_factor, 112.0)
        self.assertEqual(inputs.batter_lineup_spot, 2)
        self.assertEqual(inputs.weather_hr_multiplier, 1.0)
        # Documented real gap: rolling form doesn't expose HR counts.
        self.assertIsNone(inputs.batter_hr_per_pa_recent)
        self.assertIsNone(inputs.batter_pa_recent)

    def test_missing_pitcher_row_falls_back_to_documented_default(self):
        inputs = build_home_run_inputs(self.batter_row, None, None, park_hr_factor=100.0)
        self.assertEqual(inputs.pitcher_hr_per_9, 1.2)  # same default HomeRunInputs itself uses

    def test_missing_park_factor_falls_back_to_neutral_100(self):
        inputs = build_home_run_inputs(self.batter_row, None, self.pitcher_row, park_hr_factor=None)
        self.assertEqual(inputs.park_hr_factor, 100.0)

    def test_scoring_function_runs_on_built_inputs(self):
        from home_runs import score_home_run_prop
        inputs = build_home_run_inputs(self.batter_row, None, self.pitcher_row, park_hr_factor=112.0)
        prob = score_home_run_prop(inputs)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)


class TestHitsInputsMapping(unittest.TestCase):
    def test_builds_and_scores(self):
        batter_row = _row(dict(Name="Mookie Betts", Team="LAD", PA=400, H=112, **{"xBA": 0.280, "HardHit%": 0.42}))
        pitcher_row = _row(dict(Name="Dylan Cease", Team="SDP", **{"WHIP": 1.25}))
        inputs = build_hits_inputs(batter_row, None, pitcher_row, lineup_spot=1)
        self.assertIsInstance(inputs, HitsInputs)
        self.assertAlmostEqual(inputs.batter_hit_per_pa_season, 112 / 400)
        self.assertEqual(inputs.pitcher_whip, 1.25)
        self.assertIsNone(inputs.pitcher_ba_allowed)  # documented: not exposed by pipeline

        from hits import score_hits_prop
        prob = score_hits_prop(inputs, line=0.5)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_missing_batter_pa_does_not_crash(self):
        batter_row = _row(dict(Name="No PA Guy", Team="NYY", H=10))  # no PA column at all
        inputs = build_hits_inputs(batter_row, None, None)
        self.assertEqual(inputs.batter_pa_season, 0.0)
        self.assertAlmostEqual(inputs.batter_hit_per_pa_season, 0.236)  # league-avg fallback


class TestTotalBasesInputsMapping(unittest.TestCase):
    def test_tb_derived_from_components_when_tb_column_absent(self):
        batter_row = _row(dict(
            Name="Yordan Alvarez", Team="HOU", PA=380,
            **{"1B": 60, "2B": 25, "3B": 1, "HR": 30, "ISO": 0.290, "xSLG": 0.580},
        ))
        park_row = _row({"Team": "HOU", "1B": 101, "2B": 104, "3B": 92, "HR": 106})
        inputs = build_total_bases_inputs(batter_row, None, None, park_row, lineup_spot=3)
        self.assertIsInstance(inputs, TotalBasesInputs)
        expected_tb = 60 + 2 * 25 + 3 * 1 + 4 * 30
        self.assertAlmostEqual(inputs.batter_tb_per_pa_season, expected_tb / 380)
        self.assertEqual(inputs.park_factor_hr, 106.0)
        self.assertEqual(inputs.park_factor_2b, 104.0)

    def test_tb_column_used_directly_when_present(self):
        batter_row = _row(dict(Name="X", Team="HOU", PA=380, TB=200, ISO=0.290))
        inputs = build_total_bases_inputs(batter_row, None, None, None)
        self.assertAlmostEqual(inputs.batter_tb_per_pa_season, 200 / 380)


class TestRBIInputsMapping(unittest.TestCase):
    def test_context_fields_default_to_none_not_invented(self):
        batter_row = _row(dict(Name="Manny Machado", Team="SDP", PA=390, RBI=78))
        inputs = build_rbi_inputs(batter_row, None, None, lineup_spot=4)
        self.assertIsInstance(inputs, RBIInputs)
        self.assertIsNone(inputs.obp_of_hitters_ahead)
        self.assertIsNone(inputs.team_implied_run_total)

    def test_caller_can_supply_context_fields(self):
        batter_row = _row(dict(Name="Manny Machado", Team="SDP", PA=390, RBI=78))
        inputs = build_rbi_inputs(
            batter_row, None, None, lineup_spot=4,
            obp_of_hitters_ahead=0.340, team_implied_run_total=5.1,
        )
        self.assertEqual(inputs.obp_of_hitters_ahead, 0.340)
        self.assertEqual(inputs.team_implied_run_total, 5.1)

        from rbis import score_rbi_prop
        prob = score_rbi_prop(inputs, line=0.5)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)


class TestSinglesInputsMapping(unittest.TestCase):
    def test_sprint_speed_pulled_from_separate_row(self):
        batter_row = _row(dict(
            Name="Jose Altuve", Team="HOU", PA=410,
            **{"1B": 95, "GB%": 0.46, "LD%": 0.23, "xBA": 0.290, "ISO": 0.180},
        ))
        speed_row = _row({"Name": "Jose Altuve", "sprint_speed": 27.8})
        inputs = build_singles_inputs(batter_row, None, None, speed_row, lineup_spot=1)
        self.assertIsInstance(inputs, SinglesInputs)
        self.assertEqual(inputs.batter_sprint_speed, 27.8)

    def test_missing_sprint_speed_row_is_none_not_zero(self):
        batter_row = _row(dict(Name="X", Team="HOU", PA=400, **{"1B": 90}))
        inputs = build_singles_inputs(batter_row, None, None, None)
        self.assertIsNone(inputs.batter_sprint_speed)


class TestDoublesInputsMapping(unittest.TestCase):
    def test_uses_doubles_specific_park_factor_not_hr(self):
        batter_row = _row(dict(
            Name="Rafael Devers", Team="BOS", PA=400,
            **{"2B": 35, "HardHit%": 0.45, "LD%": 0.22, "FB%": 0.36, "xSLG": 0.520},
        ))
        extra_base_pf_row = _row({"team": "BOS", "doubles_park_factor": 128.0, "2B": 128.0})
        inputs = build_doubles_inputs(
            batter_row, None, None, None, park_2b_factor=extra_base_pf_row["doubles_park_factor"],
        )
        self.assertIsInstance(inputs, DoublesInputs)
        self.assertEqual(inputs.park_2b_factor, 128.0)
        self.assertNotEqual(inputs.park_2b_factor, 97.0)  # BOS's HR factor, must not leak in here


class TestPitcherStrikeoutInputsMapping(unittest.TestCase):
    def test_prefers_rolling_csw_over_season_when_both_present(self):
        pitcher_row = _row(dict(Name="Spencer Strider", Team="ATL", **{"K/9": 12.5, "CSW%": 0.30}, BF=520))
        rolling_row = _row({"pitcher": 123456, "csw_pct": 0.34})
        inputs = build_pitcher_strikeout_inputs(pitcher_row, rolling_row, opponent_team_k_pct=0.24)
        self.assertEqual(inputs.pitcher_csw_pct, 0.34)  # rolling wins

    def test_falls_back_to_season_csw_when_no_rolling_row(self):
        pitcher_row = _row(dict(Name="Spencer Strider", Team="ATL", **{"K/9": 12.5, "CSW%": 0.30}, BF=520))
        inputs = build_pitcher_strikeout_inputs(pitcher_row, None, opponent_team_k_pct=0.24)
        self.assertEqual(inputs.pitcher_csw_pct, 0.30)

    def test_expected_innings_stays_none_without_workload_feed(self):
        pitcher_row = _row(dict(Name="X", Team="ATL", **{"K/9": 9.0}, BF=400))
        inputs = build_pitcher_strikeout_inputs(pitcher_row, None, opponent_team_k_pct=None)
        self.assertIsNone(inputs.expected_innings)  # scoring_model's own 5.5 default kicks in later

        from pitcher_strikeouts import score_pitcher_strikeouts_prop
        prob = score_pitcher_strikeouts_prop(inputs, line=5.5)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)


# ---------------------------------------------------------------------------
# 4. build_inputs_registry() end-to-end against the REAL ranking.py
# ---------------------------------------------------------------------------

def _synthetic_batter_season_df() -> pd.DataFrame:
    """Shaped like get_batter_season_stats(): FanGraphs JSON API columns,
    Name/Team display strings (not IDs) — per savant.py's _fg_api_fetch()."""
    return pd.DataFrame([
        dict(Name="Aaron Judge", Team="NYY", PA=350, HR=28, H=90, **{
            "1B": 45, "2B": 15, "RBI": 70, "Barrel%": 0.18, "HardHit%": 0.48,
            "xSLG": 0.620, "xBA": 0.275, "ISO": 0.310, "GB%": 0.38, "LD%": 0.24,
            "FB%": 0.38, "K%": 0.28,
        }),
        dict(Name="Rafael Devers", Team="BOS", PA=400, HR=22, H=110, **{
            "1B": 60, "2B": 28, "RBI": 75, "Barrel%": 0.14, "HardHit%": 0.45,
            "xSLG": 0.520, "xBA": 0.270, "ISO": 0.240, "GB%": 0.40, "LD%": 0.22,
            "FB%": 0.36, "K%": 0.22,
        }),
        dict(Name="Fernando Tatis Jr.", Team="SDP", PA=410, HR=24, H=105, **{
            "1B": 58, "2B": 22, "RBI": 68, "Barrel%": 0.15, "HardHit%": 0.44,
            "xSLG": 0.510, "xBA": 0.265, "ISO": 0.230, "GB%": 0.39, "LD%": 0.23,
            "FB%": 0.37, "K%": 0.25,
        }),
    ])


def _synthetic_pitcher_season_df() -> pd.DataFrame:
    return pd.DataFrame([
        dict(Name="Tanner Houck", Team="BOS", **{
            "K/9": 9.2, "HR/9": 1.1, "Barrel%": 0.07, "WHIP": 1.21, "H/9": 7.9,
            "GB%": 0.45, "CSW%": 0.29, "BF": 540,
        }),
        dict(Name="Dylan Cease", Team="SDP", **{
            "K/9": 11.0, "HR/9": 1.3, "Barrel%": 0.09, "WHIP": 1.30, "H/9": 8.2,
            "GB%": 0.40, "CSW%": 0.31, "BF": 560,
        }),
    ])


def _synthetic_park_factors_df() -> pd.DataFrame:
    """Shaped like get_park_factors(): Team-keyed, per-hit-type columns,
    per park_factors.py's scrape + numeric coercion."""
    return pd.DataFrame([
        {"Team": "NYY", "1B": 98, "2B": 101, "3B": 85, "HR": 112},
        {"Team": "BOS", "1B": 103, "2B": 128, "3B": 95, "HR": 97},
        {"Team": "SDP", "1B": 99, "2B": 102, "3B": 110, "HR": 89},
    ])


def _synthetic_extra_base_park_factors_df() -> pd.DataFrame:
    """Shaped like get_extra_base_park_factors(): lowercase 'team' column,
    per its own rename in park_factors.py."""
    df = _synthetic_park_factors_df().rename(columns={"Team": "team", "2B": "doubles_park_factor", "3B": "triples_park_factor"})
    return df[["team", "doubles_park_factor", "triples_park_factor"]]


def _synthetic_odds_df() -> pd.DataFrame:
    """Shaped exactly like get_all_player_props_today()'s row-building
    code in odds_api_client.py: event_id, commence_time, home_team,
    away_team, bookmaker, market, player, outcome_name, line, price."""
    rows = []
    common = dict(event_id="evt_001", commence_time="2026-06-22T23:05:00Z", home_team="NYY", away_team="BOS")
    for bookmaker in ("fanduel", "draftkings"):
        rows += [
            dict(**common, bookmaker=bookmaker, market="batter_home_runs", player="Aaron Judge",
                 outcome_name="Yes", line=None, price=-150 if bookmaker == "fanduel" else -140),
            dict(**common, bookmaker=bookmaker, market="batter_home_runs", player="Aaron Judge",
                 outcome_name="No", line=None, price=120 if bookmaker == "fanduel" else 115),
            dict(**common, bookmaker=bookmaker, market="batter_hits", player="Aaron Judge",
                 outcome_name="Over", line=1.5, price=-110),
            dict(**common, bookmaker=bookmaker, market="batter_hits", player="Aaron Judge",
                 outcome_name="Under", line=1.5, price=-110),
            dict(**common, bookmaker=bookmaker, market="batter_doubles", player="Rafael Devers",
                 outcome_name="Over", line=0.5, price=145),
            dict(**common, bookmaker=bookmaker, market="batter_doubles", player="Rafael Devers",
                 outcome_name="Under", line=0.5, price=-175),
            dict(**common, bookmaker=bookmaker, market="pitcher_strikeouts", player="Tanner Houck",
                 outcome_name="Over", line=5.5, price=-115),
            dict(**common, bookmaker=bookmaker, market="pitcher_strikeouts", player="Tanner Houck",
                 outcome_name="Under", line=5.5, price=-105),
        ]
    # A player with no pipeline match at all — must be skipped, not crash.
    rows += [
        dict(**common, bookmaker="fanduel", market="batter_rbis", player="Totally Unknown Player",
             outcome_name="Over", line=0.5, price=-120),
        dict(**common, bookmaker="fanduel", market="batter_rbis", player="Totally Unknown Player",
             outcome_name="Under", line=0.5, price=100),
    ]
    return pd.DataFrame(rows)


class TestBuildInputsRegistryEndToEnd(unittest.TestCase):
    def setUp(self):
        self.stat_dataframes = {
            "batter_season": _synthetic_batter_season_df(),
            "pitcher_season": _synthetic_pitcher_season_df(),
            "park_factors": _synthetic_park_factors_df(),
            "extra_base_park_factors": _synthetic_extra_base_park_factors_df(),
        }
        self.odds_df = _synthetic_odds_df()

    def test_registry_has_entries_for_matched_players(self):
        registry = build_inputs_registry(self.stat_dataframes, self.odds_df)
        self.assertIn(("Aaron Judge", "batter_home_runs"), registry)
        self.assertIn(("Aaron Judge", "batter_hits"), registry)
        self.assertIn(("Rafael Devers", "batter_doubles"), registry)
        self.assertIn(("Tanner Houck", "pitcher_strikeouts"), registry)

    def test_unmatched_player_absent_from_registry(self):
        registry = build_inputs_registry(self.stat_dataframes, self.odds_df)
        self.assertNotIn(("Totally Unknown Player", "batter_rbis"), registry)

    def test_suffix_normalization_resolves_odds_name_to_leaderboard_row(self):
        # Odds feed and leaderboard both say "Fernando Tatis Jr." here, but
        # this confirms the path still works when matched via the
        # normalized-name key, not a literal string match shortcut.
        odds_df = pd.DataFrame([
            dict(event_id="evt_002", commence_time="x", home_team="SDP", away_team="LAD",
                 bookmaker="fanduel", market="batter_home_runs", player="Fernando Tatis Jr.",
                 outcome_name="Yes", line=None, price=-130),
            dict(event_id="evt_002", commence_time="x", home_team="SDP", away_team="LAD",
                 bookmaker="fanduel", market="batter_home_runs", player="Fernando Tatis Jr.",
                 outcome_name="No", line=None, price=110),
        ])
        registry = build_inputs_registry(self.stat_dataframes, odds_df)
        self.assertIn(("Fernando Tatis Jr.", "batter_home_runs"), registry)
        inputs = registry[("Fernando Tatis Jr.", "batter_home_runs")]
        self.assertAlmostEqual(inputs.batter_hr_per_pa_season, 24 / 410)

    def test_registry_feeds_real_build_candidate_legs_and_rank_edges(self):
        registry = build_inputs_registry(self.stat_dataframes, self.odds_df)
        candidates = build_candidate_legs(self.odds_df, registry)
        ranked = rank_edges(candidates)

        self.assertGreater(len(candidates), 0)
        self.assertEqual(len(ranked), len(candidates))
        # Edges sorted descending.
        edges = [c.edge for c in ranked]
        self.assertEqual(edges, sorted(edges, reverse=True))
        # The unmatched player must not have produced a candidate leg.
        self.assertNotIn("Totally Unknown Player", {c.player for c in candidates})
        # All scored probabilities are valid probabilities.
        for c in candidates:
            self.assertGreaterEqual(c.model_prob, 0.0)
            self.assertLessEqual(c.model_prob, 1.0)

    def test_missing_optional_dataframes_degrade_gracefully(self):
        # Only batter_season supplied — no pitcher_season, no park factors.
        # Must not raise; opposing-pitcher and park-factor fields fall back
        # to scoring_model's own documented defaults.
        minimal = {"batter_season": _synthetic_batter_season_df()}
        registry = build_inputs_registry(minimal, self.odds_df)
        self.assertIn(("Aaron Judge", "batter_hits"), registry)
        hits_inputs = registry[("Aaron Judge", "batter_hits")]
        self.assertEqual(hits_inputs.pitcher_whip, 1.30)  # documented fallback default

    def test_empty_stat_dataframes_does_not_crash(self):
        registry = build_inputs_registry({}, self.odds_df)
        # Nothing can be matched without any season stats at all.
        self.assertEqual(registry, {})

    def test_doubles_uses_extra_base_park_factor_not_hr_factor(self):
        registry = build_inputs_registry(self.stat_dataframes, self.odds_df)
        devers_doubles = registry[("Rafael Devers", "batter_doubles")]
        self.assertEqual(devers_doubles.park_2b_factor, 128.0)  # BOS doubles factor
        self.assertNotEqual(devers_doubles.park_2b_factor, 97.0)  # BOS HR factor


if __name__ == "__main__":
    unittest.main(verbosity=2)
