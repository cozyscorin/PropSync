"""
Runnable tests for the PropSync dashboard's non-Streamlit code: the sample
data generator (sample_data.py), the data-loading seam (data_loader.py),
and the display formatting helpers (formatting.py).

This deliberately does NOT test app.py itself — app.py is a thin Streamlit
rendering layer with no testable logic of its own (every actual
transformation it needs lives in formatting.py / data_loader.py, which is
exactly why those were split out). app.py is checked separately via
`python -c "import ast; ast.parse(...)"` for syntax validity, per the task
instructions, since there's no Streamlit runtime in this sandbox to
actually launch it.

Run with:
    cd "[C] dashboard"
    python3 tests/test_dashboard_helpers.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "[C] edge_ranking"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "[C] scoring_model"))

import pandas as pd

import data_loader
import formatting
import sample_data
from ranking import EdgeCandidate, BookPrice


# ---------------------------------------------------------------------------
# sample_data.py
# ---------------------------------------------------------------------------

class TestSampleDataRawStats(unittest.TestCase):
    def setUp(self):
        self.batter_df = sample_data.build_sample_raw_stats_df()
        self.pitcher_df = sample_data.build_sample_pitcher_stats_df()

    def test_batter_df_has_one_row_per_configured_batter(self):
        self.assertEqual(len(self.batter_df), len(sample_data._BATTERS))

    def test_pitcher_df_has_one_row_per_configured_pitcher(self):
        self.assertEqual(len(self.pitcher_df), len(sample_data._PITCHERS))

    def test_required_showcase_columns_present(self):
        required = {
            "barrel_pct", "hard_hit_pct", "fly_ball_pct", "swinging_strike_pct",
            "wOBA_vs_L", "wOBA_vs_R", "park_factor_1b", "park_factor_2b",
            "park_factor_3b", "park_factor_hr", "pulled_air_rate",
        }
        missing = required - set(self.batter_df.columns)
        self.assertEqual(missing, set(), f"Missing required showcase columns: {missing}")

    def test_park_factors_are_split_by_hit_type_not_blended(self):
        # Per the explicit Scoring Framework Notes requirement: a park's
        # 1B/2B/3B/HR factors must NOT all be identical (that would mean
        # it's silently one blended number in disguise).
        row = self.batter_df.iloc[0]
        values = {row["park_factor_1b"], row["park_factor_2b"], row["park_factor_3b"], row["park_factor_hr"]}
        self.assertGreater(len(values), 1, "Park factors should differ by hit type, not be one blended number")

    def test_rate_stats_in_plausible_bounds(self):
        for col in ("barrel_pct", "hard_hit_pct", "fly_ball_pct", "groundball_pct",
                    "line_drive_pct", "swinging_strike_pct", "pulled_air_rate"):
            self.assertTrue((self.batter_df[col] >= 0).all(), f"{col} has negative values")
            self.assertTrue((self.batter_df[col] <= 1).all(), f"{col} has values > 1")

    def test_hit_type_counting_rates_do_not_exceed_total_hit_rate(self):
        # single/PA + double/PA + hr/PA should never exceed hit/PA (a hit
        # is exactly one of single/double/triple/HR).
        total = (
            self.batter_df["single_per_pa"]
            + self.batter_df["double_per_pa"]
            + self.batter_df["hr_per_pa"]
        )
        self.assertTrue((total <= self.batter_df["hit_per_pa"] + 1e-9).all())

    def test_deterministic_across_calls(self):
        df2 = sample_data.build_sample_raw_stats_df()
        pd.testing.assert_frame_equal(self.batter_df, df2)

    def test_pitcher_stats_in_plausible_bounds(self):
        self.assertTrue((self.pitcher_df["k_per_9"] > 0).all())
        self.assertTrue((self.pitcher_df["csw_pct"] > 0).all())
        self.assertTrue((self.pitcher_df["csw_pct"] < 1).all())
        self.assertTrue((self.pitcher_df["expected_innings"] > 0).all())


class TestSampleDataEdgeCandidates(unittest.TestCase):
    def setUp(self):
        self.candidates = sample_data.build_sample_edge_candidates()

    def test_returns_edge_candidates(self):
        self.assertGreater(len(self.candidates), 0)
        self.assertIsInstance(self.candidates[0], EdgeCandidate)

    def test_covers_all_seven_prop_types(self):
        expected = {
            "batter_home_runs", "batter_hits", "batter_total_bases", "batter_rbis",
            "batter_singles", "batter_doubles", "pitcher_strikeouts",
        }
        seen = {c.market for c in self.candidates}
        self.assertEqual(seen, expected)

    def test_sorted_by_edge_descending(self):
        edges = [c.edge for c in self.candidates]
        self.assertEqual(edges, sorted(edges, reverse=True))

    def test_model_and_market_probs_in_bounds(self):
        for c in self.candidates:
            self.assertGreaterEqual(c.model_prob, 0.0)
            self.assertLessEqual(c.model_prob, 1.0)
            self.assertGreaterEqual(c.chosen_market_prob, 0.0)
            self.assertLessEqual(c.chosen_market_prob, 1.0)

    def test_same_player_can_appear_under_multiple_prop_types(self):
        # Per the Scoring Framework Notes' deliberate no-exclusion decision
        # (also tested in edge_ranking's own suite) — confirm the sample
        # data actually produces this overlap, not just that the selector
        # would allow it.
        from collections import Counter
        markets_by_player = {}
        for c in self.candidates:
            markets_by_player.setdefault(c.player, set()).add(c.market)
        multi_prop_players = [p for p, markets in markets_by_player.items() if len(markets) > 1]
        self.assertGreater(len(multi_prop_players), 0)

    def test_same_game_can_have_multiple_players_picked(self):
        from collections import Counter
        game_counts = Counter(c.game_key for c in self.candidates)
        self.assertTrue(any(n > 1 for n in game_counts.values()))

    def test_chosen_book_is_one_of_the_available_books(self):
        for c in self.candidates:
            available = {bp.bookmaker for bp in c.all_book_prices}
            self.assertIn(c.chosen_book, available)

    def test_deterministic_across_calls(self):
        candidates2 = sample_data.build_sample_edge_candidates()
        self.assertEqual(len(self.candidates), len(candidates2))
        for c1, c2 in zip(self.candidates, candidates2):
            self.assertEqual(c1.player, c2.player)
            self.assertEqual(c1.market, c2.market)
            self.assertAlmostEqual(c1.edge, c2.edge, places=9)


# ---------------------------------------------------------------------------
# data_loader.py
# ---------------------------------------------------------------------------

class TestDataLoader(unittest.TestCase):
    def test_is_live_data_flag_is_false_by_default(self):
        self.assertFalse(data_loader.IS_LIVE_DATA)

    def test_load_raw_stats_returns_two_dataframes(self):
        batter_df, pitcher_df = data_loader.load_raw_stats()
        self.assertIsInstance(batter_df, pd.DataFrame)
        self.assertIsInstance(pitcher_df, pd.DataFrame)
        self.assertGreater(len(batter_df), 0)
        self.assertGreater(len(pitcher_df), 0)

    def test_load_ranked_edges_returns_edge_candidate_list(self):
        candidates = data_loader.load_ranked_edges()
        self.assertIsInstance(candidates, list)
        self.assertIsInstance(candidates[0], EdgeCandidate)


# ---------------------------------------------------------------------------
# formatting.py
# ---------------------------------------------------------------------------

def _make_fake_candidate(player, market, line, model_prob, chosen_prob, edge, book="fanduel", event_id="evt_999"):
    other_book = "draftkings" if book == "fanduel" else "fanduel"
    bp_chosen = BookPrice(bookmaker=book, price=-120, opposing_price=110, fair_prob=chosen_prob, devig_method="two_way")
    bp_other = BookPrice(bookmaker=other_book, price=-115, opposing_price=105, fair_prob=chosen_prob + 0.03, devig_method="two_way")
    return EdgeCandidate(
        player=player, market=market, line=line, event_id=event_id,
        home_team="NYY", away_team="BOS", model_prob=model_prob,
        chosen_book=book, chosen_book_price=-120, chosen_market_prob=chosen_prob,
        edge=edge, all_book_prices=(bp_chosen, bp_other),
    )


class TestFormattingPicksView(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            _make_fake_candidate("Player A", "batter_hits", 1.5, 0.60, 0.50, 0.10),
            _make_fake_candidate("Player B", "batter_home_runs", None, 0.20, 0.15, 0.05),
            _make_fake_candidate("Player A", "batter_home_runs", None, 0.18, 0.22, -0.04),
        ]

    def test_dataframe_has_expected_columns(self):
        df = formatting.edge_candidates_to_dataframe(self.candidates)
        for col in ("player", "prop_type", "market", "line", "side", "sportsbook",
                    "model_prob", "market_prob", "edge"):
            self.assertIn(col, df.columns)

    def test_sorted_by_edge_descending(self):
        df = formatting.edge_candidates_to_dataframe(self.candidates)
        self.assertEqual(list(df["edge"]), sorted(df["edge"], reverse=True))

    def test_hr_market_line_displays_as_em_dash_and_yes_side(self):
        df = formatting.edge_candidates_to_dataframe(self.candidates)
        hr_row = df[df["market"] == "batter_home_runs"].iloc[0]
        self.assertEqual(hr_row["line"], "—")
        self.assertIn("Yes", hr_row["side"])

    def test_over_under_market_shows_over_side(self):
        df = formatting.edge_candidates_to_dataframe(self.candidates)
        hits_row = df[df["market"] == "batter_hits"].iloc[0]
        self.assertEqual(hits_row["side"], "Over 1.5")

    def test_empty_candidates_returns_empty_dataframe_with_columns(self):
        df = formatting.edge_candidates_to_dataframe([])
        self.assertEqual(len(df), 0)
        self.assertIn("player", df.columns)

    def test_prop_display_name_uses_market_map(self):
        self.assertEqual(formatting.prop_display_name("batter_home_runs"), "Home Run (1+)")
        self.assertEqual(formatting.prop_display_name("pitcher_strikeouts"), "Pitcher Strikeouts")

    def test_prop_display_name_falls_back_to_key_for_unknown_market(self):
        self.assertEqual(formatting.prop_display_name("totally_made_up"), "totally_made_up")


class TestFilterPicks(unittest.TestCase):
    def setUp(self):
        candidates = [
            _make_fake_candidate("Player A", "batter_hits", 1.5, 0.60, 0.50, 0.10),
            _make_fake_candidate("Player B", "batter_home_runs", None, 0.20, 0.15, 0.05),
            _make_fake_candidate("Player C", "batter_doubles", 0.5, 0.30, 0.35, -0.05),
        ]
        self.df = formatting.edge_candidates_to_dataframe(candidates)

    def test_filter_by_prop_type(self):
        out = formatting.filter_picks(self.df, prop_types=["batter_hits"])
        self.assertEqual(set(out["market"]), {"batter_hits"})

    def test_filter_by_min_edge(self):
        out = formatting.filter_picks(self.df, min_edge=0.0)
        self.assertTrue((out["edge"] >= 0.0).all())
        self.assertEqual(len(out), 2)

    def test_filter_by_player(self):
        out = formatting.filter_picks(self.df, players=["Player A"])
        self.assertEqual(set(out["player"]), {"Player A"})

    def test_filter_by_sportsbook(self):
        out = formatting.filter_picks(self.df, sportsbooks=["fanduel"])
        self.assertTrue((out["sportsbook"] == "fanduel").all())

    def test_no_filters_returns_everything(self):
        out = formatting.filter_picks(self.df)
        self.assertEqual(len(out), len(self.df))

    def test_filters_compose(self):
        out = formatting.filter_picks(self.df, prop_types=["batter_hits", "batter_home_runs"], min_edge=0.06)
        self.assertEqual(set(out["player"]), {"Player A"})


class TestTopNParlayView(unittest.TestCase):
    def test_top_n_matches_select_top_n_size_and_order(self):
        candidates = sample_data.build_sample_edge_candidates()
        df = formatting.top_n_parlay_view(candidates, 5)
        self.assertEqual(len(df), 5)
        self.assertEqual(list(df["edge_pct"]), [f"{c.edge:+.1%}" for c in candidates[:5]])

    def test_top_n_larger_than_list_returns_all(self):
        candidates = sample_data.build_sample_edge_candidates()
        df = formatting.top_n_parlay_view(candidates, len(candidates) + 50)
        self.assertEqual(len(df), len(candidates))

    def test_top_n_allows_same_player_repeats(self):
        # Confirms the dashboard's parlay view doesn't quietly re-introduce
        # exclusion logic the project deliberately removed.
        candidates = sample_data.build_sample_edge_candidates()
        df = formatting.top_n_parlay_view(candidates, len(candidates))
        self.assertGreater(df["player"].duplicated().sum(), 0)


class TestRawStatsShowcaseFormatting(unittest.TestCase):
    def setUp(self):
        self.batter_df, self.pitcher_df = data_loader.load_raw_stats()

    def test_batter_showcase_renames_expected_columns(self):
        out = formatting.batter_showcase_table(self.batter_df)
        for label in ("Barrel %", "Hard-Hit %", "Fly Ball %", "SwStr %", "wOBA vs LHP", "wOBA vs RHP", "PF: HR"):
            self.assertIn(label, out.columns)

    def test_batter_showcase_formats_percentages_as_strings(self):
        out = formatting.batter_showcase_table(self.batter_df)
        sample_val = out["Barrel %"].iloc[0]
        self.assertIsInstance(sample_val, str)
        self.assertTrue(sample_val.endswith("%"))

    def test_pitcher_showcase_renames_expected_columns(self):
        out = formatting.pitcher_showcase_table(self.pitcher_df)
        for label in ("K/9", "CSW %", "GB % Allowed", "Exp. Innings"):
            self.assertIn(label, out.columns)

    def test_park_factor_table_one_row_per_team_split_by_hit_type(self):
        out = formatting.park_factor_table(self.batter_df)
        self.assertEqual(len(out), out["Team"].nunique())
        for col in ("1B Park Factor", "2B Park Factor", "3B Park Factor", "HR Park Factor"):
            self.assertIn(col, out.columns)
        # Confirm at least one park has differing factors per hit type.
        row = out.iloc[0]
        values = {row["1B Park Factor"], row["2B Park Factor"], row["3B Park Factor"], row["HR Park Factor"]}
        self.assertGreater(len(values), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
