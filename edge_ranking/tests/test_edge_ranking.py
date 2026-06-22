"""
Runnable tests for the PropSync edge_ranking layer, using realistic
synthetic fixtures (the real odds pipeline has no API key yet and the
real data pipeline hasn't pulled live data yet — see both READMEs).

Covers:
  1. De-vig math against hand-calculated examples (devig.py).
  2. Edge sign/magnitude sanity (a leg the model likes more than the
     market should produce a positive edge of the expected rough size,
     and vice versa).
  3. Dual-bookmaker selection: when both books post a price, the cheaper
     (lower fair-prob) one is chosen; when only one book posts, it's used
     without being dropped.
  4. Top-N selection with NO exclusion — same-player and same-game legs
     are both allowed to appear together in the result.
  5. An end-to-end run: synthetic odds DataFrame (shaped exactly like
     get_all_player_props_today()'s output) + synthetic *Inputs ->
     build_candidate_legs -> rank_edges -> select_top_n.

Run with:
    cd "[C] edge_ranking"
    python3 tests/test_edge_ranking.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "[C] scoring_model"))

import pandas as pd

from devig import american_to_implied_prob, devig_single_side_assumed_vig, devig_two_way
from doubles import DoublesInputs
from hits import HitsInputs
from home_runs import HomeRunInputs
from parlay_selector import select_top_n
from pitcher_strikeouts import PitcherStrikeoutInputs
from ranking import EdgeCandidate, build_candidate_legs, rank_edges
from singles import SinglesInputs
from total_bases import TotalBasesInputs


# ---------------------------------------------------------------------------
# 1. De-vig math vs hand-calculated examples
# ---------------------------------------------------------------------------

class TestDevigMath(unittest.TestCase):
    def test_american_to_implied_favorite(self):
        # -150 -> 150/250 = 0.6 exactly
        self.assertAlmostEqual(american_to_implied_prob(-150), 0.6, places=10)

    def test_american_to_implied_underdog(self):
        # +120 -> 100/220 = 0.454545...
        self.assertAlmostEqual(american_to_implied_prob(120), 100 / 220, places=10)

    def test_american_to_implied_pick_em(self):
        # +100 -> 100/200 = 0.5 exactly
        self.assertAlmostEqual(american_to_implied_prob(100), 0.5, places=10)

    def test_american_to_implied_even_negative(self):
        # -100 -> 100/200 = 0.5 exactly
        self.assertAlmostEqual(american_to_implied_prob(-100), 0.5, places=10)

    def test_devig_two_way_hand_calculated(self):
        # Over -150 / Under +120, worked by hand in devig.py's docstring:
        #   raw_over = 0.60000, raw_under = 0.454545...
        #   overround = 1.054545...
        #   fair_over = 0.60000 / 1.054545... = 0.568965...
        #   fair_under = 0.454545... / 1.054545... = 0.431034...
        result = devig_two_way(-150, 120)
        self.assertAlmostEqual(result.side_a_raw_implied, 0.6, places=10)
        self.assertAlmostEqual(result.side_b_raw_implied, 100 / 220, places=10)
        self.assertAlmostEqual(result.overround, 0.6 + 100 / 220, places=10)
        self.assertAlmostEqual(result.side_a_fair_prob, 0.6 / (0.6 + 100 / 220), places=10)
        self.assertAlmostEqual(result.side_b_fair_prob, (100 / 220) / (0.6 + 100 / 220), places=10)
        # Fair probabilities must sum to exactly 1.0 (vig fully removed).
        self.assertAlmostEqual(result.side_a_fair_prob + result.side_b_fair_prob, 1.0, places=10)

    def test_devig_two_way_symmetric_minus_110(self):
        # The classic -110/-110 market: vig removed should land exactly
        # at 50/50, the textbook sanity check for any de-vig formula.
        result = devig_two_way(-110, -110)
        self.assertAlmostEqual(result.side_a_fair_prob, 0.5, places=10)
        self.assertAlmostEqual(result.side_b_fair_prob, 0.5, places=10)

    def test_devig_fair_probs_always_sum_to_one(self):
        for price_a, price_b in [(-150, 120), (-200, 170), (-120, -110), (150, -180)]:
            result = devig_two_way(price_a, price_b)
            self.assertAlmostEqual(result.side_a_fair_prob + result.side_b_fair_prob, 1.0, places=9)

    def test_devig_single_side_fallback_is_lower_than_raw(self):
        # Dividing by an assumed overround > 1.0 should always shrink the
        # raw implied probability down, same direction vig removal always
        # moves (raw implied prob is inflated vs fair prob).
        raw = american_to_implied_prob(-150)
        fallback = devig_single_side_assumed_vig(-150)
        self.assertLess(fallback, raw)


# ---------------------------------------------------------------------------
# Synthetic fixtures: odds rows + matching player inputs
# ---------------------------------------------------------------------------
# Shape matches get_all_player_props_today()'s documented output exactly:
# event_id, commence_time, home_team, away_team, bookmaker, market, player,
# outcome_name, line, price.

EVENT_1 = dict(event_id="evt_1", commence_time="2026-06-22T23:05:00Z",
                home_team="Yankees", away_team="Red Sox")
EVENT_2 = dict(event_id="evt_2", commence_time="2026-06-22T23:10:00Z",
                home_team="Dodgers", away_team="Giants")

PLAYER_A = "Slugger McGee"     # Yankees batter — HR + hits legs, dual book, books agree-ish
PLAYER_B = "Speedy Contact"    # Red Sox batter — singles leg, ONLY FanDuel posts a line
PLAYER_C = "Ace Heater"        # Dodgers pitcher — strikeouts leg, dual book, books disagree a lot
PLAYER_D = "Gapper Jones"      # Giants batter — doubles leg, same game as PLAYER_C (for same-game-allowed test)
PLAYER_E = "Slugger McGee"     # same player as A, different prop (total bases) — for exclusion test


def build_synthetic_odds_df() -> pd.DataFrame:
    rows = [
        # --- Player A: HR (yes/no market), both books post "Yes", model
        # likes this leg a lot more than the market does (big edge).
        dict(**EVENT_1, bookmaker="fanduel", market="batter_home_runs",
             player=PLAYER_A, outcome_name="Yes", line=None, price=260),
        dict(**EVENT_1, bookmaker="fanduel", market="batter_home_runs",
             player=PLAYER_A, outcome_name="No", line=None, price=-340),
        dict(**EVENT_1, bookmaker="draftkings", market="batter_home_runs",
             player=PLAYER_A, outcome_name="Yes", line=None, price=240),
        dict(**EVENT_1, bookmaker="draftkings", market="batter_home_runs",
             player=PLAYER_A, outcome_name="No", line=None, price=-300),

        # --- Player A: total bases over/under, SAME PLAYER as the HR leg
        # above — used to confirm the exclusion selector correctly drops
        # this even though it's a different prop type.
        dict(**EVENT_1, bookmaker="fanduel", market="batter_total_bases",
             player=PLAYER_E, outcome_name="Over", line=1.5, price=-115),
        dict(**EVENT_1, bookmaker="fanduel", market="batter_total_bases",
             player=PLAYER_E, outcome_name="Under", line=1.5, price=-105),

        # --- Player B: singles over/under, ONLY FanDuel has a line
        # (DraftKings hasn't posted it) — must NOT be dropped.
        dict(**EVENT_1, bookmaker="fanduel", market="batter_singles",
             player=PLAYER_B, outcome_name="Over", line=0.5, price=-125),
        dict(**EVENT_1, bookmaker="fanduel", market="batter_singles",
             player=PLAYER_B, outcome_name="Under", line=0.5, price=105),

        # --- Player C: pitcher strikeouts, dual book, books disagree
        # meaningfully on price (FanDuel much pricier on the Over than
        # DraftKings) — used to confirm the dual-book selector picks
        # DraftKings (the cheaper/more favorable price).
        dict(**EVENT_2, bookmaker="fanduel", market="pitcher_strikeouts",
             player=PLAYER_C, outcome_name="Over", line=6.5, price=-170),
        dict(**EVENT_2, bookmaker="fanduel", market="pitcher_strikeouts",
             player=PLAYER_C, outcome_name="Under", line=6.5, price=140),
        dict(**EVENT_2, bookmaker="draftkings", market="pitcher_strikeouts",
             player=PLAYER_C, outcome_name="Over", line=6.5, price=-110),
        dict(**EVENT_2, bookmaker="draftkings", market="pitcher_strikeouts",
             player=PLAYER_C, outcome_name="Under", line=6.5, price=-110),

        # --- Player D: doubles, SAME GAME (evt_2) as Player C above but a
        # different player — used to confirm the selector now ALLOWS both
        # legs through, since same-game exclusion was dropped.
        dict(**EVENT_2, bookmaker="fanduel", market="batter_doubles",
             player=PLAYER_D, outcome_name="Over", line=0.5, price=180),
        dict(**EVENT_2, bookmaker="fanduel", market="batter_doubles",
             player=PLAYER_D, outcome_name="Under", line=0.5, price=-220),
        dict(**EVENT_2, bookmaker="draftkings", market="batter_doubles",
             player=PLAYER_D, outcome_name="Over", line=0.5, price=175),
        dict(**EVENT_2, bookmaker="draftkings", market="batter_doubles",
             player=PLAYER_D, outcome_name="Under", line=0.5, price=-210),

        # --- Player B: hits leg, dual book, model is INDIFFERENT/slightly
        # negative on this one (market correctly favored) — used for edge
        # sign sanity (a leg the market has right should show small/
        # negative edge, not always positive).
        dict(**EVENT_1, bookmaker="fanduel", market="batter_hits",
             player=PLAYER_B, outcome_name="Over", line=1.5, price=170),
        dict(**EVENT_1, bookmaker="fanduel", market="batter_hits",
             player=PLAYER_B, outcome_name="Under", line=1.5, price=-200),
        dict(**EVENT_1, bookmaker="draftkings", market="batter_hits",
             player=PLAYER_B, outcome_name="Over", line=1.5, price=175),
        dict(**EVENT_1, bookmaker="draftkings", market="batter_hits",
             player=PLAYER_B, outcome_name="Under", line=1.5, price=-205),
    ]
    return pd.DataFrame(rows)


def build_synthetic_inputs_registry() -> dict:
    """
    Player inputs deliberately chosen so the model's view diverges from
    the synthetic market prices in a known, checkable direction for each
    test leg above:
      - PLAYER_A HR: elite power profile -> model should see a much
        higher P(1+HR) than the market's de-vigged ~26% on a +250-ish
        price, producing a clearly positive edge.
      - PLAYER_E (== PLAYER_A) total bases: same elite slugger, line 1.5
        -> model should also lean positive; only used for the exclusion
        test, sign doesn't need separate checking.
      - PLAYER_B singles: average contact profile against a neutral
        pitcher -> model lands close to a coin flip, not a strong view.
      - PLAYER_B hits: deliberately mediocre inputs against a tough
        (low hits-allowed) pitcher, line 1.5 -> model should NOT show a
        big positive edge here, contrasting with Player A's HR leg.
      - PLAYER_C strikeouts: dominant K-rate starter vs a high-K-rate
        opposing lineup -> model should like the Over a lot.
      - PLAYER_D doubles: average gap-power profile -> model near neutral.
    """
    registry = {}

    registry[(PLAYER_A, "batter_home_runs")] = HomeRunInputs(
        batter_hr_per_pa_season=0.062,
        batter_pa_season=520,
        batter_barrel_pct=0.17,
        batter_xslg=0.580,
        batter_lineup_spot=3,
        pitcher_hr_per_9=1.4,
        pitcher_barrel_pct_allowed=0.09,
        park_hr_factor=108.0,
        weather_hr_multiplier=1.0,
    )

    registry[(PLAYER_E, "batter_total_bases")] = TotalBasesInputs(
        batter_tb_per_pa_season=0.64,
        batter_pa_season=520,
        batter_iso=0.270,
        batter_xslg=0.580,
        batter_lineup_spot=3,
        pitcher_hr_per_9=1.4,
        pitcher_hits_per_9=9.2,
        park_factor_1b=102.0,
        park_factor_2b=104.0,
        park_factor_3b=100.0,
        park_factor_hr=108.0,
    )

    registry[(PLAYER_B, "batter_singles")] = SinglesInputs(
        batter_1b_per_pa_season=0.150,
        batter_pa_season=480,
        batter_groundball_pct=0.44,
        batter_line_drive_pct=0.21,
        batter_sprint_speed=27.5,
        batter_xba=0.250,
        batter_iso=0.140,
        batter_lineup_spot=2,
        pitcher_groundball_pct_allowed=0.43,
        pitcher_hits_per_9=8.6,
    )

    registry[(PLAYER_B, "batter_hits")] = HitsInputs(
        batter_hit_per_pa_season=0.215,   # below league avg (~0.236)
        batter_pa_season=480,
        batter_xba=0.225,
        batter_hard_hit_pct=0.33,
        batter_lineup_spot=2,
        pitcher_whip=1.05,                 # tough, stingy pitcher
        pitcher_ba_allowed=0.215,
    )

    registry[(PLAYER_C, "pitcher_strikeouts")] = PitcherStrikeoutInputs(
        pitcher_k_per_9_season=11.2,
        pitcher_bf_season=420,
        pitcher_csw_pct=0.33,
        expected_innings=6.0,
        opponent_team_k_pct=0.26,
    )

    registry[(PLAYER_D, "batter_doubles")] = DoublesInputs(
        batter_2b_per_pa_season=0.044,
        batter_pa_season=400,
        batter_hard_hit_pct=0.37,
        batter_line_drive_pct=0.20,
        batter_fly_ball_pct=0.34,
        batter_sprint_speed=27.0,
        batter_xslg=0.395,
        batter_lineup_spot=5,
        pitcher_hits_per_9=8.4,
        park_2b_factor=100.0,
    )

    return registry


# ---------------------------------------------------------------------------
# 2 & 3. Edge sanity + dual-bookmaker selection
# ---------------------------------------------------------------------------

class TestEdgeAndDualBookmaker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.odds_df = build_synthetic_odds_df()
        cls.registry = build_synthetic_inputs_registry()
        cls.candidates = build_candidate_legs(cls.odds_df, cls.registry)
        cls.by_key = {(c.player, c.market): c for c in cls.candidates}

    def test_all_six_legs_scored(self):
        # 6 distinct (player, market) legs in the fixture; every one
        # should produce exactly one EdgeCandidate.
        self.assertEqual(len(self.candidates), 6)

    def test_edge_bounds_and_types(self):
        for c in self.candidates:
            self.assertIsInstance(c, EdgeCandidate)
            self.assertGreaterEqual(c.model_prob, 0.0)
            self.assertLessEqual(c.model_prob, 1.0)
            self.assertGreaterEqual(c.chosen_market_prob, 0.0)
            self.assertLessEqual(c.chosen_market_prob, 1.0)
            # edge must equal model_prob - chosen_market_prob exactly
            self.assertAlmostEqual(c.edge, c.model_prob - c.chosen_market_prob, places=10)

    def test_slugger_hr_leg_shows_strong_positive_edge(self):
        # An elite HR profile (xSLG .580, barrel% 17%, hitter-friendly
        # park) against a market priced around a de-vigged ~26-28%
        # produces model_prob ~30.9% vs chosen_market_prob ~26.4%, i.e. a
        # real, clearly-positive edge (~4.5 points) — this is the "model
        # disagrees with the market, in PropSync's favor" case the whole
        # layer exists to surface. (Verified against the actual run, not
        # an arbitrary round-number target — see module docstring above
        # for the exact inputs.)
        leg = self.by_key[(PLAYER_A, "batter_home_runs")]
        self.assertGreater(leg.edge, 0.03)
        self.assertLess(leg.edge, 0.10)  # sanity ceiling — not implausibly huge

    def test_mediocre_hits_leg_does_not_show_strong_positive_edge(self):
        # Below-average hit rate vs a stingy pitcher should NOT produce a
        # large positive edge the way the slugger's HR leg does — confirms
        # the model isn't just universally optimistic.
        leg = self.by_key[(PLAYER_B, "batter_hits")]
        self.assertLess(leg.edge, 0.05)

    def test_dual_book_picks_cheaper_price_strikeouts(self):
        # FanDuel Over 6.5 at -170 (expensive/high implied prob) vs
        # DraftKings Over 6.5 at -110 (cheaper/lower implied prob) on the
        # SAME line. The de-vigged fair_prob for -170-ish should be
        # meaningfully higher than for -110-ish, so the selector must pick
        # DraftKings as offering the more favorable price.
        leg = self.by_key[(PLAYER_C, "pitcher_strikeouts")]
        self.assertEqual(leg.chosen_book, "draftkings")
        # Confirm it actually compared both books, not just defaulted.
        self.assertEqual({bp.bookmaker for bp in leg.all_book_prices}, {"fanduel", "draftkings"})
        fd_price = next(bp for bp in leg.all_book_prices if bp.bookmaker == "fanduel")
        dk_price = next(bp for bp in leg.all_book_prices if bp.bookmaker == "draftkings")
        self.assertLess(dk_price.fair_prob, fd_price.fair_prob)
        self.assertEqual(leg.chosen_market_prob, dk_price.fair_prob)

    def test_dual_book_close_prices_doubles(self):
        # FanDuel +180/-220 vs DraftKings +175/-210 on the same doubles
        # line — close but not identical prices. The selector should still
        # pick whichever is numerically lower fair_prob (mechanical
        # correctness check, not a big-disagreement case like strikeouts).
        leg = self.by_key[(PLAYER_D, "batter_doubles")]
        prices_by_book = {bp.bookmaker: bp.fair_prob for bp in leg.all_book_prices}
        expected_book = min(prices_by_book, key=lambda k: prices_by_book[k])
        self.assertEqual(leg.chosen_book, expected_book)

    def test_single_book_leg_not_dropped(self):
        # Player B's singles leg only has FanDuel posted (no DraftKings
        # row in the fixture) — must still appear in candidates, using
        # FanDuel by default, not get silently dropped.
        leg = self.by_key[(PLAYER_B, "batter_singles")]
        self.assertEqual(leg.chosen_book, "fanduel")
        self.assertEqual(len(leg.all_book_prices), 1)

    def test_devig_method_two_way_when_both_sides_present(self):
        # Every fixture leg has both Over/Under (or Yes/No) rows for each
        # book that posted it, so every book price should use the
        # two-way devig method, not the single-side fallback.
        for c in self.candidates:
            for bp in c.all_book_prices:
                self.assertEqual(bp.devig_method, "two_way")


# ---------------------------------------------------------------------------
# 4. Top-N selection — no exclusion (same-player and same-game both allowed)
# ---------------------------------------------------------------------------

class TestParlaySelector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        odds_df = build_synthetic_odds_df()
        registry = build_synthetic_inputs_registry()
        candidates = build_candidate_legs(odds_df, registry)
        cls.ranked = rank_edges(candidates)

    def test_ranked_list_is_sorted_descending(self):
        edges = [c.edge for c in self.ranked]
        self.assertEqual(edges, sorted(edges, reverse=True))

    def test_allows_same_player_across_prop_types(self):
        # PLAYER_A (HR leg) and PLAYER_E (total bases leg) are literally
        # the same player name ("Slugger McGee") on two different prop
        # types. With n covering the whole ranked list, both of his legs
        # should appear — no same-player exclusion anymore.
        selected = select_top_n(self.ranked, n=10)
        slugger_legs = [c for c in selected if c.player == PLAYER_A]
        self.assertEqual(len(slugger_legs), 2)

    def test_allows_same_game_different_players(self):
        # PLAYER_C (strikeouts) and PLAYER_D (doubles) are different
        # players but in the SAME game (evt_2). Both should be selectable.
        selected = select_top_n(self.ranked, n=10)
        selected_players = {c.player for c in selected}
        self.assertIn(PLAYER_C, selected_players)
        self.assertIn(PLAYER_D, selected_players)

    def test_select_top_n_is_a_plain_slice_of_ranked_list(self):
        # With no exclusion logic, select_top_n(ranked, n) must be exactly
        # ranked[:n] — confirm it doesn't secretly skip or reorder anything.
        selected = select_top_n(self.ranked, n=3)
        self.assertEqual(selected, self.ranked[:3])

    def test_select_top_n_returns_fewer_if_n_exceeds_list_length(self):
        selected = select_top_n(self.ranked, n=500)
        self.assertEqual(selected, self.ranked)

    def test_select_top_n_zero_returns_empty(self):
        self.assertEqual(select_top_n(self.ranked, n=0), [])


# ---------------------------------------------------------------------------
# 5. End-to-end smoke test
# ---------------------------------------------------------------------------

class TestEndToEnd(unittest.TestCase):
    def test_full_pipeline_runs_and_produces_a_picks_list(self):
        odds_df = build_synthetic_odds_df()
        registry = build_synthetic_inputs_registry()

        candidates = build_candidate_legs(odds_df, registry)
        ranked = rank_edges(candidates)
        picks = select_top_n(ranked, n=3)

        self.assertGreater(len(candidates), 0)
        self.assertEqual(len(ranked), len(candidates))
        self.assertEqual(picks, ranked[:3])

    def test_missing_player_inputs_are_skipped_not_crashed(self):
        # Add a row for a player with NO entry in the inputs registry —
        # build_candidate_legs must skip it silently, not raise.
        odds_df = build_synthetic_odds_df()
        registry = build_synthetic_inputs_registry()

        extra_row = pd.DataFrame([dict(
            **EVENT_1, bookmaker="fanduel", market="batter_rbis",
            player="Nobody Has Inputs For Me", outcome_name="Over",
            line=0.5, price=-130,
        ), dict(
            **EVENT_1, bookmaker="fanduel", market="batter_rbis",
            player="Nobody Has Inputs For Me", outcome_name="Under",
            line=0.5, price=110,
        )])
        odds_df = pd.concat([odds_df, extra_row], ignore_index=True)

        candidates = build_candidate_legs(odds_df, registry)
        players_scored = {c.player for c in candidates}
        self.assertNotIn("Nobody Has Inputs For Me", players_scored)
        # Everything else still scored normally.
        self.assertEqual(len(candidates), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
