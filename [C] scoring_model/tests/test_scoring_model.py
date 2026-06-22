"""
Runnable tests for the PropSync scoring model, using realistic synthetic
fixtures (the real data pipeline hasn't pulled live data yet — see
data_pipeline/README.md). These tests check three things for every prop
type:

  1. Bounds: every probability is in [0.0, 1.0].
  2. Monotonicity: the model moves in the right direction for known
     better/worse inputs (e.g. higher barrel% -> higher HR probability).
  3. Shrinkage sanity: an extreme low-PA rate gets pulled toward league
     average rather than producing an overconfident probability spike.

Run with:
    cd "[C] scoring_model"
    python3 tests/test_scoring_model.py

(Uses stdlib unittest — no pytest dependency required, in keeping with
the project's zero-extra-deps stance for this layer.)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from doubles import DoublesInputs, score_doubles_prop
from expected_opportunities import expected_pa_for_lineup_spot
from hits import HitsInputs, score_hits_prop
from home_runs import HomeRunInputs, score_home_run_prop
from pitcher_strikeouts import PitcherStrikeoutInputs, score_pitcher_strikeouts_prop
from probability_utils import (
    blend_form,
    log5,
    poisson_cdf,
    poisson_pmf,
    prob_at_least_one,
    prob_over_line,
    shrink_rate,
)
from rbis import RBIInputs, score_rbi_prop
from singles import SinglesInputs, score_singles_prop
from total_bases import TotalBasesInputs, score_total_bases_prop


# ---------------------------------------------------------------------------
# Synthetic player archetypes
# ---------------------------------------------------------------------------

# High-barrel-rate slugger: big power, lots of PAs (stable sample), strikes
# out a fair amount, average speed.
SLUGGER = dict(
    pa_season=550,
    hr_per_pa=0.058,      # well above league avg (~0.032)
    hit_per_pa=0.235,
    tb_per_pa=0.62,
    rbi_per_pa=0.15,
    single_per_pa=0.13,
    double_per_pa=0.05,
    barrel_pct=0.16,
    hard_hit_pct=0.48,
    xslg=0.560,
    xba=0.250,
    iso=0.260,
    groundball_pct=0.36,
    line_drive_pct=0.19,
    fly_ball_pct=0.45,
    sprint_speed=26.0,
    lineup_spot=3,
)

# Contact hitter: low power, high contact, good speed, lots of PAs.
CONTACT_HITTER = dict(
    pa_season=600,
    hr_per_pa=0.012,      # well below league avg
    hit_per_pa=0.275,     # above-average hit rate
    tb_per_pa=0.34,
    rbi_per_pa=0.09,
    single_per_pa=0.21,   # most hits are singles
    double_per_pa=0.04,
    barrel_pct=0.04,
    hard_hit_pct=0.32,
    xslg=0.380,
    xba=0.280,
    iso=0.110,
    groundball_pct=0.48,
    line_drive_pct=0.23,
    fly_ball_pct=0.29,
    sprint_speed=29.0,
    lineup_spot=2,
)

# Low-PA rookie callup with a hot 10-PA streak — should get shrunk hard.
HOT_STREAK_ROOKIE = dict(
    pa_season=10,
    hr_per_pa=0.30,       # 3 HRs in 10 PA — wildly unsustainable
    hit_per_pa=0.50,
    tb_per_pa=1.50,
    rbi_per_pa=0.40,
    single_per_pa=0.20,
    double_per_pa=0.20,
    barrel_pct=0.30,
    hard_hit_pct=0.55,
    xslg=0.700,
    xba=0.300,
    iso=0.400,
    groundball_pct=0.30,
    line_drive_pct=0.30,
    fly_ball_pct=0.40,
    sprint_speed=27.0,
    lineup_spot=6,
)

# Strikeout pitcher: high K/9, high CSW%, stable sample.
STRIKEOUT_PITCHER = dict(
    bf_season=650,
    k_per_9=11.5,
    hr_per_9=1.0,
    hits_per_9=7.2,
    whip=1.10,
    barrel_pct_allowed=0.06,
    groundball_pct_allowed=0.40,
    csw_pct=0.33,
)

# Contact-manager pitcher: low K/9, low CSW%, more hits/contact allowed.
CONTACT_MANAGER_PITCHER = dict(
    bf_season=650,
    k_per_9=6.0,
    hr_per_9=1.3,
    hits_per_9=9.5,
    whip=1.45,
    barrel_pct_allowed=0.09,
    groundball_pct_allowed=0.52,
    csw_pct=0.24,
)


class TestProbabilityUtils(unittest.TestCase):
    def test_poisson_pmf_sums_to_one(self):
        lam = 2.3
        total = sum(poisson_pmf(k, lam) for k in range(0, 50))
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_poisson_cdf_monotonic(self):
        lam = 1.8
        prev = 0.0
        for k in range(0, 10):
            cur = poisson_cdf(k, lam)
            self.assertGreaterEqual(cur, prev)
            prev = cur
        self.assertAlmostEqual(poisson_cdf(50, lam), 1.0, places=6)

    def test_prob_over_line_half_integer(self):
        # P(over 0.5) with lam=1.0 should equal P(count >= 1) = 1 - e^-1
        import math

        result = prob_over_line(1.0, 0.5)
        self.assertAlmostEqual(result, 1 - math.exp(-1.0), places=6)

    def test_prob_at_least_one_bounds(self):
        self.assertGreaterEqual(prob_at_least_one(0.0), 0.0)
        self.assertLessEqual(prob_at_least_one(10.0), 1.0)
        self.assertAlmostEqual(prob_at_least_one(0.0), 0.0, places=6)

    def test_shrinkage_pulls_toward_league_average(self):
        # 3 HR in 10 PA (0.30 rate) should be shrunk well below 0.30,
        # toward the league average of 0.032, given a stabilization point
        # of 150 PA.
        result = shrink_rate(
            observed_rate=0.30, sample_size=10, league_avg=0.032, stabilization_point=150
        )
        self.assertLess(result.shrunk_rate, 0.10)
        self.assertGreater(result.shrunk_rate, 0.032)
        # With n=10 and k=150, league average should dominate (weight > 0.9)
        self.assertGreater(result.shrinkage_weight, 0.9)

    def test_shrinkage_barely_moves_large_sample(self):
        # 0.058 HR/PA over 550 PA (a real, large sample) should barely move.
        result = shrink_rate(
            observed_rate=0.058, sample_size=550, league_avg=0.032, stabilization_point=150
        )
        self.assertAlmostEqual(result.shrunk_rate, 0.058, delta=0.01)

    def test_log5_returns_league_avg_for_two_average_inputs(self):
        result = log5(0.032, 0.032, 0.032)
        self.assertAlmostEqual(result, 0.032, places=6)

    def test_log5_above_average_matchup_exceeds_league_avg(self):
        # Good batter vs. bad-for-pitcher (allows lots) should be > league avg
        result = log5(0.06, 0.05, 0.032)
        self.assertGreater(result, 0.032)

    def test_blend_form_weights_recent_more_heavily(self):
        # Season rate flat at league average, recent rate hot — blended
        # rate should sit above season rate (recency tilt working) but
        # not equal the recent rate outright (season data still counts).
        blended, eff_n = blend_form(
            season_rate=0.032, season_n=400, recent_rate=0.08, recent_n=20
        )
        self.assertGreater(blended, 0.032)
        self.assertLess(blended, 0.08)
        self.assertEqual(eff_n, 420)


class TestExpectedOpportunities(unittest.TestCase):
    def test_lineup_spot_pa_descends(self):
        pa_1 = expected_pa_for_lineup_spot(1)
        pa_9 = expected_pa_for_lineup_spot(9)
        self.assertGreater(pa_1, pa_9)

    def test_unknown_lineup_spot_falls_back(self):
        result = expected_pa_for_lineup_spot(None)
        self.assertGreater(result, 0)


class TestHomeRunProp(unittest.TestCase):
    def _make_inputs(self, profile: dict, pitcher: dict) -> HomeRunInputs:
        return HomeRunInputs(
            batter_hr_per_pa_season=profile["hr_per_pa"],
            batter_pa_season=profile["pa_season"],
            batter_barrel_pct=profile["barrel_pct"],
            batter_xslg=profile["xslg"],
            batter_lineup_spot=profile["lineup_spot"],
            pitcher_hr_per_9=pitcher["hr_per_9"],
            pitcher_barrel_pct_allowed=pitcher["barrel_pct_allowed"],
            park_hr_factor=100.0,
        )

    def test_bounds(self):
        p = score_home_run_prop(self._make_inputs(SLUGGER, CONTACT_MANAGER_PITCHER))
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_slugger_beats_contact_hitter(self):
        slugger_p = score_home_run_prop(self._make_inputs(SLUGGER, CONTACT_MANAGER_PITCHER))
        contact_p = score_home_run_prop(self._make_inputs(CONTACT_HITTER, CONTACT_MANAGER_PITCHER))
        self.assertGreater(slugger_p, contact_p)

    def test_higher_barrel_pct_increases_probability(self):
        low_barrel = dict(SLUGGER, barrel_pct=0.08)
        high_barrel = dict(SLUGGER, barrel_pct=0.22)
        p_low = score_home_run_prop(self._make_inputs(low_barrel, CONTACT_MANAGER_PITCHER))
        p_high = score_home_run_prop(self._make_inputs(high_barrel, CONTACT_MANAGER_PITCHER))
        self.assertGreater(p_high, p_low)

    def test_facing_strikeout_pitcher_lowers_hr_probability_vs_contact_manager(self):
        # Strikeout pitcher here also allows fewer HR/9 than the contact
        # manager fixture, so the slugger's HR prob should be lower
        # against the strikeout pitcher.
        p_vs_k_pitcher = score_home_run_prop(self._make_inputs(SLUGGER, STRIKEOUT_PITCHER))
        p_vs_contact_mgr = score_home_run_prop(self._make_inputs(SLUGGER, CONTACT_MANAGER_PITCHER))
        self.assertLess(p_vs_k_pitcher, p_vs_contact_mgr)

    def test_park_factor_increases_probability(self):
        base_inputs = self._make_inputs(SLUGGER, CONTACT_MANAGER_PITCHER)
        hitter_park = HomeRunInputs(**{**base_inputs.__dict__, "park_hr_factor": 130.0})
        neutral_park = HomeRunInputs(**{**base_inputs.__dict__, "park_hr_factor": 100.0})
        self.assertGreater(score_home_run_prop(hitter_park), score_home_run_prop(neutral_park))

    def test_weather_multiplier_increases_probability(self):
        base_inputs = self._make_inputs(SLUGGER, CONTACT_MANAGER_PITCHER)
        wind_out = HomeRunInputs(**{**base_inputs.__dict__, "weather_hr_multiplier": 1.15})
        no_wind = HomeRunInputs(**{**base_inputs.__dict__, "weather_hr_multiplier": 1.0})
        self.assertGreater(score_home_run_prop(wind_out), score_home_run_prop(no_wind))

    def test_hot_streak_rookie_does_not_spike(self):
        rookie_p = score_home_run_prop(self._make_inputs(HOT_STREAK_ROOKIE, CONTACT_MANAGER_PITCHER))
        slugger_p = score_home_run_prop(self._make_inputs(SLUGGER, CONTACT_MANAGER_PITCHER))
        # A 10-PA, 30%-HR-rate rookie should NOT outrank an established
        # slugger with a large, stable sample — shrinkage should prevent
        # the small sample from dominating.
        self.assertLess(rookie_p, slugger_p)
        # And it shouldn't be absurdly high in absolute terms either.
        self.assertLess(rookie_p, 0.30)


class TestHitsProp(unittest.TestCase):
    def _make_inputs(self, profile: dict, pitcher: dict) -> HitsInputs:
        return HitsInputs(
            batter_hit_per_pa_season=profile["hit_per_pa"],
            batter_pa_season=profile["pa_season"],
            batter_xba=profile["xba"],
            batter_hard_hit_pct=profile["hard_hit_pct"],
            batter_lineup_spot=profile["lineup_spot"],
            pitcher_whip=pitcher["whip"],
        )

    def test_bounds(self):
        p = score_hits_prop(self._make_inputs(CONTACT_HITTER, STRIKEOUT_PITCHER), line=0.5)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_facing_high_whip_pitcher_increases_hit_probability(self):
        p_vs_contact_mgr = score_hits_prop(
            self._make_inputs(CONTACT_HITTER, CONTACT_MANAGER_PITCHER), line=0.5
        )
        p_vs_k_pitcher = score_hits_prop(self._make_inputs(CONTACT_HITTER, STRIKEOUT_PITCHER), line=0.5)
        self.assertGreater(p_vs_contact_mgr, p_vs_k_pitcher)

    def test_higher_line_lowers_probability(self):
        inputs = self._make_inputs(CONTACT_HITTER, CONTACT_MANAGER_PITCHER)
        p_05 = score_hits_prop(inputs, line=0.5)
        p_15 = score_hits_prop(inputs, line=1.5)
        self.assertGreater(p_05, p_15)

    def test_contact_hitter_beats_slugger_on_1plus_hits(self):
        # Per the notes, hits props favor contact rate over power — the
        # contact hitter fixture has a higher hit_per_pa than the slugger.
        p_contact = score_hits_prop(self._make_inputs(CONTACT_HITTER, CONTACT_MANAGER_PITCHER), line=0.5)
        p_slugger = score_hits_prop(self._make_inputs(SLUGGER, CONTACT_MANAGER_PITCHER), line=0.5)
        self.assertGreater(p_contact, p_slugger)


class TestTotalBasesProp(unittest.TestCase):
    def _make_inputs(self, profile: dict, pitcher: dict, **overrides) -> TotalBasesInputs:
        base = dict(
            batter_tb_per_pa_season=profile["tb_per_pa"],
            batter_pa_season=profile["pa_season"],
            batter_iso=profile["iso"],
            batter_xslg=profile["xslg"],
            batter_lineup_spot=profile["lineup_spot"],
            pitcher_hr_per_9=pitcher["hr_per_9"],
            pitcher_hits_per_9=pitcher["hits_per_9"],
        )
        base.update(overrides)
        return TotalBasesInputs(**base)

    def test_bounds(self):
        p = score_total_bases_prop(self._make_inputs(SLUGGER, CONTACT_MANAGER_PITCHER), line=1.5)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_slugger_beats_contact_hitter(self):
        p_slugger = score_total_bases_prop(self._make_inputs(SLUGGER, CONTACT_MANAGER_PITCHER), line=1.5)
        p_contact = score_total_bases_prop(self._make_inputs(CONTACT_HITTER, CONTACT_MANAGER_PITCHER), line=1.5)
        self.assertGreater(p_slugger, p_contact)

    def test_doubles_specific_park_factor_not_hr_factor(self):
        base = self._make_inputs(SLUGGER, CONTACT_MANAGER_PITCHER)
        hr_friendly_only = TotalBasesInputs(**{**base.__dict__, "park_factor_hr": 140.0, "park_factor_2b": 100.0})
        gap_friendly_only = TotalBasesInputs(**{**base.__dict__, "park_factor_hr": 100.0, "park_factor_2b": 140.0})
        neutral = TotalBasesInputs(**{**base.__dict__})
        p_hr_park = score_total_bases_prop(hr_friendly_only, line=1.5)
        p_gap_park = score_total_bases_prop(gap_friendly_only, line=1.5)
        p_neutral = score_total_bases_prop(neutral, line=1.5)
        # Both a HR-friendly park and a gap(2B)-friendly park should raise
        # total-bases probability above neutral, since both feed into the
        # composite TB park factor (confirms hit-type-specific factors are
        # actually wired in, not just the HR number).
        self.assertGreater(p_hr_park, p_neutral)
        self.assertGreater(p_gap_park, p_neutral)


class TestRBIProp(unittest.TestCase):
    def _make_inputs(self, profile: dict, pitcher: dict, **overrides) -> RBIInputs:
        base = dict(
            batter_rbi_per_pa_season=profile["rbi_per_pa"],
            batter_pa_season=profile["pa_season"],
            batter_lineup_spot=profile["lineup_spot"],
            pitcher_hits_per_9=pitcher["hits_per_9"],
            pitcher_hr_per_9=pitcher["hr_per_9"],
        )
        base.update(overrides)
        return RBIInputs(**base)

    def test_bounds(self):
        p = score_rbi_prop(self._make_inputs(SLUGGER, CONTACT_MANAGER_PITCHER), line=0.5)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_higher_team_run_total_increases_probability(self):
        low_total = self._make_inputs(SLUGGER, CONTACT_MANAGER_PITCHER, team_implied_run_total=3.0)
        high_total = self._make_inputs(SLUGGER, CONTACT_MANAGER_PITCHER, team_implied_run_total=6.5)
        self.assertGreater(score_rbi_prop(high_total, line=0.5), score_rbi_prop(low_total, line=0.5))

    def test_higher_obp_ahead_increases_probability(self):
        low_obp = self._make_inputs(SLUGGER, CONTACT_MANAGER_PITCHER, obp_of_hitters_ahead=0.280)
        high_obp = self._make_inputs(SLUGGER, CONTACT_MANAGER_PITCHER, obp_of_hitters_ahead=0.380)
        self.assertGreater(score_rbi_prop(high_obp, line=0.5), score_rbi_prop(low_obp, line=0.5))


class TestSinglesProp(unittest.TestCase):
    def _make_inputs(self, profile: dict, pitcher: dict) -> SinglesInputs:
        return SinglesInputs(
            batter_1b_per_pa_season=profile["single_per_pa"],
            batter_pa_season=profile["pa_season"],
            batter_groundball_pct=profile["groundball_pct"],
            batter_line_drive_pct=profile["line_drive_pct"],
            batter_sprint_speed=profile["sprint_speed"],
            batter_xba=profile["xba"],
            batter_iso=profile["iso"],
            batter_lineup_spot=profile["lineup_spot"],
            pitcher_groundball_pct_allowed=pitcher["groundball_pct_allowed"],
            pitcher_hits_per_9=pitcher["hits_per_9"],
        )

    def test_bounds(self):
        p = score_singles_prop(self._make_inputs(CONTACT_HITTER, CONTACT_MANAGER_PITCHER), line=0.5)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_contact_hitter_beats_slugger(self):
        # Per the notes, singles favor contact/speed over power.
        p_contact = score_singles_prop(self._make_inputs(CONTACT_HITTER, CONTACT_MANAGER_PITCHER), line=0.5)
        p_slugger = score_singles_prop(self._make_inputs(SLUGGER, CONTACT_MANAGER_PITCHER), line=0.5)
        self.assertGreater(p_contact, p_slugger)

    def test_facing_groundball_pitcher_increases_probability(self):
        high_gb_pitcher = dict(CONTACT_MANAGER_PITCHER, groundball_pct_allowed=0.58)
        low_gb_pitcher = dict(CONTACT_MANAGER_PITCHER, groundball_pct_allowed=0.36)
        p_high = score_singles_prop(self._make_inputs(CONTACT_HITTER, high_gb_pitcher), line=0.5)
        p_low = score_singles_prop(self._make_inputs(CONTACT_HITTER, low_gb_pitcher), line=0.5)
        self.assertGreater(p_high, p_low)


class TestDoublesProp(unittest.TestCase):
    def _make_inputs(self, profile: dict, pitcher: dict, **overrides) -> DoublesInputs:
        base = dict(
            batter_2b_per_pa_season=profile["double_per_pa"],
            batter_pa_season=profile["pa_season"],
            batter_hard_hit_pct=profile["hard_hit_pct"],
            batter_line_drive_pct=profile["line_drive_pct"],
            batter_fly_ball_pct=profile["fly_ball_pct"],
            batter_sprint_speed=profile["sprint_speed"],
            batter_xslg=profile["xslg"],
            batter_lineup_spot=profile["lineup_spot"],
            pitcher_hits_per_9=pitcher["hits_per_9"],
            pitcher_barrel_pct_allowed=pitcher["barrel_pct_allowed"],
        )
        base.update(overrides)
        return DoublesInputs(**base)

    def test_bounds(self):
        p = score_doubles_prop(self._make_inputs(SLUGGER, CONTACT_MANAGER_PITCHER), line=0.5)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_doubles_park_factor_increases_probability(self):
        base = self._make_inputs(SLUGGER, CONTACT_MANAGER_PITCHER)
        hitter_friendly = DoublesInputs(**{**base.__dict__, "park_2b_factor": 130.0})
        neutral = DoublesInputs(**{**base.__dict__, "park_2b_factor": 100.0})
        self.assertGreater(score_doubles_prop(hitter_friendly, line=0.5), score_doubles_prop(neutral, line=0.5))

    def test_higher_sprint_speed_increases_probability(self):
        slow = self._make_inputs(SLUGGER, CONTACT_MANAGER_PITCHER, batter_sprint_speed=24.0)
        fast = self._make_inputs(SLUGGER, CONTACT_MANAGER_PITCHER, batter_sprint_speed=30.0)
        self.assertGreater(score_doubles_prop(fast, line=0.5), score_doubles_prop(slow, line=0.5))


class TestPitcherStrikeoutsProp(unittest.TestCase):
    def _make_inputs(self, pitcher: dict, **overrides) -> PitcherStrikeoutInputs:
        base = dict(
            pitcher_k_per_9_season=pitcher["k_per_9"],
            pitcher_bf_season=pitcher["bf_season"],
            pitcher_csw_pct=pitcher["csw_pct"],
        )
        base.update(overrides)
        return PitcherStrikeoutInputs(**base)

    def test_bounds(self):
        p = score_pitcher_strikeouts_prop(self._make_inputs(STRIKEOUT_PITCHER), line=5.5)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_strikeout_pitcher_beats_contact_manager(self):
        p_k = score_pitcher_strikeouts_prop(self._make_inputs(STRIKEOUT_PITCHER), line=5.5)
        p_cm = score_pitcher_strikeouts_prop(self._make_inputs(CONTACT_MANAGER_PITCHER), line=5.5)
        self.assertGreater(p_k, p_cm)

    def test_more_expected_innings_increases_probability(self):
        short_outing = self._make_inputs(STRIKEOUT_PITCHER, expected_innings=4.0)
        long_outing = self._make_inputs(STRIKEOUT_PITCHER, expected_innings=7.0)
        p_short = score_pitcher_strikeouts_prop(short_outing, line=5.5)
        p_long = score_pitcher_strikeouts_prop(long_outing, line=5.5)
        self.assertGreater(p_long, p_short)

    def test_facing_high_k_lineup_increases_probability(self):
        low_k_lineup = self._make_inputs(STRIKEOUT_PITCHER, opponent_team_k_pct=0.18)
        high_k_lineup = self._make_inputs(STRIKEOUT_PITCHER, opponent_team_k_pct=0.30)
        p_low = score_pitcher_strikeouts_prop(low_k_lineup, line=5.5)
        p_high = score_pitcher_strikeouts_prop(high_k_lineup, line=5.5)
        self.assertGreater(p_high, p_low)

    def test_higher_line_lowers_probability(self):
        inputs = self._make_inputs(STRIKEOUT_PITCHER)
        p_55 = score_pitcher_strikeouts_prop(inputs, line=5.5)
        p_85 = score_pitcher_strikeouts_prop(inputs, line=8.5)
        self.assertGreater(p_55, p_85)


if __name__ == "__main__":
    unittest.main(verbosity=2)
