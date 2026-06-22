"""
Synthetic sample data for the PropSync dashboard.

Why this exists: neither the data pipeline (no Odds API key yet, pybaseball
never run live) nor a real lineup/odds feed exists yet — see
`../[C] data_pipeline/README.md` and `../[C] edge_ranking/README.md`. This
module fabricates a realistic-looking slate (real current MLB player names
for flavor, plausible stat values, internally consistent with what the real
pipeline/scoring/edge-ranking code expects) so the dashboard has something
to render today.

THIS IS NOT THE SEAM. `data_loader.py` is the seam — it decides whether to
call this module or the real pipeline. Nothing in here should be imported
directly by `app.py`; go through `data_loader.py` instead, so swapping to
live data later is a one-line change in one file.

Two outputs:
  1. `build_sample_raw_stats_df()` — one row per player, the Kasper-style
     raw-stats showcase (barrel%, hard-hit%, fly ball%, swinging-strike%,
     handedness splits, park factors split by hit type, pulled-air rate,
     etc). Shaped loosely like what `data_pipeline/statcast/savant.py` and
     `park_factors/park_factors.py` would return, flattened to one row per
     player for display convenience.
  2. `build_sample_edge_candidates()` — a list of `EdgeCandidate` objects
     (the real dataclass from `edge_ranking/ranking.py`, not a lookalike),
     covering all 7 prop types, built by running real `*Inputs` dataclasses
     through the real `score_<prop>_prop()` functions and real `devig.py` /
     `ranking.py` logic on fabricated odds rows. This means the picks view
     is exercising the ACTUAL scoring/ranking code end-to-end, just with
     made-up inputs instead of live ones — exactly the same "structured so
     it's a no-op swap later" idea the edge_ranking and scoring_model
     READMEs use for their own synthetic test fixtures.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

# Mirror the sys.path bootstrap convention used by edge_ranking/market_map.py
# and edge_ranking/tests/test_edge_ranking.py: add scoring_model/ and
# edge_ranking/ to sys.path by bare folder name so their bare `import x`
# statements resolve regardless of caller cwd. This is the dashboard's own
# copy of that pattern — dashboard code only ever IMPORTS from those
# packages, never edits them.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCORING_MODEL_DIR = _PROJECT_ROOT / "[C] scoring_model"
_EDGE_RANKING_DIR = _PROJECT_ROOT / "[C] edge_ranking"
for _p in (_SCORING_MODEL_DIR, _EDGE_RANKING_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from doubles import DoublesInputs  # noqa: E402
from hits import HitsInputs  # noqa: E402
from home_runs import HomeRunInputs  # noqa: E402
from pitcher_strikeouts import PitcherStrikeoutInputs  # noqa: E402
from rbis import RBIInputs  # noqa: E402
from singles import SinglesInputs  # noqa: E402
from total_bases import TotalBasesInputs  # noqa: E402

from ranking import EdgeCandidate, build_candidate_legs, rank_edges  # noqa: E402

# A fixed seed keeps the demo data stable across reruns within a session
# (Streamlit reruns the whole script on every interaction) without needing
# to cache it — same numbers every time the app loads, which is desirable
# for a labeled "sample data" demo, not a randomized one.
_RNG_SEED = 42


# ---------------------------------------------------------------------------
# Slate definition: real player names (flavor only), fabricated stat lines
# ---------------------------------------------------------------------------
# Each entry has enough fields to power BOTH the raw-stats showcase view and
# to construct every *Inputs dataclass the scoring model needs for that
# player's props. Pitchers get their own, smaller record shape.

_TEAMS_TODAY = [
    ("NYY", "BOS", "evt_001"),
    ("LAD", "SDP", "evt_002"),
    ("ATL", "PHI", "evt_003"),
    ("HOU", "TEX", "evt_004"),
]

_BATTERS = [
    # name, team, opp_team, event_id, lineup_spot, bats
    dict(name="Aaron Judge", team="NYY", opp="BOS", event_id="evt_001", lineup_spot=2, bats="R"),
    dict(name="Juan Soto", team="NYY", opp="BOS", event_id="evt_001", lineup_spot=3, bats="L"),
    dict(name="Rafael Devers", team="BOS", opp="NYY", event_id="evt_001", lineup_spot=3, bats="L"),
    dict(name="Triston Casas", team="BOS", opp="NYY", event_id="evt_001", lineup_spot=4, bats="L"),
    dict(name="Mookie Betts", team="LAD", opp="SDP", event_id="evt_002", lineup_spot=1, bats="R"),
    dict(name="Freddie Freeman", team="LAD", opp="SDP", event_id="evt_002", lineup_spot=3, bats="L"),
    dict(name="Fernando Tatis Jr.", team="SDP", opp="LAD", event_id="evt_002", lineup_spot=2, bats="R"),
    dict(name="Manny Machado", team="SDP", opp="LAD", event_id="evt_002", lineup_spot=4, bats="R"),
    dict(name="Ronald Acuna Jr.", team="ATL", opp="PHI", event_id="evt_003", lineup_spot=1, bats="R"),
    dict(name="Matt Olson", team="ATL", opp="PHI", event_id="evt_003", lineup_spot=3, bats="L"),
    dict(name="Bryce Harper", team="PHI", opp="ATL", event_id="evt_003", lineup_spot=2, bats="L"),
    dict(name="Kyle Schwarber", team="PHI", opp="ATL", event_id="evt_003", lineup_spot=1, bats="L"),
    dict(name="Yordan Alvarez", team="HOU", opp="TEX", event_id="evt_004", lineup_spot=3, bats="L"),
    dict(name="Jose Altuve", team="HOU", opp="TEX", event_id="evt_004", lineup_spot=1, bats="R"),
    dict(name="Corey Seager", team="TEX", opp="HOU", event_id="evt_004", lineup_spot=2, bats="L"),
    dict(name="Adolis Garcia", team="TEX", opp="HOU", event_id="evt_004", lineup_spot=4, bats="R"),
]

_PITCHERS = [
    dict(name="Gerrit Cole", team="NYY", opp="BOS", event_id="evt_001", throws="R"),
    dict(name="Tanner Houck", team="BOS", opp="NYY", event_id="evt_001", throws="R"),
    dict(name="Yoshinobu Yamamoto", team="LAD", opp="SDP", event_id="evt_002", throws="R"),
    dict(name="Dylan Cease", team="SDP", opp="LAD", event_id="evt_002", throws="R"),
    dict(name="Spencer Strider", team="ATL", opp="PHI", event_id="evt_003", throws="R"),
    dict(name="Zack Wheeler", team="PHI", opp="ATL", event_id="evt_003", throws="R"),
    dict(name="Framber Valdez", team="HOU", opp="TEX", event_id="evt_004", throws="L"),
    dict(name="Jacob deGrom", team="TEX", opp="HOU", event_id="evt_004", throws="R"),
]

# Per-team park factors, split by hit type (FanGraphs Guts index, 100 =
# neutral). Flavor values loosely modeled on real park reputations (Fenway
# inflates doubles via the Wall, Petco suppresses HRs, Coors-style parks
# aren't in this slate). Not pulled live -- see module docstring.
_PARK_FACTORS = {
    "NYY": dict(park="Yankee Stadium", pf_1b=98, pf_2b=101, pf_3b=85, pf_hr=112),
    "BOS": dict(park="Fenway Park", pf_1b=103, pf_2b=128, pf_3b=95, pf_hr=97),
    "LAD": dict(park="Dodger Stadium", pf_1b=97, pf_2b=98, pf_3b=90, pf_hr=104),
    "SDP": dict(park="Petco Park", pf_1b=99, pf_2b=102, pf_3b=110, pf_hr=89),
    "ATL": dict(park="Truist Park", pf_1b=100, pf_2b=103, pf_3b=98, pf_hr=103),
    "PHI": dict(park="Citizens Bank Park", pf_1b=99, pf_2b=99, pf_3b=88, pf_hr=109),
    "HOU": dict(park="Minute Maid Park", pf_1b=101, pf_2b=104, pf_3b=92, pf_hr=106),
    "TEX": dict(park="Globe Life Field", pf_1b=99, pf_2b=100, pf_3b=94, pf_hr=101),
}


def _seeded_random(name: str) -> random.Random:
    """Deterministic per-player RNG so each player's fabricated stat line
    is stable across reruns but still varies player-to-player."""
    return random.Random(f"{_RNG_SEED}:{name}")


def _fabricate_batter_profile(player: dict) -> dict:
    """
    Build one full raw-stats + rate-stat record for a batter: the
    Kasper-style showcase fields (barrel%, hard-hit%, fly ball%,
    swinging-strike%, handedness splits, pulled-air rate) plus the
    counting-rate fields the scoring model's *Inputs dataclasses need
    (HR/PA, hit/PA, etc).

    Values are hand-tuned to be plausible (a "true talent" power/contact
    archetype per player, then jittered with a seeded RNG) rather than
    pulled from any real season — this is fabricated data, not a leak of
    real 2026 stats.
    """
    rng = _seeded_random(player["name"])

    # Archetype tilts: every player gets a "power" and "contact" dial in
    # [0, 1] that drives the rest of the fabricated profile coherently
    # (a high-power player should also show high barrel%/xSLG/ISO, not
    # random independent numbers for each column).
    power = rng.uniform(0.45, 0.97)
    contact = rng.uniform(0.40, 0.95)
    speed = rng.uniform(0.30, 0.95)

    pa_season = round(rng.uniform(280, 420))
    pa_recent = round(rng.uniform(45, 70))

    barrel_pct = round(0.045 + 0.16 * power, 3)
    hard_hit_pct = round(0.30 + 0.22 * power, 3)
    fly_ball_pct = round(0.28 + 0.18 * power, 3)
    groundball_pct = round(0.50 - 0.22 * power + 0.05 * (1 - contact), 3)
    line_drive_pct = round(max(0.0, 1 - fly_ball_pct - groundball_pct - 0.07), 3)
    swstr_pct = round(0.14 - 0.05 * contact, 3)
    xslg = round(0.330 + 0.260 * power, 3)
    xba = round(0.215 + 0.075 * contact, 3)
    iso = round(0.110 + 0.230 * power, 3)
    sprint_speed = round(25.5 + 4.0 * speed, 1)
    pulled_air_rate = round(0.32 + 0.30 * power, 3)

    hr_per_pa = round(0.018 + 0.052 * power**1.5, 4)
    hit_per_pa = round(0.205 + 0.075 * contact, 4)
    # 1B / 2B / HR must not exceed total hit rate; derive shares.
    hr_share = hr_per_pa / hit_per_pa
    remaining = 1 - hr_share
    two_b_share = remaining * (0.16 + 0.10 * power)
    one_b_share = remaining - two_b_share
    single_per_pa = round(hit_per_pa * one_b_share, 4)
    double_per_pa = round(hit_per_pa * two_b_share, 4)
    tb_per_pa = round(single_per_pa + 2 * double_per_pa + hr_per_pa * 4 + (hit_per_pa - single_per_pa - double_per_pa - hr_per_pa) * 3, 4)
    rbi_per_pa = round(0.085 + 0.075 * power + 0.02 * contact, 4)

    # Recent-form rates: jitter the season rate +/-25% to simulate a hot or
    # cold trailing-15-day window, independent per stat.
    def _recent(rate: float, spread: float = 0.25) -> float:
        return round(max(0.0, rate * rng.uniform(1 - spread, 1 + spread)), 4)

    park = _PARK_FACTORS[player["team"]]

    return {
        **player,
        "park": park["park"],
        "park_factor_1b": park["pf_1b"],
        "park_factor_2b": park["pf_2b"],
        "park_factor_3b": park["pf_3b"],
        "park_factor_hr": park["pf_hr"],
        "pa_season": pa_season,
        "pa_recent_15d": pa_recent,
        "barrel_pct": barrel_pct,
        "hard_hit_pct": hard_hit_pct,
        "fly_ball_pct": fly_ball_pct,
        "groundball_pct": groundball_pct,
        "line_drive_pct": line_drive_pct,
        "swinging_strike_pct": swstr_pct,
        "xslg": xslg,
        "xba": xba,
        "iso": iso,
        "sprint_speed_ft_s": sprint_speed,
        "pulled_air_rate": pulled_air_rate,
        "hr_per_pa": hr_per_pa,
        "hit_per_pa": hit_per_pa,
        "single_per_pa": single_per_pa,
        "double_per_pa": double_per_pa,
        "tb_per_pa": tb_per_pa,
        "rbi_per_pa": rbi_per_pa,
        "hr_per_pa_recent": _recent(hr_per_pa, 0.40),
        "hit_per_pa_recent": _recent(hit_per_pa),
        "single_per_pa_recent": _recent(single_per_pa),
        "double_per_pa_recent": _recent(double_per_pa, 0.35),
        "tb_per_pa_recent": _recent(tb_per_pa),
        "rbi_per_pa_recent": _recent(rbi_per_pa, 0.35),
        # Handedness-split flavor: vs-same-hand rate is suppressed relative
        # to vs-opposite-hand, the standard platoon pattern, jittered.
        "wOBA_vs_L": round(0.300 + 0.090 * power * (0.85 if player["bats"] == "L" else 1.05), 3),
        "wOBA_vs_R": round(0.300 + 0.090 * power * (0.85 if player["bats"] == "R" else 1.05), 3),
    }


def _fabricate_pitcher_profile(pitcher: dict) -> dict:
    rng = _seeded_random(pitcher["name"])

    stuff = rng.uniform(0.40, 0.97)
    command = rng.uniform(0.40, 0.95)
    workload = rng.uniform(0.55, 0.95)

    bf_season = round(rng.uniform(380, 620))
    bf_recent = round(rng.uniform(60, 95))

    k_per_9 = round(7.0 + 5.5 * stuff, 2)
    hr_per_9 = round(1.45 - 0.55 * command, 2)
    whip = round(1.45 - 0.35 * command, 3)
    hits_per_9 = round(9.5 - 2.3 * command, 2)
    gb_pct_allowed = round(0.38 + 0.14 * (1 - stuff), 3)
    barrel_pct_allowed = round(0.105 - 0.045 * command, 3)
    csw_pct = round(0.26 + 0.10 * stuff, 3)
    expected_innings = round(4.7 + 1.6 * workload, 1)

    k_per_9_recent = round(max(4.0, k_per_9 * rng.uniform(0.75, 1.25)), 2)

    return {
        **pitcher,
        "park": _PARK_FACTORS[pitcher["team"]]["park"],
        "bf_season": bf_season,
        "bf_recent_15d": bf_recent,
        "k_per_9": k_per_9,
        "k_per_9_recent": k_per_9_recent,
        "hr_per_9": hr_per_9,
        "whip": whip,
        "hits_per_9": hits_per_9,
        "groundball_pct_allowed": gb_pct_allowed,
        "barrel_pct_allowed": barrel_pct_allowed,
        "csw_pct": csw_pct,
        "expected_innings": expected_innings,
        "k_pct_vs_L": round(0.205 + 0.09 * stuff * (1.05 if pitcher["throws"] == "L" else 0.92), 3),
        "k_pct_vs_R": round(0.205 + 0.09 * stuff * (1.05 if pitcher["throws"] == "R" else 0.92), 3),
    }


def build_sample_raw_stats_df() -> pd.DataFrame:
    """
    One row per batter, the Kasper-style raw-stats showcase table: barrel%,
    hard-hit%, fly ball%, swinging-strike% (pitcher matchup), handedness
    splits, park factors split by hit type, pulled-air rate, sprint speed,
    plus the underlying rate stats. This is what view #2 (raw stats
    showcase) renders.
    """
    rows = [_fabricate_batter_profile(p) for p in _BATTERS]
    return pd.DataFrame(rows)


def build_sample_pitcher_stats_df() -> pd.DataFrame:
    """One row per starting pitcher in tonight's slate, same showcase idea
    as build_sample_raw_stats_df() but for the pitcher-side metrics
    (K/9, CSW%, groundball% allowed, etc)."""
    rows = [_fabricate_pitcher_profile(p) for p in _PITCHERS]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Building *Inputs dataclasses + a fabricated odds DataFrame, then running
# the REAL edge_ranking pipeline against them.
# ---------------------------------------------------------------------------

def _pitcher_for_team(team: str) -> dict:
    for p in _PITCHERS:
        if p["team"] == team:
            return p
    raise KeyError(f"No pitcher fabricated for team {team}")


def _build_inputs_registry(
    batter_profiles: list[dict], pitcher_profiles: dict[str, dict]
) -> dict[tuple[str, str], object]:
    """
    Build the (player_name, market_key) -> *Inputs registry that
    `ranking.build_candidate_legs()` requires (see edge_ranking README
    "How this plugs into the scoring model"). This is real integration
    work the README explicitly says is NOT edge_ranking's job and not yet
    built anywhere else in the project — this sample-data module is
    standing in for that missing "build the registry from live data" step,
    using fabricated profiles instead of live pipeline pulls.
    """
    registry: dict[tuple[str, str], object] = {}

    for b in batter_profiles:
        opp_pitcher = pitcher_profiles[b["opp"]]

        registry[(b["name"], "batter_home_runs")] = HomeRunInputs(
            batter_hr_per_pa_season=b["hr_per_pa"],
            batter_pa_season=b["pa_season"],
            batter_hr_per_pa_recent=b["hr_per_pa_recent"],
            batter_pa_recent=b["pa_recent_15d"],
            batter_barrel_pct=b["barrel_pct"],
            batter_xslg=b["xslg"],
            batter_lineup_spot=b["lineup_spot"],
            pitcher_hr_per_9=opp_pitcher["hr_per_9"],
            pitcher_barrel_pct_allowed=opp_pitcher["barrel_pct_allowed"],
            park_hr_factor=b["park_factor_hr"],
            weather_hr_multiplier=1.0,
        )
        registry[(b["name"], "batter_hits")] = HitsInputs(
            batter_hit_per_pa_season=b["hit_per_pa"],
            batter_pa_season=b["pa_season"],
            batter_hit_per_pa_recent=b["hit_per_pa_recent"],
            batter_pa_recent=b["pa_recent_15d"],
            batter_xba=b["xba"],
            batter_hard_hit_pct=b["hard_hit_pct"],
            batter_lineup_spot=b["lineup_spot"],
            pitcher_whip=opp_pitcher["whip"],
        )
        registry[(b["name"], "batter_total_bases")] = TotalBasesInputs(
            batter_tb_per_pa_season=b["tb_per_pa"],
            batter_pa_season=b["pa_season"],
            batter_tb_per_pa_recent=b["tb_per_pa_recent"],
            batter_pa_recent=b["pa_recent_15d"],
            batter_iso=b["iso"],
            batter_xslg=b["xslg"],
            batter_lineup_spot=b["lineup_spot"],
            pitcher_hr_per_9=opp_pitcher["hr_per_9"],
            pitcher_hits_per_9=opp_pitcher["hits_per_9"],
            park_factor_1b=b["park_factor_1b"],
            park_factor_2b=b["park_factor_2b"],
            park_factor_3b=b["park_factor_3b"],
            park_factor_hr=b["park_factor_hr"],
        )
        registry[(b["name"], "batter_rbis")] = RBIInputs(
            batter_rbi_per_pa_season=b["rbi_per_pa"],
            batter_pa_season=b["pa_season"],
            batter_rbi_per_pa_recent=b["rbi_per_pa_recent"],
            batter_pa_recent=b["pa_recent_15d"],
            batter_lineup_spot=b["lineup_spot"],
            obp_of_hitters_ahead=0.330 if b["lineup_spot"] and b["lineup_spot"] <= 4 else 0.300,
            team_implied_run_total=4.7,
            pitcher_hits_per_9=opp_pitcher["hits_per_9"],
            pitcher_hr_per_9=opp_pitcher["hr_per_9"],
        )
        registry[(b["name"], "batter_singles")] = SinglesInputs(
            batter_1b_per_pa_season=b["single_per_pa"],
            batter_pa_season=b["pa_season"],
            batter_1b_per_pa_recent=b["single_per_pa_recent"],
            batter_pa_recent=b["pa_recent_15d"],
            batter_groundball_pct=b["groundball_pct"],
            batter_line_drive_pct=b["line_drive_pct"],
            batter_sprint_speed=b["sprint_speed_ft_s"],
            batter_xba=b["xba"],
            batter_iso=b["iso"],
            batter_lineup_spot=b["lineup_spot"],
            pitcher_groundball_pct_allowed=opp_pitcher["groundball_pct_allowed"],
            pitcher_hits_per_9=opp_pitcher["hits_per_9"],
        )
        registry[(b["name"], "batter_doubles")] = DoublesInputs(
            batter_2b_per_pa_season=b["double_per_pa"],
            batter_pa_season=b["pa_season"],
            batter_2b_per_pa_recent=b["double_per_pa_recent"],
            batter_pa_recent=b["pa_recent_15d"],
            batter_hard_hit_pct=b["hard_hit_pct"],
            batter_line_drive_pct=b["line_drive_pct"],
            batter_fly_ball_pct=b["fly_ball_pct"],
            batter_sprint_speed=b["sprint_speed_ft_s"],
            batter_xslg=b["xslg"],
            batter_lineup_spot=b["lineup_spot"],
            pitcher_hits_per_9=opp_pitcher["hits_per_9"],
            pitcher_barrel_pct_allowed=opp_pitcher["barrel_pct_allowed"],
            park_2b_factor=b["park_factor_2b"],
        )

    for p in pitcher_profiles.values():
        registry[(p["name"], "pitcher_strikeouts")] = PitcherStrikeoutInputs(
            pitcher_k_per_9_season=p["k_per_9"],
            pitcher_bf_season=p["bf_season"],
            pitcher_k_per_9_recent=p["k_per_9_recent"],
            pitcher_bf_recent=p["bf_recent_15d"],
            pitcher_csw_pct=p["csw_pct"],
            expected_innings=p["expected_innings"],
            opponent_team_k_pct=0.225,
        )

    return registry


# Plausible lines per market, used to fabricate odds rows. Mirrors what
# The Odds API actually posts for these 7 PropSync markets.
_LINES_BY_MARKET = {
    "batter_home_runs": None,  # yes/no, no numeric line
    "batter_hits": [0.5, 1.5],
    "batter_total_bases": [1.5, 2.5],
    "batter_rbis": [0.5, 1.5],
    "batter_singles": [0.5],
    "batter_doubles": [0.5],
    "pitcher_strikeouts": [5.5, 6.5, 7.5],
}


def _fabricate_price(rng: random.Random, model_prob: float) -> tuple[int, int]:
    """
    Fabricate a (modeled_side_price, opposing_side_price) American-odds
    pair that's roughly centered on the model's own probability (plus a
    deliberate book/model disagreement so the picks view actually shows a
    mix of positive and negative edges, not a wall of identical numbers),
    then add a standard ~4-5% vig on top so de-vig has something real to
    remove.
    """
    # Deliberately mis-price relative to the model some of the time, so the
    # ranked list shows a realistic distribution of edges rather than the
    # market always agreeing with the model.
    market_prob = max(0.04, min(0.93, model_prob + rng.uniform(-0.12, 0.08)))

    def _prob_to_fair_american(p: float) -> int:
        p = min(max(p, 0.02), 0.98)
        if p >= 0.5:
            return int(round(-100 * p / (1 - p)))
        return int(round(100 * (1 - p) / p))

    fair_price = _prob_to_fair_american(market_prob)
    # Apply vig by nudging both sides' raw implied prob up ~4.5% combined,
    # approximated here by shading the favorite further negative / dog
    # further positive, which is the standard vig direction.
    if fair_price < 0:
        modeled_price = int(round(fair_price * 1.06)) - 2
    else:
        modeled_price = int(round(fair_price * 0.94)) - 2
    opposing_fair = _prob_to_fair_american(1 - market_prob)
    if opposing_fair < 0:
        opposing_price = int(round(opposing_fair * 1.06)) - 2
    else:
        opposing_price = int(round(opposing_fair * 0.94)) - 2
    return modeled_price, opposing_price


def _fabricate_odds_rows(
    batter_profiles: list[dict],
    pitcher_profiles: dict[str, dict],
    registry: dict[tuple[str, str], object],
) -> list[dict]:
    """
    Fabricate odds rows shaped exactly like
    data_pipeline/odds/odds_api_client.py's get_all_player_props_today()
    output (event_id, commence_time, home_team, away_team, bookmaker,
    market, player, outcome_name, line, price), with both FanDuel and
    DraftKings posting slightly different (jittered) prices on most legs
    so the dual-bookmaker selection logic in ranking.py has something real
    to compare.
    """
    from market_map import modeled_outcome_name, opposing_outcome_name

    rows: list[dict] = []

    all_players = [
        (b["name"], b["team"], b["opp"], b["event_id"], "batter") for b in batter_profiles
    ] + [
        (p["name"], p["team"], p["opp"], p["event_id"], "pitcher") for p in pitcher_profiles.values()
    ]

    for name, team, opp, event_id, kind in all_players:
        markets = ["pitcher_strikeouts"] if kind == "pitcher" else [
            m for m in _LINES_BY_MARKET if m != "pitcher_strikeouts"
        ]
        home_team, away_team = None, None
        for h, a, eid in _TEAMS_TODAY:
            if eid == event_id:
                home_team, away_team = h, a
                break

        for market in markets:
            key = (name, market)
            if key not in registry:
                continue
            lines = _LINES_BY_MARKET[market]
            line_values = [None] if lines is None else lines
            for line in line_values:
                rng = _seeded_random(f"{name}:{market}:{line}")
                from scoring_bridge import score_for_row

                model_prob = score_for_row(name, market, line, registry)
                modeled_name = modeled_outcome_name(market)
                opposing_name = opposing_outcome_name(market)

                for bookmaker in ("fanduel", "draftkings"):
                    book_rng = random.Random(f"{_RNG_SEED}:{name}:{market}:{line}:{bookmaker}")
                    modeled_price, opposing_price = _fabricate_price(book_rng, model_prob)
                    rows.append(dict(
                        event_id=event_id, commence_time="2026-06-22T23:05:00Z",
                        home_team=home_team, away_team=away_team, bookmaker=bookmaker,
                        market=market, player=name, outcome_name=modeled_name,
                        line=line, price=modeled_price,
                    ))
                    rows.append(dict(
                        event_id=event_id, commence_time="2026-06-22T23:05:00Z",
                        home_team=home_team, away_team=away_team, bookmaker=bookmaker,
                        market=market, player=name, outcome_name=opposing_name,
                        line=line, price=opposing_price,
                    ))
    return rows


def build_sample_edge_candidates() -> list[EdgeCandidate]:
    """
    Full synthetic pipeline run: fabricate batter/pitcher profiles ->
    build *Inputs registry -> fabricate odds rows -> run the REAL
    `build_candidate_legs()` + `rank_edges()` from edge_ranking/ranking.py.

    Returns the actual `EdgeCandidate` dataclass instances, ranked by edge
    descending — exactly what the real pipeline would hand the dashboard
    once live data exists. See data_loader.py for the seam that swaps this
    out for a real call later.
    """
    batter_profiles = [_fabricate_batter_profile(b) for b in _BATTERS]
    pitcher_profiles = {p["team"]: _fabricate_pitcher_profile(p) for p in _PITCHERS}

    registry = _build_inputs_registry(batter_profiles, pitcher_profiles)
    odds_rows = _fabricate_odds_rows(batter_profiles, pitcher_profiles, registry)
    odds_df = pd.DataFrame(odds_rows)

    candidates = build_candidate_legs(odds_df, registry)
    return rank_edges(candidates)
