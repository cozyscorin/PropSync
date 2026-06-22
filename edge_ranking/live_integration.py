"""
Live data integration layer — the missing link between data_pipeline's raw
stat pulls and edge_ranking's build_candidate_legs().

WHY THIS FILE EXISTS: every previous README in this project flagged the
same hole. scoring_model needs *Inputs dataclasses. edge_ranking's
scoring_bridge.py needs an `inputs_registry: dict[(player, market_key),
*Inputs]` to call build_candidate_legs(odds_df, inputs_registry). Nothing
anywhere built that registry from real pipeline DataFrames — sample_data.py
in the dashboard built a hand-fabricated stand-in, and every README pointed
at sample_data._build_inputs_registry() as "a worked example of the shape
needed," not the real thing. This module is the real thing.

Two hard problems live here, solved separately and explicitly:

1. PLAYER MATCHING (see `player_matching.py`-equivalent section below,
   `PlayerMatcher` class). The odds feed keys players by a display-name
   string (The Odds API outcome `description`, e.g. "Mike Trout"). The
   data pipeline's raw Statcast pulls key players by numeric MLBAM ID
   (the `batter`/`pitcher` columns in get_raw_statcast()'s output).
   FanGraphs leaderboard pulls (get_batter_season_stats() /
   get_pitcher_season_stats()) key by a `Name`/`PlayerName` display string
   — NOT a numeric ID, and not guaranteed to be formatted identically to
   the odds feed's string. There is no single shared key across all three
   sources today. This module builds a name-normalization layer as the
   primary matching path (since that's what every pipeline source the
   pipeline actually calls today provides), with an ID-based path ready to
   activate the moment a name<->MLBAM-ID lookup table is supplied (e.g.
   from pybaseball.playerid_lookup(), or a cached chadwick register dump).
   See `PlayerMatcher`'s docstring for exactly what is and isn't handled,
   and the known failure modes — this is NOT bulletproof and the
   docstring says so plainly.

2. PER-PROP FIELD MAPPING (the `build_*_inputs` functions, one per
   *Inputs dataclass). Each function takes whatever raw pipeline
   DataFrames are available for one player + their game's opposing
   pitcher, and produces exactly the dataclass scoring_model needs,
   following the field-by-field mapping documented in the scoring_model
   README's "Exactly how this plugs into the data pipeline" table. Where
   the pipeline has a real documented gap (lineup spot, team implied run
   total, expected innings, weather), this uses the SAME placeholder
   approach scoring_model already uses internally (None / fixed
   defaults), not invented precision.

Where this sits: edge_ranking, not data_pipeline and not scoring_model.
This package already has the sys.path bootstrap pattern to reach
scoring_model (see market_map.py) — this module reuses that, and adds the
same pattern for data_pipeline, since this is the one place in the
project that needs to import from all three layers at once.
"""
from __future__ import annotations

import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# --- sys.path bootstrap, mirroring market_map.py's existing convention ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCORING_MODEL_DIR = _PROJECT_ROOT / "scoring_model"
_DATA_PIPELINE_DIR = _PROJECT_ROOT / "data_pipeline"
for _p in (_SCORING_MODEL_DIR, _DATA_PIPELINE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from doubles import DoublesInputs  # noqa: E402
from hits import HitsInputs  # noqa: E402
from home_runs import HomeRunInputs  # noqa: E402
from pitcher_strikeouts import PitcherStrikeoutInputs  # noqa: E402
from rbis import RBIInputs  # noqa: E402
from singles import SinglesInputs  # noqa: E402
from total_bases import TotalBasesInputs  # noqa: E402

from market_map import PROP_TYPES  # noqa: E402

# ---------------------------------------------------------------------------
# Documented placeholder / fixed-estimate constants.
#
# These mirror the SAME judgment calls scoring_model already documents and
# uses internally (expected_opportunities.py, league_constants.py) — this
# module does not invent new fake precision, it reuses the same numbers so
# a missing pipeline field degrades to exactly the behavior scoring_model's
# own README already describes as its current limitation.
# ---------------------------------------------------------------------------

# RBIInputs.obp_of_hitters_ahead / team_implied_run_total: the pipeline has
# no lineup feed and no Vegas team-totals puller (PLAYER_PROP_MARKETS in
# data_pipeline/config.py doesn't include a team-totals market). Per the
# scoring_model README, these are real, undirected gaps, not a column-name
# issue. Leaving both as None here makes rbis.py fall back to its own
# neutral multiplier (1.0) for each — see rbis.py's
# _on_base_ahead_multiplier / _team_run_total_multiplier.
DEFAULT_OBP_OF_HITTERS_AHEAD = None
DEFAULT_TEAM_IMPLIED_RUN_TOTAL = None

# PitcherStrikeoutInputs.expected_innings: no workload/pitch-count feed in
# the pipeline (flagged in scoring_model README judgment call #7 as the
# single biggest open uncertainty for this prop). Leave None so
# pitcher_strikeouts.py's own DEFAULT_EXPECTED_INNINGS (5.5) fallback
# applies, rather than guessing a number here that looks more precise than
# it is.
DEFAULT_EXPECTED_INNINGS = None

# *Inputs.batter_lineup_spot: no lineup feed in the pipeline. Leave None so
# expected_opportunities.expected_pa_for_lineup_spot()'s own
# DEFAULT_EXPECTED_PA (4.1) fallback applies.
DEFAULT_LINEUP_SPOT = None

# HomeRunInputs.weather_hr_multiplier: no weather source in the pipeline
# (explicitly flagged as out of scope in data_pipeline README). Always 1.0
# (neutral) until a weather feed exists.
DEFAULT_WEATHER_HR_MULTIPLIER = 1.0

# opponent_team_k_pct (pitcher_strikeouts): the pipeline doesn't have a
# dedicated team-K%-as-batters aggregate function. If batter season stats
# are available for the full league, this module computes a real
# team-level aggregate (see _compute_team_k_pct below) rather than using a
# placeholder — that one IS buildable from data the pipeline already
# pulls (get_batter_season_stats() has K% per batter; group by Team). Only
# falls back to None (-> scoring_model's own league-average neutral
# fallback) if batter season stats aren't supplied at all.

# Pitcher-side rate defaults used only when a specific opposing pitcher
# can't be resolved at all (see _resolve_opposing_pitcher) — these mirror
# the *Inputs dataclasses' own dataclass-level defaults (league-average-
# ish), not new numbers invented here.
_FALLBACK_PITCHER_HR_PER_9 = 1.2
_FALLBACK_PITCHER_WHIP = 1.30
_FALLBACK_PITCHER_HITS_PER_9 = 8.5


# ---------------------------------------------------------------------------
# 1. PLAYER MATCHING
# ---------------------------------------------------------------------------

def normalize_name(raw_name: str) -> str:
    """
    Normalize a player display name into a matching key.

    Steps (in order):
      1. Strip leading/trailing whitespace, collapse internal whitespace.
      2. Remove accents/diacritics via Unicode NFKD decomposition + ASCII
         filtering (e.g. "Julio Rodríguez" -> "Julio Rodriguez", "José
         Ramírez" -> "Jose Ramirez"). Uses only Python's stdlib
         `unicodedata` — no extra dependency (e.g. `unidecode`) needed for
         the accented-Latin-character case, which covers the large
         majority of MLB roster names (Spanish/Portuguese accented
         vowels, Japanese/Korean names already romanized into plain
         ASCII by the sources PropSync uses). Does NOT handle names in a
         non-Latin native script — not a concern here since both The Odds
         API and FanGraphs/Savant already publish romanized names.
      3. Lowercase everything.
      4. Strip common generational suffixes (Jr., Sr., II, III, IV) since
         one source may include them and another may not (e.g. odds feed
         "Fernando Tatis Jr." vs. a leaderboard that drops the suffix).
      5. Strip periods and commas (handles "Jr." -> "jr" cleanly after
         suffix stripping, and stray punctuation like "A.J. Puk").
      6. Collapse to single spaces.

    This is NOT a fuzzy matcher — it's a deterministic canonicalization.
    Two names that normalize to the same string are treated as the same
    player; two names that don't, aren't matched, full stop. No edit
    distance, no nickname table (see KNOWN FAILURE MODES below for why
    that's a deliberate scope cut, not an oversight).
    """
    if raw_name is None:
        return ""
    name = str(raw_name).strip()
    name = " ".join(name.split())  # collapse internal whitespace

    # Remove accents: decompose to base char + combining marks, drop the
    # combining marks, keep the base ASCII letter.
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))

    name = name.lower()

    # Strip generational suffixes as whole tokens (case-insensitive,
    # already lowercased). Handles trailing "jr.", "jr", "sr.", "sr",
    # "ii", "iii", "iv" preceded by a space.
    suffix_tokens = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv"}
    tokens = name.replace(",", " ").split()
    tokens = [t for t in tokens if t not in suffix_tokens]
    name = " ".join(tokens)

    # Strip remaining periods (e.g. "a.j. puk" -> "aj puk").
    name = name.replace(".", "")
    name = " ".join(name.split())
    return name


@dataclass(frozen=True)
class PlayerMatchResult:
    """One resolved match from an odds-feed display name to a pipeline
    player record, or a documented miss."""
    odds_name: str
    matched: bool
    pipeline_key: Any | None       # MLBAM ID if matched via ID path, else normalized name
    match_method: str              # "mlbam_id" | "normalized_name" | "unmatched"


class PlayerMatcher:
    """
    Resolves odds-feed display names to pipeline player records.

    KNOWN MATCHING STRATEGY (in priority order):

    1. ID path (preferred, NOT currently reachable with this pipeline's
       existing functions). If a `name_to_mlbam_id: dict[str, int]`
       mapping is supplied (e.g. built once via
       `pybaseball.playerid_lookup()` and cached), names are normalized
       and looked up against that table first, and the registry keys off
       the numeric MLBAM ID. This is the right long-term path because
       MLBAM IDs are the only key every pipeline source could in
       principle share (raw Statcast pulls already use them natively as
       `batter`/`pitcher` columns) — but `statcast/savant.py` does not
       currently build or expose a name<->ID table anywhere, and
       `get_batter_season_stats()`/`get_pitcher_season_stats()` (the
       FanGraphs JSON API wrappers actually used for season rate stats)
       return `Name`/`PlayerName` strings, not MLBAM IDs, in their raw
       response. So today, this path is wired up and ready, but will
       only activate once an ID table is actually supplied by the
       caller — it is NOT a guess about a column that doesn't exist.

    2. Normalized-name path (the actual fallback used today, because it's
       the only thing every current pipeline source supports). Both the
       odds feed's `player` strings and the pipeline's `Name`-keyed
       DataFrames are run through `normalize_name()` and matched on exact
       string equality of the normalized form. This is what
       `build_inputs_registry()` actually uses end to end right now,
       since no ID table exists in the pipeline yet.

    KNOWN FAILURE MODES (documented, not hidden):

    - **Nicknames / shortened first names.** "Mike Trout" vs. a source
      that lists "Michael Trout" will NOT match — there's no nickname
      table. FanGraphs and the Odds API have historically been
      consistent about using common names ("Mike Trout," not "Michael"),
      so this is a lower-risk case in practice, but it is not handled.
    - **Same normalized name, different real players.** Two different
      active MLB players sharing a normalized full name (rare, but real
      — MLB has had this happen) would silently collide in this scheme
      and one would overwrite the other in the registry dict. The ID
      path is the only real fix; the name path cannot disambiguate this.
    - **Suffix ambiguity.** Stripping Jr./Sr./II/III/IV means a son and
      father who are both active with the same name (essentially never
      happens at the MLB level simultaneously, but worth naming) would
      also collide.
    - **Mid-name punctuation/spacing variants not covered by the suffix/
      period rules** (e.g. hyphenated surnames inconsistently hyphenated
      across sources) are not specifically handled beyond generic
      whitespace collapsing — a name like "Ha-Seong Kim" vs "Ha Seong
      Kim" would NOT match today.
    - **Traded players / multi-team rows.** Not a name-matching issue per
      se, but worth flagging here: if a player appears twice in a season
      leaderboard pull (post-trade stints split by team in some FanGraphs
      views), this module's `_first_match` helper takes the first row by
      DataFrame order — not necessarily the most recent team. Pipeline
      output should be deduplicated/aggregated upstream if this becomes a
      real issue; not handled here.
    - **The odds feed name and the pipeline name both being wrong/stale**
      relative to each other in some unanticipated way the first live run
      surfaces. This has never been tested against real odds-feed strings
      (no Odds API key has been used yet — see data_pipeline README) or
      real FanGraphs strings (no live pybaseball pull yet either). The
      normalization rules above are written defensively based on known
      common cases (accents, Jr./Sr./III, periods), but the very first
      live run is highly likely to surface at least one unanticipated
      formatting mismatch. Treat any new mismatch as "add one more
      normalization rule," not as a sign the approach is wrong.

    Bottom line: this is a best-effort deterministic name canonicalizer,
    not a guaranteed-correct identity resolution system. It will silently
    fail to match (not crash) on names it can't reconcile — those players
    just won't get registry entries and their odds rows get skipped by
    `build_candidate_legs()` (which already treats "no inputs for this
    player" as skip-not-crash, per scoring_bridge.py). Track
    `unmatched_odds_names` after building the registry to see exactly who
    got dropped and why, on every real run.
    """

    def __init__(self, name_to_mlbam_id: dict[str, int] | None = None):
        # Caller-supplied ID lookup table, pre-normalized for matching.
        # None by default since the pipeline doesn't build one yet (see
        # class docstring point 1).
        self._id_lookup: dict[str, int] = {}
        if name_to_mlbam_id:
            for name, mlbam_id in name_to_mlbam_id.items():
                self._id_lookup[normalize_name(name)] = mlbam_id

    def resolve(self, odds_name: str) -> PlayerMatchResult:
        normalized = normalize_name(odds_name)
        if not normalized:
            return PlayerMatchResult(odds_name, False, None, "unmatched")

        if normalized in self._id_lookup:
            return PlayerMatchResult(
                odds_name, True, self._id_lookup[normalized], "mlbam_id"
            )

        # Normalized-name path always "matches" at this layer (it returns
        # the canonical key); whether a pipeline DataFrame actually HAS a
        # row for that key is checked by the caller (build_*_inputs
        # functions / build_inputs_registry), not here. This class's job
        # is canonicalization, not existence-checking.
        return PlayerMatchResult(odds_name, True, normalized, "normalized_name")


def _series_lookup_by_name(df: pd.DataFrame, name_col: str, normalized_key: str) -> pd.Series | None:
    """
    Find the first row in `df` whose `name_col` value normalizes to
    `normalized_key`. Returns None if no row matches (caller decides how
    to handle a miss — usually means "use defaults for this field").
    """
    if df is None or df.empty or name_col not in df.columns:
        return None
    normalized_col = df[name_col].astype(str).map(normalize_name)
    matches = df[normalized_col == normalized_key]
    if matches.empty:
        return None
    return matches.iloc[0]


def _series_lookup_by_name_any_col(
    df: pd.DataFrame, name_cols: list[str], normalized_key: str
) -> pd.Series | None:
    """
    Same as `_series_lookup_by_name`, but tries each candidate name column
    in order and returns the first match found. Exists because a plain
    `a or b` chain doesn't work here — pandas raises `ValueError: The
    truth value of a Series is ambiguous` the moment the first call
    returns a real (non-None) Series, since Python's `or` evaluates the
    left operand's truthiness. This helper does the "try the next
    candidate only if the previous one was a real miss" logic explicitly
    instead, the way `_find_first_present` already does for column values.
    """
    for name_col in name_cols:
        row = _series_lookup_by_name(df, name_col, normalized_key)
        if row is not None:
            return row
    return None


def _row_lookup_by_id(df: pd.DataFrame, id_col: str, mlbam_id: int) -> pd.Series | None:
    """Find the first row in `df` whose `id_col` equals `mlbam_id`."""
    if df is None or df.empty or id_col not in df.columns:
        return None
    matches = df[df[id_col] == mlbam_id]
    if matches.empty:
        return None
    return matches.iloc[0]


def _safe_get(row: pd.Series | None, col: str, default=None):
    """pandas-NaN-safe field getter: missing row, missing column, or a
    real NaN value all collapse to `default` rather than propagating NaN
    into a dataclass field that downstream scoring_model code expects to
    be either a real float or None."""
    if row is None or col not in row.index:
        return default
    val = row[col]
    if pd.isna(val):
        return default
    return val


# ---------------------------------------------------------------------------
# 2. PER-PROP FIELD MAPPING
#
# Each function below follows the scoring_model README's "Exactly how
# this plugs into the data pipeline" table field-by-field. Column names
# marked there as "confirmed-expected" are used directly; column names
# marked "not explicitly confirmed" are tried via a short candidate list
# (mirroring park_factors.py's own _find_column() defensive pattern) so a
# single FanGraphs API rename doesn't take down the whole mapping.
# ---------------------------------------------------------------------------

def _find_first_present(row: pd.Series | None, candidates: list[str], default=None):
    """Try each candidate column name in order, return the first present,
    non-NaN value; default if none are present. Mirrors
    park_factors.py's _find_column() pattern, applied to a single row
    instead of a DataFrame, since FanGraphs API column-naming for some
    counting stats isn't confirmed yet (see data_pipeline + scoring_model
    READMEs, repeated "print .columns and check" guidance)."""
    if row is None:
        return default
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return default


def _per_pa_rate(count, pa, default=0.0) -> float:
    """count / PA, guarding against missing/zero PA."""
    if count is None or pa is None or pa == 0:
        return default
    try:
        return float(count) / float(pa)
    except (TypeError, ValueError, ZeroDivisionError):
        return default


def build_home_run_inputs(
    batter_row: pd.Series | None,
    batter_recent_row: pd.Series | None,
    pitcher_row: pd.Series | None,
    park_hr_factor: float,
    lineup_spot: int | None = None,
    weather_hr_multiplier: float = DEFAULT_WEATHER_HR_MULTIPLIER,
) -> HomeRunInputs:
    """
    HomeRunInputs from raw pipeline rows.

    batter_row: a row from get_batter_season_stats() for this player.
    batter_recent_row: a row from the rolling-form batter profile for the
        SAME player (batted-ball profile only — barrel_pct/hard_hit_pct;
        rolling_form.py does NOT currently return recent HR count/PA, see
        scoring_model README's explicitly flagged gap — so
        batter_hr_per_pa_recent stays None here, not faked).
    pitcher_row: a row from get_pitcher_season_stats() for the SPECIFIC
        opposing starting pitcher for this game (not a league average —
        caller is responsible for resolving which pitcher via
        _resolve_opposing_pitcher / the odds DataFrame's home/away team).
    park_hr_factor: from park_factors.get_hr_park_factor(), looked up for
        the batter's team/park for this game.
    """
    pa_season = _find_first_present(batter_row, ["PA"], default=0) or 0
    hr_count = _find_first_present(batter_row, ["HR"], default=0) or 0
    batter_hr_per_pa_season = _per_pa_rate(hr_count, pa_season, default=0.032)

    return HomeRunInputs(
        batter_hr_per_pa_season=batter_hr_per_pa_season,
        batter_pa_season=float(pa_season) if pa_season else 0.0,
        # Rolling form's batter_rolling_batted_ball_profile() does not
        # emit HR counts/PA — only batted-ball-profile rates (barrel%,
        # hard-hit%, FB/GB/LD%). Real gap, not a naming issue. See README.
        batter_hr_per_pa_recent=None,
        batter_pa_recent=None,
        batter_barrel_pct=_find_first_present(batter_row, ["Barrel%"]),
        batter_xslg=_find_first_present(batter_row, ["xSLG"]),
        batter_lineup_spot=lineup_spot,
        pitcher_hr_per_9=_find_first_present(
            pitcher_row, ["HR/9"], default=_FALLBACK_PITCHER_HR_PER_9
        ),
        pitcher_barrel_pct_allowed=_find_first_present(pitcher_row, ["Barrel%"]),
        park_hr_factor=park_hr_factor if park_hr_factor is not None else 100.0,
        weather_hr_multiplier=weather_hr_multiplier,
    )


def build_hits_inputs(
    batter_row: pd.Series | None,
    batter_recent_row: pd.Series | None,
    pitcher_row: pd.Series | None,
    lineup_spot: int | None = None,
    platoon_multiplier: float = 1.0,
) -> HitsInputs:
    """
    HitsInputs from raw pipeline rows. `pitcher_ba_allowed` is preferred
    over the WHIP approximation per hits.py's own docstring, but the
    pipeline doesn't expose opponent BA-allowed as its own column today
    (FanGraphs pitching leaderboards don't publish "BA against" directly
    in the columns this pipeline confirms) — so this always falls back to
    WHIP, same as hits.py's documented fallback path, never invents a BA-
    allowed number.
    """
    pa_season = _find_first_present(batter_row, ["PA"], default=0) or 0
    hit_count = _find_first_present(batter_row, ["H"], default=0) or 0
    batter_hit_per_pa_season = _per_pa_rate(hit_count, pa_season, default=0.236)

    return HitsInputs(
        batter_hit_per_pa_season=batter_hit_per_pa_season,
        batter_pa_season=float(pa_season) if pa_season else 0.0,
        batter_hit_per_pa_recent=None,  # same rolling-form gap as HR
        batter_pa_recent=None,
        batter_xba=_find_first_present(batter_row, ["xBA"]),
        batter_hard_hit_pct=_find_first_present(batter_row, ["HardHit%"]),
        batter_lineup_spot=lineup_spot,
        pitcher_whip=_find_first_present(
            pitcher_row, ["WHIP"], default=_FALLBACK_PITCHER_WHIP
        ),
        pitcher_ba_allowed=None,  # not exposed by the pipeline; see docstring
        platoon_multiplier=platoon_multiplier,
    )


def build_total_bases_inputs(
    batter_row: pd.Series | None,
    batter_recent_row: pd.Series | None,
    pitcher_row: pd.Series | None,
    park_factors_row: pd.Series | None,
    lineup_spot: int | None = None,
) -> TotalBasesInputs:
    """
    TotalBasesInputs from raw pipeline rows. `batter_tb_per_pa_season` is
    derived as TB / PA, where TB = 1B + 2*2B + 3*3B + 4*HR if those
    counting columns are present, else approximated from ISO (TB/PA ≈
    BA/PA + ISO, but since we may not have BA either, fall back to a
    league-average-anchored estimate via ISO alone when counting columns
    are missing).
    """
    pa_season = _find_first_present(batter_row, ["PA"], default=0) or 0
    tb_count = _find_first_present(batter_row, ["TB"], default=None)
    if tb_count is None:
        # Build TB from individual hit-type counts if TB itself isn't a
        # column (FanGraphs sometimes omits TB as its own column even
        # though it publishes the components).
        singles = _find_first_present(batter_row, ["1B"], default=0) or 0
        doubles = _find_first_present(batter_row, ["2B"], default=0) or 0
        triples = _find_first_present(batter_row, ["3B"], default=0) or 0
        hr = _find_first_present(batter_row, ["HR"], default=0) or 0
        tb_count = singles + 2 * doubles + 3 * triples + 4 * hr

    batter_tb_per_pa_season = _per_pa_rate(tb_count, pa_season, default=0.405)

    pf_1b = _find_first_present(park_factors_row, ["1B"], default=100.0)
    pf_2b = _find_first_present(park_factors_row, ["2B"], default=100.0)
    pf_3b = _find_first_present(park_factors_row, ["3B"], default=100.0)
    pf_hr = _find_first_present(park_factors_row, ["HR"], default=100.0)

    return TotalBasesInputs(
        batter_tb_per_pa_season=batter_tb_per_pa_season,
        batter_pa_season=float(pa_season) if pa_season else 0.0,
        batter_tb_per_pa_recent=None,  # rolling-form gap, same as HR/hits
        batter_pa_recent=None,
        batter_iso=_find_first_present(batter_row, ["ISO"]),
        batter_xslg=_find_first_present(batter_row, ["xSLG"]),
        batter_lineup_spot=lineup_spot,
        pitcher_hr_per_9=_find_first_present(
            pitcher_row, ["HR/9"], default=1.2
        ),
        pitcher_hits_per_9=_find_first_present(
            pitcher_row, ["H/9"], default=_FALLBACK_PITCHER_HITS_PER_9
        ),
        park_factor_1b=float(pf_1b) if pf_1b is not None else 100.0,
        park_factor_2b=float(pf_2b) if pf_2b is not None else 100.0,
        park_factor_3b=float(pf_3b) if pf_3b is not None else 100.0,
        park_factor_hr=float(pf_hr) if pf_hr is not None else 100.0,
    )


def build_rbi_inputs(
    batter_row: pd.Series | None,
    batter_recent_row: pd.Series | None,
    pitcher_row: pd.Series | None,
    lineup_spot: int | None = None,
    obp_of_hitters_ahead: float | None = DEFAULT_OBP_OF_HITTERS_AHEAD,
    team_implied_run_total: float | None = DEFAULT_TEAM_IMPLIED_RUN_TOTAL,
) -> RBIInputs:
    """
    RBIInputs from raw pipeline rows. `obp_of_hitters_ahead` and
    `team_implied_run_total` are real, undirected pipeline gaps (no
    lineup feed, no Vegas team-totals puller) — both default to None
    here unless the caller explicitly supplies them, which makes
    rbis.py's own neutral (1.0) multiplier fallbacks apply rather than
    this module guessing a number.
    """
    pa_season = _find_first_present(batter_row, ["PA"], default=0) or 0
    rbi_count = _find_first_present(batter_row, ["RBI"], default=0) or 0
    batter_rbi_per_pa_season = _per_pa_rate(rbi_count, pa_season, default=0.115)

    return RBIInputs(
        batter_rbi_per_pa_season=batter_rbi_per_pa_season,
        batter_pa_season=float(pa_season) if pa_season else 0.0,
        batter_rbi_per_pa_recent=None,
        batter_pa_recent=None,
        batter_lineup_spot=lineup_spot,
        obp_of_hitters_ahead=obp_of_hitters_ahead,
        team_implied_run_total=team_implied_run_total,
        pitcher_hits_per_9=_find_first_present(
            pitcher_row, ["H/9"], default=_FALLBACK_PITCHER_HITS_PER_9
        ),
        pitcher_hr_per_9=_find_first_present(pitcher_row, ["HR/9"], default=1.2),
    )


def build_singles_inputs(
    batter_row: pd.Series | None,
    batter_recent_row: pd.Series | None,
    pitcher_row: pd.Series | None,
    sprint_speed_row: pd.Series | None,
    lineup_spot: int | None = None,
) -> SinglesInputs:
    """
    SinglesInputs from raw pipeline rows. Sprint speed comes from a
    SEPARATE pipeline call (get_sprint_speed()), not the season batting
    leaderboard — passed in as its own row since it's keyed by a
    different pull with its own (currently unconfirmed) column name.
    """
    pa_season = _find_first_present(batter_row, ["PA"], default=0) or 0
    singles_count = _find_first_present(batter_row, ["1B"], default=0) or 0
    batter_1b_per_pa_season = _per_pa_rate(singles_count, pa_season, default=0.145)

    sprint_speed = _find_first_present(
        sprint_speed_row, ["sprint_speed", "Sprint Speed (ft/sec)", "Speed"]
    )

    return SinglesInputs(
        batter_1b_per_pa_season=batter_1b_per_pa_season,
        batter_pa_season=float(pa_season) if pa_season else 0.0,
        batter_1b_per_pa_recent=None,
        batter_pa_recent=None,
        batter_groundball_pct=_find_first_present(batter_row, ["GB%"]),
        batter_line_drive_pct=_find_first_present(batter_row, ["LD%"]),
        batter_sprint_speed=sprint_speed,
        batter_xba=_find_first_present(batter_row, ["xBA"]),
        batter_iso=_find_first_present(batter_row, ["ISO"]),
        batter_lineup_spot=lineup_spot,
        pitcher_groundball_pct_allowed=_find_first_present(pitcher_row, ["GB%"]),
        pitcher_hits_per_9=_find_first_present(
            pitcher_row, ["H/9"], default=_FALLBACK_PITCHER_HITS_PER_9
        ),
    )


def build_doubles_inputs(
    batter_row: pd.Series | None,
    batter_recent_row: pd.Series | None,
    pitcher_row: pd.Series | None,
    sprint_speed_row: pd.Series | None,
    park_2b_factor: float,
    lineup_spot: int | None = None,
) -> DoublesInputs:
    """DoublesInputs from raw pipeline rows. `park_2b_factor` must come
    from park_factors.get_extra_base_park_factors()['doubles_park_factor']
    — NOT get_hr_park_factor() — per the explicit build instruction
    repeated in every layer's docs."""
    pa_season = _find_first_present(batter_row, ["PA"], default=0) or 0
    doubles_count = _find_first_present(batter_row, ["2B"], default=0) or 0
    batter_2b_per_pa_season = _per_pa_rate(doubles_count, pa_season, default=0.045)

    sprint_speed = _find_first_present(
        sprint_speed_row, ["sprint_speed", "Sprint Speed (ft/sec)", "Speed"]
    )

    return DoublesInputs(
        batter_2b_per_pa_season=batter_2b_per_pa_season,
        batter_pa_season=float(pa_season) if pa_season else 0.0,
        batter_2b_per_pa_recent=None,
        batter_pa_recent=None,
        batter_hard_hit_pct=_find_first_present(batter_row, ["HardHit%"]),
        batter_line_drive_pct=_find_first_present(batter_row, ["LD%"]),
        batter_fly_ball_pct=_find_first_present(batter_row, ["FB%"]),
        batter_sprint_speed=sprint_speed,
        batter_xslg=_find_first_present(batter_row, ["xSLG"]),
        batter_lineup_spot=lineup_spot,
        pitcher_hits_per_9=_find_first_present(
            pitcher_row, ["H/9"], default=_FALLBACK_PITCHER_HITS_PER_9
        ),
        pitcher_barrel_pct_allowed=_find_first_present(pitcher_row, ["Barrel%"]),
        park_2b_factor=float(park_2b_factor) if park_2b_factor is not None else 100.0,
    )


def _compute_team_k_pct(batter_season_df: pd.DataFrame | None, team: str | None) -> float | None:
    """
    Team-level K% as batters, aggregated from get_batter_season_stats().
    Unlike the other documented gaps, this one IS buildable from data the
    pipeline already pulls — FanGraphs' batting leaderboard has a K%
    column per batter; group by Team and take a PA-weighted average.
    Returns None (-> scoring_model's own neutral fallback) if the team
    can't be resolved or the necessary columns aren't present, rather
    than raising.
    """
    if batter_season_df is None or batter_season_df.empty or not team:
        return None
    team_col = "Team" if "Team" in batter_season_df.columns else None
    k_col = "K%" if "K%" in batter_season_df.columns else None
    pa_col = "PA" if "PA" in batter_season_df.columns else None
    if not (team_col and k_col and pa_col):
        return None

    team_rows = batter_season_df[batter_season_df[team_col] == team]
    if team_rows.empty:
        return None
    total_pa = team_rows[pa_col].sum()
    if not total_pa:
        return None
    weighted_k = (team_rows[k_col] * team_rows[pa_col]).sum()
    return float(weighted_k / total_pa)


def build_pitcher_strikeout_inputs(
    pitcher_row: pd.Series | None,
    pitcher_recent_row: pd.Series | None,
    opponent_team_k_pct: float | None,
    expected_innings: float | None = DEFAULT_EXPECTED_INNINGS,
) -> PitcherStrikeoutInputs:
    """
    PitcherStrikeoutInputs from raw pipeline rows. `expected_innings`
    stays None (-> scoring_model's own DEFAULT_EXPECTED_INNINGS fallback)
    unless explicitly supplied — no workload/pitch-count feed exists in
    the pipeline. `pitcher_recent_row` may carry CSW% from the rolling
    pitcher profile (pitcher_rolling_profile() DOES expose csw_pct, unlike
    the batter rolling profile's counting-stat gap) — prefer the rolling
    CSW% over season if both exist, since CSW% being a better predictor
    than K/9 is specifically why it's weighted as heavily as it is in
    pitcher_strikeouts.py.
    """
    bf_season = _find_first_present(pitcher_row, ["BF", "TBF"], default=0) or 0
    csw_pct = _find_first_present(pitcher_recent_row, ["csw_pct"])
    if csw_pct is None:
        csw_pct = _find_first_present(pitcher_row, ["CSW%"])

    bf_recent = _find_first_present(pitcher_recent_row, ["pitches_total"])
    # pitcher_rolling_profile() reports CSW% off pitches_total, not
    # batters faced — these aren't the same unit. There is no rolling
    # batters-faced count in the pipeline today, so pitcher_bf_recent
    # stays None (real gap) rather than silently treating pitch count as
    # a batters-faced count, which would be wrong, not just imprecise.
    pitcher_bf_recent = None
    pitcher_k_per_9_recent = None

    return PitcherStrikeoutInputs(
        pitcher_k_per_9_season=_find_first_present(pitcher_row, ["K/9"], default=8.5),
        pitcher_bf_season=float(bf_season) if bf_season else 0.0,
        pitcher_k_per_9_recent=pitcher_k_per_9_recent,
        pitcher_bf_recent=pitcher_bf_recent,
        pitcher_csw_pct=csw_pct,
        expected_innings=expected_innings,
        opponent_team_k_pct=opponent_team_k_pct,
        platoon_multiplier=1.0,
    )


# ---------------------------------------------------------------------------
# 3. THE MAIN ENTRY POINT: build_inputs_registry
# ---------------------------------------------------------------------------

def _resolve_team_park_factor(park_factors_df: pd.DataFrame, team: str, col_candidates: list[str], default=100.0):
    if park_factors_df is None or park_factors_df.empty or not team:
        return default
    team_col = None
    for c in ("Team", "team", "team_name"):
        if c in park_factors_df.columns:
            team_col = c
            break
    if team_col is None:
        return default
    rows = park_factors_df[park_factors_df[team_col] == team]
    if rows.empty:
        return default
    row = rows.iloc[0]
    for c in col_candidates:
        if c in row.index and pd.notna(row[c]):
            return row[c]
    return default


def build_inputs_registry(
    stat_dataframes: dict[str, Any],
    odds_df: pd.DataFrame,
    name_to_mlbam_id: dict[str, int] | None = None,
) -> dict[tuple[str, str], Any]:
    """
    THE function this module exists to provide. Builds the
    `(player, market_key) -> *Inputs` registry that
    `edge_ranking.ranking.build_candidate_legs()` requires, from raw
    data_pipeline DataFrames plus the day's odds.

    Args:
        stat_dataframes: a dict of raw pipeline outputs. Expected keys
            (all optional — missing keys degrade gracefully to fallback
            defaults rather than raising, since this is exactly the kind
            of partial-pipeline-failure scenario the live dashboard path
            needs to survive):
              - "batter_season": get_batter_season_stats() output
              - "pitcher_season": get_pitcher_season_stats() output
              - "batter_rolling_15d" / "batter_rolling_30d": rolling_form
                .batter_rolling_batted_ball_profile() output
              - "pitcher_rolling_15d" / "pitcher_rolling_30d": rolling_form
                .pitcher_rolling_profile() output
              - "park_factors": park_factors.get_park_factors() output
              - "extra_base_park_factors":
                park_factors.get_extra_base_park_factors() output
              - "sprint_speed": get_sprint_speed() output
              - "lineup_spots": OPTIONAL caller-supplied dict
                {player_name: int} if a lineup feed exists; the pipeline
                doesn't source this today (see module docstring) so it's
                fine for this key to be absent.
              - "team_implied_run_totals": OPTIONAL caller-supplied dict
                {team: float}; same gap, same handling.
        odds_df: today's odds DataFrame, shaped like
            odds_api_client.get_all_player_props_today()'s output —
            specifically needs the `player`, `market`, `home_team`,
            `away_team` columns to know which (player, market) pairs to
            build inputs for and which team each player belongs to (for
            park-factor/opposing-pitcher resolution).
        name_to_mlbam_id: optional pre-built name->MLBAM-ID lookup (e.g.
            from pybaseball.playerid_lookup()) for the ID-based matching
            path. None by default since the pipeline doesn't build this
            table yet — see PlayerMatcher's docstring.

    Returns:
        dict[(player_name_as_it_appears_in_odds_df, market_key)] -> the
        matching *Inputs dataclass instance for that player/market,
        ready to pass straight into
        edge_ranking.ranking.build_candidate_legs(odds_df, registry).
        Players/markets that couldn't be resolved (no matching pipeline
        row found) are simply absent from the registry — NOT an error —
        matching scoring_bridge.py's documented "skip, don't crash"
        contract for missing inputs.
    """
    batter_season = stat_dataframes.get("batter_season")
    pitcher_season = stat_dataframes.get("pitcher_season")
    batter_rolling_15d = stat_dataframes.get("batter_rolling_15d")
    pitcher_rolling_15d = stat_dataframes.get("pitcher_rolling_15d")
    park_factors_df = stat_dataframes.get("park_factors")
    extra_base_pf_df = stat_dataframes.get("extra_base_park_factors")
    sprint_speed_df = stat_dataframes.get("sprint_speed")
    lineup_spots = stat_dataframes.get("lineup_spots") or {}
    team_implied_run_totals = stat_dataframes.get("team_implied_run_totals") or {}

    matcher = PlayerMatcher(name_to_mlbam_id)

    required_cols = {"player", "market", "home_team", "away_team"}
    missing = required_cols - set(odds_df.columns)
    if missing:
        raise ValueError(
            f"odds_df is missing columns needed to build the inputs registry: "
            f"{sorted(missing)}"
        )

    # player -> team. Odds rows don't carry the player's own team directly
    # (only home_team/away_team for the game) — pipeline season-stat
    # DataFrames DO carry Team per player, so resolve team from there once
    # the player is matched, falling back to "unknown" (park factor
    # default applies) if the player can't be found in either leaderboard.
    def _player_team(player_name: str) -> str | None:
        for df in (batter_season, pitcher_season):
            row = _series_lookup_by_name_any_col(df, ["Name", "PlayerName"], normalize_name(player_name))
            if row is not None:
                for col in ("Team", "TeamName", "TeamNameAbb"):
                    if col in row.index and pd.notna(row[col]):
                        return row[col]
        return None

    def _opposing_team(player_team: str | None, home_team: str, away_team: str) -> str | None:
        if player_team is None:
            return None
        if player_team == home_team:
            return away_team
        if player_team == away_team:
            return home_team
        return None

    def _opposing_pitcher_row(opposing_team: str | None) -> pd.Series | None:
        """The SPECIFIC opposing starting pitcher's season stats for this
        game — not a league-wide pitcher average. Resolved by team: the
        pipeline doesn't have a probable-starters feed, so this takes the
        first pitcher row for the opposing team in the season leaderboard
        as a proxy. KNOWN LIMITATION: a team's leaderboard rows include
        every pitcher who appeared for them this season (starters AND
        relievers), not just today's probable starter — without a
        starters feed, this can pick a reliever's profile instead of the
        actual starter's. Flagged here, not hidden; the fix is a real
        probable-pitchers source (e.g. The Odds API's pitcher-specific
        markets often name the starter, or a dedicated lineups API),
        which doesn't exist in this pipeline yet."""
        if pitcher_season is None or pitcher_season.empty or not opposing_team:
            return None
        team_col = "Team" if "Team" in pitcher_season.columns else None
        if team_col is None:
            return None
        rows = pitcher_season[pitcher_season[team_col] == opposing_team]
        if rows.empty:
            return None
        return rows.iloc[0]

    registry: dict[tuple[str, str], Any] = {}
    unmatched_odds_names: set[str] = set()

    odds_players = odds_df[["player", "market", "home_team", "away_team"]].drop_duplicates()

    for _, odds_row in odds_players.iterrows():
        player_name = odds_row["player"]
        market = odds_row["market"]
        home_team = odds_row["home_team"]
        away_team = odds_row["away_team"]

        match = matcher.resolve(player_name)
        if not match.matched:
            unmatched_odds_names.add(player_name)
            continue

        if market not in PROP_TYPES:
            continue  # unknown market key, not this module's concern

        normalized_key = normalize_name(player_name)
        player_team = _player_team(player_name)
        opposing_team = _opposing_team(player_team, home_team, away_team)
        opposing_pitcher_row = _opposing_pitcher_row(opposing_team)
        lineup_spot = lineup_spots.get(player_name, DEFAULT_LINEUP_SPOT)

        if market == "pitcher_strikeouts":
            pitcher_row = _series_lookup_by_name_any_col(
                pitcher_season, ["Name", "PlayerName"], normalized_key
            )
            if pitcher_row is None:
                unmatched_odds_names.add(player_name)
                continue
            pitcher_recent_row = None
            if pitcher_rolling_15d is not None and not pitcher_rolling_15d.empty \
                    and name_to_mlbam_id and normalized_key in matcher._id_lookup:
                pitcher_recent_row = _row_lookup_by_id(
                    pitcher_rolling_15d, "pitcher", matcher._id_lookup[normalized_key]
                )
            opp_k_pct = _compute_team_k_pct(batter_season, opposing_team)
            registry[(player_name, market)] = build_pitcher_strikeout_inputs(
                pitcher_row, pitcher_recent_row, opp_k_pct
            )
            continue

        # Batting props from here down.
        batter_row = _series_lookup_by_name_any_col(
            batter_season, ["Name", "PlayerName"], normalized_key
        )
        if batter_row is None:
            unmatched_odds_names.add(player_name)
            continue

        batter_recent_row = None
        if batter_rolling_15d is not None and not batter_rolling_15d.empty \
                and name_to_mlbam_id and normalized_key in matcher._id_lookup:
            batter_recent_row = _row_lookup_by_id(
                batter_rolling_15d, "batter", matcher._id_lookup[normalized_key]
            )

        sprint_speed_row = (
            _series_lookup_by_name_any_col(sprint_speed_df, ["Name", "PlayerName"], normalized_key)
            if sprint_speed_df is not None else None
        )

        if market == "batter_home_runs":
            park_hr = _resolve_team_park_factor(park_factors_df, player_team, ["HR"])
            registry[(player_name, market)] = build_home_run_inputs(
                batter_row, batter_recent_row, opposing_pitcher_row, park_hr, lineup_spot
            )
        elif market == "batter_hits":
            registry[(player_name, market)] = build_hits_inputs(
                batter_row, batter_recent_row, opposing_pitcher_row, lineup_spot
            )
        elif market == "batter_total_bases":
            park_row = None
            if park_factors_df is not None and player_team:
                team_col = next((c for c in ("Team", "team") if c in park_factors_df.columns), None)
                if team_col:
                    rows = park_factors_df[park_factors_df[team_col] == player_team]
                    if not rows.empty:
                        park_row = rows.iloc[0]
            registry[(player_name, market)] = build_total_bases_inputs(
                batter_row, batter_recent_row, opposing_pitcher_row, park_row, lineup_spot
            )
        elif market == "batter_rbis":
            registry[(player_name, market)] = build_rbi_inputs(
                batter_row, batter_recent_row, opposing_pitcher_row, lineup_spot,
                team_implied_run_total=team_implied_run_totals.get(player_team),
            )
        elif market == "batter_singles":
            registry[(player_name, market)] = build_singles_inputs(
                batter_row, batter_recent_row, opposing_pitcher_row, sprint_speed_row, lineup_spot
            )
        elif market == "batter_doubles":
            park_2b = _resolve_team_park_factor(
                extra_base_pf_df, player_team, ["doubles_park_factor", "2B"]
            )
            registry[(player_name, market)] = build_doubles_inputs(
                batter_row, batter_recent_row, opposing_pitcher_row, sprint_speed_row,
                park_2b, lineup_spot,
            )

    if unmatched_odds_names:
        # Deliberately a print, not a raise: this is exactly the kind of
        # partial-failure information the dashboard path needs visible in
        # logs without crashing the run. See data_loader.py's live-path
        # try/except for where this surfaces in practice.
        print(
            f"[live_integration] {len(unmatched_odds_names)} odds-feed player "
            f"name(s) could not be matched to pipeline stat rows and were "
            f"skipped: {sorted(unmatched_odds_names)}"
        )

    return registry
