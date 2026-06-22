"""
The single mapping table between The Odds API's market keys (what shows
up in the odds pipeline's `market` column) and PropSync's 7 scoring_model
modules.

This is the one place that needs to change if a market key, an
`*Inputs` field name, or a scoring function name ever changes — every
other edge_ranking module imports this table rather than hardcoding any
of these strings itself.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# scoring_model's modules (home_runs.py, hits.py, etc.) are imported by
# bare module name throughout that package (see its own tests/
# test_scoring_model.py, which does the same sys.path trick) rather than
# as a proper installable package. Mirror that convention here instead of
# inventing a different import style for this layer: add scoring_model's
# folder to sys.path once, at import time, so every bare `import hits`
# etc. below resolves correctly regardless of the caller's cwd.
_SCORING_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "[C] scoring_model"
)
if str(_SCORING_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_SCORING_MODEL_DIR))

import doubles
import hits
import home_runs
import pitcher_strikeouts
import rbis
import singles
import total_bases

# ---------------------------------------------------------------------------
# Yes/No vs Over/Under markets
# ---------------------------------------------------------------------------
# Every PropSync market except batter_home_runs is a numeric over/under
# line (Over 1.5, Under 1.5, etc). batter_home_runs is posted by The Odds
# API as a yes/no market (outcome_name "Yes" / "No") keyed to "1+ HR",
# matching score_home_run_prop's P(1+ HR) output directly — there's no
# `line` parameter for that one function (see scoring_model/home_runs.py).
YES_NO_MARKETS = {"batter_home_runs"}

# The outcome_name value that represents the side PropSync's model scores
# directly. For over/under markets the model computes P(count > line),
# which corresponds to the "Over" outcome. For the HR yes/no market the
# model computes P(1+ HR), which corresponds to the "Yes" outcome.
# "Under" / "No" legs are intentionally NOT scored here — the scoring
# model doesn't expose a P(under)/P(no) function, and 1 - P(over) would
# silently assume a no-vig complement, which is exactly the kind of
# implicit-vig mistake this layer exists to avoid. If Under/No legs become
# wanted later, add an explicit score_<prop>_under() rather than deriving
# it from the Over output.
MODELED_OUTCOME_NAME = {
    "batter_home_runs": "Yes",
}
DEFAULT_MODELED_OUTCOME_NAME = "Over"


def modeled_outcome_name(market: str) -> str:
    return MODELED_OUTCOME_NAME.get(market, DEFAULT_MODELED_OUTCOME_NAME)


def opposing_outcome_name(market: str) -> str:
    """The other side of the same two-way market, used for de-vig pairing."""
    modeled = modeled_outcome_name(market)
    return "No" if modeled == "Yes" else "Under"


@dataclass(frozen=True)
class PropTypeSpec:
    market_key: str            # The Odds API market key, e.g. "batter_hits"
    display_name: str          # human-readable prop type label
    inputs_cls: type           # the *Inputs dataclass from scoring_model
    score_fn: Callable         # score_<prop>_prop callable
    is_yes_no: bool            # True only for batter_home_runs


def _score_home_runs(inputs, line: float | None) -> float:
    # score_home_run_prop takes no line parameter (yes/no market) — line
    # is accepted here only so every PropTypeSpec.score_fn has the same
    # call signature for the caller in scoring_bridge.py.
    return home_runs.score_home_run_prop(inputs)


PROP_TYPES: dict[str, PropTypeSpec] = {
    "batter_home_runs": PropTypeSpec(
        market_key="batter_home_runs",
        display_name="Home Run (1+)",
        inputs_cls=home_runs.HomeRunInputs,
        score_fn=_score_home_runs,
        is_yes_no=True,
    ),
    "batter_hits": PropTypeSpec(
        market_key="batter_hits",
        display_name="Hits",
        inputs_cls=hits.HitsInputs,
        score_fn=lambda inputs, line: hits.score_hits_prop(inputs, line),
        is_yes_no=False,
    ),
    "batter_total_bases": PropTypeSpec(
        market_key="batter_total_bases",
        display_name="Total Bases",
        inputs_cls=total_bases.TotalBasesInputs,
        score_fn=lambda inputs, line: total_bases.score_total_bases_prop(inputs, line),
        is_yes_no=False,
    ),
    "batter_rbis": PropTypeSpec(
        market_key="batter_rbis",
        display_name="RBIs",
        inputs_cls=rbis.RBIInputs,
        score_fn=lambda inputs, line: rbis.score_rbi_prop(inputs, line),
        is_yes_no=False,
    ),
    "batter_singles": PropTypeSpec(
        market_key="batter_singles",
        display_name="Singles",
        inputs_cls=singles.SinglesInputs,
        score_fn=lambda inputs, line: singles.score_singles_prop(inputs, line),
        is_yes_no=False,
    ),
    "batter_doubles": PropTypeSpec(
        market_key="batter_doubles",
        display_name="Doubles",
        inputs_cls=doubles.DoublesInputs,
        score_fn=lambda inputs, line: doubles.score_doubles_prop(inputs, line),
        is_yes_no=False,
    ),
    "pitcher_strikeouts": PropTypeSpec(
        market_key="pitcher_strikeouts",
        display_name="Pitcher Strikeouts",
        inputs_cls=pitcher_strikeouts.PitcherStrikeoutInputs,
        score_fn=lambda inputs, line: pitcher_strikeouts.score_pitcher_strikeouts_prop(inputs, line),
        is_yes_no=False,
    ),
}


def get_prop_spec(market_key: str) -> PropTypeSpec:
    try:
        return PROP_TYPES[market_key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown market key '{market_key}'. Known PropSync markets: "
            f"{sorted(PROP_TYPES)}"
        ) from exc
