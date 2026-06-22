"""
Bridges odds-pipeline rows to scoring_model calls.

The odds pipeline (`../[C] data_pipeline/odds/odds_api_client.py`) and the
scoring model (`../[C] scoring_model/`) don't know about each other —
that's by design (each was built standalone, see both READMEs). This
module is the glue: given a row identifying a (player, market, line) and
a lookup of that player's *Inputs dataclass (built upstream from pipeline
stat data — out of scope for this layer, see README "how this plugs in"),
call the right score_<prop>_prop function and return PropSync's model
probability for that exact line.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from market_map import get_prop_spec

# A PlayerInputsRegistry maps (player_name, market_key) -> an already-built
# *Inputs dataclass instance (HomeRunInputs, HitsInputs, etc.) for that
# player against tonight's opposing pitcher/context. Building these
# dataclasses from real pipeline data is explicitly NOT this layer's job
# (see scoring_model README's "Exactly how this plugs into the data
# pipeline" section) — by the time a dict reaches this module, every input
# field the relevant prop module needs has already been resolved.
PlayerInputsRegistry = dict[tuple[str, str], Any]


class MissingPlayerInputsError(KeyError):
    """Raised when an odds row references a player/market with no
    corresponding entry in the inputs registry — i.e. we have a market
    line to bet on but no model inputs to score it with. This should be
    treated as 'skip this leg, can't score it' by callers, not a crash."""


def score_for_row(
    player: str,
    market: str,
    line: float | None,
    inputs_registry: PlayerInputsRegistry,
) -> float:
    """
    Look up the player's prebuilt *Inputs for this market and call the
    matching score_<prop>_prop function.

    Args:
        player: player name exactly as it appears in the odds DataFrame's
            `player` column (The Odds API's outcome `description` field —
            see data_pipeline README; this must match however the
            inputs_registry keys its players, which is an integration
            detail for whatever builds the registry).
        market: The Odds API market key, e.g. "batter_hits".
        line: the betting line for this leg (None for batter_home_runs,
            which has no numeric line).
        inputs_registry: (player, market) -> *Inputs dataclass instance.

    Returns:
        PropSync's model probability (0-1) for this exact (player, market,
        line) leg.

    Raises:
        MissingPlayerInputsError: if no inputs are registered for this
            (player, market) pair.
    """
    spec = get_prop_spec(market)
    key = (player, market)
    if key not in inputs_registry:
        raise MissingPlayerInputsError(
            f"No scoring inputs registered for player={player!r} market={market!r}. "
            f"Build that player's {spec.inputs_cls.__name__} from pipeline data "
            f"before scoring this leg."
        )
    inputs = inputs_registry[key]
    return spec.score_fn(inputs, line)


@dataclass(frozen=True)
class ScoredLeg:
    """One (player, market, line) leg with PropSync's model probability
    attached — the output of scoring_bridge, the input to edge calc."""
    player: str
    market: str
    line: float | None
    model_prob: float
