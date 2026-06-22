"""
American odds <-> implied probability conversion, plus two-way de-vig.

This is the only place in edge_ranking that touches sportsbook price math.
Every other module treats "market probability" as something this module
already produced.

Why de-vig matters: a sportsbook's posted price is not a fair estimate of
the true probability — it's inflated to bake in the house's edge (the
"vig" / "overround"). If PropSync compared its own model probability
directly against the raw price-implied probability, every comparison
would be biased against PropSync by roughly the vig amount, on both sides
of every market. Removing the vig first isolates the book's actual
assessed probability, which is the correct thing to compare PropSync's
model probability against (per the Scoring Framework Notes' "probability
+ market edge" decision).
"""
from __future__ import annotations

from dataclasses import dataclass


def american_to_implied_prob(price: int | float) -> float:
    """
    Convert a single American odds price into its raw (vig-included)
    implied probability.

    Formula:
        price < 0 (favorite):  implied = -price / (-price + 100)
        price > 0 (underdog):  implied = 100 / (price + 100)

    Examples (hand-verifiable):
        -150  -> 150 / 250       = 0.6000  (60.00%)
        +120  -> 100 / 220       = 0.4545  (45.45%)
        -110  -> 110 / 210       = 0.5238  (52.38%)
        +100  -> 100 / 200       = 0.5000  (50.00%)

    This number is deliberately NOT the book's true assessed probability
    — it includes the vig. Use devig_two_way() to remove it before
    comparing against the model.
    """
    if price == 0:
        raise ValueError("American odds price cannot be 0")
    if price < 0:
        return (-price) / (-price + 100)
    return 100 / (price + 100)


@dataclass(frozen=True)
class DevigResult:
    """
    True (de-vigged) probabilities for both sides of a two-way market,
    plus the diagnostic info used to get there.
    """
    side_a_raw_implied: float
    side_b_raw_implied: float
    overround: float          # how much over 1.0 the raw implied probs sum to
    side_a_fair_prob: float   # de-vigged "true" probability for side A
    side_b_fair_prob: float   # de-vigged "true" probability for side B


def devig_two_way(price_a: int | float, price_b: int | float) -> DevigResult:
    """
    Remove the vig from a two-way market (Over/Under, or Yes/No) using the
    standard "multiplicative" / proportional de-vig method: convert both
    prices to raw implied probabilities, then rescale both by the same
    factor so they sum to exactly 1.0.

        raw_a = american_to_implied_prob(price_a)
        raw_b = american_to_implied_prob(price_b)
        overround = raw_a + raw_b          # > 1.0, the vig
        fair_a = raw_a / overround
        fair_b = raw_b / overround

    Worked example: Over -150 / Under +120
        raw_over  = 150/250 = 0.60000
        raw_under = 100/220 = 0.45455
        overround = 1.05455
        fair_over  = 0.60000 / 1.05455 = 0.56897  (~56.9%)
        fair_under = 0.45455 / 1.05455 = 0.43103  (~43.1%)
        (fair_over + fair_under == 1.0, vig removed)

    This is the standard, widely-used de-vig method for two-sided markets
    (sometimes called "proportional" or "basic" de-vig, as opposed to the
    more involved Shin method, which requires modeling bettor information
    asymmetry and is overkill here — proportional de-vig is the right
    level of rigor for a props tool, not a market-making operation).

    Note: this assumes both sides of the market are priced by the SAME
    bookmaker for the SAME line (e.g. FanDuel's Over 1.5 and FanDuel's
    Under 1.5 for the same player/prop). Don't mix one book's Over price
    with another book's Under price — that's not a coherent market and
    the overround would be meaningless.
    """
    raw_a = american_to_implied_prob(price_a)
    raw_b = american_to_implied_prob(price_b)
    overround = raw_a + raw_b

    if overround <= 0:
        raise ValueError(f"Invalid overround {overround} from prices {price_a}, {price_b}")

    return DevigResult(
        side_a_raw_implied=raw_a,
        side_b_raw_implied=raw_b,
        overround=overround,
        side_a_fair_prob=raw_a / overround,
        side_b_fair_prob=raw_b / overround,
    )


def devig_single_side_assumed_vig(price: int | float, assumed_overround: float = 1.045) -> float:
    """
    Fallback de-vig for when only ONE side of a two-way market is
    available (e.g. the Odds API returned an Over price but the Under
    price is missing from that bookmaker's response for some reason).

    Without the opposing price, the true overround for that specific
    market can't be measured directly, so this divides by a fixed assumed
    overround instead. `assumed_overround = 1.045` (a ~4.5% vig) is a
    reasonable typical value for FanDuel/DraftKings player props based on
    commonly-cited -110-ish two-way pricing (-110/-110 implies almost
    exactly a 4.76% overround; player props often run slightly tighter on
    the favored side and wider on the long side, so 4.5% is a round,
    middle-of-the-road estimate, not a measured constant).

    PREFER devig_two_way() whenever both sides are present — this function
    exists only as a degraded fallback and callers should be able to tell
    which path was used (see EdgeCandidate.devig_method in ranking.py).
    """
    raw = american_to_implied_prob(price)
    return raw / assumed_overround
