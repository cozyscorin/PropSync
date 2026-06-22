"""
Core market-edge ranking layer.

Pipeline (per Scoring Framework Notes' "cross-prop normalization —
probability + market edge" decision):

  1. Group raw odds rows (one row per bookmaker per outcome) into
     candidate legs: a (player, market, line) tuple with up to two
     bookmaker prices (FanDuel / DraftKings) on the modeled side
     (Over / Yes), plus the opposing side's price when available for
     de-vig.
  2. Call the scoring model once per candidate leg to get PropSync's
     model probability for that exact line (scoring_bridge.py).
  3. De-vig each bookmaker's price for that leg (devig.py).
  4. edge = model_prob - market_fair_prob, computed per bookmaker.
  5. Dual-bookmaker handling: if both books have a price, keep whichever
     book's edge is more favorable to PropSync (see
     `_pick_better_bookmaker_price` docstring for exactly what
     "favorable" means here). If only one book has a price, use it.
  6. Sort every candidate leg, across all 7 prop types, by edge
     descending. That ranked list is the actual "best picks" output.

This module's output (`EdgeCandidate` / `rank_edges`) is intentionally
prop-type-agnostic: it doesn't care whether a leg is a HR yes/no market or
a hits over/under market, because by the time scoring_bridge.py has run,
every leg is expressed in the same units — a model probability and a
market-implied probability, both in [0, 1]. That uniformity is exactly
what the Scoring Framework Notes' cross-prop normalization decision is
for.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from devig import DevigResult, american_to_implied_prob, devig_single_side_assumed_vig, devig_two_way
from market_map import modeled_outcome_name, opposing_outcome_name
from scoring_bridge import PlayerInputsRegistry, score_for_row


@dataclass(frozen=True)
class BookPrice:
    """One bookmaker's price (and de-vigged fair probability) for the
    modeled side of a single candidate leg."""
    bookmaker: str             # "fanduel" or "draftkings"
    price: int                 # American odds for the modeled side (Over/Yes)
    opposing_price: int | None  # American odds for the other side, if known
    fair_prob: float            # de-vigged implied probability for the modeled side
    devig_method: str           # "two_way" or "single_side_assumed_vig"


@dataclass(frozen=True)
class EdgeCandidate:
    """
    One fully-scored candidate leg, ready to rank.

    `chosen_book` / `chosen_market_prob` / `edge` reflect whichever
    bookmaker price was selected per the dual-bookmaker rule (see
    `_pick_better_bookmaker_price`). `all_book_prices` keeps both books'
    data around for transparency/debugging — nothing downstream needs it,
    but it's useful when eyeballing why a pick was selected.
    """
    player: str
    market: str
    line: float | None
    event_id: str
    home_team: str
    away_team: str
    model_prob: float
    chosen_book: str
    chosen_book_price: int
    chosen_market_prob: float
    edge: float                       # model_prob - chosen_market_prob
    all_book_prices: tuple[BookPrice, ...]

    @property
    def game_key(self) -> str:
        """Identifies "the same game" for the exclusion selector — uses
        event_id directly since that's the pipeline's own unique game
        identifier (more reliable than reconstructing it from team names)."""
        return self.event_id


# ---------------------------------------------------------------------------
# Step 1-2: group raw odds rows into candidate legs
# ---------------------------------------------------------------------------

def _build_book_price(
    market: str,
    modeled_price: int,
    opposing_price: int | None,
) -> tuple[float, str]:
    """
    De-vig a single bookmaker's price for one leg. Prefers the two-way
    method (uses both sides of that SAME book's market) whenever the
    opposing price is available; falls back to the assumed-overround
    method otherwise (see devig.py for why, and which is preferred).
    """
    if opposing_price is not None:
        result: DevigResult = devig_two_way(modeled_price, opposing_price)
        return result.side_a_fair_prob, "two_way"
    return devig_single_side_assumed_vig(modeled_price), "single_side_assumed_vig"


def build_candidate_legs(
    odds_df: pd.DataFrame,
    inputs_registry: PlayerInputsRegistry,
) -> list[EdgeCandidate]:
    """
    Turn the odds pipeline's flat row-per-(event,bookmaker,market,outcome)
    DataFrame (see data_pipeline/odds/odds_api_client.py's
    get_all_player_props_today() — columns: event_id, commence_time,
    home_team, away_team, bookmaker, market, player, outcome_name, line,
    price) into one EdgeCandidate per (player, market, line), with the
    dual-bookmaker comparison already resolved.

    Rows whose player/market has no entry in inputs_registry are skipped
    (can't score what we don't have inputs for) rather than raising —
    a single missing player shouldn't take down the whole ranking run.
    """
    required_cols = {
        "event_id", "commence_time", "home_team", "away_team",
        "bookmaker", "market", "player", "outcome_name", "line", "price",
    }
    missing_cols = required_cols - set(odds_df.columns)
    if missing_cols:
        raise ValueError(
            f"odds_df is missing expected columns from "
            f"get_all_player_props_today(): {sorted(missing_cols)}"
        )

    candidates: list[EdgeCandidate] = []

    # Group by the leg identity: same game, same market, same player, same
    # line. The HR market has line == NaN/None for every row (yes/no
    # market, no numeric line) — group on (event_id, market, player, line)
    # still works since NaN groups correctly within a single groupby key
    # when using dropna=False.
    group_cols = ["event_id", "market", "player", "line"]
    for (event_id, market, player, line), leg_rows in odds_df.groupby(
        group_cols, dropna=False, sort=False
    ):
        modeled_name = modeled_outcome_name(market)
        opposing_name = opposing_outcome_name(market)

        modeled_rows = leg_rows[leg_rows["outcome_name"] == modeled_name]
        if modeled_rows.empty:
            # This leg only has the opposing side priced (e.g. a book
            # posted Under but not Over) — nothing for the model to
            # compare against on the side it scores. Skip.
            continue

        key = (player, market)
        if key not in inputs_registry:
            continue  # can't score without inputs; not an error, just skip

        line_value = None if pd.isna(line) else float(line)
        model_prob = score_for_row(player, market, line_value, inputs_registry)

        first_row = leg_rows.iloc[0]
        book_prices: list[BookPrice] = []
        for _, row in modeled_rows.iterrows():
            bookmaker = row["bookmaker"]
            modeled_price = int(row["price"])

            opposing_match = leg_rows[
                (leg_rows["bookmaker"] == bookmaker)
                & (leg_rows["outcome_name"] == opposing_name)
            ]
            opposing_price = (
                int(opposing_match.iloc[0]["price"]) if not opposing_match.empty else None
            )

            fair_prob, method = _build_book_price(market, modeled_price, opposing_price)
            book_prices.append(
                BookPrice(
                    bookmaker=bookmaker,
                    price=modeled_price,
                    opposing_price=opposing_price,
                    fair_prob=fair_prob,
                    devig_method=method,
                )
            )

        if not book_prices:
            continue

        chosen = _pick_better_bookmaker_price(model_prob, book_prices)

        candidates.append(
            EdgeCandidate(
                player=player,
                market=market,
                line=line_value,
                event_id=str(event_id),
                home_team=str(first_row["home_team"]),
                away_team=str(first_row["away_team"]),
                model_prob=model_prob,
                chosen_book=chosen.bookmaker,
                chosen_book_price=chosen.price,
                chosen_market_prob=chosen.fair_prob,
                edge=model_prob - chosen.fair_prob,
                all_book_prices=tuple(book_prices),
            )
        )

    return candidates


# ---------------------------------------------------------------------------
# Step 5: dual-bookmaker selection
# ---------------------------------------------------------------------------

def _pick_better_bookmaker_price(model_prob: float, book_prices: list[BookPrice]) -> BookPrice:
    """
    Per the Scoring Framework Notes' dual-bookmaker decision: when both
    FanDuel and DraftKings have a posted line for the same (player, prop,
    line), compute edge against both and surface whichever is more
    favorable.

    "More favorable" = lower de-vigged fair_prob for the side PropSync's
    model favors, because edge = model_prob - market_fair_prob — the
    LOWER the market thinks this leg is, the BIGGER PropSync's edge looks
    if the model agrees the leg should hit. Concretely: if the model likes
    Over 1.5 hits, and FanDuel prices Over 1.5 at fair_prob 0.55 while
    DraftKings prices the same Over 1.5 at fair_prob 0.50, DraftKings is
    "worse" for the bettor in absolute odds, but cheaper to buy into and
    therefore the better bet to actually place — it's giving better
    value on the side PropSync already wants. So this always picks the
    MINIMUM fair_prob across available books, full stop — that's
    equivalent to "the book giving the better price for the side
    PropSync favors" regardless of whether PropSync's edge ends up
    positive or negative on that leg.

    When only one book has a price, it's used automatically (the min of a
    one-element list is itself) — per the notes, a leg is never dropped
    just because only one book posted it.
    """
    return min(book_prices, key=lambda bp: bp.fair_prob)


# ---------------------------------------------------------------------------
# Step 6: ranking
# ---------------------------------------------------------------------------

def rank_edges(candidates: list[EdgeCandidate]) -> list[EdgeCandidate]:
    """
    Sort all candidate legs, across every prop type, by edge descending.
    This is the literal "best picks" output PropSync exists to produce —
    every leg is already expressed in the same units (model probability
    minus de-vigged market probability), so a HR leg and a pitcher-Ks leg
    sort on equal footing.
    """
    return sorted(candidates, key=lambda c: c.edge, reverse=True)
