"""
Top-N selector — no exclusion.

History, per the Scoring Framework Notes' parlay-structure decision:
PropSync originally excluded same-game legs (different players, same
game), then same-player legs (different prop types, same player) from
its top-N picks, on correlation grounds. Both exclusions were dropped:

  - Same-game exclusion was dropped because a game can have more than one
    genuinely great, mostly-independent pick, and blocking the rest of a
    game over one selected leg threw away real edge.
  - Same-player exclusion was dropped on the same logic, taken further:
    different prop types on one player (e.g. hits + HR) are different
    bets, and a player having a great game can legitimately justify
    multiple legs. Note for the record: a HR technically also counts as
    a hit, so "1+ hits" and "HR" aren't fully independent outcomes even
    though they're different markets — that tradeoff is accepted
    deliberately, not overlooked.

There is currently no exclusion logic at all: this is a plain top-N slice
of the edge-ranked list. If a future correlation concern needs guarding
against, it should be re-added deliberately (and documented here) rather
than assumed.
"""
from __future__ import annotations

from ranking import EdgeCandidate


def select_top_n(
    ranked_candidates: list[EdgeCandidate],
    n: int,
) -> list[EdgeCandidate]:
    """
    Return the top `n` legs from an edge-ranked list, highest edge first.
    No exclusion of any kind — the same player or the same game can
    appear multiple times among the results.

    Args:
        ranked_candidates: output of ranking.rank_edges() — must already
            be sorted by edge descending. This function does not re-sort;
            it trusts the input order.
        n: how many legs to select.

    Returns:
        Up to `n` EdgeCandidate legs, in the same order as the input
        (i.e. still edge-descending). Returns fewer than `n` if the input
        list is shorter than `n` — does not pad or raise.
    """
    if n <= 0:
        return []
    return list(ranked_candidates[:n])
