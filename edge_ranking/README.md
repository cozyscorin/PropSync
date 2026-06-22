# PropSync Edge Ranking

The market-edge ranking layer. Takes (1) the odds pipeline's raw
FanDuel/DraftKings player prop lines and (2) the scoring model's
per-player win probabilities, and produces a single ranked list of
"best picks" across all 7 prop types — the actual output PropSync exists
to produce.

Scope: ranking/edge calculation only. Does not touch the data pipeline or
scoring model internals (only calls into them, via their documented
interfaces — see both READMEs). Does not build a dashboard/UI; that's the
next, separate task.

## What's implemented

| File | What it does |
|---|---|
| `devig.py` | American odds -> implied probability, plus two-way de-vig (vig/overround removal) to get a bookmaker's true assessed probability. |
| `market_map.py` | The one mapping table between The Odds API's 7 market keys and `scoring_model`'s 7 `score_<prop>_prop` functions / `*Inputs` dataclasses. Also defines which `outcome_name` ("Over"/"Yes") the scoring model's output corresponds to. |
| `scoring_bridge.py` | Looks up a player's prebuilt `*Inputs` dataclass and calls the matching `score_<prop>_prop` function for a given (player, market, line). |
| `ranking.py` | Core logic: groups raw odds rows into candidate legs, de-vigs each bookmaker's price, computes edge, resolves the dual-bookmaker comparison, and sorts everything by edge descending. |
| `parlay_selector.py` | Greedy top-N selector: walks the ranked list and picks legs that don't share a player or a game with anything already picked. |
| `tests/test_edge_ranking.py` | Runnable `unittest` suite, synthetic fixtures, 24 tests — de-vig math, edge sanity, dual-bookmaker selection, exclusion logic, end-to-end smoke test. |

## The de-vig math

A sportsbook's posted American odds price is **not** a fair probability
— it's inflated to bake in the house's edge (the "vig"/"overround"). If
PropSync compared its model probability directly against the raw
price-implied probability, every comparison would be biased against
PropSync on both sides of every market. De-vig removes that bias first.

**Step 1 — American odds to raw implied probability** (`american_to_implied_prob`):

```
price < 0 (favorite):  implied = -price / (-price + 100)
price > 0 (underdog):  implied = 100 / (price + 100)
```

Examples:
- `-150` → `150 / 250` = **0.6000** (60.00%)
- `+120` → `100 / 220` = **0.4545** (45.45%)
- `-110` → `110 / 210` = **0.5238** (52.38%)

**Step 2 — two-way de-vig** (`devig_two_way`): convert both sides of the
SAME bookmaker's SAME market to raw implied probabilities, then rescale
both by the same factor so they sum to exactly 1.0:

```
raw_a = american_to_implied_prob(price_a)
raw_b = american_to_implied_prob(price_b)
overround = raw_a + raw_b          # > 1.0 — this is the vig
fair_a = raw_a / overround
fair_b = raw_b / overround
```

Worked example — Over `-150` / Under `+120`:

```
raw_over  = 150/250 = 0.60000
raw_under = 100/220 = 0.45455
overround = 1.05455
fair_over  = 0.60000 / 1.05455 = 0.56897  (~56.9%)
fair_under = 0.45455 / 1.05455 = 0.43103  (~43.1%)
```

`fair_over + fair_under == 1.0` exactly — the vig is gone. This is
verified against the hand-calculated numbers above in
`tests/test_edge_ranking.py::TestDevigMath`. Sanity check built into the
tests too: de-vigging a symmetric `-110/-110` market lands at exactly
50/50, the textbook check for any de-vig formula.

This is the standard "proportional"/"basic" two-way de-vig method — not
the more involved Shin method (which models bettor information asymmetry
and needs more market data than a single two-way price pair gives you).
Proportional de-vig is the right level of rigor for a picks tool, not a
market-making operation.

**Fallback — single-side de-vig** (`devig_single_side_assumed_vig`): used
only when one bookmaker's opposing-side price (e.g. Under, when only Over
is posted) isn't available, so the true overround for that specific
market can't be measured directly. Divides by a fixed assumed overround
(`1.045`, ~4.5% vig — a round, middle-of-the-road estimate for
FanDuel/DraftKings player props, not a measured constant). Every
`BookPrice` records which method was used (`devig_method`) so this is
never silently conflated with a real two-way de-vig.

## Edge calculation

```
edge = model_probability - market_fair_probability
```

`model_probability` comes from calling the matching `score_<prop>_prop`
function. `market_fair_probability` is the de-vigged probability from
above. A positive edge means PropSync's model thinks the leg is more
likely to hit than the (vig-free) market does — that gap is the bet.
Every leg, across all 7 prop types, is ranked by this single number
(`ranking.rank_edges`), because every prop type's model output is
expressed in the same units (a real probability) — that's the whole point
of the Scoring Framework Notes' "probability + market edge" decision.

## Dual-bookmaker comparison

When both FanDuel and DraftKings have posted a price for the same
(player, market, line), `ranking._pick_better_bookmaker_price` de-vigs
**both** prices and picks whichever has the **lower** de-vigged fair
probability for the side PropSync's model scores (Over/Yes).

Why lower is better: `edge = model_prob - market_fair_prob`. The lower
the market's fair probability, the bigger the edge looks if the model
likes the leg — that book is offering a cheaper price on the side
PropSync already favors, i.e. better value to actually bet. Concretely:
if FanDuel prices a player's Over 1.5 hits at a de-vigged 55% and
DraftKings prices the exact same line at a de-vigged 50%, DraftKings is
the better book to bet that Over at, even though its raw price might look
"worse" in absolute odds terms — it's cheaper to buy into the same bet.

When only one book has posted a line, that book is used automatically
(picking the minimum of a one-element list is a no-op) — a leg is never
dropped just because the other book hasn't posted it yet, per the
Scoring Framework Notes' explicit instruction that both books are
cross-references, not a primary/fallback pair.

Every `EdgeCandidate` keeps both books' `BookPrice` data in
`all_book_prices` (even when only one used) for transparency/debugging,
even though only the chosen one feeds the rank.

Tested in `TestEdgeAndDualBookmaker`:
- `test_dual_book_picks_cheaper_price_strikeouts` — a case where the two
  books disagree a lot (FanDuel −170 vs DraftKings −110 on the same Over
  6.5 line); confirms DraftKings (the cheaper price) is selected.
- `test_dual_book_close_prices_doubles` — a case where the books are
  close but not identical, checking the mechanical "always pick the
  lower fair_prob" rule still holds.
- `test_single_book_leg_not_dropped` — a leg with only a FanDuel price
  posted; confirms it's still scored and ranked, not skipped.

## Top-N selection — no exclusion (`parlay_selector.py`)

History, per the Scoring Framework Notes' parlay-structure decision:
PropSync originally excluded same-game legs, then same-player legs, from
its picks list, on correlation grounds. Both were dropped:

- Same-game exclusion was dropped because a game can have more than one
  real, mostly-independent good pick, and blocking the rest of a game
  over one selected leg threw away real edge.
- Same-player exclusion was dropped on the same logic, taken further:
  different prop types on one player (e.g. hits + HR) are different
  bets, and a player having a great game can legitimately justify
  multiple legs. Worth knowing for the record: a HR technically also
  counts as a hit, so "1+ hits" and "HR" aren't fully independent
  outcomes even though they're different markets — that tradeoff is
  accepted deliberately.

`select_top_n(ranked_candidates, n)` is now a plain slice — no exclusion
logic of any kind. It returns `ranked_candidates[:n]` (or fewer if the
list is shorter than `n`). The same player or the same game can appear
multiple times in the result.

If a future correlation concern needs guarding against, it should be
re-added deliberately here rather than assumed.

Tested in `TestParlaySelector`:
- `test_allows_same_player_across_prop_types` — same player name on two
  different prop types (HR + total bases); confirms both can appear.
- `test_allows_same_game_different_players` — two different players in
  the same game (a pitcher's strikeout leg + a different team's batter's
  doubles leg); confirms both can be selected.
- `test_select_top_n_is_a_plain_slice_of_ranked_list` — confirms the
  output is exactly `ranked[:n]`, nothing skipped or reordered.
- `test_select_top_n_returns_fewer_if_n_exceeds_list_length` — asking for
  more legs than exist returns what's available, not padded or raised.

## How this plugs into the (not-yet-live) odds data

`ranking.build_candidate_legs(odds_df, inputs_registry)` is the single
entry point that takes raw pipeline output and turns it into ranked
`EdgeCandidate`s. It expects `odds_df` shaped **exactly** like
`data_pipeline/odds/odds_api_client.py`'s `get_all_player_props_today()`
output — confirmed against the real function signature and column list
during this build, not guessed:

```
event_id, commence_time, home_team, away_team, bookmaker, market,
player, outcome_name, line, price
```

- `bookmaker` is `"fanduel"` or `"draftkings"` per row, both pulled —
  exactly what `_pick_better_bookmaker_price` needs.
- `market` is one of the 7 PropSync market keys (`batter_home_runs`,
  `batter_hits`, etc.) — exactly what `market_map.py` keys off of.
- `outcome_name` is `"Over"`/`"Under"` for the 6 numeric-line markets and
  `"Yes"`/`"No"` for `batter_home_runs` — `market_map.py` encodes which
  one is the side the scoring model's output corresponds to.
- `line` is `None`/`NaN` for `batter_home_runs` (no numeric line) and a
  float (e.g. `1.5`) for everything else.

**Nothing in this layer needs to change once the pipeline goes live.**
The moment `get_all_player_props_today()` returns real rows instead of
nothing (currently blocked on an Odds API key — see
`data_pipeline/README.md`), `build_candidate_legs(real_df, registry)`
should work as-is, **provided the live DataFrame's actual columns match
the above** — confirm with `real_df.columns` on the first live pull, same
"fix forward" approach the data_pipeline and scoring_model READMEs both
call out for their own first-live-run risk.

One thing to watch: The Odds API's per-event endpoint can return more
than one bookmaker's row missing an opposing side in some edge cases
(e.g. a book pulls a line mid-slate). `build_candidate_legs` already
falls back to `devig_single_side_assumed_vig` per-book when that happens
— no code change needed, just confirm it's actually triggering the
fallback path (check `BookPrice.devig_method` on real output) rather than
silently mis-scoring.

## How this plugs into the scoring model

`inputs_registry: dict[(player_name, market_key), <prop>Inputs]` is the
**other** required input to `build_candidate_legs`, and building it is
explicitly **not** this layer's job — same boundary the scoring_model
README draws around its own inputs ("Exactly how this plugs into the
data pipeline" section). This layer assumes that dict already exists,
fully populated with real `HomeRunInputs`/`HitsInputs`/etc. instances
built from live pipeline data, by the time `build_candidate_legs` is
called.

What's needed to actually build that registry once both the data
pipeline and odds pipeline are live (this is the integration work for
*whatever calls this layer* — a daily run script, eventually the
Streamlit dashboard's backend — not a gap in this layer itself):

1. Pull today's odds (`get_all_player_props_today()`) to find out which
   (player, market) pairs actually need scoring — no point building
   `*Inputs` for players with no posted line.
2. For each one, pull that player's pipeline stats (season + rolling
   form + matchup pitcher's stats + park factors, per the exact mapping
   table in `scoring_model/README.md`'s "Exactly how this plugs into the
   data pipeline" section) and construct the matching `*Inputs`
   dataclass.
3. Key the registry dict by `(player_name, market_key)` — **the player
   name string must match exactly** between the odds DataFrame's
   `player` column (The Odds API's outcome `description` field) and
   however the registry is keyed. Name mismatches (e.g. "Mike Trout" vs.
   "M. Trout") are a real integration risk once both sides are live and
   not something this layer can detect — recommend normalizing names
   (e.g. lowercase, strip punctuation) on both sides at registry-build
   time if mismatches show up.
4. Pass the finished registry + the odds DataFrame into
   `build_candidate_legs`. Anything not in the registry is silently
   skipped (not an error) — see `test_missing_player_inputs_are_skipped_not_crashed`.

`market_map.py`'s `PROP_TYPES` table is the single source of truth for
which `*Inputs` dataclass each market needs — read it before building the
registry rather than re-deriving the mapping by hand.

## Running this end-to-end (once live data exists)

```python
from ranking import build_candidate_legs, rank_edges
from parlay_selector import select_top_n

# odds_df: from data_pipeline/odds/odds_api_client.py's
#          get_all_player_props_today()
# inputs_registry: built per "How this plugs into the scoring model" above

candidates = build_candidate_legs(odds_df, inputs_registry)
ranked = rank_edges(candidates)              # full "best picks" list
top_picks = select_top_n(ranked, n=5)        # for an actual parlay
```

## Running the tests

```bash
cd "[C] edge_ranking"
python3 tests/test_edge_ranking.py
```

24 tests, stdlib `unittest`, no pip install beyond `pandas` (already a
`data_pipeline` dependency). Covers:
- De-vig math against hand-calculated examples, including the
  symmetric `-110/-110` sanity check.
- Edge sign/magnitude sanity — an elite-power slugger's HR leg shows a
  real, bounded positive edge (~4.5 points in the fixture); a mediocre
  hitter against a tough pitcher does NOT show a large positive edge,
  confirming the model isn't just universally optimistic.
- Dual-bookmaker selection, including a case where the books disagree a
  lot (different rounding of "better price"), a case where they're
  close, and a case where only one book has a line.
- Top-N selection with no exclusion logic, including the specific case of
  one player appearing under two different prop types (both can appear)
  and two different players sharing a game (both can appear).
- An end-to-end smoke test (synthetic odds → ranked list → top-N picks)
  and a missing-inputs-skips-cleanly check.

## Folder structure

```
[C] edge_ranking/
├── README.md                this file
├── devig.py                  American odds <-> implied probability, two-way de-vig
├── market_map.py             market key <-> scoring_model function/dataclass table
├── scoring_bridge.py         look up player inputs, call the right score_<prop>_prop
├── ranking.py                candidate-leg building, dual-bookmaker selection, edge ranking
├── parlay_selector.py        plain top-N slice, no exclusion
└── tests/
    └── test_edge_ranking.py   runnable unittest suite, synthetic fixtures
```
